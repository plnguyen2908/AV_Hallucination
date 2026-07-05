"""
_5_intervene.py — patched attention for Stage 5 interventions.

Monkey-patches `Qwen2_5OmniAttention.forward` (the thinker's decoder
self-attention, eager implementation) to allow post-softmax modification
of attention weights at a set of (layer, head, key) positions specified
by a module-level config dict.

Usage:
    from _5_intervene import patch_qwen_attention, set_intervention, clear_intervention
    patch_qwen_attention(model)
    set_intervention(dict(
        layer_to_heads = {0: [3, 7], 1: [5], ...},     # set of head idxs per layer
        layer_to_key_mask = {0: tensor([..]),  ...},   # (S,) bool tensor per layer
        mode = "suppress",                              # "suppress" -> multiply by gamma
                                                       # "boost"    -> multiply by (1+epsilon)
                                                       # "complement"-> apply to ~key_mask instead
        gamma = 0.5,
    ))
    # ... run forward / generate ...
    clear_intervention()
"""

import math
from typing import Optional

import torch
import torch.nn as nn

# Module-level config
_STATE: dict = dict(active=False)
_DEBUG = dict(called=0, intervened=0, modified=0)


def reset_debug():
    _DEBUG["called"] = 0
    _DEBUG["intervened"] = 0
    _DEBUG["modified"] = 0


def get_debug():
    return dict(_DEBUG)


def set_intervention(cfg):
    """cfg keys: layer_to_heads {L: set/list of head idx},
    layer_to_key_mask {L: torch.bool tensor (S,)},
    mode 'suppress' | 'boost' | 'complement',
    gamma float (suppress factor or boost epsilon)."""
    _STATE.clear()
    _STATE.update(cfg)
    _STATE["active"] = True


def clear_intervention():
    _STATE.clear()
    _STATE["active"] = False


def _get_state():
    return _STATE


_ORIG_FORWARD = None


def patched_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
):
    _DEBUG["called"] += 1
    # We use the layer_idx attribute set by the model on attn modules.
    layer_idx = getattr(self, "layer_idx", None)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    # Late import to keep this file's import side-effects minimal
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_multimodal_rotary_pos_emb,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        repeat_kv as _repeat_kv,
    )

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = _repeat_kv(key_states, self.num_key_value_groups)
    value_states = _repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if query_states.dtype == torch.float16:
        attn_weights = torch.where(
            torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )

    # ----- KILL-SWITCH 2: pre-softmax temperature flattening on ALL heads
    # at ALL layers. No sink targeting, no head taxonomy. T > 1 flattens
    # attention toward uniform; T < 1 sharpens.
    if _STATE.get("active") and _STATE.get("mode") == "temperature_flatten":
        T = float(_STATE.get("gamma", 1.0))
        if T != 1.0 and T > 0:
            attn_weights = attn_weights / T

    # ----- softmax -----
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # ----- VALUE-ZERO INTERVENTION (zeroes V at sink positions for target
    # heads, BEFORE attention output is computed). Distinct from attention
    # reweighting: leaves attn distribution untouched but subtracts the
    # sink rows' contribution from the output. -----
    if (
        _STATE.get("active")
        and layer_idx is not None
        and _STATE.get("mode") == "value_zero"
    ):
        l2h = _STATE.get("layer_to_heads")
        l2k = _STATE.get("layer_to_key_mask")
        if l2h and layer_idx in l2h and l2k and layer_idx in l2k:
            mask = l2k[layer_idx].to(value_states.device)
            cur_k = value_states.shape[-2]
            if mask.numel() != cur_k:
                new_mask = torch.zeros(
                    cur_k, dtype=torch.bool, device=value_states.device
                )
                new_mask[: mask.numel()] = mask
                mask = new_mask
            head_idx_t = torch.as_tensor(
                list(l2h[layer_idx]), dtype=torch.long, device=value_states.device
            )
            # value_states shape (B, H, K, D). Zero rows where mask is True
            # for the target heads.
            vs = value_states.clone()
            for h in head_idx_t.tolist():
                vs[:, h, mask, :] = 0
            value_states = vs
            _DEBUG["modified"] += 1
            _DEBUG["intervened"] += 1

    # ----- SINK→NON-SINK MASS-CONSERVING REDISTRIBUTE -----------------
    # For each target head and each generated query position:
    #   moved      = γ · sum_k(attn[k] for k in sink positions)
    #   attn[k in sinks]     *= (1 - γ)   (scale sinks down uniformly)
    #   attn[k in non-sinks] += moved / |non_sinks|   (even spread)
    # No row renormalization — mass is conserved by construction.
    # If --include_text_sinks is set in compute_per_layer_sink_masks,
    # the source sink mask covers audio+video+text spans (not just routed
    # modality).
    if (
        _STATE.get("active")
        and layer_idx is not None
        and _STATE.get("mode") == "sink_to_nonsink_redistribute"
    ):
        l2h = _STATE.get("layer_to_heads")
        l2k = _STATE.get("layer_to_key_mask")
        if l2h and layer_idx in l2h and l2k:
            heads_to_modify = l2h[layer_idx]
            mask = l2k.get(layer_idx)
            if mask is not None and len(heads_to_modify) > 0:
                cur_k = attn_weights.shape[-1]
                if mask.numel() != cur_k:
                    new_mask = torch.zeros(
                        cur_k, dtype=torch.bool, device=attn_weights.device)
                    new_mask[: mask.numel()] = mask.to(attn_weights.device)
                    mask = new_mask
                else:
                    mask = mask.to(attn_weights.device)
                non_sink = ~mask
                n_nonsink = int(non_sink.sum().item())
                if n_nonsink > 0 and mask.any():
                    head_idx_t = torch.as_tensor(
                        list(heads_to_modify), dtype=torch.long,
                        device=attn_weights.device)
                    gamma = float(_STATE.get("gamma", 1.0))
                    aw = attn_weights.clone()
                    sel = aw[:, head_idx_t, :, :]
                    sink_attn_sum = sel[..., mask].sum(dim=-1, keepdim=True)
                    moved = gamma * sink_attn_sum  # (B, H', Q, 1)
                    # Scale sinks down by (1-γ).
                    sel[..., mask] = sel[..., mask] * (1.0 - gamma)
                    # Route-weighted distribution: send moved mass into
                    # audio/video non-sink positions weighted by router
                    # softmax shares (parallels the BOS-redistribute design).
                    route_weighted = bool(
                        _STATE.get("route_weighted", False))
                    if route_weighted:
                        p_audio = float(_STATE.get("p_audio", 0.0))
                        p_visual = float(_STATE.get("p_visual", 0.0))
                        p_av = float(_STATE.get("p_av", 0.0))
                        alpha_av = float(_STATE.get("alpha_av", 0.5))
                        a_raw = p_audio + alpha_av * p_av
                        v_raw = p_visual + alpha_av * p_av
                        tot = a_raw + v_raw
                        a_share = a_raw / tot if tot > 1e-12 else 0.0
                        v_share = v_raw / tot if tot > 1e-12 else 0.0
                        ap = _STATE.get("audio_positions")
                        vp = _STATE.get("visual_positions")
                        if ap is not None and v_share is not None:
                            ap = ap.to(attn_weights.device)
                            ap = ap[ap < cur_k]
                            ap_nonsink = ap[~mask[ap]]
                            if a_share > 0 and len(ap_nonsink) > 0:
                                sel[..., ap_nonsink] += (
                                    moved.squeeze(-1)
                                    * (a_share / len(ap_nonsink))
                                ).unsqueeze(-1)
                        if vp is not None:
                            vp = vp.to(attn_weights.device)
                            vp = vp[vp < cur_k]
                            vp_nonsink = vp[~mask[vp]]
                            if v_share > 0 and len(vp_nonsink) > 0:
                                sel[..., vp_nonsink] += (
                                    moved.squeeze(-1)
                                    * (v_share / len(vp_nonsink))
                                ).unsqueeze(-1)
                    else:
                        sel[..., non_sink] = (sel[..., non_sink]
                                                  + moved / n_nonsink)
                    aw[:, head_idx_t, :, :] = sel
                    attn_weights = aw
                    _DEBUG["modified"] += 1
                    _DEBUG["intervened"] += 1

    # ----- BOS REDISTRIBUTE INTERVENTION ------------------------------
    # Move attention mass from BOS position (default 0) to modality
    # positions, proportional to router softmax weights:
    #   audio_share = p_audio + alpha · p_av
    #   visual_share = p_visual + alpha · p_av
    # Mass is conserved (we subtract from BOS, add to modality).
    if (
        _STATE.get("active")
        and layer_idx is not None
        and _STATE.get("mode") == "bos_redistribute"
    ):
        l2h = _STATE.get("layer_to_heads")
        if l2h and layer_idx in l2h:
            heads_to_modify = l2h[layer_idx]
            if len(heads_to_modify) > 0:
                bos_pos = int(_STATE.get("bos_pos", 0))
                p_audio = float(_STATE.get("p_audio", 0.0))
                p_visual = float(_STATE.get("p_visual", 0.0))
                p_av = float(_STATE.get("p_av", 0.0))
                alpha_av = float(_STATE.get("alpha_av", 0.5))
                # γ (master gain) scales how much of the BOS mass is moved.
                gamma = float(_STATE.get("gamma", 1.0))
                a_share_raw = p_audio + alpha_av * p_av
                v_share_raw = p_visual + alpha_av * p_av
                # Always renormalize shares so a + v = 1 → exactly the BOS
                # mass we subtracted is what we redistribute (mass-conserving).
                tot = a_share_raw + v_share_raw
                if tot > 1e-12:
                    a_share = a_share_raw / tot
                    v_share = v_share_raw / tot
                else:
                    a_share = v_share = 0.0
                # Per-modality positions (Long tensors)
                audio_positions = _STATE.get("audio_positions")
                visual_positions = _STATE.get("visual_positions")
                cur_k = attn_weights.shape[-1]
                if bos_pos < cur_k and (audio_positions is not None or
                                            visual_positions is not None):
                    head_idx_t = torch.as_tensor(
                        list(heads_to_modify), dtype=torch.long,
                        device=attn_weights.device)
                    # attn_weights: (B, H, Q, K). Pick chosen heads.
                    aw = attn_weights.clone()
                    bos_attn = aw[:, head_idx_t, :, bos_pos:bos_pos + 1].clone()
                    # γ controls fraction moved (0 → no-op, 1 → full BOS mass).
                    moved = bos_attn * gamma
                    aw[:, head_idx_t, :, bos_pos:bos_pos + 1] -= moved
                    if a_share > 0 and audio_positions is not None and \
                       len(audio_positions) > 0:
                        ap = audio_positions.to(attn_weights.device)
                        ap = ap[ap < cur_k]
                        if len(ap) > 0:
                            delta = (moved.squeeze(-1) * (a_share / len(ap))).unsqueeze(-1)
                            aw[:, head_idx_t][..., ap] += delta
                    if v_share > 0 and visual_positions is not None and \
                       len(visual_positions) > 0:
                        vp = visual_positions.to(attn_weights.device)
                        vp = vp[vp < cur_k]
                        if len(vp) > 0:
                            delta = (moved.squeeze(-1) * (v_share / len(vp))).unsqueeze(-1)
                            aw[:, head_idx_t][..., vp] += delta
                    attn_weights = aw
                    _DEBUG["modified"] += 1
                    _DEBUG["intervened"] += 1

    # ----- ATTENTION-WEIGHT INTERVENTION -----
    st = _STATE
    if (
        st.get("active")
        and layer_idx is not None
        and st.get("mode") in ("suppress", "boost", "complement")
    ):
        _DEBUG["intervened"] += 1
        l2h = st.get("layer_to_heads")
        if l2h and layer_idx in l2h:
            heads_to_modify = l2h[layer_idx]
            l2k = st.get("layer_to_key_mask")
            mask = l2k.get(layer_idx) if l2k else None
            if mask is not None and len(heads_to_modify) > 0:
                # mask is (S,) bool; align to current key length
                cur_k = attn_weights.shape[-1]
                if mask.numel() != cur_k:
                    # During generation, KV cache grows — we mask only the
                    # PROMPT positions; any newly-generated tokens are
                    # never marked as sinks.
                    new_mask = torch.zeros(
                        cur_k, dtype=torch.bool, device=attn_weights.device
                    )
                    new_mask[: mask.numel()] = mask.to(attn_weights.device)
                    mask = new_mask
                else:
                    mask = mask.to(attn_weights.device)
                mode = st.get("mode", "suppress")
                gamma = float(st.get("gamma", 1.0))
                # Per-layer scheduled gamma (audio/visual/av sink schedule,
                # already router-mixed per clip): use this layer's value.
                sched = st.get("gamma_schedule")
                if sched is not None and layer_idx is not None and layer_idx < len(sched):
                    gamma = float(sched[layer_idx])
                # Pick the positions in attn_weights to scale.
                target_mask = mask if mode != "complement" else ~mask
                # Convert head list to a long tensor on device
                head_idx_t = torch.as_tensor(
                    list(heads_to_modify), dtype=torch.long, device=attn_weights.device
                )
                if mode == "boost":
                    scale = 1.0 + gamma
                else:
                    # suppress (gamma<1 dims down) or complement-boost
                    scale = gamma
                # attn_weights: (B, H, Q, K). Modify selected heads at
                # selected keys.
                # broadcast: (1,1,1,K) * heads * keys
                sel = attn_weights[:, head_idx_t, :, :]  # (B, H', Q, K)
                # Apply scale on target keys
                scale_t = torch.where(
                    target_mask,
                    torch.full_like(target_mask, scale, dtype=sel.dtype),
                    torch.ones_like(target_mask, dtype=sel.dtype),
                )
                sel = sel * scale_t  # broadcast on K
                # Row-renormalize so each (B, H', Q) row still sums to 1
                row_sums = sel.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                sel = sel / row_sums
                attn_weights = attn_weights.clone()
                attn_weights[:, head_idx_t, :, :] = sel
                _DEBUG["modified"] += 1

    attn_weights_for_dropout = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights_for_dropout, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def patch_qwen_attention(model):
    """Monkey-patch the eager Qwen2_5OmniAttention forward — both at the
    class level AND at the instance level (accelerate's device_map hooks
    wrap each instance's forward with `_old_forward = original_class.forward`,
    so class-level patching alone is invisible)."""
    import types

    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAttention,
    )

    global _ORIG_FORWARD
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = Qwen2_5OmniAttention.forward
    Qwen2_5OmniAttention.forward = patched_forward

    # Per-instance fix: replace `_old_forward` (the bound method accelerate
    # stored before wrapping). Walk the thinker decoder layers.
    n_patched = 0
    for layer in _iter_thinker_layers(model):
        sa = getattr(layer, "self_attn", None)
        if sa is None or not isinstance(sa, Qwen2_5OmniAttention):
            continue
        if hasattr(sa, "_old_forward"):
            # Save the original bound method once.
            if not hasattr(sa, "_orig_inner_forward"):
                sa._orig_inner_forward = sa._old_forward
            sa._old_forward = types.MethodType(patched_forward, sa)
            n_patched += 1
    return n_patched


def unpatch_qwen_attention(model=None):
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAttention,
    )

    if _ORIG_FORWARD is not None:
        Qwen2_5OmniAttention.forward = _ORIG_FORWARD
    if model is not None:
        for layer in _iter_thinker_layers(model):
            sa = getattr(layer, "self_attn", None)
            if sa is not None and hasattr(sa, "_orig_inner_forward"):
                sa._old_forward = sa._orig_inner_forward


def _iter_thinker_layers(model):
    thinker = getattr(model, "thinker", model)
    text_model = getattr(thinker, "model", thinker)
    layers = getattr(text_model, "layers", None)
    if layers is None and hasattr(thinker, "layers"):
        layers = thinker.layers
    if layers is None:
        return []
    return layers
