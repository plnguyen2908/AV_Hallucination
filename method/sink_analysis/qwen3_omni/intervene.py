"""intervene.py — Qwen3-Omni sink-boost attention patch.

Port of method/sink_analysis/qwen2_5_omni/_5_intervene.py to Qwen3-Omni.
Qwen3 dispatches attention through the module-level `eager_attention_forward`
(under attn_implementation="eager"), so we patch THAT (not an inline
`Attention.forward`). We reimplement its softmax->matmul body and inject a
post-softmax reweighting that BOOSTs a set of (layer, head)'s attention to
sink-token key positions, per a per-layer gamma schedule, then row-renormalizes.

Only the thinker TEXT decoder attention is intervened (checked by the module's
class name); audio/vision encoder attention is passed through untouched.

Usage:
    import intervene as IV
    IV.patch(model)
    IV.set_intervention(dict(layer_to_heads={L:[h..]},
                             layer_to_key_mask={L: bool (S,)},
                             mode="boost", gamma=3.0,
                             gamma_schedule=[g0,g1,...gL-1]))  # optional per-layer
    ... generate ...
    IV.clear_intervention()
"""
import torch
import torch.nn as nn

_STATE = dict(active=False)
_DEBUG = dict(called=0, modified=0)
_THINKER_ATTN_NAME = "Qwen3OmniMoeThinkerTextAttention"
_ORIG_EAGER = None

# Measurement of attention received per (layer, key position). Populated by a
# prompt-only forward when _MEASURE["on"] is True, WITHOUT retaining full
# attention matrices (only a (K,) sum per layer) — safe for long AV prompts
# where output_attentions would OOM.
_MEASURE = dict(on=False, sum={}, nq={})


def start_measure():
    _MEASURE["on"] = True; _MEASURE["sum"] = {}; _MEASURE["nq"] = {}


def stop_measure():
    _MEASURE["on"] = False


def get_measure():
    """Return {layer_idx: mean attention received per key position (K,) numpy}."""
    out = {}
    for L, s in _MEASURE["sum"].items():
        out[L] = (s / max(_MEASURE["nq"].get(L, 1), 1)).float().cpu().numpy()
    return out


def set_intervention(cfg):
    _STATE.clear(); _STATE.update(cfg); _STATE["active"] = True


def clear_intervention():
    _STATE.clear(); _STATE["active"] = False


def get_debug():
    return dict(_DEBUG)


def reset_debug():
    _DEBUG["called"] = 0; _DEBUG["modified"] = 0


def _patched_eager(module, query, key, value, attention_mask, scaling,
                   dropout=0.0, **kwargs):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        repeat_kv,
    )
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

    layer_idx = getattr(module, "layer_idx", None)

    # ---- measurement: accumulate attention received per key (thinker text) ----
    if (
        _MEASURE["on"]
        and layer_idx is not None
        and type(module).__name__ == _THINKER_ATTN_NAME
    ):
        # attn_weights (B,H,Q,K) -> sum over B,H,Q -> (K,); track query count.
        recv = attn_weights.sum(dim=(0, 1, 2)).detach()
        nq = attn_weights.shape[0] * attn_weights.shape[1] * attn_weights.shape[2]
        prev = _MEASURE["sum"].get(layer_idx)
        if prev is None or prev.numel() != recv.numel():
            _MEASURE["sum"][layer_idx] = recv.clone()
            _MEASURE["nq"][layer_idx] = nq
        else:
            _MEASURE["sum"][layer_idx] += recv
            _MEASURE["nq"][layer_idx] += nq

    # ---- post-softmax sink-boost (thinker text decoder only) ----
    if (
        _STATE.get("active")
        and layer_idx is not None
        and type(module).__name__ == _THINKER_ATTN_NAME
        and _STATE.get("mode") in ("boost", "suppress")
    ):
        _DEBUG["called"] += 1
        l2h = _STATE.get("layer_to_heads")
        l2k = _STATE.get("layer_to_key_mask")
        if l2h and layer_idx in l2h and l2k is not None:
            heads = l2h[layer_idx]
            mask = l2k.get(layer_idx)
            if mask is not None and len(heads) > 0:
                cur_k = attn_weights.shape[-1]
                if mask.numel() != cur_k:  # KV cache grew: new tokens never sinks
                    nm = torch.zeros(cur_k, dtype=torch.bool, device=attn_weights.device)
                    nm[: mask.numel()] = mask.to(attn_weights.device)
                    mask = nm
                else:
                    mask = mask.to(attn_weights.device)
                if mask.any():
                    gamma = float(_STATE.get("gamma", 1.0))
                    sched = _STATE.get("gamma_schedule")
                    if sched is not None and layer_idx < len(sched):
                        gamma = float(sched[layer_idx])
                    scale = (1.0 + gamma) if _STATE.get("mode") == "boost" else gamma
                    hidx = torch.as_tensor(list(heads), dtype=torch.long,
                                           device=attn_weights.device)
                    sel = attn_weights[:, hidx, :, :]  # (B,H',Q,K)
                    scale_row = torch.where(
                        mask,
                        torch.full_like(mask, scale, dtype=sel.dtype),
                        torch.ones_like(mask, dtype=sel.dtype),
                    )
                    sel = sel * scale_row
                    sel = sel / sel.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    attn_weights = attn_weights.clone()
                    attn_weights[:, hidx, :, :] = sel
                    _DEBUG["modified"] += 1

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def patch(model=None):
    """Patch the module-level eager_attention_forward. The attention modules
    read this global at call time (`attention_interface = eager_attention_forward`
    inside forward), so reassigning it is picked up without touching each
    module. Returns True on success."""
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as m
    global _ORIG_EAGER
    if _ORIG_EAGER is None:
        _ORIG_EAGER = m.eager_attention_forward
    m.eager_attention_forward = _patched_eager
    return True


def unpatch():
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as m
    if _ORIG_EAGER is not None:
        m.eager_attention_forward = _ORIG_EAGER
