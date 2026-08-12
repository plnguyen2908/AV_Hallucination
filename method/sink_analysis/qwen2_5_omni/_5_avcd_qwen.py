"""AVCD (Audio-Visual Contrastive Decoding) reimplemented for Qwen2.5-Omni.

Faithful port of the official VideoLLaMA2 implementation
(kaistmm/AVCD, `videollama2/__init__.py::mm_infer` + `model/qwen.py`), which
targets VideoLLaMA2 only. The algorithm is model-agnostic; we reproduce it on
the Qwen2.5-Omni thinker.

Per generation step:
  1. FULL forward (no masking) -> orig_logits, per-modality attention
     dominance, and a threshold. Dominance of modality m at a layer =
     (sum of the last query's attention to m's tokens) / |m|, averaged over
     layers (last layer skipped) and heads. threshold = per-head median of the
     layer-summed last-query attention, averaged over heads.
  2. Adaptive plausibility cutoff = log(beta) + max(orig_logits), beta=0.2.
  3. Entropy gate: if H(softmax(orig_logits)) < 0.6 -> skip CD, use orig.
  4. Else pick 3 contrastive contexts by dominant modality:
       language -> (VA, A, V) ; video -> (LA, A, L) ; audio -> (LV, V, L)
     each masks the *high-attention* (> threshold) tokens of the named spans of
     the last query, renormalises, and returns last-token logits out1/out2/out3.
  5. contrastive = (2+2a)*orig - 2a*out1 + out2 + out3        (a = cd_alpha)
     next = contrastive.masked_fill(orig < cutoff, -1e-4) ; argmax.

Attention weights must be materialised (we read/modify the last query row), so
the thinker must run eager -> shares the ~10k-token length ceiling (see
results/qwen2_5_omni/worldsense/LENGTH_FINDINGS.md). Use the efficient-encoder
patch so the vision/audio encoders don't OOM.

Usage (per sample):
    from _5_avcd_qwen import patch_thinker_avcd, set_spans, avcd_answer_logits
    patch_thinker_avcd(model); patch_efficient_encoders(model)
    set_spans(video_idx, audio_idx, seq_len, n_layers=model...num_layers)
    logits = avcd_answer_logits(model, inputs, use_aiv, cd_alpha=2.5)
"""
import math
import types
from typing import Optional

import torch
import torch.nn as nn

# ---- global state ---------------------------------------------------------
_AVCD = dict(mode=None, modality=None, threshold=None,
             span_keys=None, layer_last_q=None, n_layers=28)
_ORIG_FWD = None


def reset_collect():
    _AVCD["mode"] = "collect"
    _AVCD["modality"] = None
    _AVCD["layer_last_q"] = []  # per-layer (H, S) last-query attention (prefill)


def set_mask_mode(modality, threshold):
    _AVCD["mode"] = "mask"
    _AVCD["modality"] = modality
    _AVCD["threshold"] = float(threshold)


def clear_avcd():
    _AVCD["mode"] = None
    _AVCD["modality"] = None
    _AVCD["layer_last_q"] = None


def set_spans(video_idx, audio_idx, seq_len, device=None):
    """Precompute boolean key masks (seq_len,) for video / audio / language,
    stored on CPU (the model is sharded across GPUs, so per-layer tensors live
    on different devices; masks are moved to the right device on use).
    language = every non-audio, non-video position (system+query+generated)."""
    vid = torch.zeros(seq_len, dtype=torch.bool)
    aud = torch.zeros(seq_len, dtype=torch.bool)
    if len(video_idx):
        vid[torch.as_tensor(video_idx)] = True
    if len(audio_idx):
        aud[torch.as_tensor(audio_idx)] = True
    lang = ~(vid | aud)
    _AVCD["span_keys"] = {"V": vid, "A": aud, "L": lang}
    _AVCD["span_len"] = {"V": int(vid.sum()), "A": int(aud.sum()),
                         "L": int(lang.sum())}


# ---- patched thinker attention -------------------------------------------
def _avcd_forward(self, hidden_states, attention_mask=None, position_ids=None,
                  past_key_value=None, output_attentions=False, use_cache=False,
                  cache_position=None, position_embeddings=None):
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_multimodal_rotary_pos_emb, repeat_kv as _repeat_kv)
    layer_idx = getattr(self, "layer_idx", None)
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling["mrope_section"])
    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx,
                                     {"sin": sin, "cos": cos, "cache_position": cache_position})
    k = _repeat_kv(k, self.num_key_value_groups)
    v = _repeat_kv(v, self.num_key_value_groups)

    attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn = attn + attention_mask[:, :, :, : k.shape[-2]]
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

    mode = _AVCD["mode"]
    if mode is not None and q_len != 1 and layer_idx is not None:
        last = attn[:, :, -1, :]  # (B, H, S)
        if mode == "collect" and layer_idx < _AVCD["n_layers"]:
            # .cpu() clone: last is a VIEW into the (H,S,S) attn tensor (pinning
            # it would keep all layers' full attention alive); CPU unifies the
            # per-layer tensors that otherwise live on different shard devices.
            _AVCD["layer_last_q"].append(last[0].detach().float().cpu())  # (H, S)
        elif mode == "mask" and layer_idx < 27:
            thr = _AVCD["threshold"]
            keymask = torch.zeros(last.shape[-1], dtype=torch.bool)
            for letter in _AVCD["modality"]:
                keymask |= _AVCD["span_keys"][letter]
            keymask = keymask.to(last.device)
            # suppress the > threshold tokens within the named spans (keep <=)
            row = attn[:, :, -1, :].clone()
            keep = (row <= thr) | (~keymask)  # keep low-attn or out-of-span
            row = row * keep.to(row.dtype)
            row = row / row.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn = attn.clone()
            attn[:, :, -1, :] = row

    out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    out = self.o_proj(out)
    return out, (attn if output_attentions else None), past_key_value


def patch_thinker_avcd(model):
    global _ORIG_FWD
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
    if _ORIG_FWD is None:
        _ORIG_FWD = Qwen2_5OmniAttention.forward
    Qwen2_5OmniAttention.forward = _avcd_forward
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2_5OmniAttention) and ".model.layers." in name:
            if hasattr(mod, "_old_forward"):
                mod._old_forward = types.MethodType(_avcd_forward, mod)
            else:
                mod.forward = types.MethodType(_avcd_forward, mod)
            n += 1
    try:
        _AVCD["n_layers"] = len(model.thinker.model.layers)
    except Exception:
        pass
    print(f"[avcd] patched {n} thinker attn modules; n_layers={_AVCD['n_layers']}", flush=True)
    return n


# ---- dominance + threshold from collected last-query attention ------------
def _dominance_and_threshold():
    layers = _AVCD["layer_last_q"]  # list of (H, S)
    span = _AVCD["span_keys"]
    slen = _AVCD["span_len"]
    lqsum = None
    domi = {"V": [], "A": [], "L": []}
    for ind, lq in enumerate(layers):
        if ind == 27:
            continue
        lqsum = lq.clone() if lqsum is None else lqsum + lq  # (H, S)
        mean_h = lq.mean(0)  # (S,)
        for m in ("V", "A", "L"):
            n = max(slen[m], 1)
            domi[m].append((mean_h[span[m]].sum() / n).item())
    lqsum = lqsum / (13 + 14)
    thr = torch.quantile(lqsum.float(), 0.5, dim=-1).mean().item()  # per-head median -> mean
    avg = {m: abs(sum(domi[m]) / max(len(layers), 1)) for m in ("V", "A", "L")}
    name = {"V": "video", "A": "audio", "L": "language"}
    order = sorted([(name[m], avg[m]) for m in ("V", "A", "L")], key=lambda x: x[1], reverse=True)
    return order, thr


# modality-triples keyed by dominant modality (official AVCD)
_TRIPLE = {"language": ("VA", "A", "V"),
           "video":    ("LA", "A", "L"),
           "audio":    ("LV", "V", "L")}


def _forward_logits(model, inputs, use_aiv):
    with torch.inference_mode():
        out = model.thinker(**inputs, use_audio_in_video=use_aiv, use_cache=False)
    return out.logits[:, -1, :]


def avcd_answer_logits(model, inputs, use_aiv, cd_alpha=2.5,
                       beta=0.2, entropy_gate=0.6):
    """One AVCD step. Returns next-token logits over the vocab (for MCQ argmax
    or scoring the A/B/C/D option ids). Assumes set_spans() already called."""
    # 1. full forward, collect dominance
    reset_collect()
    orig_logits = _forward_logits(model, inputs, use_aiv)
    order, thr = _dominance_and_threshold()
    clear_avcd()

    cutoff = math.log(beta) + orig_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(orig_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    if entropy.item() < entropy_gate:
        return orig_logits, dict(gated=True, dominant=order[0][0], threshold=thr)

    m1, m2, m3 = _TRIPLE[order[0][0]]
    outs = []
    for mod in (m1, m2, m3):
        set_mask_mode(mod, thr)
        outs.append(_forward_logits(model, inputs, use_aiv))
        clear_avcd()
    o1, o2, o3 = outs
    a = float(cd_alpha)
    contrastive = (2 + 2 * a) * orig_logits - 2 * a * o1 + o2 + o3
    next_logits = contrastive.masked_fill(orig_logits < cutoff, -1e-4)
    return next_logits, dict(gated=False, dominant=order[0][0], threshold=thr,
                             triple=(m1, m2, m3))
