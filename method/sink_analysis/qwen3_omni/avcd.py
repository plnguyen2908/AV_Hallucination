"""avcd.py — AVCD (Audio-Visual Contrastive Decoding) for Qwen3-Omni.

Faithful port of method/sink_analysis/qwen2_5_omni/_5_avcd_qwen.py. Qwen3
dispatches attention through the module-level `eager_attention_forward`, so we
patch THAT (collect / mask modes on the thinker-text decoder attention),
generalize the hard-coded 28-layer constants to n_layers, and read Qwen3's
audio/video token ids. Algorithm (per yes/no decision):
  1. full forward (collect last-query attention) -> modality dominance + a
     per-head-median attention threshold.
  2. entropy gate: H(softmax(orig)) < 0.6 -> skip CD, use orig logits.
  3. dominant modality picks a triple of contrastive contexts; each masks the
     high-attention (> thr) tokens of the named spans, renormalizes, re-forwards.
  4. contrastive = (2+2a)orig - 2a o1 + o2 + o3 ; plausibility-cutoff on orig.
"""
import math
import torch
import torch.nn as nn

AUDIO_TOKEN_ID = 151675
VIDEO_TOKEN_ID = 151656
_THINKER_ATTN = "Qwen3OmniMoeThinkerTextAttention"

_AVCD = dict(mode=None, modality=None, threshold=None, span_keys=None,
             layer_last_q=None, n_layers=48, span_len=None)
_ORIG_EAGER = None


def reset_collect():
    _AVCD["mode"] = "collect"; _AVCD["modality"] = None; _AVCD["layer_last_q"] = []


def set_mask_mode(modality, threshold):
    _AVCD["mode"] = "mask"; _AVCD["modality"] = modality; _AVCD["threshold"] = float(threshold)


def clear_avcd():
    _AVCD["mode"] = None; _AVCD["modality"] = None; _AVCD["layer_last_q"] = None


def set_spans_from_ids(input_ids):
    ids = input_ids[0]
    S = ids.shape[0]
    vid = (ids == VIDEO_TOKEN_ID).cpu()
    aud = (ids == AUDIO_TOKEN_ID).cpu()
    lang = ~(vid | aud)
    _AVCD["span_keys"] = {"V": vid, "A": aud, "L": lang}
    _AVCD["span_len"] = {"V": int(vid.sum()), "A": int(aud.sum()), "L": int(lang.sum())}


def _patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn = attn + attention_mask[:, :, :, : k.shape[-2]]
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)

    layer_idx = getattr(module, "layer_idx", None)
    mode = _AVCD["mode"]
    if (mode is not None and attn.shape[2] != 1 and layer_idx is not None
            and type(module).__name__ == _THINKER_ATTN):
        last = attn[:, :, -1, :]  # (B,H,S)
        if mode == "collect" and layer_idx < _AVCD["n_layers"]:
            _AVCD["layer_last_q"].append(last[0].detach().float().cpu())  # (H,S)
        elif mode == "mask" and layer_idx < _AVCD["n_layers"] - 1:
            thr = _AVCD["threshold"]
            keymask = torch.zeros(last.shape[-1], dtype=torch.bool)
            for letter in _AVCD["modality"]:
                keymask |= _AVCD["span_keys"][letter]
            keymask = keymask.to(last.device)
            row = attn[:, :, -1, :].clone()
            keep = (row <= thr) | (~keymask)
            row = row * keep.to(row.dtype)
            row = row / row.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn = attn.clone()
            attn[:, :, -1, :] = row

    out = torch.matmul(attn, v).transpose(1, 2).contiguous()
    return out, attn


def patch(model):
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as m
    global _ORIG_EAGER
    if _ORIG_EAGER is None:
        _ORIG_EAGER = m.eager_attention_forward
    m.eager_attention_forward = _patched_eager
    try:
        _AVCD["n_layers"] = len(model.thinker.model.layers)
    except Exception:
        pass
    print(f"[avcd-qwen3] patched eager attention; n_layers={_AVCD['n_layers']}", flush=True)


def unpatch():
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as m
    if _ORIG_EAGER is not None:
        m.eager_attention_forward = _ORIG_EAGER


def _dominance_and_threshold():
    layers = _AVCD["layer_last_q"]        # list of (H,S), one per decoder layer
    span = _AVCD["span_keys"]; slen = _AVCD["span_len"]
    L = len(layers)
    lqsum = None; domi = {"V": [], "A": [], "L": []}
    for ind, lq in enumerate(layers):
        if ind == L - 1:                  # skip last layer (as in the original)
            continue
        lqsum = lq.clone() if lqsum is None else lqsum + lq
        mean_h = lq.mean(0)
        for m_ in ("V", "A", "L"):
            n = max(slen[m_], 1)
            domi[m_].append((mean_h[span[m_]].sum() / n).item())
    lqsum = lqsum / max(L - 1, 1)
    thr = torch.quantile(lqsum.float(), 0.5, dim=-1).mean().item()
    avg = {m_: abs(sum(domi[m_]) / max(len(layers), 1)) for m_ in ("V", "A", "L")}
    name = {"V": "video", "A": "audio", "L": "language"}
    order = sorted([(name[m_], avg[m_]) for m_ in ("V", "A", "L")],
                   key=lambda x: x[1], reverse=True)
    return order, thr


_TRIPLE = {"language": ("VA", "A", "V"), "video": ("LA", "A", "L"),
           "audio": ("LV", "V", "L")}


def _forward_logits(model, inputs, use_aiv):
    with torch.inference_mode():
        out = model.thinker(**inputs, use_cache=False)
    return out.logits[:, -1, :]


def avcd_answer_logits(model, inputs, use_aiv, cd_alpha=2.5, beta=0.2, entropy_gate=0.6):
    reset_collect()
    orig = _forward_logits(model, inputs, use_aiv)
    order, thr = _dominance_and_threshold()
    clear_avcd()
    cutoff = math.log(beta) + orig.max(dim=-1, keepdim=True).values
    p = torch.softmax(orig, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    if ent.item() < entropy_gate:
        return orig, dict(gated=True, dominant=order[0][0])
    outs = []
    for mod in _TRIPLE[order[0][0]]:
        set_mask_mode(mod, thr)
        outs.append(_forward_logits(model, inputs, use_aiv))
        clear_avcd()
    o1, o2, o3 = outs
    a = float(cd_alpha)
    contrastive = (2 + 2 * a) * orig - 2 * a * o1 + o2 + o3
    return contrastive.masked_fill(orig < cutoff, -1e-4), dict(gated=False, dominant=order[0][0])
