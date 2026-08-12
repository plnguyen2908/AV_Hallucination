"""asd.py — ASD (Adaptive Sink-guided Decoding) for Qwen3-Omni.

Faithful port of method/sink_analysis/qwen2_5_omni/_5_asd_qwen.py with the SAME
standard config (TAU=25, alpha=0.2, mds_thr=0.3, K=400) — only D_SINK is the
model-appropriate Qwen3 sink dim [1992] (identified via VAR method). Patches the
module-level eager_attention_forward (collect / boost) on thinker-text attention.
Per yes/no decision:
  1. hidden-states forward -> sink tokens (|RMSNorm(hs)[1992]| > TAU, top-K=400
     most frequent across layers).
  2. collect: mean-head attention TO sinks from every query -> per-layer (S,n_sink).
  3. cross-modal sinks = |MDS| <= mds_thr, MDS=(v-a)/(v+a) from LATER video/audio
     query rows.
  4. boost: attn[...,-1,cross_sinks] += alpha*|attn|, renormalize -> answer logits.
"""
import math
from collections import Counter
import torch
import torch.nn as nn

D_SINK = [1992]
TAU = 25.0
AUDIO_TOKEN_ID = 151675
VIDEO_TOKEN_ID = 151656
_THINKER_ATTN = "Qwen3OmniMoeThinkerTextAttention"
_ASD = dict(mode=None, sink_cols=None, layer_attn=None, cross=None, alpha=0.2, n_layers=48)
_ORIG_EAGER = None


def _rmsnorm_abs(hs, eps=1e-6):
    hs = hs.to(torch.float32)
    return (hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + eps)).abs()


def sinks_from_hidden(hidden_states, K=400):
    allidx = []
    dsink = torch.tensor(D_SINK)
    for hs in hidden_states:
        rn = _rmsnorm_abs(hs[0].float().cpu())
        mx = rn[:, dsink].max(dim=-1).values
        allidx += torch.nonzero(mx > TAU).flatten().tolist()
    if not allidx:
        return []
    return [i for i, _ in Counter(allidx).most_common(K)]


def _patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn = attn + attention_mask[:, :, :, : k.shape[-2]]
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)

    layer_idx = getattr(module, "layer_idx", None)
    mode = _ASD["mode"]
    if (mode is not None and attn.shape[2] != 1 and layer_idx is not None
            and layer_idx < _ASD["n_layers"] and type(module).__name__ == _THINKER_ATTN):
        if mode == "collect":
            cols = _ASD["sink_cols"]
            if cols is not None and len(cols):
                mh = attn[0].mean(0)[:, cols].detach().float().cpu()  # (S,n_sink)
                _ASD["layer_attn"].append(mh)
        elif mode == "boost":
            cross = _ASD["cross"]
            if cross is not None and len(cross):
                row = attn[:, :, -1, :].clone()
                a = _ASD["alpha"]
                row[:, :, cross] = row[:, :, cross] + a * row[:, :, cross].abs()
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
        _ASD["n_layers"] = len(model.thinker.model.layers)
    except Exception:
        pass
    print(f"[asd-qwen3] patched eager attention; n_layers={_ASD['n_layers']}", flush=True)


def unpatch():
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as m
    if _ORIG_EAGER is not None:
        m.eager_attention_forward = _ORIG_EAGER


def compute_cross_modal_sinks(sink_cols, video_idx, audio_idx, mds_thr=0.3):
    if not _ASD["layer_attn"]:
        return []
    A = torch.stack(_ASD["layer_attn"]).mean(0)  # (S,n_sink)
    S = A.shape[0]
    vset = torch.zeros(S, dtype=torch.bool); vset[torch.as_tensor(video_idx)] = True if len(video_idx) else vset
    aset = torch.zeros(S, dtype=torch.bool); aset[torch.as_tensor(audio_idx)] = True if len(audio_idx) else aset
    cross = []
    for j, sink in enumerate(sink_cols):
        later = torch.arange(S) > sink
        vq = later & vset; aq = later & aset
        vs = A[vq, j].mean().item() if vq.any() else 0.0
        as_ = A[aq, j].mean().item() if aq.any() else 0.0
        mds = (vs - as_) / (vs + as_ + 1e-9)
        if abs(mds) <= mds_thr:
            cross.append(int(sink))
    return cross


def _set(mode):
    _ASD["mode"] = mode
    if mode == "collect":
        _ASD["layer_attn"] = []


def _clear():
    _ASD["mode"] = None; _ASD["layer_attn"] = None


def asd_answer_logits(model, inputs, use_aiv, mds_thr=0.3):
    """One ASD yes/no decision. Returns boosted next-token logits."""
    ids = inputs["input_ids"][0]
    video_idx = torch.nonzero(ids == VIDEO_TOKEN_ID).flatten().tolist()
    audio_idx = torch.nonzero(ids == AUDIO_TOKEN_ID).flatten().tolist()
    # 1. hidden states -> sinks
    with torch.inference_mode():
        out = model.thinker(**inputs, use_cache=False, output_hidden_states=True)
    sink_cols = sinks_from_hidden(out.hidden_states)
    if not sink_cols:
        return out.logits[:, -1, :], dict(n_sink=0, n_cross=0)
    _ASD["sink_cols"] = sink_cols
    # 2. collect attention to sinks
    _set("collect")
    with torch.inference_mode():
        model.thinker(**inputs, use_cache=False)
    _clear()
    # 3. cross-modal sinks
    cross = compute_cross_modal_sinks(sink_cols, video_idx, audio_idx, mds_thr)
    _ASD["cross"] = cross
    if not cross:
        return out.logits[:, -1, :], dict(n_sink=len(sink_cols), n_cross=0)
    # 4. boost forward
    _set("boost")
    with torch.inference_mode():
        bo = model.thinker(**inputs, use_cache=False)
    _clear()
    return bo.logits[:, -1, :], dict(n_sink=len(sink_cols), n_cross=len(cross))
