"""ASD (Adaptive Sink-guided Decoding) for Qwen2.5-Omni — reconstruction of the
official crossmodal-hub method (arXiv 2605.10815). Their released repo is the
causal-tracing code with the mitigation attention-scaling commented out
(model/modeling_qwen2_5_omni_low.py:1558-1569, self.alpha=0.2); we reproduce the
documented algorithm.

Per question (all eager -> shares the length ceiling; use trim + efficient encoders):
  1. hidden-states forward -> sink tokens: indices where
     max(|RMSNorm(hs)[458 or 2570]|) > TAU(25), pooled over 28 layers, top-K=400
     most frequent (logic/sink.py get_global_sink_token / run_logic).
  2. attention forward (patched 'collect') -> per sink, MDS =
     (v_score - a_score)/(v_score + a_score), v/a_score = mean attention from
     LATER video/audio query tokens to the sink (future_looks_sink). Cross-modal
     sinks = |MDS| <= mds_thr (attended by both modalities).
  3. boost forward (patched 'boost') -> attn[...,-1,cross_sinks] += alpha*|attn|
     (alpha=0.2), renormalise; greedy-decode the answer letter.
"""
import math
import types

import torch
import torch.nn as nn

D_SINK = [458, 2570]
TAU = 25.0

_ASD = dict(mode=None, sink_cols=None, layer_attn=None, cross=None,
            alpha=0.2, n_layers=28)
_ORIG = None


def _rmsnorm_abs(hs, eps=1e-6):
    hs = hs.to(torch.float32)
    var = hs.pow(2).mean(-1, keepdim=True)
    return (hs * torch.rsqrt(var + eps)).abs()


def sinks_from_hidden(hidden_states, K=400):
    """hidden_states: tuple/list of per-layer (1,S,D). Returns sorted sink token
    indices (top-K most frequent across layers, high-norm on D_SINK dims)."""
    from collections import Counter
    allidx = []
    dsink = torch.tensor(D_SINK)
    for hs in hidden_states:
        rn = _rmsnorm_abs(hs[0].float().cpu())           # (S,D)
        mx = rn[:, dsink].max(dim=-1).values              # (S,)
        allidx += torch.nonzero(mx > TAU).flatten().tolist()
    if not allidx:
        return []
    return [i for i, _ in Counter(allidx).most_common(K)]


# ---- patched thinker attention -------------------------------------------
def _asd_forward(self, hidden_states, attention_mask=None, position_ids=None,
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

    mode = _ASD["mode"]
    if mode is not None and q_len != 1 and layer_idx is not None and layer_idx < _ASD["n_layers"]:
        if mode == "collect":
            cols = _ASD["sink_cols"]
            if cols is not None and len(cols):
                # mean over heads, attention TO sink columns from every query row
                mh = attn[0].mean(0)[:, cols].detach().float().cpu()  # (S, n_sink)
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
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    return self.o_proj(out), None, past_key_value


def patch_thinker_asd(model):
    global _ORIG
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
    if _ORIG is None:
        _ORIG = Qwen2_5OmniAttention.forward
    Qwen2_5OmniAttention.forward = _asd_forward
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2_5OmniAttention) and ".model.layers." in name:
            if hasattr(mod, "_old_forward"):
                mod._old_forward = types.MethodType(_asd_forward, mod)
            else:
                mod.forward = types.MethodType(_asd_forward, mod)
            n += 1
    try:
        _ASD["n_layers"] = len(model.thinker.model.layers)
    except Exception:
        pass
    print(f"[asd] patched {n} thinker attn modules", flush=True)
    return n


def compute_cross_modal_sinks(sink_cols, video_idx, audio_idx, mds_thr=0.3):
    """From the collected per-layer mean-head attention to sinks (_ASD['layer_attn'],
    list of (S, n_sink)), compute MDS per sink and return cross-modal sink token
    positions (|MDS| <= mds_thr). future_looks_sink: only query rows AFTER the sink."""
    if not _ASD["layer_attn"]:
        return []
    A = torch.stack(_ASD["layer_attn"]).mean(0)          # (S, n_sink) mean over layers
    S = A.shape[0]
    vset = torch.zeros(S, dtype=torch.bool); vset[torch.as_tensor(video_idx)] = True
    aset = torch.zeros(S, dtype=torch.bool); aset[torch.as_tensor(audio_idx)] = True
    cross = []
    for j, sink in enumerate(sink_cols):
        later = torch.arange(S) > sink
        vq = later & vset
        aq = later & aset
        v_score = A[vq, j].mean().item() if vq.any() else 0.0
        a_score = A[aq, j].mean().item() if aq.any() else 0.0
        mds = (v_score - a_score) / (v_score + a_score + 1e-9)
        if abs(mds) <= mds_thr:
            cross.append(int(sink))
    return cross


def reset(mode):
    _ASD["mode"] = mode
    if mode == "collect":
        _ASD["layer_attn"] = []


def clear():
    _ASD["mode"] = None
    _ASD["layer_attn"] = None
