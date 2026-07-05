"""STEP 0 follow-up: per-head orientation check + axis verification.

(a) Axis check: verify A[h, q, :].sum() ≈ 1 across keys (softmax-correct
    layout).
(b) Per-head orientation: for each of the 16 heads at blk 31, compute
    mean(strongest-raw own-frame col) for the 3 top-norm sinks vs the 3
    random non-sinks. Print the per-head ratio.

This tells us whether the head-averaged FAIL is just averaging across
heads with opposite biases, or a genuine "no head where sinks dominate"
finding.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # noqa: E402
    Qwen2_5OmniVisionAttention, apply_rotary_pos_emb_vision,
)

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
VIDEO_TOKEN_ID = 151656
TAU_PROP = 100.0
PICK_BLOCK = 31

_attn_callback = None
_axis_check_row = [None]


def patched(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    am = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min,
                     device=q.device, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        am[..., cu_seqlens[i-1]:cu_seqlens[i],
            cu_seqlens[i-1]:cu_seqlens[i]] = 0
    q = q.transpose(0,1); k = k.transpose(0,1); v = v.transpose(0,1)
    aw = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.head_dim)
    aw = aw + am
    aw = nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    if _attn_callback is not None:
        _attn_callback(self, aw, cu_seqlens)
    ao = torch.matmul(aw, v).transpose(0,1).reshape(seq_length, -1)
    return self.proj(ao)


Qwen2_5OmniVisionAttention.forward = patched


def _extract(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def post_to_raw_indices(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    t = k_post // (H_post * W_post)
    R = (k_post % (H_post * W_post)) // W_post
    C = k_post % W_post
    return [dict(idx=t * H_raw * W_raw + (sm*R + dR) * W_raw + (sm*C + dC),
                  t=t, r=sm*R+dR, c=sm*C+dC)
             for dR in range(sm) for dC in range(sm)]


def post_to_frc(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    return (k_post // (H_post * W_post),
            (k_post % (H_post * W_post)) // W_post,
            k_post % W_post)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    args = p.parse_args()
    clip = args.clip

    print(f"Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                                   device_map="balanced_low_0")
    visual_enc = model.thinker.visual
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i

    data = json.load(open(DEFAULT_QA))
    qa_map = {x["video"]: x for x in data}
    question = qa_map[clip]["question"] if clip in qa_map else "Describe this video."
    vp = DEFAULT_VID / clip
    conv = build_conversation(str(vp), question, "v")
    inputs, use_aiv = prepare_inputs(processor, conv, "v",
                                       model.device, model.dtype)
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0]
    n_llm = int(video_pos.size)
    grid = inputs["video_grid_thw"].cpu().numpy()
    T_raw = int(grid[0,0]); H_raw = int(grid[0,1]); W_raw = int(grid[0,2])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm
    n_raw = T_raw * H_raw * W_raw

    # Pass 1: encoder norms (no attention capture)
    enc_buf = []
    def enc_hook(_m, _i, out):
        tok = _extract(out)
        enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    h = visual_enc.register_forward_hook(enc_hook)
    global _attn_callback
    _attn_callback = None
    with torch.inference_mode():
        model.thinker(**inputs, use_audio_in_video=use_aiv,
                      output_attentions=False, return_dict=True,
                      use_cache=False)
    h.remove()
    enc_norms = np.concatenate(enc_buf)
    sinks = np.where(enc_norms > TAU_PROP)[0]
    rng = np.random.default_rng(0)
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    top3 = sinks[np.argsort(-enc_norms[sinks])[:3]].tolist()
    rand_non = rng.choice(nonsinks, size=min(3, len(nonsinks)),
                            replace=False).tolist()
    sink_ids = [int(k) for k in top3]
    non_ids  = [int(k) for k in rand_non]
    all_ids  = sink_ids + non_ids

    # Pass 2: capture, at blk 31, for all 6 tokens × 4 raws × 16 heads,
    # AND capture row sums for axis verification.
    n_heads = visual_enc.blocks[0].attn.num_heads
    store = {(int(k), ri, hh): None
              for k in all_ids for ri in range(4) for hh in range(n_heads)}
    row_sum_check = [None, None]   # (h, q), row_sum

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi != PICK_BLOCK: return
        if attn_weights.shape[-1] != n_raw: return
        # axis check
        if row_sum_check[0] is None:
            # pick head=0, query=10 (some valid query)
            row_sum = float(attn_weights[0, 10, :].sum())
            col_sum = float(attn_weights[0, :, 10].sum())
            row_sum_check[0] = row_sum
            row_sum_check[1] = col_sum
        for k in all_ids:
            raws_ = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
            for ri, rm in enumerate(raws_):
                for hh in range(n_heads):
                    store[(int(k), ri, hh)] = (
                        attn_weights[hh, :, rm["idx"]].float().cpu().numpy())

    _attn_callback = cb
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        _attn_callback = None
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("STEP 0b — axis check + per-head orientation")
    print("=" * 60)

    rs, cs = row_sum_check
    print(f"\n(a) Axis check at h=0, blk {PICK_BLOCK}:")
    print(f"    A[h=0, q=10, :].sum() = {rs:.6f}  "
          f"(should be ≈ 1.0 if softmax over keys is axis -1)")
    print(f"    A[h=0, :, k=10].sum() = {cs:.6f}  "
          f"(no constraint — just for comparison)")
    if abs(rs - 1.0) < 0.02:
        print(f"    ✓ Axis correct: column = A[h, :, k_idx] is attention "
              f"FROM all queries TO key k_idx.")
    else:
        print(f"    ✗ Axis suspect: row does not sum to 1.")

    print(f"\n(b) Per-head orientation at blk {PICK_BLOCK}:")
    print(f"    For each of 16 heads, compute mean(strongest-raw "
          f"own-frame col_mean) for the 3 top sinks vs 3 random "
          f"non-sinks. Ratio = sink_mean / non_sink_mean. Reported "
          f"sorted by ratio descending.")
    print()
    print(f"    {'head':>4s}  {'sink_mean':>11s}  {'non_mean':>11s}  "
          f"{'ratio':>7s}  {'sink_max':>11s}  {'non_max':>11s}")
    per_head = []
    for hh in range(n_heads):
        sink_means = []
        non_means = []
        sink_maxes = []
        non_maxes = []
        for k in all_ids:
            own_t, _, _ = post_to_frc(int(k), T_raw, H_raw, W_raw, sm)
            best_raw_mean = -1.0; best_raw_max = -1.0
            for ri in range(4):
                col = store[(int(k), ri, hh)]
                cube = col.reshape(T_raw, H_raw, W_raw)
                m = float(cube[own_t].mean())
                if m > best_raw_mean:
                    best_raw_mean = m
                    best_raw_max = float(cube[own_t].max())
            if k in sink_ids:
                sink_means.append(best_raw_mean)
                sink_maxes.append(best_raw_max)
            else:
                non_means.append(best_raw_mean)
                non_maxes.append(best_raw_max)
        sm_ = float(np.mean(sink_means)); nm_ = float(np.mean(non_means))
        per_head.append(dict(h=hh, sink_mean=sm_, non_mean=nm_,
                               ratio=sm_/max(nm_, 1e-30),
                               sink_max=float(np.mean(sink_maxes)),
                               non_max=float(np.mean(non_maxes))))
    per_head.sort(key=lambda x: -x["ratio"])
    for ph in per_head:
        flag = "  ← SINK > NON-SINK" if ph["ratio"] > 1.0 else ""
        print(f"    {ph['h']:>4d}  {ph['sink_mean']:.4e}  "
              f"{ph['non_mean']:.4e}  {ph['ratio']:.3f}  "
              f"{ph['sink_max']:.4e}  {ph['non_max']:.4e}{flag}")

    n_pos = sum(1 for ph in per_head if ph["ratio"] > 1.0)
    print(f"\n    Summary: {n_pos}/{n_heads} heads have sink_mean > "
          f"non_sink_mean at blk {PICK_BLOCK}.")
    if n_pos > 0:
        best = per_head[0]
        print(f"    Best fg-candidate head: h{best['h']} "
              f"(ratio {best['ratio']:.3f}, sink_max {best['sink_max']*100:.2f}%, "
              f"non_max {best['non_max']*100:.2f}%)")
    else:
        print(f"    No head at blk {PICK_BLOCK} has sink > non-sink on "
              f"this clip. The head-averaged FAIL is structural for this "
              f"clip, not just averaging. May need to try blk 23 / 29 / 30.")
    print("\n=" * 30)
    print("STOP. Awaiting confirmation before head ranking.")


if __name__ == "__main__":
    main()
