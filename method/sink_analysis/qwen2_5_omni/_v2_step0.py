"""STEP 0 sanity check for Stage 5 supp v2 on the chosen clip.

Checks:
  (1) Grid-map round-trip on a known token index.
  (2) Cross-frame masking status (cite + verify cu_seqlens).
  (3) Key-column orientation at strict raw grain: mean strongest-raw
      column for top-3 sinks vs 3 random non-sinks. Raw values. Sinks
      must be >> non-sinks; otherwise the axis is transposed.

STOP after printing. No rendering, no caching.
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
_cu_log = {}


def patched(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    bi = getattr(self, "_blk_idx", -1)
    if bi == PICK_BLOCK and bi not in _cu_log:
        _cu_log[bi] = cu_seqlens.detach().cpu().numpy().copy()
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
    out = []
    for dR in range(sm):
        for dC in range(sm):
            raw_idx = t * H_raw * W_raw + (sm*R + dR) * W_raw + (sm*C + dC)
            out.append(dict(idx=raw_idx, t=t, r=sm*R+dR, c=sm*C+dC))
    return out


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

    print("\n" + "=" * 60)
    print("STEP 0 — SANITY CHECKS")
    print("=" * 60)

    print(f"\nClip: {clip}")
    print(f"  T_raw={T_raw}, H_raw={H_raw}, W_raw={W_raw}, n_raw={n_raw}")
    print(f"  T_post={T_raw}, H_post={H_post}, W_post={W_post}")
    print(f"  n_llm video tokens = {n_llm}")

    print("\n(1) Grid-map round-trip")
    mid = n_llm // 2
    t = mid // (H_post * W_post)
    r = (mid % (H_post * W_post)) // W_post
    c = mid % W_post
    back = t * H_post * W_post + r * W_post + c
    raws = post_to_raw_indices(mid, T_raw, H_raw, W_raw, sm)
    assert (T_raw * H_post * W_post) == n_llm, \
        f"layout mismatch: {T_raw*H_post*W_post} != {n_llm}"
    print(f"  T_post*H_post*W_post = {T_raw}*{H_post}*{W_post} = "
          f"{T_raw*H_post*W_post}  (matches n_llm={n_llm} ✓)")
    print(f"  k=mid={mid} → (t={t},R_post={r},C_post={c}) → back={back}  "
          f"{'✓' if back==mid else 'MISMATCH'}")
    print(f"  4 raw indices for k={mid}: "
          f"{[(rm['t'], rm['r'], rm['c']) for rm in raws]} "
          f"→ raw_idx {[rm['idx'] for rm in raws]}")

    # Pass 1: encoder norms
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
    print(f"\n  encoder norms (post-merger): n={enc_norms.size}, "
          f"P_prop sinks (>τ={TAU_PROP}) = {sinks.size}")

    # Tag blocks for the cu_seqlens probe
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i

    # Pick top-3 sinks + 3 random non-sinks (seed=0)
    rng = np.random.default_rng(0)
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    top3 = sinks[np.argsort(-enc_norms[sinks])[:3]].tolist()
    rand_non = rng.choice(nonsinks, size=min(3, len(nonsinks)),
                            replace=False).tolist()
    selected = [("sink", int(k), enc_norms[int(k)]) for k in top3] \
                + [("nonsink", int(k), enc_norms[int(k)]) for k in rand_non]

    # Pass 2: capture attention at block 31 for the 6 tokens × 4 raws
    n_heads = visual_enc.blocks[0].attn.num_heads
    store = {(int(k), ri, h): None
              for _, k, _ in selected
              for ri in range(4)
              for h in range(n_heads)}

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi != PICK_BLOCK: return
        if attn_weights.shape[-1] != n_raw: return
        for _, k, _ in selected:
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

    print("\n(2) Cross-frame masking status")
    cs = _cu_log[PICK_BLOCK]
    sizes = np.diff(cs)
    is_per_frame = (len(sizes) == T_raw and all(s == H_raw*W_raw for s in sizes))
    print(f"  cu_seqlens at blk {PICK_BLOCK}: {cs.tolist()}")
    print(f"  block sizes: {sizes.tolist()} (n_blocks={len(sizes)})")
    print(f"  T_raw frames * (H_raw*W_raw) = {T_raw} * {H_raw*W_raw} "
          f"= {T_raw * H_raw * W_raw}")
    if is_per_frame:
        print(f"  → PER-FRAME block-diagonal at this FULL-attn block "
              f"(blk {PICK_BLOCK}). Cross-frame attention = 0 by mask. "
              f"OWN-FRAME-only renders will follow.")
    else:
        print(f"  → NOT per-frame at this block — different structure.")

    print("\n(3) Key-column orientation (strongest-raw col_mean, raw values, "
          f"head-averaged at blk {PICK_BLOCK}, OWN-FRAME only)")
    sink_means = []
    non_means = []
    for kind, k, norm in selected:
        own_t, own_r, own_c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        # For each raw cell, compute own-frame col mean (averaged over heads)
        raw_own_means = []
        for ri in range(4):
            # head-averaged column
            col_h_avg = np.mean([store[(int(k), ri, hh)]
                                  for hh in range(n_heads)], axis=0)
            cube = col_h_avg.reshape(T_raw, H_raw, W_raw)
            own_mean = float(cube[own_t].mean())
            raw_own_means.append(own_mean)
        strongest = int(np.argmax(raw_own_means))
        strongest_mean = raw_own_means[strongest]
        marker = "S" if kind == "sink" else "N"
        print(f"  [{marker}] k={int(k):4d}  norm={float(norm):6.2f}  "
              f"own_t={own_t}  "
              f"raw_own_means={[f'{m:.3e}' for m in raw_own_means]}  "
              f"strongest=raw{strongest}={strongest_mean:.3e}")
        if kind == "sink":
            sink_means.append(strongest_mean)
        else:
            non_means.append(strongest_mean)

    sm_mean = np.mean(sink_means); nm_mean = np.mean(non_means)
    ratio = sm_mean / max(nm_mean, 1e-30)
    print(f"\n  mean over 3 sinks (strongest-raw own-frame col_mean): "
          f"{sm_mean:.3e}")
    print(f"  mean over 3 non-sinks: {nm_mean:.3e}")
    print(f"  RATIO sink/non-sink = {ratio:.2f}x")
    if ratio > 1.5:
        print(f"  ✓ orientation OK: sinks receive more attention than "
              f"non-sinks at the strongest-raw own-frame level.")
    elif ratio > 1.0:
        print(f"  ~ borderline: sinks slightly above non-sinks. "
              f"Proceed with caution.")
    else:
        print(f"  ✗ ORIENTATION FAIL: sinks <= non-sinks. The strongest "
              f"sink raw doesn't carry more attention than a typical "
              f"non-sink raw — investigate.")

    print("\n" + "=" * 60)
    print("STEP 0 complete. STOP for confirmation before head ranking.")
    print("=" * 60)


if __name__ == "__main__":
    main()
