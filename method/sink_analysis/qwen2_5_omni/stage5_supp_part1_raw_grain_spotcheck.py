"""
stage5_supp_part1_raw_grain_spotcheck.py — spot-check at strict raw-patch
grain. The PART 1 per-token script averages the 4 raw key columns that
compose one post-merger token (the unit at which sinks are identified).
This script renders the 4 raw key columns SEPARATELY for the top-norm
sink k=505 in clip 021bd34fe8ff, at h9 (fg) and h7 (bg), block 31.

If one of the 4 raw cells is "the real register" and mean-of-4 dilutes
it 4×, this view will show that — one of the 4 rows in the output figure
will be visibly broader/stronger than the other 3. If all 4 raw columns
look similar (same dim-or-local pattern), mean-of-4 is faithful and the
doubly-null finding stands at raw grain too.

Output: stage5_supp/clip_021bd34fe8ff/per_token/raw_grain_k505_h{9,7}.png
        (rows = 4 raw patches, cols = T_raw frames)
"""
import json, math, sys, subprocess, tempfile
from pathlib import Path

import matplotlib.pyplot as plt
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
OUT_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp/clip_021bd34fe8ff/per_token"
VIDEO_TOKEN_ID = 151656
CLIP = "021bd34fe8ff.mp4"
K_POST = 505
PICK_BLOCK = 31
HEADS = {9: "fg", 7: "bg"}

_attn_callback = None


def patched_vision_attn_forward(self, hidden_states, cu_seqlens,
                                 rotary_pos_emb=None):
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    attention_mask = torch.full(
        [1, seq_length, seq_length], torch.finfo(q.dtype).min,
        device=q.device, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i],
                       cu_seqlens[i-1]:cu_seqlens[i]] = 0
    q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                          dtype=torch.float32).to(q.dtype)
    if _attn_callback is not None:
        _attn_callback(self, attn_weights, cu_seqlens)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


Qwen2_5OmniVisionAttention.forward = patched_vision_attn_forward


def post_to_raw_indices(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    t = k_post // (H_post * W_post)
    R = (k_post % (H_post * W_post)) // W_post
    C = k_post % W_post
    out = []
    for dR in range(sm):
        for dC in range(sm):
            raw_idx = t * H_raw * W_raw + (sm*R + dR) * W_raw + (sm*C + dC)
            out.append(dict(idx=raw_idx, t=t, r=sm*R+dR, c=sm*C+dC,
                             dR=dR, dC=dC))
    return out


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    n_total = len(vr); native_fps = float(vr.get_avg_fps())
    step = native_fps / max(qwen_fps, 1e-6)
    idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
    return [vr[i].asnumpy() for i in idxs]


def render_grid(rel_4xT, frames, T_raw, H_raw, W_raw, head_idx, tag,
                 raw_meta, png_path, k_post):
    """rel_4xT shape (4, T_raw, H_raw, W_raw). 4 rows = 4 raw patches
    composing k_post; T_raw cols = frames. GLOBAL normalization across
    all 4*T panels."""
    # Per-raw col_mean (mean attention each raw key column receives)
    per_raw_mean = rel_4xT.reshape(4, -1).mean(axis=1)

    vmin = float(rel_4xT.min()); vmax = float(rel_4xT.max())
    eps = max(vmax - vmin, 1e-12)
    norm = (rel_4xT - vmin) / eps
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(4, T_raw, figsize=(3 * T_raw, 12),
                              constrained_layout=True)
    for ri in range(4):
        rm = raw_meta[ri]
        for ti in range(T_raw):
            ax = axes[ri, ti] if T_raw > 1 else axes[ri]
            h_map = norm[ri, ti]
            if ti < len(frames):
                base = frames[ti]
                from PIL import Image as _Im
                fh, fw = base.shape[:2]
                map_pil = _Im.fromarray((h_map * 255).astype(np.uint8)).resize(
                    (fw, fh), _Im.BILINEAR)
                h_up = np.asarray(map_pil).astype(np.float32) / 255.0
                heat = (cmap(h_up)[..., :3] * 255).astype(np.uint8)
                overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
            else:
                heat = (cmap(h_map)[..., :3] * 255).astype(np.uint8)
                overlay = heat
            ax.imshow(overlay)
            ax.set_xticks([]); ax.set_yticks([])
            if ti == 0:
                ax.set_ylabel(f"raw {ri} (r={rm['r']},c={rm['c']})\n"
                               f"col_mean={per_raw_mean[ri]:.4e}",
                               fontsize=9)
            if ri == 0:
                ax.set_title(f"t={ti}s", fontsize=10)
    fig.suptitle(
        f"{CLIP} blk{PICK_BLOCK} h{head_idx} ({tag}) — k_post={k_post} "
        f"split into 4 raw patches (rows). "
        f"Global vmin={vmin:.3e} vmax={vmax:.3e}.  "
        f"per-raw col_mean: {', '.join(f'{m:.3e}' for m in per_raw_mean)}",
        fontsize=10)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                                   device_map="balanced_low_0")
    visual_enc = None
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model.thinker, attr):
            visual_enc = getattr(model.thinker, attr); break

    data = json.load(open(DEFAULT_QA))
    d = next(x for x in data if x["video"] == CLIP)
    conv = build_conversation(str(Path(DEFAULT_VID) / d["video"]),
                                d["question"], "v")
    inputs, use_aiv = prepare_inputs(processor, conv, "v", model.device,
                                       model.dtype)
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0]
    n_llm = int(video_pos.size)
    grid = inputs["video_grid_thw"].cpu().numpy()
    T_raw = int(grid[0, 0]); H_raw = int(grid[0, 1]); W_raw = int(grid[0, 2])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm
    n_raw = T_raw * H_raw * W_raw
    assert T_raw * H_post * W_post == n_llm
    print(f"  T_raw={T_raw} H_raw={H_raw} W_raw={W_raw} n_raw={n_raw} "
          f"n_llm={n_llm}")

    raw_meta = post_to_raw_indices(K_POST, T_raw, H_raw, W_raw, sm)
    print(f"  k_post={K_POST} → 4 raw patches:")
    for rm in raw_meta:
        print(f"    raw_idx={rm['idx']:5d}  (t={rm['t']}, r={rm['r']}, c={rm['c']})")

    # Storage: per (head_idx in HEADS) → (4, n_raw) array of cols
    raw_cols = {h: np.zeros((4, n_raw), dtype=np.float32) for h in HEADS}

    # Tag block indices
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi != PICK_BLOCK: return
        if attn_weights.shape[-1] != n_raw: return
        for ri, rm in enumerate(raw_meta):
            for h in HEADS:
                col = attn_weights[h, :, rm["idx"]]  # (n_raw,)
                raw_cols[h][ri] = col.float().cpu().numpy()

    global _attn_callback
    _attn_callback = cb
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        _attn_callback = None
    torch.cuda.empty_cache()

    # Reshape each (4, n_raw) → (4, T_raw, H_raw, W_raw)
    raw_cols_4d = {h: raw_cols[h].reshape(4, T_raw, H_raw, W_raw)
                    for h in HEADS}

    frames = load_raw_frames(Path(DEFAULT_VID) / CLIP, T_raw, qwen_fps=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for h, tag in HEADS.items():
        out_png = OUT_DIR / f"raw_grain_k{K_POST}_h{h}_{tag}.png"
        render_grid(raw_cols_4d[h], frames, T_raw, H_raw, W_raw,
                     h, tag, raw_meta, out_png, K_POST)
        print(f"  wrote {out_png.relative_to(_REPO)}")
        per_raw_mean = raw_cols_4d[h].reshape(4, -1).mean(axis=1)
        max_raw = int(np.argmax(per_raw_mean))
        dilution = per_raw_mean[max_raw] / per_raw_mean.mean()
        print(f"    h{h} ({tag}): per-raw col_means = "
              f"{[f'{m:.3e}' for m in per_raw_mean]}; "
              f"strongest = raw {max_raw} (× mean = {dilution:.2f})")


if __name__ == "__main__":
    main()
