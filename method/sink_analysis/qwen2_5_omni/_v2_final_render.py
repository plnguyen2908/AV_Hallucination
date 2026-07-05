"""Stage 5 supp v2 — final render with SHARED GLOBAL COLOR SCALE (Fix 1).

Loads cached head_block_captures.npz (from _v2_head_block_ranking.py)
and renders 6 per-token own-frame relevance maps on ONE shared raw-value
vmax per head — sinks and non-sinks on the SAME scale.

Layout per head: 2 rows × 3 cols.
  Row 0: 3 top-norm sinks
  Row 1: 3 random non-sinks
Each panel: token's own-frame relevance map = A[h, queries_in_own_frame,
strongest_raw_of_token] reshaped to (H_raw, W_raw), bilinear-upsampled
to native frame resolution, overlaid on the actual frame.

Color scale: ONE vmax across all 6 panels of the head's figure (raw
values, no per-panel renorm). Per-token panel-max reported as fraction
of shared vmax in panel annotation.

Block 30 is WINDOWED (8×8 within-frame); attention is structurally
local. The figure caption flags this. The full-within-frame block 31
supplement is rendered if --also-blk31 is passed.

Usage:
  python _v2_final_render.py --clip 761e61816a14.mp4 \
        --block 30 --fg-head 1 --bg-head 3
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_VID = _REPO / "data/ActivityNet/videos"
OUT_ROOT = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp_v2"


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    n_total = len(vr); native_fps = float(vr.get_avg_fps())
    step = native_fps / max(qwen_fps, 1e-6)
    idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
    return [vr[i].asnumpy() for i in idxs]


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--fg-head", type=int, required=True)
    p.add_argument("--bg-head", type=int, required=True)
    p.add_argument("--also-blk31", action="store_true",
                   help="Also render block-31 supplement")
    args = p.parse_args()

    clip_stem = Path(args.clip).stem
    in_dir = OUT_ROOT / f"clip_{clip_stem}"
    npz_path = in_dir / "head_block_captures.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run head_block_ranking first.")
        return
    print(f"Loading {npz_path}")
    cap = np.load(str(npz_path), allow_pickle=True)
    T_raw = int(cap["T_raw"]); H_raw = int(cap["H_raw"])
    W_raw = int(cap["W_raw"]); n_raw = int(cap["n_raw"])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm

    sink_keys = cap["sink_keys"].tolist()
    non_keys = cap["non_keys"].tolist()
    sink_norms = cap["sink_norms"]
    non_norms = cap["non_norms"]
    sink_own_t = cap["sink_own_t"]
    non_own_t = cap["non_own_t"]

    all_meta = []
    for i, k in enumerate(sink_keys):
        all_meta.append(dict(k=int(k), kind="sink", norm=float(sink_norms[i]),
                               own_t=int(sink_own_t[i])))
    for i, k in enumerate(non_keys):
        all_meta.append(dict(k=int(k), kind="nonsink", norm=float(non_norms[i]),
                               own_t=int(non_own_t[i])))

    frames = load_raw_frames(DEFAULT_VID / args.clip, T_raw, qwen_fps=1.0)
    blocks_to_render = [args.block]
    if args.also_blk31 and 31 not in blocks_to_render:
        blocks_to_render.append(31)

    for blk in blocks_to_render:
        for kind_label, head in [("fg", args.fg_head), ("bg", args.bg_head)]:
            # Pull each token's strongest-raw own-frame map at (block, head)
            per_token = []  # list of dict(k, kind, norm, own_t, raw_idx, map, vmax)
            for m in all_meta:
                store_arr = cap[f"store_b{blk}_k{m['k']}"]  # (4, n_heads, n_raw)
                head_arr = store_arr[:, head, :]   # (4, n_raw)
                cube = head_arr.reshape(4, T_raw, H_raw, W_raw)
                own_means = cube[:, m["own_t"]].reshape(4, -1).mean(axis=1)
                ri = int(np.argmax(own_means))
                relevance_map = cube[ri, m["own_t"]]  # (H_raw, W_raw)
                per_token.append(dict(
                    k=m["k"], kind=m["kind"], norm=m["norm"],
                    own_t=m["own_t"], raw_idx=ri,
                    map=relevance_map,
                    vmax=float(relevance_map.max()),
                    col_mean=float(relevance_map.mean())))

            # SHARED vmax across all 6 panels (Fix 1)
            shared_vmax = max(pt["vmax"] for pt in per_token)
            print(f"\nblk {blk} h{head} ({kind_label}): shared_vmax = "
                  f"{shared_vmax:.4e} ({shared_vmax*100:.2f}%)")
            print(f"  per-token panel max / shared vmax fraction:")
            for pt in per_token:
                marker = "S" if pt["kind"] == "sink" else "N"
                frac = pt["vmax"] / shared_vmax * 100
                print(f"    [{marker}] k={pt['k']:4d}  "
                      f"norm={pt['norm']:6.2f}  "
                      f"panel_max={pt['vmax']*100:.2f}%  "
                      f"(={frac:.1f}% of shared vmax)  "
                      f"col_mean={pt['col_mean']*100:.3f}%")

            # Render 2x3 grid
            cmap = plt.get_cmap("turbo")
            fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                                       constrained_layout=True)
            sink_panel_maxes = []
            non_panel_maxes = []
            for ri_row, kind in enumerate(["sink", "nonsink"]):
                tokens_this_row = [pt for pt in per_token
                                     if pt["kind"] == kind]
                for ci, pt in enumerate(tokens_this_row):
                    ax = axes[ri_row, ci]
                    h_map = pt["map"]
                    norm_map = h_map / max(shared_vmax, 1e-30)  # [0,1] on shared scale
                    norm_map = np.clip(norm_map, 0, 1)
                    base = frames[pt["own_t"]]
                    from PIL import Image as _Im
                    fh, fw = base.shape[:2]
                    map_pil = _Im.fromarray(
                        (norm_map * 255).astype(np.uint8)).resize(
                        (fw, fh), _Im.BILINEAR)
                    h_up = np.asarray(map_pil).astype(np.float32) / 255.0
                    heat = (cmap(h_up)[..., :3] * 255).astype(np.uint8)
                    overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
                    ax.imshow(overlay)
                    ax.set_xticks([]); ax.set_yticks([])
                    kind_str = "SINK" if kind == "sink" else "non-sink"
                    title = (f"{kind_str} k={pt['k']}  "
                              f"norm={pt['norm']:.1f}  t={pt['own_t']}s\n"
                              f"panel max = {pt['vmax']*100:.2f}% "
                              f"(={pt['vmax']/shared_vmax*100:.0f}% of "
                              f"shared)")
                    ax.set_title(title, fontsize=10,
                                  color="darkred" if kind == "sink" else "darkblue")
                    if kind == "sink":
                        sink_panel_maxes.append(pt["vmax"])
                    else:
                        non_panel_maxes.append(pt["vmax"])
            sink_mean_max = float(np.mean(sink_panel_maxes))
            non_mean_max = float(np.mean(non_panel_maxes))

            block_type = "WINDOWED 8×8 within-frame" if blk in (29, 30) else "FULL within-frame"
            head_role = "fg-type (top sink fg_share)" if kind_label == "fg" else "bg-type (lowest sink fg_share)"
            fig.suptitle(
                f"Clip {args.clip}  |  ViT block {blk} ({block_type})  "
                f"|  head h{head} ({head_role})\n"
                f"Per-token OWN-FRAME relevance map, strict raw grain, "
                f"SHARED color scale. "
                f"Shared vmax = {shared_vmax*100:.2f}% (any-query peak attention to a sink raw).\n"
                f"Mean sink panel-max = {sink_mean_max*100:.2f}%, "
                f"mean non-sink panel-max = {non_mean_max*100:.2f}%, "
                f"ratio sink/non = {sink_mean_max/max(non_mean_max,1e-30):.2f}×.",
                fontsize=11)
            out_path = in_dir / f"final_blk{blk}_h{head}_{kind_label}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {out_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
