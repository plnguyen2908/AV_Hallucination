"""Stage 5 supp v2 — blk-31 multi-scale render.

Per user request: switch to blk 31 (FULL within-frame attention), use
blk-31-native top heads h9 (fg) and h6 (bg) from the ranking, and do
NOT upsample to native frame resolution. Instead, bilinear-downsample
the raw (H_raw, W_raw) = (26, 34) heatmap to N×N for
N ∈ {5, 10, 14, 16} and render pure heatmaps with shared vmax across
the 6 panels per (head, scale).

N=16 roughly matches sink_or_not's CLIP-ViT-L/14 @ 224 patch grid.

Loads cached head_block_captures.npz from the head_block_ranking step.

Layout per (head, N): 1 row of frame strip on top + 2 rows × 3 cols of
pure heatmaps (sinks top row, non-sinks bottom row).

Output: stage5_supp_v2/clip_<id>/final_blk31_h<head>_<tag>_N<N>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_VID = _REPO / "data/ActivityNet/videos"
OUT_ROOT = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp_v2"

BLK = 31
HEADS = {9: "fg", 6: "bg"}     # blk-31-native top heads from ranking
SCALES = [5, 10, 14, 16]


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    n_total = len(vr); native_fps = float(vr.get_avg_fps())
    step = native_fps / max(qwen_fps, 1e-6)
    idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
    return [vr[i].asnumpy() for i in idxs]


def downsample_to_NxN(rel_2d, N):
    """Bilinear downsample (H, W) -> (N, N)."""
    from PIL import Image
    img = Image.fromarray(rel_2d.astype(np.float32), mode="F")
    img_d = img.resize((N, N), Image.BILINEAR)
    return np.asarray(img_d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    args = p.parse_args()

    clip_stem = Path(args.clip).stem
    in_dir = OUT_ROOT / f"clip_{clip_stem}"
    npz_path = in_dir / "head_block_captures.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        return
    print(f"Loading {npz_path}")
    cap = np.load(str(npz_path), allow_pickle=True)
    T_raw = int(cap["T_raw"]); H_raw = int(cap["H_raw"])
    W_raw = int(cap["W_raw"])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm

    sink_keys = cap["sink_keys"].tolist()
    non_keys = cap["non_keys"].tolist()
    sink_norms = cap["sink_norms"]; non_norms = cap["non_norms"]
    sink_own_t = cap["sink_own_t"]; non_own_t = cap["non_own_t"]

    all_meta = []
    for i, k in enumerate(sink_keys):
        all_meta.append(dict(k=int(k), kind="sink", norm=float(sink_norms[i]),
                               own_t=int(sink_own_t[i])))
    for i, k in enumerate(non_keys):
        all_meta.append(dict(k=int(k), kind="nonsink", norm=float(non_norms[i]),
                               own_t=int(non_own_t[i])))

    frames = load_raw_frames(DEFAULT_VID / args.clip, T_raw, qwen_fps=1.0)

    for head, kind_label in HEADS.items():
        # First: get each token's strongest-raw own-frame raw heatmap
        # at (BLK, head) — same procedure as the chosen-block render.
        per_token = []
        for m in all_meta:
            store_arr = cap[f"store_b{BLK}_k{m['k']}"]   # (4, n_heads, n_raw)
            head_arr = store_arr[:, head, :]              # (4, n_raw)
            cube = head_arr.reshape(4, T_raw, H_raw, W_raw)
            own_means = cube[:, m["own_t"]].reshape(4, -1).mean(axis=1)
            ri = int(np.argmax(own_means))
            raw_map = cube[ri, m["own_t"]]                # (H_raw, W_raw)
            per_token.append(dict(k=m["k"], kind=m["kind"], norm=m["norm"],
                                    own_t=m["own_t"], raw_idx=ri,
                                    raw_map=raw_map,
                                    raw_vmax=float(raw_map.max()),
                                    raw_col_mean=float(raw_map.mean())))

        for N in SCALES:
            # Downsample each token's raw_map to N×N
            for pt in per_token:
                pt[f"map_{N}"] = downsample_to_NxN(pt["raw_map"], N)
                pt[f"vmax_{N}"] = float(pt[f"map_{N}"].max())

            shared_vmax = max(pt[f"vmax_{N}"] for pt in per_token)
            print(f"\nblk {BLK} h{head} ({kind_label}) N={N}: "
                  f"shared_vmax = {shared_vmax:.4e} "
                  f"({shared_vmax*100:.3f}%)")
            sink_panel = []; non_panel = []
            for pt in per_token:
                m = pt[f"vmax_{N}"]; frac = m / max(shared_vmax, 1e-30) * 100
                marker = "S" if pt["kind"] == "sink" else "N"
                print(f"    [{marker}] k={pt['k']:4d} norm={pt['norm']:6.2f} "
                      f"panel_max={m*100:.3f}% (={frac:.1f}% of shared)")
                (sink_panel if pt["kind"] == "sink" else non_panel).append(m)
            sm_ = float(np.mean(sink_panel)); nm_ = float(np.mean(non_panel))
            print(f"    mean sink panel-max = {sm_*100:.3f}%, "
                  f"mean non-sink panel-max = {nm_*100:.3f}%, "
                  f"ratio = {sm_/max(nm_,1e-30):.2f}×")

            # Render: 3 rows × 3 cols. Row 0 = frames (context), row 1 =
            # sink heatmaps (3 tokens), row 2 = non-sink heatmaps (3 tokens).
            # All heatmaps share the same vmax. Pure heatmaps, no overlay,
            # to honor "do not resize to original image size".
            cmap = plt.get_cmap("turbo")
            fig, axes = plt.subplots(3, 3, figsize=(13, 13),
                                       constrained_layout=True,
                                       gridspec_kw={"height_ratios":[1.0, 2.4, 2.4]})

            # Row 0: frame thumbnails for context (just so reader knows
            # which frames the tokens live in)
            for ti in range(min(3, T_raw)):
                ax = axes[0, ti]
                ax.imshow(frames[ti])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"frame t={ti}s (context)", fontsize=9)

            # Row 1: 3 sinks
            sinks_pt = [pt for pt in per_token if pt["kind"] == "sink"]
            non_pt = [pt for pt in per_token if pt["kind"] == "nonsink"]

            def draw_heatmap_panel(ax, pt, shared_vmax, N):
                m = pt[f"map_{N}"]
                ax.imshow(m, cmap=cmap, vmin=0, vmax=shared_vmax,
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                kind_str = "SINK" if pt["kind"] == "sink" else "non-sink"
                color = "darkred" if pt["kind"] == "sink" else "darkblue"
                ax.set_title(
                    f"{kind_str} k={pt['k']} norm={pt['norm']:.0f} "
                    f"t={pt['own_t']}s\n"
                    f"panel max = {pt[f'vmax_{N}']*100:.3f}% "
                    f"(={pt[f'vmax_{N}']/shared_vmax*100:.0f}% of shared)",
                    fontsize=10, color=color)
                # Annotate per-cell values for small N
                if N <= 10:
                    for r in range(N):
                        for c in range(N):
                            v = m[r, c]
                            ax.text(c, r, f"{v*100:.1f}", ha="center",
                                    va="center",
                                    color=("white" if v/max(shared_vmax,1e-12) > 0.5 else "black"),
                                    fontsize=7)
                # add a faint border per cell
                for s in range(N+1):
                    ax.axhline(s - 0.5, color="white", lw=0.3, alpha=0.6)
                    ax.axvline(s - 0.5, color="white", lw=0.3, alpha=0.6)

            for ci, pt in enumerate(sinks_pt):
                draw_heatmap_panel(axes[1, ci], pt, shared_vmax, N)
            for ci, pt in enumerate(non_pt):
                draw_heatmap_panel(axes[2, ci], pt, shared_vmax, N)

            # Add a colorbar on the right of rows 1 + 2
            sm_obj = plt.cm.ScalarMappable(cmap=cmap,
                                              norm=plt.Normalize(vmin=0, vmax=shared_vmax))
            sm_obj.set_array([])
            fig.colorbar(sm_obj, ax=axes[1:, :].ravel().tolist(),
                          location="right", shrink=0.6, label="attention from any query to this raw key (fraction)")

            head_role = "fg-type (top sink fg_share)" if kind_label == "fg" else "bg-type (lowest sink fg_share)"
            fig.suptitle(
                f"Clip {args.clip}  |  ViT block {BLK} (FULL within-frame)  "
                f"|  head h{head} ({head_role})\n"
                f"Pure heatmap at {N}×{N} (bilinear-downsampled from raw "
                f"{H_raw}×{W_raw}); SHARED color scale across 6 panels.  "
                f"Shared vmax = {shared_vmax*100:.3f}%.\n"
                f"Mean sink panel-max = {sm_*100:.3f}%, "
                f"mean non-sink panel-max = {nm_*100:.3f}%, "
                f"sink/non = {sm_/max(nm_,1e-30):.2f}×.",
                fontsize=11)
            out_path = in_dir / f"final_blk{BLK}_h{head}_{kind_label}_N{N:02d}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {out_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
