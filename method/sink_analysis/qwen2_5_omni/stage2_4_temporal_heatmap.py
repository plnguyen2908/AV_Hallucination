"""
stage2_4_temporal_heatmap.py

Temporal analog of Stage 2.2's spatial heatmap, for Stage 2.4 audio data.

Stage 2.2 pools sinks across clips into a 2D (row × col) frame-grid heatmap.
For 1D audio time the direct pooled analog is just a histogram (Stage 2.4
already has `temporal_sink_distribution.png`). The informative 2D analog is
a (clip × time-bin) strip view: each row is one clip, columns are normalized
time bins on [0, 1], cell color = sink density in that bin for that clip.
Sorts clips by total sink count so the most sink-heavy clips are on top.

Inputs:
  per_clip_temporal_arrays.npz  from Stage 2.4 (or 2.4-describe), containing
    per-layer CSR-flat sink indices and per-clip n_audio.

Outputs (one figure per requested layer):
  temporal_sink_heatmap_L<L>.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_NPZ_DESCRIBE = (_REPO / "results/qwen2_5_omni/sink_analysis/"
                        "stage2_4_temporal_sinks_describe/"
                        "per_clip_temporal_arrays.npz")
DEFAULT_NPZ_ORIG = (_REPO / "results/qwen2_5_omni/sink_analysis/"
                    "stage2_4_temporal_sinks/per_clip_temporal_arrays.npz")


def build_clip_time_matrix(sink_idx_flat, sink_idx_offsets, n_audio_arr,
                            n_bins=32):
    """Returns (matrix, sorted_idx, n_sink_per_clip).
    matrix[c, t] = # sinks in time bin t of clip c, divided by clip's n_audio
    so rows are normalized to the clip's audio length (= per-clip sink rate
    in that time bin)."""
    n_clips = len(n_audio_arr)
    mat = np.zeros((n_clips, n_bins), dtype=np.float64)
    n_sink_per_clip = np.zeros(n_clips, dtype=np.int64)
    edges = np.linspace(0, 1, n_bins + 1)
    for c in range(n_clips):
        a, b = int(sink_idx_offsets[c]), int(sink_idx_offsets[c + 1])
        n_a = int(n_audio_arr[c])
        n_sink_per_clip[c] = b - a
        if b > a and n_a > 0:
            norm_pos = sink_idx_flat[a:b].astype(np.float64) / n_a
            hist, _ = np.histogram(norm_pos, bins=edges)
            # Divide by audio-tokens-per-bin to get sink density (sinks per
            # audio token in that bin); makes rows comparable across clips
            # with different n_audio.
            tokens_per_bin = n_a / n_bins
            mat[c] = hist / max(tokens_per_bin, 1)
    # Sort clips by total sink count, descending.
    order = np.argsort(n_sink_per_clip)[::-1]
    return mat[order], order, n_sink_per_clip[order]


def build_clip_token_matrix(sink_idx_flat, sink_idx_offsets, n_audio_arr):
    """Per-token (1 bin = 1 audio token, absolute position) matrix.
    matrix[c, t] = 1.0 if token t is a sink in clip c, 0.0 otherwise; cells
    with t >= n_audio[c] are NaN so they render transparent. Width = max
    n_audio across all clips."""
    n_clips = len(n_audio_arr)
    max_n = int(n_audio_arr.max())
    mat = np.full((n_clips, max_n), np.nan, dtype=np.float32)
    n_sink_per_clip = np.zeros(n_clips, dtype=np.int64)
    for c in range(n_clips):
        a, b = int(sink_idx_offsets[c]), int(sink_idx_offsets[c + 1])
        n_a = int(n_audio_arr[c])
        n_sink_per_clip[c] = b - a
        # Initialize the in-span tokens to 0 (non-sink), then set sinks to 1.
        if n_a > 0:
            mat[c, :n_a] = 0.0
            if b > a:
                idx = sink_idx_flat[a:b].astype(np.int64)
                idx = idx[(idx >= 0) & (idx < n_a)]
                mat[c, idx] = 1.0
    order = np.argsort(n_sink_per_clip)[::-1]
    return mat[order], order, n_sink_per_clip[order], max_n


def plot_temporal_heatmap(mat, n_sink_sorted, layer, out_path, n_bins=32,
                            mode="normalized", max_n=None):
    """mode='normalized' → x-axis is fractional within-span position (0..1).
    mode='token'        → x-axis is absolute audio-token position (0..max_n).
                          mat is expected to have NaN beyond each clip's span;
                          NaN renders transparent (cmap bad-color)."""
    n_clips = mat.shape[0]
    fig, (ax_h, ax_m) = plt.subplots(
        2, 1, figsize=(13, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.18}, sharex=True)

    if mode == "token":
        x_max = int(max_n) if max_n is not None else mat.shape[1]
        x_lo, x_hi = 0, x_max
        x_lab = f"audio token position (0 = BOS-of-audio, max = {x_max})"
        cb_lab = "sink (1) / non-sink (0); NaN beyond clip span"
        bin_centers = np.arange(mat.shape[1]) + 0.5
        bar_width = 1.0
    else:
        x_lo, x_hi = 0.0, 1.0
        x_lab = "normalized within-span position"
        cb_lab = "sink density (sinks / token in bin)"
        bin_centers = (np.arange(n_bins) + 0.5) / n_bins
        bar_width = 1.0 / n_bins

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#222222")          # NaN cells = dark grey
    im = ax_h.imshow(mat, aspect="auto", origin="upper", cmap=cmap,
                      interpolation="nearest",
                      extent=[x_lo, x_hi, n_clips, 0])
    cbar = plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.01)
    cbar.set_label(cb_lab, fontsize=9)
    ax_h.set_ylabel("clip (sorted by total sink count; top = most)", fontsize=11)
    title_extra = (f"n_bins = {mat.shape[1]} (1/token)" if mode == "token"
                    else f"n_bins = {n_bins}")
    ax_h.set_title(f"Stage 2.4 — per-clip temporal sink heatmap @ L{layer}  "
                   f"(n_clips = {n_clips}, {title_extra})", fontsize=12)
    ax_h.set_yticks([])

    # Pooled marginal (column sums, normalized to a fraction).
    if mode == "token":
        col_sum = np.nansum(mat, axis=0)
        n_valid = (~np.isnan(mat)).sum(axis=0)
        p = col_sum / np.clip(n_valid, 1, None)
    else:
        col_sum = mat.sum(axis=0)
        p = col_sum / max(col_sum.sum(), 1e-12)
    ax_m.bar(bin_centers, p, width=bar_width, color="#1f77b4",
             edgecolor="black", linewidth=0.2, alpha=0.85)
    ax_m.set_xlabel(x_lab, fontsize=11)
    ax_m.set_ylabel("sink fraction\nover valid clips" if mode == "token"
                    else "pooled\nfraction", fontsize=10)
    ax_m.set_xlim(x_lo, x_hi)
    ax_m.grid(True, ls=":", alpha=0.4, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(args):
    npz_path = Path(args.npz)
    if not npz_path.is_file():
        raise SystemExit(f"not found: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    # Auto-discover layers present (keys like 'L21_sink_idx_flat').
    layers = sorted({int(k[1:].split("_", 1)[0])
                      for k in z.files
                      if k.startswith("L") and "_sink_idx_flat" in k})
    if args.layers:
        layers = [int(L) for L in args.layers if int(L) in layers]
    out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {npz_path}; layers found: {layers}; "
          f"output dir: {out_dir}")
    for L in layers:
        flat = z[f"L{L}_sink_idx_flat"]
        offs = z[f"L{L}_sink_idx_offsets"]
        n_audio = z[f"L{L}_n_audio"]
        n_clips = len(n_audio)
        if args.mode == "token":
            mat, order, n_sink_sorted, max_n = build_clip_token_matrix(
                flat, offs, n_audio)
            print(f"  L{L}: n_clips={n_clips}, "
                  f"mean n_sink/clip={n_sink_sorted.mean():.1f}, "
                  f"max_n_audio={max_n}  (mode=token, 1 token/bin)")
            plot_temporal_heatmap(
                mat, n_sink_sorted, L,
                out_dir / f"temporal_sink_heatmap_L{L}_per_token.png",
                mode="token", max_n=max_n)
        else:
            mat, order, n_sink_sorted = build_clip_time_matrix(
                flat, offs, n_audio, n_bins=args.n_bins)
            print(f"  L{L}: n_clips={n_clips}, "
                  f"mean n_sink/clip={n_sink_sorted.mean():.1f}  "
                  f"(mode=normalized, n_bins={args.n_bins})")
            plot_temporal_heatmap(
                mat, n_sink_sorted, L,
                out_dir / f"temporal_sink_heatmap_L{L}.png",
                n_bins=args.n_bins, mode="normalized")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ_DESCRIBE),
                   help="Stage 2.4 (or 2.4-describe) per_clip_temporal_arrays.npz.")
    p.add_argument("--output_dir", default="",
                   help="Where to save figures (default: alongside the npz).")
    p.add_argument("--n_bins", type=int, default=32,
                   help="Normalized-mode: number of time bins on [0, 1].")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Which layers to plot (default = all in the npz).")
    p.add_argument("--mode", choices=["normalized", "token"], default="token",
                   help="'token' (default): 1 bin = 1 audio token (absolute "
                        "position; NaN beyond clip span). 'normalized': "
                        "fixed-bin density over [0, 1].")
    args = p.parse_args()
    main(args)
