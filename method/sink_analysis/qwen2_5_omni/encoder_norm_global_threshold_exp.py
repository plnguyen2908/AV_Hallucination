"""
encoder_norm_global_threshold_exp.py

Stage 1.1 (v2) — Redo the encoder-side bimodality analysis using a single
GLOBAL percentile threshold per modality, instead of the per-clip robust-z
threshold from v1.

Motivation: per-clip (norm − median) / IQR self-suppresses. When a clip has
multiple high-norm tokens, those tokens inflate the clip's own IQR and pull
the per-clip threshold up with them, hiding the very outliers we're trying
to detect. Switching to a global percentile threshold avoids the loop.

Prerequisites:
    Per-token L2 norm arrays cached on disk at --norms_npz by a previous run
    of `encoder_norm_bimodality_exp.py`. That script writes
        encoder_norms.npz
    inside its --output_dir (default
    results/qwen2_5_omni/sink_analysis/stage1_1_encoder_norms/), containing:
        audio_flat, audio_offsets, audio_names
        video_flat, video_offsets, video_names
    Forward passes are NOT recomputed here.

Procedure:
    1. Per modality, pool all per-clip norms and compute four thresholds:
            tau_95     = 95th    percentile
            tau_99     = 99th    percentile
            tau_995    = 99.5th  percentile
            tau_abs100 = absolute cutoff of 100  (Sink-or-Not-to-Sink criterion)
    2. Per (modality, threshold): per-clip count of tokens whose norm > τ.
    3. Plot a 2×4 grid of per-clip count histograms.
    4. Verdict per cell:
            median > 0 AND %zero_clips < 30  → "Propagated-sink signature ..."
            median > 0 AND %zero_clips ≥ 30  → "Weak / content-conditional ..."
            median == 0                      → "No propagated-sink signature ..."
    5. Print a summary table.

Outputs (--output_dir):
    encoder_norm_global_threshold.png
    encoder_norm_global_threshold_summary.txt

Stop after the summary table — do not pick a downstream threshold without
human confirmation.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

THRESHOLD_KEYS = ("tau_95", "tau_99", "tau_995", "tau_abs100")
MODALITY_COLORS = {"audio": "#1f77b4", "video": "#d62728"}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def load_cached_norms(npz_path: Path):
    """Load the encoder_norms.npz cache.

    Returns dict {modality: {"per_clip": list[np.ndarray], "names": list[str]}}.
    """
    if not npz_path.exists():
        raise SystemExit(
            f"Cached norms not found at {npz_path}.\n"
            "Run\n"
            "    python method/sink_analysis/qwen2_5_omni/"
            "encoder_norm_bimodality_exp.py\n"
            "first — it now caches per-clip norms by default.\n"
        )
    d = np.load(npz_path, allow_pickle=True)
    out: dict = {}
    for modality in ("audio", "video"):
        flat_key = f"{modality}_flat"
        off_key = f"{modality}_offsets"
        names_key = f"{modality}_names"
        if flat_key not in d.files or off_key not in d.files:
            continue
        flat = d[flat_key]
        offs = d[off_key]
        per_clip = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        names = (
            list(d[names_key]) if names_key in d.files else
            [f"clip_{i}" for i in range(len(per_clip))]
        )
        out[modality] = {"per_clip": per_clip, "names": names}
    if not out:
        raise SystemExit(f"Cache at {npz_path} has no audio or video arrays.")
    return out


# --------------------------------------------------------------------------
# Thresholds + counts
# --------------------------------------------------------------------------

def compute_thresholds(pooled: np.ndarray) -> dict[str, float]:
    return {
        "tau_95":     float(np.percentile(pooled, 95)),
        "tau_99":     float(np.percentile(pooled, 99)),
        "tau_995":    float(np.percentile(pooled, 99.5)),
        "tau_abs100": 100.0,
    }


def per_clip_counts(per_clip: list, threshold: float) -> np.ndarray:
    return np.array([int((n > threshold).sum()) for n in per_clip])


def verdict(median_count: float, frac_zero: float) -> str:
    if median_count > 0 and frac_zero < 0.30:
        return "Propagated-sink signature present at this threshold."
    if median_count > 0 and frac_zero >= 0.30:
        return "Weak / content-conditional propagated sinks at this threshold."
    return "No propagated-sink signature at this threshold."


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_grid(stats: dict, png_path: Path) -> None:
    modalities = list(stats.keys())
    n_rows = len(modalities)
    n_cols = len(THRESHOLD_KEYS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows)
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, modality in enumerate(modalities):
        color = MODALITY_COLORS.get(modality, "#888888")
        per_clip = stats[modality]["per_clip"]
        thresholds = stats[modality]["thresholds"]

        for c, key in enumerate(THRESHOLD_KEYS):
            tau = thresholds[key]
            counts = stats[modality]["counts"][key]
            mean_c = float(counts.mean())
            median_c = float(np.median(counts))
            frac_zero = float((counts == 0).mean())

            ax = axes[r, c]
            max_c = max(int(counts.max()) if counts.size else 0, 5)
            bins = np.arange(0, max_c + 2)
            ax.hist(
                counts, bins=bins, color=color, alpha=0.80,
                edgecolor="white", linewidth=0.6,
            )
            ax.set_xlabel("# tokens > τ per clip")
            ax.set_ylabel("# clips")
            ax.set_title(
                f"{modality} — {key} = {tau:.3g}",
                fontsize=11,
            )
            ax.grid(True, linestyle=":", alpha=0.4)

            annot = (
                f"median = {median_c:.1f}\n"
                f"mean   = {mean_c:.1f}\n"
                f"%zero  = {frac_zero * 100:.1f}%"
            )
            ax.text(
                0.97, 0.95, annot,
                ha="right", va="top", transform=ax.transAxes,
                fontsize=9, family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white", alpha=0.88, edgecolor="lightgray",
                ),
            )

    fig.suptitle(
        "Per-clip outlier counts at global percentile thresholds  "
        "(rows: modality, cols: τ)",
        fontsize=13, y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    npz_path = Path(args.norms_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached norms from {npz_path}")
    cached = load_cached_norms(npz_path)

    # ----- sanity: shape + dtype -----
    print("\n" + "=" * 70)
    print("Sanity checks (loaded norms)")
    print("=" * 70)
    for modality, info in cached.items():
        per_clip = info["per_clip"]
        pooled = np.concatenate(per_clip) if per_clip else np.array([])
        print(
            f"  {modality}: n_clips={len(per_clip)}, "
            f"total_tokens={pooled.size}, "
            f"dtype={pooled.dtype}, "
            f"shape(pooled)={pooled.shape}"
        )

    # ----- thresholds -----
    stats: dict = {}
    for modality, info in cached.items():
        per_clip = info["per_clip"]
        pooled = np.concatenate(per_clip)
        thresholds = compute_thresholds(pooled)
        stats[modality] = {
            "per_clip": per_clip,
            "names": info["names"],
            "pooled": pooled,
            "thresholds": thresholds,
        }

    print("\nThresholds (per modality, from pooled distribution):")
    for modality, s in stats.items():
        t = s["thresholds"]
        print(
            f"  {modality}: "
            f"tau_95={t['tau_95']:.3g}  "
            f"tau_99={t['tau_99']:.3g}  "
            f"tau_995={t['tau_995']:.3g}  "
            f"tau_abs100={t['tau_abs100']:.3g}"
        )
        assert t["tau_95"] > 0, f"{modality}: tau_95 must be positive"
        assert t["tau_95"] <= t["tau_99"] <= t["tau_995"], (
            f"{modality}: percentile thresholds must be ordered "
            f"(got {t['tau_95']:.3g}, {t['tau_99']:.3g}, {t['tau_995']:.3g})"
        )

    # ----- sanity: example clip -----
    if "audio" in stats and stats["audio"]["per_clip"]:
        ex = stats["audio"]["per_clip"][0]
        t99 = stats["audio"]["thresholds"]["tau_99"]
        print(
            f"\nExample audio clip 0 ({stats['audio']['names'][0]}): "
            f"{ex.size} tokens, "
            f"count > tau_99 ({t99:.3g}) = {int((ex > t99).sum())}"
        )
    if "video" in stats and stats["video"]["per_clip"]:
        ex = stats["video"]["per_clip"][0]
        t99 = stats["video"]["thresholds"]["tau_99"]
        print(
            f"Example video clip 0 ({stats['video']['names'][0]}): "
            f"{ex.size} tokens, "
            f"count > tau_99 ({t99:.3g}) = {int((ex > t99).sum())}"
        )

    # ----- per-clip counts for every (modality, threshold) -----
    for modality, s in stats.items():
        s["counts"] = {
            key: per_clip_counts(s["per_clip"], s["thresholds"][key])
            for key in THRESHOLD_KEYS
        }

    # ----- summary table -----
    summary_rows: list[dict] = []
    for modality, s in stats.items():
        for key in THRESHOLD_KEYS:
            counts = s["counts"][key]
            median_c = float(np.median(counts))
            mean_c = float(counts.mean())
            frac_zero = float((counts == 0).mean())
            summary_rows.append({
                "modality": modality,
                "threshold_key": key,
                "threshold_value": s["thresholds"][key],
                "median": median_c,
                "mean": mean_c,
                "pct_zero": frac_zero * 100,
                "verdict": verdict(median_c, frac_zero),
            })

    header = (
        f"{'modality':<10}{'threshold':<12}{'value':>10}"
        f"{'median':>9}{'mean':>9}{'%zero':>9}   verdict"
    )
    sep = "=" * 100
    print()
    print(sep)
    print("Summary  (global percentile thresholds, no per-clip normalization)")
    print(sep)
    print(header)
    print("-" * 100)
    for row in summary_rows:
        line = (
            f"{row['modality']:<10}{row['threshold_key']:<12}"
            f"{row['threshold_value']:>10.3g}"
            f"{row['median']:>9.1f}{row['mean']:>9.1f}"
            f"{row['pct_zero']:>8.1f}%   {row['verdict']}"
        )
        print(line)

    # ----- write text summary -----
    txt_path = out_dir / "encoder_norm_global_threshold_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Stage 1.1 (v2) — global percentile thresholds\n")
        f.write(f"loaded from: {npz_path}\n\n")
        for modality, s in stats.items():
            t = s["thresholds"]
            f.write(
                f"{modality}: n_clips={len(s['per_clip'])}, "
                f"total_tokens={s['pooled'].size}, "
                f"tau_95={t['tau_95']:.3g}, tau_99={t['tau_99']:.3g}, "
                f"tau_995={t['tau_995']:.3g}, tau_abs100={t['tau_abs100']:.3g}\n"
            )
        f.write("\n" + header + "\n" + ("-" * 100) + "\n")
        for row in summary_rows:
            f.write(
                f"{row['modality']:<10}{row['threshold_key']:<12}"
                f"{row['threshold_value']:>10.3g}"
                f"{row['median']:>9.1f}{row['mean']:>9.1f}"
                f"{row['pct_zero']:>8.1f}%   {row['verdict']}\n"
            )
    print(f"\nwrote {txt_path}")

    # ----- figure -----
    png_path = out_dir / "encoder_norm_global_threshold.png"
    plot_grid(stats, png_path)
    print(f"wrote {png_path}")
    print(
        "\nDone — Stage 1.1 (v2) complete. Pick a τ before any downstream "
        "analysis runs."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--norms_npz",
        default=str(
            _REPO / "results/qwen2_5_omni/sink_analysis/"
            "stage1_1_encoder_norms/encoder_norms.npz"
        ),
        help="Cached per-token norms from the previous Stage 1.1 run "
             "(written by encoder_norm_bimodality_exp.py).",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/stage1_1_encoder_norms"),
    )
    args = p.parse_args()
    main(args)
