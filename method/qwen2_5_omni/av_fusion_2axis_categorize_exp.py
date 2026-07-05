"""
av_fusion_2axis_categorize_exp.py

Stage 4.1 variant: 2-dataset head categorization (drop VGGSounder).

Combine per-head hallucination influence scores from TWO Qwen2.5-Omni
attribution runs:

    A = AudioSet describe       (audio-only modality)
    V = ActivityNet describe    (video-only modality)

Score per (layer, head, modality) is the contrastive
`mean_hal - mean_non_hal` recomputed from the per-sample .pth influence
files (mirrors `contrastive_score` in identify_halluc_head.py).

A single threshold τ = `--main_percentile`th percentile of |scores|
pooled across A ∪ V. Membership in H_X = `score_X > τ` (strictly
positive: heads that DRIVE hallucination).

Categories (4 disjoint cells from the (in_H_A, in_H_V) signature):

    (1, 0) Audio head
    (0, 1) Visual head
    (1, 1) Audiovisual head    (intersection of H_A and H_V)
    (0, 0) Inert

The "Audiovisual head" cell replaces the 3-dataset AV-dataset cell — a
head is called AV here when it drives hallucination in both audio-only
and video-only probes, regardless of any explicit AV probe.

Outputs (under --output_dir, default
`results/qwen2_5_omni/categorize_exp_2axis/`):
    heads.csv               one row per head, category at τ_main
    counts.csv              category counts at τ_90 / τ_95 / τ_99
    categories_scatter.png  single A-vs-V scatter, 300 dpi
    category_counts.png     horizontal bar chart of the 4 counts

Usage:
    python method/qwen2_5_omni/av_fusion_2axis_categorize_exp.py
    python method/qwen2_5_omni/av_fusion_2axis_categorize_exp.py --main_percentile 95
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_REPO = Path(__file__).resolve().parent.parent.parent

_DEFAULT_AUDIO_DIR = _REPO / "results/qwen2_5_omni/AudioSet_describe/attribution"
_DEFAULT_VISUAL_DIR = _REPO / "results/qwen2_5_omni/ActivityNet_describe/attribution"
_DEFAULT_OUT = _REPO / "results/qwen2_5_omni/categorize_exp_2axis"


# 4-cell taxonomy (no VGGSounder dimension)
CATEGORY_ORDER: list[str] = [
    "Audiovisual head",
    "Audio head",
    "Visual head",
    "Inert",
]

CATEGORY_RULES: dict[tuple[int, int], str] = {
    (1, 1): "Audiovisual head",
    (1, 0): "Audio head",
    (0, 1): "Visual head",
    (0, 0): "Inert",
}

CATEGORY_COLORS: dict[str, str] = {
    "Audiovisual head": "#9467bd",   # purple — A ∩ V
    "Audio head":       "#d62728",   # red
    "Visual head":      "#1f77b4",   # blue
    "Inert":            "#cccccc",   # light gray
}


# --------------------------------------------------------------------------
# Input loading (same aggregator as the 3-dataset script)
# --------------------------------------------------------------------------

def compute_difference(attribution_dir: Path) -> tuple[np.ndarray, int, int]:
    """Walk `<attribution_dir>/pth/` and return
    (mean_hal − mean_non_hal, layer_num, head_num)."""
    pth_dir = attribution_dir / "pth"
    if not pth_dir.is_dir():
        raise SystemExit(f"pth dir not found: {pth_dir}")

    hal_samples: list[torch.Tensor] = []
    non_hal_samples: list[torch.Tensor] = []
    layer_num = head_num = None

    for fname in sorted(os.listdir(pth_dir)):
        if not fname.endswith(".pth"):
            continue
        is_hal = fname.startswith("hal")
        data = torch.load(pth_dir / fname, weights_only=False)
        if not data:
            continue
        for _, v in data.items():
            if layer_num is None:
                layer_num = len(v)
                head_num = len(v[0])
            influence = torch.zeros(layer_num, head_num)
            for li in range(layer_num):
                for hi in range(head_num):
                    influence[li][hi] = v[li][hi]["influence"]
            influence = torch.nan_to_num(influence, nan=0.0)
            (hal_samples if is_hal else non_hal_samples).append(influence)

    if not hal_samples or not non_hal_samples:
        raise SystemExit(
            f"Need both hal and non_hal samples in {pth_dir}; "
            f"got {len(hal_samples)} / {len(non_hal_samples)}"
        )

    mean_hal = torch.stack(hal_samples).mean(0).float()
    mean_non_hal = torch.stack(non_hal_samples).mean(0).float()
    return (mean_hal - mean_non_hal).cpu().numpy(), layer_num, head_num


def assert_input_sane(name: str, arr: np.ndarray) -> None:
    if arr is None or not isinstance(arr, np.ndarray):
        raise SystemExit(f"[input check][{name}] bad input")
    if arr.size == 0 or np.isnan(arr).all() or (arr == 0).all():
        raise SystemExit(f"[input check][{name}] empty / all-nan / all-zero")


def summarize(name: str, arr: np.ndarray) -> None:
    print(
        f"  [{name}] shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():+.4g} max={arr.max():+.4g} "
        f"mean={arr.mean():+.4g} median={np.median(arr):+.4g}"
    )


# --------------------------------------------------------------------------
# Categorization
# --------------------------------------------------------------------------

def assign_categories(
    score_A: np.ndarray, score_V: np.ndarray, tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_A = score_A > tau
    in_V = score_V > tau
    cat = np.empty(score_A.shape, dtype=object)
    L, H = score_A.shape
    for li in range(L):
        for hi in range(H):
            cat[li, hi] = CATEGORY_RULES[(int(in_A[li, hi]), int(in_V[li, hi]))]
    return in_A, in_V, cat


def category_counts(cat: np.ndarray) -> dict[str, int]:
    return {c: int((cat == c).sum()) for c in CATEGORY_ORDER}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_scatter(
    score_A: np.ndarray, score_V: np.ndarray,
    cat_main: np.ndarray, counts_main: dict[str, int],
    tau_main: float, main_percentile: int,
    output_path: Path,
) -> None:
    flat_A = score_A.ravel()
    flat_V = score_V.ravel()
    flat_cat = cat_main.ravel()

    smin = float(min(flat_A.min(), flat_V.min()))
    smax = float(max(flat_A.max(), flat_V.max()))
    margin = 0.05 * (smax - smin) if smax > smin else 1e-6
    lo, hi = smin - margin, smax + margin

    fig, ax = plt.subplots(figsize=(9, 9))

    # Inert as background
    inert_mask = flat_cat == "Inert"
    ax.scatter(
        flat_A[inert_mask], flat_V[inert_mask],
        s=12, c=CATEGORY_COLORS["Inert"], alpha=0.35,
        edgecolors="none", label=None,
    )
    # 3 hal categories on top
    for cat in ["Audio head", "Visual head", "Audiovisual head"]:
        mask = flat_cat == cat
        if not mask.any():
            continue
        ax.scatter(
            flat_A[mask], flat_V[mask],
            s=70, c=CATEGORY_COLORS[cat], alpha=0.9,
            edgecolors="black", linewidths=0.5,
        )

    # Threshold lines at tau (vertical for A, horizontal for V)
    ax.axhline(tau_main, color="gray", linestyle="--", linewidth=1.2,
                alpha=0.7, label=f"τ_V = τ_A = {tau_main:.4g}")
    ax.axvline(tau_main, color="gray", linestyle="--", linewidth=1.2,
                alpha=0.7)
    # Zero lines (lighter)
    ax.axhline(0, color="lightgray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="lightgray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Audio score (AudioSet)  mean_hal − mean_non_hal", fontsize=12)
    ax.set_ylabel("Visual score (ActivityNet)  mean_hal − mean_non_hal", fontsize=12)
    ax.set_title(
        f"Per-head 2-axis categorization (τ = {main_percentile}th percentile = "
        f"{tau_main:.4g})",
        fontsize=13, pad=10,
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Legend
    legend_handles = []
    for cat in CATEGORY_ORDER:
        is_inert = cat == "Inert"
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=CATEGORY_COLORS[cat],
                markeredgecolor="none" if is_inert else "black",
                markeredgewidth=0.5,
                markersize=8 if is_inert else 11,
                label=f"{cat} ({counts_main[cat]})",
            )
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
                    label=f"τ = {tau_main:.4g}")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
                framealpha=0.92)

    # Quadrant annotations (tiny, in the corners outside the data hull)
    ax.text(0.97, 0.97, "Audiovisual\n(in H_A ∩ H_V)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=CATEGORY_COLORS["Audiovisual head"],
            alpha=0.6)
    ax.text(0.97, 0.03, "Audio only",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=CATEGORY_COLORS["Audio head"], alpha=0.6)
    ax.text(0.03, 0.97, "Visual only",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=CATEGORY_COLORS["Visual head"], alpha=0.6)
    ax.text(0.03, 0.03, "Inert",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="#888", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_counts(counts: dict[str, int], tau_main: float,
                 main_percentile: int, output_path: Path) -> None:
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    cats = [c for c, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(range(len(cats)), vals, color=colors,
                    edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(
        f"Head categories at τ = {main_percentile}th percentile = {tau_main:.4g}",
        fontsize=13, pad=8,
    )
    pad = 0.01 * max(vals)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=11,
        )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(vals) * 1.10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Loading attribution scores (2-axis: A + V only, no VGGSounder)")
    print("=" * 70)
    score_A, La, Ha = compute_difference(Path(args.audio_dir))
    score_V, Lv, Hv = compute_difference(Path(args.visual_dir))

    print("Per-modality stats:")
    for name, arr in [("A", score_A), ("V", score_V)]:
        assert_input_sane(name, arr)
        summarize(name, arr)

    if (La, Ha) != (Lv, Hv):
        raise SystemExit(
            f"Layer/head shape mismatch: A=({La},{Ha}) V=({Lv},{Hv}). "
            "Both runs must use the same Qwen2.5-Omni model."
        )
    L, H = La, Ha
    total = L * H

    print("\n" + "=" * 70)
    print(f"Thresholds (|scores| pooled across A ∪ V, {2 * total} values)")
    print("=" * 70)
    combined_abs = np.concatenate([
        np.abs(score_A).ravel(), np.abs(score_V).ravel(),
    ])
    percentiles = sorted({90, 95, 99, int(args.main_percentile)})
    tau: dict[int, float] = {
        p: float(np.percentile(combined_abs, p)) for p in percentiles
    }
    for p in sorted(tau):
        marker = "  ←main" if p == args.main_percentile else ""
        print(f"  τ_{p:>2d} = {tau[p]:.6g}{marker}")

    main_p = int(args.main_percentile)
    tau_main = tau[main_p]

    print("\n" + "=" * 70)
    print(f"Membership at τ_{main_p} (= {tau_main:.6g})")
    print("=" * 70)
    in_A_main, in_V_main, cat_main = assign_categories(
        score_A, score_V, tau_main
    )
    print(f"  |H_A| = {int(in_A_main.sum())}")
    print(f"  |H_V| = {int(in_V_main.sum())}")

    counts_main = category_counts(cat_main)
    counts_sum = sum(counts_main.values())
    print(f"\n  Category counts at τ_{main_p}:")
    for c in CATEGORY_ORDER:
        print(f"    {c:<20s} {counts_main[c]:>6d}")
    print(f"    {'TOTAL':<20s} {counts_sum:>6d}  (expected {total})")
    assert counts_sum == total, (
        f"Counts ({counts_sum}) != total heads ({total})"
    )

    # Counts at robustness thresholds 90/95/99
    counts_per_tau: dict[int, dict[str, int]] = {}
    for p in (90, 95, 99):
        _, _, cat_p = assign_categories(score_A, score_V, tau[p])
        counts_per_tau[p] = category_counts(cat_p)

    rows = []
    for c in CATEGORY_ORDER:
        row = {"category": c}
        for p in (90, 95, 99):
            n = counts_per_tau[p][c]
            row[f"count_tau{p}"] = n
            row[f"fraction_tau{p}"] = n / total
        rows.append(row)
    counts_df = pd.DataFrame(rows)
    counts_csv = out_dir / "counts.csv"
    counts_df.to_csv(counts_csv, index=False)
    print("\n" + "=" * 70)
    print("Counts at all three robustness thresholds (90 / 95 / 99)")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(counts_df.to_string(index=False))

    # Per-head csv
    layers, heads = np.meshgrid(np.arange(L), np.arange(H), indexing="ij")
    heads_df = pd.DataFrame({
        "layer":    layers.ravel(),
        "head":     heads.ravel(),
        "score_A":  score_A.ravel(),
        "score_V":  score_V.ravel(),
        "in_H_A":   in_A_main.ravel(),
        "in_H_V":   in_V_main.ravel(),
        "category": cat_main.ravel(),
    })
    heads_csv = out_dir / "heads.csv"
    heads_df.to_csv(heads_csv, index=False)

    print("\n" + "=" * 70)
    print(f"Outputs (τ_main = {main_p}th percentile)")
    print("=" * 70)
    print(f"  wrote {heads_csv}")
    print(f"  wrote {counts_csv}")
    plot_scatter(
        score_A, score_V, cat_main, counts_main, tau_main, main_p,
        out_dir / "categories_scatter.png",
    )
    plot_counts(counts_main, tau_main, main_p,
                 out_dir / "category_counts.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default=str(_DEFAULT_AUDIO_DIR))
    parser.add_argument("--visual_dir", default=str(_DEFAULT_VISUAL_DIR))
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUT))
    parser.add_argument("--main_percentile", type=int, default=90)
    args = parser.parse_args()
    main(args)
