"""
av_fusion_categorize_exp.py

Combine per-head hallucination influence scores from three Qwen2.5-Omni
attribution runs:

    A  = AudioSet describe        (audio-only modality)
    V  = ActivityNet describe     (video-only modality)
    AV = VGGSounder describe      (audio-visual modality)

Score per (layer, head, modality) is the contrastive
`mean_hal - mean_non_hal` matrix recomputed from the per-sample .pth
influence files (mirrors `contrastive_score` in identify_halluc_head.py).

A single threshold τ is taken across ALL THREE maps combined — specifically
the 95th percentile of |scores| pooled across A ∪ V ∪ AV. Membership in
H_X is `score_X > τ` (strictly positive: heads that DRIVE hallucination,
not protect against it). The same categorization is recomputed at τ_90
and τ_99 for robustness.

Every head lands in exactly one of 8 disjoint cells based on its
(in_H_A, in_H_V, in_H_AV) signature:

    (1,0,0) Audio-only        (0,1,0) Visual-only
    (0,0,1) Cross-modal-only  (1,0,1) Audio + AV
    (0,1,1) Visual + AV       (1,1,0) Compensated
    (1,1,1) Generic           (0,0,0) Inert

Outputs (under --output_dir, default
`results/qwen2_5_omni/categorize_exp/`):
    heads.csv               one row per head, category at τ_95
    counts.csv              category counts at τ_90 / τ_95 / τ_99
    categories_scatter.png  3 paired scatters, 300 dpi
    category_counts.png     horizontal bar chart of the 8 counts

Usage:
    python method/qwen2_5_omni/av_fusion_categorize_exp.py
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------
# Defaults — point at the three attribution dirs you already have.
# --------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent.parent

_DEFAULT_AUDIO_DIR = _REPO / "results/qwen2_5_omni/AudioSet_describe/attribution"
_DEFAULT_VISUAL_DIR = _REPO / "results/qwen2_5_omni/ActivityNet_describe/attribution"
_DEFAULT_AV_DIR = _REPO / "results/qwen2_5_omni/VGGSounder_describe/attribution"
_DEFAULT_OUT = _REPO / "results/qwen2_5_omni/categorize_exp"


# --------------------------------------------------------------------------
# Category taxonomy and palette
# --------------------------------------------------------------------------

# Order is the canonical order for tables, legends, and the bar chart's
# tie-breaker when two categories have identical counts.
CATEGORY_ORDER: list[str] = [
    "Generic",
    "Audio + AV",
    "Visual + AV",
    "Cross-modal-only",
    "Compensated",
    "Audio-only",
    "Visual-only",
    "Inert",
]

# (in_H_A, in_H_V, in_H_AV) -> category name. The 8 keys cover the cube.
CATEGORY_RULES: dict[tuple[int, int, int], str] = {
    (1, 0, 0): "Audio-only",
    (0, 1, 0): "Visual-only",
    (0, 0, 1): "Cross-modal-only",
    (1, 0, 1): "Audio + AV",
    (0, 1, 1): "Visual + AV",
    (1, 1, 0): "Compensated",
    (1, 1, 1): "Generic",
    (0, 0, 0): "Inert",
}

# Distinct, saturated colours; Inert is the only desaturated one so it
# recedes when plotted as background.
CATEGORY_COLORS: dict[str, str] = {
    "Generic":          "#000000",  # black — stands out, most interesting
    "Audio + AV":       "#d62728",  # red
    "Visual + AV":      "#1f77b4",  # blue
    "Cross-modal-only": "#2ca02c",  # green
    "Compensated":      "#9467bd",  # purple
    "Audio-only":       "#ff7f0e",  # orange
    "Visual-only":      "#17becf",  # cyan
    "Inert":            "#cccccc",  # light gray
}


# --------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------

def compute_difference(attribution_dir: Path) -> tuple[np.ndarray, int, int]:
    """Walk `<attribution_dir>/pth/` and return
    (mean_hal − mean_non_hal, layer_num, head_num).

    Same aggregation logic as identify_halluc_head.contrastive_score; kept
    inline so this experiment script avoids the heavy generation imports.
    """
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


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------

def assert_input_sane(name: str, arr: np.ndarray) -> None:
    if arr is None:
        raise SystemExit(f"[input check][{name}] missing input array")
    if not isinstance(arr, np.ndarray):
        raise SystemExit(
            f"[input check][{name}] expected np.ndarray, got {type(arr)}"
        )
    if arr.size == 0:
        raise SystemExit(f"[input check][{name}] empty array")
    if np.isnan(arr).all():
        raise SystemExit(f"[input check][{name}] all-NaN array")
    if (arr == 0).all():
        raise SystemExit(f"[input check][{name}] all-zero array")


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
    score_A: np.ndarray,
    score_V: np.ndarray,
    score_AV: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (in_A, in_V, in_AV, category) arrays — all (L, H)."""
    in_A = score_A > tau
    in_V = score_V > tau
    in_AV = score_AV > tau
    cat = np.empty(score_A.shape, dtype=object)
    L, H = score_A.shape
    for li in range(L):
        for hi in range(H):
            cat[li, hi] = CATEGORY_RULES[
                (int(in_A[li, hi]), int(in_V[li, hi]), int(in_AV[li, hi]))
            ]
    return in_A, in_V, in_AV, cat


def category_counts(cat: np.ndarray) -> dict[str, int]:
    return {c: int((cat == c).sum()) for c in CATEGORY_ORDER}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_scatter(
    score_A: np.ndarray,
    score_V: np.ndarray,
    score_AV: np.ndarray,
    cat_main: np.ndarray,
    counts_main: dict[str, int],
    main_percentile: int,
    output_path: Path,
) -> None:
    flat_A = score_A.ravel()
    flat_V = score_V.ravel()
    flat_AV = score_AV.ravel()
    flat_cat = cat_main.ravel()

    # Shared axes limits.
    all_scores = np.concatenate([flat_A, flat_V, flat_AV])
    smin, smax = float(all_scores.min()), float(all_scores.max())
    margin = 0.05 * (smax - smin) if smax > smin else 1e-6
    lo, hi = smin - margin, smax + margin

    pairs = [
        ("Audio vs Visual",         flat_A,  flat_V,
         "Audio score (AudioSet)",        "Visual score (ActivityNet)"),
        ("Audio vs Audio-Visual",   flat_A,  flat_AV,
         "Audio score (AudioSet)",        "Audio-Visual score (VGGSounder)"),
        ("Visual vs Audio-Visual",  flat_V,  flat_AV,
         "Visual score (ActivityNet)",    "Audio-Visual score (VGGSounder)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    for ax, (title, x_arr, y_arr, xlabel, ylabel) in zip(axes, pairs):
        # Inert first as background.
        inert_mask = flat_cat == "Inert"
        ax.scatter(
            x_arr[inert_mask], y_arr[inert_mask],
            s=10, c=CATEGORY_COLORS["Inert"], alpha=0.35,
            edgecolors="none",
        )
        # Then the 7 hal categories on top, in CATEGORY_ORDER (Generic
        # last so it sits on top of everything).
        for cat in reversed([c for c in CATEGORY_ORDER if c != "Inert"]):
            mask = flat_cat == cat
            if not mask.any():
                continue
            ax.scatter(
                x_arr[mask], y_arr[mask],
                s=55, c=CATEGORY_COLORS[cat], alpha=0.9,
                edgecolors="black", linewidths=0.4,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=8)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

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
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Per-head hallucination category assignment (τ = {main_percentile}th percentile)",
        fontsize=14, y=0.98,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_counts(counts: dict[str, int], main_percentile: int, output_path: Path) -> None:
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    cats = [c for c, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        range(len(cats)), vals, color=colors,
        edgecolor="black", linewidth=0.5,
    )
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=11)
    ax.invert_yaxis()  # largest count at the top
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(
        f"Head categories at τ = {main_percentile}th percentile",
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
    print("Loading attribution scores")
    print("=" * 70)
    score_A, La, Ha = compute_difference(Path(args.audio_dir))
    score_V, Lv, Hv = compute_difference(Path(args.visual_dir))
    score_AV, Lav, Hav = compute_difference(Path(args.av_dir))

    print("Per-modality stats:")
    for name, arr in [("A", score_A), ("V", score_V), ("AV", score_AV)]:
        assert_input_sane(name, arr)
        summarize(name, arr)

    if (La, Ha) != (Lv, Hv) or (La, Ha) != (Lav, Hav):
        raise SystemExit(
            f"Layer/head shape mismatch: "
            f"A=({La},{Ha}) V=({Lv},{Hv}) AV=({Lav},{Hav}). "
            "All three runs must use the same Qwen2.5-Omni model."
        )
    L, H = La, Ha
    total = L * H

    print("\n" + "=" * 70)
    print(f"Thresholds (over |scores| pooled across all 3 maps, {3 * total} values)")
    print("=" * 70)
    combined_abs = np.concatenate(
        [np.abs(score_A).ravel(), np.abs(score_V).ravel(), np.abs(score_AV).ravel()]
    )
    # Always compute the three robustness thresholds (90/95/99) AND the
    # user-selected --main_percentile so it's a guaranteed key in `tau`.
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
    print(f"Membership at τ_{main_p} (= {tau_main:.6g}) — main threshold")
    print("=" * 70)
    in_A_main, in_V_main, in_AV_main, cat_main = assign_categories(
        score_A, score_V, score_AV, tau_main
    )
    print(f"  |H_A|  = {int(in_A_main.sum())}")
    print(f"  |H_V|  = {int(in_V_main.sum())}")
    print(f"  |H_AV| = {int(in_AV_main.sum())}")

    counts_main = category_counts(cat_main)
    counts_sum = sum(counts_main.values())
    print(f"\n  Category counts at τ_{main_p} (sum must equal total heads):")
    for c in CATEGORY_ORDER:
        print(f"    {c:<20s} {counts_main[c]:>6d}")
    print(f"    {'TOTAL':<20s} {counts_sum:>6d}  (expected {total})")
    assert counts_sum == total, (
        f"Category counts ({counts_sum}) do not sum to total heads ({total})"
    )

    # --- counts at all three robustness thresholds (always 90/95/99) ---
    counts_per_tau: dict[int, dict[str, int]] = {}
    for p in (90, 95, 99):
        _, _, _, cat_p = assign_categories(score_A, score_V, score_AV, tau[p])
        counts_per_tau[p] = category_counts(cat_p)

    # --- counts.csv (always carries 90/95/99 side-by-side) ---
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

    # --- heads.csv (at the MAIN percentile) ---
    layers, heads = np.meshgrid(np.arange(L), np.arange(H), indexing="ij")
    heads_df = pd.DataFrame({
        "layer":    layers.ravel(),
        "head":     heads.ravel(),
        "score_A":  score_A.ravel(),
        "score_V":  score_V.ravel(),
        "score_AV": score_AV.ravel(),
        "in_H_A":   in_A_main.ravel(),
        "in_H_V":   in_V_main.ravel(),
        "in_H_AV":  in_AV_main.ravel(),
        "category": cat_main.ravel(),
    })
    heads_csv = out_dir / "heads.csv"
    heads_df.to_csv(heads_csv, index=False)

    print("\n" + "=" * 70)
    print(f"Outputs (main τ = {main_p}th percentile)")
    print("=" * 70)
    print(f"  wrote {heads_csv}")
    print(f"  wrote {counts_csv}")
    plot_scatter(
        score_A, score_V, score_AV, cat_main, counts_main, main_p,
        out_dir / "categories_scatter.png",
    )
    plot_counts(counts_main, main_p, out_dir / "category_counts.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir", default=str(_DEFAULT_AUDIO_DIR),
        help="AudioSet describe attribution dir (with pth/ inside).",
    )
    parser.add_argument(
        "--visual_dir", default=str(_DEFAULT_VISUAL_DIR),
        help="ActivityNet describe attribution dir.",
    )
    parser.add_argument(
        "--av_dir", default=str(_DEFAULT_AV_DIR),
        help="VGGSounder describe attribution dir.",
    )
    parser.add_argument(
        "--output_dir", default=str(_DEFAULT_OUT),
        help="Where to dump heads.csv, counts.csv, and the two PNGs.",
    )
    parser.add_argument(
        "--main_percentile", type=int, default=90,
        help=(
            "Which percentile of pooled |scores| to use as the headline τ "
            "(default: 90). The plots and the `category` column in heads.csv "
            "are computed at this τ; counts.csv always carries the 90/95/99 "
            "robustness table regardless."
        ),
    )
    args = parser.parse_args()
    main(args)
