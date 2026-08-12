"""Per-head influence-score heatmaps for the 4 proxy datasets — one PNG per dataset.

Real heatmap style from identify_halluc_head.py `_heatmap`: seaborn coolwarm,
center=0, symmetric limits = ±P(vlim_pct)(|score|). High positive -> red,
high negative -> blue, middle white/grey. No overlay squares.

YouTube-VOS has ~3 outlier heads (max |score| 0.09) that dominate the P99.9
scale and wash the map out, so it uses a lower percentile (P99) by default.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paper-figure style, matched to head_umap_508.py: no in-figure title (the
# caption/subfigure label carries the dataset name), serif text, large type,
# vector output, opaque white canvas.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 25,
    "axes.labelsize": 30,      # axis text
    "xtick.labelsize": 25,     # tick numbers
    "ytick.labelsize": 25,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,        # embed TrueType, not Type3 (camera-ready)
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.transparent": False,
})

_REPO = Path(__file__).resolve().parent.parent.parent
CSV = _REPO / "results/qwen2_5_omni/head_pca_4datasets/head_pca_4datasets.csv"
OUT = _REPO / "results/qwen2_5_omni/head_pca_4datasets"
DATASETS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
# per-dataset color-scale percentile (YouTube-VOS reduced: outlier-dominated)
VLIM_PCT = {"AudioSet": 99.9, "LibriSpeech": 99.9, "ActivityNet": 99.9, "YouTubeVOS": 99.0}
NL, NH = 28, 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ytvos_pct", type=float, default=99.0,
                    help="color-scale percentile for YouTube-VOS")
    ap.add_argument("--vlim_scale", type=float, default=2.5,
                    help=">1 widens the color scale -> paler / more grey")
    args = ap.parse_args()
    VLIM_PCT["YouTubeVOS"] = args.ytvos_pct
    # per-dataset extra widening; YouTube-VOS is brighter (wider spread) -> more
    SCALE = {ds: args.vlim_scale for ds in DATASETS}
    SCALE["YouTubeVOS"] = args.vlim_scale * 1.8
    df = pd.read_csv(CSV)

    for ds in DATASETS:
        data = np.zeros((NL, NH))
        for _, r in df.iterrows():
            data[int(r["layer"]), int(r["head"])] = r[f"score_{ds}"]
        # center on the per-dataset MEDIAN so the global positive offset of some
        # datasets (AudioSet/YouTube-VOS/LibriSpeech skew positive) doesn't paint
        # the whole map red; balances red/blue like ActivityNet (median ≈ 0).
        med = float(np.median(data))
        v = np.percentile(np.abs(data - med), VLIM_PCT[ds]) * SCALE[ds]

        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        sns.heatmap(data, cmap="coolwarm", center=med, vmin=med - v, vmax=med + v, ax=ax,
                    cbar_kws={"shrink": 0.85, "pad": 0.02,
                              "label": "influence score"})
        # 28x28 ticks cannot be labelled at 25pt -- show every 4th, unrotated.
        ticks = np.arange(0, NL, 4)
        ax.set_xticks(ticks + 0.5); ax.set_xticklabels(ticks, rotation=0)
        ax.set_yticks(ticks + 0.5); ax.set_yticklabels(ticks, rotation=0)
        ax.tick_params(colors="#555555", length=4, pad=4)
        ax.set_xlabel("head index"); ax.set_ylabel("layer index")
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=20, colors="#555555")
        cb.set_label("influence score", size=26)
        cb.outline.set_visible(False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"head_heatmap_{ds}.{ext}", dpi=400,
                        bbox_inches="tight", facecolor="white",
                        edgecolor="none", transparent=False)
        fn = OUT / f"head_heatmap_{ds}.png"
        plt.close(fig)
        print(f"{ds}: vlim=±{v:.4f} (P{VLIM_PCT[ds]}) -> {fn.name}")


if __name__ == "__main__":
    main()
