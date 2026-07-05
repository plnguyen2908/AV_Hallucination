"""
_4_1b_aggregate.py — aggregate per-head share + plot.

Loads:
  results/qwen2_5_omni/sink_analysis/stage4_1b/share_per_head_<dataset>.csv
    cols: clip, layer, head, n_gen, n_audio, n_video,
          n_audio_sink_L, n_video_sink_L,
          num_audio, num_video, den_audio, den_video
  results/qwen2_5_omni/categorize_exp_2axis/heads.csv
    cols: layer, head, score_A, score_V, in_H_A, in_H_V, category
        category ∈ {Audiovisual head, Audio head, Visual head, Inert}

Computes per head:
    share_{L,h}(D) = mean_{c ∈ D} [ num(D,c,L,h) / den(D,c,L,h) ]
where the (num, den) bin pair is dataset-dependent:
    AudioSet     → num_audio, den_audio
    ActivityNet  → num_video, den_video
    VGGSounder   → num_audio + num_video, den_audio + den_video

Per-clip ratio is skipped if den == 0; per-head mean takes only the
valid (finite, well-defined) clips. Head is reported as NaN if no clip
yielded a defined ratio for it.

Comparisons (each = one category vs Inert on the corresponding dataset):
    Audio head        vs Inert  on AudioSet     (107 vs 654)
    Visual head       vs Inert  on ActivityNet  (17 vs 654)
    Audiovisual head  vs Inert  on VGGSounder   (6 vs 654)

Outputs (`results/qwen2_5_omni/sink_analysis/stage4_1b/`):
    per_head_share_<dataset>.csv  per-head clip-mean share + n_valid_clips
    comparison_summary.csv        per (cat, dataset): n_heads, median,
                                    mean, mw_u, mw_p
    boxplots.png                  3-panel boxplot, 300 dpi
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_IN_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1b"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"
DEFAULT_OUT = DEFAULT_IN_DIR

DATASETS = {
    "AudioSet":    dict(label="Audio head",         num="num_audio", den="den_audio"),
    "ActivityNet": dict(label="Visual head",        num="num_video", den="den_video"),
    "VGGSounder":  dict(label="Audiovisual head",   num="num_av",    den="den_av"),
}

CATEGORY_COLORS = {
    "Audiovisual head": "#9467bd",
    "Audio head":       "#d62728",
    "Visual head":      "#1f77b4",
    "Inert":            "#cccccc",
}


def per_head_share(df: pd.DataFrame, num_col: str, den_col: str) -> pd.DataFrame:
    """Per (clip, layer, head): per-clip ratio = num/den (NaN if den=0).
    Per (layer, head): mean over clips of per-clip ratio, plus n_valid_clips."""
    d = df.copy()
    if num_col == "num_av":
        d["num_av"] = d["num_audio"] + d["num_video"]
    if den_col == "den_av":
        d["den_av"] = d["den_audio"] + d["den_video"]
    valid = d[den_col] > 0
    d["share_per_clip"] = np.where(valid, d[num_col] / d[den_col].clip(lower=1e-30),
                                     np.nan)
    out = (d.groupby(["layer", "head"])
              .agg(share=("share_per_clip", "mean"),
                    n_valid_clips=("share_per_clip", lambda s: s.notna().sum()),
                    n_clips=("share_per_clip", "size"))
              .reset_index())
    return out


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load category assignments
    heads = pd.read_csv(args.heads_csv)
    print(f"Loaded heads.csv: {len(heads)} rows, "
          f"categories: {dict(heads.category.value_counts())}")

    # Per dataset: load share CSV, compute per-head clip-mean
    per_head: dict[str, pd.DataFrame] = {}
    for ds, cfg in DATASETS.items():
        csv = in_dir / f"share_per_head_{ds}.csv"
        if not csv.exists():
            print(f"  [WARN] missing {csv}")
            continue
        raw = pd.read_csv(csv)
        ph = per_head_share(raw, cfg["num"], cfg["den"])
        ph = ph.merge(heads[["layer", "head", "category"]],
                       on=["layer", "head"], how="left")
        ph.to_csv(out_dir / f"per_head_share_{ds}.csv", index=False)
        per_head[ds] = ph
        n_clips = raw["clip"].nunique()
        n_valid = ph["n_valid_clips"].mean()
        print(f"\n  {ds}: n_clips={n_clips}, "
              f"per-head mean n_valid_clips = {n_valid:.1f}")

    # Comparisons + plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    summary_rows = []

    pairs = [
        ("AudioSet",    "Audio head"),
        ("ActivityNet", "Visual head"),
        ("VGGSounder",  "Audiovisual head"),
    ]
    for ax, (ds, cat) in zip(axes, pairs):
        if ds not in per_head:
            ax.set_title(f"{ds} — missing"); continue
        ph = per_head[ds]
        cat_shares = ph[(ph.category == cat) & ph.share.notna()]["share"].values
        inert_shares = ph[(ph.category == "Inert") & ph.share.notna()]["share"].values

        # Mann-Whitney U (one-sided greater AND two-sided)
        try:
            mw_g = stats.mannwhitneyu(cat_shares, inert_shares,
                                         alternative="greater")
            u_stat, p_val = float(mw_g.statistic), float(mw_g.pvalue)
            mw_t = stats.mannwhitneyu(cat_shares, inert_shares,
                                         alternative="two-sided")
            p_two = float(mw_t.pvalue)
        except Exception:
            u_stat = p_val = p_two = float("nan")

        # Boxplot
        bp = ax.boxplot(
            [inert_shares, cat_shares],
            tick_labels=[f"Inert\n(n={len(inert_shares)})",
                          f"{cat}\n(n={len(cat_shares)})"],
            widths=0.55, patch_artist=True, showfliers=True,
            medianprops=dict(color="black", linewidth=1.6),
            boxprops=dict(linewidth=0.8, edgecolor="black"),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
            flierprops=dict(marker="o", markersize=3, alpha=0.5,
                              markerfacecolor="black", markeredgecolor="none"),
        )
        bp["boxes"][0].set_facecolor(CATEGORY_COLORS["Inert"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(CATEGORY_COLORS[cat])
        bp["boxes"][1].set_alpha(0.85)

        # Overlay per-head dots (jittered) for the smaller-n category
        if len(cat_shares) <= 200:
            rng = np.random.default_rng(0)
            xs = 2 + rng.uniform(-0.18, 0.18, size=len(cat_shares))
            ax.scatter(xs, cat_shares, s=12,
                        color=CATEGORY_COLORS[cat], edgecolor="black",
                        linewidths=0.4, alpha=0.85, zorder=5)

        med_cat = float(np.median(cat_shares)) if len(cat_shares) else float("nan")
        med_in  = float(np.median(inert_shares)) if len(inert_shares) else float("nan")
        ax.set_title(
            f"{cat} vs Inert on {ds}\n"
            f"median: {cat} {med_cat:.3f}  vs  Inert {med_in:.3f}  |  "
            f"M-W p={p_two:.2g} (2-sided)  p={p_val:.2g} (1-sided >)",
            fontsize=11)
        ax.set_ylabel(
            "share = Σ_q∈gen Σ_k∈LLM-emerged-sink A[q,k] / Σ_q∈gen Σ_k∈modality A[q,k]",
            fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(bottom=0)

        summary_rows.append(dict(
            comparison=f"{cat} vs Inert",
            dataset=ds,
            n_heads_cat=len(cat_shares),
            n_heads_inert=len(inert_shares),
            median_cat=med_cat,
            median_inert=med_in,
            mean_cat=float(np.mean(cat_shares)) if len(cat_shares) else float("nan"),
            mean_inert=float(np.mean(inert_shares)) if len(inert_shares) else float("nan"),
            mw_u=u_stat,
            mw_p_one_sided_greater=p_val,
            mw_p_two_sided=p_two,
        ))

    fig.suptitle(
        "Stage 4.1b — share of attention from generated text to LLM-emerged "
        "modality sinks (excluding propagated), per head category × dataset.\n"
        f"Heads from categorize_exp_2axis at τ_90.  n=50 clips per dataset.  "
        "LLM-emerged sink = p_llm[L] AND NOT p_prop.",
        fontsize=12,
    )
    out_png = out_dir / "boxplots.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")
    print("\n" + summary_df.to_string(index=False, float_format="{:.4g}".format))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=str(DEFAULT_IN_DIR))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
