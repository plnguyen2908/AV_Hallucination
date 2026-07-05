"""
_4_1bc_filtered_full.py — re-do 4.1b boxplot + 4.1c overlap with
modality_attn ≥ 0.2 filter applied SYMMETRICALLY to both halluc heads
and Inert heads.

Per dataset, the filter is on the matched modality_attn:
    AudioSet      → den_audio
    ActivityNet   → den_video
    VGGSounder    → den_audio + den_video
A head with mean modality_attn < 0.2 across clips is dropped from BOTH
the halluc category and the Inert pool. This removes the "share is
artefactually 0 or 1 because the head barely attends to modality"
contamination from both sides of the comparison.

Outputs (in `stage4_1b/`):
    boxplots_filtered.png          — 4.1b boxplot redo on filtered pool
    boxplots_filtered_summary.csv  — same as 4.1b summary on filtered pool
    overlap_filtered_symmetric.csv — 4.1c K' = M_f matched table (the
                                     correct comparison when both sides
                                     are restricted to the filtered pool)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_IN = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1b"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"
TOTAL_HEADS = 784

CATEGORY_COLORS = {
    "Audiovisual head": "#9467bd",
    "Audio head":       "#d62728",
    "Visual head":      "#1f77b4",
    "Inert":            "#cccccc",
}

DATASETS = {
    "AudioSet":    dict(label="Audio head",        num="num_audio", den="den_audio"),
    "ActivityNet": dict(label="Visual head",       num="num_video", den="den_video"),
    "VGGSounder":  dict(label="Audiovisual head",  num="num_av",    den="den_av"),
}


def per_head_share(df: pd.DataFrame, num_col: str, den_col: str) -> pd.DataFrame:
    d = df.copy()
    if num_col == "num_av":
        d["num_av"] = d["num_audio"] + d["num_video"]
    if den_col == "den_av":
        d["den_av"] = d["den_audio"] + d["den_video"]
    valid = d[den_col] > 0
    d["share_per_clip"] = np.where(valid, d[num_col] / d[den_col].clip(lower=1e-30), np.nan)
    out = (d.groupby(["layer", "head"])
              .agg(share=("share_per_clip", "mean"),
                    modality_attn=(den_col, "mean"))
              .reset_index())
    return out


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = in_dir
    heads = pd.read_csv(args.heads_csv)[["layer", "head", "category"]]
    print(f"Filter: modality_attn >= {args.filter_min}\n")

    # ------------------------------------------------------------
    # Part 1 — 4.1b boxplot redo, but symmetric filter applied
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    bx_rows = []
    overlap_rows = []
    pairs = [
        ("AudioSet",    "Audio head"),
        ("ActivityNet", "Visual head"),
        ("VGGSounder",  "Audiovisual head"),
    ]
    for ax, (ds, cat) in zip(axes, pairs):
        cfg = DATASETS[ds]
        raw = pd.read_csv(in_dir / f"share_per_head_{ds}.csv")
        ph = per_head_share(raw, cfg["num"], cfg["den"])
        ph = ph.merge(heads, on=["layer", "head"], how="left")

        # Symmetric filter on modality_attn
        filt = ph[ph.modality_attn >= args.filter_min].copy()
        n_full = len(ph)
        n_filt = len(filt)

        cat_pre = (ph.category == cat).sum()
        inert_pre = (ph.category == "Inert").sum()
        cat_pos = filt[(filt.category == cat) & filt.share.notna()]["share"].values
        inert_pos = filt[(filt.category == "Inert") & filt.share.notna()]["share"].values

        try:
            mw_t = stats.mannwhitneyu(cat_pos, inert_pos, alternative="two-sided")
            p_two = float(mw_t.pvalue)
            mw_g = stats.mannwhitneyu(cat_pos, inert_pos, alternative="greater")
            p_g = float(mw_g.pvalue)
            mw_l = stats.mannwhitneyu(cat_pos, inert_pos, alternative="less")
            p_l = float(mw_l.pvalue)
        except Exception:
            p_two = p_g = p_l = float("nan")

        bp = ax.boxplot(
            [inert_pos, cat_pos],
            tick_labels=[f"Inert (filt)\n(n={len(inert_pos)}/{inert_pre})",
                          f"{cat} (filt)\n(n={len(cat_pos)}/{cat_pre})"],
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

        if len(cat_pos) <= 200 and len(cat_pos) > 0:
            rng = np.random.default_rng(0)
            xs = 2 + rng.uniform(-0.18, 0.18, size=len(cat_pos))
            ax.scatter(xs, cat_pos, s=12,
                        color=CATEGORY_COLORS[cat], edgecolor="black",
                        linewidths=0.4, alpha=0.85, zorder=5)

        med_cat = float(np.median(cat_pos)) if len(cat_pos) else float("nan")
        med_in  = float(np.median(inert_pos)) if len(inert_pos) else float("nan")
        ax.set_title(
            f"{cat} vs Inert on {ds} (filtered)\n"
            f"median: {cat} {med_cat:.3f}  vs  Inert {med_in:.3f}  |  "
            f"M-W p={p_two:.2g} (2-sided)",
            fontsize=11)
        ax.set_ylabel(
            "share = Σ A(gen→LLM-emerged-sink∩modality) / Σ A(gen→modality)",
            fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(bottom=0)

        bx_rows.append(dict(
            comparison=f"{cat} vs Inert",
            dataset=ds,
            filter_min=args.filter_min,
            n_cat_filt=len(cat_pos), n_cat_full=int(cat_pre),
            n_inert_filt=len(inert_pos), n_inert_full=int(inert_pre),
            median_cat=med_cat, median_inert=med_in,
            mean_cat=float(np.mean(cat_pos)) if len(cat_pos) else float("nan"),
            mean_inert=float(np.mean(inert_pos)) if len(inert_pos) else float("nan"),
            mw_p_two=p_two, mw_p_greater=p_g, mw_p_less=p_l,
        ))

        # 4.1c filtered-symmetric overlap (K' = M_f)
        # Within the filtered pool, take top-K' by rate where K' = M_f
        ph_f = filt.copy()
        ph_f["rate"] = 1.0 - ph_f["share"]
        ph_f_sorted = ph_f.sort_values(
            ["rate", "modality_attn"], ascending=[False, False])
        K_alt = max(int((filt.category == cat).sum()), 1)
        topKalt = ph_f_sorted.head(K_alt)
        overlap_alt = int((topKalt.category == cat).sum())
        chance_alt = K_alt * K_alt / max(n_filt, 1)
        p_alt = float(stats.hypergeom.sf(
            overlap_alt - 1, n_filt, K_alt, K_alt))
        overlap_rows.append(dict(
            modality=cat.split()[0].lower(),
            dataset=ds, halluc_category=cat,
            filter_min=args.filter_min,
            N_filt=n_filt, K_prime=K_alt,
            overlap=overlap_alt,
            pct=100*overlap_alt/max(K_alt,1),
            chance=chance_alt, p_hyper=p_alt,
            odds=(overlap_alt/max(K_alt,1)) / max(K_alt/max(n_filt,1), 1e-30),
        ))

    fig.suptitle(
        f"Stage 4.1b (filtered) — share of attention from generated text to "
        f"LLM-emerged modality sinks, per head category × dataset.\n"
        f"Symmetric filter modality_attn ≥ {args.filter_min} on BOTH halluc "
        f"and Inert. n=50 clips per dataset.",
        fontsize=12)
    out_png = out_dir / "boxplots_filtered.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    bx_df = pd.DataFrame(bx_rows)
    bx_csv = out_dir / "boxplots_filtered_summary.csv"
    bx_df.to_csv(bx_csv, index=False)
    print(f"wrote {bx_csv}")
    print("\n" + bx_df.to_string(index=False, float_format="{:.4g}".format))

    ov_df = pd.DataFrame(overlap_rows)
    ov_csv = out_dir / "overlap_filtered_symmetric.csv"
    ov_df.to_csv(ov_csv, index=False)
    print(f"\nwrote {ov_csv}")
    print("\n" + ov_df.to_string(index=False, float_format="{:.4g}".format))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=str(DEFAULT_IN))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--filter_min", type=float, default=0.2)
    args = p.parse_args()
    main(args)
