"""
_4_1c_centric_overlap.py — overlap between attention-based "centric"
heads and attribution-based "halluc" heads (categorize_exp_2axis τ_90).

For each modality and matched dataset:
    non_sink_rate_h(D) = 1 - sink_share_h(D)   (Stage 4.1b per-head metric)

A head is *centric* if it has high non-sink rate on its matched dataset
(i.e., routes its modality attention mostly to NON-sink content tokens).

We take top-K heads by non_sink_rate where K = size of the matching
categorize_exp_2axis category, so the sets are size-matched (cleanest
overlap-vs-chance test).

Comparisons:
    audio-centric (top-K=107 on AudioSet)        vs Audio head category
    vision-centric (top-K=17 on ActivityNet)     vs Visual head category
    AV-centric (top-K=6 on VGGSounder)           vs Audiovisual head category

Per pair:
    overlap     = |centric ∩ halluc|
    P(halluc|centric) = overlap / K
    P(centric|halluc) = overlap / K (symmetric since |sets| = K)
    expected_overlap_by_chance = K^2 / 784
    hypergeometric p          = P(X >= overlap)  with N=784, K=K, n=K

Robustness: also report at top-K ∈ {top half of M-category, K, 2K} (K
fixed at category size).

Outputs (`results/qwen2_5_omni/sink_analysis/stage4_1b/`):
    centric_overlap_summary.csv  one row per (modality, top_K choice)
    centric_overlap.png          bar chart + chance line
    centric_overlap_table.md     readable verdict
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
DEFAULT_IN = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1b"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"

MODALITIES = [
    # (modality_label, dataset_for_metric, halluc_category)
    ("audio",  "AudioSet",    "Audio head"),
    ("visual", "ActivityNet", "Visual head"),
    ("av",     "VGGSounder",  "Audiovisual head"),
]

TOTAL_HEADS = 784      # 28 layers × 28 heads

# Hypergeometric: P(X >= k) with population N, M positives, sample n.
# stats.hypergeom.sf(k-1, N, M, n) = P(X >= k)


def main(args):
    in_dir = Path(args.in_dir)
    heads = pd.read_csv(args.heads_csv)
    heads = heads[["layer", "head", "category"]]
    print(f"Loaded heads.csv: {len(heads)} rows; "
          f"categories: {dict(heads['category'].value_counts())}")

    summary_rows = []
    for mod, ds, cat_name in MODALITIES:
        ph = pd.read_csv(in_dir / f"per_head_share_{ds}.csv")
        ph["non_sink_rate"] = 1.0 - ph["share"]
        # Heads with NaN share = no valid clips; exclude from ranking
        ph_valid = ph.dropna(subset=["non_sink_rate"]).copy()
        n_valid = len(ph_valid)
        K = int((heads["category"] == cat_name).sum())  # category size

        # Define centric: top-K by non_sink_rate (highest first)
        ph_valid = ph_valid.sort_values("non_sink_rate", ascending=False)
        centric_K = ph_valid.head(K)[["layer", "head"]]
        centric_K = centric_K.merge(heads, on=["layer", "head"], how="left")
        in_cat_K = int((centric_K["category"] == cat_name).sum())
        expected = K * K / TOTAL_HEADS
        # Hypergeometric: population = TOTAL_HEADS, M = halluc cat size = K,
        # sample = centric top-K, k observed.
        p_hyper = float(stats.hypergeom.sf(in_cat_K - 1, TOTAL_HEADS, K, K))
        odds = (in_cat_K / max(K, 1)) / (K / TOTAL_HEADS)

        # Also at top-2K (richer set of "centric" heads)
        TWOK = min(2 * K, n_valid)
        centric_2K = ph_valid.head(TWOK)[["layer", "head"]]
        centric_2K = centric_2K.merge(heads, on=["layer", "head"], how="left")
        in_cat_2K = int((centric_2K["category"] == cat_name).sum())
        p_hyper_2K = float(stats.hypergeom.sf(in_cat_2K - 1, TOTAL_HEADS, K, TWOK))
        expected_2K = K * TWOK / TOTAL_HEADS

        # Also at fixed top-100 (orthogonal to halluc-cat size, common scale)
        TOP100 = min(100, n_valid)
        centric_100 = ph_valid.head(TOP100)[["layer", "head"]]
        centric_100 = centric_100.merge(heads, on=["layer", "head"], how="left")
        in_cat_100 = int((centric_100["category"] == cat_name).sum())
        p_hyper_100 = float(stats.hypergeom.sf(in_cat_100 - 1, TOTAL_HEADS, K, TOP100))
        expected_100 = K * TOP100 / TOTAL_HEADS

        print(f"\n=== {mod}-centric vs {cat_name} on {ds} ===")
        print(f"  N_valid heads on this dataset: {n_valid}/{TOTAL_HEADS}")
        print(f"  halluc category size K = {K}")
        print(f"  top-K (matched, K={K}):")
        print(f"    overlap = {in_cat_K}/{K}  "
              f"(= {100*in_cat_K/max(K,1):.1f}% of centric are {cat_name})")
        print(f"    expected by chance = {expected:.2f}")
        print(f"    hypergeometric p(X≥obs) = {p_hyper:.3g}")
        print(f"    odds = obs/chance = {odds:.2f}×")
        print(f"  top-2K (2K={TWOK}):")
        print(f"    overlap = {in_cat_2K} (chance {expected_2K:.1f}, p={p_hyper_2K:.3g})")
        print(f"  top-100:")
        print(f"    overlap = {in_cat_100} (chance {expected_100:.1f}, p={p_hyper_100:.3g})")

        # Reverse direction: of the halluc heads, how many are in centric top-K?
        # Equal to the same overlap when K = K.
        # For top-2K: P(centric|halluc) = overlap/K (asymmetric numerator).
        summary_rows.append(dict(
            modality=mod, dataset=ds, halluc_category=cat_name,
            K=K, n_valid=n_valid,
            overlap_topK=in_cat_K,
            pct_centric_in_halluc_topK=100*in_cat_K/max(K,1),
            expected_topK=expected,
            p_hyper_topK=p_hyper,
            odds_topK=odds,
            overlap_top2K=in_cat_2K, p_hyper_top2K=p_hyper_2K,
            overlap_top100=in_cat_100, p_hyper_top100=p_hyper_100,
        ))

    df_summary = pd.DataFrame(summary_rows)
    out_csv = in_dir / "centric_overlap_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n" + df_summary.to_string(index=False, float_format="{:.4g}".format))

    # Plot bar chart: observed vs chance for top-K
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    labels = [f"{r.modality}-centric ∩ {r.halluc_category}\n"
              f"(K={r.K}, on {r.dataset})"
              for r in df_summary.itertuples()]
    x = np.arange(len(labels))
    obs = df_summary["overlap_topK"].values
    chance = df_summary["expected_topK"].values
    width = 0.35

    bars_obs = ax.bar(x - width/2, obs, width,
                       label="observed", color="#d62728",
                       edgecolor="black", linewidth=0.6)
    bars_chance = ax.bar(x + width/2, chance, width,
                          label="expected (chance)", color="#cccccc",
                          edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars_obs, obs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{int(val)}", ha="center", va="bottom", fontsize=10,
                 color="darkred", weight="bold")
    for bar, val in zip(bars_chance, chance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                 color="black")
    # Annotate p-values
    for xi, p in zip(x, df_summary["p_hyper_topK"].values):
        marker = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(xi, max(obs[xi], chance[xi]) * 1.18,
                 f"hyperg. p={p:.2g} {marker}",
                 ha="center", va="bottom", fontsize=9,
                 color="black" if marker != "ns" else "gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("# heads in overlap (top-K matched)", fontsize=11)
    ax.set_title(
        "Stage 4.1c — Centric→Halluc overlap (top-K matched to halluc-category size)\n"
        "Centric = top-K heads by non-sink rate (1 − sink share) on matched dataset",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    out_png = in_dir / "centric_overlap.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # Write a short markdown table
    md_lines = [
        "# Centric → Halluc overlap (Stage 4.1c)",
        "",
        "Centric = top-K heads by **non-sink rate = 1 − sink share** on "
        "the matched dataset. K = size of the matching halluc category "
        "from categorize_exp_2axis at τ_90. Chance = K²/784.",
        "",
        "| centric (top-K) | halluc category | K | overlap | chance | "
        "% centric→halluc | odds | hyperg. p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df_summary.itertuples():
        md_lines.append(
            f"| {r.modality}-centric (on {r.dataset}) | {r.halluc_category} | "
            f"{r.K} | **{r.overlap_topK}** | {r.expected_topK:.1f} | "
            f"{r.pct_centric_in_halluc_topK:.1f}% | "
            f"{r.odds_topK:.2f}× | {r.p_hyper_topK:.3g} |"
        )
    md_lines += [
        "",
        "## Robustness — overlap at top-2K and top-100",
        "",
        "| centric set | overlap (top-K) | overlap (top-2K) | p (top-2K) | "
        "overlap (top-100) | p (top-100) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in df_summary.itertuples():
        md_lines.append(
            f"| {r.modality}-centric on {r.dataset} | {r.overlap_topK} | "
            f"{r.overlap_top2K} | {r.p_hyper_top2K:.3g} | "
            f"{r.overlap_top100} | {r.p_hyper_top100:.3g} |"
        )

    (in_dir / "centric_overlap_table.md").write_text("\n".join(md_lines))
    print(f"wrote {in_dir / 'centric_overlap_table.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=str(DEFAULT_IN))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    args = p.parse_args()
    main(args)
