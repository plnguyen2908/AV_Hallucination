"""
_4_1c_filtered_overlap.py — overlap with modality_attn ≥ 0.2 filter.

Refines _4_1c_centric_overlap.py by filtering the head universe to
heads whose mean per-clip generated-text-to-modality attention mass
satisfies `den_modality >= 0.2`. This kills the rate-near-1 artifact
where Inert heads with vanishing modality attention monopolise the
top of the rate ranking.

Within the filtered universe, take top-K by non-sink rate (1 − share)
where K = size of the matching halluc category in the FULL universe.
Compute overlap with the halluc category and report:
  - hypergeometric p (population = filtered universe size N_f,
                       positives = halluc heads in filtered pool M_f,
                       sample = K, observed = overlap)
  - chance baseline = K * M_f / N_f

Also reports the alternative K' = M_f (matched to filtered halluc size).

Output: results/qwen2_5_omni/sink_analysis/stage4_1b/
    centric_overlap_filtered.{csv,png,md}
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
MODALITIES = [
    ("audio",  "AudioSet",    "Audio head",        "num_audio", "den_audio"),
    ("visual", "ActivityNet", "Visual head",       "num_video", "den_video"),
    ("av",     "VGGSounder",  "Audiovisual head",  "num_av",    "den_av"),
]


def main(args):
    in_dir = Path(args.in_dir)
    heads = pd.read_csv(args.heads_csv)[["layer", "head", "category"]]
    K_full = {c: int((heads.category == c).sum())
               for c in heads.category.unique()}
    print(f"Full-universe halluc category sizes: {K_full}")
    print(f"Filter: modality_attn >= {args.filter_min}\n")

    summary_rows = []
    for mod, ds, cat, num_col, den_col in MODALITIES:
        raw = pd.read_csv(in_dir / f"share_per_head_{ds}.csv")
        if mod == "av":
            raw["num_av"] = raw["num_audio"] + raw["num_video"]
            raw["den_av"] = raw["den_audio"] + raw["den_video"]
        # Per-clip ratio
        raw["rate_clip"] = np.where(
            raw[den_col] > 0,
            1.0 - raw[num_col] / raw[den_col].clip(lower=1e-30),
            np.nan)
        per_head = raw.groupby(["layer", "head"]).agg(
            non_sink_rate=("rate_clip", "mean"),
            modality_attn=(den_col, "mean"),
        ).reset_index().merge(heads, on=["layer", "head"])
        # Filter
        filt = per_head[per_head.modality_attn >= args.filter_min].copy()
        N_f = len(filt)
        K = K_full[cat]
        M_f = int((filt.category == cat).sum())

        if K > N_f:
            K_eff = N_f
        else:
            K_eff = K

        # Top-K by rate among filtered. Deterministic tie-break: when
        # many heads tie at rate=1.0, prefer the one with higher
        # modality_attn (i.e., actually attends a lot to modality, not a
        # marginal pass through the 0.2 filter).
        filt_sorted = filt.sort_values(
            ["non_sink_rate", "modality_attn"], ascending=[False, False])
        topK = filt_sorted.head(K_eff)
        overlap_K = int((topK.category == cat).sum())
        chance_K = K_eff * M_f / max(N_f, 1)
        # Hypergeometric: P(X >= overlap_K)
        p_K = float(stats.hypergeom.sf(overlap_K - 1, N_f, M_f, K_eff))
        odds_K = (overlap_K / max(K_eff, 1)) / max(M_f / max(N_f, 1), 1e-30)

        # Alternative K' = M_f (matched to filtered halluc size)
        K_alt = max(M_f, 1)
        topKalt = filt_sorted.head(K_alt)
        overlap_alt = int((topKalt.category == cat).sum())
        chance_alt = K_alt * M_f / max(N_f, 1)
        p_alt = float(stats.hypergeom.sf(overlap_alt - 1, N_f, M_f, K_alt))

        print(f"=== {mod}-centric (filtered modality_attn >= {args.filter_min}) "
              f"vs {cat} on {ds} ===")
        print(f"  filtered universe size N_f = {N_f} / {TOTAL_HEADS}")
        print(f"  halluc heads in filtered pool M_f = {M_f} / {K} "
              f"(filter survival)")
        print(f"  top-K (K={K_eff}, matched to full-universe halluc size):")
        print(f"    overlap = {overlap_K} / {K_eff} "
              f"({100*overlap_K/max(K_eff,1):.1f}% of centric are {cat})")
        print(f"    chance = {chance_K:.2f}  odds = {odds_K:.2f}×  "
              f"hyperg. p = {p_K:.3g}")
        print(f"  top-K' (K'={K_alt}, matched to filtered halluc size):")
        print(f"    overlap = {overlap_alt} / {K_alt} "
              f"({100*overlap_alt/max(K_alt,1):.1f}%)  "
              f"chance = {chance_alt:.3f}  p = {p_alt:.3g}\n")

        summary_rows.append(dict(
            modality=mod, dataset=ds, halluc_category=cat,
            filter_min=args.filter_min,
            N_full=TOTAL_HEADS, N_filt=N_f,
            K_full=K, M_filt=M_f,
            overlap_K=overlap_K, pct_K=100*overlap_K/max(K_eff,1),
            chance_K=chance_K, odds_K=odds_K, p_hyper_K=p_K,
            overlap_Kalt=overlap_alt, Kalt=K_alt,
            chance_Kalt=chance_alt, p_hyper_Kalt=p_alt,
        ))

    df_sum = pd.DataFrame(summary_rows)
    out_csv = in_dir / "centric_overlap_filtered.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print("\n" + df_sum.to_string(index=False, float_format="{:.4g}".format))

    # Bar chart: observed vs chance, top-K and top-K_alt side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                                constrained_layout=True)

    # Top-K (matched to FULL halluc size)
    ax = axes[0]
    labels = [f"{r.modality}-centric ∩ {r.halluc_category}\n"
              f"(K={r.K_full}, on {r.dataset})"
              for r in df_sum.itertuples()]
    x = np.arange(len(labels))
    obs = df_sum["overlap_K"].values
    chance = df_sum["chance_K"].values
    width = 0.35
    ax.bar(x - width/2, obs, width, label="observed",
            color="#d62728", edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, chance, width, label="chance",
            color="#cccccc", edgecolor="black", linewidth=0.6)
    for xi, (o, c, p) in enumerate(zip(obs, chance,
                                         df_sum["p_hyper_K"].values)):
        marker = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(xi - width/2, o + 0.5, f"{int(o)}", ha="center",
                fontsize=10, color="darkred", weight="bold")
        ax.text(xi + width/2, c + 0.5, f"{c:.1f}", ha="center",
                fontsize=9)
        ax.text(xi, max(o, c) * 1.15, f"p={p:.2g} {marker}",
                ha="center", va="bottom", fontsize=9,
                color="black" if marker != "ns" else "gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# heads in overlap", fontsize=11)
    ax.set_title(
        f"top-K matched to FULL halluc size "
        f"(filter: modality_attn ≥ {args.filter_min})",
        fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Top-K_alt (matched to FILTERED halluc size)
    ax = axes[1]
    labels = [f"{r.modality}-centric ∩ {r.halluc_category}\n"
              f"(K'={r.Kalt} = filtered {r.halluc_category}, M_f={r.M_filt})"
              for r in df_sum.itertuples()]
    obs = df_sum["overlap_Kalt"].values
    chance = df_sum["chance_Kalt"].values
    ax.bar(x - width/2, obs, width, label="observed",
            color="#d62728", edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, chance, width, label="chance",
            color="#cccccc", edgecolor="black", linewidth=0.6)
    for xi, (o, c, p) in enumerate(zip(obs, chance,
                                         df_sum["p_hyper_Kalt"].values)):
        marker = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(xi - width/2, o + 0.1, f"{int(o)}", ha="center",
                fontsize=10, color="darkred", weight="bold")
        ax.text(xi + width/2, c + 0.1, f"{c:.2f}", ha="center",
                fontsize=9)
        ax.text(xi, max(o, c) * 1.15 + 0.05, f"p={p:.2g} {marker}",
                ha="center", va="bottom", fontsize=9,
                color="black" if marker != "ns" else "gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# heads in overlap", fontsize=11)
    ax.set_title(
        f"top-K' matched to FILTERED halluc size",
        fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Stage 4.1c (filtered) — Centric→Halluc overlap on heads with "
        f"modality_attn ≥ {args.filter_min}",
        fontsize=13)

    out_png = in_dir / "centric_overlap_filtered.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # Markdown
    md_lines = [
        f"# Stage 4.1c (filtered) — centric→halluc with modality_attn ≥ "
        f"{args.filter_min}",
        "",
        f"Filter: keep heads with mean per-clip generated-text-to-modality "
        f"attention mass ≥ {args.filter_min}. Filtered universe N_f and "
        f"halluc-survival M_f reported per dataset.",
        "",
        "## top-K matched to FULL halluc-category size (K_full)",
        "",
        "| centric set | filter | N_f | K_full | M_f | overlap | "
        "% K_eff→halluc | chance | odds | hyperg. p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df_sum.itertuples():
        md_lines.append(
            f"| {r.modality}-centric ({r.dataset}) | "
            f"≥{args.filter_min} | {r.N_filt} | {r.K_full} | {r.M_filt} | "
            f"**{r.overlap_K}** | "
            f"{r.pct_K:.1f}% | {r.chance_K:.2f} | {r.odds_K:.2f}× | "
            f"{r.p_hyper_K:.3g} |")
    md_lines += [
        "",
        "## top-K' matched to FILTERED halluc-category size (K' = M_f)",
        "",
        "| centric set | K' | overlap | chance | hyperg. p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in df_sum.itertuples():
        md_lines.append(
            f"| {r.modality}-centric ({r.dataset}) | {r.Kalt} | "
            f"**{r.overlap_Kalt}** | {r.chance_Kalt:.3f} | "
            f"{r.p_hyper_Kalt:.3g} |")

    (in_dir / "centric_overlap_filtered.md").write_text("\n".join(md_lines))
    print(f"wrote {in_dir / 'centric_overlap_filtered.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=str(DEFAULT_IN))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--filter_min", type=float, default=0.2,
                   help="Minimum mean modality_attn (per-head clip-mean of "
                         "den_modality) to keep in centric ranking pool.")
    args = p.parse_args()
    main(args)
