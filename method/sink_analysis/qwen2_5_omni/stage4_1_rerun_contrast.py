"""
stage4_1_rerun_contrast.py — Stage 4.1 RERUN (count-fix + pre-registered
H1 + dual-τ + BH-FDR exploratory matrix).

Reads:
  results/qwen2_5_omni/sink_analysis/stage4_1_rerun/head_sink_attention.csv
                                            (300-clip head×sink matrix)
  results/qwen2_5_omni/categorize_exp/heads.csv
                                            (per-head score_A / score_V / score_AV)

Class assignment is recomputed from the score columns at BOTH τ_90 and
τ_95 (percentiles of |scores| pooled across A∪V∪AV). The heads.csv
`category` column is NOT trusted (it reflects τ_90 despite the
docstring; see `count_fix.md`).

Outputs (`stage4_1_rerun/`):
  stage4_1_taxonomy_contrast.csv
      one row per (tau, head_class, bin):
        n_heads_in_class (correct, counted on (layer,head) pairs)
        mean_per_token_attention, ci95_low, ci95_high
        mean_delta_vs_inert_layer, delta_ci95_low, delta_ci95_high
        delta_pval, delta_pval_bh_q,
        delta_excludes_zero_uncorrected, delta_fdr_q05_survives,
        mean_n_tokens_in_bin
  stage4_1_per_head_mean.csv
      per (tau_label, layer, head, head_class, bin) clip-averaged values
  stage4_1_h1.csv
      the single pre-registered H1 test, at τ_95 and τ_90, with
      P_prop-specificity check (H1b)

The exploratory matrix uses Benjamini–Hochberg with q=0.05 across all
exploratory cells PER τ separately. Small-n classes (<10 heads) are
flagged `small_n=True` and excluded from FDR.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_ATTN = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1_rerun/head_sink_attention.csv"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1_rerun"

ALL_BINS = ["p_prop_cross", "p_prop_uni_video", "p_prop_uni_audio",
            "p_llm_cross",  "p_llm_uni_video",  "p_llm_uni_audio",
            "nonsink_audio", "nonsink_video", "text_and_bos"]
CLASSES = ["Audio-only", "Visual-only", "Cross-modal-only",
           "Audio + AV", "Visual + AV", "Compensated", "Generic", "Inert"]
SMALL_N_FLAG = 10        # spec: <~10 heads = small-n, flagged not interpreted


def categorize(in_A, in_V, in_AV):
    if in_A and in_V and in_AV: return "Generic"
    if in_A and in_V:            return "Compensated"
    if in_A and in_AV:           return "Audio + AV"
    if in_V and in_AV:           return "Visual + AV"
    if in_A:                     return "Audio-only"
    if in_V:                     return "Visual-only"
    if in_AV:                    return "Cross-modal-only"
    return "Inert"


def build_class_table(heads_df: pd.DataFrame):
    """Recompute (layer, head, head_class) at τ_90 / τ_95 / τ_99 from
    scores. Returns dict {tau_label: DataFrame(layer, head, head_class)}."""
    pool = np.abs(np.concatenate([heads_df["score_A"].values,
                                    heads_df["score_V"].values,
                                    heads_df["score_AV"].values]))
    out = {}
    for tau_p in (90, 95, 99):
        tau_val = float(np.percentile(pool, tau_p))
        cats = []
        for i in range(len(heads_df)):
            row = heads_df.iloc[i]
            cats.append(categorize(row["score_A"] > tau_val,
                                     row["score_V"] > tau_val,
                                     row["score_AV"] > tau_val))
        tbl = heads_df[["layer", "head"]].copy()
        tbl["head_class"] = cats
        out[f"tau_{tau_p}"] = tbl
        print(f"  τ_{tau_p} (={tau_val:.6g}) class counts:")
        print(tbl["head_class"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_string())
        print(f"    sum = {len(tbl)}")
    return out


def _ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2: return float("nan"), float("nan"), float("nan")
    m = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(n))
    # two-sided p via t-distribution (paired one-sample test on zero)
    t = m / se if se > 0 else 0.0
    pval = float(2 * stats.t.sf(abs(t), df=n - 1))
    return m - 1.96 * se, m + 1.96 * se, pval


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: return boolean array of survivors and adjusted q."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0: return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    sorted_p = p[order]
    bh_q = sorted_p * n / ranks
    # enforce monotonicity (running min from the right)
    bh_q = np.minimum.accumulate(bh_q[::-1])[::-1]
    survives_sorted = bh_q <= q
    survives = np.zeros(n, dtype=bool)
    survives[order] = survives_sorted
    q_adj = np.zeros(n)
    q_adj[order] = bh_q
    return survives, q_adj


def fill_empty_bins_with_zero(attn: pd.DataFrame) -> pd.DataFrame:
    """For any (clip, layer, bin) where the bin has n_tokens=0 at that
    layer, no row was emitted at the forward-pass stage (the in-hook
    bin builder skips n_tokens==0). Expand to the full Cartesian
    (clip × layer × head × bin) and insert zero-valued rows for the
    missing combinations: per_token attention to a 0-token bin is 0,
    total_inflow is 0, n_tokens is 0.

    The downstream per-head aggregation now sees these 0 contributions
    on every clip where the bin was structurally absent, and the
    layer-matched Inert control (also 0 at those layers) is computed
    against them, so Δ = 0 - 0 = 0 cleanly instead of being undefined."""
    clips  = attn["clip"].unique()
    layers = sorted(attn["layer"].unique())
    heads  = sorted(attn["head"].unique())
    bins_  = sorted(attn["bin"].unique())
    print(f"  expanding to full grid: |clips|={len(clips)} × |layers|={len(layers)} "
          f"× |heads|={len(heads)} × |bins|={len(bins_)} = "
          f"{len(clips) * len(layers) * len(heads) * len(bins_):,}")
    full = pd.MultiIndex.from_product(
        [clips, layers, heads, bins_],
        names=["clip", "layer", "head", "bin"]).to_frame(index=False)
    merged = full.merge(attn, on=["clip", "layer", "head", "bin"], how="left")
    n_filled = int(merged["per_token_inflow"].isna().sum())
    merged["n_tokens"]         = merged["n_tokens"].fillna(0).astype(int)
    merged["per_token_inflow"] = merged["per_token_inflow"].fillna(0.0)
    merged["total_inflow"]     = merged["total_inflow"].fillna(0.0)
    print(f"  filled {n_filled:,} structurally-zero rows "
          f"(orig attn had {len(attn):,}, post-expansion {len(merged):,})")
    return merged


def per_clip_per_head_means(attn: pd.DataFrame, class_tbl: pd.DataFrame):
    """Merge class assignment, return per-(clip, layer, head, bin) rows
    with delta vs layer-matched Inert at same clip."""
    df = attn.merge(class_tbl, on=["layer", "head"], how="left")
    inert = df[df["head_class"] == "Inert"]
    inert_layer = (inert.groupby(["clip", "layer", "bin"], as_index=False)
                       .agg(inert_layer_pt=("per_token_inflow", "mean")))
    df = df.merge(inert_layer, on=["clip", "layer", "bin"], how="left")
    df["delta_vs_inert_layer"] = df["per_token_inflow"] - df["inert_layer_pt"]
    return df


def aggregate_per_head(df_with_class: pd.DataFrame, tau_label: str):
    return (df_with_class.groupby(["layer", "head", "head_class", "bin"],
                                    as_index=False)
                .agg(mean_pt=("per_token_inflow", "mean"),
                     mean_delta_vs_inert=("delta_vs_inert_layer", "mean"),
                     mean_n_tokens=("n_tokens", "mean"),
                     n_clips=("clip", "nunique"))
                .assign(tau=tau_label))


def class_level_summary(per_head: pd.DataFrame, tau_label: str):
    rows = []
    for (hc, b), g in per_head.groupby(["head_class", "bin"]):
        vals = g["mean_pt"].dropna().values
        deltas = g["mean_delta_vs_inert"].dropna().values
        m_pt, lo_pt, hi_pt = (float(vals.mean()), *_ci95(vals)[:2]) if vals.size else (float("nan"),)*3
        if deltas.size:
            d_lo, d_hi, d_p = _ci95(deltas)
            m_d = float(deltas.mean())
        else:
            d_lo, d_hi, d_p, m_d = float("nan"), float("nan"), float("nan"), float("nan")
        n_heads = int(g[["layer", "head"]].drop_duplicates().shape[0])
        rows.append(dict(
            tau=tau_label, head_class=hc, bin=b,
            n_heads_in_class=n_heads,
            small_n=bool(n_heads < SMALL_N_FLAG),
            mean_per_token_attention=m_pt,
            ci95_low=lo_pt, ci95_high=hi_pt,
            mean_delta_vs_inert_layer=m_d,
            delta_ci95_low=d_lo, delta_ci95_high=d_hi,
            delta_pval=d_p,
            delta_excludes_zero_uncorrected=(bool((d_lo > 0) or (d_hi < 0))
                                              if not np.isnan(d_lo) else False),
            mean_n_tokens_in_bin=float(g["mean_n_tokens"].mean()),
        ))
    return pd.DataFrame(rows)


def apply_fdr_per_tau(summary: pd.DataFrame, q=0.05):
    """BH-FDR across cells WITHIN each tau, EXCLUDING small_n classes
    and the head_class == Inert row (Inert is the baseline, its delta is
    structurally 0 — meaningless to test)."""
    summary["delta_fdr_q05_survives"] = False
    summary["delta_bh_q"] = float("nan")
    for tau_label, sub in summary.groupby("tau"):
        elig = sub[(~sub["small_n"]) & (sub["head_class"] != "Inert")
                    & (sub["delta_pval"].notna())]
        if elig.empty: continue
        survives, q_adj = bh_fdr(elig["delta_pval"].values, q=q)
        summary.loc[elig.index, "delta_fdr_q05_survives"] = survives
        summary.loc[elig.index, "delta_bh_q"] = q_adj
    return summary


def compute_h1(per_head_by_tau: dict):
    """H1a: Visual-only → p_prop_uni_video Δ > 0 (one-sided test on the
    paired delta) and CI excludes 0.
    H1b: same class's p_llm_uni_video Δ NOT positive (specificity).
    Audio-only NULL control: Δ on {p_prop_uni_audio, p_llm_uni_audio} ≈ 0.
    """
    rows = []
    for tau_label, per_head in per_head_by_tau.items():
        for hc, bin_name, tag in [
            ("Visual-only", "p_prop_uni_video",  "H1a (primary)"),
            ("Visual-only", "p_llm_uni_video",   "H1b (specificity)"),
            ("Audio-only",  "p_prop_uni_audio",  "control (audio null)"),
            ("Audio-only",  "p_llm_uni_audio",   "control (audio null)"),
        ]:
            sub = per_head[(per_head["head_class"] == hc) & (per_head["bin"] == bin_name)]
            deltas = sub["mean_delta_vs_inert"].dropna().values
            if deltas.size < 2:
                rows.append(dict(tau=tau_label, tag=tag, head_class=hc, bin=bin_name,
                                  n_heads=len(deltas), mean_delta=float("nan"),
                                  ci95_low=float("nan"), ci95_high=float("nan"),
                                  pval_two_sided=float("nan"), excludes_zero=False))
                continue
            lo, hi, p = _ci95(deltas)
            m = float(deltas.mean())
            rows.append(dict(tau=tau_label, tag=tag,
                              head_class=hc, bin=bin_name,
                              n_heads=int(sub[["layer","head"]].drop_duplicates().shape[0]),
                              mean_delta=m,
                              ci95_low=lo, ci95_high=hi,
                              pval_two_sided=p,
                              excludes_zero=bool((lo > 0) or (hi < 0))))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_csv",  default=str(DEFAULT_ATTN))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.attn_csv} ...")
    attn = pd.read_csv(args.attn_csv)
    print(f"  {len(attn):,} rows; n_clips={attn['clip'].nunique()}")

    print("\nFilling structurally-empty (clip, layer, head, bin) cells with 0 ...")
    attn = fill_empty_bins_with_zero(attn)

    print(f"\nloading {args.heads_csv} (recomputing class tables at τ_90 / τ_95 / τ_99) ...")
    heads = pd.read_csv(args.heads_csv)
    class_tables = build_class_table(heads)

    print("\nMerging classes + computing per-head deltas (per τ) ...")
    per_head_concat = []
    summary_concat = []
    per_head_by_tau = {}
    for tau_label, tbl in class_tables.items():
        df_with_class = per_clip_per_head_means(attn, tbl)
        per_head = aggregate_per_head(df_with_class, tau_label)
        per_head_by_tau[tau_label] = per_head
        per_head_concat.append(per_head)
        summary = class_level_summary(per_head, tau_label)
        summary_concat.append(summary)

    per_head_all = pd.concat(per_head_concat, ignore_index=True)
    summary_all  = pd.concat(summary_concat,  ignore_index=True)
    summary_all  = apply_fdr_per_tau(summary_all, q=0.05)

    ph_path = out_dir / "stage4_1_per_head_mean.csv"
    sm_path = out_dir / "stage4_1_taxonomy_contrast.csv"
    per_head_all.to_csv(ph_path, index=False)
    summary_all.to_csv(sm_path, index=False)
    print(f"\nwrote {ph_path}  ({len(per_head_all)} rows)")
    print(f"wrote {sm_path}  ({len(summary_all)} rows)")

    h1 = compute_h1(per_head_by_tau)
    h1_path = out_dir / "stage4_1_h1.csv"
    h1.to_csv(h1_path, index=False)
    print(f"wrote {h1_path}  ({len(h1)} rows)")

    print("\n" + "=" * 80)
    print("PRE-REGISTERED H1 TEST (Visual-only → P_prop video, n_clips=300):")
    print("=" * 80)
    print(h1.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n" + "=" * 80)
    print("EXPLORATORY MATRIX FDR (q=0.05) SURVIVORS — eligible classes (≥10 heads, non-Inert):")
    print("=" * 80)
    surv = summary_all[summary_all["delta_fdr_q05_survives"]]
    if surv.empty:
        print("  (no cells survived FDR)")
    else:
        print(surv[["tau","head_class","n_heads_in_class","bin",
                     "mean_delta_vs_inert_layer","delta_ci95_low","delta_ci95_high",
                     "delta_pval","delta_bh_q"]].sort_values(["tau","head_class","bin"])
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n" + "=" * 80)
    print("CORRECTED COUNTS per τ (gate verification):")
    print("=" * 80)
    counts_pivot = (summary_all[summary_all["bin"] == "p_prop_cross"]
                      [["tau", "head_class", "n_heads_in_class"]]
                      .pivot(index="head_class", columns="tau", values="n_heads_in_class")
                      .reindex(CLASSES).fillna(0).astype(int))
    print(counts_pivot.to_string())
    for tau_label in counts_pivot.columns:
        print(f"  {tau_label} sum = {counts_pivot[tau_label].sum()}")


if __name__ == "__main__":
    main()
