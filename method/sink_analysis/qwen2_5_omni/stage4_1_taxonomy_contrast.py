"""
stage4_1_taxonomy_contrast.py — Stage 4.1 (static contrast by head taxonomy).

Pure post-processing of:
  results/qwen2_5_omni/sink_analysis/stage4/head_sink_attention.csv
  results/qwen2_5_omni/categorize_exp/heads.csv          (τ_95 categories)
  results/qwen2_5_omni/categorize_exp/counts.csv         (for τ_90 control)

For each (head_class, sink_bin, population):
  mean per-token attention pooled over (clip, head ∈ class),
  paired delta vs Inert heads (mean of Inert at SAME layer per head),
  paired delta vs layer-matched random heads (= Inert heads at same layer),
  per-clip CI (paired with clip).

Layer-matched random control: for each head h at layer L, the "random head"
control is the mean per-token attention OVER INERT HEADS AT LAYER L for the
same clip and same sink bin. This isolates "hallucination-head-ness" from
depth, because (a) heads cluster in certain layers per category and
(b) sink-attention varies by depth.

Outputs (`stage4/`):
  stage4_1_taxonomy_contrast.csv   per (head_class × sink_bin × pop):
                                   mean_pt, ci_low, ci_high,
                                   vs_inert_mean, vs_inert_ci_*,
                                   n_heads_in_class
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_ATTN = _REPO / "results/qwen2_5_omni/sink_analysis/stage4/head_sink_attention.csv"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
DEFAULT_OUT_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage4"

ALL_BINS = ("p_prop_cross", "p_prop_uni_video", "p_prop_uni_audio",
            "p_llm_cross",  "p_llm_uni_video",  "p_llm_uni_audio",
            "nonsink_audio", "nonsink_video", "text_and_bos")


def _ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    return m - 1.96 * se, m + 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_csv",  default=str(DEFAULT_ATTN))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.attn_csv} ...")
    df = pd.read_csv(args.attn_csv)
    print(f"  {len(df):,} rows; bins={sorted(df['bin'].unique())}")
    print(f"loading {args.heads_csv} ...")
    heads = pd.read_csv(args.heads_csv)
    heads = heads[["layer", "head", "category"]].rename(columns={"category": "head_class"})
    print(f"  {len(heads)} heads; classes={sorted(heads['head_class'].unique())}")

    # Tag each row with the head's class
    df = df.merge(heads, on=["layer", "head"], how="left")

    # Per-clip per-head mean of per-token attention (each clip contributes
    # one value per (head, bin)). With one clip per row in attn CSV, no
    # extra aggregation needed.
    # Layer-matched control: per (clip, layer, bin), mean over INERT heads
    inert = df[df["head_class"] == "Inert"]
    inert_layer = (inert.groupby(["clip", "layer", "bin"], as_index=False)
                       .agg(inert_layer_pt=("per_token_inflow", "mean"),
                            inert_n_heads=("head", "nunique")))
    df = df.merge(inert_layer, on=["clip", "layer", "bin"], how="left")
    df["delta_vs_inert_layer"] = df["per_token_inflow"] - df["inert_layer_pt"]

    # Per-head per-bin means across clips (each head's signature)
    per_head = (df.groupby(["layer", "head", "head_class", "bin"], as_index=False)
                  .agg(mean_pt=("per_token_inflow", "mean"),
                       mean_delta_vs_inert=("delta_vs_inert_layer", "mean"),
                       mean_n_tokens=("n_tokens", "mean"),
                       n_clips=("clip", "nunique")))
    per_head_path = out_dir / "stage4_1_per_head_mean.csv"
    per_head.to_csv(per_head_path, index=False)
    print(f"wrote {per_head_path}  ({len(per_head)} rows)")

    # Class-level summary: for each (head_class, bin), aggregate per-head means
    rows = []
    for (hc, b), g in per_head.groupby(["head_class", "bin"]):
        vals = g["mean_pt"].dropna().values
        deltas = g["mean_delta_vs_inert"].dropna().values
        m_pt = float(vals.mean()) if vals.size else float("nan")
        ci_pt = _ci95(vals)
        m_d = float(deltas.mean()) if deltas.size else float("nan")
        ci_d = _ci95(deltas)
        rows.append(dict(
            head_class=hc, bin=b,
            n_heads_in_class=int(g["head"].nunique()),
            mean_per_token_attention=m_pt,
            ci95_low=ci_pt[0], ci95_high=ci_pt[1],
            mean_delta_vs_inert_layer=m_d,
            delta_ci95_low=ci_d[0], delta_ci95_high=ci_d[1],
            delta_excludes_zero=bool((ci_d[0] > 0) or (ci_d[1] < 0))
                                  if not np.isnan(ci_d[0]) else False,
            mean_n_tokens_in_bin=float(g["mean_n_tokens"].mean()),
        ))
    summary = pd.DataFrame(rows).sort_values(["head_class", "bin"]).reset_index(drop=True)
    sum_path = out_dir / "stage4_1_taxonomy_contrast.csv"
    summary.to_csv(sum_path, index=False)
    print(f"wrote {sum_path}  ({len(summary)} rows)")

    # Pretty print headline pivots
    print("\n=== Mean per-token attention (× 1e3) by head_class × bin ===")
    pivot = summary.pivot(index="head_class", columns="bin",
                           values="mean_per_token_attention").reindex(
        index=["Audio-only", "Visual-only", "Cross-modal-only", "Audio + AV",
                "Visual + AV", "Compensated", "Generic", "Inert"],
        columns=list(ALL_BINS))
    print((pivot * 1e3).round(3).to_string())

    print("\n=== Delta vs layer-matched Inert (× 1e3) — positive = head over-attends bin vs Inert ===")
    pivot_d = summary.pivot(index="head_class", columns="bin",
                             values="mean_delta_vs_inert_layer").reindex(
        index=["Audio-only", "Visual-only", "Cross-modal-only", "Audio + AV",
                "Visual + AV", "Compensated", "Generic", "Inert"],
        columns=list(ALL_BINS))
    print((pivot_d * 1e3).round(3).to_string())

    print("\n=== Delta excludes 0 (per-class paired CI) — ★ = significant ===")
    pivot_s = summary.pivot(index="head_class", columns="bin",
                             values="delta_excludes_zero").reindex(
        index=["Audio-only", "Visual-only", "Cross-modal-only", "Audio + AV",
                "Visual + AV", "Compensated", "Generic", "Inert"],
        columns=list(ALL_BINS))
    print(pivot_s.fillna(False).replace({True: "★", False: "·"}).to_string())


if __name__ == "__main__":
    main()
