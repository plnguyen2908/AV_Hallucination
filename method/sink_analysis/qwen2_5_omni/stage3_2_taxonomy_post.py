"""
stage3_2_taxonomy_post.py — Part 1 of Stage 3.2.

Pure post-processing of stage3_1_preSA/ref_all/stage3_1_per_clip_layer.csv.
For each (layer, population in {prop, llm}), sum cell counts across the 50
clips and emit fractions. Cells are uni-video, uni-audio, cross-modal
(mutually exclusive in the revised MDS formulation since cross = neither).

Sanity-checks (printed, no STOP unless cell fractions disagree with the
Stage 3.1 classification_counts figure within rounding):
  - P_prop cross_frac ≈ 0.40-0.55 early dropping toward ~0.40 late.
  - P_llm  cross_frac high mid-stack (~0.74-0.93) dropping toward ~0.5 late.

Output: stage3_2/taxonomy_by_layer.csv
        layer, pop, n_uv, n_ua, n_cross, n_total, uv_frac, ua_frac, cross_frac
And prints the P_prop ∩ cross_modal count per layer (the Stage 5.3 target
population size).

Read-only on Stage 3.1 outputs; writes only under stage3_2/.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_SRC = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_1_preSA/ref_all/stage3_1_per_clip_layer.csv"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2"


def build_taxonomy(src_csv: Path, out_csv: Path):
    df = pd.read_csv(src_csv)
    needed = {"layer", "n_uv_llm", "n_ua_llm", "n_cross_llm", "n_total_sinks_llm",
              "n_uv_prop", "n_ua_prop", "n_cross_prop", "n_total_sinks_prop"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"src CSV missing cols: {missing}")

    rows = []
    for L, g in df.groupby("layer"):
        for pop in ("llm", "prop"):
            n_uv    = int(g[f"n_uv_{pop}"].sum())
            n_ua    = int(g[f"n_ua_{pop}"].sum())
            n_cross = int(g[f"n_cross_{pop}"].sum())
            n_tot   = int(g[f"n_total_sinks_{pop}"].sum())
            denom = n_tot if n_tot > 0 else np.nan
            rows.append(dict(
                layer=int(L), pop=pop,
                n_uv=n_uv, n_ua=n_ua, n_cross=n_cross, n_total=n_tot,
                uv_frac    = float(n_uv    / denom) if n_tot > 0 else float("nan"),
                ua_frac    = float(n_ua    / denom) if n_tot > 0 else float("nan"),
                cross_frac = float(n_cross / denom) if n_tot > 0 else float("nan"),
            ))
    out = pd.DataFrame(rows).sort_values(["layer", "pop"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(out)} rows)")
    return out


def sanity_check(tax: pd.DataFrame):
    """Cross-frac shape vs the Stage 3.1 classification_counts figure.
    Note: tax['pop'], not tax.pop — `.pop` is a DataFrame method."""
    prop = tax[tax["pop"] == "prop"].set_index("layer").sort_index()
    llm  = tax[tax["pop"] == "llm" ].set_index("layer").sort_index()
    early_prop = prop.loc[1:5,  "cross_frac"].mean()
    late_prop  = prop.loc[20:27, "cross_frac"].mean()
    mid_llm    = llm.loc[8:17,  "cross_frac"].mean()
    late_llm   = llm.loc[20:27, "cross_frac"].mean()
    print(f"\nsanity check (cross_frac):")
    print(f"  P_prop L1-L5  mean cross_frac = {early_prop:.3f}  (expect ~0.40-0.55)")
    print(f"  P_prop L20-27 mean cross_frac = {late_prop :.3f}  (expect ~0.40)")
    print(f"  P_llm  L8-17  mean cross_frac = {mid_llm   :.3f}  (expect ~0.74-0.93)")
    print(f"  P_llm  L20-27 mean cross_frac = {late_llm  :.3f}  (expect ~0.50)")
    flags = []
    if not (0.30 <= early_prop <= 0.60):  flags.append("P_prop early cross_frac out of [0.30,0.60]")
    if not (0.30 <= late_prop  <= 0.55):  flags.append("P_prop late  cross_frac out of [0.30,0.55]")
    if not (0.60 <= mid_llm    <= 0.95):  flags.append("P_llm  mid   cross_frac out of [0.60,0.95]")
    if not (0.30 <= late_llm   <= 0.70):  flags.append("P_llm  late  cross_frac out of [0.30,0.70]")
    if flags:
        print("\n  STOP: sanity check failed:")
        for f in flags: print(f"    - {f}")
        raise SystemExit(2)
    else:
        print("  -> within expected envelope.")


def report_prop_cross(tax: pd.DataFrame):
    prop = tax[tax["pop"] == "prop"].set_index("layer").sort_index()
    print("\nP_prop ∩ cross-modal count per layer (Stage 5.3 target population):")
    print(prop[["n_cross", "n_total", "cross_frac"]]
          .rename(columns={"n_cross": "n_prop_cross",
                            "n_total": "n_prop_total",
                            "cross_frac": "prop_cross_frac"})
          .to_string(float_format=lambda x: f"{x:.3f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_csv", default=str(DEFAULT_SRC))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    out_csv = Path(args.out_dir) / "taxonomy_by_layer.csv"
    tax = build_taxonomy(Path(args.src_csv), out_csv)
    sanity_check(tax)
    report_prop_cross(tax)


if __name__ == "__main__":
    main()
