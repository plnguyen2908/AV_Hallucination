"""
stage4_2_did_finalize.py — finish Stage 4.2 aggregation after the forward
pass without depending on statsmodels.

Reads:
  stage4_2_entity_attention.csv   (long-form per-(clip, layer, head, query, bin))
Writes:
  stage4_2_did_position_controlled.csv
  stage4_2_dose_response.csv

The position-controlled regression is plain numpy OLS with
cluster-robust (Liang-Zeger) standard errors clustered by clip.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_2"
HEADS_CSV = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"


def categorize(in_A, in_V, in_AV):
    if in_A and in_V and in_AV: return "Generic"
    if in_A and in_V:            return "Compensated"
    if in_A and in_AV:           return "Audio + AV"
    if in_V and in_AV:           return "Visual + AV"
    if in_A:                     return "Audio-only"
    if in_V:                     return "Visual-only"
    if in_AV:                    return "Cross-modal-only"
    return "Inert"


def load_head_classes(heads_csv: Path, tau_percentile: int = 99):
    h = pd.read_csv(heads_csv)
    pool = np.abs(np.concatenate([h["score_A"].values, h["score_V"].values,
                                    h["score_AV"].values]))
    tau_val = float(np.percentile(pool, tau_percentile))
    h["head_class_t99"] = [   # column name kept for downstream stability
        categorize(h["score_A"].iloc[i] > tau_val,
                   h["score_V"].iloc[i] > tau_val,
                   h["score_AV"].iloc[i] > tau_val)
        for i in range(len(h))
    ]
    ao = (h[h["head_class_t99"] == "Audio-only"]
            [["layer", "head", "score_A"]]
            .sort_values("score_A", ascending=False)
            .reset_index(drop=True))
    ao["score_A_rank"] = np.arange(1, len(ao) + 1)
    return h, ao


def cluster_robust_ols(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS with Liang-Zeger cluster-robust SE on `groups`.
    Returns (β, SE_cluster_robust, z, p, ci_lo, ci_hi) per coefficient."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # Σ X_g'e_g e_g'X_g over clusters g
    meat = np.zeros((k, k))
    G = 0
    for g in np.unique(groups):
        idx = groups == g
        Xg = X[idx]
        rg = resid[idx]
        Xg_rg = Xg.T @ rg
        meat += np.outer(Xg_rg, Xg_rg)
        G += 1
    cov_cl = XtX_inv @ meat @ XtX_inv
    # small-sample correction (Stata-style): G/(G-1) * (n-1)/(n-k)
    cov_cl *= (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(cov_cl))
    z = beta / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(z)))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    return beta, se, z, p, ci_lo, ci_hi


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--tau_percentile", type=int, default=99, choices=[90, 95, 99])
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    print(f"output dir: {out_dir}")
    print(f"τ percentile: {args.tau_percentile}")

    print("loading records …")
    df = pd.read_csv(out_dir / "stage4_2_entity_attention.csv")
    print(f"  {len(df):,} rows")

    heads_df, ao_df = load_head_classes(HEADS_CSV, args.tau_percentile)
    ao_set = set(zip(ao_df["layer"], ao_df["head"]))
    inert = heads_df[heads_df["head_class_t99"] == "Inert"]
    inert_set = set(zip(inert["layer"], inert["head"]))

    df["head_class"] = df.apply(
        lambda r: "Audio-only" if (r["layer"], r["head"]) in ao_set
                  else ("Inert" if (r["layer"], r["head"]) in inert_set else "other"),
        axis=1)
    df = df[df["head_class"].isin(["Audio-only", "Inert"])]
    ao_layers = set(ao_df["layer"].tolist())
    df = df[df["layer"].isin(ao_layers)]      # layer-matched only
    print(f"  after class / layer filter: {len(df):,}")

    df["is_hal"] = (df["token_type"] == "hal").astype(int)
    df["is_ao"]  = (df["head_class"] == "Audio-only").astype(int)

    # Position-controlled DiD: per-bin OLS
    #   per_token_inflow ~ 1 + token_position_in_caption + is_hal + is_ao + is_hal*is_ao
    #   cluster-robust SE by clip
    rows = []
    for bname in sorted(df["bin"].unique()):
        sub = df[df["bin"] == bname]
        X = np.column_stack([
            np.ones(len(sub)),                                  # const
            sub["token_position_in_caption"].values,            # position
            sub["is_hal"].values,                                # token_type
            sub["is_ao"].values,                                 # head_class
            (sub["is_hal"].values * sub["is_ao"].values),       # interaction = DiD
        ])
        y = sub["per_token_inflow"].values
        groups = sub["clip"].values
        beta, se, z, p, lo, hi = cluster_robust_ols(X, y, groups)
        # interaction is index 4
        rows.append(dict(
            bin=bname, n_obs=int(len(sub)),
            interaction_coef_DiD=float(beta[4]),
            cluster_se=float(se[4]),
            ci95_low=float(lo[4]),
            ci95_high=float(hi[4]),
            pval_two_sided=float(p[4]),
            excludes_zero=bool((lo[4] > 0) or (hi[4] < 0)),
            position_coef=float(beta[1]),
            position_se=float(se[1]),
            position_pval=float(p[1]),
        ))
    pc = pd.DataFrame(rows)
    pc_path = out_dir / "stage4_2_did_position_controlled.csv"
    pc.to_csv(pc_path, index=False)
    print(f"\nwrote {pc_path}")
    print(pc.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # ---- Dose-response across Audio-only heads ----
    print("\n=== Dose-response (per-head hal−non gap, by |score_A| rank) ===")
    per_head_pivot = (df[df["head_class"] == "Audio-only"]
                       .groupby(["clip", "layer", "head", "bin", "token_type"],
                                 as_index=False)
                       .agg(mean_per_token=("per_token_inflow", "mean")))
    wide = per_head_pivot.pivot_table(
        index=["clip", "layer", "head", "bin"],
        columns="token_type",
        values="mean_per_token").reset_index()
    wide["hal_minus_non"] = wide.get("hal", np.nan) - wide.get("non", np.nan)
    # Aggregate over clips to per-head per-bin gap
    per_head = (wide.groupby(["layer", "head", "bin"], as_index=False)
                       .agg(mean_hal_minus_non=("hal_minus_non", "mean"),
                            n_clips=("clip", "nunique")))
    per_head = per_head.merge(ao_df[["layer", "head", "score_A", "score_A_rank"]],
                                on=["layer", "head"], how="inner")
    dose_path = out_dir / "stage4_2_dose_response.csv"
    per_head.sort_values(["bin", "score_A_rank"]).to_csv(dose_path, index=False)
    print(f"wrote {dose_path}")

    # Compute per-bin Spearman correlation rank vs |hal_minus_non|
    print("\nPer-bin Spearman ρ(rank, hal−non) — monotone iff dose-response:")
    from scipy.stats import spearmanr
    for bname, g in per_head.groupby("bin"):
        rho, p = spearmanr(g["score_A_rank"].values,
                            g["mean_hal_minus_non"].values)
        print(f"  {bname:<14s}  ρ={rho:+.3f}  p={p:.3f}  (n={len(g)} heads)")


if __name__ == "__main__":
    main()
