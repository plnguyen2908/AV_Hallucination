"""
stage4_3b_l27_isolation.py — Stage 4.3b L27 noise control.

Pure re-aggregation of `stage4_3/stage4_3_output_records.csv`. No new
forward passes; no new head definitions. Decompose the τ_99 audio set
(n=12) into {non-L27 audio n=9} vs {L27 audio n=3} and ask whether the
positive Stage 4.3 DiD survives without L27, AND whether the 3
audio-L27 heads sit OUTSIDE the L27 Inert hal-non gap distribution.

Tests (all on ‖c_h(t)‖ primary and cos_to_total secondary):
  TEST 1  drop L27 audio heads → 9 audio vs layer-matched Inert at L0/1/4/16/17/20.
          Raw + position-controlled DiD; same cluster-robust OLS as 4.3.
  TEST 2  isolate the 3 audio-L27 heads vs Inert-L27; per-head hal−non gaps.
  TEST 3  compare the 3 audio-L27 hal−non gaps to the 25 Inert-L27
          hal−non gap distribution (per-head means across 37 clips).
          The decisive panel.
  TEST 4  same as 1–3 on cos_to_total.

Outputs (`stage4_3b/`):
  l27_isolation.csv  rows for each (head_set, readout, model) of the DiD
                     plus per-head hal−non summary rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
RECORDS_CSV = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_3/stage4_3_output_records.csv"
HEADS_CSV   = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
OUT_DIR     = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_3b"


def categorize(in_A, in_V, in_AV):
    if in_A and in_V and in_AV: return "Generic"
    if in_A and in_V:            return "Compensated"
    if in_A and in_AV:           return "Audio + AV"
    if in_V and in_AV:           return "Visual + AV"
    if in_A:                     return "Audio-only"
    if in_V:                     return "Visual-only"
    if in_AV:                    return "Cross-modal-only"
    return "Inert"


def load_head_sets():
    h = pd.read_csv(HEADS_CSV)
    pool = np.abs(np.concatenate([h["score_A"].values, h["score_V"].values,
                                    h["score_AV"].values]))
    tau99 = float(np.percentile(pool, 99))
    h["cat"] = [
        categorize(h["score_A"].iloc[i] > tau99,
                   h["score_V"].iloc[i] > tau99,
                   h["score_AV"].iloc[i] > tau99)
        for i in range(len(h))]
    ao = h[h["cat"] == "Audio-only"][["layer", "head", "score_A"]] \
            .sort_values("score_A", ascending=False).reset_index(drop=True)
    inert = h[h["cat"] == "Inert"][["layer", "head"]]
    return ao, inert


def cluster_robust_ols(X, y, groups):
    from scipy.stats import norm
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((k, k))
    G = 0
    for g in np.unique(groups):
        idx = groups == g
        Xg = X[idx]
        eg = resid[idx]
        Xg_eg = Xg.T @ eg
        meat += np.outer(Xg_eg, Xg_eg); G += 1
    cov = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return beta, se, beta - 1.96 * se, beta + 1.96 * se, p


def did_for_head_set(records: pd.DataFrame, ao_heads: pd.DataFrame,
                      inert_heads: pd.DataFrame, readout: str, label: str):
    """Compute raw DiD (per-clip paired) and position-controlled DiD
    (cluster-robust OLS) for an Audio-only subset vs layer-matched Inert.

    Layer-matched: Inert heads are restricted to the layers where the
    Audio-only subset has heads.
    """
    layers_used = sorted(set(ao_heads["layer"].tolist()))
    ao_set = set(zip(ao_heads["layer"], ao_heads["head"]))
    inert_set = set(zip(inert_heads["layer"], inert_heads["head"]))

    df = records.copy()
    df["head_class"] = df.apply(
        lambda r: ("Audio-only" if (r["layer"], r["head"]) in ao_set
                   else ("Inert" if (r["layer"], r["head"]) in inert_set else "other")),
        axis=1)
    df = df[df["head_class"].isin(["Audio-only", "Inert"])]
    df = df[df["layer"].isin(layers_used)]

    # Per (clip, class, layer, head, token_type) mean
    per_head = (df.groupby(["clip", "head_class", "layer", "head", "token_type"],
                             as_index=False).agg(mean_val=(readout, "mean")))
    wide = per_head.pivot_table(
        index=["clip", "head_class", "layer", "head"],
        columns="token_type", values="mean_val").reset_index()
    wide["hal_minus_non"] = wide.get("hal", np.nan) - wide.get("non", np.nan)
    ao_part = wide[wide["head_class"] == "Audio-only"]
    in_part = wide[wide["head_class"] == "Inert"]
    g_ao = ao_part.groupby("clip", as_index=False).agg(gap_H=("hal_minus_non", "mean"))
    g_in = in_part.groupby("clip", as_index=False).agg(gap_I=("hal_minus_non", "mean"))
    by_clip = g_ao.merge(g_in, on="clip", how="inner")
    by_clip["DiD"] = by_clip["gap_H"] - by_clip["gap_I"]

    x = by_clip["DiD"].dropna().values
    n = len(x)
    m = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else float("nan")
    from scipy.stats import t as _t
    p = float(2 * _t.sf(abs(m / se if se else 0.0), df=n - 1)) if n > 1 else float("nan")
    raw = dict(label=label, readout=readout, test="raw_DiD", n_clips=n,
               n_audio_heads=int(ao_part[["layer","head"]].drop_duplicates().shape[0]),
               n_inert_heads=int(in_part[["layer","head"]].drop_duplicates().shape[0]),
               mean_gap_H=float(by_clip["gap_H"].mean()),
               mean_gap_I=float(by_clip["gap_I"].mean()),
               coef=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se, pval=p,
               excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0)))

    # Position-controlled
    sub = df.copy()
    sub["is_hal"] = (sub["token_type"] == "hal").astype(int)
    sub["is_ao"]  = (sub["head_class"] == "Audio-only").astype(int)
    X_ = np.column_stack([
        np.ones(len(sub)),
        sub["token_position_in_caption"].values,
        sub["is_hal"].values, sub["is_ao"].values,
        sub["is_hal"].values * sub["is_ao"].values,
    ])
    y_ = sub[readout].values
    beta, se_, lo, hi, pv = cluster_robust_ols(X_, y_, sub["clip"].values)
    pc = dict(label=label, readout=readout, test="position_controlled_DiD",
              n_obs=int(len(sub)),
              n_audio_heads=raw["n_audio_heads"], n_inert_heads=raw["n_inert_heads"],
              coef=float(beta[4]), ci95_low=float(lo[4]), ci95_high=float(hi[4]),
              pval=float(pv[4]),
              excludes_zero=bool((lo[4] > 0) or (hi[4] < 0)),
              position_coef=float(beta[1]), position_pval=float(pv[1]))
    return [raw, pc], by_clip


def per_head_hal_non(records: pd.DataFrame, head_set: pd.DataFrame, readout: str,
                      tag: str):
    """For each (layer, head) in head_set, return per-clip and aggregate
    hal-non gap. Returns a long-form DataFrame."""
    rows = []
    for _, hrow in head_set.iterrows():
        L, h_idx = int(hrow["layer"]), int(hrow["head"])
        sub = records[(records["layer"] == L) & (records["head"] == h_idx)]
        per = (sub.groupby(["clip", "token_type"], as_index=False)
                   .agg(m=(readout, "mean")))
        w = per.pivot_table(index="clip", columns="token_type", values="m").reset_index()
        w["hal_minus_non"] = w.get("hal", np.nan) - w.get("non", np.nan)
        n = int(w["hal_minus_non"].notna().sum())
        vals = w["hal_minus_non"].dropna().values
        m_   = float(vals.mean()) if vals.size else float("nan")
        sd_  = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        rows.append(dict(tag=tag, readout=readout, layer=L, head=h_idx,
                          mean_non=float(w.get("non", pd.Series([np.nan])).mean()),
                          mean_hal=float(w.get("hal", pd.Series([np.nan])).mean()),
                          mean_hal_minus_non=m_, sd_across_clips=sd_,
                          n_clips=n))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(RECORDS_CSV)
    ao_all, inert_all = load_head_sets()
    ao_l27   = ao_all[ao_all["layer"] == 27]
    ao_nol27 = ao_all[ao_all["layer"] != 27]
    inert_l27 = inert_all[inert_all["layer"] == 27]
    print(f"audio τ_99 total = {len(ao_all)}  ; L27 audio = {len(ao_l27)} (excluded) "
          f"; non-L27 audio = {len(ao_nol27)}; Inert L27 = {len(inert_l27)}")

    all_rows = []
    for readout in ("norm", "cos_to_total"):
        # TEST 1 — drop L27 audio
        rows1, _ = did_for_head_set(
            records, ao_nol27, inert_all,
            readout=readout, label="TEST1_audio_NoL27_vs_InertLayerMatched_NoL27")
        all_rows.extend(rows1)
        # TEST 2 — L27 audio (n=3) vs L27 Inert
        rows2, _ = did_for_head_set(
            records, ao_l27, inert_all,
            readout=readout, label="TEST2_audio_L27_vs_Inert_L27")
        all_rows.extend(rows2)

        # PER-HEAD for the 3 audio-L27 + the 25 Inert-L27 — for TEST 3
        ph_audio_l27 = per_head_hal_non(records, ao_l27, readout,
                                          tag="audio_L27_per_head")
        ph_inert_l27 = per_head_hal_non(records, inert_l27, readout,
                                          tag="inert_L27_per_head")
        per_head_csv = OUT_DIR / f"per_head_l27_{readout}.csv"
        pd.concat([ph_audio_l27, ph_inert_l27], ignore_index=True).to_csv(
            per_head_csv, index=False)
        print(f"\n[{readout}]  wrote {per_head_csv}")
        print(f"  audio-L27 per-head hal-non gaps:")
        print(ph_audio_l27[["layer","head","mean_non","mean_hal","mean_hal_minus_non","sd_across_clips"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(f"  inert-L27 hal-non gap distribution (n={len(ph_inert_l27)} heads):")
        vals = ph_inert_l27["mean_hal_minus_non"].dropna().values
        print(f"    mean={vals.mean():+.4f}  sd={vals.std(ddof=1):+.4f}  "
              f"min={vals.min():+.4f}  q25={np.percentile(vals, 25):+.4f}  "
              f"median={np.median(vals):+.4f}  q75={np.percentile(vals, 75):+.4f}  "
              f"max={vals.max():+.4f}")
        # TEST 3 verdict numbers — where do the 3 audio heads fall in the Inert dist
        for _, ar in ph_audio_l27.iterrows():
            g = ar["mean_hal_minus_non"]
            pctile = float((vals < g).mean() * 100)
            z = (g - vals.mean()) / vals.std(ddof=1)
            print(f"    audio L{int(ar['layer'])}H{int(ar['head'])} gap={g:+.4f}  "
                  f"→ pctile in Inert-L27 = {pctile:.1f}  z = {z:+.2f}")

    summary = pd.DataFrame(all_rows)
    summary_path = OUT_DIR / "l27_isolation.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}  ({len(summary)} rows)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
