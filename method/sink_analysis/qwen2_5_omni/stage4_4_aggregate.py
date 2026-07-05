"""
stage4_4_aggregate.py — DiD on target-token push + qualitative top-10 dump.

Reads:
  stage4_4_target_logit_records.csv
  stage4_4_top10_L27_primary.csv
Writes:
  stage4_4_output_decode_did.csv
  top_tokens_L27H12_L27H13.md
"""
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
OUT_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_4"
HEADS_CSV = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
PRIMARY_L27 = [(27, 12), (27, 13)]


def categorize(in_A, in_V, in_AV):
    if in_A and in_V and in_AV: return "Generic"
    if in_A and in_V:            return "Compensated"
    if in_A and in_AV:           return "Audio + AV"
    if in_V and in_AV:           return "Visual + AV"
    if in_A:                     return "Audio-only"
    if in_V:                     return "Visual-only"
    if in_AV:                    return "Cross-modal-only"
    return "Inert"


def load_head_classes():
    h = pd.read_csv(HEADS_CSV)
    pool = np.abs(np.concatenate([h["score_A"].values, h["score_V"].values,
                                    h["score_AV"].values]))
    tau95 = float(np.percentile(pool, 95))
    tau99 = float(np.percentile(pool, 99))
    h["t99_class"] = [
        categorize(h["score_A"].iloc[i] > tau99,
                   h["score_V"].iloc[i] > tau99,
                   h["score_AV"].iloc[i] > tau99) for i in range(len(h))]
    h["t95_class"] = [
        categorize(h["score_A"].iloc[i] > tau95,
                   h["score_V"].iloc[i] > tau95,
                   h["score_AV"].iloc[i] > tau95) for i in range(len(h))]
    return h


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
        Xg = X[idx]; eg = resid[idx]
        Xg_eg = Xg.T @ eg
        meat += np.outer(Xg_eg, Xg_eg); G += 1
    cov = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return beta, se, beta - 1.96 * se, beta + 1.96 * se, p


def did_for_subset(records: pd.DataFrame, audio_subset, inert_subset, label: str):
    """Same DiD shape as 4.3/4.3b but on target_logit. Inert is layer-matched
    to the audio subset's layer footprint. Returns rows for raw + position-controlled."""
    layers_used = sorted({L for (L, _) in audio_subset})
    ao_set = set(audio_subset)
    in_set = set(inert_subset)

    df = records.copy()
    df["head_class"] = df.apply(
        lambda r: ("Audio" if (r["layer"], r["head"]) in ao_set
                   else ("Inert" if (r["layer"], r["head"]) in in_set else "other")),
        axis=1)
    df = df[df["head_class"].isin(["Audio", "Inert"])]
    df = df[df["layer"].isin(layers_used)]
    if df.empty:
        return []

    per_head = (df.groupby(["clip", "head_class", "layer", "head", "token_type"],
                             as_index=False)
                  .agg(mean_val=("target_logit", "mean")))
    wide = per_head.pivot_table(
        index=["clip", "head_class", "layer", "head"],
        columns="token_type", values="mean_val").reset_index()
    wide["hal_minus_non"] = wide.get("hal", np.nan) - wide.get("non", np.nan)
    ao_part = wide[wide["head_class"] == "Audio"]
    in_part = wide[wide["head_class"] == "Inert"]
    g_ao = ao_part.groupby("clip", as_index=False).agg(gap_H=("hal_minus_non", "mean"))
    g_in = in_part.groupby("clip", as_index=False).agg(gap_I=("hal_minus_non", "mean"))
    by_clip = g_ao.merge(g_in, on="clip", how="inner")
    by_clip["DiD"] = by_clip["gap_H"] - by_clip["gap_I"]

    rows = []
    x = by_clip["DiD"].dropna().values; n = len(x)
    m  = float(x.mean()) if n else float("nan")
    se = float(x.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else float("nan")
    from scipy.stats import t as _t
    pval = float(2 * _t.sf(abs(m / se if se else 0.0), df=n - 1)) if n > 1 else float("nan")
    rows.append(dict(
        label=label, test="raw_DiD", n_clips=n,
        n_audio_heads=int(ao_part[["layer","head"]].drop_duplicates().shape[0]),
        n_inert_heads=int(in_part[["layer","head"]].drop_duplicates().shape[0]),
        mean_gap_H=float(by_clip["gap_H"].mean()),
        mean_gap_I=float(by_clip["gap_I"].mean()),
        coef=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se, pval=pval,
        excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0))))

    # Position-controlled
    sub = df.copy()
    sub["is_hal"] = (sub["token_type"] == "hal").astype(int)
    sub["is_ao"]  = (sub["head_class"] == "Audio").astype(int)
    X_ = np.column_stack([
        np.ones(len(sub)),
        sub["token_position_in_caption"].values,
        sub["is_hal"].values, sub["is_ao"].values,
        sub["is_hal"].values * sub["is_ao"].values,
    ])
    y_ = sub["target_logit"].values
    beta, se_, lo, hi, pv = cluster_robust_ols(X_, y_, sub["clip"].values)
    rows.append(dict(
        label=label, test="position_controlled_DiD", n_obs=int(len(sub)),
        n_audio_heads=rows[0]["n_audio_heads"],
        n_inert_heads=rows[0]["n_inert_heads"],
        coef=float(beta[4]), ci95_low=float(lo[4]), ci95_high=float(hi[4]),
        pval=float(pv[4]),
        excludes_zero=bool((lo[4] > 0) or (hi[4] < 0)),
        position_coef=float(beta[1]), position_pval=float(pv[1])))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading records …")
    records = pd.read_csv(OUT_DIR / "stage4_4_target_logit_records.csv")
    print(f"  {len(records):,} rows")
    heads = load_head_classes()
    ao_t99 = [(int(r.layer), int(r.head)) for r in heads[heads.t99_class == "Audio-only"].itertuples()]
    ao_t95 = [(int(r.layer), int(r.head)) for r in heads[heads.t95_class == "Audio-only"].itertuples()]
    inert  = [(int(r.layer), int(r.head)) for r in heads[heads.t99_class == "Inert"].itertuples()]

    # Head subsets
    primary_pool  = PRIMARY_L27
    h12_only      = [(27, 12)]
    h13_only      = [(27, 13)]
    h_other_t99   = [(L, H) for (L, H) in ao_t99 if (L, H) not in PRIMARY_L27]
    full_t99      = ao_t99
    full_t95      = ao_t95

    all_rows = []
    all_rows += did_for_subset(records, primary_pool, inert,
                                  "PRIMARY pooled  (L27H12 + L27H13)")
    all_rows += did_for_subset(records, h12_only, inert,
                                  "L27H12 alone")
    all_rows += did_for_subset(records, h13_only, inert,
                                  "L27H13 alone")
    all_rows += did_for_subset(records, h_other_t99, inert,
                                  "other-10 τ_99 (audio − L27H10/H12/H13)")
    all_rows += did_for_subset(records, full_t99, inert,
                                  "full τ_99 (n=12)")
    all_rows += did_for_subset(records, full_t95, inert,
                                  "τ_95 class (n=68, mid-stack lens — approximate)")
    did_df = pd.DataFrame(all_rows)
    did_path = OUT_DIR / "output_decode_did.csv"
    did_df.to_csv(did_path, index=False)
    print(f"\nwrote {did_path}\n")
    print(did_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---- Top-10 decode markdown ----
    top10 = pd.read_csv(OUT_DIR / "stage4_4_top10_L27_primary.csv")
    md_lines = ["# Stage 4.4 — Top-10 logit-lens decode of c_h(t) at L27H12 & L27H13\n"]
    md_lines.append(
        "Each block shows the model's `final_norm + lm_head` applied to the head's "
        "single-token contribution `c_h(t)` at a caption query position. `tok(t)` is "
        "the token sitting at that position in the teacher-forced caption. Top-10 = "
        "the head's most-promoted vocab tokens at that position.\n\n"
        "L27 is the last decoder layer, so the lens is near-exact for these heads. "
        "Showing 3 hallucinated and 3 grounded-token positions per head.\n")
    # Sample 3 hal + 3 non per head
    # NOTE: bracket access (top10["head"]) — `top10.head` is the DataFrame
    # `.head()` method, not the column.
    for (L, H) in PRIMARY_L27:
        md_lines.append(f"\n---\n\n## L{L}H{H}\n")
        for ttype, label in (("hal", "Hallucinated positions"),
                                ("non", "Grounded positions")):
            md_lines.append(f"\n### {label}\n")
            sub = top10[(top10["layer"] == L) & (top10["head"] == H)
                          & (top10["token_type"] == ttype)]
            # Group by (clip, position) and pick 3 samples
            keys = sub[["clip", "token_position_in_caption", "target_token_id"]].drop_duplicates()
            samples = keys.sample(min(3, len(keys)), random_state=42)
            tok = None
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
            except Exception:
                tok = None
            for _, k in samples.iterrows():
                tgt_str = (tok.decode([int(k['target_token_id'])],
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                            if tok else f"id={k['target_token_id']}")
                md_lines.append(f"\n- `{k['clip']}`, pos={int(k['token_position_in_caption'])}, "
                                  f"tok(t) = `{tgt_str!r}` (id {int(k['target_token_id'])})")
                ents = sub[(sub["clip"] == k["clip"])
                            & (sub["token_position_in_caption"] == k["token_position_in_caption"])].sort_values("rank")
                lines = []
                for _, r in ents.iterrows():
                    lines.append(f"      {int(r['rank']):>2d}  id={int(r['vocab_id']):>7d}  "
                                 f"logit={r['logit']:+.3f}  {r['decoded']!r}")
                md_lines.append("```")
                md_lines.append("\n".join(lines))
                md_lines.append("```")
    md_path = OUT_DIR / "top_tokens_L27H12_L27H13.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
