"""
stage4_3_output_did.py — Stage 4.3 DiD on per-head OUTPUT contribution.

Identical setup to Stage 4.2 — same 37 paired clips, same `generated_caption`
teacher-force, same hal/non token-position assignment, same head sets
(τ_99 Audio-only n=12 + layer-matched Inert), same DiD machinery. The
ONLY change is the per-(clip, head, query_pos) quantity measured.

Quantity — per-head output contribution to the residual stream:
  Let z_h(t) = head h's attention output at query t BEFORE o_proj mixing
              (one slice of the o_proj input).
  W_o_h     = the h-th head's block of self_attn.o_proj.weight (head_dim → hidden)
  c_h(t)    = W_o_h · z_h(t)   ∈ R^hidden
  PRIMARY READOUT      : ||c_h(t)||_2
  SECONDARY READOUT    : cosine(c_h(t), o_proj_out(t))     where o_proj_out(t) = Σ_h c_h(t)
                          (cheap by-product of the same hook; how aligned this head's
                          write is with the consensus self-attn write at this token)

Captured via `register_forward_pre_hook` on `self_attn.o_proj` — the input
tensor is exactly the concatenated per-head outputs (B, q, H*hd). We
reshape to (B, q, H, hd), restrict to caption query positions, multiply
by per-head W_o blocks, take norm + cosine.

Output:
  stage4_3_output_records.csv      per (clip, layer, head, token_type, position, norm, cos_to_total)
  stage4_3_output_did.csv          raw + position-controlled DiD per readout
  stage4_3_dose_response.csv       per-head [hal − non] gap by |score_A| rank
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402

DEFAULT_QA   = _REPO / "results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json"
DEFAULT_VID  = _REPO / "data/VGGSounder/videos"
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_3"


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
    """Parameterized over τ percentile; column name `head_class_t99` is kept
    as a stable alias so downstream code stays unchanged."""
    h = pd.read_csv(heads_csv)
    pool = np.abs(np.concatenate([h["score_A"].values, h["score_V"].values,
                                    h["score_AV"].values]))
    tau_val = float(np.percentile(pool, tau_percentile))
    col = f"head_class_t{tau_percentile}"
    h[col] = [
        categorize(h["score_A"].iloc[i] > tau_val,
                   h["score_V"].iloc[i] > tau_val,
                   h["score_AV"].iloc[i] > tau_val)
        for i in range(len(h))
    ]
    h["head_class_t99"] = h[col]
    ao = (h[h[col] == "Audio-only"]
            [["layer", "head", "score_A"]]
            .sort_values("score_A", ascending=False)
            .reset_index(drop=True))
    ao["score_A_rank"] = np.arange(1, len(ao) + 1)
    return h, ao, tau_val


def assign_token_types(caption_ids, hal_tokens, non_tokens):
    """Consume-in-order — same convention as 4.2."""
    hal_left = Counter(hal_tokens)
    non_left = Counter(non_tokens)
    hal_positions, non_positions = [], []
    for pos, tid in enumerate(caption_ids):
        in_hal = hal_left[tid] > 0
        in_non = non_left[tid] > 0
        if in_hal:
            hal_positions.append(pos); hal_left[tid] -= 1
        elif in_non:
            non_positions.append(pos); non_left[tid] -= 1
    return hal_positions, non_positions


def process_clip(d, model, processor, layers, n_layers, ao_layers_set,
                  out_records):
    """Forward + capture per-head output norm + cosine to o_proj total
    at caption query positions split by hal/non. Returns
    (n_hal, n_non, err)."""
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists():
        return 0, 0, "missing_video"

    conv = build_conversation(str(video_path), d["question"], "av")
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "av",
                                           model.device, model.dtype)
    except Exception as e:
        return 0, 0, f"prep:{type(e).__name__}"

    prompt_S = int(inputs["input_ids"].shape[1])
    caption_ids = processor.tokenizer(d["generated_caption"],
                                        add_special_tokens=False).input_ids
    if not caption_ids:
        return 0, 0, "empty_caption"

    hal_pos, non_pos = assign_token_types(
        caption_ids,
        d.get("hallucinated_tokens", []),
        d.get("non_hallucinated_tokens", []))
    if not hal_pos or not non_pos:
        return 0, 0, "no_both_types"

    # Append caption ids to input_ids
    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Absolute caption query positions
    hal_q_abs = [prompt_S + p for p in hal_pos]
    non_q_abs = [prompt_S + p for p in non_pos]
    # Combined list with (abs_pos, position_in_caption, token_type)
    q_records = ([(q, r, "hal") for q, r in zip(hal_q_abs, hal_pos)]
                 + [(q, r, "non") for q, r in zip(non_q_abs, non_pos)])
    abs_positions = torch.tensor([q for q, _, _ in q_records], dtype=torch.long)
    pos_in_cap    = [r for _, r, _ in q_records]
    token_types   = [t for _, _, t in q_records]

    # Install pre-hook on each self_attn.o_proj that ONLY fires on the
    # final forward (full sequence). We only act at layers in ao_layers_set.
    def make_hook(L_idx, layer):
        sa = layer.self_attn
        num_heads = sa.num_heads
        head_dim  = sa.head_dim
        W_o = sa.o_proj.weight        # (hidden, H*hd)
        hidden = W_o.shape[0]

        def _h(_m, args):
            if L_idx not in ao_layers_set:
                return None         # don't modify args, still consume
            x = args[0]              # (B, q, H*hd) — concat per-head outputs
            if x.shape[0] != 1 or x.shape[1] < (prompt_S + len(caption_ids)):
                return None
            # Subset to caption query positions
            qs = abs_positions.to(x.device)
            z = x[0, qs, :]                                       # (n_q, H*hd)
            z = z.reshape(z.shape[0], num_heads, head_dim).float() # (n_q, H, hd)
            # Per-head contribution c_h(t) = W_o_h · z_h(t)
            # W_o reshaped: (hidden, H, hd)
            W_o_r = W_o.reshape(hidden, num_heads, head_dim).float()
            # einsum: c[q, h, o] = sum_d z[q, h, d] * W_o_r[o, h, d]
            c = torch.einsum('qhd,ohd->qho', z, W_o_r)            # (n_q, H, hidden)
            # primary: ||c_h(t)||
            c_norm = c.norm(dim=-1)                                # (n_q, H)
            # o_proj total output: sum over heads
            c_total = c.sum(dim=1)                                 # (n_q, hidden)
            c_total_norm = c_total.norm(dim=-1).clamp(min=1e-12)   # (n_q,)
            # cosine of c_h(t) with c_total(t)
            cos = (c * c_total.unsqueeze(1)).sum(dim=-1) \
                  / (c_norm.clamp(min=1e-12) * c_total_norm.unsqueeze(-1))
            c_norm_np = c_norm.cpu().numpy()
            cos_np    = cos.cpu().numpy()
            for q_idx in range(c_norm_np.shape[0]):
                rank_in_cap = pos_in_cap[q_idx]
                ttype = token_types[q_idx]
                for h_idx in range(num_heads):
                    out_records.append((
                        d["video"], int(L_idx), int(h_idx),
                        ttype, int(rank_in_cap),
                        float(c_norm_np[q_idx, h_idx]),
                        float(cos_np[q_idx, h_idx])))
            return None
        return _h

    handles = []
    for L in range(n_layers):
        if L in ao_layers_set:
            handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(
                make_hook(L, layers[L])))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False, output_hidden_states=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return 0, 0, f"fwd:{type(e).__name__}"
    finally:
        for h in handles: h.remove()
    torch.cuda.empty_cache()
    return len(hal_pos), len(non_pos), None


def cluster_robust_ols(X, y, groups):
    """Liang-Zeger cluster-robust SE. Returns β, SE, p, CI."""
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
        Xg_eg = Xg.T @ resid[idx]
        meat += np.outer(Xg_eg, Xg_eg)
        G += 1
    cov_cl = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(cov_cl))
    z = beta / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return beta, se, beta - 1.96 * se, beta + 1.96 * se, p


def aggregate_did(df: pd.DataFrame, ao_df: pd.DataFrame, heads_df: pd.DataFrame,
                    out_dir: Path):
    """For each readout (norm, cos), compute the DiD per (clip), pooled."""
    inert = heads_df[heads_df["head_class_t99"] == "Inert"]
    ao_set = set(zip(ao_df["layer"], ao_df["head"]))
    inert_set = set(zip(inert["layer"], inert["head"]))
    ao_layers = set(ao_df["layer"].tolist())

    def _hc(r):
        k = (r["layer"], r["head"])
        if k in ao_set:    return "Audio-only"
        if k in inert_set: return "Inert"
        return "other"
    df["head_class"] = df.apply(_hc, axis=1)
    df = df[df["head_class"].isin(["Audio-only", "Inert"])]
    df = df[df["layer"].isin(ao_layers)]

    READOUTS = ["norm", "cos_to_total"]
    raw_rows = []
    per_clip_rows = []
    pc_rows = []
    dose_rows = []

    for readout in READOUTS:
        # per (clip, head_class, layer, head, token_type) mean across queries
        per_head = (df.groupby(["clip", "head_class", "layer", "head", "token_type"],
                                 as_index=False)
                       .agg(mean_val=(readout, "mean")))
        wide = per_head.pivot_table(
            index=["clip", "head_class", "layer", "head"],
            columns="token_type", values="mean_val").reset_index()
        wide["hal_minus_non"] = wide.get("hal", np.nan) - wide.get("non", np.nan)

        ao_part = wide[wide["head_class"] == "Audio-only"]
        inert_part = wide[wide["head_class"] == "Inert"]
        g_ao = (ao_part.groupby(["clip"], as_index=False)
                       .agg(gap_H=("hal_minus_non", "mean"),
                            mean_hal_H=("hal", "mean"),
                            mean_non_H=("non", "mean")))
        g_in = (inert_part.groupby(["clip"], as_index=False)
                          .agg(gap_I=("hal_minus_non", "mean"),
                               mean_hal_I=("hal", "mean"),
                               mean_non_I=("non", "mean")))
        by_clip = g_ao.merge(g_in, on="clip", how="inner")
        by_clip["DiD"] = by_clip["gap_H"] - by_clip["gap_I"]
        by_clip["readout"] = readout
        per_clip_rows.append(by_clip)

        # Raw DiD summary
        x = by_clip["DiD"].dropna().values
        n = len(x)
        m = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(max(n, 1)))
        from scipy.stats import t as _t
        p = float(2 * _t.sf(abs(m / se if se > 0 else 0.0), df=n - 1)) if n > 1 else float("nan")
        raw_rows.append(dict(
            readout=readout, n_clips=n,
            mean_gap_H=float(by_clip["gap_H"].mean()),
            mean_gap_I=float(by_clip["gap_I"].mean()),
            mean_DiD=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se,
            pval_two_sided=p,
            excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0))))

        # Position-controlled DiD (cluster-robust OLS, clusters = clip)
        sub = df.copy()
        sub["is_hal"] = (sub["token_type"] == "hal").astype(int)
        sub["is_ao"]  = (sub["head_class"] == "Audio-only").astype(int)
        X_ = np.column_stack([
            np.ones(len(sub)),
            sub["token_position_in_caption"].values,
            sub["is_hal"].values,
            sub["is_ao"].values,
            (sub["is_hal"].values * sub["is_ao"].values),
        ])
        y_ = sub[readout].values
        beta, se_, lo, hi, pv = cluster_robust_ols(X_, y_, sub["clip"].values)
        pc_rows.append(dict(
            readout=readout, n_obs=len(sub),
            interaction_coef_DiD=float(beta[4]),
            cluster_se=float(se_[4]),
            ci95_low=float(lo[4]), ci95_high=float(hi[4]),
            pval_two_sided=float(pv[4]),
            excludes_zero=bool((lo[4] > 0) or (hi[4] < 0)),
            position_coef=float(beta[1]),
            position_pval=float(pv[1])))

        # Dose-response by |score_A| rank
        ao_part2 = ao_part.merge(
            ao_df[["layer", "head", "score_A", "score_A_rank"]],
            on=["layer", "head"], how="inner")
        per_head = (ao_part2.groupby(["layer", "head", "score_A_rank"],
                                       as_index=False)
                            .agg(mean_hal_minus_non=("hal_minus_non", "mean"),
                                 n_clips=("clip", "nunique")))
        per_head["readout"] = readout
        dose_rows.append(per_head)

    raw_df = pd.DataFrame(raw_rows)
    pc_df  = pd.DataFrame(pc_rows)
    per_clip_df = pd.concat(per_clip_rows, ignore_index=True)
    dose_df = pd.concat(dose_rows, ignore_index=True)

    raw_path = out_dir / "stage4_3_output_did.csv"
    raw_df.to_csv(raw_path, index=False)
    pc_path = out_dir / "stage4_3_output_did_position_controlled.csv"
    pc_df.to_csv(pc_path, index=False)
    pcl_path = out_dir / "stage4_3_output_did_per_clip.csv"
    per_clip_df.to_csv(pcl_path, index=False)
    dose_path = out_dir / "stage4_3_dose_response.csv"
    dose_df.to_csv(dose_path, index=False)
    print(f"wrote {raw_path}\n  {raw_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}')}")
    print(f"\nwrote {pc_path}\n  {pc_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}')}")
    print(f"wrote {pcl_path}  ({len(per_clip_df)} rows)")
    print(f"wrote {dose_path}  ({len(dose_df)} rows)")

    # Dose-response Spearman
    print("\n=== dose-response (per-head hal−non gap vs |score_A| rank, Spearman ρ) ===")
    from scipy.stats import spearmanr
    for ro, g in dose_df.groupby("readout"):
        rho, p = spearmanr(g["score_A_rank"].values,
                            g["mean_hal_minus_non"].values)
        print(f"  {ro:<14s}  ρ={rho:+.3f}  p={p:.3f}  (n={len(g)} heads)")


def main(args):
    out_dir = Path(args.output_dir)
    if args.output_subdir:
        out_dir = out_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)

    heads_df, ao_df, tau_val = load_head_classes(Path(args.heads_csv),
                                                    tau_percentile=args.tau_percentile)
    ao_layers_set = set(ao_df["layer"].tolist())
    print(f"τ_{args.tau_percentile} = {tau_val:.6g}; Audio-only n={len(ao_df)}; "
          f"layers covered = {sorted(ao_layers_set)}")

    data = json.load(open(args.sampled_entities_json))
    paired = [d for d in data
              if d.get("hallucinated_tokens") and d.get("non_hallucinated_tokens")]
    print(f"\n{len(data)} total clips; {len(paired)} have BOTH hal and non-hal tokens")

    records = []
    n_processed = 0
    failures = {}
    for d in tqdm(paired, desc="clips"):
        n_h, n_n, err = process_clip(d, model, processor, layers, n_layers,
                                       ao_layers_set, records)
        if err is not None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
            continue
        n_processed += 1
    if failures:
        print(f"failures: {failures}")
    print(f"\nprocessed {n_processed} clips, collected {len(records):,} records")

    df = pd.DataFrame(records, columns=[
        "clip", "layer", "head", "token_type", "token_position_in_caption",
        "norm", "cos_to_total"])
    rec_path = out_dir / "stage4_3_output_records.csv"
    df.to_csv(rec_path, index=False)
    print(f"wrote {rec_path}  ({len(df):,} rows)")

    aggregate_did(df, ao_df, heads_df, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--output_subdir", default="",
                   help="If set, write to {output_dir}/{output_subdir}/ "
                        "(used to keep multiple τ runs side-by-side).")
    p.add_argument("--tau_percentile", type=int, default=99, choices=[90, 95, 99],
                   help="Percentile of pooled |scores| for head-class assignment.")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
