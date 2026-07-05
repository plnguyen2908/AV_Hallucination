"""
stage4_2_did.py — Stage 4.2 DiD test.

Hypothesis (pre-registered, directional):
  On AV input, the τ_99 Audio-only "audio-hallucination" heads (n=12,
  AudioSet-defined) attend LESS to audio content and MORE to BOS at
  HALLUCINATED generated-caption token positions than at NON-hallucinated
  positions — MORE THAN layer-matched Inert heads do (DiD).

Test:
  per clip:
    gap_H(audio) = mean over τ_99-audio heads of [hal − non]
    gap_I(audio) = mean over layer-matched Inert heads of [hal − non]
    DiD(audio)   = gap_H − gap_I
  CONFIRMED iff audio DiD < 0 (and bos DiD > 0), paired across clips, CI excludes 0.

Forced decode: build the constrained-captioning prompt (from
`sampled_entities.json::question`) + the model's own
`generated_caption` as the assistant continuation. Forward pass with
attention captured; query positions = caption-token positions, split
by id-membership in `hallucinated_tokens` / `non_hallucinated_tokens`
(consume-in-order to handle duplicate token ids).

Bins per key position:
  audio_content : input_ids == audio_token_index (151646)
  bos           : position 0 (the `<|im_start|>` token)
  video_content : input_ids == video_token_index (151656)
  text_sys      : everything else

Per (clip, layer, head, query_pos, bin) record:
  per_token_attention = (sum over k in bin of A_h[q,k]) / n_tokens(bin)

Output:
  stage4_2_entity_attention.csv  — long form, one row per record
  stage4_2_did.csv               — per-bin DiD (raw + position-controlled +
                                   dose-response by |score_A| rank)

Reads:
  results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json
  results/qwen2_5_omni/categorize_exp/heads.csv
  data/VGGSounder/videos/*.mp4
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
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_2"

AUDIO_TOKEN_ID = 151646
VIDEO_TOKEN_ID = 151656


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
    """Return (heads_df with head_class_t{p} column, ao_df sorted by |score_A|,
    tau_val) for the requested percentile. The head_class column name carries
    the percentile so downstream code can pivot consistently."""
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
    # Keep a stable alias for the downstream code that expects a fixed name
    h["head_class_t99"] = h[col]
    ao = (h[h[col] == "Audio-only"]
            [["layer", "head", "score_A"]]
            .sort_values("score_A", ascending=False)
            .reset_index(drop=True))
    ao["score_A_rank"] = np.arange(1, len(ao) + 1)
    return h, ao, tau_val


def assign_token_types(caption_ids, hal_tokens, non_tokens):
    """Returns parallel lists (hal_positions, non_positions, ambig_count).
    Consume-in-order to handle duplicate ids: a position with id `t` whose
    `t` appears in both hal & non gets assigned to hal first (until the
    hal count for t is exhausted), then non; pure-other tokens are
    skipped (they were neither labeled as grounded nor as hallucinated)."""
    hal_left = Counter(hal_tokens)
    non_left = Counter(non_tokens)
    hal_positions, non_positions = [], []
    ambig = 0
    for pos, tid in enumerate(caption_ids):
        in_hal = hal_left[tid] > 0
        in_non = non_left[tid] > 0
        if in_hal and in_non:
            ambig += 1
            hal_positions.append(pos)
            hal_left[tid] -= 1
        elif in_hal:
            hal_positions.append(pos)
            hal_left[tid] -= 1
        elif in_non:
            non_positions.append(pos)
            non_left[tid] -= 1
    return hal_positions, non_positions, ambig


def build_bin_masks(input_ids_1d_np, S: int):
    """Return dict {bin: bool mask of length S, n_tokens}.
    Bins: audio_content, video_content, bos, text_sys."""
    is_audio = (input_ids_1d_np == AUDIO_TOKEN_ID)
    is_video = (input_ids_1d_np == VIDEO_TOKEN_ID)
    bos = np.zeros(S, dtype=bool); bos[0] = True
    text_sys = ~(is_audio | is_video | bos)
    return {
        "audio_content": is_audio,
        "video_content": is_video,
        "bos":           bos,
        "text_sys":      text_sys,
    }


def process_clip(d, model, processor, layers, n_layers, ao_layers_set,
                  out_records):
    """Forward + capture; append records to `out_records`. Returns
    n_hal, n_non assigned for this clip (0,0 if skipped)."""
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

    # Tokenize the model's own generated_caption (no special tokens)
    caption_ids = processor.tokenizer(d["generated_caption"],
                                        add_special_tokens=False).input_ids
    if not caption_ids:
        return 0, 0, "empty_caption"

    hal_pos_in_cap, non_pos_in_cap, n_ambig = assign_token_types(
        caption_ids,
        d.get("hallucinated_tokens", []),
        d.get("non_hallucinated_tokens", []))
    if not hal_pos_in_cap or not non_pos_in_cap:
        return 0, 0, "no_both_types"

    # Append caption ids to input_ids (teacher-forced)
    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    full_S = int(inputs["input_ids"].shape[1])
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Absolute query positions (in full sequence)
    hal_q_abs = [prompt_S + p for p in hal_pos_in_cap]
    non_q_abs = [prompt_S + p for p in non_pos_in_cap]
    # token_position_in_caption (rank starting at 0)
    hal_q_rank = list(hal_pos_in_cap)
    non_q_rank = list(non_pos_in_cap)

    # Bin masks
    ids_np = inputs["input_ids"][0].cpu().numpy()
    bins = build_bin_masks(ids_np, full_S)
    bin_masks_t = {name: (torch.from_numpy(mask), int(mask.sum()))
                    for name, mask in bins.items()}

    # In-hook: only fire at layers Audio-only heads occupy
    def make_hook(L_idx):
        def _h(_m, _i, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            if L_idx not in ao_layers_set:
                # still null out the attention tensor to save memory
                return (out[0], None) + tuple(out[2:])
            aw = out[1]                              # (1, H, q, kv)
            a = aw[0].float()                        # (H, q, k)
            # subset query rows
            for q_abs, q_rank, ttype in (
                [(q, r, "hal") for q, r in zip(hal_q_abs, hal_q_rank)]
                + [(q, r, "non") for q, r in zip(non_q_abs, non_q_rank)]
            ):
                if q_abs >= a.shape[1]:
                    continue
                row = a[:, q_abs, :]                 # (H, k)
                for bname, (mask_cpu, n_tok) in bin_masks_t.items():
                    if n_tok == 0:
                        for h_idx in range(row.shape[0]):
                            out_records.append((
                                d["video"], int(L_idx), int(h_idx),
                                ttype, int(q_rank), bname, 0, 0.0, 0.0))
                        continue
                    mask = mask_cpu.to(row.device)
                    s = row[:, mask].sum(dim=-1).cpu().numpy()
                    per_tok = s / n_tok
                    for h_idx in range(row.shape[0]):
                        out_records.append((
                            d["video"], int(L_idx), int(h_idx),
                            ttype, int(q_rank), bname,
                            int(n_tok), float(s[h_idx]), float(per_tok[h_idx])))
            return (out[0], None) + tuple(out[2:])
        return _h

    handles = [layers[L].self_attn.register_forward_hook(make_hook(L))
               for L in range(n_layers)]
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=True, return_dict=True,
                          use_cache=False, output_hidden_states=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return 0, 0, f"fwd:{type(e).__name__}"
    finally:
        for h in handles: h.remove()
    torch.cuda.empty_cache()
    return len(hal_pos_in_cap), len(non_pos_in_cap), None


def aggregate_did(records_df: pd.DataFrame, ao_df: pd.DataFrame,
                    heads_df: pd.DataFrame, out_dir: Path):
    """Compute DiD per (clip, bin):
       gap_H = mean over Audio-only heads of (mean_per_token at hal − at non) per (clip)
       gap_I = mean over layer-matched Inert heads of same
       DiD   = gap_H − gap_I
    Also: position-controlled DiD (regression of per_token ~ position +
    token_type + head_class with interaction); dose-response by |score_A|
    rank."""
    # Merge head_class onto records_df
    inert_at_layers = heads_df[heads_df["head_class_t99"] == "Inert"]
    ao_set = set(zip(ao_df["layer"], ao_df["head"]))
    inert_set = set(zip(inert_at_layers["layer"], inert_at_layers["head"]))

    def _class(row):
        k = (row["layer"], row["head"])
        if k in ao_set:    return "Audio-only"
        if k in inert_set: return "Inert"
        return "other"
    records_df["head_class"] = records_df.apply(_class, axis=1)
    # Drop "other" (only Audio-only + Inert needed for DiD)
    records_df = records_df[records_df["head_class"].isin(["Audio-only", "Inert"])]

    # Per (clip, head_class, layer, head, token_type, bin): mean per_token over query positions
    per_head = (records_df.groupby(
        ["clip", "head_class", "layer", "head", "token_type", "bin"],
        as_index=False)
                .agg(mean_per_token=("per_token_inflow", "mean"),
                     mean_position=("token_position_in_caption", "mean"),
                     n_queries=("token_position_in_caption", "count")))

    # Pivot to hal vs non
    wide = per_head.pivot_table(
        index=["clip", "head_class", "layer", "head", "bin"],
        columns="token_type",
        values="mean_per_token").reset_index()
    wide["hal_minus_non"] = wide.get("hal", np.nan) - wide.get("non", np.nan)

    # Per (clip, head_class, bin): mean across heads; for Inert use heads in
    # the layers where Audio-only sits (layer-matched).
    ao_layers = set(ao_df["layer"].tolist())
    inert_lm = wide[(wide["head_class"] == "Inert")
                    & (wide["layer"].isin(ao_layers))]
    ao_part = wide[wide["head_class"] == "Audio-only"]

    def _mean_per_clip(df, label):
        g = (df.groupby(["clip", "bin"], as_index=False)
               .agg(mean_hal_minus_non=("hal_minus_non", "mean"),
                    n_heads=("head", "count")))
        g["head_class"] = label
        return g

    g_ao    = _mean_per_clip(ao_part,    "Audio-only")
    g_inert = _mean_per_clip(inert_lm,   "Inert (layer-matched)")
    by_clip = pd.concat([g_ao, g_inert], ignore_index=True)

    # DiD per clip per bin
    did = by_clip.pivot(index=["clip", "bin"], columns="head_class",
                          values="mean_hal_minus_non").reset_index()
    did["DiD"] = did["Audio-only"] - did["Inert (layer-matched)"]
    did = did.rename(columns={"Audio-only": "gap_H",
                                "Inert (layer-matched)": "gap_I"})
    did_path = out_dir / "stage4_2_did_per_clip.csv"
    did.to_csv(did_path, index=False)
    print(f"wrote {did_path}  ({len(did)} rows)")

    # Aggregate over clips
    summary_rows = []
    for bname, g in did.groupby("bin"):
        x = g["DiD"].dropna().values
        n = len(x)
        if n < 2:
            continue
        m = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(n))
        from scipy import stats as _stats
        t = m / se if se > 0 else 0.0
        p = float(2 * _stats.t.sf(abs(t), df=n - 1))
        ci_lo, ci_hi = m - 1.96 * se, m + 1.96 * se
        # raw gap_H / gap_I means too
        summary_rows.append(dict(
            bin=bname, n_clips=n,
            mean_gap_H=float(g["gap_H"].mean()),
            mean_gap_I=float(g["gap_I"].mean()),
            mean_DiD=m, ci95_low=ci_lo, ci95_high=ci_hi, pval_two_sided=p,
            excludes_zero=bool((ci_lo > 0) or (ci_hi < 0)),
        ))
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "stage4_2_did.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # ---- Position-controlled regression ----
    print("\n=== Position-controlled regression ===")
    pc_rows = []
    for bname in ("audio_content", "bos", "video_content", "text_sys"):
        sub = records_df[records_df["bin"] == bname].copy()
        sub["is_hal"] = (sub["token_type"] == "hal").astype(int)
        sub["is_ao"]  = (sub["head_class"] == "Audio-only").astype(int)
        # Linear model: per_token ~ position + is_hal + is_ao + is_hal:is_ao + (clip FE)
        # Use simple OLS with clip dummies; for speed, demean within clip.
        sub["per_token_dm"] = sub["per_token_inflow"] - sub.groupby("clip")["per_token_inflow"].transform("mean")
        # The interaction coefficient is the DiD controlling for position + clip FE
        import statsmodels.api as sm
        X = sm.add_constant(sub[["token_position_in_caption", "is_hal",
                                   "is_ao"]].assign(
            hal_x_ao=sub["is_hal"] * sub["is_ao"]))
        y = sub["per_token_inflow"].values
        # cluster SE by clip
        model_ = sm.OLS(y, X).fit(cov_type="cluster",
                                    cov_kwds={"groups": sub["clip"].values})
        coef = float(model_.params["hal_x_ao"])
        se   = float(model_.bse["hal_x_ao"])
        pval = float(model_.pvalues["hal_x_ao"])
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        pc_rows.append(dict(
            bin=bname, n_obs=int(len(sub)),
            interaction_coef_DiD=coef, cluster_se=se,
            ci95_low=ci_lo, ci95_high=ci_hi, pval=pval,
            excludes_zero=bool((ci_lo > 0) or (ci_hi < 0)),
        ))
    pc_df = pd.DataFrame(pc_rows)
    pc_path = out_dir / "stage4_2_did_position_controlled.csv"
    pc_df.to_csv(pc_path, index=False)
    print(pc_df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"wrote {pc_path}")

    # ---- Dose-response by |score_A| rank ----
    print("\n=== Dose-response across Audio-only heads (per-head hal−non, by rank) ===")
    ao_rank = ao_df[["layer", "head", "score_A", "score_A_rank"]]
    ao_part2 = ao_part.merge(ao_rank, on=["layer", "head"], how="inner")
    dose = (ao_part2.groupby(["score_A_rank", "layer", "head", "bin"], as_index=False)
                     .agg(mean_hal_minus_non=("hal_minus_non", "mean"),
                          n_clips=("clip", "nunique")))
    dose_path = out_dir / "stage4_2_dose_response.csv"
    dose.to_csv(dose_path, index=False)
    print(dose.head(30).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"wrote {dose_path}")

    return summary, pc_df, dose


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
    print(f"τ_{args.tau_percentile} = {tau_val:.6g}; Audio-only heads "
          f"(n={len(ao_df)}); layers covered = {sorted(ao_layers_set)}")

    data = json.load(open(args.sampled_entities_json))
    paired = [d for d in data
              if d.get("hallucinated_tokens") and d.get("non_hallucinated_tokens")]
    print(f"\n{len(data)} total clips; {len(paired)} have BOTH hal and non-hal tokens")

    records = []                                # huge — flush periodically?
    n_processed = 0
    failures = {}
    pos_hist_rows = []                          # for position-distribution check
    for d in tqdm(paired, desc="clips"):
        before = len(records)
        n_h, n_n, err = process_clip(d, model, processor, layers, n_layers,
                                       ao_layers_set, records)
        if err is not None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
            continue
        n_processed += 1
        # position-distribution recording
        for q in range(len(processor.tokenizer(d["generated_caption"],
                                                  add_special_tokens=False).input_ids)):
            pass     # placeholder; real records below
        # use what we already wrote in records to derive positions:
        new = records[before:]
        # collect first per-(clip, query) position+type
        seen = set()
        for r in new:
            key = (r[0], r[3], r[4])
            if key in seen: continue
            seen.add(key)
            pos_hist_rows.append(dict(clip=r[0], token_type=r[3],
                                        position_in_caption=r[4]))

    if failures:
        print(f"failures: {failures}")
    print(f"\nprocessed {n_processed} clips, collected {len(records)} records")

    df = pd.DataFrame(records, columns=[
        "clip", "layer", "head", "token_type", "token_position_in_caption",
        "bin", "n_tokens_in_bin", "total_inflow", "per_token_inflow"])
    rec_path = out_dir / "stage4_2_entity_attention.csv"
    df.to_csv(rec_path, index=False)
    print(f"wrote {rec_path}  ({len(df):,} rows)")

    # Position histogram csv
    ph = pd.DataFrame(pos_hist_rows)
    ph_path = out_dir / "stage4_2_position_distribution.csv"
    ph.to_csv(ph_path, index=False)
    print(f"wrote {ph_path}")

    # DiD
    aggregate_did(df, ao_df, heads_df, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--output_subdir", default="",
                   help="If set, write outputs to {output_dir}/{output_subdir}/ "
                        "(used to keep multiple τ runs side-by-side).")
    p.add_argument("--tau_percentile", type=int, default=99, choices=[90, 95, 99],
                   help="Percentile of pooled |scores| for the Audio-only / Inert "
                        "head-class assignment (heads.csv). Default 99 preserves "
                        "the original Stage 4.2 run.")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
