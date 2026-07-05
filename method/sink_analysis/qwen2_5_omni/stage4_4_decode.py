"""
stage4_4_decode.py — Stage 4.4 logit-lens decode of per-head output c_h(t).

For each (clip, layer, head h, query position t in the caption):
  ℓ_h(t) = lm_head(final_RMSNorm(c_h(t)))   ∈  R^V
where c_h(t) is the head's o_proj contribution (the same vector whose
norm was measured in Stage 4.3).

PRIMARY readout — target-token push:
  target_logit = ℓ_h(t)[ tok(t) ]
where tok(t) is the token actually at position t in the teacher-forced
caption. Does the head's write decode to the token sitting at the same
position? Tested per spec.

SECONDARY (interpretability) — for the L27H12 + L27H13 heads only, the
top-10 vocab tokens of ℓ_h(t) are dumped at hal and non query positions
for the qualitative table.

Reuse from 4.3 (no change): 37 paired VGGSounder clips, same teacher-
force, same hal vs non token-position assignment. The only new compute is
the lm_head matmul on c_h(t).

Outputs (`stage4_4/`):
  stage4_4_target_logit_records.csv   per (clip, layer, head, ttype, qpos,
                                       target_token_id, target_logit)
  stage4_4_top10_L27_primary.csv      per (clip, head, ttype, qpos, rank,
                                       vocab_id, decoded, logit)

Honesty: logit lens is exact ONLY at the last layer (L27). For non-L27
heads it is approximate (mid-stack lens noise; Stage 3.3 lesson). The
primary heads of interest (L27H12, L27H13) live AT L27, so for them
this is near-exact; the SECONDARY τ_95 class result is flagged as
approximate.
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
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_4"

PRIMARY_HEADS = {(27, 12), (27, 13)}        # L27H12, L27H13


def categorize(in_A, in_V, in_AV):
    if in_A and in_V and in_AV: return "Generic"
    if in_A and in_V:            return "Compensated"
    if in_A and in_AV:           return "Audio + AV"
    if in_V and in_AV:           return "Visual + AV"
    if in_A:                     return "Audio-only"
    if in_V:                     return "Visual-only"
    if in_AV:                    return "Cross-modal-only"
    return "Inert"


def assign_token_types(caption_ids, hal_tokens, non_tokens):
    hal_left = Counter(hal_tokens)
    non_left = Counter(non_tokens)
    hal_positions, non_positions = [], []
    for pos, tid in enumerate(caption_ids):
        if hal_left[tid] > 0:
            hal_positions.append(pos); hal_left[tid] -= 1
        elif non_left[tid] > 0:
            non_positions.append(pos); non_left[tid] -= 1
    return hal_positions, non_positions


def process_clip(d, model, processor, layers, ao_layers, final_norm, lm_head,
                  out_records, out_top10, top_k=10):
    """Forward + capture target-token logits and top-K decode (primary).
    Returns (n_hal, n_non, err)."""
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

    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Per spec: ℓ_h(t)[tok(t)] uses the token sitting at the same position t.
    # Token at caption position p is caption_ids[p].
    hal_q_abs = [prompt_S + p for p in hal_pos]
    non_q_abs = [prompt_S + p for p in non_pos]
    q_records = ([(q, r, "hal", caption_ids[r]) for q, r in zip(hal_q_abs, hal_pos)]
                 + [(q, r, "non", caption_ids[r]) for q, r in zip(non_q_abs, non_pos)])
    abs_positions = torch.tensor([q for q, _, _, _ in q_records], dtype=torch.long)
    rank_in_cap   = [r   for _, r, _, _ in q_records]
    token_types   = [tt  for _, _, tt, _ in q_records]
    target_tokens = [tid for _, _, _, tid in q_records]

    def make_hook(L_idx, layer):
        sa = layer.self_attn
        num_heads = sa.num_heads
        head_dim  = sa.head_dim
        W_o = sa.o_proj.weight                              # (hidden, H*hd)
        hidden = W_o.shape[0]

        def _h(_m, args):
            if L_idx not in ao_layers:
                return None
            x = args[0]                                       # (B, q, H*hd)
            full_S_expected = prompt_S + len(caption_ids)
            if x.shape[0] != 1 or x.shape[1] < full_S_expected:
                return None
            qs = abs_positions.to(x.device)
            z = x[0, qs, :]                                   # (n_q, H*hd)
            n_q = z.shape[0]
            z = z.reshape(n_q, num_heads, head_dim).float()   # (n_q, H, hd)
            W_o_r = W_o.reshape(hidden, num_heads, head_dim).float()
            c = torch.einsum('qhd,ohd->qho', z, W_o_r)        # (n_q, H, hidden)

            # Move to lm_head device + cast to its dtype (avoid bf16↔fp32 mismatch)
            dev_lm = lm_head.weight.device
            dt_lm  = lm_head.weight.dtype
            c2 = c.to(device=dev_lm, dtype=dt_lm)
            # final_norm expects (..., hidden); reshape to (n_q*H, hidden)
            c_flat = c2.reshape(-1, hidden)
            c_norm_flat = final_norm(c_flat)
            logits = lm_head(c_norm_flat).float()             # (n_q*H, V)
            V = logits.shape[-1]
            logits = logits.reshape(n_q, num_heads, V)        # (n_q, H, V)

            # target-token logit per (q, head)
            t_ids_t = torch.tensor(target_tokens, device=logits.device,
                                    dtype=torch.long).unsqueeze(1).expand(-1, num_heads)
            target_logit = logits.gather(2, t_ids_t.unsqueeze(-1)).squeeze(-1)
            target_logit_np = target_logit.cpu().numpy()

            # Records for all heads
            for q_i in range(n_q):
                for h_i in range(num_heads):
                    out_records.append((
                        d["video"], int(L_idx), int(h_i),
                        token_types[q_i], int(rank_in_cap[q_i]),
                        int(target_tokens[q_i]),
                        float(target_logit_np[q_i, h_i])))

            # Top-K dump for primary heads only
            for h_i in range(num_heads):
                if (L_idx, h_i) not in PRIMARY_HEADS:
                    continue
                top_vals, top_idx = torch.topk(logits[:, h_i, :], top_k, dim=-1)
                tv = top_vals.cpu().numpy()
                ti = top_idx.cpu().numpy()
                for q_i in range(n_q):
                    for k in range(top_k):
                        out_top10.append((
                            d["video"], int(L_idx), int(h_i),
                            token_types[q_i], int(rank_in_cap[q_i]),
                            int(target_tokens[q_i]),
                            int(k), int(ti[q_i, k]), float(tv[q_i, k])))
            return None
        return _h

    handles = []
    for L in range(len(layers)):
        if L in ao_layers:
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


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)
    final_norm = model.thinker.model.norm
    lm_head    = model.thinker.lm_head
    print(f"  n_layers={n_layers}; final_norm.dev={final_norm.weight.device}; "
          f"lm_head.dev={lm_head.weight.device}; V={lm_head.weight.shape[0]}")

    # Hook at the τ_95 Audio-only layer set (superset of τ_99 set).
    heads_df = pd.read_csv(args.heads_csv)
    pool = np.abs(np.concatenate([heads_df["score_A"].values,
                                    heads_df["score_V"].values,
                                    heads_df["score_AV"].values]))
    tau_95 = float(np.percentile(pool, 95))
    ao95 = heads_df[(heads_df["score_A"] > tau_95)
                     & (heads_df["score_V"] <= tau_95)
                     & (heads_df["score_AV"] <= tau_95)]
    ao_layers = set(ao95["layer"].tolist())
    print(f"τ_95 = {tau_95:.6g}; Audio-only n={len(ao95)}; "
          f"layers covered = {sorted(ao_layers)} ({len(ao_layers)} layers)")

    data = json.load(open(args.sampled_entities_json))
    paired = [d for d in data
              if d.get("hallucinated_tokens") and d.get("non_hallucinated_tokens")]
    print(f"\n{len(data)} total clips; {len(paired)} have BOTH hal and non-hal tokens")

    records = []
    top10   = []
    n_processed = 0
    failures = {}
    for d in tqdm(paired, desc="clips"):
        n_h, n_n, err = process_clip(d, model, processor, layers, ao_layers,
                                       final_norm, lm_head, records, top10,
                                       top_k=10)
        if err is not None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
            continue
        n_processed += 1
    if failures:
        print(f"failures: {failures}")
    print(f"\nprocessed {n_processed} clips, "
          f"target-logit records: {len(records):,}, top10 rows: {len(top10):,}")

    df = pd.DataFrame(records, columns=[
        "clip", "layer", "head", "token_type", "token_position_in_caption",
        "target_token_id", "target_logit"])
    rec_path = out_dir / "stage4_4_target_logit_records.csv"
    df.to_csv(rec_path, index=False)
    print(f"wrote {rec_path}  ({len(df):,} rows)")

    df10 = pd.DataFrame(top10, columns=[
        "clip", "layer", "head", "token_type", "token_position_in_caption",
        "target_token_id", "rank", "vocab_id", "logit"])
    # decode vocab_id
    if len(df10):
        ids = df10["vocab_id"].unique().tolist()
        tok = processor.tokenizer
        decoded_map = {i: tok.decode([int(i)], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
                        for i in ids}
        df10["decoded"] = df10["vocab_id"].map(decoded_map)
    top_path = out_dir / "stage4_4_top10_L27_primary.csv"
    df10.to_csv(top_path, index=False)
    print(f"wrote {top_path}  ({len(df10):,} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
