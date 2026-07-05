"""
stage3_3b_causal_probe.py — necessity-knockout probe of the P_prop sink
cells at L13 (and L20 control).

Pre-registered decision gate (decided BEFORE running):

  CROSS-IS-SPECIAL    iff  Δ(cross) > Δ(uni_audio)  AND  Δ(cross) > Δ(random),
                          both paired across clips with 95% CI excluding 0.
  SINKS-MATTER-NOT-SPECIAL iff Δ(cross) ≈ Δ(uni_*) > Δ(random).
  NULL                iff Δ(cross) ≈ Δ(random).

Δ is measured on the model's final-position next-token distribution:
  KL(clean ‖ intervened)        — primary, label-free
  Δlogp(argmax_clean)            — primary, label-free
  Δlogp(GT-first-content-token)  — secondary, label-dependent (noisier)

Two interventions:
  attn_knockout  PRIMARY  — modify the layer's attention_mask kwarg to
                            insert -inf at KEY positions in the target
                            set (zeros attention to them across all
                            queries / heads; mass redistributes).
  value_zero     SECONDARY at L13 only — zero the v_proj output rows at
                            target positions (attention can still route
                            there, but the value contribution is 0;
                            isolates "stored content" from "routing").

Conditions at each probe layer:
  1. P_prop ∩ cross-modal
  2. P_prop ∩ uni_audio
  3. P_prop ∩ uni_video
  4. random non-sink, span-matched to cross
  5. full P_prop set (ASD-style comparison)

Two matching modes:
  count_matched : K=5 random subsamples each to k = min(|cross|, |ua|, |uv|)
                  per clip. Mean Δ over draws.
  per_token     : full cell knockout, Δ / n_knockout (effect normalized by
                  number of tokens removed).

Reuses Stage 3.2 per-clip sink masks (stage3_2/per_clip_tokens/*.npz);
sinks/MDS not recomputed.

Outputs:
  causal_probe_by_condition.csv       per-condition per-layer means + CIs
  per_clip/<clip_stem>.npz            per-clip raw deltas (for paired tests)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402

PROMPT_AV = "Describe what you see and hear in detail."

DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_QA_JSON = _REPO / "data/VGGSounder/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_3b_causal"

CELL_CROSS, CELL_UNI_V, CELL_UNI_A = 0, +1, -1
PROBE_LAYERS = (13, 20)
K_DRAWS = 5                      # spec: K=5 count-matched subsamples


# ----------------------------------------------------------------------
# Intervention hooks
# ----------------------------------------------------------------------

def make_attn_knockout_pre_hook(positions: np.ndarray, S: int):
    """register_forward_pre_hook(with_kwargs=True) on a decoder layer.
    Mutates the attention_mask kwarg by adding -inf at the kv columns
    corresponding to `positions`, across all queries / heads / batch.
    No-op if attention_mask is None or prompt length doesn't match S."""
    pos_t = torch.as_tensor(positions, dtype=torch.long)

    def _hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None:
            return args, kwargs
        # eager-attention 4D mask: (B, 1, q_len, kv_len) added pre-softmax
        if am.shape[-1] != S:
            return args, kwargs
        new_am = am.clone()
        idx = pos_t.to(new_am.device)
        # broadcast: zero out a key column means attention to that key is 0.
        # We add a very large negative value (use the dtype's most-negative
        # representable so the softmax goes exactly to 0 after exp).
        neg = torch.finfo(new_am.dtype).min
        new_am[..., :, idx] = neg
        kwargs["attention_mask"] = new_am
        return args, kwargs
    return _hook


def make_value_zero_hook(positions: np.ndarray, S: int):
    """register_forward_hook on self_attn.v_proj.
    The v_proj output is (B, S, num_kv_heads * head_dim); zero the rows
    at `positions` so those positions contribute 0 value vectors. No-op
    if the sequence length doesn't match S (i.e., not the prompt forward).
    """
    pos_t = torch.as_tensor(positions, dtype=torch.long)

    def _hook(module, inp, out):
        if out.shape[1] != S:
            return out
        new_out = out.clone()
        idx = pos_t.to(new_out.device)
        new_out[:, idx, :] = 0
        return new_out
    return _hook


# ----------------------------------------------------------------------
# Forward + last-position logits
# ----------------------------------------------------------------------

def forward_last_logits(model, inputs, use_aiv):
    with torch.inference_mode():
        out = model.thinker(
            **inputs,
            use_audio_in_video=use_aiv,
            return_dict=True,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    logits = out.logits[0, -1, :].float()                # (V,)
    return logits


# ----------------------------------------------------------------------
# Cells / random control
# ----------------------------------------------------------------------

def collect_cells(p_prop, p_llm_L, mds_cell_L, video_pos, audio_pos):
    """Return dict of {name: positions array} for one layer L.
    full_prop  = all P_prop sinks
    cross_prop = P_prop AND mds_cell == 0
    uv_prop    = P_prop AND mds_cell == +1
    ua_prop    = P_prop AND mds_cell == -1
    Span-matched random non-sink pool (audio+video, non-sink at L) is
    derived in the caller from cross's span composition."""
    cell = {
        "cross_prop": np.where(p_prop & (mds_cell_L == CELL_CROSS))[0],
        "uv_prop":    np.where(p_prop & (mds_cell_L == CELL_UNI_V))[0],
        "ua_prop":    np.where(p_prop & (mds_cell_L == CELL_UNI_A))[0],
        "full_prop":  np.where(p_prop)[0],
    }
    return cell


def span_matched_random(cross_positions, video_pos, audio_pos,
                         non_sink_mask, rng):
    """Draw a span-matched non-sink random control of |cross| tokens
    with cross's audio-span / video-span composition. Returns positions
    array. If a span pool is too small, draws with replacement and flags
    via an integer return value."""
    v_set = set(video_pos.tolist())
    a_set = set(audio_pos.tolist())
    n_v = int(sum(1 for p in cross_positions if p in v_set))
    n_a = int(sum(1 for p in cross_positions if p in a_set))
    ns_v = np.array([p for p in video_pos if non_sink_mask[p]], dtype=np.int64)
    ns_a = np.array([p for p in audio_pos if non_sink_mask[p]], dtype=np.int64)
    pieces = []
    if n_v > 0:
        replace = len(ns_v) < n_v
        pieces.append(rng.choice(ns_v, size=n_v, replace=replace) if len(ns_v) else np.array([], dtype=np.int64))
    if n_a > 0:
        replace = len(ns_a) < n_a
        pieces.append(rng.choice(ns_a, size=n_a, replace=replace) if len(ns_a) else np.array([], dtype=np.int64))
    if not pieces:
        return np.array([], dtype=np.int64)
    return np.concatenate(pieces)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def deltas(logits_clean: torch.Tensor, logits_int: torch.Tensor,
            top1_clean: int, gt_token_id: int | None):
    """Returns (kl, dlogp_argmax, dlogp_gt). gt is NaN if no GT id."""
    logp_c = torch.log_softmax(logits_clean, dim=-1)
    logp_i = torch.log_softmax(logits_int,   dim=-1)
    probs_c = torch.exp(logp_c)
    kl = float((probs_c * (logp_c - logp_i)).sum().item())
    dlp_a = float((logp_c[top1_clean] - logp_i[top1_clean]).item())
    dlp_g = (float((logp_c[gt_token_id] - logp_i[gt_token_id]).item())
             if gt_token_id is not None else float("nan"))
    return kl, dlp_a, dlp_g


# ----------------------------------------------------------------------
# GT token id helper
# ----------------------------------------------------------------------

_WORD_RE = __import__("re").compile(r"[A-Za-z]{3,}")
_LABEL_STOP = {"playing","sound","sounds","noise","noises","music","audio",
               "a","an","the","of","with","and","or","in","on","at"}


def gt_token_id_for_clip(tokenizer, label_list):
    """Pick the first sub-token id of the first non-stop content word from
    the clip's GT label list. Returns None if no usable label."""
    for lab in label_list or []:
        for w in _WORD_RE.findall(str(lab).lower()):
            if w in _LABEL_STOP:
                continue
            # try with leading space (more common in BPE)
            for cand in (f" {w}", w):
                ids = tokenizer.encode(cand, add_special_tokens=False)
                if ids:
                    return int(ids[0])
    return None


# ----------------------------------------------------------------------
# Per-clip probe
# ----------------------------------------------------------------------

def probe_clip(clip_path, dump_path, label_list, tokenizer,
                model, processor, layers, n_layers, K_draws=K_DRAWS,
                seed=0, per_clip_dir=None):
    dump = np.load(dump_path, allow_pickle=True)
    p_llm    = dump["p_llm"]              # (n_layers, S) bool
    p_prop   = dump["p_prop"]             # (S,) bool
    mds_cell = dump["mds_cell"]           # (n_layers, S) int8
    video_pos = dump["video_pos"]
    audio_pos = dump["audio_pos"]
    dump_S = int(dump["S"])

    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "av",
                                           model.device, model.dtype)
    except Exception:
        return None
    S = int(inputs["input_ids"].shape[1])
    if S != dump_S:
        return None

    # Clean forward
    logits_clean = forward_last_logits(model, inputs, use_aiv)
    top1_clean = int(logits_clean.argmax().item())
    gt_token_id = gt_token_id_for_clip(tokenizer, label_list)

    rng = np.random.default_rng(seed)
    records = []

    # ---- per probe layer ----
    for L in PROBE_LAYERS:
        cells = collect_cells(p_prop, p_llm[L], mds_cell[L],
                                video_pos, audio_pos)
        n_cross = len(cells["cross_prop"])
        n_uv    = len(cells["uv_prop"])
        n_ua    = len(cells["ua_prop"])
        if min(n_cross, n_uv, n_ua) == 0:
            continue
        k = int(min(n_cross, n_uv, n_ua))

        non_sink_mask = ~(p_prop | p_llm[L])
        rand_full = span_matched_random(cells["cross_prop"],
                                          video_pos, audio_pos,
                                          non_sink_mask, rng)
        if rand_full.size == 0:
            continue

        # Build the test-set dict
        full_sets = {
            "cross_prop": cells["cross_prop"],
            "uv_prop":    cells["uv_prop"],
            "ua_prop":    cells["ua_prop"],
            "random_span_matched": rand_full,
            "full_prop":  cells["full_prop"],
        }

        # Interventions at this layer
        intervention_specs = [("attn_knockout", "layer")]
        if L == 13:                                   # value_zero only at L13
            intervention_specs.append(("value_zero", "v_proj"))

        for inter_name, hook_target in intervention_specs:
            # ---- full-cell (per_token) measurements ----
            for cond_name, positions in full_sets.items():
                if positions.size == 0:
                    continue
                kl, dla, dlg = run_one_intervention(
                    model, inputs, use_aiv, logits_clean, top1_clean,
                    gt_token_id, layers[L], inter_name, positions, S)
                records.append(dict(
                    layer=L, intervention=inter_name, condition=cond_name,
                    matching="per_token", draw=0, n_knockout=int(positions.size),
                    kl=kl, dlogp_argmax=dla, dlogp_gt=dlg,
                ))

            # ---- count-matched K-draw measurements ----
            for cond_name, positions in full_sets.items():
                if cond_name == "full_prop":
                    continue   # full_prop has no subsampling; per_token only
                if positions.size == 0:
                    continue
                for d in range(K_draws):
                    if positions.size <= k:
                        sub = positions
                    else:
                        sub = rng.choice(positions, size=k, replace=False)
                    kl, dla, dlg = run_one_intervention(
                        model, inputs, use_aiv, logits_clean, top1_clean,
                        gt_token_id, layers[L], inter_name, sub, S)
                    records.append(dict(
                        layer=L, intervention=inter_name, condition=cond_name,
                        matching="count_matched", draw=d + 1, n_knockout=int(sub.size),
                        kl=kl, dlogp_argmax=dla, dlogp_gt=dlg,
                    ))

    if per_clip_dir is not None:
        # Save raw records for paired CI computation downstream
        out_npz = per_clip_dir / f"{clip_path.stem}.npz"
        if records:
            arr = np.array([(r["layer"], r["intervention"], r["condition"],
                              r["matching"], r["draw"], r["n_knockout"],
                              r["kl"], r["dlogp_argmax"], r["dlogp_gt"])
                             for r in records],
                            dtype=[("layer", "i4"), ("intervention", "U16"),
                                   ("condition", "U24"), ("matching", "U16"),
                                   ("draw", "i4"), ("n_knockout", "i4"),
                                   ("kl", "f4"), ("dlogp_argmax", "f4"),
                                   ("dlogp_gt", "f4")])
            np.savez_compressed(out_npz, records=arr,
                                 clip=np.array(clip_path.name),
                                 gt_token_id=np.int64(gt_token_id if gt_token_id is not None else -1),
                                 top1_clean=np.int64(top1_clean))
    torch.cuda.empty_cache()
    return records


def run_one_intervention(model, inputs, use_aiv, logits_clean, top1_clean,
                          gt_token_id, layer_module, intervention,
                          positions, S):
    """Install hook, forward, compute deltas, remove hook. Returns the
    three deltas."""
    if intervention == "attn_knockout":
        handle = layer_module.register_forward_pre_hook(
            make_attn_knockout_pre_hook(positions, S), with_kwargs=True)
    elif intervention == "value_zero":
        handle = layer_module.self_attn.v_proj.register_forward_hook(
            make_value_zero_hook(positions, S))
    else:
        raise ValueError(intervention)
    try:
        logits_int = forward_last_logits(model, inputs, use_aiv)
    finally:
        handle.remove()
    return deltas(logits_clean, logits_int, top1_clean, gt_token_id)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate_records(all_records: list, out_csv: Path):
    import pandas as pd
    df = pd.DataFrame(all_records)
    if df.empty:
        raise SystemExit("no records collected")

    # Average over draws per (clip, layer, intervention, condition, matching)
    per_clip = (df.groupby(["clip", "layer", "intervention", "condition", "matching"],
                            as_index=False)
                  .agg(n_knockout=("n_knockout", "mean"),
                       kl=("kl", "mean"),
                       dlogp_argmax=("dlogp_argmax", "mean"),
                       dlogp_gt=("dlogp_gt", "mean")))
    per_clip_path = out_csv.parent / "causal_probe_per_clip.csv"
    per_clip.to_csv(per_clip_path, index=False)
    print(f"wrote {per_clip_path}  ({len(per_clip)} per-clip rows)")

    def _ci(x):
        x = x.dropna().values
        n = len(x)
        if n < 2:
            return float("nan"), float("nan")
        m = float(x.mean())
        se = float(x.std(ddof=1) / max(np.sqrt(n), 1e-12))
        # 95% CI via t-distribution approx; for n=50 z=1.96 is fine
        return m - 1.96 * se, m + 1.96 * se

    def _per_token(x_kl, x_n):
        if x_n.dropna().empty:
            return float("nan")
        return float((x_kl / x_n.clip(lower=1)).mean())

    grouped = per_clip.groupby(["layer", "intervention", "condition", "matching"])
    rows = []
    for (L, inter, cond, match), g in grouped:
        kl_ci = _ci(g["kl"])
        dla_ci = _ci(g["dlogp_argmax"])
        dlg_ci = _ci(g["dlogp_gt"])
        rows.append(dict(
            layer=L, intervention=inter, condition=cond, matching=match,
            n_clips=int(g["clip"].nunique()),
            mean_n_knockout=float(g["n_knockout"].mean()),
            mean_kl=float(g["kl"].mean()),
            ci95_kl_low=kl_ci[0], ci95_kl_high=kl_ci[1],
            mean_dlogp_argmax=float(g["dlogp_argmax"].mean()),
            ci95_dlogp_argmax_low=dla_ci[0], ci95_dlogp_argmax_high=dla_ci[1],
            mean_dlogp_gt=float(g["dlogp_gt"].mean()),
            ci95_dlogp_gt_low=dlg_ci[0], ci95_dlogp_gt_high=dlg_ci[1],
            per_token_kl=_per_token(g["kl"], g["n_knockout"]),
        ))
    agg = pd.DataFrame(rows).sort_values(["layer", "intervention", "matching", "condition"])
    agg.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(agg)} rows)")

    # Paired tests: cross vs each of (uni_audio, uni_video, random, full_prop)
    paired_rows = []
    for (L, inter, match), g in per_clip.groupby(["layer", "intervention", "matching"]):
        wide = g.pivot(index="clip", columns="condition",
                        values=["kl", "dlogp_argmax", "dlogp_gt"])
        if "cross_prop" not in wide["kl"].columns:
            continue
        for other in ("ua_prop", "uv_prop", "random_span_matched", "full_prop"):
            if other not in wide["kl"].columns:
                continue
            for metric in ("kl", "dlogp_argmax", "dlogp_gt"):
                d = (wide[metric]["cross_prop"] - wide[metric][other]).dropna()
                if len(d) < 2:
                    continue
                m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d)))
                paired_rows.append(dict(
                    layer=L, intervention=inter, matching=match,
                    comparison=f"cross_minus_{other}", metric=metric,
                    n_clips=int(len(d)),
                    mean_diff=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se,
                    excludes_zero=bool((m - 1.96 * se) > 0 or (m + 1.96 * se) < 0),
                ))
    paired = pd.DataFrame(paired_rows)
    p_path = out_csv.parent / "causal_probe_paired.csv"
    paired.to_csv(p_path, index=False)
    print(f"wrote {p_path}  ({len(paired)} paired rows)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_clip_dir = out_dir / "per_clip"; per_clip_dir.mkdir(exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)
    print(f"  n_layers={n_layers}; probe layers={PROBE_LAYERS}; K_draws={K_DRAWS}")

    label_lookup = {}
    qa = json.loads(Path(args.qa_json).read_text())
    for e in qa:
        v = e.get("video_id")
        lab = e.get("label") or []
        if isinstance(lab, str): lab = [lab]
        label_lookup.setdefault(v, []).extend(lab)

    dump_dir = Path(args.dump_dir)
    video_dir = Path(args.video_dir)
    dumps = sorted(dump_dir.glob("*.npz"))
    if not dumps:
        raise SystemExit(f"no .npz under {dump_dir}")

    all_records = []
    failures = {}
    rng_seed = 12345
    for i, dump_path in enumerate(tqdm(dumps, desc="clips")):
        clip_name = dump_path.stem + ".mp4"
        clip_path = video_dir / clip_name
        if not clip_path.exists():
            failures["missing_video"] = failures.get("missing_video", 0) + 1
            continue
        labs = label_lookup.get(clip_name, [])
        recs = probe_clip(clip_path, dump_path, labs, processor.tokenizer,
                           model, processor, layers, n_layers,
                           K_draws=K_DRAWS, seed=rng_seed + i,
                           per_clip_dir=per_clip_dir)
        if recs is None:
            failures["fwd_fail"] = failures.get("fwd_fail", 0) + 1
            continue
        for r in recs:
            r["clip"] = clip_name
        all_records.extend(recs)

    if failures:
        print(f"  failures: {failures}")

    csv = out_dir / "causal_probe_by_condition.csv"
    aggregate_records(all_records, csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",  default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--dump_dir",   default=str(DEFAULT_DUMP))
    p.add_argument("--qa_json",    default=str(DEFAULT_QA_JSON))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
