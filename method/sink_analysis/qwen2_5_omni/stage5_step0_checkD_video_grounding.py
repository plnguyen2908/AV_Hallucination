"""
stage5_step0_checkD_video_grounding.py — CHECK D.

Same primitive shape as CHECK C v2 but suppressing P_prop-video sink
attention (encoder L2 norm > 100, fixed across layers, video span)
instead of P_llm-audio. Tests whether suppressing P_prop-video sinks
hurts visual grounding — operationalized as Δlogp on non-hal
("grounded") caption tokens.

Pass / fail (informs policy design, not gate):
  Δlogp_non DROPS under P_prop-video suppression → P_prop-video IS
    grounding-critical → "PROTECT P_prop-video" rule justified for the
    video arm.
  Δlogp_non UNCHANGED → P_prop-video not grounding-critical → the video
    arm can suppress P_prop-video freely.

γ ∈ {0.0, 0.5, 1.0}. Same row-renormalized soft-suppression primitive,
applied to the same per-layer P_prop-video positions (since P_prop is
fixed across layers by encoder definition, the mask is constant per
layer).

Outputs (`stage5_step0/checkD/`):
  checkD_efficacy.csv
  checkD_summary.csv  + paired-differential Δ(non - hal) per γ
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
_MODALITY = "av"
_MEDIA_DIR = str(DEFAULT_VID)
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_step0/checkD"

TAU_PROP = 100.0
AUDIO_TOKEN_ID = 151646
VIDEO_TOKEN_ID = 151656
GAMMAS = [0.0, 0.5, 1.0]


def assign_token_types(caption_ids, hal_tokens, non_tokens):
    hal_left = Counter(hal_tokens); non_left = Counter(non_tokens)
    hal_positions, non_positions = [], []
    for pos, tid in enumerate(caption_ids):
        if hal_left[tid] > 0:
            hal_positions.append(pos); hal_left[tid] -= 1
        elif non_left[tid] > 0:
            non_positions.append(pos); non_left[tid] -= 1
    return hal_positions, non_positions


def _resolve_encoders(model):
    thinker = model.thinker
    audio = visual = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr): audio = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr): visual = getattr(thinker, attr); break
    return audio, visual


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def _align_norms(enc_norms, n_llm):
    n_enc = len(enc_norms)
    if n_enc == n_llm: return enc_norms
    if n_enc > n_llm and n_enc % n_llm == 0:
        return enc_norms.reshape(n_llm, n_enc // n_llm).mean(axis=1)
    if n_llm > n_enc and n_llm % n_enc == 0:
        return np.repeat(enc_norms, n_llm // n_enc)
    return None


def make_suppress_hook(positions: np.ndarray, S: int, gamma: float):
    """Suppress attention TO `positions` at every query/layer, identical
    shape to CHECK C v2 hook but with a fixed per-clip key set
    (P_prop-video is layer-independent)."""
    if gamma == 1.0:
        def _noop(m, args, kwargs): return args, kwargs
        return _noop
    log_gamma = float(np.log(max(gamma, 1e-30)))
    pos_t = torch.as_tensor(positions, dtype=torch.long)

    def _hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None or am.shape[-1] != S: return args, kwargs
        if pos_t.numel() == 0: return args, kwargs
        new_am = am.clone()
        idx = pos_t.to(new_am.device)
        if gamma == 0.0:
            new_am[..., :, idx] = torch.finfo(new_am.dtype).min
        else:
            new_am[..., :, idx] = new_am[..., :, idx] + log_gamma
        kwargs["attention_mask"] = new_am
        return args, kwargs
    return _hook


def forward_caption_logprobs(model, inputs, use_aiv, prompt_S, caption_ids):
    with torch.inference_mode():
        out = model.thinker(**inputs, use_audio_in_video=use_aiv,
                            output_attentions=False, return_dict=True,
                            use_cache=False)
    logits = out.logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    per_pos = []
    for p, tid in enumerate(caption_ids):
        abs_q = prompt_S - 1 + p
        per_pos.append((p, int(tid), float(logp[abs_q, int(tid)].item())))
    return per_pos


def process_clip(d, model, processor, layers, n_layers, visual_enc,
                  gammas, out_records):
    video_path = Path(_MEDIA_DIR) / d["video"]
    if not video_path.exists(): return "missing_video"
    conv = build_conversation(str(video_path), d["question"], _MODALITY)
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, _MODALITY,
                                           model.device, model.dtype)
    except Exception as e:
        return f"prep:{type(e).__name__}"
    prompt_S = int(inputs["input_ids"].shape[1])
    caption_ids = processor.tokenizer(d["generated_caption"],
                                        add_special_tokens=False).input_ids
    if not caption_ids: return "empty_caption"
    hal_pos, non_pos = assign_token_types(
        caption_ids,
        d.get("hallucinated_tokens", []),
        d.get("non_hallucinated_tokens", []))
    if not hal_pos and not non_pos: return "no_labels"
    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    full_S = int(inputs["input_ids"].shape[1])

    # Baseline forward — capture logits + visual encoder norms.
    v_enc_buf = []
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        v_enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    handle = visual_enc.register_forward_hook(enc_hook) if visual_enc else None
    try:
        with torch.inference_mode():
            base_out = model.thinker(**inputs, use_audio_in_video=use_aiv,
                                       output_attentions=False,
                                       return_dict=True, use_cache=False)
    except Exception as e:
        if handle: handle.remove()
        torch.cuda.empty_cache()
        return f"fwd_base:{type(e).__name__}"
    finally:
        if handle: handle.remove()
    if not v_enc_buf: return "no_encoder_capture"

    base_logits = base_out.logits[0].float()
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0].astype(np.int64)
    if video_pos.size == 0: return "no_video_tokens"

    v_enc = np.concatenate(v_enc_buf)
    aligned = _align_norms(v_enc, len(video_pos))
    p_prop_video_positions = (
        video_pos[aligned > TAU_PROP] if aligned is not None
        else np.array([], dtype=np.int64))

    hal_set = set(hal_pos); non_set = set(non_pos)
    logp_base = torch.log_softmax(base_logits, dim=-1)
    for p, tid in enumerate(caption_ids):
        abs_q = prompt_S - 1 + p
        lp = float(logp_base[abs_q, int(tid)].item())
        tt = "hal" if p in hal_set else ("non" if p in non_set else "other")
        out_records.append((d["video"], 1.0, tt, int(p), int(tid), lp,
                              int(p_prop_video_positions.size)))
    del base_logits, logp_base
    torch.cuda.empty_cache()

    for gamma in gammas:
        if gamma == 1.0: continue
        hook = make_suppress_hook(p_prop_video_positions, full_S, gamma)
        handles = [layers[L].register_forward_pre_hook(hook, with_kwargs=True)
                    for L in range(n_layers)]
        try:
            per_pos = forward_caption_logprobs(model, inputs, use_aiv,
                                                  prompt_S, caption_ids)
        except Exception as e:
            for h in handles: h.remove()
            torch.cuda.empty_cache()
            return f"fwd_gamma{gamma}:{type(e).__name__}"
        finally:
            for h in handles: h.remove()
        for p, tid, lp in per_pos:
            tt = "hal" if p in hal_set else ("non" if p in non_set else "other")
            out_records.append((d["video"], float(gamma), tt, int(p),
                                  int(tid), float(lp),
                                  int(p_prop_video_positions.size)))
    torch.cuda.empty_cache()
    return None


def main(args):
    global _MODALITY, _MEDIA_DIR
    _MODALITY = args.modality
    if args.media_dir:
        _MEDIA_DIR = args.media_dir
    print(f"  modality={_MODALITY}; media_dir={_MEDIA_DIR}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)
    _, visual_enc = _resolve_encoders(model)
    print(f"n_layers={n_layers}; visual_enc={'OK' if visual_enc else 'MISSING'}")

    data = json.load(open(args.sampled_entities_json))
    # CHECK D measures Δlogp on hal vs non. For paired-DiD style (VGGSounder)
    # we want both. For one-sided "does grounding hurt?" (ActivityNet has
    # 161 non-only clips, 0 paired), we accept clips with EITHER label set
    # populated; process_clip will skip ones where it can't classify any
    # positions but uses the available tokens.
    paired = [d for d in data
              if d.get("generated_caption")
              and (d.get("hallucinated_tokens") or d.get("non_hallucinated_tokens"))]
    print(f"{len(paired)} clips (any-label; CHECK D doesn't require paired)")

    records = []
    failures = {}
    for d in tqdm(paired, desc="clips"):
        err = process_clip(d, model, processor, layers, n_layers, visual_enc,
                            GAMMAS, records)
        if err:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
    if failures: print(f"failures: {failures}")
    print(f"records={len(records):,}")

    df = pd.DataFrame(records, columns=[
        "clip", "gamma", "token_type", "token_position_in_caption",
        "target_token_id", "logp", "n_p_prop_video"])
    df.to_csv(out_dir / "checkD_efficacy.csv", index=False)

    per = (df.groupby(["clip", "gamma", "token_type"], as_index=False)
              .agg(mean_logp=("logp", "mean")))
    base = per[per["gamma"] == 1.0].rename(columns={"mean_logp": "baseline_logp"})\
                                       [["clip", "token_type", "baseline_logp"]]
    per = per.merge(base, on=["clip", "token_type"], how="left")
    per["delta_logp"] = per["mean_logp"] - per["baseline_logp"]

    sum_rows = []
    from scipy.stats import t as _t
    for (gamma, tt), g in per.groupby(["gamma", "token_type"]):
        x = g["delta_logp"].dropna().values
        n = len(x); m = float(x.mean()) if n else float("nan")
        se = float(x.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else float("nan")
        p = float(2 * _t.sf(abs(m / se if se else 0.0), df=n - 1)) if n > 1 else float("nan")
        sum_rows.append(dict(
            gamma=float(gamma), token_type=tt, n_clips=n,
            mean_delta_logp=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se,
            pval=p,
            excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0))))
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(out_dir / "checkD_summary.csv", index=False)
    print(sum_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Paired differential Δ(non - hal) per γ — grounding-specific
    pivot = per.pivot_table(index=["clip","gamma"], columns="token_type",
                              values="delta_logp").reset_index()
    if "hal" in pivot.columns and "non" in pivot.columns:
        pivot["delta_non_minus_hal"] = pivot["non"] - pivot["hal"]
        diff_rows = []
        for gamma in sorted(pivot["gamma"].unique()):
            if gamma == 1.0: continue
            sub = pivot[pivot["gamma"] == gamma]["delta_non_minus_hal"].dropna().values
            n = len(sub); m = float(sub.mean()); se = float(sub.std(ddof=1) / np.sqrt(n))
            p = float(2 * _t.sf(abs(m / se if se else 0.0), df=n - 1))
            diff_rows.append(dict(
                gamma=float(gamma), n_clips=n, mean_delta_non_minus_hal=m,
                ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se, pval=p,
                excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0))))
        diff_df = pd.DataFrame(diff_rows)
        diff_df.to_csv(out_dir / "checkD_paired_diff.csv", index=False)
        print("\n=== paired Δ(non - hal) per γ ===")
        print(diff_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--modality", default="av", choices=["a", "v", "av"],
                   help="Modal type passed to build_conversation/prepare_inputs.")
    p.add_argument("--media_dir", default=None,
                   help="Override the default media directory (audio/video files).")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
