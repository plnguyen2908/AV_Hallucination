"""
stage5_step0_checkC_v2.py — CHECK C v2: redistribute properly.

v1 (already run) implemented the spec literally: suppress P_llm-audio
sinks only, let softmax row-renormalize. The renormalization
redistributed the freed mass proportionally to ALL remaining keys —
which in Qwen-Omni is dominated by BOS + text_sys (~50× the per-token
attention of any sink bin per Stage 4.1/4.2). Net: the freed mass
went to structural destinations, not to non-sink audio CONTENT, and
hal-token logp didn't move.

v2 fixes this: ALSO suppress BOS and text_sys at the same γ as
P_llm-audio. With those dominant junk destinations down-weighted,
softmax renormalization concentrates the residual mass on the actual
content keys (non-sink audio + P_prop audio + video). This is the
"redistribute freed mass to content" primitive the spec intended.

This is still SUPPRESSION (γ ≤ 1 on the unwanted destinations); the
up-weighting of content is implicit via renormalization. No explicit
boost.

Three γ settings:
  γ=1.0 baseline
  γ=0.5 partial suppression on {P_llm-audio, BOS, text_sys}
  γ=0.0 full knockout on {P_llm-audio, BOS, text_sys}

Pass: Δlogp_hal < 0 with CI excluding 0 (suppression-with-proper-
redistribution reduces hal-token prob).
Fail: still ≈ 0 → the lever does not exist in any form; strong NO-GO.

Outputs (`stage5_step0/v2/`):
  checkC_v2_efficacy.csv
  checkC_v2_summary.csv
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
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_step0/v2"

D_SINK = [458, 2570]
TAU_SINK = 20.0
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


def _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm):
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    masks = {}
    for L in range(n_layers):
        h = per_layer_h[L].float()
        rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h / rms).abs()
        d_t = d_sink_t.to(h.device)
        masks[L] = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
    return masks


def make_v2_pre_hook(target_positions_per_layer: dict, S: int, gamma: float):
    """v2 hook — same primitive as v1 but `target_positions_per_layer`
    now contains the union of {P_llm-audio sinks, BOS, text_sys} per
    layer. Adds log(γ) to the attention_mask at those key columns; the
    softmax renorm then concentrates surviving mass on non-target keys."""
    if gamma == 1.0:
        def _noop(m, args, kwargs): return args, kwargs
        return _noop
    log_gamma = float(np.log(max(gamma, 1e-30)))

    def _hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None or am.shape[-1] != S: return args, kwargs
        L_idx = getattr(module, "_layer_idx_for_hook", None)
        if L_idx is None: return args, kwargs
        positions = target_positions_per_layer.get(L_idx, np.array([], dtype=np.int64))
        if positions.size == 0: return args, kwargs
        new_am = am.clone()
        idx = torch.as_tensor(positions, dtype=torch.long, device=new_am.device)
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


def process_clip(d, model, processor, layers, n_layers, eps_norm,
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
    if not hal_pos or not non_pos: return "no_both_types"

    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    full_S = int(inputs["input_ids"].shape[1])

    # Baseline forward with pre-SA capture
    per_layer_h = [None] * n_layers
    def _make_cap(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach()
        return _h
    handles = [layers[L].register_forward_pre_hook(_make_cap(L))
                for L in range(n_layers)]
    try:
        with torch.inference_mode():
            base_out = model.thinker(**inputs, use_audio_in_video=use_aiv,
                                       output_attentions=False,
                                       return_dict=True, use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        return f"fwd_base:{type(e).__name__}"
    finally:
        for h in handles: h.remove()
    if any(h is None for h in per_layer_h): return "no_capture"
    base_logits = base_out.logits[0].float()
    p_llm_masks = _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm)
    per_layer_h = [None] * n_layers
    torch.cuda.empty_cache()

    # Build the v2 target set per layer:
    #   target = P_llm-audio  ∪  {bos = pos 0}  ∪  text_sys (non-modal, non-bos)
    ids_np = inputs["input_ids"][0].cpu().numpy()
    is_audio = (ids_np[:prompt_S] == AUDIO_TOKEN_ID)
    is_video = (ids_np[:prompt_S] == VIDEO_TOKEN_ID)
    audio_pos = np.where(is_audio)[0]
    is_bos = np.zeros(prompt_S, dtype=bool); is_bos[0] = True
    text_sys_mask = ~(is_audio | is_video | is_bos)
    text_sys_positions = np.where(text_sys_mask)[0].astype(np.int64)
    bos_positions = np.array([0], dtype=np.int64)

    audio_set = set(audio_pos.tolist())
    target_per_layer: dict[int, np.ndarray] = {}
    for L in range(n_layers):
        p_llm_audio = np.array(
            [i for i, b in enumerate(p_llm_masks[L]) if b and (i in audio_set)],
            dtype=np.int64)
        # union P_llm-audio ∪ text_sys ∪ bos
        target_per_layer[L] = np.unique(np.concatenate([
            p_llm_audio, text_sys_positions, bos_positions]))

    for L in range(n_layers):
        layers[L]._layer_idx_for_hook = L

    hal_set = set(hal_pos); non_set = set(non_pos)
    logp_base = torch.log_softmax(base_logits, dim=-1)
    for p, tid in enumerate(caption_ids):
        abs_q = prompt_S - 1 + p
        lp = float(logp_base[abs_q, int(tid)].item())
        tt = "hal" if p in hal_set else ("non" if p in non_set else "other")
        out_records.append((d["video"], 1.0, tt, int(p), int(tid), lp))
    del base_logits, logp_base
    torch.cuda.empty_cache()

    for gamma in gammas:
        if gamma == 1.0: continue
        hook = make_v2_pre_hook(target_per_layer, full_S, gamma)
        handles = [layers[L].register_forward_pre_hook(hook, with_kwargs=True)
                    for L in range(n_layers)]
        try:
            per_pos = forward_caption_logprobs(model, inputs, use_aiv, prompt_S, caption_ids)
        except Exception as e:
            for h in handles: h.remove()
            torch.cuda.empty_cache()
            return f"fwd_gamma{gamma}:{type(e).__name__}"
        finally:
            for h in handles: h.remove()
        for p, tid, lp in per_pos:
            tt = "hal" if p in hal_set else ("non" if p in non_set else "other")
            out_records.append((d["video"], float(gamma), tt, int(p),
                                  int(tid), float(lp)))
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
    eps_norm = float(getattr(
        getattr(model.thinker.config, "text_config", model.thinker.config),
        "rms_norm_eps", 1e-6))

    data = json.load(open(args.sampled_entities_json))
    paired = [d for d in data
              if d.get("hallucinated_tokens") and d.get("non_hallucinated_tokens")]
    print(f"{len(paired)} paired clips; rms_eps={eps_norm}")
    print("v2 target = P_llm-audio ∪ BOS ∪ text_sys (the dominant non-content destinations)")

    records = []
    failures = {}
    for d in tqdm(paired, desc="clips"):
        err = process_clip(d, model, processor, layers, n_layers,
                            eps_norm, GAMMAS, records)
        if err:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
    if failures: print(f"failures: {failures}")
    print(f"records: {len(records):,}")

    df = pd.DataFrame(records, columns=[
        "clip", "gamma", "token_type", "token_position_in_caption",
        "target_token_id", "logp"])
    df.to_csv(out_dir / "checkC_v2_efficacy.csv", index=False)

    per = (df.groupby(["clip", "gamma", "token_type"], as_index=False)
              .agg(mean_logp=("logp", "mean")))
    base = per[per["gamma"] == 1.0].rename(columns={"mean_logp": "baseline_logp"})\
                                       [["clip", "token_type", "baseline_logp"]]
    per = per.merge(base, on=["clip", "token_type"], how="left")
    per["delta_logp"] = per["mean_logp"] - per["baseline_logp"]
    sum_rows = []
    for (gamma, tt), g in per.groupby(["gamma", "token_type"]):
        x = g["delta_logp"].dropna().values
        n = len(x); m = float(x.mean()) if n else float("nan")
        se = float(x.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else float("nan")
        from scipy.stats import t as _t
        p = float(2 * _t.sf(abs(m / se if se else 0.0), df=n - 1)) if n > 1 else float("nan")
        sum_rows.append(dict(
            gamma=float(gamma), token_type=tt, n_clips=n,
            mean_delta_logp=m, ci95_low=m - 1.96 * se, ci95_high=m + 1.96 * se,
            pval=p,
            excludes_zero=bool((m - 1.96 * se > 0) or (m + 1.96 * se < 0))))
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(out_dir / "checkC_v2_summary.csv", index=False)
    print(sum_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


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
