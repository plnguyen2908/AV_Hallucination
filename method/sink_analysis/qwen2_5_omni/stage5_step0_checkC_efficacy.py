"""
stage5_step0_checkC_efficacy.py — CHECK C, the Stage 5 go/no-go gate.

Two parts on the same 37 paired clips used in Stages 4.2/4.3/4.4:

  (i) CORRELATION (clip-level, NOT head-level): does higher attention mass
      on P_llm-audio sinks predict more hallucinated tokens per clip?
      Reuses Stage 4.2 records (attention to audio sink bins by clip).
  (ii) MINIMAL INTERVENTION: at γ ∈ {0.0, 0.5, 1.0=baseline} apply the
      reweighting primitive (multiplicative down-weight of attention to
      P_llm-audio sink KEYS + row-renormalize via softmax) and measure
      the change in HAL-token log-probability vs the baseline.

The reweighting is implemented as an ADD to the pre-softmax attention
mask: adding `log(γ)` at target key columns multiplies post-softmax
attention to those keys by γ and lets the renormalization redistribute
freed mass to remaining keys (NOT masking; this is the row-renorm form).

Pass: suppressing P_llm-audio sinks measurably reduces hallucinated-token
log-probability (paired across clips, position-controlled, CI excludes 0
in the expected direction).
Fail: no correlation AND no movement → safe target is inert (consistent
with 3.3b) → STOP, no router, consolidate the negative paper.

Outputs:
  checkC_efficacy.csv                — per-(clip, γ, token_type) mean log-prob
  checkC_summary.csv                 — paired Δlogp_hal across γ
  checkC_correlation.csv             — clip-level correlation table
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
DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_step0"

GAMMAS = [0.0, 0.5, 1.0]    # 1.0 = baseline (no intervention)


def assign_token_types(caption_ids, hal_tokens, non_tokens):
    hal_left = Counter(hal_tokens); non_left = Counter(non_tokens)
    hal_positions, non_positions = [], []
    for pos, tid in enumerate(caption_ids):
        if hal_left[tid] > 0:
            hal_positions.append(pos); hal_left[tid] -= 1
        elif non_left[tid] > 0:
            non_positions.append(pos); non_left[tid] -= 1
    return hal_positions, non_positions


def make_reweight_pre_hook(p_llm_audio_per_layer: dict, S: int, gamma: float):
    """register_forward_pre_hook(with_kwargs=True) on a decoder layer.
    Adds log(gamma) to the attention_mask kwarg at target key columns
    (P_llm-audio sink positions at THIS layer). For γ=1 a no-op; for
    γ=0 the keys are knocked out; in between, the row-renormalization
    of softmax redistributes the freed mass to remaining keys (the
    spec's reweighting primitive)."""
    if gamma == 1.0:
        def _noop(m, args, kwargs): return args, kwargs
        return _noop
    log_gamma = float(np.log(max(gamma, 1e-30)))   # log(0) handled below

    def _hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None: return args, kwargs
        if am.shape[-1] != S: return args, kwargs
        # Per-layer P_llm-audio key positions
        layer_idx = getattr(module, "_layer_idx_for_hook", None)
        if layer_idx is None: return args, kwargs
        positions = p_llm_audio_per_layer.get(layer_idx, np.array([], dtype=np.int64))
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
    """Forward thinker (teacher-forced); return per-caption-position
    log-prob of the actually-emitted next token."""
    with torch.inference_mode():
        out = model.thinker(**inputs, use_audio_in_video=use_aiv,
                            output_attentions=False, return_dict=True,
                            use_cache=False, output_hidden_states=False)
    logits = out.logits[0]                                # (S, V)
    logp = torch.log_softmax(logits.float(), dim=-1)
    # For caption position p (in [0..len(caption)-1]), the model predicts
    # at absolute position prompt_S - 1 + p, and the target is caption_ids[p].
    # That is: logits at position (prompt_S - 1 + p) → next token = caption_ids[p].
    per_pos = []
    for p, tid in enumerate(caption_ids):
        abs_q = prompt_S - 1 + p
        per_pos.append((p, int(tid), float(logp[abs_q, int(tid)].item())))
    return per_pos


D_SINK = [458, 2570]
TAU_SINK = 20.0
AUDIO_TOKEN_ID = 151646


def _build_p_llm_per_layer(per_layer_h_input, n_layers, eps_norm):
    """From per-layer pre-SA hidden states, compute per-layer P_llm masks
    using the Stage 3.1/3.2 convention: RMSNorm + D_sink + τ=20."""
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    masks = {}
    for L in range(n_layers):
        h = per_layer_h_input[L].float()      # (S, D)
        rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h / rms).abs()
        d_t = d_sink_t.to(h.device)
        is_sink = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
        masks[L] = is_sink                    # bool array length S
    return masks


def process_clip(d, model, processor, layers, n_layers,
                  eps_norm, gammas, out_records, out_attn_mass):
    video_path = Path(_MEDIA_DIR) / d["video"]
    if not video_path.exists():
        return "missing_video"

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

    # Append caption ids (teacher-force)
    cap_t = torch.tensor([caption_ids],
                          device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], cap_t], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    full_S = int(inputs["input_ids"].shape[1])

    # Baseline forward with pre-hooks to capture h_input[L] for the P_llm
    # masks, AND record baseline logits at caption positions.
    per_layer_h = [None] * n_layers

    def _make_capture(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach()
        return _h

    handles = [layers[L].register_forward_pre_hook(_make_capture(L))
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
    if any(h is None for h in per_layer_h):
        return "no_capture"
    base_logits = base_out.logits[0].float()              # (S, V)

    # Compute P_llm masks per layer from h_input
    p_llm_masks = _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm)
    # Free hidden states
    per_layer_h = [None] * n_layers
    torch.cuda.empty_cache()

    # Audio positions from input_ids
    ids_np = inputs["input_ids"][0].cpu().numpy()
    audio_pos = np.where(ids_np[:prompt_S] == AUDIO_TOKEN_ID)[0].astype(np.int64)

    audio_set = set(audio_pos.tolist())
    p_llm_audio_per_layer: dict[int, np.ndarray] = {}
    for L in range(n_layers):
        idxs = [i for i, b in enumerate(p_llm_masks[L]) if b and (i in audio_set)]
        p_llm_audio_per_layer[L] = np.asarray(idxs, dtype=np.int64)

    # Tag layers for the hook (so it knows its own index)
    for L in range(n_layers):
        layers[L]._layer_idx_for_hook = L

    hal_set = set(hal_pos); non_set = set(non_pos)
    # Record γ=1 baseline directly from the already-captured base_logits
    logp_base = torch.log_softmax(base_logits, dim=-1)
    for p, tid in enumerate(caption_ids):
        abs_q = prompt_S - 1 + p
        lp = float(logp_base[abs_q, int(tid)].item())
        tt = "hal" if p in hal_set else ("non" if p in non_set else "other")
        out_records.append((d["video"], 1.0, tt, int(p), int(tid), lp))
    # Free base_logits to save memory before re-running
    del base_logits, logp_base
    torch.cuda.empty_cache()

    # Forward at each γ != 1
    for gamma in gammas:
        if gamma == 1.0: continue
        hook = make_reweight_pre_hook(p_llm_audio_per_layer, full_S, gamma)
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

    # Attention-mass on P_llm-audio sinks per clip (mean per-token attention
    # over caption-query positions): take a single γ=1 forward with
    # output_attentions=True and one-hook capturing the values.
    # Done lazily below; for now record n_hal / n_non and target ids.
    out_attn_mass.append((d["video"], int(len(hal_pos)), int(len(non_pos)),
                          float(np.mean([len(v) for v in p_llm_audio_per_layer.values()])),
                          int(audio_pos.size)))

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

    records = []
    attn_meta = []
    failures = {}
    for d in tqdm(paired, desc="clips"):
        err = process_clip(d, model, processor, layers, n_layers,
                            eps_norm, GAMMAS, records, attn_meta)
        if err:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
    if failures: print(f"failures: {failures}")
    print(f"records: {len(records):,}; attn_meta: {len(attn_meta)}")

    df = pd.DataFrame(records, columns=[
        "clip", "gamma", "token_type", "token_position_in_caption",
        "target_token_id", "logp"])
    rec_path = out_dir / "checkC_efficacy.csv"
    df.to_csv(rec_path, index=False)
    print(f"wrote {rec_path}")

    # Per-(clip, gamma, ttype) mean logp + per-clip Δ vs baseline
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
    sum_path = out_dir / "checkC_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"wrote {sum_path}")
    print(sum_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---- correlation: attention mass on P_llm audio sinks vs hal-token count ----
    # Use Stage 4.2's entity_attention CSV (has per-clip mean per_token_inflow per bin).
    s42 = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_2/stage4_2_entity_attention.csv"
    if s42.exists():
        print(f"\nloading {s42} for correlation ...")
        attn = pd.read_csv(s42)
        sub = attn[(attn["bin"] == "p_llm_uni_audio")
                    & (attn["token_type"].isin(["hal", "non"]))]
        clip_mass = (sub.groupby("clip", as_index=False)
                          .agg(mean_per_token=("per_token_inflow", "mean")))
        # hal count from sampled entities
        hal_count = {d["video"]: len(d.get("hallucinated_tokens", [])) for d in paired}
        clip_mass["hal_count"] = clip_mass["clip"].map(hal_count)
        from scipy.stats import spearmanr, pearsonr
        if not clip_mass["hal_count"].isna().all():
            cm = clip_mass.dropna()
            rho_s, p_s = spearmanr(cm["mean_per_token"].values, cm["hal_count"].values)
            rho_p, p_p = pearsonr(cm["mean_per_token"].values, cm["hal_count"].values)
            corr_df = pd.DataFrame([dict(
                metric="P_llm_audio attn-mass vs hal_count",
                n_clips=int(len(cm)),
                spearman_rho=float(rho_s), spearman_p=float(p_s),
                pearson_r=float(rho_p), pearson_p=float(p_p),
            )])
            corr_path = out_dir / "checkC_correlation.csv"
            corr_df.to_csv(corr_path, index=False)
            print(f"wrote {corr_path}")
            print(corr_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--dump_dir", default=str(DEFAULT_DUMP))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--modality", default="av", choices=["a", "v", "av"],
                   help="Modal type passed to build_conversation/prepare_inputs.")
    p.add_argument("--media_dir", default=None,
                   help="Override the default media directory (audio/video files).")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
