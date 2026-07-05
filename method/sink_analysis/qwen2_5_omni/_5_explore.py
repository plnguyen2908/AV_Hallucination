"""
_5_explore.py — exploration loop for Stage 5 interventions on AVHBench.

Runs the routed-modality intervention with a chosen config on a split
(DEV by default). For each clip:
  Pass 1 (cheap): forward thinker on prompt only, capture per-layer
    pre-SA hidden states + encoder norms → per-clip p_llm AND p_prop.
  Pass 2 (generation): set the intervention config (target heads +
    per-layer key mask = LLM-emerged sinks in the routed modality),
    run generate, parse Yes/No.

The intervention parameters:
  --variant {suppress_halluc_sink, suppress_inert_sink,
              boost_halluc_content, boost_inert_content,
              suppress_centric_sink, none}
  --gamma   <float>  for suppress: multiplier (e.g. 0.5) on sink keys
                       for boost:    epsilon (e.g. 0.5 → x1.5) on non-sinks
  --use_routing      if set, only apply to heads of the routed modality
                       (audio/visual/AV halluc head set).
                       Otherwise apply to all heads of that head-category
                       across modalities (e.g. all Audio-halluc heads
                       regardless of routed modality).

Inputs:
  --split_csv      results/.../stage5_intervention/split.csv
  --router_csv     router_v2 csv (per-question routed modality)
  --heads_csv      categorize_exp_2axis/heads.csv  (category per (L,h))

Outputs:
  --output_dir     by default the same dir; per-config result csv
                     `interv_<tag>_DEV.csv` and a row appended to
                     `method_log.md` with method+intuition+accuracy.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, thinker_layers, OMNI_SYSTEM_PROMPT  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

import _5_intervene as IV  # noqa: E402

# Match the offline 4.1b definitions exactly
D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0

DEFAULT_VIDEO_DIR = _REPO / "data/AVHBench/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention"
YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."

TASK_TO_GT_MODALITY = {
    "Video-driven Audio Hallucination": "AUDIO",
    "Audio-driven Video Hallucination": "VISUAL",
    "AV Matching":                       "AV",
}


# ---------------------------------------------------------------------
# Helpers (matching _5_phaseB_prep)
# ---------------------------------------------------------------------

def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_token_index"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _thinker_rms_eps(model) -> float:
    cfg = getattr(model.thinker.config, "text_config", model.thinker.config)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def _resolve_encoders(model):
    thinker = model.thinker
    audio_mod = visual_mod = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr); break
    return audio_mod, visual_mod


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


def _modal_positions(input_ids, thinker_cfg):
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


def parse_yes_no(text: str) -> str:
    m = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unk"


# ---------------------------------------------------------------------
# Head-set selection from categorize_exp_2axis
# ---------------------------------------------------------------------

def load_head_sets(heads_csv: Path) -> Dict[str, List]:
    """Return dict modality -> list of (layer, head) tuples."""
    df = pd.read_csv(heads_csv)
    sets: dict = {}
    for cat in ["Audio head", "Visual head", "Audiovisual head", "Inert"]:
        sub = df[df.category == cat]
        sets[cat] = [(int(r["layer"]), int(r["head"]))
                     for _, r in sub.iterrows()]
    return sets


def heads_for_variant_routing(head_sets, variant: str,
                                routed_modality: str) -> List:
    """Return the (layer, head) list for the chosen variant given the
    routed modality."""
    # routed_modality ∈ AUDIO, VISUAL, AV
    cat_for_routed = {
        "AUDIO": "Audio head",
        "VISUAL": "Visual head",
        "AV": "Audiovisual head",
    }[routed_modality]
    # New: union variants ignore routing and combine head sets.
    if "all_halluc" in variant:
        return (head_sets["Audio head"]
                  + head_sets["Visual head"]
                  + head_sets["Audiovisual head"])
    if "halluc_and_inert" in variant or "inert_and_halluc" in variant:
        return (head_sets["Inert"]
                  + head_sets["Audio head"]
                  + head_sets["Visual head"]
                  + head_sets["Audiovisual head"])
    if "all_heads" in variant:
        # All 784 heads regardless of category.
        return (head_sets["Inert"]
                  + head_sets["Audio head"]
                  + head_sets["Visual head"]
                  + head_sets["Audiovisual head"])
    if "halluc" in variant:
        return head_sets[cat_for_routed]
    if "inert" in variant:
        return head_sets["Inert"]
    return []


def heads_by_layer(heads_list, layer_band: str = "full"):
    """Convert list of (L, h) tuples to dict {L: [h, ...]}, optionally
    restricting to a layer band: early=0-9, mid=10-18, late=19-27."""
    bands = {
        "full": range(0, 28),
        "early": range(0, 10),
        "mid": range(10, 19),
        "late": range(19, 28),
    }
    allowed = set(bands[layer_band])
    d: dict = {}
    for L, h in heads_list:
        if L in allowed:
            d.setdefault(L, []).append(h)
    return d


# ---------------------------------------------------------------------
# Per-clip sink mask computation (Pass 1)
# ---------------------------------------------------------------------

def _compute_asd_mds(attentions, audio_pos, video_pos, sinks_positions,
                      target_layers):
    """ASD MDS — for each sink position, compute attention from video-query
    positions vs audio-query positions, return |v_attn − a_attn|/(v+a).
    Aggregated across target_layers, head-averaged.

    Returns: dict {sink_position: |MDS|}, and an "adaptive_weight" =
    cross_mass / (cross_mass + uni_mass) for the whole clip.
    """
    if len(sinks_positions) == 0:
        return {}, 0.0
    cross_sum = 0.0; uni_sum = 0.0
    mds_per_sink = {}
    for L in target_layers:
        if L >= len(attentions): continue
        attn = attentions[L][0]  # (H, Q, K)
        if attn is None: continue
        # video-query rows: mean over heads × queries in video_pos
        if len(video_pos) > 0:
            v_attn_per_key = attn[:, video_pos, :].mean(dim=(0, 1))  # (K,)
        else:
            v_attn_per_key = torch.zeros(attn.shape[-1], device=attn.device)
        if len(audio_pos) > 0:
            a_attn_per_key = attn[:, audio_pos, :].mean(dim=(0, 1))
        else:
            a_attn_per_key = torch.zeros(attn.shape[-1], device=attn.device)
        for sink in sinks_positions:
            v = float(v_attn_per_key[sink])
            a = float(a_attn_per_key[sink])
            denom = v + a + 1e-8
            mds = abs(v - a) / denom
            mds_per_sink[sink] = mds_per_sink.get(sink, []) + [mds]
            # cross-modal contribution if low |MDS|, uni-modal if high
            if mds < 0.3:
                cross_sum += v + a
            else:
                uni_sum += v + a
    # Average MDS per sink
    avg_mds = {k: sum(vs)/len(vs) for k, vs in mds_per_sink.items()}
    adaptive_weight = cross_sum / (cross_sum + uni_sum + 1e-8)
    return avg_mds, adaptive_weight


def compute_per_layer_sink_masks(model, processor, conv, use_aiv,
                                   layers, audio_enc, visual_enc,
                                   eps_norm, d_sink_t,
                                   routed_modality: str) -> Dict[int, torch.Tensor]:
    """Run a prompt forward to capture p_llm and p_prop, then build per-
    layer key masks restricted to the routed modality (AUDIO/VISUAL/AV).
    With --asd: also computes per-sink MDS and restricts mask to
    cross-modal sinks. Stores `_asd_adaptive_weight` on the module-level
    `_RUNTIME_STATE` for the intervention to read."""
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_aiv)
    text = processor.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False)
    if isinstance(text, list): text = text[0]
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True,
                        use_audio_in_video=use_aiv)
    inputs = inputs.to(model.device).to(model.dtype)
    thinker_cfg = _resolve_thinker_cfg(model)
    audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
    S = int(inputs["input_ids"].shape[1])
    n_layers = len(layers)

    per_layer_h = [None] * n_layers
    a_enc_buf, v_enc_buf = [], []

    def make_pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach().cpu()
        return _h

    def make_enc_hook(buf):
        def _h(_m, _i, out):
            tok = _extract_tokens(out)
            buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
        return _h

    handles = []
    if audio_enc is not None:
        handles.append(audio_enc.register_forward_hook(make_enc_hook(a_enc_buf)))
    if visual_enc is not None:
        handles.append(visual_enc.register_forward_hook(make_enc_hook(v_enc_buf)))
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(make_pre_hook(L)))
    # NOTE: intervention state should be cleared during this pass
    IV.clear_intervention()
    asd_on = bool(getattr(_RUNTIME_ARGS, "asd", False))
    try:
        with torch.inference_mode():
            out = model.thinker(
                **inputs, use_audio_in_video=use_aiv,
                output_attentions=asd_on, return_dict=True,
                use_cache=False)
    finally:
        for h in handles: h.remove()

    # p_llm per layer
    p_llm = np.zeros((n_layers, S), dtype=bool)
    for L in range(n_layers):
        if per_layer_h[L] is None: continue
        h_L = per_layer_h[L].float()
        rms = torch.sqrt(h_L.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_L / rms).abs()
        p_llm[L] = (normed_abs[:, d_sink_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
    # p_prop
    p_prop = np.zeros(S, dtype=bool)
    if a_enc_buf and len(audio_pos) > 0:
        a_aligned = _align_norms(np.concatenate(a_enc_buf), len(audio_pos))
        if a_aligned is not None:
            p_prop[audio_pos] = a_aligned > TAU_PROP
    if v_enc_buf and len(video_pos) > 0:
        v_aligned = _align_norms(np.concatenate(v_enc_buf), len(video_pos))
        if v_aligned is not None:
            p_prop[video_pos] = v_aligned > TAU_PROP

    # modality mask
    in_audio = np.zeros(S, dtype=bool); in_audio[audio_pos] = True
    in_video = np.zeros(S, dtype=bool); in_video[video_pos] = True
    if routed_modality == "AUDIO":
        modality_mask = in_audio
    elif routed_modality == "VISUAL":
        modality_mask = in_video
    else:  # AV
        modality_mask = in_audio | in_video

    # Build per-layer sink key masks
    # Default: LLM-emerged only = p_llm AND NOT p_prop.
    # If --sink_mask=all: use ANY sink = p_llm OR p_prop (wider).
    # If --sink_mask=prop: use only propagated = p_prop (narrower).
    sink_mask_mode = getattr(_RUNTIME_ARGS, "sink_mask", "llm_emerged")
    random_mask = bool(getattr(_RUNTIME_ARGS, "random_mask", False))
    include_text_sinks = bool(getattr(
        _RUNTIME_ARGS, "include_text_sinks", False))
    rng = np.random.default_rng(0)  # deterministic per call
    layer_to_key_mask = {}
    for L in range(n_layers):
        if sink_mask_mode == "all":
            sinks_L = (p_llm[L] | p_prop)
        elif sink_mask_mode == "prop":
            sinks_L = p_prop.copy()  # static, but same shape per L
        else:
            sinks_L = p_llm[L] & ~p_prop
        # AND with modality unless we're keeping text/system sinks too.
        if not include_text_sinks:
            sinks_L = sinks_L & modality_mask
        # Probe A — random control: replace sinks_L with a same-size random
        # subset of the modality_mask, so the intervention boosts a random
        # sparse subset of equal cardinality. Tests whether the gain is
        # specific to sink positions or just to "any sparse subset".
        if random_mask and sinks_L.any():
            n_sinks = int(sinks_L.sum())
            mod_positions = np.where(modality_mask)[0]
            if len(mod_positions) >= n_sinks:
                chosen = rng.choice(mod_positions, size=n_sinks, replace=False)
                sinks_L = np.zeros_like(sinks_L)
                sinks_L[chosen] = True
        # --modality_complement: replace sinks_L with the COMPLEMENT inside
        # the modality span (= modality non-sinks). Tests "are sinks special,
        # or is what matters just concentrating attention on routed modality"?
        if bool(getattr(_RUNTIME_ARGS, "modality_complement", False)):
            sinks_L = modality_mask & ~sinks_L
        if sinks_L.any():
            layer_to_key_mask[L] = torch.from_numpy(sinks_L)

    # ASD — restrict to cross-modal sinks (low |MDS|) using attention
    # collected from the forward pass above.
    if asd_on and "attentions" in dir(out) and out.attentions is not None:
        all_sinks = sorted({int(p) for L, m in layer_to_key_mask.items()
                              for p in np.where(m.cpu().numpy())[0]})
        target_layers = list(range(18, min(26, n_layers)))
        mds, adaptive_w = _compute_asd_mds(
            out.attentions, audio_pos.tolist(), video_pos.tolist(),
            all_sinks, target_layers)
        cm_thresh = float(getattr(_RUNTIME_ARGS, "asd_mds_threshold", 0.3))
        cross_modal = {s for s, m in mds.items() if m < cm_thresh}
        # Restrict each layer's mask to cross-modal sinks
        for L, m in list(layer_to_key_mask.items()):
            arr = m.cpu().numpy()
            keep = np.zeros_like(arr)
            for p in cross_modal:
                if p < len(arr) and arr[p]:
                    keep[p] = True
            if keep.any():
                layer_to_key_mask[L] = torch.from_numpy(keep)
            else:
                layer_to_key_mask.pop(L)
        # Store adaptive weight for the intervention to read
        _RUNTIME_STATE["asd_adaptive_weight"] = float(adaptive_w)
    torch.cuda.empty_cache()
    return layer_to_key_mask, S


# Set by main() to make args visible to compute_per_layer_sink_masks
_RUNTIME_ARGS = type("A", (), dict(sink_mask="llm_emerged"))()
_RUNTIME_STATE: dict = {}


# ---------------------------------------------------------------------
# Generation with intervention active
# ---------------------------------------------------------------------

def generate_with_intervention(model, processor, conv, use_aiv,
                                 max_new_tokens: int = 8) -> str:
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_aiv)
    text = processor.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False)
    if isinstance(text, list): text = text[0]
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True,
                        use_audio_in_video=use_aiv)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        text_ids = model.generate(
            **inputs, use_audio_in_video=use_aiv, return_audio=False,
            do_sample=False, max_new_tokens=max_new_tokens)
    gen_ids = text_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def _compute_modality_positions(model, processor, conv):
    """Quick re-tokenization to find audio / video token positions in input."""
    from qwen_omni_utils import process_mm_info
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    text = processor.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False)
    if isinstance(text, list): text = text[0]
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True,
                        use_audio_in_video=True)
    ids = inputs["input_ids"][0].cpu().numpy()
    thinker_cfg = model.thinker.config
    if not hasattr(thinker_cfg, "audio_token_index"):
        thinker_cfg = getattr(thinker_cfg, "text_config", thinker_cfg)
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    ap = torch.tensor(np.where(ids == a_id)[0], dtype=torch.long)
    vp = torch.tensor(np.where(ids == v_id)[0], dtype=torch.long)
    return ap, vp


def run_intervention(args, model, processor, layers, audio_enc, visual_enc,
                      eps_norm, d_sink_t, head_sets, split_df, router_df):
    routed = dict(zip(router_df.question_id, router_df.predicted))
    # Per-clip router softmax probabilities (for MAD-soft / Season-JSD gamma
    # weighting). Defaults to one-hot on the predicted class if missing.
    router_probs = {}
    if {"p_audio", "p_visual", "p_av"}.issubset(router_df.columns):
        for _, rr in router_df.iterrows():
            router_probs[rr["question_id"]] = (
                float(rr["p_audio"]), float(rr["p_visual"]), float(rr["p_av"]))

    sched_shapes = None
    if args.gamma_schedule_npz:
        _sz = np.load(args.gamma_schedule_npz)
        sched_shapes = (_sz["shape_a"], _sz["shape_v"], _sz["shape_av"])
        print(f"[schedule] loaded {args.gamma_schedule_npz}, g_base={args.g_base}, "
              f"nLayers={len(_sz['shape_a'])}", flush=True)

    rows = []
    failures = 0
    t0 = time.time()
    for _, r in tqdm(split_df.iterrows(), total=len(split_df),
                       desc=f"interv-{args.tag}"):
        vid = str(r["video_id"]).zfill(5)
        vp = Path(args.video_dir) / f"{vid}.mp4"
        if not vp.exists():
            failures += 1; continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        routed_mod = routed.get(r["question_id"], None) or \
                       TASK_TO_GT_MODALITY[r["task"]]  # fallback to gt
        if args.variant == "none" and args.temperature_flatten is not None:
            # KILL-SWITCH 2 — uniform pre-softmax temperature flattening.
            IV.set_intervention(dict(
                mode="temperature_flatten",
                gamma=float(args.temperature_flatten)))
            try:
                out = generate_with_intervention(model, processor, conv,
                                                   use_aiv=True)
            finally:
                IV.clear_intervention()
        elif args.variant == "none":
            # Baseline replay (no intervention). Let any model error
            # propagate — OOMs etc. must be visible, not swallowed.
            IV.clear_intervention()
            out = generate_with_intervention(model, processor, conv,
                                               use_aiv=True)
        else:
            # Pass 1: sink masks (errors propagate)
            layer_to_key_mask, S = compute_per_layer_sink_masks(
                model, processor, conv, True,
                layers, audio_enc, visual_enc, eps_norm, d_sink_t,
                routed_mod)
            heads_list = heads_for_variant_routing(head_sets, args.variant,
                                                      routed_mod)
            # NEW ARM — override with top-N most-negative-Δ (grounding) heads
            # from the categorize_exp_2axis heads.csv (score_min < 0).
            if args.grounding_heads_top_n is not None and args.grounding_heads_top_n > 0:
                hdf = pd.read_csv(args.heads_csv)
                hdf["score_min"] = hdf[["score_A", "score_V"]].min(axis=1)
                top = hdf.nsmallest(args.grounding_heads_top_n, "score_min")
                heads_list = [(int(L), int(h))
                                for L, h in zip(top["layer"], top["head"])]
            # CONTROL 1 — random-head baseline: replace the candidate set
            # with a uniformly-random subset of all heads. --random_n_heads
            # overrides the size (defaults to len(heads_list)).
            if args.random_heads_seed is not None:
                rng_local = np.random.default_rng(args.random_heads_seed)
                n_layers_tot = len(layers)
                n_heads_tot = 28  # qwen2.5-omni-7b
                all_pairs = np.array(
                    [(L, h) for L in range(n_layers_tot)
                     for h in range(n_heads_tot)])
                k = (args.random_n_heads if args.random_n_heads is not None
                     else len(heads_list))
                k = min(int(k), len(all_pairs))
                idx = rng_local.choice(len(all_pairs), size=k, replace=False)
                heads_list = [tuple(all_pairs[i].tolist()) for i in idx]
            # Inert subset: take the first N Inert heads from heads.csv
            # sorted by (layer, head). Reproducible.
            if args.inert_n_first is not None and args.inert_n_first > 0:
                hdf = pd.read_csv(args.heads_csv)
                inert_only = hdf[hdf["category"] == "Inert"].sort_values(
                    ["layer", "head"]).head(args.inert_n_first)
                heads_list = [(int(L), int(h))
                                for L, h in zip(inert_only["layer"],
                                                  inert_only["head"])]
            l2h = heads_by_layer(heads_list, args.layer_band)
            # Per-modality gamma override: if --gamma_a/_v/_av set,
            # use them based on the routed modality.
            effective_gamma = args.gamma
            per_mod = {"AUDIO": args.gamma_a, "VISUAL": args.gamma_v,
                         "AV": args.gamma_av}
            if per_mod.get(routed_mod) is not None:
                effective_gamma = per_mod[routed_mod]
            # MAD-style soft γ: γ_clip = Σ_M  g_M × p_M(router)
            # where g_M = per-mod γ override (or base γ) and p_M is the
            # router's softmax probability. Single hyper-param γ when no
            # per-mod overrides are set.
            if args.mad_soft and r["question_id"] in router_probs:
                pa, pv, pav = router_probs[r["question_id"]]
                g_a = args.gamma_a if args.gamma_a is not None else args.gamma
                g_v = args.gamma_v if args.gamma_v is not None else args.gamma
                g_av = args.gamma_av if args.gamma_av is not None else args.gamma
                effective_gamma = g_a * pa + g_v * pv + g_av * pav
            # ASD adaptive γ — scale γ by cross-modal attention share, clamped.
            if args.asd:
                aw = _RUNTIME_STATE.get("asd_adaptive_weight", 1.0)
                effective_gamma = min(
                    args.asd_strength_clamp, effective_gamma * aw)
            if "sink_to_nonsink_redistribute" in args.variant:
                mode = "sink_to_nonsink_redistribute"
            elif "bos_redistribute" in args.variant:
                mode = "bos_redistribute"
            elif "value_zero" in args.variant:
                mode = "value_zero"
            elif "boost" in args.variant:
                mode = "boost"
            else:
                mode = "suppress"
            iv_cfg = dict(
                layer_to_heads=l2h, layer_to_key_mask=layer_to_key_mask,
                mode=mode, gamma=effective_gamma)
            if mode == "sink_to_nonsink_redistribute":
                # If --route_weighted: distribute the moved sink mass into
                # audio/video non-sinks weighted by router shares.
                if args.route_weighted:
                    pa = pv = pav = 0.0
                    if r["question_id"] in router_probs:
                        pa, pv, pav = router_probs[r["question_id"]]
                    else:
                        if routed_mod == "AUDIO": pa = 1.0
                        elif routed_mod == "VISUAL": pv = 1.0
                        else: pav = 1.0
                    ap, vp = _compute_modality_positions(model, processor, conv)
                    iv_cfg.update(
                        route_weighted=True,
                        audio_positions=ap, visual_positions=vp,
                        p_audio=pa, p_visual=pv, p_av=pav,
                        alpha_av=float(args.alpha_av))
            if mode == "bos_redistribute":
                # Pass router probs + modality positions so the patched
                # attention can move BOS mass into the modality spans.
                pa = pv = pav = 0.0
                if r["question_id"] in router_probs:
                    pa, pv, pav = router_probs[r["question_id"]]
                else:
                    # Fallback: one-hot on routed_mod.
                    if routed_mod == "AUDIO": pa = 1.0
                    elif routed_mod == "VISUAL": pv = 1.0
                    else: pav = 1.0
                # Modality positions = audio/video token slots in the prompt.
                # Rebuild from the conversation since we don't have the
                # processor's input_ids handy here; compute inside the
                # attention forward via cached layer-input would be cleaner,
                # but we precompute once via a quick processor call.
                ap, vp = _compute_modality_positions(model, processor, conv)
                iv_cfg.update(
                    audio_positions=ap, visual_positions=vp,
                    p_audio=pa, p_visual=pv, p_av=pav,
                    bos_pos=int(args.bos_pos),
                    alpha_av=float(args.alpha_av))
            # Per-layer scheduled gamma (a/v/av): mix the 3 modality shapes
            # by this clip's router probs, scale by g_base.
            if sched_shapes is not None:
                sa, sv, sav = sched_shapes
                if r["question_id"] in router_probs:
                    pa, pv, pav = router_probs[r["question_id"]]
                else:
                    pa = pv = pav = 0.0
                    if routed_mod == "AUDIO": pa = 1.0
                    elif routed_mod == "VISUAL": pv = 1.0
                    else: pav = 1.0
                iv_cfg["gamma_schedule"] = (
                    args.g_base * (sa * pa + sv * pv + sav * pav)).tolist()
            IV.set_intervention(iv_cfg)
            try:
                out = generate_with_intervention(model, processor, conv,
                                                   use_aiv=True)
            finally:
                IV.clear_intervention()
        pred = parse_yes_no(out)
        rows.append(dict(question_id=r["question_id"], task=r["task"],
                          label=r["label"], routed=routed_mod,
                          generated=out[:120], predicted=pred,
                          correct=int(pred == r["label"])))
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    fname = f"interv_{args.tag}_{args.split.upper()}.csv"
    df.to_csv(out_dir / fname, index=False)
    overall = df.correct.mean() * 100
    per_task = df.groupby("task")["correct"].mean() * 100
    return df, overall, per_task, dt, failures


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model + patching attention ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    eps_norm = _thinker_rms_eps(model)
    audio_enc, visual_enc = _resolve_encoders(model)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    IV.patch_qwen_attention(model)
    IV.clear_intervention()

    head_sets = load_head_sets(args.heads_csv)
    print("Head-set sizes:")
    for k, v in head_sets.items():
        print(f"  {k:<20s} {len(v):4d}")

    split_df = pd.read_csv(args.split_csv, dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    if args.split.upper() == "DEV":
        df = split_df[split_df.split == "DEV"].reset_index(drop=True)
    elif args.split.upper() == "HELDOUT":
        df = split_df[split_df.split == "HELDOUT"].reset_index(drop=True)
    elif args.split.upper() == "TEST":
        df = split_df[split_df.split != "DEV"].reset_index(drop=True)
    elif args.split.upper() == "FULL":
        df = split_df.reset_index(drop=True)
    else:
        raise SystemExit(f"unknown split {args.split}")
    print(f"Eval split={args.split} n={len(df)}")

    # Router predictions
    router_csv = Path(args.router_csv)
    if not router_csv.exists():
        raise SystemExit(f"missing {router_csv}")
    router_df = pd.read_csv(router_csv)
    if args.use_gt_routing:
        router_df["predicted"] = router_df.task.map(TASK_TO_GT_MODALITY)
    out_df, overall, per_task, dt, fail = run_intervention(
        args, model, processor, layers, audio_enc, visual_enc,
        eps_norm, d_sink_t, head_sets, df, router_df)
    print(f"\n=== Result: tag={args.tag}, split={args.split} ===")
    print(f"  n={len(out_df)}, failures={fail}, elapsed={dt/60:.1f} min")
    print(f"  overall accuracy = {overall:.2f}%")
    for t, a in per_task.items():
        print(f"    {t:<40s} {a:5.2f}%")

    # Append to method log
    log_path = out_dir / "method_log.md"
    if not log_path.exists():
        log_path.write_text(
            "# Stage 5 exploration log\n\n"
            "Format per entry: title, intuition, config, DEV/TEST result,\n"
            "and per-task breakdown.\n\n"
            "Baselines: DEV 73.0%, Full AVHBench 75.22% (VDAH 81.14%, "
            "ADVH 81.51%, AV-Matching 64.18%).\n"
            "Target: ≥3% absolute over baseline on DEV + holds on TEST.\n\n"
        )
    with log_path.open("a") as f:
        f.write("---\n\n")
        f.write(f"## {args.tag}\n\n")
        if args.note:
            f.write(f"**Method / intuition:** {args.note}\n\n")
        f.write(f"**Config:** variant=`{args.variant}` "
                  f"gamma=`{args.gamma}` "
                  f"routing=`{'gt' if args.use_gt_routing else 'predicted'}` "
                  f"split=`{args.split}` n={len(out_df)}\n\n")
        f.write(f"**Result:** overall **{overall:.2f}%** "
                  f"(baseline {73.0 if args.split.upper()=='DEV' else 75.22:.2f}%, "
                  f"Δ = **{overall - (73.0 if args.split.upper()=='DEV' else 75.22):+.2f}%**)\n\n")
        f.write("| task | n | accuracy |\n|---|---:|---:|\n")
        for t, sub in out_df.groupby("task"):
            f.write(f"| {t} | {len(sub)} | {sub.correct.mean()*100:.2f}% |\n")
        f.write(f"\nElapsed: {dt/60:.1f} min, failures: {fail}\n\n")
    print(f"  -> appended to {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split_csv", default=str(DEFAULT_OUT / "split.csv"))
    p.add_argument("--router_csv",
                   default=str(DEFAULT_OUT / "router_v2_dev.csv"))
    p.add_argument("--heads_csv",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"))
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--split", default="DEV",
                   choices=["DEV", "HELDOUT", "TEST", "FULL"])
    p.add_argument("--variant", required=True,
                   choices=["none",
                              "suppress_halluc_sink",
                              "suppress_inert_sink",
                              "boost_halluc_content",
                              "boost_inert_content",
                              "boost_all_halluc_content",
                              "boost_inert_and_halluc_content",
                              "boost_all_heads_content",
                              "suppress_all_heads_sink",
                              "suppress_inert_and_halluc_sink",
                              "value_zero_inert_sink",
                              "value_zero_halluc_sink",
                              "value_zero_all_heads_sink",
                              "bos_redistribute_inert",
                              "bos_redistribute_halluc",
                              "bos_redistribute_all_heads",
                              "sink_to_nonsink_redistribute_inert",
                              "sink_to_nonsink_redistribute_halluc",
                              "sink_to_nonsink_redistribute_all_heads"])
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--use_gt_routing", action="store_true",
                   help="Use AVHBench gt task label instead of router output. "
                         "Useful for isolating intervention effect from router.")
    p.add_argument("--tag", required=True,
                   help="Short identifier for this config (used in filenames "
                         "and the log).")
    p.add_argument("--note", default="",
                   help="Method + intuition note appended to method_log.md.")
    p.add_argument("--layer_band", default="full",
                   choices=["full", "early", "mid", "late"],
                   help="Restrict intervention to a layer band: "
                         "early(L0-9), mid(L10-18), late(L19-27), full(all).")
    p.add_argument("--sink_mask", default="llm_emerged",
                   choices=["llm_emerged", "all", "prop"],
                   help="Sink mask definition: llm_emerged (p_llm AND NOT p_prop, default), "
                         "all (p_llm OR p_prop, widest), prop (p_prop only).")
    p.add_argument("--gamma_a", type=float, default=None,
                   help="Per-modality γ override for AUDIO-routed clips. "
                         "Falls back to --gamma if unset.")
    p.add_argument("--gamma_v", type=float, default=None,
                   help="Per-modality γ override for VISUAL-routed clips.")
    p.add_argument("--gamma_av", type=float, default=None,
                   help="Per-modality γ override for AV-routed clips.")
    p.add_argument("--gamma_schedule_npz", type=str, default=None,
                   help="npz with shape_a/shape_v/shape_av (len=nLayers): per-layer "
                        "gamma schedule. Per clip per layer gamma = g_base * "
                        "(shape_a*p_a + shape_v*p_v + shape_av*p_av) via router probs.")
    p.add_argument("--g_base", type=float, default=3.0,
                   help="Overall strength for the per-layer gamma schedule.")
    p.add_argument("--tau_sink", type=float, default=None,
                   help="Override TAU_SINK threshold (default 20.0). Lower = more sinks.")
    p.add_argument("--mad_soft", action="store_true",
                   help="MAD-style soft γ: per-clip γ = Σ_M g_M × p_M(router). "
                         "Mixes per-modality γ overrides via router softmax probs.")
    p.add_argument("--random_mask", action="store_true",
                   help="Probe A: replace each layer's sink mask with a "
                         "same-size random subset of the modality span. "
                         "Tests whether the gain is sink-specific or just "
                         "any-sparse-subset.")
    p.add_argument("--modality_complement", action="store_true",
                   help="Probe A2: replace each layer's sink mask with the "
                         "COMPLEMENT inside the modality span (modality "
                         "non-sinks). Tests sink-specificity vs "
                         "modality-concentration.")
    p.add_argument("--include_text_sinks", action="store_true",
                   help="Include sinks in text/system spans in addition to "
                         "the routed-modality sinks (don't AND with "
                         "modality_mask).")
    p.add_argument("--route_weighted", action="store_true",
                   help="In sink_to_nonsink_redistribute: distribute the "
                         "moved sink mass into audio/video non-sinks "
                         "weighted by router softmax shares (parallels "
                         "BOS-redistribute). Text non-sinks get zero share.")
    p.add_argument("--random_heads_seed", type=int, default=None,
                   help="CONTROL 1 — replace the candidate head set with a "
                         "size-matched RANDOM subset of all heads. If set, "
                         "drives the random-head kill-switch baseline.")
    p.add_argument("--random_n_heads", type=int, default=None,
                   help="Override the random-head subset size (default = "
                         "len(variant heads)).")
    p.add_argument("--inert_n_first", type=int, default=None,
                   help="Take the first N Inert heads (sorted by layer, head). "
                         "Used for the Inert-130 vs random-130 kill-switch.")
    p.add_argument("--asd", action="store_true",
                   help="ASD-inspired: restrict sink mask to cross-modal sinks "
                         "(low |MDS| via attention symmetry from audio/video "
                         "queries) and scale γ by cross-modal attention share.")
    p.add_argument("--asd_mds_threshold", type=float, default=0.3,
                   help="|MDS| threshold below which a sink counts as "
                         "cross-modal. Default 0.3.")
    p.add_argument("--asd_strength_clamp", type=float, default=0.6,
                   help="Max adaptive-γ multiplier (Chung et al. 2026 use 0.6).")
    p.add_argument("--temperature_flatten", type=float, default=None,
                   help=("KILL-SWITCH 2 — divide pre-softmax attention logits "
                         "by T > 1 for ALL heads at ALL layers. No sinks, no "
                         "head taxonomy. Tests if a mild attention flattening "
                         "reproduces the +5%% gain."))
    p.add_argument("--grounding_heads_top_n", type=int, default=None,
                   help="NEW ARM — override candidate head set with the "
                         "top-N most-negative-Δ (grounding) heads from "
                         "heads.csv. These are heads currently labeled "
                         "Inert due to SIGNED selection in 2-axis categorize "
                         "but actually protect against hallucination.")
    p.add_argument("--bos_pos", type=int, default=0,
                   help="BOS position for bos_redistribute mode (default 0).")
    p.add_argument("--alpha_av", type=float, default=0.5,
                   help="Fraction of p_av weight assigned to EACH modality "
                         "(audio_share = p_audio + alpha_av*p_av, "
                         " visual_share = p_visual + alpha_av*p_av).")
    args = p.parse_args()
    if args.tau_sink is not None:
        globals()["TAU_SINK"] = float(args.tau_sink)
        print(f"[tau_sink override] TAU_SINK = {TAU_SINK}", flush=True)
    globals()["_RUNTIME_ARGS"] = args
    main(args)
