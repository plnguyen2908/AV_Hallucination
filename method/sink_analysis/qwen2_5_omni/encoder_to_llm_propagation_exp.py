"""
encoder_to_llm_propagation_exp.py

Stage 1.2 — Encoder-to-LLM propagation analysis (PER-LAYER).

Tests whether encoder-side high-norm tokens (from Stage 1.1) receive
disproportionately high LLM attention during decoding. Reproduces
Sink-or-Not-to-Sink Figure 3A for Qwen2.5-Omni at EVERY decoder layer.

Layer dependence is a real phenomenon in Qwen2.5-Omni (ASD, arXiv 2605.10815,
shows sink behavior varies strongly across layers), so we report the full
trajectory rather than averaging it away or sampling a couple of layers.

The output of this stage determines the project's framing:
    (A) Both modalities propagate           → symmetric two-population story
    (B) Only video propagates clearly       → asymmetric (audio-deficit) story
    (C) Neither propagates clearly          → ASD refinement, not Sink-extension

Prerequisites:
    - Stage 1.1 cached encoder_norms.npz (per-clip pre-projection norms).
    - Same clips/modality used in 1.1 (token positions must align).

Procedure:
    1. For each cached clip, greedy-decode (to EOS) capturing, at EVERY decoder
       layer, the last-query-row attention averaged over heads. Average over
       (heads, generated query positions) → one attention value per (layer,
       modal token). Raw values, no renormalization (matches the paper).
    2. Align encoder norms with LLM-side modal positions (1:1 for Qwen2.5-Omni).
    3. Pool tokens across clips; bin by encoder L2 norm (fixed width 5).
    4. For every layer l in 0..n_layers-1, TWO metrics → a pattern:
         top3_ratio = mean_attn(top-3 reliable hi-norm bins) / mean_attn(norm<50)
         p95_ratio  = mean_attn(norm>p95) / mean_attn(norm<p50), token-weighted
       Pattern: Sharp (top3>=2.0 AND p95>=1.2, extreme-tail concentration) |
       Spread (p95>=1.2, broad propagation) | None (otherwise). Also reported:
       Pearson r, n_high_tokens (>100), the τ=100 prop_ratio (mean+sum), and an
       audio-only norm>2*median alternative.
    5. PRIMARY figure (figure_3a_layer_trajectory.png): per modality, thick
       top3_ratio line + thin p95_ratio line vs. layer; each layer's marker is
       colored by its pattern (Sharp=red, Spread=orange, None=gray) with the
       classification rule in the legend, and the dominant-pattern peak starred.
    6. Secondary figure (figure_3a_representative_layers.png): Figure 3A panels
       at three representative layers (early=2, middle=n//2, late=n-3).
    7. Per-layer table (stdout + propagation_per_layer.csv):
         modality | layer | pearson_r | top3_ratio | p95_ratio | pattern
    8. Framing from each modality's dominant pattern + its band:
         A        video Sharp-early + audio Sharp-early (symmetric classical)
         B        video Sharp-early + audio Spread-anywhere (asymmetric mechanism)
         B-strict video Sharp-early + audio None (audio doesn't propagate)
         C        otherwise
    9. Sanity: per-layer attention budget (system / modal / query / generated)
       printed at every 4th layer.

    --from_csv recomputes 4-8 and regenerates figures from an existing
    propagation_layers.csv with no model load / no GPU.

Outputs (--output_dir, default results/qwen2_5_omni/sink_analysis/stage1_2_propagation):
    figure_3a_layer_trajectory.png       PRIMARY (tail ratio vs layer)
    figure_3a_representative_layers.png   secondary (Fig 3A at 3 layers)
    propagation_layers.csv                per-layer per-bin stats (all layers)
    propagation_per_layer.csv             per-(modality, layer) verdict metrics
    propagation_decision.txt              peak verdicts + framing recommendation

Stop after writing these. Do not proceed to Stage 1.3 without confirmation.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import (  # noqa: E402
    build_conversation,
    find_modality_spans,
    load_omni,
    prepare_inputs,
    thinker_layers,
)


SEED = 42
# Decoding stops at EOS (natural caption length, typically tens of tokens, up to
# ~200 observed). This is only a safety ceiling so a pathological clip can't run
# away — it does NOT normally bind. Passed as thinker_max_new_tokens because the
# composite generate() shadows the generic max_new_tokens with that kwarg.
MAX_GEN_TOKENS = 512
BIN_WIDTH = 5.0
RELIABLE_MIN_TOKENS = 10        # reliability threshold for a bin
LOW_NORM_PCT = 50.0            # low-norm regime = below this percentile of norms
HIGH_NORM_PCT = 95.0          # high-norm regime = above this percentile (+ reliable)
LOW50_NORM = 50.0             # low-norm regime: norm < 50 (denominator)
TAU_HIGH = 100.0              # high-norm regime: norm > 100 (Sink-or-Not-to-Sink τ)
TAIL_TOPN = 3                 # # of highest-norm reliable bins for the tail metric
# audio bimodal characterization windows
LEFT_NORM_HI = 10.0           # leftmost / positional-sink bin: norm in [0, 10)
SEC_NORM_LO, SEC_NORM_HI = 60.0, 100.0   # secondary peak window
HITAIL_NORM = 120.0           # high-norm tail: norm > 120
# Two-metric pattern classification thresholds:
#   Sharp  := top3_ratio >= 2.0 AND p95_ratio >= 1.2  (extreme-tail concentration)
#   Spread := p95_ratio >= 1.2                         (broad high-norm propagation)
#   None   := otherwise
TOP3_SHARP = 2.0
P95_MIN = 1.2
PATTERN_COLOR = {"Sharp": "#d62728", "Spread": "#ff7f0e", "None": "#999999"}
# Representative layers for the secondary Figure 3A (early / middle / late).
REP_EARLY = 2
REP_LATE_OFFSET = 3           # late = n_layers - REP_LATE_OFFSET
PROMPT_BY_MODAL = {
    "a": "Describe what you hear in detail.",
    "v": "Describe what you see in detail.",
}
COLOR_BAR = "#9467bd"           # violet bars (attention)
COLOR_LINE = "#ff7f0e"          # orange line (token count)
_MODAL_COLOR = {"audio": "#1f77b4", "video": "#d62728"}

# ----------------------------------------------------------------------
# Stage 1.2 — LLM-emerged sink (P_llm) track, merged in.
# Encoder-norm pipeline above asks "do encoder-high-norm tokens get more
# LLM attention?" (P_prop, the propagated sinks). This track asks the
# DUAL question "do D_sink-activated LLM tokens get more attention?"
# (P_llm, the LLM-emerged sinks). The two populations are mostly disjoint
# in Qwen2.5-Omni (Stage 1.3), so the answers can differ.
#
# Sink criterion identical to Stage 2.1/2.2/2.4: pure RMSNorm (no learned
# weight), max over D_sink of |RMSNorm(x)[d]| >= τ.
D_SINK_LLM = [458, 2570]
TAU_SINK_LLM = 20.0
# Layers reported in the L2/L21 side-by-side panel and decision text.
COMPARE_LAYERS_LLM = [2, 21]
# Verdict thresholds for sink_attn / nonsink_attn ratio (mirrors the
# encoder-norm tail-ratio cutoffs at 1.5 / 1.0).
LLM_RATIO_PRESENT = 1.5
LLM_RATIO_AMBIG = 1.0


# --------------------------------------------------------------------------
# Cache loading + alignment + span finding
# --------------------------------------------------------------------------

def load_cache(npz_path: Path):
    """Returns {modality: list[(name, norms)]}."""
    if not npz_path.exists():
        raise SystemExit(f"Cache not found: {npz_path}. Run Stage 1.1 first.")
    d = np.load(npz_path, allow_pickle=True)
    out: dict = {}
    for modality in ("audio", "video"):
        if f"{modality}_flat" not in d.files:
            continue
        flat = d[f"{modality}_flat"]
        offs = d[f"{modality}_offsets"]
        names = list(d[f"{modality}_names"])
        out[modality] = [
            (names[i], flat[offs[i]:offs[i + 1]]) for i in range(len(names))
        ]
    return out


def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_start_token_id"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def align_norms(encoder_norms: np.ndarray, n_llm: int):
    """Chunked-mean (or repeat) so encoder norms have len == n_llm."""
    n_enc = len(encoder_norms)
    if n_enc == n_llm:
        return encoder_norms, "equal"
    if n_enc > n_llm and n_enc % n_llm == 0:
        k = n_enc // n_llm
        return encoder_norms.reshape(n_llm, k).mean(axis=1), f"down(x{k})"
    if n_llm > n_enc and n_llm % n_enc == 0:
        k = n_llm // n_enc
        return np.repeat(encoder_norms, k), f"up(x{k})"
    return None, f"mismatch({n_enc}->{n_llm})"


def find_all_spans(input_ids, tokenizer, thinker_cfg) -> dict:
    """Return {'system': (s,e), 'audio': (s,e)|absent, 'video': (s,e)|absent,
    'query': (s,e)} — same logic as Stage 0.2 bookkeeping."""
    ids = input_ids[0].tolist()
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_starts = [i for i, t in enumerate(ids) if t == im_start]
    im_ends = [i for i, t in enumerate(ids) if t == im_end]

    regions: dict = {}
    if im_starts and im_ends:
        sys_s = im_starts[0]
        sys_e = next((e for e in im_ends if e > sys_s), None)
        if sys_e is not None:
            regions["system"] = (sys_s, sys_e + 1)

    modal = find_modality_spans(input_ids, thinker_cfg)
    if modal["audio"][1] > 0:
        regions["audio"] = modal["audio"]
    if modal["video"][1] > 0:
        regions["video"] = modal["video"]

    if len(im_starts) >= 2:
        user_s = im_starts[1]
        user_e = next((e for e in im_ends if e > user_s), None)
        if user_e is not None:
            modal_end = user_s + 3  # rough header skip
            for k in ("audio", "video"):
                if k in regions and user_s < regions[k][1] < user_e:
                    modal_end = max(modal_end, regions[k][1] + 1)
            regions["query"] = (modal_end, user_e)
    return regions


# --------------------------------------------------------------------------
# Per-clip processing — single decode, ALL layers (per-layer attention + budget)
# --------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors. These are NOT a per-clip data
    problem — they mean the run is over-subscribing GPU memory — so they must
    abort the whole run rather than be silently tallied as a skipped clip."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def process_clip(
    model, processor, clip_path: Path, modal_type: str,
    thinker_cfg, tokenizer, max_gen_tokens: int = MAX_GEN_TOKENS,
):
    """For one clip, capture attention to every modal token from each generated
    query position (decode runs to EOS, capped at max_gen_tokens), averaged
    across heads, at EVERY decoder layer. Returns:
        {
            'perlayer_attn': (n_layers, N_llm)  mean over (heads, gen steps),
            'budget_layers': {span: (n_layers,)} per-layer attention budget,
            'N_llm': int, 'n_layers': int,
        },
        spans dict, error str or None
    """
    conv = build_conversation(
        str(clip_path), PROMPT_BY_MODAL[modal_type], modal_type
    )
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, modal_type, model.device, model.dtype
        )
    except Exception as e:
        if _is_oom(e):
            print(f"\n[FATAL] CUDA OOM preparing inputs for {clip_path.name}.",
                  file=sys.stderr)
            raise
        return None, None, f"prep:{e}"

    prompt_len = inputs["input_ids"].shape[1]
    spans = find_all_spans(inputs["input_ids"], tokenizer, thinker_cfg)
    mod_key = "audio" if modal_type == "a" else "video"
    if mod_key not in spans:
        return None, spans, f"no_{mod_key}_span"
    mod_start, mod_end = spans[mod_key]

    # ---- Hook-based attention capture (memory-safe) -----------------------
    # The naive approach (output_attentions=True + return_dict_in_generate=True)
    # makes the model RETAIN the full (heads x q_len x kv_len) probability
    # matrix for EVERY decoder layer, all decode steps, on-device until generate
    # returns — the step-0 prompt forward alone is O(n_layers*n_heads*S^2) in
    # fp32 (>10 GB for long video), which OOMs.
    #
    # Instead: output_attentions=True still makes each eager attention module
    # RETURN its weight tensor (modeling_qwen2_5_omni.py:1542 nulls it
    # otherwise), but we hook every thinker decoder layer's self_attn. The hook
    # extracts ONLY the last-query row (mean over heads) to CPU, then returns a
    # modified output with attn_weights replaced by None — so the model never
    # accumulates the full per-layer matrices. Peak GPU memory drops to ONE
    # transient (1, H, q, kv) block at a time. An OOM that still occurs is a
    # config/memory problem, not a bad clip, so _is_oom re-raises to abort.
    layers = thinker_layers(model)
    n_layers = len(layers)
    captured: list = []          # per forward: list[n_layers] of (kv,) CPU arrays

    # --- LLM-emerged sink mask (P_llm) capture, fires once at step 0 ---
    # On each decoder layer's main forward output: at the prompt forward
    # (seq_len > 1), compute is_sink for the modal-token slice via
    # D_SINK_LLM + τ. Generation steps (seq_len = 1) are no-op.
    text_cfg = getattr(model.thinker.config, "text_config",
                        model.thinker.config)
    rms_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-6))
    d_sink_t = torch.tensor(D_SINK_LLM, dtype=torch.long)
    is_sink_per_layer: list = [None] * n_layers

    def _make_sink_hook(layer_idx: int):
        def _h(_module, _inp, out):
            if is_sink_per_layer[layer_idx] is not None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] <= 1:
                return out
            x = hs[0, mod_start:mod_end].float()              # (n_modal, H)
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
            normed_abs = (x / rms).abs()
            d_t = d_sink_t.to(x.device)
            sink_act = normed_abs[:, d_t].amax(dim=-1)        # (n_modal,)
            is_sink_per_layer[layer_idx] = (sink_act >= TAU_SINK_LLM).cpu().numpy()
            return out
        return _h

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            aw = out[1]                                       # (1, H, q, kv)
            row = aw[0, :, -1, :].float().mean(dim=0).cpu().numpy()  # (kv,)
            if layer_idx == 0:                                # new forward step
                captured.append([None] * n_layers)
            if captured:
                captured[-1][layer_idx] = row
            # Drop the heavy tensor so nothing downstream retains it.
            return (out[0], None) + tuple(out[2:])
        return _hook

    handles = [
        layer.self_attn.register_forward_hook(_make_hook(i))
        for i, layer in enumerate(layers)
    ]
    sink_handles = [
        layer.register_forward_hook(_make_sink_hook(i))
        for i, layer in enumerate(layers)
    ]
    try:
        with torch.inference_mode():
            model.generate(
                **inputs,
                use_audio_in_video=use_aiv,
                return_audio=False,
                do_sample=False,
                # Composite generate() shadows the generic max_new_tokens with
                # thinker_max_new_tokens, so the cap must be set here. Decoding
                # stops at EOS first; this is only a runaway ceiling.
                thinker_max_new_tokens=max_gen_tokens,
                output_attentions=True,      # makes the modules return weights
                return_dict_in_generate=True,
            )
    except Exception as e:
        if _is_oom(e):
            print(
                f"\n[FATAL] CUDA OOM generating attentions for {clip_path.name} "
                f"(prompt_len={prompt_len}, modal_tokens={mod_end - mod_start}). "
                f"Even with hook-based capture, one layer's (H x S x S) block "
                f"exceeds GPU memory — shorten the sequence (fps/max_pixels) or "
                f"spread layers across more GPUs; not a bad clip.",
                file=sys.stderr,
            )
            raise
        return None, spans, f"generate:{e}"
    finally:
        for h in handles:
            h.remove()
        for h in sink_handles:
            h.remove()

    if not captured:
        torch.cuda.empty_cache()
        return None, spans, "no_attentions"

    n_steps = len(captured)
    n_llm = mod_end - mod_start
    # Accumulate per-layer modal attention + per-layer attention budget, both
    # summed over generation steps.
    perlayer_modal_sum = np.zeros((n_layers, n_llm), dtype=np.float64)
    budget_layers = {sp: np.zeros(n_layers, dtype=np.float64)
                     for sp in ("system", "modal", "query", "generated")}
    span_items = (("system", "system"), ("modal", mod_key), ("query", "query"))

    for s_idx, step_rows in enumerate(captured):
        # step_rows: per-layer last-query attention rows (kv grows each step).
        if any(r is None for r in step_rows):
            torch.cuda.empty_cache()
            return None, spans, "incomplete_capture"
        arr = np.stack(step_rows, axis=0)                # (n_layers, kv_len)
        perlayer_modal_sum += arr[:, mod_start:mod_end]
        for span_name, span_key in span_items:
            if span_key in spans:
                s, e = spans[span_key]
                budget_layers[span_name] += arr[:, s:e].sum(axis=1)  # (n_layers,)
        gen_end = prompt_len + s_idx
        if gen_end > prompt_len:
            budget_layers["generated"] += arr[:, prompt_len:gen_end].sum(axis=1)

    perlayer_attn = perlayer_modal_sum / n_steps         # mean over gen steps
    for sp in budget_layers:
        budget_layers[sp] /= n_steps

    # Stack sink masks (n_layers, n_modal); fall back to all-False if any layer
    # didn't capture (rare; e.g. if prompt_len was unexpectedly 1).
    if all(s is not None for s in is_sink_per_layer):
        is_sink_arr = np.stack(is_sink_per_layer, axis=0)
    else:
        is_sink_arr = np.zeros((n_layers, n_llm), dtype=bool)

    torch.cuda.empty_cache()
    return {
        "perlayer_attn": perlayer_attn,
        "budget_layers": budget_layers,
        "N_llm": n_llm,
        "n_layers": n_layers,
        "is_sink_per_layer": is_sink_arr,    # (n_layers, n_modal) bool, P_llm mask
    }, spans, None


# --------------------------------------------------------------------------
# Binning + summary metrics
# --------------------------------------------------------------------------

def bin_norms_attn(
    pooled_norms: np.ndarray, pooled_attn: np.ndarray,
    n_clips: int, bin_width: float = BIN_WIDTH,
):
    """Fine-grained linear binning. Returns dict with bin arrays."""
    if pooled_norms.size == 0:
        return None
    max_norm = float(pooled_norms.max())
    n_bins = int(np.ceil((max_norm + bin_width) / bin_width))
    edges = np.arange(n_bins + 1) * bin_width
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.clip(np.digitize(pooled_norms, edges) - 1, 0, n_bins - 1)

    mean_norm = np.full(n_bins, np.nan)
    mean_attn = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_norm[b] = pooled_norms[mask].mean()
            mean_attn[b] = pooled_attn[mask].mean()
            counts[b] = int(mask.sum())
    n_per_clip = counts / max(n_clips, 1)
    return dict(
        bin_edges=edges, bin_centers=centers,
        mean_norm=mean_norm, mean_attn=mean_attn,
        bin_counts=counts, n_per_clip=n_per_clip,
        n_bins=n_bins,
    )


def _wmean_attn(mean_attn, counts, mask) -> float:
    """Token-count-weighted mean attention over the selected bins."""
    if mask is None or not np.any(mask):
        return float("nan")
    a = mean_attn[mask]
    w = counts[mask].astype(np.float64)
    ok = np.isfinite(a) & (w > 0)
    if not ok.any() or w[ok].sum() == 0:
        return float("nan")
    return float((a[ok] * w[ok]).sum() / w[ok].sum())


def _wsum_attn(mean_attn, counts, mask) -> float:
    """Total attention mass over the selected bins = Σ (mean_attn_b * count_b)
    = Σ over tokens in the regime of their attention (the 'sum' version)."""
    if mask is None or not np.any(mask):
        return float("nan")
    a = mean_attn[mask]
    w = counts[mask].astype(np.float64)
    ok = np.isfinite(a) & (w > 0)
    if not ok.any():
        return float("nan")
    return float((a[ok] * w[ok]).sum())


def _wpercentile(values, weights, q_frac) -> float:
    """Weighted percentile of `values` (q_frac in [0,1]) using the per-bin
    histogram (value = bin mean_norm, weight = bin token count). Accurate to
    the bin resolution (5 norm units), which is sufficient here."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return float("nan")
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return float("nan")
    return float(v[min(int(np.searchsorted(cw, q_frac * cw[-1])), len(v) - 1)])


def compute_summary(b: dict) -> dict:
    """Propagation metrics for one (layer's) binned distribution.

    - high_low_ratio (robust): token-weighted mean_attn over the HIGH-norm
      regime (mean_norm > p95 AND bin count >= RELIABLE_MIN_TOKENS) divided by
      the same over the LOW-norm regime (mean_norm < p50).
    - tail_ratio: top-`TAIL_TOPN` highest-norm reliable bins (token-weighted
      mean_attn) divided by token-weighted mean_attn of bins with norm < 50.
      This is the verdict driver.
    - bimodal characterization: leftmost bin (norm 0-10, positional sink?),
      secondary window (norm 60-100), high-norm tail (norm > 120).
    """
    counts = b["bin_counts"]
    mean_attn = b["mean_attn"]
    mean_norm = b["mean_norm"]
    reliable = counts >= RELIABLE_MIN_TOKENS
    n_reliable = int(reliable.sum())
    n_noisy = int(((counts > 0) & ~reliable).sum())
    populated = counts > 0

    # Pearson r: bin index vs mean_attn over reliable bins (kept as context).
    if reliable.sum() >= 2:
        idxs = np.where(reliable)[0]
        pearson_r = float(np.corrcoef(idxs, mean_attn[idxs])[0, 1])
    else:
        pearson_r = float("nan")

    # Norm percentiles from the histogram.
    p50 = _wpercentile(mean_norm[populated], counts[populated], LOW_NORM_PCT / 100.0)
    p95 = _wpercentile(mean_norm[populated], counts[populated], HIGH_NORM_PCT / 100.0)

    # Robust high/low ratio (reliable high bins only; kept for context).
    low_mask = populated & np.isfinite(mean_norm) & (mean_norm < p50)
    high_mask = (counts >= RELIABLE_MIN_TOKENS) & np.isfinite(mean_norm) & (mean_norm > p95)
    low_attn = _wmean_attn(mean_attn, counts, low_mask)
    high_attn = _wmean_attn(mean_attn, counts, high_mask)
    high_low_ratio = (
        high_attn / low_attn
        if np.isfinite(low_attn) and low_attn > 0 and np.isfinite(high_attn)
        else float("nan")
    )

    # p95_ratio (classification input): mean_attn over tokens with norm > p95 ÷
    # mean_attn over tokens with norm < p50, token-weighted. NO reliability
    # filter (token weighting handles noise). Captures BROAD high-norm propagation.
    high_pct_mask = populated & np.isfinite(mean_norm) & (mean_norm > p95)
    high_pct_attn = _wmean_attn(mean_attn, counts, high_pct_mask)
    p95_ratio = (
        high_pct_attn / low_attn
        if np.isfinite(low_attn) and low_attn > 0 and np.isfinite(high_pct_attn)
        else float("nan")
    )

    # Tail-focused ratio (legacy comparison line).
    reli_idx = np.where(reliable)[0]
    top_idx = sorted(reli_idx, key=lambda i: mean_norm[i], reverse=True)[:TAIL_TOPN]
    top_mask = np.zeros_like(counts, dtype=bool)
    top_mask[top_idx] = True
    top3_attn = _wmean_attn(mean_attn, counts, top_mask)
    low50_mask = populated & np.isfinite(mean_norm) & (mean_norm < LOW50_NORM)
    mean_low50 = _wmean_attn(mean_attn, counts, low50_mask)
    tail_ratio = (
        top3_attn / mean_low50
        if np.isfinite(mean_low50) and mean_low50 > 0 and np.isfinite(top3_attn)
        else float("nan")
    )

    # PRIMARY metric — absolute-threshold propagation ratio (Sink-or-Not τ=100):
    #   high regime = norm > 100, low regime = norm < 50, token-weighted.
    # Reported in two forms: 'mean' (per-token preference, the verdict driver)
    # and 'sum' (total attention MASS to high vs low; tiny because there are far
    # fewer high-norm tokens). _wmean_attn over a regime's bins equals the exact
    # token-level mean; _wsum_attn equals the exact token-level sum.
    high_abs_mask = populated & np.isfinite(mean_norm) & (mean_norm > TAU_HIGH)
    n_high_tokens = int(counts[high_abs_mask].sum())
    n_low_tokens = int(counts[low50_mask].sum())
    high_abs_mean = _wmean_attn(mean_attn, counts, high_abs_mask)
    prop_ratio = (
        high_abs_mean / mean_low50
        if np.isfinite(mean_low50) and mean_low50 > 0 and np.isfinite(high_abs_mean)
        else float("nan")
    )
    high_abs_sum = _wsum_attn(mean_attn, counts, high_abs_mask)
    low_abs_sum = _wsum_attn(mean_attn, counts, low50_mask)
    prop_ratio_sum = (
        high_abs_sum / low_abs_sum
        if np.isfinite(low_abs_sum) and low_abs_sum > 0 and np.isfinite(high_abs_sum)
        else float("nan")
    )
    # Modality-relative alternative (norm > 2*median high cutoff; <50 low). Audio
    # has few tokens > 100 absolute, so this characterizes it on its own scale.
    rel_cut = 2.0 * p50 if np.isfinite(p50) else float("nan")
    if np.isfinite(rel_cut):
        high_rel_mask = populated & np.isfinite(mean_norm) & (mean_norm > rel_cut)
    else:
        high_rel_mask = np.zeros_like(counts, dtype=bool)
    n_high_rel_tokens = int(counts[high_rel_mask].sum())
    high_rel_mean = _wmean_attn(mean_attn, counts, high_rel_mask)
    rel_ratio = (
        high_rel_mean / mean_low50
        if np.isfinite(mean_low50) and mean_low50 > 0 and np.isfinite(high_rel_mean)
        else float("nan")
    )

    # Bimodal characterization windows.
    left_mask = populated & np.isfinite(mean_norm) & (mean_norm < LEFT_NORM_HI)
    sec_mask = populated & np.isfinite(mean_norm) & (mean_norm >= SEC_NORM_LO) & (mean_norm < SEC_NORM_HI)
    hitail_mask = populated & np.isfinite(mean_norm) & (mean_norm > HITAIL_NORM)
    left_attn = _wmean_attn(mean_attn, counts, left_mask)
    sec_attn = _wmean_attn(mean_attn, counts, sec_mask)
    hitail_attn = _wmean_attn(mean_attn, counts, hitail_mask)

    if reliable.any():
        max_bin = int(np.argmax(np.where(reliable, mean_attn, -np.inf)))
        max_attn_bin_norm = float(mean_norm[max_bin])
    else:
        max_attn_bin_norm = float("nan")

    top3_norms = [float(mean_norm[i]) for i in top_idx]

    return dict(
        n_reliable_bins=n_reliable,
        n_noisy_bins=n_noisy,
        pearson_r=pearson_r,
        p50_norm=p50,
        p95_norm=p95,
        high_attn=high_attn,
        low_attn=low_attn,
        high_low_ratio=high_low_ratio,
        p95_ratio=p95_ratio,
        prop_ratio=prop_ratio,
        prop_ratio_sum=prop_ratio_sum,
        n_high_tokens=n_high_tokens,
        n_low_tokens=n_low_tokens,
        rel_cut=rel_cut,
        rel_ratio=rel_ratio,
        n_high_rel_tokens=n_high_rel_tokens,
        top3_attn=top3_attn,
        mean_low50=mean_low50,
        tail_ratio=tail_ratio,
        top3_high_norms=top3_norms,
        max_attn_bin_norm=max_attn_bin_norm,
        left_attn_0_10=left_attn,
        attn_60_100=sec_attn,
        attn_gt_120=hitail_attn,
    )


def classify_layer(top3_ratio: float, p95_ratio: float) -> str:
    """Two-metric pattern classification of one (modality, layer):
        Sharp  — top3_ratio >= 2.0 AND p95_ratio >= 1.2 (extreme-tail
                 concentration; classical Sink-or-Not-to-Sink)
        Spread — p95_ratio >= 1.2 (broad high-norm propagation, distributed)
        None   — otherwise (no propagation effect)."""
    has95 = np.isfinite(p95_ratio) and p95_ratio >= P95_MIN
    has_top3 = np.isfinite(top3_ratio) and top3_ratio >= TOP3_SHARP
    if has_top3 and has95:
        return "Sharp"
    if has95:
        return "Spread"
    return "None"


def _ranges_str(layers_sorted) -> str:
    """Compress a sorted layer list into a range string, e.g. [1,2,3,5,6]→'1-3,5-6'."""
    if not layers_sorted:
        return "—"
    parts, start, prev = [], layers_sorted[0], layers_sorted[0]
    for L in layers_sorted[1:]:
        if L == prev + 1:
            prev = L
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = L
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def layer_band(layer) -> str:
    """Layer-band classification of a peak. early=0-7 (encoder-propagation per
    Sink-or-Not-to-Sink), mid=8-19, late=20+ (LLM-emerged, a novel finding)."""
    if layer is None:
        return "n/a"
    if layer <= 7:
        return "early"
    if layer <= 19:
        return "mid"
    return "late"


def describe_pattern(modality: str, peak: dict) -> str:
    """One-sentence mechanistic read from the dominant pattern + its band."""
    pat, pl, band = peak["pattern"], peak["peak_layer"], peak["band"]
    shape = peak["shape"]
    M = modality.capitalize()
    if pat == "None" or pl is None:
        return f"{M}: no propagation effect at any layer (absent)."
    if pat == "Sharp":
        kind = ("consistent with Sink-or-Not-to-Sink's encoder-propagation "
                "prediction" if band == "early" else
                "but late — would suggest LLM-emerged rather than encoder-propagated "
                "sinks" if band == "late" else "at a middle layer")
        return (f"{M}: {shape} — extreme-tail concentration peaks at L{pl} "
                f"(top3={peak['peak_top3']:.2f}x, p95={peak['peak_p95']:.2f}x), {kind}.")
    # Spread
    return (f"{M}: {shape} — broad/distributed high-norm propagation (no extreme-tail "
            f"concentration), peak at L{pl} (top3={peak['peak_top3']:.2f}x, "
            f"p95={peak['peak_p95']:.2f}x).")


def framing_from_peaks(peaks: dict) -> str:
    """Framing from the per-modality dominant pattern + band."""
    v, a = peaks.get("video", {}), peaks.get("audio", {})
    v_sharp_early = (v.get("pattern") == "Sharp" and v.get("band") == "early")
    if not v_sharp_early:
        return ("(C) Video is not Sharp-early — no clean classical/asymmetric "
                "split; project becomes an ASD refinement, not a "
                "Sink-or-Not-to-Sink extension.")
    a_pat, a_band = a.get("pattern"), a.get("band")
    if a_pat == "Sharp" and a_band == "early":
        return ("(A) Symmetric classical — both modalities show early Sharp "
                "(extreme-tail) propagation; encoder-propagated sinks in both.")
    if a_pat == "Spread":
        return ("(B) Asymmetric mechanism — vision does extreme-tail register "
                "propagation (early Sharp) while audio does distributed propagation "
                "(Spread), possibly at different layers.")
    if a_pat == "None":
        return ("(B-strict) Asymmetric — vision propagates (early Sharp) but audio "
                "does not propagate at all (None).")
    return ("(C) Video Sharp-early but audio pattern (Sharp non-early) fits neither "
            "A nor B; project becomes an ASD refinement.")


# --------------------------------------------------------------------------
# Per-layer metrics assembly
# --------------------------------------------------------------------------

def per_layer_metrics(per_bin: dict, n_layers_by_modality: dict) -> dict:
    """metrics_by_modality[modality] = list (len n_layers) of metric dicts
    (compute_summary output + 'layer', 'verdict', 'verdict_str'); None for any
    missing layer."""
    out: dict = {}
    for modality in per_bin:
        nL = n_layers_by_modality.get(modality)
        if not nL:
            continue
        ms = []
        for L in range(nL):
            b = per_bin[modality].get(L)
            if b is None:
                ms.append(None)
                continue
            m = compute_summary(b)
            m["layer"] = L
            m["top3_ratio"] = m["tail_ratio"]
            m["pattern"] = classify_layer(m["top3_ratio"], m["p95_ratio"])
            ms.append(m)
        out[modality] = ms
    return out


def _first_metric(ms):
    for m in ms:
        if m is not None:
            return m
    return None


def _peak_of(ms, key):
    """(peak_value, peak_layer) for metric `key` across layers (nan-safe)."""
    cand = [(m[key], m["layer"]) for m in ms
            if m is not None and np.isfinite(m.get(key, float("nan")))]
    if not cand:
        return (float("nan"), None)
    return max(cand, key=lambda x: x[0])


def peak_per_modality(metrics_by_modality: dict) -> dict:
    """modality -> dominant pattern + peak layer + ranges + trajectory shape.

    dominant pattern = Sharp if any layer is Sharp, else Spread if any Spread,
    else None. The peak layer is the strongest layer of the dominant pattern
    (Sharp → max top3_ratio; Spread → max p95_ratio)."""
    peaks: dict = {}
    for modality, ms in metrics_by_modality.items():
        ms = [m for m in ms if m is not None]
        sharp = [m["layer"] for m in ms if m["pattern"] == "Sharp"]
        spread = [m["layer"] for m in ms if m["pattern"] == "Spread"]
        none = [m["layer"] for m in ms if m["pattern"] == "None"]

        if sharp:
            dom = "Sharp"
            peak_m = max((m for m in ms if m["pattern"] == "Sharp"),
                         key=lambda m: (m["top3_ratio"] if np.isfinite(m["top3_ratio"]) else -np.inf))
        elif spread:
            dom = "Spread"
            peak_m = max((m for m in ms if m["pattern"] == "Spread"),
                         key=lambda m: (m["p95_ratio"] if np.isfinite(m["p95_ratio"]) else -np.inf))
        else:
            dom = "None"
            peak_m = None

        if peak_m is not None:
            pl = peak_m["layer"]
            band = layer_band(pl)
            shape = f"{band}-{dom.lower()}"
            peaks[modality] = dict(
                pattern=dom, peak_layer=pl, band=band, shape=shape,
                peak_top3=peak_m["top3_ratio"], peak_p95=peak_m["p95_ratio"],
                sharp_layers=sorted(sharp), spread_layers=sorted(spread),
                none_layers=sorted(none),
            )
        else:
            peaks[modality] = dict(
                pattern="None", peak_layer=None, band="n/a", shape="absent",
                peak_top3=float("nan"), peak_p95=float("nan"),
                sharp_layers=[], spread_layers=[], none_layers=sorted(none),
            )
    return peaks


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _plot_panel(ax, b: dict, title: str):
    """Sink-or-Not Figure 3A panel: violet attn bars (left y), orange
    tokens-per-clip line (right y, log). <10-token bins hatched."""
    centers = b["bin_centers"]
    mean_attn = b["mean_attn"]
    n_per_clip = b["n_per_clip"]
    counts = b["bin_counts"]
    reliable = counts >= RELIABLE_MIN_TOKENS

    for i in range(len(centers)):
        if np.isnan(mean_attn[i]) or counts[i] == 0:
            continue
        ax.bar(
            centers[i], mean_attn[i], width=BIN_WIDTH * 0.85,
            color=COLOR_BAR, alpha=0.75 if reliable[i] else 0.4,
            hatch=None if reliable[i] else "///",
            edgecolor="black", linewidth=0.4, zorder=2,
        )
    ax.set_xlabel("Encoder L2 norm (bin center)", fontsize=11)
    ax.set_ylabel("Avg Attn for LLM Outputs", color=COLOR_BAR, fontsize=11)
    ax.tick_params(axis="y", labelcolor=COLOR_BAR)
    ax.set_xlim(left=0)

    ax_r = ax.twinx()
    valid_line = n_per_clip > 0
    ax_r.plot(
        centers[valid_line], n_per_clip[valid_line],
        color=COLOR_LINE, marker="o", markersize=4, linewidth=1.5, zorder=3,
    )
    ax_r.set_yscale("log")
    ax_r.set_ylabel("Avg # of Tokens per Clip (log)", color=COLOR_LINE, fontsize=11)
    ax_r.tick_params(axis="y", labelcolor=COLOR_LINE)

    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=1)
    ax.legend(handles=[
        Patch(facecolor=COLOR_BAR, alpha=0.75, label="Avg Attn for LLM Outputs"),
        Line2D([0], [0], marker="o", color=COLOR_LINE, linewidth=1.5,
               label="Avg # of Tokens per Clip"),
        Patch(facecolor=COLOR_BAR, alpha=0.4, hatch="///",
              label=f"bin with <{RELIABLE_MIN_TOKENS} tokens"),
    ], loc="upper right", fontsize=9)


def plot_layer_trajectory(metrics_by_modality: dict, peaks: dict, out_path: Path):
    """PRIMARY figure: one panel per modality. X = layer index. Thick line =
    top3_ratio (extreme-tail); thin line = p95_ratio (broad). Each layer's marker
    is colored by its two-metric pattern (Sharp=red, Spread=orange, None=gray) so
    the classification is visible. Threshold lines at top3=2.0 and p95=1.2; the
    dominant-pattern peak layer is starred."""
    mods = [m for m in ("audio", "video") if m in metrics_by_modality]
    if not mods:
        return
    fig, axes = plt.subplots(1, len(mods), figsize=(8 * len(mods), 5.2),
                             squeeze=False)
    for ax, modality in zip(axes[0], mods):
        ms = [m for m in metrics_by_modality[modality] if m is not None]
        layers = np.array([m["layer"] for m in ms])
        top3 = np.array([m["top3_ratio"] for m in ms], dtype=float)
        p95 = np.array([m["p95_ratio"] for m in ms], dtype=float)
        pat_colors = [PATTERN_COLOR[m["pattern"]] for m in ms]

        ax.plot(layers, top3, lw=2.4, color="#444444", zorder=2,
                label="top3_ratio (extreme-tail)  [thick]")
        ax.plot(layers, p95, lw=1.0, ls="--", color="#888888", alpha=0.8, zorder=2,
                label="p95_ratio (broad)  [thin]")
        # Pattern-colored markers on the top3 line.
        ax.scatter(layers, top3, c=pat_colors, s=42, zorder=4,
                   edgecolor="black", linewidth=0.3)
        ax.axhline(TOP3_SHARP, ls="--", color="#d62728", lw=1.0, alpha=0.7,
                   label=f"top3 Sharp threshold = {TOP3_SHARP}")
        ax.axhline(P95_MIN, ls="--", color="#ff7f0e", lw=1.0, alpha=0.7,
                   label=f"p95 propagation threshold = {P95_MIN}")
        ax.axhline(1.0, ls="-", color="lightgray", lw=0.8, zorder=1)

        pk = peaks.get(modality, {})
        pl = pk.get("peak_layer")
        if pl is not None:
            yval = pk.get("peak_top3")
            if not np.isfinite(yval):
                yval = pk.get("peak_p95", 1.0)
            ax.scatter([pl], [yval], color="black", marker="*", s=170, zorder=6)
            ax.annotate(
                f"peak L={pl}  ({pk['shape']})\n"
                f"top3={pk['peak_top3']:.2f}x  p95={pk['peak_p95']:.2f}x",
                xy=(pl, yval), xytext=(6, 8), textcoords="offset points",
                fontsize=9, fontweight="bold",
            )
        ax.set_xlabel("LLM layer index", fontsize=11)
        ax.set_ylabel("propagation ratio", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{modality.capitalize()} encoder — propagation by layer",
                     fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.4)
        # Classification-rule legend cell.
        rule_handles = [
            Line2D([0], [0], color="#444444", lw=2.4, label="top3_ratio (thick)"),
            Line2D([0], [0], color="#888888", lw=1.0, ls="--", label="p95_ratio (thin)"),
            Patch(facecolor=PATTERN_COLOR["Sharp"],
                  label="Sharp: top3≥2.0 AND p95≥1.2"),
            Patch(facecolor=PATTERN_COLOR["Spread"], label="Spread: p95≥1.2"),
            Patch(facecolor=PATTERN_COLOR["None"], label="None: otherwise"),
        ]
        ax.legend(handles=rule_handles, loc="upper right", fontsize=8,
                  title="pattern rule")
    fig.suptitle(
        "Per-layer encoder→LLM propagation trajectory "
        "(two-metric pattern classification; markers colored by pattern)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layers_grid(per_bin: dict, layers, out_path: Path):
    """SECONDARY figure: rows = representative layers, cols = modality."""
    layers = [l for l in layers]
    fig, axes = plt.subplots(len(layers), 2, figsize=(16, 6 * len(layers)),
                             squeeze=False)
    for ri, layer_idx in enumerate(layers):
        for ci, modality in enumerate(("audio", "video")):
            ax = axes[ri][ci]
            if modality not in per_bin or layer_idx not in per_bin[modality] \
                    or per_bin[modality][layer_idx] is None:
                ax.set_visible(False)
                continue
            _plot_panel(ax, per_bin[modality][layer_idx],
                        f"{modality.capitalize()} Encoder (layer {layer_idx})")
    fig.suptitle("Figure 3A at representative layers (early / middle / late)",
                 fontsize=15, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporting (shared by full-run and --from_csv paths)
# --------------------------------------------------------------------------

def representative_layers(n_layers: int) -> list:
    """Early / middle / late layer indices, clamped & de-duplicated."""
    cand = [REP_EARLY, n_layers // 2, n_layers - REP_LATE_OFFSET]
    return sorted({max(0, min(n_layers - 1, c)) for c in cand})


def _bindict_from_group(g) -> dict:
    g = g.sort_values("bin")
    return dict(
        bin_centers=g["bin_center"].to_numpy(dtype=np.float64),
        mean_norm=g["mean_norm"].to_numpy(dtype=np.float64),
        mean_attn=g["mean_attn"].to_numpy(dtype=np.float64),
        bin_counts=g["bin_count_total"].to_numpy(dtype=np.int64),
        n_per_clip=g["n_tokens_per_clip"].to_numpy(dtype=np.float64),
        n_bins=len(g),
    )


def perbin_rows(modality, layer_label, b) -> list:
    return [{
        "modality": modality, "layer": layer_label, "bin": i,
        "bin_center": b["bin_centers"][i], "mean_norm": b["mean_norm"][i],
        "n_tokens_per_clip": b["n_per_clip"][i], "mean_attn": b["mean_attn"][i],
        "bin_count_total": b["bin_counts"][i],
        "reliable": bool(b["bin_counts"][i] >= RELIABLE_MIN_TOKENS),
    } for i in range(b["n_bins"])]


def load_per_layer_from_csv(layers_csv: Path):
    """Reconstruct per_bin[modality][int] from propagation_layers.csv.
    Returns (per_bin, n_layers_by_modality)."""
    if not Path(layers_csv).exists():
        raise SystemExit(f"{layers_csv} not found. Run the full pipeline first.")
    dl = pd.read_csv(layers_csv)
    per_bin: dict = {}
    for (modality, layer), g in dl.groupby(["modality", "layer"]):
        per_bin.setdefault(modality, {})[int(layer)] = _bindict_from_group(g)
    n_layers_by_modality = {m: (max(per_bin[m]) + 1) for m in per_bin}
    return per_bin, n_layers_by_modality


def print_top_coverage(per_bin: dict, n_layers_by_modality: dict):
    """Top-5 high-norm bins by mean_norm (norm distribution is layer-independent,
    so any populated layer's bins suffice)."""
    print("\n" + "=" * 78)
    print("Top-5 high-norm bins by mean_norm (coverage check)")
    print("=" * 78)
    for modality in per_bin:
        layers = [L for L in per_bin[modality] if per_bin[modality][L] is not None]
        if not layers:
            continue
        b = per_bin[modality][layers[0]]
        valid = (b["bin_counts"] > 0) & np.isfinite(b["mean_norm"])
        order = np.argsort(np.where(valid, b["mean_norm"], -np.inf))[::-1][:5]
        print(f"\n  {modality}:")
        for vi in order:
            c = b["bin_counts"][vi]
            tag = "OK" if c >= RELIABLE_MIN_TOKENS else "noisy (<10)"
            small = "  ⚠ <5 tokens" if c < 5 else ""
            print(
                f"    bin={vi:3d}  norm={b['mean_norm'][vi]:7.2f}  "
                f"count={c:6d}  n/clip={b['n_per_clip'][vi]:6.3f}  [{tag}]{small}"
            )


def print_and_save_per_layer_table(metrics_by_modality: dict, out_dir: Path):
    """Per-layer table to stdout AND propagation_per_layer.csv."""
    def _f(v, fmt):
        return fmt.format(v) if np.isfinite(v) else "   nan  "

    print("\n" + "=" * 86)
    print("Per-layer pattern classification  (Sharp: top3>=2.0 & p95>=1.2 | "
          "Spread: p95>=1.2 | None)")
    print("=" * 86)
    print(f"{'modality':<9}{'layer':<6}{'pearson_r':<11}{'top3_ratio':<12}"
          f"{'p95_ratio':<12}{'pattern':<9}")
    print("-" * 86)
    rows = []
    for modality in ("audio", "video"):
        ms = metrics_by_modality.get(modality)
        if not ms:
            continue
        for m in ms:
            if m is None:
                continue
            print(
                f"{modality:<9}{m['layer']:<6}"
                f"{_f(m['pearson_r'], '{:+.3f} '):<11}"
                f"{_f(m['top3_ratio'], '{:.2f}x '):<12}"
                f"{_f(m['p95_ratio'], '{:.2f}x '):<12}"
                f"{m['pattern']:<9}"
            )
            rows.append({
                "modality": modality, "layer": m["layer"],
                "pearson_r": m["pearson_r"],
                "top3_ratio": m["top3_ratio"],
                "p95_ratio": m["p95_ratio"],
                "pattern": m["pattern"],
                "prop_ratio_mean": m["prop_ratio"],
                "prop_ratio_sum": m["prop_ratio_sum"],
                "rel_ratio": m["rel_ratio"],
            })
    path = out_dir / "propagation_per_layer.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")


def print_budget(budget_by_modality: dict, n_layers_by_modality: dict):
    """Attention budget (system / modal / query / generated) at every 4th layer.
    Each value is the fraction of a query row's attention landing on that span,
    averaged over generated query positions and clips."""
    print("\n" + "=" * 78)
    print("Per-layer attention budget  (% of each query row, every 4th layer)")
    print("=" * 78)
    for modality, spans in budget_by_modality.items():
        nL = n_layers_by_modality.get(modality, 0)
        print(f"\n  {modality}:")
        print(f"    {'layer':<7}{'system':>9}{'modal':>9}{'query':>9}{'generated':>11}")
        for L in range(0, nL, 4):
            def g(sp):
                a = spans.get(sp)
                return a[L] * 100 if a is not None and L < len(a) else float("nan")
            mflag = "  ⚠ modal<1%" if np.isfinite(g("modal")) and g("modal") < 1.0 else ""
            print(f"    {L:<7}{g('system'):>8.2f}%{g('modal'):>8.2f}%"
                  f"{g('query'):>8.2f}%{g('generated'):>10.2f}%{mflag}")


def decide_and_write(metrics_by_modality: dict, peaks: dict,
                     n_layers_by_modality: dict, out_dir: Path, n_clips: int):
    framing = framing_from_peaks(peaks)

    def _modality_lines(modality) -> list:
        """Reusable per-modality report lines (used for stdout and file)."""
        pk = peaks[modality]
        lines = [
            f"{modality}: dominant pattern = {pk['pattern']}  (trajectory: {pk['shape']})",
            f"  Sharp at L {_ranges_str(pk['sharp_layers'])}; "
            f"Spread at L {_ranges_str(pk['spread_layers'])}; None elsewhere",
        ]
        if pk["peak_layer"] is not None:
            lines.append(f"  peak layer L{pk['peak_layer']}: "
                         f"top3={pk['peak_top3']:.2f}x, p95={pk['peak_p95']:.2f}x")
        lines.append(f"  → {describe_pattern(modality, pk)}")
        m0 = _first_metric(metrics_by_modality.get(modality, []))
        if m0 is not None:
            lines.append(f"  regime tokens: n(norm>100)={m0['n_high_tokens']}, "
                         f"n(norm<50)={m0['n_low_tokens']}")
            if modality == "audio":
                rpr, rpl = _peak_of(metrics_by_modality[modality], "rel_ratio")
                lines.append(f"  audio-relative (norm>{m0['rel_cut']:.0f}=2*median, "
                             f"n={m0['n_high_rel_tokens']}): peak p95-style "
                             f"{rpr:.2f}x at layer {rpl}")
        return lines

    print("\n" + "=" * 78)
    print("Pattern verdicts  (Sharp=extreme-tail | Spread=broad | None;  "
          "band early 0-7 / mid 8-19 / late 20+)")
    print("=" * 78)
    for modality in ("audio", "video"):
        if modality in peaks:
            for ln in _modality_lines(modality):
                print("  " + ln)
    print(f"\n  Framing recommendation: {framing}")

    with open(out_dir / "propagation_decision.txt", "w") as f:
        f.write("Stage 1.2 — Encoder-to-LLM propagation (per-layer, two-metric)\n")
        f.write("Per (modality, layer) pattern from TWO metrics:\n")
        f.write("  top3_ratio = mean_attn(top-3 reliable hi-norm bins) / mean_attn(norm<50)\n")
        f.write("  p95_ratio  = mean_attn(norm>p95) / mean_attn(norm<p50), token-weighted\n")
        f.write("  Sharp: top3>=2.0 AND p95>=1.2 | Spread: p95>=1.2 | None: otherwise\n")
        f.write("Band: early 0-7 (encoder-propagated) | mid 8-19 | late 20+ (LLM-emerged)\n")
        f.write(f"n_clips/mod   : {n_clips}\n\n")
        for modality in ("audio", "video"):
            if modality in peaks:
                f.write("\n".join(_modality_lines(modality)) + "\n\n")
        f.write(f"Framing recommendation: {framing}\n")
    print(f"\nwrote {out_dir / 'propagation_decision.txt'}")
    print("\nStopping. Confirm the framing before Stage 1.3.")


def report(per_bin: dict, n_layers_by_modality: dict, out_dir: Path,
           n_clips: int, budget_by_modality: dict = None,
           write_figures: bool = True):
    """Shared reporting: per-layer metrics → table + figures + peak framing."""
    metrics_by_modality = per_layer_metrics(per_bin, n_layers_by_modality)
    peaks = peak_per_modality(metrics_by_modality)

    print_top_coverage(per_bin, n_layers_by_modality)
    print_and_save_per_layer_table(metrics_by_modality, out_dir)
    if budget_by_modality:
        print_budget(budget_by_modality, n_layers_by_modality)

    if write_figures:
        # PRIMARY: per-layer tail-ratio trajectory.
        p = out_dir / "figure_3a_layer_trajectory.png"
        plot_layer_trajectory(metrics_by_modality, peaks, p)
        print(f"wrote {p}")
        # SECONDARY: Figure 3A at representative layers.
        nL = max(n_layers_by_modality.values()) if n_layers_by_modality else 0
        if nL:
            reps = representative_layers(nL)
            p = out_dir / "figure_3a_representative_layers.png"
            plot_layers_grid(per_bin, reps, p)
            print(f"wrote {p}  (layers {reps})")

    decide_and_write(metrics_by_modality, peaks, n_layers_by_modality,
                     out_dir, n_clips)


def run_from_csv(args):
    """Recompute metrics/verdicts/framing and regenerate figures from an
    existing propagation_layers.csv — no model load, no GPU. The per-layer
    attention budget is unavailable from CSV (it needs the forward pass)."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers_csv = Path(args.from_csv)
    if layers_csv.is_dir():
        layers_csv = layers_csv / "propagation_layers.csv"
    print(f"--from_csv: recomputing from {layers_csv} (no model load)")
    per_bin, n_layers_by_modality = load_per_layer_from_csv(layers_csv)
    print(f"  modalities: {list(per_bin)}   n_layers: {n_layers_by_modality}")
    report(per_bin, n_layers_by_modality, out_dir, args.n_clips,
           budget_by_modality=None, write_figures=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

# ----------------------------------------------------------------------
# LLM-emerged sink (P_llm) aggregation + figures
# ----------------------------------------------------------------------

def aggregate_llm_emerged(per_clip_list: list) -> dict | None:
    """Aggregate per-clip P_llm sink masks + attention received into per-layer
    sink/non-sink attention means. per_clip_list is a list of dicts each with
    'is_sink' (n_layers, n_modal) and 'perlayer_attn' (n_layers, n_modal)."""
    if not per_clip_list:
        return None
    n_layers = per_clip_list[0]["perlayer_attn"].shape[0]
    n_clips = len(per_clip_list)
    sink_means    = np.full((n_layers, n_clips), np.nan)
    nonsink_means = np.full((n_layers, n_clips), np.nan)
    sink_counts   = np.zeros((n_layers, n_clips), dtype=np.int64)
    total_counts  = np.zeros((n_layers, n_clips), dtype=np.int64)
    for i, c in enumerate(per_clip_list):
        is_sink = c["is_sink"].astype(bool)              # (n_layers, n_modal)
        attn = c["perlayer_attn"]
        total_counts[:, i] = is_sink.shape[1]
        sink_counts[:, i] = is_sink.sum(axis=1)
        for L in range(n_layers):
            m = is_sink[L]
            if m.any():
                sink_means[L, i] = float(attn[L, m].mean())
            if (~m).any():
                nonsink_means[L, i] = float(attn[L, ~m].mean())
    # Per-layer aggregates (mean over clips, ignoring NaN clips).
    with np.errstate(invalid="ignore"):
        layer_sink     = np.nanmean(sink_means, axis=1)
        layer_sink_std = np.nanstd(sink_means, axis=1)
        layer_nonsink  = np.nanmean(nonsink_means, axis=1)
        layer_nonsink_std = np.nanstd(nonsink_means, axis=1)
        layer_ratio = layer_sink / np.clip(layer_nonsink, 1e-12, None)
    cross_sink    = float(np.nanmean(layer_sink))
    cross_nonsink = float(np.nanmean(layer_nonsink))
    cross_ratio   = cross_sink / max(cross_nonsink, 1e-12)
    mean_sink_per_clip = sink_counts.mean(axis=1)
    mean_total_per_clip = total_counts.mean(axis=1)
    return dict(
        n_clips=n_clips, n_layers=n_layers,
        layer_sink=layer_sink, layer_sink_std=layer_sink_std,
        layer_nonsink=layer_nonsink, layer_nonsink_std=layer_nonsink_std,
        layer_ratio=layer_ratio,
        crosslayer_sink=cross_sink, crosslayer_nonsink=cross_nonsink,
        crosslayer_ratio=cross_ratio,
        sink_count_per_layer=mean_sink_per_clip,
        total_count_per_layer=mean_total_per_clip,
        sink_means_per_clip=sink_means,        # (n_layers, n_clips) for later
        nonsink_means_per_clip=nonsink_means,
    )


def llm_emerged_verdict(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    if ratio >= LLM_RATIO_PRESENT:
        return "present"
    if ratio >= LLM_RATIO_AMBIG:
        return "ambiguous"
    return "none"


def plot_llm_emerged_crosslayer(stats_by_modality: dict, out_path: Path):
    """Bar plot: cross-layer mean attn received, sink vs non-sink per modality,
    with the ratio annotated above each pair."""
    import matplotlib.pyplot as plt
    mods = [m for m in ("audio", "video") if stats_by_modality.get(m)]
    if not mods:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(mods))
    w = 0.36
    sinks    = [stats_by_modality[m]["crosslayer_sink"]    for m in mods]
    nonsinks = [stats_by_modality[m]["crosslayer_nonsink"] for m in mods]
    ratios   = [stats_by_modality[m]["crosslayer_ratio"]   for m in mods]
    ax.bar(x - w / 2, sinks,    width=w, color="#d62728", alpha=0.9,
           edgecolor="black", linewidth=0.5, label="sink-token mean attn")
    ax.bar(x + w / 2, nonsinks, width=w, color="#9ecae1", alpha=0.9,
           edgecolor="black", linewidth=0.5, label="non-sink mean attn")
    for i, (s, ns, r) in enumerate(zip(sinks, nonsinks, ratios)):
        top = max(s, ns) * 1.02
        ax.text(i, top, f"ratio = {r:.2f}×  ({llm_emerged_verdict(r)})",
                ha="center", fontweight="bold", fontsize=10)
        ax.text(i - w / 2, s + max(s, ns) * 0.01, f"{s:.2e}",
                ha="center", fontsize=8)
        ax.text(i + w / 2, ns + max(s, ns) * 0.01, f"{ns:.2e}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mods)
    ax.set_ylabel("mean attention received (averaged over layers)", fontsize=11)
    ax.set_title("Stage 1.2 LLM-emerged sink (P_llm, D_sink={458,2570}, τ=20) — "
                  "cross-layer", fontsize=11)
    ax.grid(True, ls=":", alpha=0.4, axis="y"); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def plot_llm_emerged_trajectory(stats_by_modality: dict, out_path: Path,
                                 compare_layers=COMPARE_LAYERS_LLM):
    """Per modality (2 panels): mean attn to sink (left y) and to non-sink, with
    the per-layer ratio on the right y-axis. Compare layers marked vertically."""
    import matplotlib.pyplot as plt
    mods = [m for m in ("audio", "video") if stats_by_modality.get(m)]
    if not mods:
        return
    fig, axes = plt.subplots(1, len(mods), figsize=(7 * len(mods), 5),
                              squeeze=False)
    axes = axes[0]
    for ax, mod in zip(axes, mods):
        s = stats_by_modality[mod]
        Ls = np.arange(s["n_layers"])
        ax.plot(Ls, s["layer_sink"], marker="o", ms=4, lw=2.0,
                color="#d62728", label="sink-token mean attn")
        ax.fill_between(Ls, s["layer_sink"] - s["layer_sink_std"],
                         s["layer_sink"] + s["layer_sink_std"],
                         color="#d62728", alpha=0.12)
        ax.plot(Ls, s["layer_nonsink"], marker="o", ms=4, lw=2.0,
                color="#9ecae1", label="non-sink mean attn")
        ax.set_xlabel("LLM decoder layer L"); ax.set_ylabel("mean attention received")
        ax.set_title(f"{mod}  (n_clips={s['n_clips']})", fontsize=12)
        for L in compare_layers:
            if 0 <= L < s["n_layers"]:
                ax.axvline(L, ls=":", color="black", alpha=0.4)
        ax_r = ax.twinx()
        ax_r.plot(Ls, s["layer_ratio"], color="#2ca02c", lw=1.4, ls="--",
                  label="ratio sink/non-sink")
        ax_r.axhline(1.0, color="gray", ls=":", alpha=0.6)
        ax_r.axhline(LLM_RATIO_PRESENT, color="#2ca02c", ls=":", alpha=0.5)
        ax_r.set_ylabel("ratio sink / non-sink", color="#2ca02c")
        ax_r.tick_params(axis="y", labelcolor="#2ca02c")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)
        ax_r.legend(loc="upper right", fontsize=8)
    fig.suptitle("Stage 1.2 LLM-emerged sink (P_llm) — per-layer attention "
                 "received  (dashed green = sink/non-sink ratio)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def plot_llm_emerged_layers(stats_by_modality: dict, out_path: Path,
                             layers=COMPARE_LAYERS_LLM):
    """Bar plot at the COMPARE_LAYERS_LLM layers: sink vs non-sink per layer
    per modality, with ratios annotated."""
    import matplotlib.pyplot as plt
    mods = [m for m in ("audio", "video") if stats_by_modality.get(m)]
    if not mods:
        return
    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 5),
                              squeeze=False)
    axes = axes[0]
    w = 0.36
    for ax, mod in zip(axes, mods):
        s = stats_by_modality[mod]
        xs = np.arange(len(layers))
        sinks    = [s["layer_sink"][L]    for L in layers if L < s["n_layers"]]
        nonsinks = [s["layer_nonsink"][L] for L in layers if L < s["n_layers"]]
        ratios   = [s["layer_ratio"][L]   for L in layers if L < s["n_layers"]]
        xs = xs[:len(sinks)]
        ax.bar(xs - w / 2, sinks,    width=w, color="#d62728", alpha=0.9,
               edgecolor="black", linewidth=0.5, label="sink mean attn")
        ax.bar(xs + w / 2, nonsinks, width=w, color="#9ecae1", alpha=0.9,
               edgecolor="black", linewidth=0.5, label="non-sink mean attn")
        for i, (sv, nv, r) in enumerate(zip(sinks, nonsinks, ratios)):
            top = max(sv, nv) * 1.02
            ax.text(i, top, f"ratio = {r:.2f}×\n({llm_emerged_verdict(r)})",
                    ha="center", fontweight="bold", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in layers[:len(xs)]])
        ax.set_ylabel("mean attention received")
        ax.set_title(f"{mod}  (n_clips={s['n_clips']})", fontsize=11)
        ax.grid(True, ls=":", alpha=0.4, axis="y"); ax.legend(fontsize=9)
    fig.suptitle(f"Stage 1.2 LLM-emerged sink (P_llm) — at L{', L'.join(str(L) for L in layers)}",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def write_llm_emerged_outputs(stats_by_modality: dict, out_dir: Path) -> str:
    """Write the per-layer CSV + the 3 figures, return the decision-text block."""
    # Per-layer CSV
    rows = []
    for mod, s in stats_by_modality.items():
        if s is None: continue
        for L in range(s["n_layers"]):
            rows.append(dict(
                modality=mod, layer=L,
                sink_attn_mean=float(s["layer_sink"][L]),
                sink_attn_std=float(s["layer_sink_std"][L]),
                nonsink_attn_mean=float(s["layer_nonsink"][L]),
                nonsink_attn_std=float(s["layer_nonsink_std"][L]),
                ratio=float(s["layer_ratio"][L]),
                mean_sink_count=float(s["sink_count_per_layer"][L]),
                mean_total_count=float(s["total_count_per_layer"][L]),
            ))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "llm_emerged_per_layer.csv", index=False)
        print(f"wrote {out_dir / 'llm_emerged_per_layer.csv'}")
    plot_llm_emerged_crosslayer(stats_by_modality,
                                 out_dir / "llm_emerged_crosslayer.png")
    plot_llm_emerged_trajectory(stats_by_modality,
                                 out_dir / "llm_emerged_trajectory.png")
    plot_llm_emerged_layers(stats_by_modality,
                             out_dir / "llm_emerged_layers_2_21.png")
    # Decision-text block (returned so main can append to propagation_decision.txt)
    lines = ["",
             "=" * 80,
             f"LLM-emerged sink track  (P_llm: D_sink={D_SINK_LLM}, τ={TAU_SINK_LLM})",
             "  Question: do D_sink-activated LLM tokens receive more attention",
             "  than non-D_sink tokens, layer by layer?  (vs Stage 1.2's primary",
             "  question, which is about P_prop, the encoder-norm-defined population.)",
             "=" * 80]
    for mod, s in stats_by_modality.items():
        if s is None: continue
        v_cross = llm_emerged_verdict(s["crosslayer_ratio"])
        lines.append(
            f"  {mod}: cross-layer  sink_attn = {s['crosslayer_sink']:.3e},  "
            f"nonsink_attn = {s['crosslayer_nonsink']:.3e},  "
            f"ratio = {s['crosslayer_ratio']:.2f}×  → {v_cross}")
        for L in COMPARE_LAYERS_LLM:
            if 0 <= L < s["n_layers"]:
                r = float(s["layer_ratio"][L])
                lines.append(f"           L{L:<3d}  sink = {s['layer_sink'][L]:.3e}, "
                              f"nonsink = {s['layer_nonsink'][L]:.3e}, "
                              f"ratio = {r:.2f}×  → {llm_emerged_verdict(r)}")
        n_sink_l21 = (s["sink_count_per_layer"][min(21, s["n_layers"]-1)]
                       if s["n_layers"] > 21 else float("nan"))
        n_sink_l2 = (s["sink_count_per_layer"][2] if s["n_layers"] > 2 else float("nan"))
        lines.append(f"           mean sink tokens/clip:  L2 = {n_sink_l2:.1f},  "
                      f"L21 = {n_sink_l21:.1f},  of ~"
                      f"{s['total_count_per_layer'][0]:.0f} modal tokens")
    lines.append("")
    return "\n".join(lines)


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Primary metric: per-layer tail_ratio trajectory over ALL layers.")

    print(f"Loading Stage 1.1 norms from {args.norms_npz}")
    cache = load_cache(Path(args.norms_npz))
    for modality, entries in cache.items():
        print(f"  {modality}: {len(entries)} cached clips")

    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    max_memory = None
    if args.max_memory_per_gpu:
        max_memory = {i: args.max_memory_per_gpu for i in range(n_gpu)}
        max_memory["cpu"] = args.cpu_memory
    if n_gpu == 1 and args.device_map != "auto":
        print(f"  [note] only 1 visible GPU — overriding device_map "
              f"{args.device_map!r} → 'auto'.")
        args.device_map = "auto"
    model, processor = load_omni(
        args.model_path, device_map=args.device_map, max_memory=max_memory,
    )
    try:
        print(f"  device_map placement: {model.hf_device_map}")
    except AttributeError:
        pass
    tokenizer = processor.tokenizer
    thinker_cfg = _resolve_thinker_cfg(model)

    # ----- per-clip data collection -----
    #   norms:    list of (N_llm,)
    #   perlayer: list of (n_layers, N_llm)
    #   is_sink:  list of (n_layers, N_llm) bool  (P_llm)
    #   budget:   per span, list of (n_layers,) per-clip arrays
    per_modality: dict = {}
    budget_pool: dict = {}
    n_layers_by_modality: dict = {}
    span_names = ("system", "modal", "query", "generated")
    # Parallel per-clip storage for the LLM-emerged (P_llm) track. Same clip
    # order as per_modality; each entry holds {"is_sink", "perlayer_attn"}.
    per_clip_llm: dict = {}

    for modality, entries in cache.items():
        modal_type = "a" if modality == "audio" else "v"
        clip_dir = Path(args.audio_dir if modality == "audio" else args.video_dir)
        sliced = entries if args.no_subset else entries[: args.n_clips]
        print(f"\n{modality} pass — {len(sliced)} clips, modal_type={modal_type!r}")
        per_modality[modality] = {"norms": [], "perlayer": []}
        per_clip_llm[modality] = []
        budget_pool[modality] = {sp: [] for sp in span_names}
        tags: dict[str, int] = {}
        failures: dict[str, int] = {}
        first_verified = False
        for cached_name, cached_norms in tqdm(sliced, desc=modality):
            clip_path = clip_dir / cached_name
            if not clip_path.exists():
                failures["missing_file"] = failures.get("missing_file", 0) + 1
                continue
            result, spans, err = process_clip(
                model, processor, clip_path, modal_type, thinker_cfg, tokenizer,
                max_gen_tokens=args.max_gen_tokens,
            )
            if result is None:
                failures[err] = failures.get(err, 0) + 1
                continue
            n_llm = result["N_llm"]
            n_layers_by_modality[modality] = result["n_layers"]
            aligned, tag = align_norms(cached_norms, n_llm)
            tags[tag] = tags.get(tag, 0) + 1
            if aligned is None:
                failures[f"align:{tag}"] = failures.get(f"align:{tag}", 0) + 1
                continue

            if not first_verified:
                print(
                    f"  [sanity] alignment check on first clip {cached_name}:\n"
                    f"    cached_norms[:3] = {cached_norms[:3].tolist()}\n"
                    f"    N_enc={len(cached_norms)}, N_llm={n_llm}, "
                    f"n_layers={result['n_layers']}, alignment={tag}\n"
                    f"    aligned_norms[:3] = {aligned[:3].tolist()}\n"
                    f"    perlayer_attn.mean(0)[:3] = "
                    f"{result['perlayer_attn'].mean(0)[:3].tolist()}"
                )
                first_verified = True

            per_modality[modality]["norms"].append(aligned)
            per_modality[modality]["perlayer"].append(result["perlayer_attn"])
            for sp in span_names:
                budget_pool[modality][sp].append(result["budget_layers"][sp])
            # P_llm: same per-clip alignment, just a parallel store
            if "is_sink_per_layer" in result:
                per_clip_llm[modality].append(dict(
                    is_sink=result["is_sink_per_layer"],
                    perlayer_attn=result["perlayer_attn"],
                ))
        print(f"  alignment tags: {tags}")
        if failures:
            print(f"  failures: {failures}")

    # ----- per-layer attention budget (mean over clips) -----
    budget_by_modality: dict = {}
    for modality in per_modality:
        bsp = budget_pool[modality]
        budget_by_modality[modality] = {
            sp: (np.mean(np.stack(v), axis=0) if v else None)
            for sp, v in bsp.items()
        }

    # ----- per-bin stats at EVERY layer -----
    per_bin: dict = {}
    for modality in per_modality:
        per_bin[modality] = {}
        norms_list = per_modality[modality]["norms"]
        n_clips_used = len(norms_list)
        if n_clips_used == 0:
            continue
        pooled_n = np.concatenate(norms_list)                       # (T,)
        pooled_pl = np.concatenate(
            per_modality[modality]["perlayer"], axis=1)             # (n_layers, T)
        n_layers = n_layers_by_modality[modality]
        for L in range(n_layers):
            per_bin[modality][L] = bin_norms_attn(
                pooled_n, pooled_pl[L], n_clips_used, BIN_WIDTH)

    # ----- raw per-bin CSV (read back by --from_csv) -----
    layer_rows = []
    for modality in per_bin:
        for L in range(n_layers_by_modality.get(modality, 0)):
            if per_bin[modality].get(L) is not None:
                layer_rows += perbin_rows(modality, L, per_bin[modality][L])
    pd.DataFrame(layer_rows).to_csv(out_dir / "propagation_layers.csv", index=False)
    print(f"\nwrote {out_dir / 'propagation_layers.csv'}  (per-layer per-bin, all layers)")

    # ----- coverage, per-layer table, budget, figures, framing -----
    report(per_bin, n_layers_by_modality, out_dir, args.n_clips,
           budget_by_modality=budget_by_modality, write_figures=True)

    # ----- LLM-emerged (P_llm) track -----
    print("\n" + "=" * 80)
    print("LLM-emerged sink (P_llm) track — D_sink={458, 2570}, τ=20")
    print("=" * 80)
    llm_stats = {mod: aggregate_llm_emerged(per_clip_llm[mod])
                 for mod in per_clip_llm if per_clip_llm[mod]}
    if llm_stats:
        block = write_llm_emerged_outputs(llm_stats, out_dir)
        print(block)
        # Append to the decision file so both verdicts live together.
        dec_path = out_dir / "propagation_decision.txt"
        with open(dec_path, "a") as f:
            f.write(block + "\n")
        print(f"appended LLM-emerged block to {dec_path}")
    else:
        print("  (no clips contributed P_llm data — skipping)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument(
        "--norms_npz",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                    "stage1_1_encoder_norms/encoder_norms.npz"),
        help="Stage 1.1 cached norms.",
    )
    p.add_argument(
        "--audio_dir", default=str(_REPO / "data/AudioSet/audios"),
    )
    p.add_argument(
        "--video_dir", default=str(_REPO / "data/ActivityNet/videos"),
    )
    p.add_argument("--n_clips", type=int, default=300)
    p.add_argument(
        "--max_gen_tokens", type=int, default=MAX_GEN_TOKENS,
        help="Safety ceiling on generated tokens (passed as "
             "thinker_max_new_tokens; the generic max_new_tokens is shadowed by "
             "the composite generate). Decoding stops at EOS first, so this "
             "rarely binds — captions are typically tens of tokens.",
    )
    p.add_argument(
        "--no_subset", action="store_true",
        help="Use ALL cached clips (overrides --n_clips slice).",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/stage1_2_propagation"),
    )
    p.add_argument(
        "--device_map", default="balanced_low_0",
        help="HF device_map. Default 'balanced_low_0' is the only multi-GPU map "
             "that works on this model — plain 'auto'/'balanced' crash with a "
             "cross-device mismatch. Auto-falls back to 'auto' on a single GPU.",
    )
    p.add_argument(
        "--max_memory_per_gpu", default=None,
        help="Optional per-GPU cap (e.g. '18GiB') forwarded as max_memory with "
             "an additional CPU spill bucket. NOTE: capping changes placement and "
             "can re-trigger the device-mismatch bug — leave unset unless needed.",
    )
    p.add_argument(
        "--cpu_memory", default="64GiB",
        help="CPU spill budget when --max_memory_per_gpu is set.",
    )
    p.add_argument(
        "--from_csv", default=None,
        help="Skip the model entirely and recompute metrics/verdicts/framing + "
             "figures from an existing propagation_layers.csv (no GPU). Accepts "
             "the CSV path or its directory.",
    )
    args = p.parse_args()
    if args.from_csv:
        run_from_csv(args)
    else:
        main(args)
