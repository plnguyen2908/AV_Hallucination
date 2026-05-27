"""
encoder_to_llm_propagation_exp.py

Stage 1.2 — Encoder-to-LLM propagation analysis.

Tests whether encoder-side high-norm tokens (from Stage 1.1) receive
disproportionately high LLM attention during decoding. Reproduces
Sink-or-Not-to-Sink Figure 3A for Qwen2.5-Omni at two layers (2 = early,
14 = middle) on both modalities.

The output of this stage determines the project's framing:
    (A) Both modalities propagate           → symmetric two-population story
    (B) Only video propagates clearly       → asymmetric (audio-deficit) story
    (C) Neither propagates clearly          → ASD refinement, not Sink-extension

Prerequisites:
    - Stage 1.1 cached encoder_norms.npz (per-clip pre-projection norms).
    - Same 300 clips/modality used in 1.1 (token positions must align).

PRIMARY metric = CROSS-LAYER-AVERAGED attention. For each modal token we average
its received attention over ALL decoder layers and all heads (and the 20 gen
query positions), giving one attention number per token, then bin by encoder
norm. The per-layer view (layers 2 & 14) is kept only as a supplementary
breakdown — it is no longer the primary evidence.

Procedure:
    1. Load cached norms; for each cached clip, re-run a forward pass with
       output_attentions=True and max_new_tokens=20.
    2. At each of the 20 generation steps, for EVERY decoder layer, take the
       last-query-row attention averaged across heads; slice to the modal
       columns. Average over (heads, layers, gen steps) → cross-layer per-token
       attention. Also keep the per-layer means (for the trajectory + per-layer
       figures). Raw values, no renormalization (matches the paper).
    3. Align encoder norms with LLM-side modal positions if the projector
       downsamples (chunked-mean); in practice 1:1 for Qwen2.5-Omni.
    4. Fine-grained binning: width = 5 norm units, range [0, max_norm + 5].
    5. PRIMARY Figure 3A on the cross-layer attention (figure_3a_crosslayer.png):
       violet bars "Avg Attn for LLM Outputs", orange log line "Avg # of Tokens
       per Clip", <10-token bins hatched.
    6. Quantitative summary on the cross-layer data: n_reliable_bins (≥10),
       Pearson r (bin-index vs mean_attn over reliable bins), high_low_ratio
       (token-weighted mean_attn of norm>p95 & reliable ÷ norm<p50), tail_ratio
       (top-3 reliable high-norm bins ÷ norm<50) — the verdict driver — plus the
       audio bimodal characterization (0-10, 60-100, >120 norm windows).
    7. Framing decision from the cross-layer verdicts.
    8. Supplementary figures: figure_3a_layer_trajectory.png (per-(layer, norm
       bin) attention heatmap, per-layer-normalized) and figure_3a_layers_2_14.png
       (per-layer Figure 3A panels for layers 2 & 14).

    --from_csv recomputes 6-8 and regenerates figures from existing CSVs with no
    model load (propagation_summary.csv = cross-layer; propagation_layers.csv =
    per-layer, optional, needed for the trajectory + per-layer figures).

Sanity checks (before plotting):
    - Per-modality cross-layer attention budget split among system / query /
      modal / generated. modal_attn budget < 1% is flagged — itself a finding.
    - Coverage of top-5 high-norm bins.
    - One-clip alignment verification: cached norms vs cross-layer attention.

Outputs (--output_dir, default results/qwen2_5_omni/sink_analysis/stage1_2_propagation):
    figure_3a_crosslayer.png         PRIMARY
    figure_3a_layer_trajectory.png   supplementary heatmap
    figure_3a_layers_2_14.png        tertiary per-layer panels
    propagation_summary.csv          cross-layer per-bin stats
    propagation_layers.csv           per-layer per-bin stats (all layers)
    propagation_metrics.csv          per-(modality, layer) summary metrics
    propagation_decision.txt         verdicts + framing recommendation

Stop after writing these. Do not proceed to Stage 1.3 without confirmation.
"""

import argparse
import sys
import warnings
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
)


SEED = 42
N_GEN = 20
BIN_WIDTH = 5.0
RELIABLE_MIN_TOKENS = 10        # reliability threshold for a bin
LOW_NORM_PCT = 50.0            # low-norm regime = below this percentile of norms
HIGH_NORM_PCT = 95.0          # high-norm regime = above this percentile (+ reliable)
LOW50_NORM = 50.0             # "norm < 50" reference for the tail-focused metric
TAIL_TOPN = 3                 # # of highest-norm reliable bins for the tail metric
# audio bimodal characterization windows
LEFT_NORM_HI = 10.0           # leftmost / positional-sink bin: norm in [0, 10)
SEC_NORM_LO, SEC_NORM_HI = 60.0, 100.0   # secondary peak window
HITAIL_NORM = 120.0           # high-norm tail: norm > 120
# verdict thresholds on the tail-focused metric (top3_reliable_high / mean_low)
TAIL_PRESENT = 1.5
TAIL_AMBIGUOUS = 1.0
PROMPT_BY_MODAL = {
    "a": "Describe what you hear in detail.",
    "v": "Describe what you see in detail.",
}
COLOR_BAR = "#9467bd"           # violet bars (attention)
COLOR_LINE = "#ff7f0e"          # orange line (token count)


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
# Per-clip processing — single forward, ALL layers + cross-layer average
# --------------------------------------------------------------------------

def process_clip(
    model, processor, clip_path: Path, modal_type: str,
    thinker_cfg, tokenizer,
):
    """For one clip, capture attention to every modal token from each of the
    N_GEN generated query positions, averaged across heads, at EVERY decoder
    layer. Returns:
        {
            'xlayer_attn':   (N_llm,)  mean over (layers, heads, gen steps),
            'perlayer_attn': (n_layers, N_llm)  mean over (heads, gen steps),
            'budget_per_step': {span: [...]}  cross-layer-averaged budget,
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
        return None, None, f"prep:{e}"

    prompt_len = inputs["input_ids"].shape[1]
    spans = find_all_spans(inputs["input_ids"], tokenizer, thinker_cfg)
    mod_key = "audio" if modal_type == "a" else "video"
    if mod_key not in spans:
        return None, spans, f"no_{mod_key}_span"
    mod_start, mod_end = spans[mod_key]

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                use_audio_in_video=use_aiv,
                return_audio=False,
                do_sample=False,
                max_new_tokens=N_GEN,
                output_attentions=True,
                return_dict_in_generate=True,
            )
    except Exception as e:
        return None, spans, f"generate:{e}"

    gen_attns = getattr(outputs, "attentions", None)
    if gen_attns is None or len(gen_attns) == 0:
        del outputs
        torch.cuda.empty_cache()
        return None, spans, "no_attentions"

    n_steps = len(gen_attns)
    n_layers = len(gen_attns[0])
    n_llm = mod_end - mod_start
    # Accumulate per-layer modal attention summed over generation steps.
    perlayer_modal_sum = np.zeros((n_layers, n_llm), dtype=np.float64)
    budget = {"system": [], "modal": [], "query": [], "generated": []}

    for s_idx, step in enumerate(gen_attns):
        # Each layer's attention tensor may live on a different device under
        # device_map="auto", so move every layer's last-query row to CPU before
        # stacking. mean over heads → (kv_len,) per layer; stack → (n_layers, kv_len).
        rows = [
            step[L][0, :, -1, :].float().mean(dim=0).cpu()
            for L in range(n_layers)
        ]
        arr = torch.stack(rows, dim=0).numpy()           # (n_layers, kv_len)
        perlayer_modal_sum += arr[:, mod_start:mod_end]
        # Cross-layer-averaged full row for the budget sanity check.
        xlayer_full = arr.mean(axis=0)                   # (kv_len,)
        for span_name, span_key in (
            ("system", "system"), ("modal", mod_key), ("query", "query"),
        ):
            if span_key in spans:
                s, e = spans[span_key]
                budget[span_name].append(float(xlayer_full[s:e].sum()))
            else:
                budget[span_name].append(0.0)
        gen_end = prompt_len + s_idx
        budget["generated"].append(
            float(xlayer_full[prompt_len:gen_end].sum()) if gen_end > prompt_len else 0.0
        )

    perlayer_attn = perlayer_modal_sum / n_steps         # mean over gen steps
    xlayer_attn = perlayer_attn.mean(axis=0)             # mean over layers too

    del outputs
    torch.cuda.empty_cache()
    return {
        "xlayer_attn": xlayer_attn,
        "perlayer_attn": perlayer_attn,
        "budget_per_step": budget,
        "N_llm": n_llm,
        "n_layers": n_layers,
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
    """Corrected propagation metrics.

    - high_low_ratio (robust): token-weighted mean_attn over the HIGH-norm
      regime (mean_norm > p95 AND bin count >= RELIABLE_MIN_TOKENS) divided by
      the same over the LOW-norm regime (mean_norm < p50). Replaces the old
      n_per_clip split, which collapsed to NaN because the first sub-1 bin sits
      at the sparse low-norm edge, not in the high-norm tail.
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

    # Robust high/low ratio.
    low_mask = populated & np.isfinite(mean_norm) & (mean_norm < p50)
    high_mask = (counts >= RELIABLE_MIN_TOKENS) & np.isfinite(mean_norm) & (mean_norm > p95)
    low_attn = _wmean_attn(mean_attn, counts, low_mask)
    high_attn = _wmean_attn(mean_attn, counts, high_mask)
    high_low_ratio = (
        high_attn / low_attn
        if np.isfinite(low_attn) and low_attn > 0 and np.isfinite(high_attn)
        else float("nan")
    )

    # Tail-focused ratio (verdict driver).
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
        top3_attn=top3_attn,
        mean_low50=mean_low50,
        tail_ratio=tail_ratio,
        top3_high_norms=top3_norms,
        max_attn_bin_norm=max_attn_bin_norm,
        left_attn_0_10=left_attn,
        attn_60_100=sec_attn,
        attn_gt_120=hitail_attn,
    )


def verdict_for(metric: dict) -> tuple:
    """Verdict from the tail-focused metric (top3_reliable_high / mean_low).
    Returns (verdict_string, strength_tag in {'present','ambiguous','none'})."""
    tr = metric["tail_ratio"]
    if not np.isfinite(tr):
        return (
            "Indeterminate — no reliable high-norm bins to estimate the tail.",
            "none",
        )
    if tr >= TAIL_PRESENT:
        return (
            f"Propagation present — high-norm tail attended {tr:.2f}x the "
            f"low-norm (<50) regime.",
            "present",
        )
    if tr >= TAIL_AMBIGUOUS:
        return (
            f"Weak / ambiguous — high-norm tail attention {tr:.2f}x the "
            f"low-norm regime (between 1.0 and 1.5).",
            "ambiguous",
        )
    return (
        f"No propagation — high-norm tail attention {tr:.2f}x the low-norm "
        f"regime (below 1.0).",
        "none",
    )


def framing_from_strengths(strengths: dict) -> str:
    """Project framing from per-modality verdict tags."""
    a = strengths.get("audio", "none")
    v = strengths.get("video", "none")
    a_prop, v_prop = (a == "present"), (v == "present")
    if a_prop and v_prop:
        return "(A) Both modalities propagate — symmetric two-population story"
    if (not a_prop) and v_prop:
        return (
            "(B) Only video propagates clearly — asymmetric (audio-deficit) "
            "framing"
        )
    return (
        "(C) Neither propagates clearly — project becomes ASD refinement, "
        "not Sink-or-Not-to-Sink extension"
    )


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


def plot_primary_crosslayer(xl_bin: dict, out_path: Path):
    """PRIMARY Figure 3A using cross-layer-averaged attention. xl_bin[modality]
    is a bindict built from attention averaged over all layers + heads."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, modality in zip(axes, ("audio", "video")):
        if modality not in xl_bin or xl_bin[modality] is None:
            ax.set_visible(False)
            continue
        _plot_panel(ax, xl_bin[modality], f"{modality.capitalize()} Encoder (all layers)")
    fig.suptitle(
        "Encoder norm vs LLM attention (averaged across all layers)",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layers_grid(per_bin: dict, layers, out_path: Path):
    """TERTIARY supplementary figure: rows = the named layers, cols = modality."""
    layers = [l for l in layers]
    fig, axes = plt.subplots(len(layers), 2, figsize=(16, 6 * len(layers)))
    if len(layers) == 1:
        axes = np.array([axes])
    for ri, layer_idx in enumerate(layers):
        for ci, modality in enumerate(("audio", "video")):
            ax = axes[ri][ci]
            if modality not in per_bin or layer_idx not in per_bin[modality]:
                ax.set_visible(False)
                continue
            _plot_panel(ax, per_bin[modality][layer_idx],
                        f"{modality.capitalize()} Encoder (layer {layer_idx})")
    fig.suptitle("Per-layer breakdown (supplementary)", fontsize=15, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layer_trajectory(traj: dict, out_path: Path):
    """SUPPLEMENTARY heatmap: x = layer index, y = norm bin (high-norm at top),
    color = mean attention per (layer, bin), normalized within each layer
    (per-column max). Only reliable bins (>=10 tokens) are shown; the rest are
    masked. Reveals whether propagation is layer-localized or distributed."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, modality in zip(axes, ("audio", "video")):
        if modality not in traj or traj[modality] is None:
            ax.set_visible(False)
            continue
        t = traj[modality]
        M = t["matrix"].astype(np.float64).copy()        # (n_bins, n_layers)
        counts = t["bin_counts"]
        centers = t["bin_centers"]
        n_layers = M.shape[1]
        # Mask unreliable bins (count identical across layers — same tokens).
        M[counts < RELIABLE_MIN_TOKENS, :] = np.nan
        # Restrict y to the populated-reliable range for legibility.
        reli_rows = np.where(counts >= RELIABLE_MIN_TOKENS)[0]
        if reli_rows.size == 0:
            ax.set_visible(False)
            continue
        top_row = reli_rows.max()
        M = M[: top_row + 1]
        centers = centers[: top_row + 1]
        # Per-layer (per-column) normalization by column max.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            colmax = np.nanmax(M, axis=0, keepdims=True)
        colmax[~np.isfinite(colmax) | (colmax == 0)] = 1.0
        Mn = M / colmax
        im = ax.imshow(
            np.ma.masked_invalid(Mn), aspect="auto", origin="lower",
            cmap=cmap, vmin=0.0, vmax=1.0,
            extent=[-0.5, n_layers - 0.5, -0.5, len(centers) - 0.5],
        )
        ax.set_xlabel("LLM layer index", fontsize=11)
        ax.set_ylabel("Encoder L2 norm", fontsize=11)
        ystep = max(1, len(centers) // 12)
        yt = np.arange(0, len(centers), ystep)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{centers[i]:.0f}" for i in yt])
        ax.set_title(f"{modality.capitalize()} Encoder — attn(layer, norm bin)",
                     fontsize=13)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("per-layer-normalized mean attn", fontsize=10)
    fig.suptitle(
        "Layer trajectory of encoder-norm → LLM attention "
        "(per-layer-normalized; high-norm at top)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporting (shared by full-run and --from_csv paths)
# --------------------------------------------------------------------------

DECISION_KEY = "all"            # cross-layer-averaged is the primary evidence


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


def load_per_bin_from_csv(summary_csv: Path, layers_csv: Path = None):
    """Reconstruct per_bin[modality]['all'] from the cross-layer
    propagation_summary.csv, and per_bin[modality][int] from the optional
    per-layer propagation_layers.csv. Returns (per_bin, n_layers_by_modality)."""
    df = pd.read_csv(summary_csv)
    df["layer"] = df["layer"].astype(str)
    if DECISION_KEY not in set(df["layer"]):
        raise SystemExit(
            f"{summary_csv} has no '{DECISION_KEY}' (cross-layer) rows — it "
            "predates the cross-layer change. Re-run the full pipeline (no "
            "--from_csv) to regenerate it."
        )
    per_bin: dict = {}
    for (modality, _), g in df[df["layer"] == DECISION_KEY].groupby(["modality", "layer"]):
        per_bin.setdefault(modality, {})[DECISION_KEY] = _bindict_from_group(g)

    n_layers_by_modality: dict = {}
    if layers_csv and Path(layers_csv).exists():
        dl = pd.read_csv(layers_csv)
        for (modality, layer), g in dl.groupby(["modality", "layer"]):
            per_bin.setdefault(modality, {})[int(layer)] = _bindict_from_group(g)
        for modality in per_bin:
            int_layers = [k for k in per_bin[modality] if isinstance(k, int)]
            if int_layers:
                n_layers_by_modality[modality] = max(int_layers) + 1
    return per_bin, n_layers_by_modality


def print_top5_coverage(per_bin: dict):
    print("\n" + "=" * 78)
    print("Top-5 high-norm bins by mean_norm (coverage check; norm dist is "
          "layer-independent)")
    print("=" * 78)
    for modality in per_bin:
        b = per_bin[modality].get(DECISION_KEY)
        if b is None:
            continue
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


def build_summary_rows(per_bin: dict, layers_to_report: list) -> list:
    """layers_to_report: ordered list of keys, e.g. ['all', 2, 14]."""
    rows = []
    for modality in per_bin:
        for layer_label in layers_to_report:
            if layer_label not in per_bin[modality]:
                continue
            metric = compute_summary(per_bin[modality][layer_label])
            rows.append({"modality": modality, "layer": layer_label, **metric})
    return rows


def print_summary_table(summary_rows: list):
    def _f(v, fmt):
        return fmt.format(v) if np.isfinite(v) else "  nan "

    print("\n" + "=" * 92)
    print("Quantitative summary  (layer 'all' = cross-layer-averaged = PRIMARY; "
          "2/14 supplementary)")
    print("  high/low: norm>p95&reliable vs norm<p50 | tail3/low: top-3 reliable "
          "hi-norm bins / norm<50")
    print("=" * 92)
    print(
        f"{'modality':<9}{'layer':<6}{'reliable':<9}{'noisy':<7}"
        f"{'pearson_r':<11}{'high/low':<10}{'tail3/low':<11}{'max_attn_norm':<14}"
    )
    print("-" * 92)
    for r in summary_rows:
        print(
            f"{r['modality']:<9}{str(r['layer']):<6}{r['n_reliable_bins']:<9}"
            f"{r['n_noisy_bins']:<7}"
            f"{_f(r['pearson_r'], '{:+.3f} '):<11}"
            f"{_f(r['high_low_ratio'], '{:.2f}x '):<10}"
            f"{_f(r['tail_ratio'], '{:.2f}x '):<11}"
            f"{_f(r['max_attn_bin_norm'], '{:.1f}'):<14}"
        )


def print_audio_characterization(summary_rows: list):
    audio_rows = [r for r in summary_rows if r["modality"] == "audio"]
    if not audio_rows:
        return
    print("\n" + "=" * 78)
    print("Audio bimodal characterization (token-weighted mean_attn per window)")
    print("=" * 78)

    def _e(v):
        return f"{v:.3e}" if np.isfinite(v) else "  nan  "

    for r in audio_rows:
        left, sec, tail = r["left_attn_0_10"], r["attn_60_100"], r["attn_gt_120"]
        ratio = left / sec if np.isfinite(left) and np.isfinite(sec) and sec > 0 else float("nan")
        print(
            f"  layer {str(r['layer']):<4}:  leftmost(0-10)={_e(left)}   "
            f"secondary(60-100)={_e(sec)}   high-tail(>120)={_e(tail)}"
        )
        if np.isfinite(ratio):
            note = (
                "leftmost >> secondary → positional sink dominates"
                if ratio >= 2 else
                "leftmost ~ secondary → no single dominant positional sink"
            )
            print(f"             leftmost/secondary = {ratio:.2f}x  ({note})")


def write_metrics_csv(out_dir: Path, summary_rows: list):
    rows = []
    for r in summary_rows:
        v, s = verdict_for(r)
        rows.append({
            "modality": r["modality"], "layer": r["layer"],
            "is_primary": (r["layer"] == DECISION_KEY),
            "n_reliable_bins": r["n_reliable_bins"],
            "n_noisy_bins": r["n_noisy_bins"],
            "pearson_r": r["pearson_r"],
            "p50_norm": r["p50_norm"], "p95_norm": r["p95_norm"],
            "low_attn": r["low_attn"], "high_attn": r["high_attn"],
            "high_low_ratio": r["high_low_ratio"],
            "top3_attn": r["top3_attn"], "mean_low50": r["mean_low50"],
            "tail_ratio": r["tail_ratio"],
            "top3_high_norms": ";".join(f"{x:.1f}" for x in r["top3_high_norms"]),
            "max_attn_bin_norm": r["max_attn_bin_norm"],
            "left_attn_0_10": r["left_attn_0_10"],
            "attn_60_100": r["attn_60_100"],
            "attn_gt_120": r["attn_gt_120"],
            "strength": s,
        })
    path = out_dir / "propagation_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


def build_trajectory(per_bin_modality: dict, n_layers: int) -> dict:
    """(n_bins, n_layers) matrix of mean_attn per (bin, layer)."""
    ref = per_bin_modality.get(0) or per_bin_modality.get(DECISION_KEY)
    if ref is None:
        return None
    n_bins = ref["n_bins"]
    M = np.full((n_bins, n_layers), np.nan)
    for L in range(n_layers):
        if L in per_bin_modality:
            M[:, L] = per_bin_modality[L]["mean_attn"]
    return dict(matrix=M, bin_centers=ref["bin_centers"],
                bin_counts=ref["bin_counts"], n_layers=n_layers)


def decide_and_write(summary_rows: list, out_dir: Path, n_clips: int,
                     tertiary_layers, probed_layers_desc: str):
    verdicts, strengths = {}, {}
    for r in summary_rows:
        if r["layer"] != DECISION_KEY:
            continue
        v, s = verdict_for(r)
        verdicts[r["modality"]] = v
        strengths[r["modality"]] = s
    framing = framing_from_strengths(strengths)

    print("\n" + "=" * 78)
    print("Per-modality verdicts  (PRIMARY = cross-layer-averaged attention)")
    print("=" * 78)
    for modality in ("audio", "video"):
        if modality in verdicts:
            print(f"  {modality}: {verdicts[modality]}")
    print(f"\n  Framing recommendation: {framing}")

    with open(out_dir / "propagation_decision.txt", "w") as f:
        f.write("Stage 1.2 — Encoder-to-LLM propagation (cross-layer primary)\n")
        f.write("Primary metric: attention averaged across ALL layers + heads.\n")
        f.write(f"Supplementary per-layer panels: {probed_layers_desc}\n")
        f.write(f"n_clips/mod   : {n_clips}\n")
        f.write("Verdict driver: tail_ratio = (top-3 reliable high-norm bins, "
                "token-weighted mean_attn) / (mean_attn of norm<50 bins)\n")
        f.write("  >=1.5 present | 1.0-1.5 ambiguous | <1.0 none\n\n")
        for modality in ("audio", "video"):
            if modality in verdicts:
                f.write(f"{modality}: {verdicts[modality]}\n")
        f.write(f"\nFraming recommendation: {framing}\n")
    print(f"\nwrote {out_dir / 'propagation_decision.txt'}")
    print("\nStopping. Confirm the framing before Stage 1.3.")


def report(per_bin: dict, n_layers_by_modality: dict, tertiary_layers,
           out_dir: Path, n_clips: int, write_figures: bool = True):
    """Shared reporting. PRIMARY evidence = cross-layer ('all'); layers in
    `tertiary_layers` are a supplementary per-layer breakdown."""
    tertiary_layers = [l for l in tertiary_layers]
    report_layers = [DECISION_KEY] + tertiary_layers

    print_top5_coverage(per_bin)
    summary_rows = build_summary_rows(per_bin, report_layers)
    print_summary_table(summary_rows)
    print_audio_characterization(summary_rows)
    write_metrics_csv(out_dir, summary_rows)

    if write_figures:
        # PRIMARY: cross-layer Figure 3A.
        xl_bin = {m: per_bin[m].get(DECISION_KEY) for m in per_bin}
        p = out_dir / "figure_3a_crosslayer.png"
        plot_primary_crosslayer(xl_bin, p)
        print(f"wrote {p}")
        # SUPPLEMENTARY: layer trajectory heatmap.
        traj = {}
        for m in per_bin:
            nl = n_layers_by_modality.get(m)
            if nl:
                traj[m] = build_trajectory(per_bin[m], nl)
        if any(v is not None for v in traj.values()):
            p = out_dir / "figure_3a_layer_trajectory.png"
            plot_layer_trajectory(traj, p)
            print(f"wrote {p}")
        # TERTIARY: per-layer breakdown for the named layers.
        if any(l in per_bin[m] for m in per_bin for l in tertiary_layers):
            p = out_dir / "figure_3a_layers_2_14.png"
            plot_layers_grid(per_bin, tertiary_layers, p)
            print(f"wrote {p}")

    desc = f"layers {tertiary_layers} (tertiary figure_3a_layers_2_14.png)"
    decide_and_write(summary_rows, out_dir, n_clips, tertiary_layers, desc)


def run_from_csv(args):
    """Recompute metrics/verdicts/framing (and regenerate figures) from existing
    CSVs — no model load, no GPU. Needs the cross-layer propagation_summary.csv;
    propagation_layers.csv (optional) enables the trajectory + per-layer figures."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.from_csv)
    layers_csv = summary_csv.parent / "propagation_layers.csv"
    print(f"--from_csv: recomputing from {summary_csv} (no model load)")
    per_bin, n_layers_by_modality = load_per_bin_from_csv(summary_csv, layers_csv)
    print(f"  modalities: {list(per_bin)}   n_layers: {n_layers_by_modality}")
    report(per_bin, n_layers_by_modality, (args.layer_a, args.layer_b),
           out_dir, args.n_clips, write_figures=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tertiary_layers = (args.layer_a, args.layer_b)
    print(f"Primary metric: cross-layer average over ALL layers.")
    print(f"Tertiary per-layer panels: {tertiary_layers}")

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
    # per_modality[modality] = {"norms": [...], "xlayer": [...], "perlayer": [...]}
    #   norms:    list of (N_llm,)
    #   xlayer:   list of (N_llm,)  cross-layer-averaged attention
    #   perlayer: list of (n_layers, N_llm)
    per_modality: dict = {}
    budget_pool: dict = {}                 # cross-layer per-span budgets
    n_layers_by_modality: dict = {}

    for modality, entries in cache.items():
        modal_type = "a" if modality == "audio" else "v"
        clip_dir = Path(args.audio_dir if modality == "audio" else args.video_dir)
        sliced = entries if args.no_subset else entries[: args.n_clips]
        print(f"\n{modality} pass — {len(sliced)} clips, modal_type={modal_type!r}")
        per_modality[modality] = {"norms": [], "xlayer": [], "perlayer": []}
        budget_pool[modality] = {sp: [] for sp in ("system", "modal",
                                                   "query", "generated")}
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
                    f"    xlayer_attn[:3]   = {result['xlayer_attn'][:3].tolist()}"
                )
                first_verified = True

            per_modality[modality]["norms"].append(aligned)
            per_modality[modality]["xlayer"].append(result["xlayer_attn"])
            per_modality[modality]["perlayer"].append(result["perlayer_attn"])
            for sp, vals in result["budget_per_step"].items():
                budget_pool[modality][sp].extend(vals)
        print(f"  alignment tags: {tags}")
        if failures:
            print(f"  failures: {failures}")

    # ----- sanity: attention budget (cross-layer-averaged) -----
    print("\n" + "=" * 78)
    print("Attention budget sanity check  (cross-layer, per-(clip,query) mean, %)")
    print("=" * 78)
    for modality in per_modality:
        print(f"\n  {modality}:")
        tot = 0.0
        for sp in ("system", "modal", "query", "generated"):
            vals = budget_pool[modality][sp]
            if not vals:
                continue
            pct = float(np.mean(vals)) * 100
            tot += pct
            flag = "  ⚠ MODAL <1% — LLM is ignoring this modality" if (
                sp == "modal" and pct < 1.0) else ""
            print(f"    {sp:<10s}: {pct:6.2f}%{flag}")
        print(f"    {'(sum)':<10s}: {tot:6.2f}%  (≈100% minus modal markers/header)")

    # ----- per-bin stats: cross-layer ('all') + every layer -----
    per_bin: dict = {}
    for modality in per_modality:
        per_bin[modality] = {}
        norms_list = per_modality[modality]["norms"]
        n_clips_used = len(norms_list)
        if n_clips_used == 0:
            continue
        pooled_n = np.concatenate(norms_list)                       # (T,)
        pooled_xl = np.concatenate(per_modality[modality]["xlayer"])  # (T,)
        pooled_pl = np.concatenate(
            per_modality[modality]["perlayer"], axis=1)             # (n_layers, T)
        per_bin[modality][DECISION_KEY] = bin_norms_attn(
            pooled_n, pooled_xl, n_clips_used, BIN_WIDTH)
        n_layers = n_layers_by_modality[modality]
        for L in range(n_layers):
            per_bin[modality][L] = bin_norms_attn(
                pooled_n, pooled_pl[L], n_clips_used, BIN_WIDTH)

    # ----- raw per-bin CSVs (read back by --from_csv) -----
    summary_rows, layer_rows = [], []
    for modality in per_bin:
        if DECISION_KEY in per_bin[modality]:
            summary_rows += perbin_rows(modality, DECISION_KEY,
                                        per_bin[modality][DECISION_KEY])
        for L in range(n_layers_by_modality.get(modality, 0)):
            if L in per_bin[modality]:
                layer_rows += perbin_rows(modality, L, per_bin[modality][L])
    pd.DataFrame(summary_rows).to_csv(out_dir / "propagation_summary.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out_dir / "propagation_layers.csv", index=False)
    print(f"\nwrote {out_dir / 'propagation_summary.csv'}  (cross-layer)")
    print(f"wrote {out_dir / 'propagation_layers.csv'}  (per-layer, all layers)")

    # ----- coverage, metrics, figures, framing decision -----
    report(per_bin, n_layers_by_modality, tertiary_layers, out_dir,
           args.n_clips, write_figures=True)


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
        "--no_subset", action="store_true",
        help="Use ALL cached clips (overrides --n_clips slice).",
    )
    p.add_argument(
        "--layer_a", type=int, default=2,
        help="First layer for the TERTIARY per-layer breakdown figure "
             "(default 2). No longer the decision layer — the verdict now uses "
             "cross-layer-averaged attention.",
    )
    p.add_argument(
        "--layer_b", type=int, default=14,
        help="Second layer for the tertiary per-layer figure (default 14).",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/stage1_2_propagation"),
    )
    p.add_argument(
        "--device_map", default="balanced_low_0",
        help="HF device_map. Default 'balanced_low_0' shards the model across "
             "ALL visible GPUs and keeps GPU 0 light — required for 4-GPU runs, "
             "since output_attentions on long video OOMs a single card. Plain "
             "'auto' packs everything onto GPU 0. Auto-falls back to 'auto' if "
             "only 1 GPU is visible.",
    )
    p.add_argument(
        "--max_memory_per_gpu", default=None,
        help="Optional per-GPU cap (e.g. '18GiB') forwarded as max_memory with "
             "an additional CPU spill bucket. Use to fine-tune sharding if "
             "balanced_low_0 still OOMs.",
    )
    p.add_argument(
        "--cpu_memory", default="64GiB",
        help="CPU spill budget when --max_memory_per_gpu is set.",
    )
    p.add_argument(
        "--from_csv", default=None,
        help="Skip the model entirely and recompute metrics/verdicts/framing "
             "from an existing propagation_summary.csv (no GPU).",
    )
    args = p.parse_args()
    if args.from_csv:
        run_from_csv(args)
    else:
        main(args)
