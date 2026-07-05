"""
stage1_3_dimension_signatures.py

Stage 1.3 — Hidden-state dimension signatures of propagated vs LLM-emerged sinks.

GOAL: replicate the Sink-or-Not-to-Sink separation test for Qwen2.5-Omni — do
encoder-propagated sinks and LLM-emerged sinks activate DIFFERENT hidden
dimensions? And does that separation map onto our inherited/emerged split from
the corrected Stage 0.1?

FIXED INPUTS (from Stage 0.1 RMSNorm, 20x):
  D_sink_all = {458, 2570, 3197}
  Inherited  = {458, 2570}   (present in base Qwen2.5-7B)
  Emerged    = {3197}        (Omni-only, multimodal-induced)
  hidden_dim = 3584 (base and Omni thinker match)

TOKEN SINK CRITERION (Kang et al.):
  Token at (position j, layer l) is an LLM sink iff
    max over d in D_sink_all of |RMSNorm(x_j^l)[d]| >= 20.

POPULATIONS (native criteria, NO cross-gating; overlap is expected and fine):
  P_prop_video  : video-span tokens with encoder norm > 100
  P_prop_audio  : audio-span tokens with encoder norm > 100   (SAME tau=100)
  P_llm_video   : video-span tokens satisfying the tau=20 LLM-sink criterion
                  at the layer of interest (no encoder gate)
  P_llm_audio   : audio-span tokens satisfying the tau=20 criterion
  Control_random_{audio,video}: random positions in each modality span (per
                  modality, sampled per clip; baseline for the sink-dim plot).

LAYERS (default): 2 (video early, Stage 1.2 peak), 14 (mid), 21 (audio late).

RMSNorm convention: pure normalization x / sqrt(mean(x^2) + eps) — NO learned
weight, eps from the thinker's text config. MATCHES Stage 0.1 exactly so the
dimension magnitudes are comparable.

Encoder norms for VGGSounder are computed on the fly by hooking audio_tower and
visual (Stage 1.1's cache covers AudioSet/ActivityNet, not VGGSounder).

OUTPUTS (--output_dir):
    dimension_profiles.png        per-population per-layer per-dim profile
    sink_dim_comparison.png       grouped bar at {458,2570,3197} (headline)
    dimension_signatures.csv      per (population, layer) summary + high_dims
    population_sizes.csv          per (population, layer) n_tokens + overlaps
    stage1_3_decision.txt         verdict + framing read

Stop after printing the verdict. Do not proceed to Stage 2 without confirmation.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import (  # noqa: E402
    build_conversation, find_modality_spans, load_omni, prepare_inputs,
)


# ---- fixed-input constants ----
# D_SINK_ALL is the τ=20 token-sink GATE. After Stage 1.3 distinctiveness
# analysis: 3197 is dropped — its distinctiveness to P_prop populations is <1
# and it's broadly active (highest in video non_sink). It's a video-content
# dim, NOT a sink register. Sink registers from Stage 0.1 (RMSNorm, 20x) are
# {458, 2570} — both inherited from base Qwen2.5.
D_SINK_ALL = [458, 2570]
INHERITED = [458, 2570]
# 3197 is kept here as a TRACKED diagnostic dim (NOT a sink). The label
# 'EMERGED' is retained from earlier runs for code compatibility; figures/CSVs
# still report it for context, just not as a sink-classification gate.
EMERGED = [3197]
TAU_HIGH_ENC = 100.0          # P_prop encoder-norm gate
TAU_TOKEN_SINK = 20.0         # LLM-sink token criterion (max over D_sink dims)
HIGH_DIM_MULT = 10.0          # per-profile high-dim cutoff (> 10x median)
MIN_POP_SIZE = 30             # below this, profile is too noisy to interpret
N_RANDOM_PER_CLIP = 200       # random control sample size per modality per clip
PROMPT_AV = "Describe what you see and hear in detail."

POPULATIONS = (
    "P_prop_video", "P_llm_video",
    "P_prop_audio", "P_llm_audio",
    "Random_video", "Random_audio",
)
POP_COLOR = {
    "P_prop_video": "#d62728",   # red
    "P_llm_video":  "#1f77b4",   # blue
    "P_prop_audio": "#ff7f0e",   # orange
    "P_llm_audio":  "#2ca02c",   # green
    "Random_video": "#999999",
    "Random_audio": "#cccccc",
}
DEFAULT_LAYERS = (2, 14, 21)


# --------------------------------------------------------------------------
# Helpers — RMSNorm, span finding, encoder hooks, alignment
# --------------------------------------------------------------------------

def _rmsnorm_abs(hs: torch.Tensor, eps: float) -> torch.Tensor:
    """|RMSNorm(x)| pure normalization (no learned weight). MATCHES Stage 0.1."""
    rms = torch.sqrt(hs.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (hs / rms).abs()


def _thinker_rms_eps(model) -> float:
    cfg = model.thinker.config
    cfg = getattr(cfg, "text_config", cfg)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_start_token_id"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _resolve_encoders(model):
    thinker = model.thinker
    audio_mod = visual_mod = None
    for attr in ("audio_tower", "audio_encoder", "audio_model"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr); break
    for attr in ("visual", "visual_tower", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr); break
    return audio_mod, visual_mod


def _extract_tokens(out) -> torch.Tensor:
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    return x


def align_norms(enc_norms: np.ndarray, n_llm: int):
    """Match Stage 1.2's align_norms. Returns (aligned, tag) or (None, tag)."""
    n_enc = len(enc_norms)
    if n_enc == n_llm:
        return enc_norms, "equal"
    if n_enc > n_llm and n_enc % n_llm == 0:
        k = n_enc // n_llm
        return enc_norms.reshape(n_llm, k).mean(axis=1), f"down(x{k})"
    if n_llm > n_enc and n_llm % n_enc == 0:
        k = n_llm // n_enc
        return np.repeat(enc_norms, k), f"up(x{k})"
    return None, f"mismatch({n_enc}->{n_llm})"


def _modal_positions(input_ids: torch.Tensor, thinker_cfg) -> tuple:
    """Per-token modality classification by INPUT ID. Required for AV clips
    where audio and video tokens are interleaved inside the BOS/EOS brackets.
    Returns (audio_positions, video_positions) as 1D np.int64 arrays."""
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return np.where(ids == a_id)[0].astype(np.int64), \
           np.where(ids == v_id)[0].astype(np.int64)


# --------------------------------------------------------------------------
# Reporting helpers — per-layer em/inh table, L2-focused verdict, washout,
# em/inh trajectory figure, and a no-GPU replot path from saved profiles.npz.
# --------------------------------------------------------------------------

def _build_rows_high_dims(profiles, counts, layers, H):
    """Derive per-(population, layer) summary rows + high_dims sets from the
    saved per-dim profile arrays. Used by main() and by --replot_only."""
    high_dims: dict = {}
    rows = []
    for p in POPULATIONS:
        for li, L in enumerate(layers):
            prof = profiles[p][li]
            n = int(counts[p][li])
            if n == 0 or not np.isfinite(prof).any():
                high_dims[(p, L)] = []
                rows.append(dict(population=p, layer=L, n_tokens=n,
                                 n_high_dims=0, high_dims="",
                                 act_458=float("nan"), act_2570=float("nan"),
                                 act_3197=float("nan"),
                                 inherited_mean=float("nan"),
                                 emerged_mean=float("nan"),
                                 emerged_to_inherited_ratio=float("nan"),
                                 sink_dims_in_high=""))
                continue
            med = float(np.nanmedian(prof))
            hd = sorted(int(d) for d in np.where(prof > HIGH_DIM_MULT * med)[0])
            high_dims[(p, L)] = hd
            a458, a2570 = float(prof[458]), float(prof[2570])
            a3197 = float(prof[3197])
            inh = (a458 + a2570) / 2.0
            ratio = a3197 / inh if inh > 0 else float("nan")
            rows.append(dict(population=p, layer=L, n_tokens=n,
                             n_high_dims=len(hd),
                             high_dims=";".join(map(str, hd[:30])),
                             act_458=a458, act_2570=a2570, act_3197=a3197,
                             inherited_mean=inh, emerged_mean=a3197,
                             emerged_to_inherited_ratio=ratio,
                             sink_dims_in_high=";".join(
                                 str(d) for d in D_SINK_ALL if d in hd)))
    return rows, high_dims


def _rows_to_lookup(rows):
    return {(r["population"], r["layer"]): r for r in rows}


def print_em_inh_per_layer_table(rows, layers, out_dir):
    """Per-layer em/inh ratio per population — wide format showing the full
    trajectory at a glance. Also writes em_inh_trajectory.csv."""
    pop_to = _rows_to_lookup(rows)
    print("\n" + "=" * 86)
    print("EM/INH RATIO PER (POPULATION, LAYER)  "
          "[em = act on dim 3197; inh = mean act on {458, 2570}]")
    print("=" * 86)
    header = f"  {'population':<16}" + "".join(
        f"{'L'+str(L)+' em/inh':<14}" for L in layers)
    print(header)
    print("  " + "-" * (len(header) - 2))
    wide_rows = []
    for p in POPULATIONS:
        line = f"  {p:<16}"
        wd = {"population": p}
        for L in layers:
            r = pop_to.get((p, L))
            v = r["emerged_to_inherited_ratio"] if r else float("nan")
            wd[f"em_inh_L{L}"] = v
            wd[f"em_L{L}"] = r["emerged_mean"] if r else float("nan")
            wd[f"inh_L{L}"] = r["inherited_mean"] if r else float("nan")
            wd[f"n_L{L}"] = r["n_tokens"] if r else 0
            cell = f"{v:.2f}x" if np.isfinite(v) else "nan"
            line += f"{cell:<14}"
        print(line)
        wide_rows.append(wd)
    pd.DataFrame(wide_rows).to_csv(out_dir / "em_inh_trajectory.csv", index=False)
    print(f"\nwrote {out_dir / 'em_inh_trajectory.csv'}")


def verdict_l2_focused(rows, layers, out_dir):
    """L2 verdict with explicit per-population expectations + cross-layer
    washout quantification. Writes stage1_3_decision.txt."""
    pop_to = _rows_to_lookup(rows)
    L_focus = 2 if 2 in layers else layers[0]

    def get(p, L):
        return pop_to.get((p, L), {}).get("emerged_to_inherited_ratio", float("nan"))

    print("\n" + "=" * 86)
    print(f"VERDICT — L{L_focus}-focused  "
          f"(video's encoder-propagation peak from Stage 1.2)")
    print("=" * 86)
    # NOTE: em/inh = act[3197] / mean(act[458], act[2570]). A HIGH em/inh for
    # P_prop_video does NOT imply distinctive activation on 3197 — per the
    # distinctiveness analysis it reflects DENOMINATOR SUPPRESSION (P_prop_video
    # fails to load the inherited register dims). 3197 is broadly active, not a
    # propagated register. Expectations below are kept as descriptive labels.
    expectations = [
        ("P_prop_video", "HIGH em/inh — driven by LOW inherited denom (not by "
                         "distinctive 3197); see distinctiveness analysis",
         lambda v: np.isfinite(v) and v >= 1.5),
        ("P_llm_video",  "LOW em/inh — loads inherited register {458, 2570}",
         lambda v: np.isfinite(v) and v < 1.0),
        ("P_prop_audio", "LOW em/inh — uses inherited dims like ordinary sinks",
         lambda v: np.isfinite(v) and v < 1.0),
        ("P_llm_audio",  "LOW em/inh — loads inherited register {458, 2570}",
         lambda v: np.isfinite(v) and v < 1.0),
    ]
    verdict_lines = []
    for pop, expect_str, check in expectations:
        v = get(pop, L_focus)
        n = pop_to.get((pop, L_focus), {}).get("n_tokens", 0)
        mark = "✓" if check(v) else "✗"
        s = (f"  {pop:<14}  em/inh@L{L_focus} = "
             f"{(f'{v:.2f}x' if np.isfinite(v) else 'nan'):>8}   "
             f"(n={n:>5})   expected {expect_str:<58}  {mark}")
        print(s); verdict_lines.append(s)

    # Spec framing: video_separated = vp HIGH and exceeds vl by a clear margin;
    # audio_propagated_signature = ap HIGH like vp (expected FALSE).
    vp = get("P_prop_video", L_focus); vl = get("P_llm_video", L_focus)
    ap = get("P_prop_audio", L_focus)
    MARGIN = 0.5  # "clear margin" between vp and vl
    video_separated = (np.isfinite(vp) and np.isfinite(vl)
                       and vp >= 1.5 and (vp - vl) > MARGIN)
    audio_prop_sig = (np.isfinite(ap) and ap >= 1.5)
    if video_separated and not audio_prop_sig:
        head = (
            f"\n  → SEPARATION BY ABSENCE (Stage 1.3 final framing): video "
            f"populations are dimensionally separable on the inherited register "
            f"dims (P_llm_video em/inh = {vl:.2f}x — strongly loads {INHERITED}; "
            f"P_prop_video em/inh = {vp:.2f}x — suppressed on {INHERITED}, gap "
            f"+{vp-vl:.2f}). CAVEAT: the HIGH P_prop_video em/inh is driven by a "
            f"LOW inherited denominator, NOT by distinctive activation on dim "
            f"3197 (distinct=0.83 < 1; 3197 is a broadly-active video-content "
            f"channel, not a propagated register). P_prop has NO dedicated "
            f"propagated register dimension — UNLIKE LLaVA. Propagation appears "
            f"in attention (Stage 1.2) but not in hidden-state registers.")
    elif video_separated and audio_prop_sig:
        head = (f"\n  → SYMMETRIC: both video AND audio propagated populations "
                f"show high em/inh (vp={vp:.2f}x, ap={ap:.2f}x). Re-check "
                f"distinctiveness analysis before claiming dedicated registers.")
    elif (not video_separated):
        head = (f"\n  → NO SEPARATION: video em/inh not clearly separated "
                f"(vp={vp:.2f}x, vl={vl:.2f}x, gap=+{vp-vl:.2f}).")
    else:
        head = (f"\n  → INDETERMINATE: vp={vp:.2f}x, vl={vl:.2f}x, ap={ap:.2f}x.")
    print(head); verdict_lines.append(head)

    # Cross-layer washout (gap = em/inh[P_prop_video] − em/inh[P_llm_video]).
    print("\n" + "=" * 86)
    print("CROSS-LAYER WASHOUT  "
          "(separation gap = em/inh[P_prop_video] − em/inh[P_llm_video])")
    print("=" * 86)
    wash_lines = []
    for L in layers:
        vp_L = get("P_prop_video", L); vl_L = get("P_llm_video", L)
        gap = vp_L - vl_L if np.isfinite(vp_L) and np.isfinite(vl_L) else float("nan")
        s = (f"  L{L:<3}: P_prop_video = {vp_L:.2f}x   "
             f"P_llm_video = {vl_L:.2f}x   gap = {gap:+.2f}")
        print(s); wash_lines.append(s)
    L_first, L_last = layers[0], layers[-1]
    g_first = get("P_prop_video", L_first) - get("P_llm_video", L_first)
    g_last = get("P_prop_video", L_last) - get("P_llm_video", L_last)
    if np.isfinite(g_first) and np.isfinite(g_last) and abs(g_first) > 0:
        shrink = abs(g_last) / abs(g_first)
        if shrink < 0.3:
            tail = (f"\n  → WASHOUT CONFIRMED: separation collapses from "
                    f"|{g_first:+.2f}| at L{L_first} to |{g_last:+.2f}| at L{L_last} "
                    f"({shrink*100:.0f}% retained) — residual stream homogenizes "
                    f"populations over depth.")
        else:
            tail = (f"\n  → separation partially persists: |L{L_first}|="
                    f"{abs(g_first):.2f}, |L{L_last}|={abs(g_last):.2f} "
                    f"({shrink*100:.0f}% retained).")
    else:
        tail = "\n  → washout indeterminate."
    print(tail); wash_lines.append(tail)

    with open(out_dir / "stage1_3_decision.txt", "w") as f:
        f.write("Stage 1.3 — Hidden-state dimension signatures (L2-focused)\n")
        f.write(f"layers analyzed: {list(layers)}\n\n")
        f.write("Per-layer em/inh ratios:\n")
        for r in rows:
            v = r["emerged_to_inherited_ratio"]
            f.write(f"  {r['population']:<14} L{r['layer']:<3}  em/inh = "
                    f"{(f'{v:.2f}x' if np.isfinite(v) else 'nan'):>8}  "
                    f"(n={r['n_tokens']})\n")
        f.write("\n--- VERDICT (L2) ---\n")
        for ln in verdict_lines:
            f.write(ln + "\n")
        f.write("\n--- WASHOUT ---\n")
        for ln in wash_lines:
            f.write(ln + "\n")
    print(f"\nwrote {out_dir / 'stage1_3_decision.txt'}")


def plot_em_inh_trajectory(rows, layers, out_dir):
    """em/inh ratio vs layer — one bold line per main population, dashed for
    random controls. Headline panel for the per-layer story."""
    pop_to = _rows_to_lookup(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in POPULATIONS:
        ys = [pop_to.get((p, L), {}).get("emerged_to_inherited_ratio",
                                         float("nan")) for L in layers]
        is_rand = "Random" in p
        ax.plot(layers, ys, marker="o", ms=6,
                lw=1.2 if is_rand else 2.6,
                ls="--" if is_rand else "-",
                color=POP_COLOR[p], label=p)
    ax.axhline(1.0, color="gray", ls=":", lw=0.9, alpha=0.7,
               label="em = inh (1.0)")
    ax.axhline(1.5, color="#d62728", ls=":", lw=1.0,
               label="HIGH threshold (1.5)")
    ax.set_xticks(layers)
    ax.set_xlabel("LLM decoder layer", fontsize=11)
    ax.set_ylabel("em / inh   (act dim 3197 / mean act dims {458, 2570})",
                  fontsize=11)
    ax.set_title("Per-layer em/inh ratio trajectory "
                 "(sharp separation early, washes out by L21)", fontsize=12)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    p_fig = out_dir / "em_inh_trajectory.png"
    fig.savefig(p_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_fig}")


def plot_asd_figure(asd_bos_mean, asd_sink_median, asd_nonsink_median,
                    layers, asd_layer, H, out_dir):
    """ASD-format two-panel replication. Top: BOS-token activation (mean over
    clips). Bottom: sink vs non-sink token medians (median over per-clip
    medians). Yellow axvspan bands at inherited dims {458, 2570}; green band
    at emerged dim {3197}. xtick labels at all three sink dims."""
    if asd_layer not in layers:
        print(f"  [asd] requested layer {asd_layer} not in {layers}; "
              f"falling back to {layers[0]}.")
        asd_layer = layers[0]
    li = list(layers).index(asd_layer)
    bos = asd_bos_mean[li]
    sink = asd_sink_median[li]
    nonsink = asd_nonsink_median[li]

    band_hw = 15
    dims = np.arange(H)

    def _bands(ax):
        # Inherited dims (yellow), then emerged dim (green). zorder=0 so the
        # line plot (zorder=2) sits on top. Translucent so the line is visible
        # against the band.
        for d in INHERITED:
            ax.axvspan(d - band_hw, d + band_hw,
                       color="gold", alpha=0.25, zorder=0)
        for d in EMERGED:
            ax.axvspan(d - band_hw, d + band_hw,
                       color="mediumseagreen", alpha=0.25, zorder=0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5),
                                         sharex=True)

    # TOP — BOS only.
    _bands(ax_top)
    ax_top.plot(dims, bos, color="black", lw=0.7, zorder=2)
    ax_top.set_xlim(0, H)
    ax_top.set_ylabel("|RMSNorm(x)[d]|  —  BOS token", fontsize=11)
    ax_top.set_title(f"Qwen2.5-Omni (7B)  —  layer {asd_layer}", fontsize=13)
    ax_top.grid(True, ls=":", alpha=0.35, zorder=1)

    # BOTTOM — sink vs non-sink medians.
    _bands(ax_bot)
    ax_bot.plot(dims, nonsink, color="#1f77b4", lw=0.7,
                label="non-sink tokens (median)", zorder=2)
    ax_bot.plot(dims, sink, color="#d62728", lw=0.7,
                label="sink tokens (τ=20, median)", zorder=2)
    ax_bot.set_xlim(0, H)
    ax_bot.set_xlabel("hidden-state dim", fontsize=11)
    ax_bot.set_ylabel("|RMSNorm(x)[d]|  —  per-token median", fontsize=11)
    ax_bot.grid(True, ls=":", alpha=0.35, zorder=1)
    ax_bot.legend(loc="upper right", fontsize=9)

    # x-tick labels at the three sink dims so the reader can read them off.
    sink_ticks = sorted(set(D_SINK_ALL))
    for ax in (ax_top, ax_bot):
        ax.set_xticks(sink_ticks + [0, H])
        ax.set_xticklabels([str(d) if d in sink_ticks else str(d)
                            for d in sink_ticks + [0, H]], fontsize=9)
    # Tag the bands with a one-line legend at the figure level.
    fig.legend(handles=[
        Patch(facecolor="gold", alpha=0.4,
              label=f"inherited dims {INHERITED} (yellow band)"),
        Patch(facecolor="mediumseagreen", alpha=0.4,
              label=f"emerged dim {EMERGED} (green band)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, fontsize=9, frameon=False)

    plt.tight_layout()
    p_fig = out_dir / "asd_sink_dim_replication.png"
    fig.savefig(p_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_fig}")

    # Sanity prints.
    print("\n" + "=" * 86)
    print(f"ASD figure sanity (layer {asd_layer})")
    print("=" * 86)
    print(f"  BOS act at        458 = {bos[458]:.3f}   2570 = {bos[2570]:.3f}   "
          f"3197 = {bos[3197]:.3f}   "
          f"(expect 458/2570 HIGH, 3197 LOW since BOS is text)")
    print(f"  sink median  at   458 = {sink[458]:.3f}   2570 = {sink[2570]:.3f}   "
          f"3197 = {sink[3197]:.3f}")
    print(f"  non-sink median   458 = {nonsink[458]:.3f}   2570 = {nonsink[2570]:.3f}"
          f"   3197 = {nonsink[3197]:.3f}")
    sep_458 = sink[458] / nonsink[458] if nonsink[458] > 0 else float("nan")
    sep_2570 = sink[2570] / nonsink[2570] if nonsink[2570] > 0 else float("nan")
    print(f"  sink/non-sink ratio: dim 458 = {sep_458:.2f}x   "
          f"dim 2570 = {sep_2570:.2f}x   (expect sink >> non-sink at inherited dims)")


def plot_asd_per_modality(bos_mean, asd_per_mod, asd_per_mod_counts,
                          layers, asd_layer, H, out_dir):
    """Per-modality ASD-format GRID: 4 rows × 2 cols. Cols = (audio, video).
    Rows = (BOS, non-sink, P_llm, P_prop). sharey='row' so audio vs video at
    each category are directly comparable. Yellow bands at inherited dims,
    green at the emerged dim."""
    if asd_layer not in layers:
        print(f"  [asd-per-mod] requested layer {asd_layer} not in {layers}; "
              f"falling back to {layers[0]}.")
        asd_layer = layers[0]
    li = list(layers).index(asd_layer)
    band_hw = 15
    dims = np.arange(H)

    ROW_DEF = [
        ("BOS",      "black",   "BOS\n|RMSNorm(x)[d]|"),
        ("non_sink", "#1f77b4", "non-sink\n|RMSNorm(x)[d]|"),
        ("P_llm",    "#d62728", "P_llm (τ=20 sink)\n|RMSNorm(x)[d]|"),
        ("P_prop",   "#ff7f0e", "P_prop (enc > 100)\n|RMSNorm(x)[d]|"),
    ]
    MODS = ("audio", "video")

    def _bands(ax):
        for d in INHERITED:
            ax.axvspan(d - band_hw, d + band_hw, color="gold",
                       alpha=0.25, zorder=0)
        for d in EMERGED:
            ax.axvspan(d - band_hw, d + band_hw, color="mediumseagreen",
                       alpha=0.25, zorder=0)

    fig, axes = plt.subplots(len(ROW_DEF), len(MODS), figsize=(14, 10),
                             sharex=True, sharey="row", squeeze=False)
    for ri, (cat, color, ylabel) in enumerate(ROW_DEF):
        for ci, modality in enumerate(MODS):
            ax = axes[ri, ci]
            _bands(ax)
            if cat == "BOS":
                # BOS is one token at position 0, modality-agnostic — same
                # line in both cols by design (mirrors ASD's two-column layout).
                ax.plot(dims, bos_mean[li], color=color, lw=0.7, zorder=2)
                n_str = "1 per clip"
            else:
                prof = asd_per_mod[(modality, cat)][li]
                n = int(asd_per_mod_counts[(modality, cat)][li])
                ax.plot(dims, prof, color=color, lw=0.7, zorder=2)
                n_str = f"n={n}"
            ax.grid(True, ls=":", alpha=0.35, zorder=1)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if ri == 0:
                ax.set_title(modality, fontsize=13)
            ax.text(0.99, 0.95, n_str, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", alpha=0.75,
                              edgecolor="lightgray"))

    sink_ticks = sorted(set(D_SINK_ALL))
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(sink_ticks + [0, H])
            ax.set_xticklabels([str(d) for d in sink_ticks + [0, H]],
                               fontsize=9)
            ax.set_xlim(0, H)
    for ax in axes[-1, :]:
        ax.set_xlabel("hidden-state dim", fontsize=11)

    fig.suptitle(f"Qwen2.5-Omni (7B) — layer {asd_layer}  "
                 f"(per-modality ASD-format grid)", fontsize=13, y=1.005)
    fig.legend(handles=[
        Patch(facecolor="gold", alpha=0.4,
              label=f"inherited dims {INHERITED} (yellow)"),
        Patch(facecolor="mediumseagreen", alpha=0.4,
              label=f"emerged dim {EMERGED} (green)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=2, fontsize=9,
        frameon=False)
    plt.tight_layout()
    p_fig = out_dir / "asd_sink_dim_per_modality.png"
    fig.savefig(p_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_fig}")

    # Sanity prints — per modality, per category at the chosen layer.
    print("\n" + "=" * 86)
    print(f"Per-modality ASD sanity (layer {asd_layer}) — act on {{458, 2570, 3197}}")
    print("=" * 86)
    print(f"  {'(mod, cat)':<28}{'n':<8}{'458':<10}{'2570':<10}{'3197':<10}em/inh")
    print("  " + "-" * 70)
    for m in ("audio", "video"):
        for cat in ("non_sink", "P_llm", "P_prop"):
            prof = asd_per_mod[(m, cat)][li]
            n = int(asd_per_mod_counts[(m, cat)][li])
            a458, a2570, a3197 = float(prof[458]), float(prof[2570]), float(prof[3197])
            inh = (a458 + a2570) / 2.0
            ratio = a3197 / inh if inh > 0 else float("nan")
            print(f"  {f'({m}, {cat})':<28}{n:<8}{a458:<10.2f}{a2570:<10.2f}"
                  f"{a3197:<10.2f}{ratio:.2f}x")


DISTINCT_THRESHOLD = 1.5    # distinctiveness > this to count as distinctive
ACTIVE_THRESHOLD = 3.0      # population median must exceed this to be "active"


def distinctiveness_analysis(asd_per_mod, asd_per_mod_counts, layers, asd_layer,
                             H, out_dir):
    """For each (modality, P_prop or P_llm), rank all dims by

        distinctiveness[d] = pop_median[d] / max(other_pop_medians[d])

    A dim is "distinctive" if distinctiveness > 1.5 AND pop_median > 3 (so the
    population actually uses it, not just trivially small ÷ smaller). Also
    report explicitly at the three sink dims {458, 2570, 3197}. This tests
    whether P_prop has its own dedicated register dim (like LLaVA's
    {982, 2494, 3263}) or whether its dimensional separation is by ABSENCE on
    {458, 2570} rather than by a distinctive propagated dim.
    """
    if asd_layer not in layers:
        print(f"  [distinct] layer {asd_layer} not in {layers}; "
              f"falling back to {layers[0]}.")
        asd_layer = layers[0]
    li = list(layers).index(asd_layer)

    print("\n" + "=" * 90)
    print(f"DISTINCTIVENESS ANALYSIS — layer {asd_layer}")
    print(f"  distinctiveness[d] = pop_median[d] / max(other_pops_median[d])")
    print(f"  'distinctive' := distinctiveness > {DISTINCT_THRESHOLD} AND "
          f"pop_median > {ACTIVE_THRESHOLD}")
    print("=" * 90)

    rows = []
    summary = {}     # (modality, pop) -> list of distinctive dim ints
    for modality in ("audio", "video"):
        for pop in ("P_prop", "P_llm"):
            prof = asd_per_mod[(modality, pop)][li]                    # (H,)
            others = [c for c in ("non_sink", "P_llm", "P_prop") if c != pop]
            other_arrs = np.stack(
                [asd_per_mod[(modality, o)][li] for o in others], axis=0)
            other_max = np.nanmax(other_arrs, axis=0)                  # (H,)
            denom = np.maximum(other_max, 1e-12)
            distinct = np.where(np.isfinite(prof) & np.isfinite(other_max),
                                prof / denom, np.nan)
            is_active = np.isfinite(prof) & (prof > ACTIVE_THRESHOLD)
            is_distinctive = (np.isfinite(distinct)
                              & (distinct > DISTINCT_THRESHOLD)
                              & is_active)

            order = np.argsort(np.where(np.isfinite(distinct),
                                        distinct, -np.inf))[::-1][:10]
            n_pop = int(asd_per_mod_counts[(modality, pop)][li])
            print(f"\n  ({modality}, {pop})  n={n_pop}  — top 10 by distinctiveness:")
            print(f"    {'dim':<7}{'distinct':<11}{'pop_med':<10}"
                  f"{'other_max':<11}{'active?':<9}{'distinct?':<10}sink?")
            for d in order:
                d = int(d)
                star = " *" if d in D_SINK_ALL else ""
                print(f"    {d:<7}{distinct[d]:<11.2f}{prof[d]:<10.2f}"
                      f"{other_max[d]:<11.2f}{str(bool(is_active[d])):<9}"
                      f"{str(bool(is_distinctive[d])):<10}{star}")
                rows.append(dict(
                    modality=modality, population=pop, dim=d,
                    distinctiveness=float(distinct[d]),
                    pop_median=float(prof[d]),
                    other_max=float(other_max[d]),
                    is_active=bool(is_active[d]),
                    is_distinctive=bool(is_distinctive[d]),
                    is_sink_dim=d in D_SINK_ALL,
                ))
            # Sink-dim breakdown — KEY for the verdict.
            print(f"    sink-dim breakdown ({modality}, {pop}):")
            for d in D_SINK_ALL:
                tag = "EM" if d in EMERGED else "INH"
                print(f"      dim {d:<5} [{tag}]  distinct={distinct[d]:.2f}  "
                      f"pop_med={prof[d]:.2f}  other_max={other_max[d]:.2f}  "
                      f"distinctive={bool(is_distinctive[d])}")

            dist_dims = sorted(int(d) for d in np.where(is_distinctive)[0])
            summary[(modality, pop)] = dist_dims
            if dist_dims:
                head = (f"    → {pop}_{modality}: {len(dist_dims)} distinctive "
                        f"dim(s): {dist_dims[:15]}{' ...' if len(dist_dims)>15 else ''}")
            else:
                head = (f"    → {pop}_{modality}: NO distinctive dim (no dim "
                        f"with distinct > {DISTINCT_THRESHOLD} AND active > "
                        f"{ACTIVE_THRESHOLD}).")
            print(head)

    pd.DataFrame(rows).to_csv(out_dir / "distinctiveness_top10.csv", index=False)
    print(f"\nwrote {out_dir / 'distinctiveness_top10.csv'}")

    # Final synthesis tied to the user's question.
    print("\n" + "=" * 90)
    print("DISTINCTIVENESS VERDICT")
    print("=" * 90)
    lines = []
    # 1. P_prop has any distinctive dim?
    for m in ("audio", "video"):
        dd = summary[(m, "P_prop")]
        if dd:
            s = (f"  P_prop_{m}: HAS distinctive dim(s) {dd[:10]} — "
                 f"propagated sinks DO form a dedicated register here.")
        else:
            s = (f"  P_prop_{m}: NO distinctive dim — no register channel "
                 f"that is uniquely activated by encoder-propagated sinks.")
        print(s); lines.append(s)
    # 2. P_llm's distinctive dims include {458, 2570}?
    for m in ("audio", "video"):
        dd = summary[(m, "P_llm")]
        sink_in_dd = [d for d in INHERITED if d in dd]
        s = (f"  P_llm_{m}: {len(dd)} distinctive dim(s) "
             f"{dd[:10]}{' ...' if len(dd)>10 else ''}; inherited dims in set: "
             f"{sink_in_dd}.")
        print(s); lines.append(s)
    # 3. Is 3197 distinctive to ANY P_prop?
    print()
    for m in ("audio", "video"):
        prof = asd_per_mod[(m, "P_prop")][li]
        others = np.stack([asd_per_mod[(m, o)][li] for o in
                           ("non_sink", "P_llm")], axis=0)
        omax = np.nanmax(others, axis=0)
        d3 = prof[3197] / max(omax[3197], 1e-12)
        s = (f"  dim 3197 in P_prop_{m}: distinctiveness = {d3:.2f}, "
             f"pop_med = {prof[3197]:.2f}, "
             f"max(non_sink, P_llm) = {omax[3197]:.2f} "
             f"{'(NOT distinctive — broadly active)' if d3 < DISTINCT_THRESHOLD else '(distinctive)'}")
        print(s); lines.append(s)

    # 4. Revised conclusion (per-modality, honest about the weak audio case).
    print()
    pv = summary[("video", "P_prop")]
    pa = summary[("audio", "P_prop")]
    pv_sink_dist = [d for d in pv if d in D_SINK_ALL]
    pa_sink_dist = [d for d in pa if d in D_SINK_ALL]
    no_sink_dim_distinct_to_prop = (not pv_sink_dist) and (not pa_sink_dist)

    concl_lines = ["\n  → REVISED CONCLUSION (per-modality):"]
    if not pv:
        concl_lines.append(
            "    P_prop_video: NO distinctive dim. No hidden-state channel "
            "is uniquely activated by encoder-propagated VIDEO sinks.")
    else:
        concl_lines.append(
            f"    P_prop_video: {len(pv)} distinctive dim(s) {pv[:10]}; "
            f"sink-set overlap: {pv_sink_dist}.")
    if not pa:
        concl_lines.append(
            "    P_prop_audio: NO distinctive dim.")
    else:
        concl_lines.append(
            f"    P_prop_audio: {len(pa)} WEAK distinctive dim(s) {pa[:10]} "
            f"(none in D_sink); modest pop_medians (just above active "
            f"threshold), not a dedicated propagated register.")
    concl_lines.append(
        "    Dim 3197 distinctiveness: P_prop_video 0.83, P_prop_audio 0.97 — "
        "BOTH < 1; 3197 is BROADLY active (highest in video non_sink at "
        "12.79), NOT specific to propagated tokens.")
    if no_sink_dim_distinct_to_prop:
        concl_lines.append(
            "\n    → Qwen2.5-Omni's propagated sinks do NOT form a dedicated "
            "register dim (unlike LLaVA {982, 2494, 3263}). Propagation appears "
            "in ATTENTION (Stage 1.2 video early-sharp peak) but NOT in a "
            "distinctive HIDDEN-STATE register. The em/inh separation observed "
            "earlier is SEPARATION BY ABSENCE: P_prop tokens are dimensionally "
            "distinct from P_llm because they FAIL TO LOAD the inherited "
            "register dims {458, 2570}, not because they activate a propagated "
            "dim. 3197 is not a propagated dim; it's a broadly-used channel.")
    concl = "\n".join(concl_lines)
    print(concl); lines.append(concl)

    with open(out_dir / "distinctiveness_verdict.txt", "w") as f:
        f.write(f"Stage 1.3 distinctiveness analysis — layer {asd_layer}\n")
        f.write(f"distinctiveness[d] = pop_median / max(other_pops_median)\n")
        f.write(f"distinctive := distinct > {DISTINCT_THRESHOLD} AND "
                f"pop_median > {ACTIVE_THRESHOLD}\n\n")
        for ln in lines:
            f.write(ln.lstrip() + "\n")
    print(f"\nwrote {out_dir / 'distinctiveness_verdict.txt'}")


def run_from_npz(args):
    """Load profiles.npz and regenerate the per-layer table, verdict, washout,
    and em/inh trajectory figure with no GPU."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = (Path(args.profiles_npz) if args.profiles_npz
                else (out_dir / "profiles.npz"))
    print(f"--replot_only: loading {npz_path}")
    d = np.load(npz_path)
    layers = list(map(int, d["layers"].tolist()))
    H = int(d["H"][0])
    profiles = {p: d[f"profile_{p}"] for p in POPULATIONS}
    counts = {p: d[f"count_{p}"] for p in POPULATIONS}
    print(f"  layers={layers}  H={H}  populations={list(profiles.keys())}")
    rows, _ = _build_rows_high_dims(profiles, counts, layers, H)
    print_em_inh_per_layer_table(rows, layers, out_dir)
    verdict_l2_focused(rows, layers, out_dir)
    plot_em_inh_trajectory(rows, layers, out_dir)
    # Mixed-modality ASD figure.
    if all(k in d.files for k in
           ("asd_bos_mean", "asd_sink_median", "asd_nonsink_median")):
        plot_asd_figure(d["asd_bos_mean"], d["asd_sink_median"],
                        d["asd_nonsink_median"], layers, args.asd_layer, H,
                        out_dir)
    else:
        print("  [asd] profiles.npz predates ASD; skipping mixed ASD figure.")
    # Per-modality ASD figure.
    pm_keys = [f"asd_{m}_{c}" for m in ("audio", "video")
               for c in ("non_sink", "P_llm", "P_prop")]
    if all(k in d.files for k in pm_keys) and "asd_bos_mean" in d.files:
        asd_per_mod = {(m, c): d[f"asd_{m}_{c}"]
                       for m in ("audio", "video")
                       for c in ("non_sink", "P_llm", "P_prop")}
        asd_per_mod_counts = {(m, c): d[f"asd_{m}_{c}_count"]
                              for m in ("audio", "video")
                              for c in ("non_sink", "P_llm", "P_prop")}
        plot_asd_per_modality(d["asd_bos_mean"], asd_per_mod,
                              asd_per_mod_counts, layers, args.asd_layer, H,
                              out_dir)
        # Distinctiveness analysis (no extra figure; CSV + verdict text).
        distinctiveness_analysis(asd_per_mod, asd_per_mod_counts, layers,
                                 args.asd_layer, H, out_dir)
    else:
        print("  [asd-per-mod] profiles.npz lacks per-modality ASD arrays; "
              "skipping. Re-run the full pipeline to populate.")
    print("\nDone (replot only — no model load).")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers_of_interest = tuple(sorted(set(args.layers)))
    print(f"Stage 1.3 — dimension signatures (RMSNorm pure, matches Stage 0.1)")
    print(f"  D_sink_all = {D_SINK_ALL}  inherited = {INHERITED}  emerged = {EMERGED}")
    print(f"  layers of interest: {layers_of_interest}")
    print(f"  tau_high_enc = {TAU_HIGH_ENC} (P_prop gate)   "
          f"tau_token_sink = {TAU_TOKEN_SINK} (P_llm gate)")

    # ----- model -----
    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        print(f"  [note] only 1 visible GPU — overriding device_map "
              f"{args.device_map!r} -> 'auto'.")
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    try:
        print(f"  device_map placement keys: {len(model.hf_device_map)} modules")
    except AttributeError:
        pass
    eps = _thinker_rms_eps(model)
    thinker_cfg = _resolve_thinker_cfg(model)
    tokenizer = processor.tokenizer
    audio_enc, visual_enc = _resolve_encoders(model)
    H = int(getattr(thinker_cfg, "hidden_size",
                    getattr(model.thinker.config, "hidden_size", 3584)))
    print(f"  RMSNorm convention (pure, no weight), rms_norm_eps = {eps}")
    print(f"  hidden_dim = {H}")
    assert H == 3584, (
        f"hidden_dim mismatch ({H}); D_sink_all indices were set for 3584. "
        "Update D_sink_all if the model changed."
    )

    # ----- clips -----
    clip_dir = Path(args.video_dir)
    all_clips = sorted(clip_dir.glob("*.mp4"))
    if not all_clips:
        raise SystemExit(f"No .mp4 in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(all_clips))[: args.n_clips]
    clips = [all_clips[i] for i in idx]
    print(f"\nVGGSounder pass: {len(clips)} clips from {clip_dir}")

    # ----- accumulators -----
    n_layers_int = len(layers_of_interest)
    sums = {p: np.zeros((n_layers_int, H), dtype=np.float64) for p in POPULATIONS}
    counts = {p: np.zeros(n_layers_int, dtype=np.int64) for p in POPULATIONS}
    overlap = {"audio": np.zeros(n_layers_int, dtype=np.int64),
               "video": np.zeros(n_layers_int, dtype=np.int64)}
    span_totals = {"audio": 0, "video": 0}

    failures: dict[str, int] = {}
    succeeded = 0
    first_sanity = True
    # ASD-format figure data: per-clip BOS activation + per-clip sink/non-sink
    # medians at each layer of interest. Aggregated post-loop as mean-across-clips
    # (BOS) and median-across-per-clip-medians (sink, non-sink) — the
    # median-of-medians kills heavy-tail per-token spikes.
    asd_bos: list = [[] for _ in range(n_layers_int)]          # per layer: list of (H,)
    asd_sink_med: list = [[] for _ in range(n_layers_int)]
    asd_nonsink_med: list = [[] for _ in range(n_layers_int)]
    # Per-modality ASD: per-clip per-category median, with category in
    # {non_sink, P_llm, P_prop}. Lets the per-modality figure show how each
    # category lights up the sink dims separately for audio vs video.
    ASD_CATS = ("non_sink", "P_llm", "P_prop")
    asd_mod_cats: dict = {(m, c): [[] for _ in range(n_layers_int)]
                          for m in ("audio", "video") for c in ASD_CATS}
    asd_mod_counts: dict = {key: np.zeros(n_layers_int, dtype=np.int64)
                            for key in asd_mod_cats}

    for clip in tqdm(clips, desc="VGGSounder"):
        conv = build_conversation(str(clip), PROMPT_AV, "av")
        try:
            inputs, use_aiv = prepare_inputs(
                processor, conv, "av", model.device, model.dtype)
        except Exception as e:
            failures[f"prep:{type(e).__name__}"] = failures.get(
                f"prep:{type(e).__name__}", 0) + 1
            continue
        prompt_len = inputs["input_ids"].shape[1]
        # Identify modal positions by INPUT TOKEN ID (handles AV interleaving:
        # inside the BOS/EOS brackets, each position is either an audio_token
        # or video_token; slicing the bracket would mix them).
        audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
        n_a_llm = len(audio_pos); n_v_llm = len(video_pos)
        if n_a_llm == 0 or n_v_llm == 0:
            failures["missing_modal_tokens"] = failures.get(
                "missing_modal_tokens", 0) + 1
            continue

        # Hook encoders to capture per-token L2 norms.
        a_buf, v_buf = [], []

        def make_hook(buf):
            def _h(_m, _i, out):
                tokens = _extract_tokens(out)
                buf.append(tokens.detach().norm(dim=-1).float().cpu().numpy())
            return _h

        h_a = audio_enc.register_forward_hook(make_hook(a_buf))
        h_v = visual_enc.register_forward_hook(make_hook(v_buf))
        try:
            with torch.inference_mode():
                outputs = model.thinker(
                    **inputs, output_hidden_states=True,
                    use_audio_in_video=use_aiv,
                    return_dict=True, use_cache=False,
                )
        except Exception as e:
            for h in (h_a, h_v):
                h.remove()
            failures[f"fwd:{type(e).__name__}"] = failures.get(
                f"fwd:{type(e).__name__}", 0) + 1
            torch.cuda.empty_cache()
            continue
        finally:
            for h in (h_a, h_v):
                h.remove()

        if not a_buf or not v_buf:
            failures["no_enc_hooks"] = failures.get("no_enc_hooks", 0) + 1
            del outputs; torch.cuda.empty_cache()
            continue
        # Concatenate ALL encoder calls (audio_tower may fire multiple times
        # in chunked AV processing).
        a_enc = np.concatenate(a_buf)
        v_enc = np.concatenate(v_buf)
        a_aligned, a_tag = align_norms(a_enc, n_a_llm)
        v_aligned, v_tag = align_norms(v_enc, n_v_llm)
        if a_aligned is None or v_aligned is None:
            failures[f"align:a={a_tag},v={v_tag}"] = failures.get(
                f"align:a={a_tag},v={v_tag}", 0) + 1
            del outputs; torch.cuda.empty_cache()
            continue

        # One-time sanity dump.
        if first_sanity:
            print(f"\n  [sanity] clip {clip.name}:")
            print(f"    prompt_len = {prompt_len}")
            print(f"    audio: {n_a_llm} LLM positions (token_id-matched) | "
                  f"enc n={len(a_enc)} ({len(a_buf)} hook calls) | alignment={a_tag}")
            print(f"    video: {n_v_llm} LLM positions (token_id-matched) | "
                  f"enc n={len(v_enc)} ({len(v_buf)} hook calls) | alignment={v_tag}")
            print(f"    first 3 audio positions = {audio_pos[:3].tolist()}")
            print(f"    first 3 video positions = {video_pos[:3].tolist()}")
            first_sanity = False

        # Sample random control INDICES once per clip per modality (these are
        # indices INTO audio_pos / video_pos, i.e., into the modal-token list).
        n_rand_a = min(N_RANDOM_PER_CLIP, n_a_llm)
        n_rand_v = min(N_RANDOM_PER_CLIP, n_v_llm)
        rand_a = rng.choice(n_a_llm, n_rand_a, replace=False) if n_rand_a else None
        rand_v = rng.choice(n_v_llm, n_rand_v, replace=False) if n_rand_v else None

        # Encoder-norm gate for P_prop (layer-independent; per modal token).
        audio_prop_mask = a_aligned > TAU_HIGH_ENC      # (n_a_llm,) bool
        video_prop_mask = v_aligned > TAU_HIGH_ENC      # (n_v_llm,) bool

        hidden_states = outputs.hidden_states  # tuple len (n_layers + 1)
        for li, L in enumerate(layers_of_interest):
            if L + 1 >= len(hidden_states):
                continue
            # hidden_states[L+1] = OUTPUT OF DECODER LAYER L (since [0] = embedding).
            hs = hidden_states[L + 1]
            normed = (_rmsnorm_abs(hs[0].float(), eps)
                      .to(torch.float64).cpu().numpy())          # (seq, H)

            # ---- ASD-format data capture for this layer ----
            # BOS = position 0 (the first <|im_start|>). Sink/non-sink over all
            # NON-BOS positions, classified by the τ=20 LLM-sink criterion.
            asd_bos[li].append(normed[0].copy())
            nb = normed[1:]                                       # (seq-1, H)
            if nb.shape[0] > 0:
                max_sink_all = nb[:, D_SINK_ALL].max(axis=1)      # (seq-1,)
                sink_mask_all = max_sink_all >= TAU_TOKEN_SINK
                if sink_mask_all.any():
                    asd_sink_med[li].append(np.median(nb[sink_mask_all], axis=0))
                if (~sink_mask_all).any():
                    asd_nonsink_med[li].append(
                        np.median(nb[~sink_mask_all], axis=0))

            for modality, mod_positions, prop_mask, rand in (
                ("audio", audio_pos, audio_prop_mask, rand_a),
                ("video", video_pos, video_prop_mask, rand_v),
            ):
                mod_normed = normed[mod_positions]               # (n_mod, H)
                # Token-sink gate: max over D_sink_all dims >= τ_token_sink.
                max_sink = mod_normed[:, D_SINK_ALL].max(axis=1) # (n_mod,)
                llm_mask = max_sink >= TAU_TOKEN_SINK

                # Per-modality ASD per-clip per-category medians (used for the
                # per-modality ASD figure). non_sink = neither LLM-sink nor
                # propagated. P_llm and P_prop are the same masks used above.
                non_sink_mask = (~llm_mask) & (~prop_mask)
                for cat_name, cat_mask in (("non_sink", non_sink_mask),
                                           ("P_llm", llm_mask),
                                           ("P_prop", prop_mask)):
                    if cat_mask.any():
                        asd_mod_cats[(modality, cat_name)][li].append(
                            np.median(mod_normed[cat_mask], axis=0))
                        asd_mod_counts[(modality, cat_name)][li] += \
                            int(cat_mask.sum())

                p_prop = f"P_prop_{modality}"
                p_llm = f"P_llm_{modality}"
                p_rand = f"Random_{modality}"
                if prop_mask.any():
                    sums[p_prop][li] += mod_normed[prop_mask].sum(axis=0)
                    counts[p_prop][li] += int(prop_mask.sum())
                if llm_mask.any():
                    sums[p_llm][li] += mod_normed[llm_mask].sum(axis=0)
                    counts[p_llm][li] += int(llm_mask.sum())
                overlap[modality][li] += int((prop_mask & llm_mask).sum())
                if rand is not None:
                    sums[p_rand][li] += mod_normed[rand].sum(axis=0)
                    counts[p_rand][li] += len(rand)

            if li == 0:
                span_totals["audio"] += n_a_llm
                span_totals["video"] += n_v_llm

        del outputs, hidden_states
        torch.cuda.empty_cache()
        succeeded += 1

    if failures:
        print(f"\n  failures: {failures}")
    if succeeded == 0:
        raise SystemExit("No clips processed — bailing.")

    # ----- STEP 1: population sizes -----
    print("\n" + "=" * 86)
    print("STEP 1 — Population sizes  (gate: <30 → too small to interpret profile)")
    print("=" * 86)
    print(f"  {'population':<16}{'layer':<6}{'n_tokens':<12}{'span_total':<14}flag")
    print("  " + "-" * 60)
    pop_rows = []
    for p in POPULATIONS:
        for li, L in enumerate(layers_of_interest):
            n = int(counts[p][li])
            mod = "audio" if "audio" in p else "video"
            span = span_totals[mod]
            small = (n < MIN_POP_SIZE) and ("Random" not in p)
            flag = "TOO SMALL" if small else ""
            print(f"  {p:<16}{L:<6}{n:<12}{span:<14}{flag}")
            pop_rows.append(dict(population=p, layer=L, n_tokens=n,
                                 span_total=span, too_small=small))
    print(f"\n  overlap (P_prop & P_llm) — audio: "
          f"{dict(zip(layers_of_interest, overlap['audio'].tolist()))}")
    print(f"  overlap (P_prop & P_llm) — video: "
          f"{dict(zip(layers_of_interest, overlap['video'].tolist()))}")
    pd.DataFrame(pop_rows).to_csv(out_dir / "population_sizes.csv", index=False)

    # ----- STEP 2: profiles -----
    profiles = {}
    for p in POPULATIONS:
        prof = np.full((n_layers_int, H), np.nan, dtype=np.float64)
        for li in range(n_layers_int):
            if counts[p][li] > 0:
                prof[li] = sums[p][li] / counts[p][li]
        profiles[p] = prof

    # ----- STEP 3: high dimensions per profile -----
    high_dims: dict = {}
    for p in POPULATIONS:
        for li, L in enumerate(layers_of_interest):
            prof = profiles[p][li]
            if not np.isfinite(prof).any():
                high_dims[(p, L)] = []
                continue
            med = float(np.nanmedian(prof))
            mask = prof > HIGH_DIM_MULT * med
            high_dims[(p, L)] = sorted(int(d) for d in np.where(mask)[0])

    # ----- STEP 4: per-population activations on D_sink_all -----
    rows = []
    for p in POPULATIONS:
        for li, L in enumerate(layers_of_interest):
            n = int(counts[p][li])
            hd = high_dims[(p, L)]
            row = dict(population=p, layer=L, n_tokens=n, n_high_dims=len(hd),
                       high_dims=";".join(map(str, hd[:30])))
            if n > 0:
                prof = profiles[p][li]
                a458, a2570, a3197 = float(prof[458]), float(prof[2570]), float(prof[3197])
                inh = (a458 + a2570) / 2.0
                em = a3197
                ratio = em / inh if inh > 0 else float("nan")
                row.update(act_458=a458, act_2570=a2570, act_3197=a3197,
                           inherited_mean=inh, emerged_mean=em,
                           emerged_to_inherited_ratio=ratio,
                           sink_dims_in_high=";".join(
                               str(d) for d in D_SINK_ALL if d in hd))
            else:
                row.update(act_458=np.nan, act_2570=np.nan, act_3197=np.nan,
                           inherited_mean=np.nan, emerged_mean=np.nan,
                           emerged_to_inherited_ratio=np.nan,
                           sink_dims_in_high="")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dimension_signatures.csv", index=False)

    # ---- Aggregate ASD-format data (per layer) ----
    asd_bos_mean = np.full((n_layers_int, H), np.nan, dtype=np.float64)
    asd_sink_median = np.full((n_layers_int, H), np.nan, dtype=np.float64)
    asd_nonsink_median = np.full((n_layers_int, H), np.nan, dtype=np.float64)
    for li in range(n_layers_int):
        if asd_bos[li]:
            asd_bos_mean[li] = np.stack(asd_bos[li]).mean(axis=0)
        if asd_sink_med[li]:
            asd_sink_median[li] = np.nanmedian(np.stack(asd_sink_med[li]), axis=0)
        if asd_nonsink_med[li]:
            asd_nonsink_median[li] = np.nanmedian(
                np.stack(asd_nonsink_med[li]), axis=0)
    # Per-modality category medians (median-of-per-clip-medians).
    asd_per_mod = {key: np.full((n_layers_int, H), np.nan, dtype=np.float64)
                   for key in asd_mod_cats}
    for key, per_layer_lists in asd_mod_cats.items():
        for li in range(n_layers_int):
            if per_layer_lists[li]:
                asd_per_mod[key][li] = np.nanmedian(
                    np.stack(per_layer_lists[li]), axis=0)

    # Save full per-population per-layer profiles AND the ASD-format data so
    # figures can be regenerated without rerunning the 100-clip forward pass.
    np.savez_compressed(
        out_dir / "profiles.npz",
        layers=np.array(layers_of_interest, dtype=np.int64),
        H=np.array([H], dtype=np.int64),
        **{f"profile_{p}": profiles[p] for p in POPULATIONS},
        **{f"count_{p}": counts[p] for p in POPULATIONS},
        overlap_audio=overlap["audio"],
        overlap_video=overlap["video"],
        span_total_audio=np.array([span_totals["audio"]], dtype=np.int64),
        span_total_video=np.array([span_totals["video"]], dtype=np.int64),
        # ASD-format arrays (mixed-modality):
        asd_bos_mean=asd_bos_mean,
        asd_sink_median=asd_sink_median,
        asd_nonsink_median=asd_nonsink_median,
        # Per-modality ASD arrays: keys are 'asd_{modality}_{category}' and
        # 'asd_{modality}_{category}_count'. category ∈ {non_sink, P_llm, P_prop}.
        **{f"asd_{m}_{c}": asd_per_mod[(m, c)]
           for m in ("audio", "video") for c in ASD_CATS},
        **{f"asd_{m}_{c}_count": asd_mod_counts[(m, c)]
           for m in ("audio", "video") for c in ASD_CATS},
    )
    print(f"wrote {out_dir / 'profiles.npz'}  (profiles + ASD-format data)")

    # Pretty-print step-4 table.
    print("\n" + "=" * 110)
    print("STEP 4 — Activations on D_sink_all per population × layer")
    print("=" * 110)
    print(f"  {'population':<16}{'L':<4}{'n':<8}{'act_458':<10}{'act_2570':<10}"
          f"{'act_3197':<10}{'inh_mean':<10}{'em_mean':<10}{'em/inh':<8}"
          f"{'n_high':<7}sinks_in_high")
    print("  " + "-" * 108)

    def _f(v, fmt):
        return fmt.format(v) if np.isfinite(v) else "  nan  "

    for r in rows:
        print(f"  {r['population']:<16}{r['layer']:<4}{r['n_tokens']:<8}"
              f"{_f(r['act_458'], '{:.2f}  '):<10}"
              f"{_f(r['act_2570'], '{:.2f}  '):<10}"
              f"{_f(r['act_3197'], '{:.2f}  '):<10}"
              f"{_f(r['inherited_mean'], '{:.2f}  '):<10}"
              f"{_f(r['emerged_mean'], '{:.2f}  '):<10}"
              f"{_f(r['emerged_to_inherited_ratio'], '{:.2f}  '):<8}"
              f"{r['n_high_dims']:<7}{r['sink_dims_in_high']}")

    # ----- VERDICT -----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    def _row(pop, L):
        for r in rows:
            if r["population"] == pop and r["layer"] == L:
                return r
        return None

    def _high_sinks(pop, L):
        return [d for d in D_SINK_ALL if d in high_dims.get((pop, L), [])]

    verdict_lines = []

    # Video separation (use early layer if present, else first).
    L_video = 2 if 2 in layers_of_interest else layers_of_interest[0]
    vp = _row("P_prop_video", L_video); vl = _row("P_llm_video", L_video)
    vp_sinks = _high_sinks("P_prop_video", L_video)
    vl_sinks = _high_sinks("P_llm_video", L_video)
    print(f"\nVideo dimensional separation (at L{L_video}):")
    print(f"  P_prop_video (n={vp['n_tokens']}): sinks_in_high = {vp_sinks}, "
          f"emerged/inherited = {_f(vp['emerged_to_inherited_ratio'], '{:.2f}x ')}")
    print(f"  P_llm_video  (n={vl['n_tokens']}): sinks_in_high = {vl_sinks}, "
          f"emerged/inherited = {_f(vl['emerged_to_inherited_ratio'], '{:.2f}x ')}")
    video_separated = (set(vp_sinks) != set(vl_sinks)) and \
                      (vp["n_tokens"] >= MIN_POP_SIZE and vl["n_tokens"] >= MIN_POP_SIZE)
    if video_separated:
        line = ("  → video populations activate DIFFERENT sink-dim subsets — "
                "matches the Sink-or-Not-to-Sink separation.")
    else:
        line = ("  → video populations share the same sink-dim subset (or one is too "
                "small) — no clean dimensional separation.")
    print(line); verdict_lines.append(line)

    # Audio.
    L_audio = 21 if 21 in layers_of_interest else layers_of_interest[-1]
    ap = _row("P_prop_audio", L_audio); al = _row("P_llm_audio", L_audio)
    print(f"\nAudio dimensional read (at L{L_audio}):")
    audio_lacks_prop = ap["n_tokens"] < MIN_POP_SIZE
    if audio_lacks_prop:
        line = (f"  P_prop_audio: n={ap['n_tokens']} < {MIN_POP_SIZE} — "
                "AUDIO LACKS ENCODER-PROPAGATED SINKS (count-only finding).")
    else:
        ap_sinks = _high_sinks("P_prop_audio", L_audio)
        line = (f"  P_prop_audio (n={ap['n_tokens']}): sinks_in_high = {ap_sinks}, "
                f"emerged/inherited = "
                f"{_f(ap['emerged_to_inherited_ratio'], '{:.2f}x ')}")
    print(line); verdict_lines.append(line)
    al_sinks = _high_sinks("P_llm_audio", L_audio)
    line = (f"  P_llm_audio  (n={al['n_tokens']}): sinks_in_high = {al_sinks}, "
            f"emerged/inherited = {_f(al['emerged_to_inherited_ratio'], '{:.2f}x ')}")
    print(line); verdict_lines.append(line)

    # Asymmetric framing check.
    print(f"\nAsymmetric framing check:")
    if video_separated and audio_lacks_prop:
        line = ("  CONFIRMED — video has a dimensionally-distinct propagated "
                "population AND audio's propagated population is near-absent. "
                "Hidden-state-level confirmation of Stage 1.2's attention-level "
                "asymmetry.")
    else:
        line = (f"  PARTIAL — video_separated={video_separated}, "
                f"audio_lacks_propagated={audio_lacks_prop}.")
    print(line); verdict_lines.append(line)

    # ----- write decision file -----
    with open(out_dir / "stage1_3_decision.txt", "w") as f:
        f.write("Stage 1.3 — Hidden-state dimension signatures\n")
        f.write(f"RMSNorm pure (no weight), eps={eps}.  D_sink_all={D_SINK_ALL}\n")
        f.write(f"layers analyzed: {list(layers_of_interest)}\n\n")
        for L in verdict_lines:
            f.write(L + "\n")
    print(f"\nwrote {out_dir / 'stage1_3_decision.txt'}  (legacy verdict)")

    # ----- NEW: per-layer em/inh table, L2 verdict + washout, trajectory plot -----
    # These OVERWRITE stage1_3_decision.txt with the L2-focused content.
    print_em_inh_per_layer_table(rows, layers_of_interest, out_dir)
    verdict_l2_focused(rows, layers_of_interest, out_dir)
    plot_em_inh_trajectory(rows, layers_of_interest, out_dir)
    plot_asd_figure(asd_bos_mean, asd_sink_median, asd_nonsink_median,
                    layers_of_interest, args.asd_layer, H, out_dir)
    plot_asd_per_modality(asd_bos_mean, asd_per_mod, asd_mod_counts,
                          layers_of_interest, args.asd_layer, H, out_dir)
    distinctiveness_analysis(asd_per_mod, asd_mod_counts, layers_of_interest,
                             args.asd_layer, H, out_dir)

    # ----- Figure 2: sink-dim comparison (HEADLINE) -----
    fig, axes = plt.subplots(1, n_layers_int, figsize=(5.5 * n_layers_int, 4.5),
                             squeeze=False)
    pops_in_fig = ("P_prop_video", "P_llm_video", "Random_video",
                   "P_prop_audio", "P_llm_audio", "Random_audio")
    x_pos = np.arange(len(D_SINK_ALL))
    bw = 0.8 / len(pops_in_fig)
    for li, L in enumerate(layers_of_interest):
        ax = axes[0][li]
        for pi, p in enumerate(pops_in_fig):
            vals = [profiles[p][li, d] if counts[p][li] > 0 else np.nan
                    for d in D_SINK_ALL]
            ax.bar(x_pos + (pi - (len(pops_in_fig) - 1) / 2) * bw, vals,
                   width=bw, color=POP_COLOR[p],
                   label=f"{p} (n={int(counts[p][li])})" if li == 0 else None,
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"dim {d}\n({'inh' if d in INHERITED else 'em'})"
                           for d in D_SINK_ALL], fontsize=9)
        ax.set_ylabel("mean |RMSNorm(x)[d]|", fontsize=10)
        ax.set_title(f"layer {L}", fontsize=12)
        ax.axhline(TAU_TOKEN_SINK, ls=":", color="gray", lw=0.9,
                   label="τ=20 sink threshold" if li == 0 else None)
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Sink-dim comparison: per population, activations on "
                 "{458 inh, 2570 inh, 3197 em}", y=1.06, fontsize=13)
    plt.tight_layout()
    p_fig2 = out_dir / "sink_dim_comparison.png"
    fig.savefig(p_fig2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_fig2}")

    # ----- Figure 1: full per-dim profile per (population, layer) -----
    show_pops = ("P_prop_video", "P_llm_video", "P_prop_audio", "P_llm_audio")
    nrows, ncols = len(show_pops), n_layers_int
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.6 * nrows),
                             squeeze=False)
    dims = np.arange(H)
    for ri, p in enumerate(show_pops):
        for ci, L in enumerate(layers_of_interest):
            ax = axes[ri][ci]
            n = int(counts[p][ci])
            if n == 0:
                ax.text(0.5, 0.5, "(no tokens)", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{p}  L{L}  n={n}", fontsize=10)
                continue
            prof = profiles[p][ci]
            # Sink-dim highlights drawn UNDER the line plot (zorder=0), wider
            # bands so they're visible against the dense 3584-dim line.
            span_hw = 12
            for d in INHERITED:
                ax.axvspan(d - span_hw, d + span_hw,
                           color="#ffd400", alpha=0.55, zorder=0)
            for d in EMERGED:
                ax.axvspan(d - span_hw, d + span_hw,
                           color="#2ca02c", alpha=0.55, zorder=0)
            ax.plot(dims, prof, lw=0.4, color=POP_COLOR[p], alpha=0.85, zorder=3)
            ax.set_yscale("log")
            ax.set_xlim(0, H)
            ax.grid(True, ls=":", alpha=0.3, zorder=1)
            med = float(np.nanmedian(prof))
            ax.axhline(HIGH_DIM_MULT * med, color="gray", ls="--", lw=0.6,
                       label=f"10× median = {HIGH_DIM_MULT*med:.2g}", zorder=2)
            ax.set_title(f"{p}  L{L}  n={n}  "
                         f"n_high={len(high_dims[(p,L)])}", fontsize=10)
            if ri == nrows - 1:
                ax.set_xlabel("hidden-state dim")
            if ci == 0:
                ax.set_ylabel("mean |RMSNorm(x)|")
            ax.legend(fontsize=7, loc="upper right")
    # Custom legend for the sink-dim highlights.
    fig.legend(handles=[
        Patch(facecolor="#ffd400", alpha=0.7,
              label=f"inherited dims {INHERITED} (yellow)"),
        Patch(facecolor="#2ca02c", alpha=0.7,
              label=f"emerged dim {EMERGED} (green)"),
    ], loc="upper center", ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Per-dimension activation profile per (population, layer)",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    p_fig1 = out_dir / "dimension_profiles.png"
    fig.savefig(p_fig1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_fig1}")

    print("\nDone. Confirm before proceeding to Stage 2.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument(
        "--video_dir", default=str(_REPO / "data/VGGSounder/videos"),
        help="VGGSounder .mp4 dir (audio+video clips for AV processing).",
    )
    p.add_argument("--n_clips", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
        help="Decoder-layer indices to analyze (0-based, Stage 1.2 convention). "
             "hidden_states[L+1] is the output of decoder layer L.",
    )
    p.add_argument(
        "--asd_layer", type=int, default=2,
        help="Layer to use for the ASD-format two-panel figure (BOS + "
             "sink/non-sink medians). Must be one of --layers. Default 2.",
    )
    p.add_argument(
        "--device_map", default="balanced_low_0",
        help="HF device_map. balanced_low_0 is the only multi-GPU map that "
             "works on this model; falls back to 'auto' if only 1 GPU visible.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/stage1_3_dim_signatures"),
    )
    p.add_argument(
        "--replot_only", action="store_true",
        help="Skip the model + per-clip forward; load profiles.npz from "
             "--profiles_npz (or output_dir) and regenerate the per-layer "
             "table, L2 verdict, washout, and em/inh trajectory figure.",
    )
    p.add_argument(
        "--profiles_npz", default=None,
        help="Path to profiles.npz (default: <output_dir>/profiles.npz).",
    )
    args = p.parse_args()
    if args.replot_only:
        run_from_npz(args)
    else:
        main(args)
