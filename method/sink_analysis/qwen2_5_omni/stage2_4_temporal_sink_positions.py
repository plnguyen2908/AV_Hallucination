"""
stage2_4_temporal_sink_positions.py

Stage 2.4 — Temporal position of audio late-spread LLM-emerged sinks.

Question: are audio sinks at L21 (Stage 1.2's late-spread peak) positional
(fixed locations regardless of content), content-conditional (at acoustic
events, varying per clip), or distributed (no temporal structure)?

Inputs:
- 300 AudioSet clips, audio-only forward (modal_type="a").
- D_sink = {458, 2570}, τ=20, RMSNorm pure (no weight) — same as Stage 1.3/2.1.
- Primary layer: 21 (Stage 1.2 audio peak). Also captured at L26 for the
  deep-layer saturation comparison.

Notes:
- Qwen2.5-Omni audio is 25 tok/s after avg_pool (40 ms/token). The task spec
  mentions 20 ms; that's the Whisper pre-pool rate. We use 40 ms (matches
  Stage 0.2 bookkeeping: AudioSet 10 s clips → 250 audio tokens).
- "Position 0" = the FIRST audio token (in-span index 0, right after the
  <|audio_bos|> marker). Often called the "BOS-of-audio" candidate.

Procedure (per the spec):
  STEP 1  — Per-clip: forward thinker, hook L21/L26, slice sink mask to audio
            positions, record sink in-span indices + normalized positions.
  STEP 2  — Marginal histogram of pooled normalized sink positions (50 bins).
  STEP 3  — Per-clip distribution shape: KS vs uniform + IQR span across the
            clip's sink positions.
  STEP 4  — Per-clip sink count histogram.
  STEP 4B — Per-clip sink PROPORTION (sink_count / audio_span_length) and
            audio span length stats; plus Stage 2.1 cross-stage comparison
            at L21 (loads Stage 2.1's per_clip_counts.npz; if proportions
            match, the 100-vs-183 absolute-count gap is just span-length).
  STEP 5  — Position-0 sensitivity: re-do Step 2 excluding position-0 sinks.

Outputs (--output_dir):
  temporal_sink_distribution.png         Figure 1 (Step 2 marginal)
  per_clip_distribution_shape.png        Figure 2 (Step 3 KS/IQR)
  per_clip_count_distribution.png        Figure 3 (Step 4)
  per_clip_proportion_distribution.png   Figure 3B (Step 4B proportion + span)
  temporal_sink_distribution_no_pos0.png Figure 4 (Step 5)
  per_clip_temporal_stats.csv            per-clip stats (now incl. proportion)
  per_clip_temporal_arrays.npz           per-clip raw arrays (replot-friendly)
  stage2_4_decision.txt                  verdict text (incl. saturation
                                         diagnostic + S2.1 comparison)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scistats
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import (  # noqa: E402
    build_conversation, load_omni, prepare_inputs, thinker_layers,
)


D_SINK = [458, 2570]
TAU_SINK = 20.0
PROMPT_AUDIO = "Describe what you hear in detail."
TOKEN_DUR_SEC = 0.040          # 25 tok/s after avg_pool
PRIMARY_LAYER = 21             # Stage 1.2 audio late-spread peak
COMPARE_LAYER = 27             # late-saturation comparison (defaults to last)

POS0_POSITIONAL_THR = 0.70     # ≥70% pos-0 sink fraction → positional candidate
KS_HIGH_THR = 0.30             # median KS ≥ this → clip-level clustering
IQR_SMALL_THR = 0.20           # median IQR ≤ this → clip-level concentration
PROPORTION_CV_SAT_THR = 0.10   # std/mean(sink_proportion) < this → saturation
PROP_AGREE_TOL = 0.20          # |ratio-1| ≤ this between AudioSet/VGGSounder
                               # proportions → cross-stage AGREE (≤0.40 → ambiguous)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_start_token_id"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _thinker_rms_eps(model) -> float:
    cfg = model.thinker.config
    cfg = getattr(cfg, "text_config", cfg)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def _audio_positions(input_ids: torch.Tensor, thinker_cfg) -> np.ndarray:
    """Indices in the LLM sequence that are audio tokens (audio_token_index).
    For modal_type='a' clips this is a contiguous run after <|audio_bos|>."""
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    return np.where(ids == a_id)[0].astype(np.int64)


# --------------------------------------------------------------------------
# Per-clip processing
# --------------------------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers, eps,
                 d_sink_t, layer_indices):
    """Forward audio-only thinker, hook each layer in layer_indices, return
    per-layer sink mask over the audio span."""
    conv = build_conversation(str(clip_path), PROMPT_AUDIO, "a")
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, "a", model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}"

    audio_pos = _audio_positions(inputs["input_ids"], thinker_cfg)
    if len(audio_pos) == 0:
        return None, "no_audio"
    n_audio = int(len(audio_pos))
    a_pos_t = torch.from_numpy(audio_pos)
    sink_masks: dict = {L: None for L in layer_indices}

    def make_hook(L_idx):
        def _h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            x = hs[0].float()
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            normed_abs = (x / rms).abs()
            sink_act = normed_abs[:, d_sink_t].amax(dim=-1)
            sink_mask = (sink_act >= TAU_SINK)
            ap = a_pos_t.to(sink_mask.device)
            sink_masks[L_idx] = sink_mask[ap].cpu().numpy()
        return _h

    handles = [layers[L].register_forward_hook(make_hook(L))
               for L in layer_indices]
    try:
        with torch.inference_mode():
            model.thinker(**inputs, output_hidden_states=False,
                          use_audio_in_video=use_aiv,
                          return_dict=True, use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return None, f"fwd:{type(e).__name__}"
    for h in handles: h.remove()
    torch.cuda.empty_cache()

    if any(m is None for m in sink_masks.values()):
        return None, "no_hook"

    return {"n_audio": n_audio, "sink_masks": sink_masks}, None


def per_clip_stats(sink_mask: np.ndarray, n_audio: int) -> dict:
    """Per-clip stats from a (n_audio,) boolean sink mask."""
    sink_idx = np.where(sink_mask)[0]
    n_sink = int(len(sink_idx))
    if n_audio == 0:
        return None
    # Normalized to [0, 1] by within-span position / n_audio.
    sink_norm = sink_idx / max(n_audio, 1) if n_sink > 0 else np.array([])
    pos0 = bool(sink_mask[0])
    ks, iqr = float("nan"), float("nan")
    if n_sink >= 2:
        ks_res = scistats.kstest(sink_norm, "uniform")
        ks = float(ks_res.statistic)
        iqr = float(np.percentile(sink_norm, 75) - np.percentile(sink_norm, 25))
    return dict(n_audio=n_audio, n_sink=n_sink, sink_idx=sink_idx,
                sink_norm=sink_norm, pos0_is_sink=pos0, ks=ks, iqr=iqr)


# --------------------------------------------------------------------------
# Aggregation + plotting + verdict
# --------------------------------------------------------------------------

def _density(pooled, lo, hi):
    """Fraction of mass in [lo, hi] for the pooled normalized positions."""
    if pooled.size == 0:
        return float("nan")
    return float(((pooled >= lo) & (pooled < hi)).mean())


def plot_marginal(pooled, n_clips_used, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(pooled, bins=50, range=(0, 1), color="#1f77b4",
            edgecolor="black", linewidth=0.3, alpha=0.85)
    low = _density(pooled, 0.0, 0.05) * 100
    mid = _density(pooled, 0.4, 0.6) * 100
    high = _density(pooled, 0.95, 1.0001) * 100
    ax.set_xlabel("normalized within-span position", fontsize=11)
    ax.set_ylabel("# sink tokens (pooled across clips)", fontsize=11)
    ax.set_title(f"Audio L{PRIMARY_LAYER} sink positions — pooled marginal "
                 f"(n_clips={n_clips_used}{title_suffix})", fontsize=12)
    ax.text(0.5, 0.97,
            f"density: low (<0.05) = {low:.1f}%   "
            f"middle (0.4-0.6) = {mid:.1f}%   "
            f"high (>0.95) = {high:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85))
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return dict(low=low, mid=mid, high=high)


def plot_per_clip_shape(stats_l, out_path, layer):
    ks_arr = np.array([s["ks"] for s in stats_l if np.isfinite(s["ks"])])
    iqr_arr = np.array([s["iqr"] for s in stats_l if np.isfinite(s["iqr"])])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(ks_arr, bins=40, range=(0, 1), color="#2ca02c",
                 edgecolor="black", linewidth=0.3, alpha=0.85)
    axes[0].axvline(KS_HIGH_THR, ls="--", color="gray", lw=1,
                    label=f"KS={KS_HIGH_THR} (clustering threshold)")
    axes[0].set_xlabel("clip-level KS vs uniform", fontsize=11)
    axes[0].set_ylabel("# clips", fontsize=11)
    axes[0].set_title(f"L{layer} — per-clip KS  "
                      f"(median={np.median(ks_arr):.3f})", fontsize=12)
    axes[0].grid(True, ls=":", alpha=0.4, axis="y"); axes[0].legend(fontsize=9)

    axes[1].hist(iqr_arr, bins=40, range=(0, 1), color="#9467bd",
                 edgecolor="black", linewidth=0.3, alpha=0.85)
    axes[1].axvline(IQR_SMALL_THR, ls="--", color="gray", lw=1,
                    label=f"IQR={IQR_SMALL_THR} (concentration threshold)")
    axes[1].set_xlabel("clip-level IQR of sink positions", fontsize=11)
    axes[1].set_ylabel("# clips", fontsize=11)
    axes[1].set_title(f"L{layer} — per-clip IQR  "
                      f"(median={np.median(iqr_arr):.3f})", fontsize=12)
    axes[1].grid(True, ls=":", alpha=0.4, axis="y"); axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return float(np.median(ks_arr)), float(np.median(iqr_arr))


def plot_count_hist(counts, out_path, layer):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(counts, bins=40, color="#ff7f0e", edgecolor="black",
            linewidth=0.3, alpha=0.85)
    mn, sd = float(counts.mean()), float(counts.std())
    ax.axvline(mn, ls="-", color="black", lw=1.2,
               label=f"mean={mn:.1f}, std={sd:.1f}, std/mean={sd/max(mn,1e-9):.2f}")
    ax.set_xlabel("# sink tokens per clip", fontsize=11)
    ax.set_ylabel("# clips", fontsize=11)
    ax.set_title(f"L{layer} — per-clip sink count distribution", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, ls=":", alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_proportion_and_span(proportions, span_lens, out_path, layer):
    """Two-panel: per-clip sink PROPORTION (sink_count / audio_span_length) +
    per-clip audio span length distribution. The proportion plot's std/mean
    is the cleaner content-dependence diagnostic — span-length differences
    that confound the absolute count cancel out."""
    p = np.asarray(proportions, dtype=np.float64)
    sl = np.asarray(span_lens, dtype=np.float64)
    pm, ps = float(p.mean()), float(p.std()); pcv = ps / max(pm, 1e-12)
    sm, ss = float(sl.mean()), float(sl.std())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].hist(p, bins=40, color="#17becf", edgecolor="black",
                 linewidth=0.3, alpha=0.85)
    axes[0].axvline(pm, ls="-", color="black", lw=1.2,
                    label=f"mean={pm:.3f}, std={ps:.3f}, std/mean={pcv:.3f}")
    axes[0].set_xlabel("sink count / audio span length", fontsize=11)
    axes[0].set_ylabel("# clips", fontsize=11)
    axes[0].set_title(f"L{layer} — per-clip sink PROPORTION", fontsize=12)
    axes[0].legend(fontsize=10); axes[0].grid(True, ls=":", alpha=0.4, axis="y")

    axes[1].hist(sl, bins=40, color="#bcbd22", edgecolor="black",
                 linewidth=0.3, alpha=0.85)
    axes[1].axvline(sm, ls="-", color="black", lw=1.2,
                    label=f"mean={sm:.1f}, std={ss:.1f}, "
                          f"std/mean={ss/max(sm,1e-12):.3f}")
    axes[1].set_xlabel("audio span length (LLM audio tokens)", fontsize=11)
    axes[1].set_ylabel("# clips", fontsize=11)
    axes[1].set_title("per-clip audio span length distribution", fontsize=12)
    axes[1].legend(fontsize=10); axes[1].grid(True, ls=":", alpha=0.4, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return dict(prop_mean=pm, prop_std=ps, prop_cv=pcv,
                span_mean=sm, span_std=ss)


def stage21_audio_proportion(npz_path: Path, layer: int):
    """Load Stage 2.1's cached per_clip_counts.npz and compute the per-clip
    audio LLM-sink PROPORTION at the requested layer on VGGSounder.

    Stage 2.1 caches `audio_count` (n_clips, n_layers) and `n_a_llm` (n_clips,)
    — both with the same D_sink={458,2570} and τ=20 used here, so the
    proportions are directly comparable. Returns None if the file/layer is
    missing."""
    if not npz_path.is_file():
        return None
    with np.load(npz_path) as z:
        if "audio_count" not in z.files or "n_a_llm" not in z.files:
            return None
        ac = z["audio_count"]
        n_a = z["n_a_llm"]
        if layer >= ac.shape[1]:
            return None
        counts = ac[:, layer].astype(np.float64)
        spans = n_a.astype(np.float64)
        prop = counts / np.clip(spans, 1, None)
        pm, ps = float(prop.mean()), float(prop.std())
        sm, ss = float(spans.mean()), float(spans.std())
        return dict(
            n_clips=int(len(counts)),
            count_mean=float(counts.mean()), count_std=float(counts.std()),
            span_mean=sm, span_std=ss,
            prop_mean=pm, prop_std=ps,
            prop_cv=ps / max(pm, 1e-12),
        )


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    eps = _thinker_rms_eps(model)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    layer_indices = sorted(set(args.layers))
    for L in layer_indices:
        if L >= n_layers:
            raise SystemExit(f"layer {L} >= n_layers {n_layers}")
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ={TAU_SINK}, "
          f"layers={layer_indices}, primary=L{PRIMARY_LAYER}")

    clip_dir = Path(args.audio_dir)
    clips = sorted(clip_dir.glob("*.wav"))
    if not clips:
        raise SystemExit(f"No .wav in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips))[: args.n_clips]
    clips = [clips[i] for i in idx]
    print(f"\nAudioSet pass: {len(clips)} clips\n")

    per_clip = {L: [] for L in layer_indices}
    failures: dict = {}
    for clip in tqdm(clips, desc="AudioSet"):
        result, err = process_clip(model, processor, clip, thinker_cfg,
                                   layers, eps, d_sink_t, layer_indices)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            continue
        for L in layer_indices:
            s = per_clip_stats(result["sink_masks"][L], result["n_audio"])
            if s:
                per_clip[L].append(s)
    if failures:
        print(f"  failures: {failures}")

    if not per_clip[PRIMARY_LAYER]:
        raise SystemExit("No clips contributed at primary layer.")

    # ----- analysis at the PRIMARY layer (L21) -----
    L = PRIMARY_LAYER
    stats_L = per_clip[L]
    n_clips_used = len(stats_L)

    pooled = np.concatenate([s["sink_norm"] for s in stats_L if s["n_sink"] > 0])
    counts = np.array([s["n_sink"] for s in stats_L])
    pos0_frac = float(np.mean([s["pos0_is_sink"] for s in stats_L]))

    # Step 2 figure: marginal.
    dens = plot_marginal(pooled, n_clips_used,
                         out_dir / "temporal_sink_distribution.png")

    # Step 3: per-clip shape.
    ks_median, iqr_median = plot_per_clip_shape(
        stats_L, out_dir / "per_clip_distribution_shape.png", L)

    # Step 4: count distribution.
    plot_count_hist(counts, out_dir / "per_clip_count_distribution.png", L)

    # Step 4B: per-clip PROPORTION (sink_count / audio_span_length).
    # This is the cleaner content-dependence diagnostic — span-length variation
    # (the confound on absolute counts) cancels out.
    spans = np.array([s["n_audio"] for s in stats_L], dtype=np.int64)
    proportions = (counts.astype(np.float64)
                   / np.clip(spans.astype(np.float64), 1, None))
    prop_stats = plot_proportion_and_span(
        proportions, spans,
        out_dir / "per_clip_proportion_distribution.png", L)

    # Step 4B cross-stage: Stage 2.1's L21 audio sink proportion on VGGSounder.
    s21 = stage21_audio_proportion(Path(args.stage21_npz), PRIMARY_LAYER)

    # Step 5: marginal excluding position 0.
    pooled_no0 = np.concatenate([
        (s["sink_norm"][s["sink_idx"] != 0] if s["n_sink"] > 0 else np.array([]))
        for s in stats_L
    ])
    dens_no0 = plot_marginal(
        pooled_no0, n_clips_used,
        out_dir / "temporal_sink_distribution_no_pos0.png",
        title_suffix=", excluding position 0")

    # ----- comparison at L26 (deep saturation) -----
    if COMPARE_LAYER in per_clip and per_clip[COMPARE_LAYER]:
        stats_LC = per_clip[COMPARE_LAYER]
        pooled_LC = np.concatenate([
            s["sink_norm"] for s in stats_LC if s["n_sink"] > 0])
        plot_marginal(
            pooled_LC, len(stats_LC),
            out_dir / f"temporal_sink_distribution_L{COMPARE_LAYER}.png",
            title_suffix=f" — comparison layer L{COMPARE_LAYER}")

    # ----- per-clip CSV -----
    rows = []
    for s in stats_L:
        rows.append(dict(
            n_audio=s["n_audio"], n_sink=s["n_sink"],
            sink_proportion=(s["n_sink"] / s["n_audio"]
                             if s["n_audio"] > 0 else float("nan")),
            pos0_is_sink=int(s["pos0_is_sink"]),
            ks=s["ks"], iqr=s["iqr"],
            first_sink_norm=(float(s["sink_norm"][0])
                             if s["n_sink"] > 0 else float("nan")),
        ))
    pd.DataFrame(rows).to_csv(out_dir / "per_clip_temporal_stats.csv",
                              index=False)
    print(f"wrote {out_dir / 'per_clip_temporal_stats.csv'}")

    # Save raw arrays per layer for future replot.
    np.savez_compressed(
        out_dir / "per_clip_temporal_arrays.npz",
        **{f"L{L_}_sink_idx_offsets": np.cumsum([0] + [s["n_sink"] for s in per_clip[L_]])
           for L_ in layer_indices},
        **{f"L{L_}_sink_idx_flat": (np.concatenate([s["sink_idx"] for s in per_clip[L_]])
                                     if per_clip[L_] else np.array([], dtype=np.int64))
           for L_ in layer_indices},
        **{f"L{L_}_n_audio": np.array([s["n_audio"] for s in per_clip[L_]],
                                       dtype=np.int64)
           for L_ in layer_indices},
        **{f"L{L_}_n_sink": np.array([s["n_sink"] for s in per_clip[L_]],
                                      dtype=np.int64)
           for L_ in layer_indices},
        **{f"L{L_}_pos0": np.array([int(s["pos0_is_sink"]) for s in per_clip[L_]],
                                    dtype=np.int8)
           for L_ in layer_indices},
        **{f"L{L_}_ks": np.array([s["ks"] for s in per_clip[L_]])
           for L_ in layer_indices},
        **{f"L{L_}_iqr": np.array([s["iqr"] for s in per_clip[L_]])
           for L_ in layer_indices},
    )
    print(f"wrote {out_dir / 'per_clip_temporal_arrays.npz'}")

    # ----- verdict -----
    print("\n" + "=" * 86)
    print(f"STAGE 2.4 VERDICT  (layer L{PRIMARY_LAYER}, n_clips={n_clips_used}, "
          f"AudioSet, D_sink={{458, 2570}})")
    print("=" * 86)
    lines = []
    s = f"  total pooled sink positions: {pooled.size}"
    print(s); lines.append(s)
    s = f"  per-clip sink count: mean={counts.mean():.1f}, std={counts.std():.1f}, std/mean={counts.std()/max(counts.mean(),1e-9):.2f}"
    print(s); lines.append(s)
    s = (f"  per-clip audio span length: mean={prop_stats['span_mean']:.1f}, "
         f"std={prop_stats['span_std']:.1f}, "
         f"std/mean={prop_stats['span_std']/max(prop_stats['span_mean'],1e-9):.3f}")
    print(s); lines.append(s)
    s = (f"  per-clip sink PROPORTION (count/span): "
         f"mean={prop_stats['prop_mean']:.3f}, std={prop_stats['prop_std']:.3f}, "
         f"std/mean={prop_stats['prop_cv']:.3f}")
    print(s); lines.append(s)
    s = f"  pos-0 sink fraction (clips where in-span position 0 is a sink): {pos0_frac*100:.1f}%"
    print(s); lines.append(s)
    s = f"  marginal density low (<0.05) = {dens['low']:.1f}%, middle (0.4-0.6) = {dens['mid']:.1f}%, high (>0.95) = {dens['high']:.1f}%"
    print(s); lines.append(s)
    s = f"  excluding pos 0: low (<0.05) = {dens_no0['low']:.1f}%, middle (0.4-0.6) = {dens_no0['mid']:.1f}%, high (>0.95) = {dens_no0['high']:.1f}%"
    print(s); lines.append(s)
    s = f"  per-clip distribution shape: median KS = {ks_median:.3f}, median IQR = {iqr_median:.3f}"
    print(s); lines.append(s)

    # Cross-stage proportion comparison (AudioSet here vs VGGSounder in S2.1).
    # Resolves the apparent 100-vs-183 absolute-count gap: if proportions agree,
    # the gap is span-length only (VGGSounder audio span < AudioSet 10 s).
    if s21 is not None:
        s = (f"  Stage 2.1 comparison (VGGSounder, L{PRIMARY_LAYER}, "
             f"n_clips={s21['n_clips']}, D_sink={{458, 2570}}, τ=20):")
        print(s); lines.append(s)
        s = (f"    audio sink count: mean={s21['count_mean']:.1f} "
             f"(std={s21['count_std']:.1f}); "
             f"audio span: mean={s21['span_mean']:.1f} "
             f"(std={s21['span_std']:.1f}); "
             f"PROPORTION: mean={s21['prop_mean']:.3f}, "
             f"std={s21['prop_std']:.3f}, std/mean={s21['prop_cv']:.3f}")
        print(s); lines.append(s)
        ratio = prop_stats["prop_mean"] / max(s21["prop_mean"], 1e-12)
        if abs(ratio - 1.0) <= PROP_AGREE_TOL:
            verdict_cross = (
                f"AGREE — proportions match within {PROP_AGREE_TOL*100:.0f}%, "
                f"so the AudioSet/VGGSounder count gap is span-length, not rate")
        elif abs(ratio - 1.0) <= 2 * PROP_AGREE_TOL:
            verdict_cross = (
                f"ambiguous — {abs(ratio-1)*100:.0f}% off (tol {PROP_AGREE_TOL*100:.0f}%)")
        else:
            verdict_cross = (
                f"DISAGREE — {abs(ratio-1)*100:.0f}% off; "
                f"per-token sink rate differs across datasets/modality contexts")
        s = (f"    proportion ratio AudioSet/VGGSounder = {ratio:.2f}x  "
             f"→ {verdict_cross}")
        print(s); lines.append(s)
    else:
        s = (f"  Stage 2.1 npz not found at {args.stage21_npz} "
             f"— skipping cross-stage proportion comparison.")
        print(s); lines.append(s)

    # Saturation diagnostic — proportion is content-independent (rate) if
    # std/mean is low; otherwise the rate itself depends on the clip.
    sat = prop_stats["prop_cv"] < PROPORTION_CV_SAT_THR
    sat_msg = (
        "CONFIRMED — content-independent rate (saturation framing holds)"
        if sat else
        "NOT confirmed — proportion varies clip-to-clip → content-dependent rate "
        "(saturation framing weakens)")
    s = (f"  saturation diagnostic: sink_proportion std/mean = "
         f"{prop_stats['prop_cv']:.3f}  vs threshold "
         f"{PROPORTION_CV_SAT_THR:.2f}  →  {sat_msg}")
    print(s); lines.append(s)

    # Classification.
    positional = (pos0_frac >= POS0_POSITIONAL_THR
                  and dens["low"] > dens["mid"] * 3
                  and dens_no0["low"] < dens["low"] / 2)
    content_cond = (ks_median >= KS_HIGH_THR and iqr_median <= IQR_SMALL_THR)
    distributed = (ks_median < 0.20 and dens["mid"] > 0.10)

    if positional:
        verdict = (f"\n  → POSITIONAL: ≥{POS0_POSITIONAL_THR*100:.0f}% of clips "
                   f"have pos-0 as sink ({pos0_frac*100:.1f}%); low-position "
                   f"density {dens['low']:.1f}% >> middle {dens['mid']:.1f}%; "
                   f"and excluding pos 0 flattens the distribution. "
                   f"Audio late-spread sinks are mostly the BOS-of-audio token.")
    elif content_cond:
        verdict = (f"\n  → CONTENT-CONDITIONAL: per-clip KS median "
                   f"{ks_median:.3f} ≥ {KS_HIGH_THR} (clip-level clustering) "
                   f"and IQR median {iqr_median:.3f} ≤ {IQR_SMALL_THR} "
                   f"(concentration within clip), but positions vary across "
                   f"clips. Sinks track acoustic events.")
    elif distributed:
        verdict = (f"\n  → DISTRIBUTED: per-clip KS median {ks_median:.3f} is "
                   f"low and middle-position density {dens['mid']:.1f}% is "
                   f"substantial. No clear temporal structure.")
    else:
        verdict = (f"\n  → MIXED: pos-0 fraction {pos0_frac*100:.1f}% / "
                   f"KS median {ks_median:.3f} / IQR median {iqr_median:.3f} — "
                   f"doesn't cleanly fit positional / content-conditional / "
                   f"distributed. See per-clip CSV for finer breakdown.")
    print(verdict); lines.append(verdict)

    with open(out_dir / "stage2_4_decision.txt", "w") as f:
        f.write(f"Stage 2.4 — audio temporal sink positions  "
                f"(layer L{PRIMARY_LAYER}, n_clips={n_clips_used}, "
                f"D_sink={D_SINK}, τ={TAU_SINK})\n\n")
        for ln in lines:
            f.write(ln.lstrip() + "\n")
    print(f"\nwrote {out_dir / 'stage2_4_decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--audio_dir",
                   default=str(_REPO / "data/AudioSet/audios"))
    p.add_argument("--n_clips", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[PRIMARY_LAYER, COMPARE_LAYER],
                   help="Decoder layers to analyze. Primary = "
                        f"L{PRIMARY_LAYER}, comparison default = L{COMPARE_LAYER}.")
    p.add_argument("--device_map", default="balanced_low_0",
                   help="balanced_low_0 only — auto crashes this model.")
    p.add_argument("--stage21_npz",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_1_layer_sinks/per_clip_counts.npz"),
                   help="Stage 2.1 cached per-clip arrays. Used to compute the "
                        f"VGGSounder audio sink PROPORTION at L{PRIMARY_LAYER} "
                        "for the cross-stage sanity check. Skipped if missing.")
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_4_temporal_sinks"))
    args = p.parse_args()
    main(args)
