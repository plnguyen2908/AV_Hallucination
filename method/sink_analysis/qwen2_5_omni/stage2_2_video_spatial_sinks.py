"""
stage2_2_video_spatial_sinks.py

Stage 2.2 — Spatial distribution of video LLM-emerged sinks in Qwen2.5-Omni.

Matched-modality counterpart to Stage 2.4 (audio temporal). Stage 2.4 found
audio sinks at L21 are temporally UNIFORM within each clip (median per-clip
KS = 0.062). This stage asks the analogous spatial question for video:
where in the (frame × row × col) grid do sinks sit, and are they
concentrated or uniform across the spatial plane?

Inputs:
  - Same 300 VGGSounder clips as Stage 2.1 (sorted glob + seed=42 + modal_type="av").
    Override with --video_dir / --modal_type for ActivityNet (modal_type="v").
  - D_sink = {458, 2570}, τ=20, RMSNorm pure (no weight) eps from thinker config.
  - Primary layer L2 (Stage 1.2 video propagation peak).
    Secondary layer L21 (Stage 1.2 audio peak — cross-modality comparison).

Vision tokenization (Qwen2.5-Omni, from `vision_config`):
  patch_size = 14, spatial_merge_size = 2, temporal_patch_size = 2.
  Processor returns `video_grid_thw` per clip as (T, H, W) where T is ALREADY
  the post-temporal-merge frame-group count, while H and W are pre-spatial-
  merge 14×14 patch counts. So the LLM-side video token grid is
      (T_eff, H_eff, W_eff) = (T, H // spatial_merge_size, W // spatial_merge_size)
  with total token count T_eff × H_eff × W_eff. STEP 0 validates this
  identity per-clip; aborts if it doesn't hold.

Procedure (per the spec):
  STEP 0 — Sanity. Print one example clip's (T_eff, H_eff, W_eff), the
            first 10 video tokens' inferred (frame, row, col), and the
            video span length distribution across all 300 clips.
  STEP 1 — Per-clip forward + hook L2 and L21; record per-token sink mask
            on D_sink; compute (frame, row, col) for each sink position.
  STEP 2 — Pooled 2D spatial heatmap (collapse frames + clips). Normalize
            row/col to [0,1] and bin into a fixed POOL_GRID×POOL_GRID grid
            so clips with different H_eff/W_eff can be aggregated. Compute
            Shannon entropy, entropy ratio vs uniform, top-5 cells, edge
            vs center concentration.
  STEP 3 — Per-clip spatial concentration metrics (KS-row, KS-col, IQR-row,
            IQR-col, edge fraction).
  STEP 4 — Per-clip sink proportion at L2 and L21.

Output dir: results/qwen2_5_omni/sink_analysis/stage2_2_spatial/
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
PROMPT_AV = "Describe what you see and hear in detail."
PROMPT_V = "Describe what you see in detail."

PRIMARY_LAYER = 2          # Stage 1.2 video propagation peak
COMPARE_LAYER = 21         # Stage 1.2 audio peak; cross-modality comparison

POOL_GRID = 16             # POOL_GRID × POOL_GRID heatmap (clip-grid-agnostic)
EDGE_FRAC = 0.20           # outer 20% in either axis → "edge"
ENTROPY_CONC_THR = 0.70    # ratio < this → concentrated
ENTROPY_UNIF_THR = 0.95    # ratio > this → uniform
TOP5_CONC_THR = 0.20       # top-5 cell mass > this → concentrated

VIDEO_TOKEN_ID = 151656    # qwen2_5_omni video_token_index


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


def _vision_merge_sizes(model):
    """Return (temporal_patch_size, spatial_merge_size)."""
    vc = getattr(model.thinker.config, "vision_config", None)
    if vc is None:
        vc = getattr(model.thinker.config, "visual_config", None)
    t = int(getattr(vc, "temporal_patch_size", 2))
    s = int(getattr(vc, "spatial_merge_size", 2))
    return t, s


def _video_positions(input_ids: torch.Tensor) -> np.ndarray:
    ids = input_ids[0].cpu().numpy()
    return np.where(ids == VIDEO_TOKEN_ID)[0].astype(np.int64)


def _grid_thw(inputs) -> tuple:
    """Extract (T, H, W) in 14×14 patch units (pre-merge) for the (one) video
    in this clip's batch. Returns None if no video_grid_thw present."""
    if "video_grid_thw" not in inputs:
        return None
    thw = inputs["video_grid_thw"]
    if hasattr(thw, "cpu"):
        thw = thw.cpu().numpy()
    thw = np.asarray(thw)
    if thw.ndim == 2:
        thw = thw[0]
    return int(thw[0]), int(thw[1]), int(thw[2])


def _effective_grid(thw, t_ps, s_m):
    """For Qwen2.5-Omni, the processor's video_grid_thw returns T already
    post-temporal-merge; H and W are pre-spatial-merge 14×14 patch counts.
    Confirmed empirically: n_video_tokens == T * (H//s_m) * (W//s_m).
    The t_ps argument is retained for documentation / cross-model safety —
    it would be applied if a future config reports T pre-merge."""
    T, H, W = thw
    return T, H // s_m, W // s_m


def _coords_from_idx(idx_in_span: np.ndarray, T_eff, H_eff, W_eff):
    """Map flat in-span token index → (frame, row, col) given the effective
    grid. Row-major frame-major layout: idx = f * H * W + r * W + c."""
    HW = H_eff * W_eff
    f = idx_in_span // HW
    r = (idx_in_span % HW) // W_eff
    c = (idx_in_span % HW) % W_eff
    return f.astype(np.int32), r.astype(np.int32), c.astype(np.int32)


# --------------------------------------------------------------------------
# Per-clip processing
# --------------------------------------------------------------------------

def process_clip(model, processor, clip_path, modal_type, thinker_cfg,
                 layers, eps, d_sink_t, t_ps, s_m, layer_indices):
    prompt = PROMPT_AV if modal_type == "av" else PROMPT_V
    conv = build_conversation(str(clip_path), prompt, modal_type)
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, modal_type, model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}:{e}"

    video_pos = _video_positions(inputs["input_ids"])
    if len(video_pos) == 0:
        return None, "no_video"
    n_video = int(len(video_pos))

    thw = _grid_thw(inputs)
    if thw is None:
        return None, "no_video_grid_thw"
    T_eff, H_eff, W_eff = _effective_grid(thw, t_ps, s_m)
    if T_eff * H_eff * W_eff != n_video:
        return None, (f"grid_mismatch: thw={thw} → eff=({T_eff},{H_eff},{W_eff}) "
                      f"prod={T_eff*H_eff*W_eff} ≠ n_video={n_video}")

    v_pos_t = torch.from_numpy(video_pos)
    sink_masks: dict = {L: None for L in layer_indices}

    def make_hook(L_idx):
        def _h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            x = hs[0].float()
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            normed_abs = (x / rms).abs()
            sink_act = normed_abs[:, d_sink_t].amax(dim=-1)
            sink_mask = (sink_act >= TAU_SINK)
            vp = v_pos_t.to(sink_mask.device)
            sink_masks[L_idx] = sink_mask[vp].cpu().numpy()
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
        return None, f"fwd:{type(e).__name__}:{e}"
    for h in handles: h.remove()
    torch.cuda.empty_cache()

    if any(m is None for m in sink_masks.values()):
        return None, "no_hook"

    per_layer = {}
    for L, mask in sink_masks.items():
        idx_in_span = np.where(mask)[0]
        f, r, c = _coords_from_idx(idx_in_span, T_eff, H_eff, W_eff)
        per_layer[L] = dict(
            n_sink=int(len(idx_in_span)),
            sink_idx=idx_in_span.astype(np.int32),
            frames=f, rows=r, cols=c,
        )
    return dict(
        thw=thw, T_eff=T_eff, H_eff=H_eff, W_eff=W_eff,
        n_video=n_video, per_layer=per_layer,
    ), None


# --------------------------------------------------------------------------
# Per-clip spatial metrics
# --------------------------------------------------------------------------

def per_clip_spatial(per_layer_entry, H_eff, W_eff):
    """Returns dict of per-clip spatial metrics for one layer's sinks."""
    n = per_layer_entry["n_sink"]
    if n < 2:
        return dict(n_sink=n,
                    ks_row=float("nan"), ks_col=float("nan"),
                    ks_max=float("nan"),
                    iqr_row=float("nan"), iqr_col=float("nan"),
                    edge_frac=float("nan"))
    r = per_layer_entry["rows"].astype(np.float64)
    c = per_layer_entry["cols"].astype(np.float64)
    r_n = (r + 0.5) / max(H_eff, 1)
    c_n = (c + 0.5) / max(W_eff, 1)
    ks_r = float(scistats.kstest(r_n, "uniform").statistic)
    ks_c = float(scistats.kstest(c_n, "uniform").statistic)
    iqr_r = float(np.percentile(r_n, 75) - np.percentile(r_n, 25))
    iqr_c = float(np.percentile(c_n, 75) - np.percentile(c_n, 25))
    edge_mask = ((r_n < EDGE_FRAC) | (r_n > 1 - EDGE_FRAC)
                 | (c_n < EDGE_FRAC) | (c_n > 1 - EDGE_FRAC))
    edge_frac = float(edge_mask.mean())
    return dict(n_sink=n, ks_row=ks_r, ks_col=ks_c, ks_max=max(ks_r, ks_c),
                iqr_row=iqr_r, iqr_col=iqr_c, edge_frac=edge_frac)


def accumulate_heatmap(heatmap: np.ndarray, per_layer_entry, H_eff, W_eff):
    """Add this clip's L-layer sinks (all frames pooled) into the fixed
    POOL_GRID×POOL_GRID heatmap by normalized-coordinate binning."""
    n = per_layer_entry["n_sink"]
    if n == 0:
        return
    r = per_layer_entry["rows"].astype(np.float64)
    c = per_layer_entry["cols"].astype(np.float64)
    rn = (r + 0.5) / max(H_eff, 1)
    cn = (c + 0.5) / max(W_eff, 1)
    rb = np.clip((rn * POOL_GRID).astype(np.int64), 0, POOL_GRID - 1)
    cb = np.clip((cn * POOL_GRID).astype(np.int64), 0, POOL_GRID - 1)
    np.add.at(heatmap, (rb, cb), 1)


def shannon_entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_heatmap(heatmap, layer, n_clips, total_sinks, out_path):
    p = heatmap / max(heatmap.sum(), 1)
    H_uniform = np.log2(heatmap.size)
    H_obs = shannon_entropy_bits(p)
    ratio = H_obs / max(H_uniform, 1e-12)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(p, origin="upper", cmap="magma", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="P(sink at this cell)")
    ax.set_xticks(np.arange(0, POOL_GRID, max(1, POOL_GRID // 8)))
    ax.set_yticks(np.arange(0, POOL_GRID, max(1, POOL_GRID // 8)))
    ax.set_xlabel(f"col-bin (1/{POOL_GRID})", fontsize=10)
    ax.set_ylabel(f"row-bin (1/{POOL_GRID})", fontsize=10)
    ax.set_title(f"Stage 2.2 — pooled video-sink heatmap @ L{layer}\n"
                 f"n_clips={n_clips}, sinks={total_sinks}, "
                 f"entropy={H_obs:.3f} / uniform={H_uniform:.3f} = "
                 f"ratio {ratio:.3f}",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return dict(entropy_obs=H_obs, entropy_uniform=H_uniform,
                entropy_ratio=ratio)


def plot_per_clip_metrics(stats_df, layer, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # KS_max
    ks = stats_df["ks_max"].dropna().values
    axes[0].hist(ks, bins=40, range=(0, 1), color="#2ca02c",
                 edgecolor="black", linewidth=0.3, alpha=0.85)
    axes[0].axvline(0.30, ls="--", color="gray", lw=1,
                    label="0.30 (cluster threshold)")
    axes[0].set_xlabel("max(KS_row, KS_col) vs uniform", fontsize=11)
    axes[0].set_ylabel("# clips", fontsize=11)
    axes[0].set_title(f"L{layer} per-clip spatial KS  "
                      f"(median={np.median(ks):.3f})", fontsize=11)
    axes[0].grid(True, ls=":", alpha=0.4, axis="y")
    axes[0].legend(fontsize=9)
    # mean IQR
    iqr_mean = (stats_df["iqr_row"].dropna().values
                + stats_df["iqr_col"].dropna().values) / 2.0
    axes[1].hist(iqr_mean, bins=40, range=(0, 1), color="#9467bd",
                 edgecolor="black", linewidth=0.3, alpha=0.85)
    axes[1].axvline(0.20, ls="--", color="gray", lw=1,
                    label="0.20 (concentration threshold)")
    axes[1].set_xlabel("mean(IQR_row, IQR_col)", fontsize=11)
    axes[1].set_ylabel("# clips", fontsize=11)
    axes[1].set_title(f"L{layer} per-clip mean IQR  "
                      f"(median={np.median(iqr_mean):.3f})", fontsize=11)
    axes[1].grid(True, ls=":", alpha=0.4, axis="y")
    axes[1].legend(fontsize=9)
    # Edge fraction
    ef = stats_df["edge_frac"].dropna().values
    axes[2].hist(ef, bins=40, range=(0, 1), color="#ff7f0e",
                 edgecolor="black", linewidth=0.3, alpha=0.85)
    edge_uniform = 1 - (1 - 2 * EDGE_FRAC) ** 2  # fraction of unit square in edge ring
    axes[2].axvline(edge_uniform, ls="--", color="gray", lw=1,
                    label=f"uniform = {edge_uniform:.2f}")
    axes[2].set_xlabel(f"edge fraction (outer {int(EDGE_FRAC*100)}%)",
                       fontsize=11)
    axes[2].set_ylabel("# clips", fontsize=11)
    axes[2].set_title(f"L{layer} per-clip edge concentration  "
                      f"(median={np.median(ef):.3f})", fontsize=11)
    axes[2].grid(True, ls=":", alpha=0.4, axis="y")
    axes[2].legend(fontsize=9)
    fig.suptitle(f"Stage 2.2 — per-clip spatial concentration @ L{layer}  "
                 f"(n_clips={len(stats_df)})", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_proportion_hist(stats_l2, stats_l21, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, df, L, color in (
        (axes[0], stats_l2, PRIMARY_LAYER, "#d62728"),
        (axes[1], stats_l21, COMPARE_LAYER, "#1f77b4")):
        p = df["sink_proportion"].values
        ax.hist(p, bins=40, color=color, edgecolor="black", linewidth=0.3,
                alpha=0.85)
        m, s = float(p.mean()), float(p.std())
        ax.axvline(m, ls="-", color="black", lw=1.2,
                   label=f"mean={m:.3f}, std={s:.3f}, "
                         f"std/mean={s/max(m,1e-9):.3f}")
        ax.set_xlabel("sink count / video span length", fontsize=11)
        ax.set_ylabel("# clips", fontsize=11)
        ax.set_title(f"L{L} per-clip sink PROPORTION", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, ls=":", alpha=0.4, axis="y")
    fig.suptitle("Stage 2.2 — per-clip video sink proportion  "
                 f"(n_clips={len(stats_l2)})", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def verdict_for_layer(L, heat_stats, per_clip_df, top5_mass):
    er = heat_stats["entropy_ratio"]
    median_ks = float(np.nanmedian(per_clip_df["ks_max"].values))
    median_iqr = float(np.nanmedian(
        (per_clip_df["iqr_row"].values + per_clip_df["iqr_col"].values) / 2.0))
    median_edge = float(np.nanmedian(per_clip_df["edge_frac"].values))
    if er < ENTROPY_CONC_THR and top5_mass > TOP5_CONC_THR:
        v = (f"L{L}: CONCENTRATED — entropy ratio = {er:.3f} < "
             f"{ENTROPY_CONC_THR}, top-5 cells = {top5_mass*100:.1f}% > "
             f"{TOP5_CONC_THR*100:.0f}%. Spatially structured (opposite "
             f"of audio's temporal uniformity).")
    elif er > ENTROPY_UNIF_THR:
        v = (f"L{L}: UNIFORM — entropy ratio = {er:.3f} > {ENTROPY_UNIF_THR}, "
             f"top-5 cells = {top5_mass*100:.1f}%. Spatially uniform like audio.")
    else:
        v = (f"L{L}: PARTIAL — entropy ratio = {er:.3f}, "
             f"top-5 cells = {top5_mass*100:.1f}%. Intermediate concentration.")
    v += (f"  per-clip medians: KS={median_ks:.3f}, IQR={median_iqr:.3f}, "
          f"edge_frac={median_edge:.3f}.")
    return v, dict(entropy_ratio=er, top5_mass=top5_mass,
                   median_ks=median_ks, median_iqr=median_iqr,
                   median_edge=median_edge)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni …")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        print(f"  [note] 1 GPU visible — overriding device_map → 'auto'")
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    eps = _thinker_rms_eps(model)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    t_ps, s_m = _vision_merge_sizes(model)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    layer_indices = sorted(set(args.layers))
    for L in layer_indices:
        if L >= n_layers:
            raise SystemExit(f"layer {L} ≥ n_layers {n_layers}")
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ={TAU_SINK}, "
          f"vision: t_ps={t_ps}, s_m={s_m}, layers={layer_indices}, "
          f"POOL_GRID={POOL_GRID}")

    clip_dir = Path(args.video_dir)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No .mp4 in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips))[:args.n_clips]
    clips = [clips[i] for i in idx]
    print(f"\n{clip_dir.name} pass: {len(clips)} clips, modal_type={args.modal_type}\n")

    # -------------------------- STEP 0 sanity --------------------------
    print("=" * 80)
    print("STEP 0 — sanity check on one example clip")
    print("=" * 80)
    sanity_done = False
    for clip in clips:
        result, err = process_clip(model, processor, clip, args.modal_type,
                                   thinker_cfg, layers, eps, d_sink_t,
                                   t_ps, s_m, layer_indices)
        if result is None:
            print(f"  [skip] {clip.name}: {err}"); continue
        T_e, H_e, W_e = result["T_eff"], result["H_eff"], result["W_eff"]
        print(f"  clip: {clip.name}")
        print(f"  video_grid_thw (pre-merge 14×14 patches): {result['thw']}")
        print(f"  effective grid (after t_ps={t_ps}, s_m={s_m}): "
              f"T_eff={T_e}, H_eff={H_e}, W_eff={W_e}")
        print(f"  n_video_tokens = {result['n_video']}  "
              f"(expected T_eff·H_eff·W_eff = {T_e*H_e*W_e})  ✓")
        print(f"  first-10 token coords (frame, row, col):")
        for i in range(min(10, result['n_video'])):
            f = i // (H_e * W_e); rem = i % (H_e * W_e)
            print(f"    tok {i:4d}  ({f:3d}, {rem // W_e:3d}, {rem % W_e:3d})")
        for L in layer_indices:
            n_s = result["per_layer"][L]["n_sink"]
            print(f"  layer {L}: {n_s} sinks "
                  f"({n_s / max(result['n_video'], 1) * 100:.1f}% of video tokens)")
        sanity_done = True
        break
    if not sanity_done:
        raise SystemExit("STEP 0 failed: no clip could be processed.")

    # video span length distribution from the saved n_video values (collected
    # below); print here as a placeholder using the sanity clip.
    print("\n  (full video-span distribution printed after STEP 1.)")

    # -------------------------- STEP 1: process 300 clips --------------------------
    print("\n" + "=" * 80)
    print(f"STEP 1 — processing {len(clips)} clips, hooks at layers {layer_indices}")
    print("=" * 80)
    per_layer_heatmaps = {L: np.zeros((POOL_GRID, POOL_GRID),
                                       dtype=np.float64)
                          for L in layer_indices}
    per_clip_rows = {L: [] for L in layer_indices}
    span_lengths = []
    failures: dict = {}
    raw_dump: dict = {L: [] for L in layer_indices}     # for the npz

    for clip in tqdm(clips, desc="clips"):
        result, err = process_clip(model, processor, clip, args.modal_type,
                                   thinker_cfg, layers, eps, d_sink_t,
                                   t_ps, s_m, layer_indices)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            continue
        T_e, H_e, W_e = result["T_eff"], result["H_eff"], result["W_eff"]
        n_v = result["n_video"]
        span_lengths.append(n_v)
        for L in layer_indices:
            entry = result["per_layer"][L]
            accumulate_heatmap(per_layer_heatmaps[L], entry, H_e, W_e)
            sp = entry["n_sink"] / max(n_v, 1)
            metrics = per_clip_spatial(entry, H_e, W_e)
            metrics.update(dict(
                clip=clip.name, layer=L,
                n_video=n_v, T_eff=T_e, H_eff=H_e, W_eff=W_e,
                sink_proportion=sp))
            per_clip_rows[L].append(metrics)
            raw_dump[L].append(dict(
                clip=clip.name, n_video=n_v, n_sink=entry["n_sink"],
                T_eff=T_e, H_eff=H_e, W_eff=W_e,
                rows=entry["rows"], cols=entry["cols"],
                frames=entry["frames"]))
    if failures:
        print(f"  failures: {failures}")
    n_used = len(per_clip_rows[layer_indices[0]])
    print(f"\n  succeeded on {n_used}/{len(clips)} clips.")
    if n_used == 0:
        raise SystemExit("No clips processed successfully.")

    sl = np.array(span_lengths, dtype=np.int64)
    print(f"  video span length: mean={sl.mean():.1f}, std={sl.std():.1f}, "
          f"min={sl.min()}, max={sl.max()}")

    # -------------------------- STEP 2: pooled heatmap aggregates --------------------------
    print("\n" + "=" * 80)
    print(f"STEP 2 — pooled spatial heatmap ({POOL_GRID}×{POOL_GRID})")
    print("=" * 80)
    heat_results = {}
    for L in layer_indices:
        heat = per_layer_heatmaps[L]
        total = int(heat.sum())
        # Save raw heatmap.
        np.savez_compressed(out_dir / f"pooled_heatmap_L{L}.npz",
                            heatmap=heat,
                            pool_grid=POOL_GRID,
                            n_clips=n_used,
                            total_sinks=total)
        # Top-5 cells
        flat = heat.ravel().copy()
        order = np.argsort(flat)[::-1]
        top5_mass = float(flat[order[:5]].sum() / max(flat.sum(), 1))
        top5_coords = [(int(o // POOL_GRID), int(o % POOL_GRID),
                        int(flat[o])) for o in order[:5]]
        # Edge vs center
        p = heat / max(heat.sum(), 1)
        rr, cc = np.meshgrid(np.arange(POOL_GRID), np.arange(POOL_GRID),
                             indexing="ij")
        rn = (rr + 0.5) / POOL_GRID; cn = (cc + 0.5) / POOL_GRID
        edge_mask = ((rn < EDGE_FRAC) | (rn > 1 - EDGE_FRAC)
                     | (cn < EDGE_FRAC) | (cn > 1 - EDGE_FRAC))
        edge_mass = float(p[edge_mask].sum())
        center_mass = float(p[~edge_mask].sum())
        edge_uniform = float(edge_mask.mean())   # what uniform would give
        heat_stats = plot_heatmap(heat, L, n_used, total,
                                  out_dir / f"spatial_sink_heatmap_L{L}.png")
        heat_stats.update(
            top5_mass=top5_mass, top5_coords=top5_coords,
            edge_mass=edge_mass, center_mass=center_mass,
            edge_uniform_baseline=edge_uniform,
            n_clips=n_used, total_sinks=total,
        )
        heat_results[L] = heat_stats
        print(f"\n  L{L}: total_sinks={total}, "
              f"entropy={heat_stats['entropy_obs']:.3f} "
              f"(uniform={heat_stats['entropy_uniform']:.3f}, "
              f"ratio={heat_stats['entropy_ratio']:.3f})")
        print(f"       top-5 mass = {top5_mass*100:.1f}%   "
              f"top-5 cells (row,col,count) = {top5_coords}")
        print(f"       edge_mass = {edge_mass:.3f}  vs uniform baseline "
              f"{edge_uniform:.3f}   center_mass = {center_mass:.3f}")

    # -------------------------- STEP 3: per-clip metrics --------------------------
    print("\n" + "=" * 80)
    print("STEP 3 — per-clip spatial concentration")
    print("=" * 80)
    per_clip_dfs = {L: pd.DataFrame(per_clip_rows[L]) for L in layer_indices}
    for L in layer_indices:
        df = per_clip_dfs[L]
        plot_per_clip_metrics(df, L,
                              out_dir / f"per_clip_spatial_concentration_L{L}.png")

    # -------------------------- STEP 4: per-clip proportion --------------------------
    print("\n" + "=" * 80)
    print("STEP 4 — per-clip sink proportion")
    print("=" * 80)
    if PRIMARY_LAYER in per_clip_dfs and COMPARE_LAYER in per_clip_dfs:
        plot_proportion_hist(per_clip_dfs[PRIMARY_LAYER],
                             per_clip_dfs[COMPARE_LAYER],
                             out_dir / "per_clip_sink_proportion_video.png")

    # -------------------------- combined CSV --------------------------
    combined = pd.concat([per_clip_dfs[L] for L in layer_indices],
                         ignore_index=True)
    combined.to_csv(out_dir / "per_clip_spatial_stats.csv", index=False)
    print(f"\nwrote {out_dir / 'per_clip_spatial_stats.csv'}  "
          f"({len(combined)} rows)")

    # Save raw sink coordinates for replot.
    np.savez_compressed(
        out_dir / "per_clip_sink_coords.npz",
        **{f"L{L}_clips": np.array([d["clip"] for d in raw_dump[L]], dtype=object)
           for L in layer_indices},
        **{f"L{L}_n_video": np.array([d["n_video"] for d in raw_dump[L]],
                                      dtype=np.int64)
           for L in layer_indices},
        **{f"L{L}_n_sink": np.array([d["n_sink"] for d in raw_dump[L]],
                                     dtype=np.int64)
           for L in layer_indices},
        **{f"L{L}_T_eff": np.array([d["T_eff"] for d in raw_dump[L]],
                                    dtype=np.int64)
           for L in layer_indices},
        **{f"L{L}_H_eff": np.array([d["H_eff"] for d in raw_dump[L]],
                                    dtype=np.int64)
           for L in layer_indices},
        **{f"L{L}_W_eff": np.array([d["W_eff"] for d in raw_dump[L]],
                                    dtype=np.int64)
           for L in layer_indices},
        **{f"L{L}_offsets": np.cumsum([0] + [d["n_sink"] for d in raw_dump[L]])
           for L in layer_indices},
        **{f"L{L}_rows_flat": (np.concatenate([d["rows"] for d in raw_dump[L]])
                                if raw_dump[L] else np.array([], dtype=np.int32))
           for L in layer_indices},
        **{f"L{L}_cols_flat": (np.concatenate([d["cols"] for d in raw_dump[L]])
                                if raw_dump[L] else np.array([], dtype=np.int32))
           for L in layer_indices},
        **{f"L{L}_frames_flat": (np.concatenate([d["frames"] for d in raw_dump[L]])
                                  if raw_dump[L] else np.array([], dtype=np.int32))
           for L in layer_indices},
    )
    print(f"wrote {out_dir / 'per_clip_sink_coords.npz'}")

    # -------------------------- Verdict + decision.txt --------------------------
    print("\n" + "=" * 86)
    print(f"STAGE 2.2 VERDICT  (n_clips={n_used}, dataset={clip_dir.name})")
    print("=" * 86)

    decision_lines = []
    decision_lines.append(
        f"Stage 2.2 — video spatial sink distribution  "
        f"(n_clips={n_used}, D_sink={D_SINK}, τ={TAU_SINK})")
    decision_lines.append(
        f"Video span: mean={sl.mean():.1f}, std={sl.std():.1f}, "
        f"min={sl.min()}, max={sl.max()}")
    decision_lines.append("")
    for L in layer_indices:
        v, summary = verdict_for_layer(L, heat_results[L],
                                        per_clip_dfs[L],
                                        heat_results[L]["top5_mass"])
        print("  " + v); decision_lines.append(v)
    # Stage 2.4 audio comparison
    decision_lines.append("")
    decision_lines.append("Comparison vs Stage 2.4 (audio L21 within-clip):")
    decision_lines.append(
        "  Stage 2.4 audio:  median KS = 0.062,  median IQR = 0.494,  "
        "marginal essentially uniform.")
    for L in layer_indices:
        df = per_clip_dfs[L]
        median_ks = float(np.nanmedian(df["ks_max"].values))
        median_iqr = float(np.nanmedian(
            (df["iqr_row"].values + df["iqr_col"].values) / 2.0))
        median_edge = float(np.nanmedian(df["edge_frac"].values))
        decision_lines.append(
            f"  L{L} video:        median max-KS = {median_ks:.3f},  "
            f"median mean-IQR = {median_iqr:.3f},  "
            f"median edge_frac = {median_edge:.3f}")

    with open(out_dir / "decision.txt", "w") as f:
        for ln in decision_lines:
            f.write(ln + "\n")
    print(f"\nwrote {out_dir / 'decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",
                   default=str(_REPO / "data/VGGSounder/videos"),
                   help="Default: VGGSounder (matches Stage 2.1). For "
                        "ActivityNet, override + --modal_type v.")
    p.add_argument("--modal_type", default="av", choices=["av", "v"])
    p.add_argument("--n_clips", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[PRIMARY_LAYER, COMPARE_LAYER],
                   help=f"Decoder layers. Primary = L{PRIMARY_LAYER}, "
                        f"comparison = L{COMPARE_LAYER}.")
    p.add_argument("--device_map", default="balanced_low_0",
                   help="balanced_low_0 only — auto crashes this model.")
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_2_spatial"))
    args = p.parse_args()
    main(args)
