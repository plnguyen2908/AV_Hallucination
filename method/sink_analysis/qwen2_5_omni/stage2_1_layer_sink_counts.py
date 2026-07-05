"""
stage2_1_layer_sink_counts.py

Stage 2.1 — Layer-wise LLM sink counts (ASD Figure 7 reproduction for
Qwen2.5-Omni). For each decoder layer, count tokens crossing the τ=20 LLM-sink
criterion on D_sink = {458, 2570} (UPDATED — 3197 dropped after Stage 1.3
distinctiveness analysis showed it's a broadly-active video-content dim, not a
sink). Separately for audio and video tokens, separately per clip.

Per-clip processing uses per-layer forward HOOKS (not output_hidden_states),
so we compute the per-layer mask on the fly and never retain all 28 hidden
states on GPU simultaneously — keeps memory low for 300 clips.

Inputs:
- 300 VGGSounder clips (audio + video both present).
- D_sink = {458, 2570}; RMSNorm pure (no weight), eps from thinker config.

Outputs (--output_dir):
    layer_sink_counts.csv             per (layer): audio/video mean ± std
    figure_2_1_layer_sink_counts.png  2 panels (audio, video) with prop baseline
    per_clip_counts.npz               cached per-clip per-layer arrays (replot)
    stage2_1_decision.txt             verdict text + sanity comparison to S1.3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import (  # noqa: E402
    build_conversation, find_modality_spans, load_omni, prepare_inputs,
    thinker_layers,
)


D_SINK = [458, 2570]                # Stage 1.3-corrected sink set (drop 3197)
TAU_SINK = 20.0                     # token-sink criterion on D_sink (max ≥ τ)
TAU_HIGH_ENC = 100.0                # encoder-propagated criterion
PROMPT_AV = "Describe what you see and hear in detail."

# Stage 1.2 reference peaks for vertical markers.
LAYER_VIDEO_PEAK = 2
LAYER_AUDIO_PEAK = 21
# Stage 1.3 reference numbers (D_sink = {458, 2570, 3197}, 100 VGGSounder clips,
# per-clip totals — listed here so the sanity check can compare scale).
S1_3_OLD_3DIM = {
    "video": {2: 4805, 14: 64324, 21: 96922},
    "audio": {2: 446,  14: 7397,  21: 14366},
}
S1_3_NEW_2DIM = {                    # after dropping 3197
    "video": {2: 3100, 14: 37350, 21: 55106},
    "audio": {2: 446,  14: 6701,  21: 10315},
}
S1_3_N_CLIPS = 100                  # reference clip count for the above


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


def _extract_tokens(out) -> torch.Tensor:
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    return x


def _modal_positions(input_ids: torch.Tensor, thinker_cfg) -> tuple:
    """Per-token modality classification by INPUT TOKEN ID (handles AV
    interleaving — see Stage 1.3 logic)."""
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


def align_norms(enc_norms: np.ndarray, n_llm: int):
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


# --------------------------------------------------------------------------
# Per-clip processing — forward with PER-LAYER HOOKS (no hidden_states retention)
# --------------------------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, audio_enc, visual_enc,
                 layers, eps, d_sink_tensor):
    """Per-clip: compute per-layer audio/video sink counts via forward hooks,
    + propagated counts via encoder hooks."""
    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, "av", model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}"

    audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
    if len(audio_pos) == 0 or len(video_pos) == 0:
        return None, "no_modal_tokens"

    n_layers = len(layers)
    audio_count = np.zeros(n_layers, dtype=np.int64)
    video_count = np.zeros(n_layers, dtype=np.int64)
    # Store per-layer sink masks over the modal positions ONLY (small: per clip
    # ~ 28 * (n_a_llm + n_v_llm) ~ 50KB), so we can intersect with the prop
    # mask post-forward (prop mask isn't known until encoder hooks have fired
    # AND we've aligned encoder norms — both happen AFTER the forward).
    a_sink_per_layer = np.zeros((n_layers, len(audio_pos)), dtype=bool)
    v_sink_per_layer = np.zeros((n_layers, len(video_pos)), dtype=bool)
    a_pos_t = torch.from_numpy(audio_pos)
    v_pos_t = torch.from_numpy(video_pos)

    def make_layer_hook(L_idx):
        def _h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out      # (1, seq, H)
            x = hs[0].float()                                    # (seq, H)
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            normed_abs = (x / rms).abs()                         # (seq, H)
            sink_act = normed_abs[:, d_sink_tensor].amax(dim=-1)  # (seq,)
            sink_mask = (sink_act >= TAU_SINK)
            ap = a_pos_t.to(sink_mask.device); vp = v_pos_t.to(sink_mask.device)
            a_sub = sink_mask[ap].cpu().numpy()
            v_sub = sink_mask[vp].cpu().numpy()
            a_sink_per_layer[L_idx] = a_sub
            v_sink_per_layer[L_idx] = v_sub
            audio_count[L_idx] = int(a_sub.sum())
            video_count[L_idx] = int(v_sub.sum())
        return _h

    # Encoder-norm hooks (for propagated counts).
    a_buf, v_buf = [], []
    def make_enc_hook(buf):
        def _h(_m, _i, out):
            tok = _extract_tokens(out)
            buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
        return _h

    handles = []
    handles.append(audio_enc.register_forward_hook(make_enc_hook(a_buf)))
    handles.append(visual_enc.register_forward_hook(make_enc_hook(v_buf)))
    for L_idx in range(n_layers):
        handles.append(layers[L_idx].register_forward_hook(make_layer_hook(L_idx)))

    try:
        with torch.inference_mode():
            model.thinker(**inputs, output_hidden_states=False,
                          use_audio_in_video=use_aiv,
                          return_dict=True, use_cache=False)
    except Exception as e:
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        return None, f"fwd:{type(e).__name__}"
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()

    if not a_buf or not v_buf:
        return None, "no_enc_hooks"

    a_enc = np.concatenate(a_buf)
    v_enc = np.concatenate(v_buf)
    n_a_llm = len(audio_pos); n_v_llm = len(video_pos)
    a_aligned, _ = align_norms(a_enc, n_a_llm)
    v_aligned, _ = align_norms(v_enc, n_v_llm)
    if a_aligned is None or v_aligned is None:
        return None, "align_fail"

    a_prop_mask = a_aligned > TAU_HIGH_ENC          # (n_a_llm,) bool
    v_prop_mask = v_aligned > TAU_HIGH_ENC
    # Per-layer count of P_prop ∩ P_llm@L: how many of the propagated tokens
    # also satisfy the LLM-sink criterion at each layer. Bounded by sum(prop_mask).
    audio_prop_sink = (a_sink_per_layer & a_prop_mask[None, :]).sum(axis=1)
    video_prop_sink = (v_sink_per_layer & v_prop_mask[None, :]).sum(axis=1)
    return {
        "n_a_llm": int(n_a_llm), "n_v_llm": int(n_v_llm),
        "audio_count": audio_count, "video_count": video_count,
        "audio_prop": int(a_prop_mask.sum()),
        "video_prop": int(v_prop_mask.sum()),
        "audio_prop_sink": audio_prop_sink.astype(np.int64),
        "video_prop_sink": video_prop_sink.astype(np.int64),
    }, None


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------

def aggregate_and_report(per_clip, n_layers, out_dir):
    """per_clip: list of dicts from process_clip."""
    n_clips = len(per_clip)
    audio_mat = np.stack([d["audio_count"] for d in per_clip])   # (n_clips, n_layers)
    video_mat = np.stack([d["video_count"] for d in per_clip])
    a_prop_sink_mat = np.stack([d["audio_prop_sink"] for d in per_clip])
    v_prop_sink_mat = np.stack([d["video_prop_sink"] for d in per_clip])
    a_prop = np.array([d["audio_prop"] for d in per_clip])
    v_prop = np.array([d["video_prop"] for d in per_clip])
    n_a_llm = np.array([d["n_a_llm"] for d in per_clip])
    n_v_llm = np.array([d["n_v_llm"] for d in per_clip])

    a_mean = audio_mat.mean(axis=0); a_std = audio_mat.std(axis=0)
    v_mean = video_mat.mean(axis=0); v_std = video_mat.std(axis=0)
    a_ps_mean = a_prop_sink_mat.mean(axis=0); a_ps_std = a_prop_sink_mat.std(axis=0)
    v_ps_mean = v_prop_sink_mat.mean(axis=0); v_ps_std = v_prop_sink_mat.std(axis=0)
    a_prop_mean = float(a_prop.mean()); a_prop_std = float(a_prop.std())
    v_prop_mean = float(v_prop.mean()); v_prop_std = float(v_prop.std())

    layers_idx = np.arange(n_layers)
    df = pd.DataFrame({
        "layer": layers_idx,
        "audio_llm_mean": a_mean, "audio_llm_std": a_std,
        "video_llm_mean": v_mean, "video_llm_std": v_std,
        # P_prop ∩ P_llm@L — # propagated tokens that ALSO satisfy the LLM-sink
        # criterion at this layer (bounded by the per-clip P_prop count).
        "audio_prop_sink_mean": a_ps_mean, "audio_prop_sink_std": a_ps_std,
        "video_prop_sink_mean": v_ps_mean, "video_prop_sink_std": v_ps_std,
    })
    df.to_csv(out_dir / "layer_sink_counts.csv", index=False)

    return {
        "n_clips": n_clips,
        "audio_mat": audio_mat, "video_mat": video_mat,
        "audio_prop_sink_mat": a_prop_sink_mat,
        "video_prop_sink_mat": v_prop_sink_mat,
        "a_mean": a_mean, "a_std": a_std,
        "v_mean": v_mean, "v_std": v_std,
        "a_ps_mean": a_ps_mean, "a_ps_std": a_ps_std,
        "v_ps_mean": v_ps_mean, "v_ps_std": v_ps_std,
        "a_prop_mean": a_prop_mean, "a_prop_std": a_prop_std,
        "v_prop_mean": v_prop_mean, "v_prop_std": v_prop_std,
        "n_a_llm_total": int(n_a_llm.sum()), "n_v_llm_total": int(n_v_llm.sum()),
        "n_a_llm_mean": float(n_a_llm.mean()), "n_v_llm_mean": float(n_v_llm.mean()),
        "layers_idx": layers_idx,
    }


def plot_figure(stats, out_dir):
    """Writes TWO figures (overlaying them at one scale crams the prop∩llm
    line down to invisible):
      figure_2_1_llm_emerged.png  — LLM-emerged sinks per layer (audio+video).
      figure_2_1_prop_overlap.png — P_prop ∩ P_llm per layer + P_prop baseline.
    """
    layers = stats["layers_idx"]

    # --- Figure A: LLM-emerged sinks per layer ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, m, color, mean, std in (
        (axes[0], "audio", "#1f77b4", stats["a_mean"], stats["a_std"]),
        (axes[1], "video", "#d62728", stats["v_mean"], stats["v_std"]),
    ):
        ax.plot(layers, mean, marker="o", ms=4, lw=2.2, color=color,
                label=f"{m} LLM-emerged sinks (τ=20 on {{458, 2570}})")
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2,
                        label="±1 std")
        ax.axvline(LAYER_VIDEO_PEAK, ls=":", color="#d62728", alpha=0.55,
                   label=f"L{LAYER_VIDEO_PEAK} (video peak, S1.2)")
        ax.axvline(LAYER_AUDIO_PEAK, ls=":", color="#1f77b4", alpha=0.55,
                   label=f"L{LAYER_AUDIO_PEAK} (audio peak, S1.2)")
        ax.set_xlabel("LLM decoder layer L", fontsize=11)
        ax.set_ylabel("# sink tokens per clip", fontsize=11)
        ax.set_title(f"{m} — layer-wise LLM-emerged sink count", fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Stage 2.1 — LLM-emerged sinks per layer  "
                 f"(VGGSounder, n_clips={stats['n_clips']}, "
                 f"D_sink={{458, 2570}}, τ=20)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p = out_dir / "figure_2_1_llm_emerged.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")

    # --- Figure B: P_prop ∩ P_llm per layer + P_prop baseline ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    panels = (
        (axes[0], "audio", "#0a3d62",
         stats["a_ps_mean"], stats["a_ps_std"],
         stats["a_prop_mean"], stats["a_prop_std"]),
        (axes[1], "video", "#7a1010",
         stats["v_ps_mean"], stats["v_ps_std"],
         stats["v_prop_mean"], stats["v_prop_std"]),
    )
    for ax, m, c_ps, ps_mean, ps_std, p_mean, p_std in panels:
        ax.plot(layers, ps_mean, marker="s", ms=5, lw=2.2, color=c_ps,
                label="P_prop ∩ P_llm @ L  (propagated AND sink at layer)")
        ax.fill_between(layers, ps_mean - ps_std, ps_mean + ps_std,
                        color=c_ps, alpha=0.2, label="±1 std")
        ax.axhline(p_mean, ls="--", color="gray", lw=1.6,
                   label=f"P_prop count (enc>100, fixed): "
                         f"mean={p_mean:.1f}±{p_std:.1f}")
        ax.axvline(LAYER_VIDEO_PEAK, ls=":", color="#d62728", alpha=0.55,
                   label=f"L{LAYER_VIDEO_PEAK} (video peak, S1.2)")
        ax.axvline(LAYER_AUDIO_PEAK, ls=":", color="#1f77b4", alpha=0.55,
                   label=f"L{LAYER_AUDIO_PEAK} (audio peak, S1.2)")
        # Saturation annotation at the peak.
        if len(ps_mean) > 0:
            peak_L = int(np.argmax(ps_mean))
            pct = ps_mean[peak_L] / max(p_mean, 1e-9) * 100
            ax.annotate(f"peak L{peak_L}\n{ps_mean[peak_L]:.2f}/clip\n"
                        f"({pct:.1f}% of P_prop)",
                        xy=(peak_L, ps_mean[peak_L]),
                        xytext=(peak_L + 0.5, ps_mean[peak_L] * 0.65),
                        fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", lw=0.7))
        ax.set_ylim(bottom=0, top=max(p_mean * 1.7, ps_mean.max() * 1.2))
        ax.set_xlabel("LLM decoder layer L", fontsize=11)
        ax.set_ylabel("# tokens per clip", fontsize=11)
        ax.set_title(f"{m} — propagated tokens acting as LLM-sinks",
                     fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Stage 2.1 — P_prop ∩ P_llm per layer  "
                 "(propagated tokens are absorbed into the sink population at "
                 f"depth; n_clips={stats['n_clips']})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p = out_dir / "figure_2_1_prop_overlap.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def print_sanity_and_verdict(stats, n_clips, out_dir):
    a_mean, v_mean = stats["a_mean"], stats["v_mean"]
    n_layers = len(a_mean)
    a_total = stats["n_a_llm_total"]; v_total = stats["n_v_llm_total"]

    print("\n" + "=" * 86)
    print(f"SANITY CHECKS  (n_clips={n_clips}, D_sink={D_SINK}, τ={TAU_SINK})")
    print("=" * 86)
    print(f"  audio tokens total: {a_total} (mean/clip {stats['n_a_llm_mean']:.1f})")
    print(f"  video tokens total: {v_total} (mean/clip {stats['n_v_llm_mean']:.1f})")
    print(f"  audio propagated (enc>100): mean/clip {stats['a_prop_mean']:.1f}")
    print(f"  video propagated (enc>100): mean/clip {stats['v_prop_mean']:.1f}")
    print(f"\n  Comparison to Stage 1.3 totals at L2/14/21  (per-clip totals, "
          f"so multiply S1.3 by {n_clips/S1_3_N_CLIPS:.2f} to compare):")
    print(f"    {'modality':<8}{'layer':<7}{'S1.3 old (3dim)':<18}"
          f"{'S1.3 new (2dim)':<18}{'this run (2dim)':<18}"
          f"{'S1.3-new scaled':<18}")
    for mod, mean_arr in (("audio", a_mean), ("video", v_mean)):
        for L in (2, 14, 21):
            old = S1_3_OLD_3DIM[mod][L]
            new = S1_3_NEW_2DIM[mod][L]
            here = float(mean_arr[L]) * n_clips        # per-clip × n_clips
            new_scaled = new * (n_clips / S1_3_N_CLIPS)
            print(f"    {mod:<8}{L:<7}{old:<18}{new:<18}"
                  f"{here:<18.0f}{new_scaled:<18.0f}")

    # Verdict.
    print("\n" + "=" * 86)
    print("STAGE 2.1 VERDICT")
    print("=" * 86)
    lines = []
    a_peak_L = int(np.argmax(a_mean)); v_peak_L = int(np.argmax(v_mean))
    # Use a robust early baseline (mean of first 5 layers, floored at 1) for the
    # peak/early ratio — min can be ~0 in early layers and explodes the ratio.
    a_early = max(float(a_mean[:5].mean()), 1.0)
    v_early = max(float(v_mean[:5].mean()), 1.0)
    s = (f"  audio LLM-emerged sinks: peak at L{a_peak_L} "
         f"({a_mean[a_peak_L]:.1f}/clip); L21={a_mean[21]:.1f}/clip; "
         f"peak/mean(L0-4) = {a_mean[a_peak_L]/a_early:.1f}x")
    print(s); lines.append(s)
    s = (f"  video LLM-emerged sinks: peak at L{v_peak_L} "
         f"({v_mean[v_peak_L]:.1f}/clip); L2={v_mean[2]:.1f}/clip; "
         f"peak/mean(L0-4) = {v_mean[v_peak_L]/v_early:.1f}x")
    print(s); lines.append(s)
    s = (f"  propagated audio mean: {stats['a_prop_mean']:.1f}/clip   "
         f"propagated video mean: {stats['v_prop_mean']:.1f}/clip")
    print(s); lines.append(s)
    # P_prop ∩ P_llm trajectory: where do the propagated tokens themselves act
    # as LLM-sinks?  Bounded by the prop count baseline.
    aps, vps = stats["a_ps_mean"], stats["v_ps_mean"]
    a_ps_pk = int(np.argmax(aps)); v_ps_pk = int(np.argmax(vps))
    s = (f"  audio P_prop∩P_llm: peak at L{a_ps_pk} ({aps[a_ps_pk]:.2f}/clip "
         f"of {stats['a_prop_mean']:.1f} propagated total = "
         f"{aps[a_ps_pk]/max(stats['a_prop_mean'],1e-9)*100:.1f}% of prop tokens "
         f"act as sinks here)")
    print(s); lines.append(s)
    s = (f"  video P_prop∩P_llm: peak at L{v_ps_pk} ({vps[v_ps_pk]:.2f}/clip "
         f"of {stats['v_prop_mean']:.1f} propagated total = "
         f"{vps[v_ps_pk]/max(stats['v_prop_mean'],1e-9)*100:.1f}% of prop tokens "
         f"act as sinks here)")
    print(s); lines.append(s)
    # LLM-emerged vs propagated comparison at the peak layers.
    s = (f"  audio LLM-emerged @L21 ({a_mean[21]:.1f}) vs audio propagated "
         f"({stats['a_prop_mean']:.1f}) → "
         f"ratio = {a_mean[21]/max(stats['a_prop_mean'],1e-9):.1f}x")
    print(s); lines.append(s)
    s = (f"  video LLM-emerged @ peak L{v_peak_L} ({v_mean[v_peak_L]:.1f}) vs "
         f"video propagated ({stats['v_prop_mean']:.1f}) → "
         f"ratio = {v_mean[v_peak_L]/max(stats['v_prop_mean'],1e-9):.1f}x")
    print(s); lines.append(s)
    # Late-layer growth check (ASD's "sharp rise in deeper layers").
    early = float(a_mean[:5].mean()); late = float(a_mean[-5:].mean())
    s = (f"  audio early-vs-late: mean(L0-4)={early:.1f}  mean(L{n_layers-5}-{n_layers-1})={late:.1f}  "
         f"late/early={late/max(early,1e-9):.2f}x")
    print(s); lines.append(s)
    early_v = float(v_mean[:5].mean()); late_v = float(v_mean[-5:].mean())
    s = (f"  video early-vs-late: mean(L0-4)={early_v:.1f}  mean(L{n_layers-5}-{n_layers-1})={late_v:.1f}  "
         f"late/early={late_v/max(early_v,1e-9):.2f}x")
    print(s); lines.append(s)

    with open(out_dir / "stage2_1_decision.txt", "w") as f:
        f.write(f"Stage 2.1 — layer-wise sink counts  (n_clips={n_clips}, "
                f"D_sink={D_SINK}, τ={TAU_SINK})\n\n")
        for ln in lines:
            f.write(ln.lstrip() + "\n")
    print(f"\nwrote {out_dir / 'stage2_1_decision.txt'}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        print(f"  [note] 1 GPU visible — overriding device_map → 'auto'")
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    eps = _thinker_rms_eps(model)
    thinker_cfg = _resolve_thinker_cfg(model)
    audio_mod, visual_mod = _resolve_encoders(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    # D_sink as a long tensor on each layer's device — torch indexing copies it,
    # so we just create once and let the hook re-host if needed.
    d_sink_tensor = torch.tensor(D_SINK, dtype=torch.long)
    print(f"  n_layers={n_layers}, eps={eps}, D_sink={D_SINK}, τ={TAU_SINK}")

    clip_dir = Path(args.video_dir)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No .mp4 in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips))[: args.n_clips]
    clips = [clips[i] for i in idx]
    print(f"\nVGGSounder pass: {len(clips)} clips")

    per_clip: list = []
    failures: dict = {}
    for clip in tqdm(clips, desc="VGGSounder"):
        result, err = process_clip(
            model, processor, clip, thinker_cfg, audio_mod, visual_mod,
            layers, eps, d_sink_tensor)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            continue
        per_clip.append(result)
    if failures:
        print(f"  failures: {failures}")
    if not per_clip:
        raise SystemExit("No clips succeeded.")

    stats = aggregate_and_report(per_clip, n_layers, out_dir)

    # Save per-clip arrays so future replots are free.
    np.savez_compressed(
        out_dir / "per_clip_counts.npz",
        layers=np.arange(n_layers),
        audio_count=stats["audio_mat"],
        video_count=stats["video_mat"],
        audio_prop_sink=stats["audio_prop_sink_mat"],
        video_prop_sink=stats["video_prop_sink_mat"],
        n_a_llm=np.array([d["n_a_llm"] for d in per_clip]),
        n_v_llm=np.array([d["n_v_llm"] for d in per_clip]),
        a_prop=np.array([d["audio_prop"] for d in per_clip]),
        v_prop=np.array([d["video_prop"] for d in per_clip]),
    )
    print(f"wrote {out_dir / 'per_clip_counts.npz'}")
    print(f"wrote {out_dir / 'layer_sink_counts.csv'}")

    plot_figure(stats, out_dir)
    print_sanity_and_verdict(stats, len(per_clip), out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",
                   default=str(_REPO / "data/VGGSounder/videos"))
    p.add_argument("--n_clips", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_map", default="balanced_low_0",
                   help="balanced_low_0 only — auto crashes this model.")
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_1_layer_sinks"))
    args = p.parse_args()
    main(args)
