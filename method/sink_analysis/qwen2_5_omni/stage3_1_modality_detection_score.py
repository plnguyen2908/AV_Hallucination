"""
stage3_1_modality_detection_score.py

Stage 3.1 — ASD Modality Detection Score (MDS) replication on Qwen2.5-Omni.

ASD's MDS (arXiv 2605.10815) is a per-token, per-layer cross-modal probe:
for a sink token i at layer l, compare the attention token i RECEIVES from
the video span vs from the audio span, classify it, and aggregate over
sink tokens to a per-layer score.

Classification (per ASD):
  Let  a_v(i) = sum over q in VIDEO span of attn[L][q, i], mean over heads
       a_a(i) = sum over q in AUDIO span of attn[L][q, i], mean over heads
       med_v = median of a_v over the REFERENCE token set at layer L
       med_a = median of a_a over the same set
  Then
       uni_video(i) iff   a_v(i) >= med_v   AND   a_a(i) <  med_a
       uni_audio(i) iff   a_v(i) <  med_v   AND   a_a(i) >= med_a
       cross-modal  iff   anything else
  Per-token signed score:
       score(i) = +1 (uni_video)  /  -1 (uni_audio)  /  0 (cross-modal)
  Per-layer per-modality MDS:
       MDS_video[L] = mean over (sink positions in the VIDEO span) of score(i)
       MDS_audio[L] = mean over (sink positions in the AUDIO span) of score(i)
  ASD published targets to reproduce (LLM-emerged sinks):
       video MDS ≈ +0.45     audio MDS ≈ −0.49
  (Positive = the sink behaves video-uni; negative = audio-uni.)

REFERENCE SET (the median's denominator)
  ASD's exact convention is not transcribed in this codebase. The default
  here is "all tokens in the prompt sequence at layer L" — i.e., the
  median is taken over the full per-token a_v and a_a vectors. This is
  NOT the median over sinks (which would be circular). Flagged in stdout.
  --ref_set non_sink switches to median over non-sink positions per layer.

POPULATIONS — compute MDS twice, with the two sink criteria the project
already uses (do not conflate):
  P_llm (LLM-emerged) — per Stage 2.1 / 2.4:
        pure RMSNorm (no learned weight), D_sink={458, 2570}, τ=20.
        Per-layer mask.  THIS is the ASD-comparable population.
  P_prop (encoder-propagated) — per Stage 1.1 / 1.2:
        encoder L2 norm > 100 (Sink-or-Not τ), aligned encoder→LLM.
        Fixed across layers.  Novel measurement.

SATURATION WARNING (P_llm only)
  Audio is ~74% sink by L21 and ~96% by L25. When nearly all audio tokens
  are sinks, MDS over the P_llm column becomes near-tautological ("audio
  attends to audio"). The CSV reports the per-layer sink fraction next to
  every MDS so this is visible; lean on P_prop (sparse, fixed) and on
  layers where the P_llm sink fraction is well under 1.

PER-LAYER, never averaged across layers.

Outputs:
    stage3_1_per_clip_layer.csv     long-form per (clip, layer, population, modality)
    stage3_1_per_layer.csv          aggregated per (layer, population, modality)
    stage3_1_mds_trajectory.png     line graph with ASD target bands
    stage3_1_decision.txt           verdict + key numbers
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
    build_conversation, load_omni, prepare_inputs, thinker_layers,
)

# Constants — match Stages 1.2 / 2.1 / 2.4 verbatim.
D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0
PROMPT_AV = "Describe what you see and hear in detail."

# NOTE: ASD's paper reports specific MDS values (~+0.45 video, ~−0.49 audio)
# but these are MODEL-SPECIFIC outputs from their pipeline, not normative
# targets to hit. The MDS itself is fully defined by the median-based
# classification above; values reported here come straight from that
# computation. No fixed comparison band on the plot or in the verdict.

DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_1_mds"


# ----------------------------------------------------------------------
# Helpers — match Stage 2.1 / Stage 1.2 exactly
# ----------------------------------------------------------------------

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
    if audio_mod is None or visual_mod is None:
        raise RuntimeError("missing audio/visual encoder modules on thinker")
    return audio_mod, visual_mod


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    return x


def align_norms(enc_norms, n_llm):
    """Same as Stage 1.2 / 2.1 align_norms."""
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


def _modal_positions(input_ids, thinker_cfg):
    """Match Stage 2.1's _modal_positions — token-ID based, handles AV
    interleaving."""
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


# ----------------------------------------------------------------------
# Per-clip forward — hook attention + hidden state + encoder norms
# ----------------------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers,
                 audio_enc, visual_enc, eps_norm, d_sink_t):
    """Return per-clip dict or None on failure. Hooks:
       - layer.self_attn (capture attention received from V and A spans;
         null out attn_weights afterwards à la Stage 1.2's memory pattern)
       - layer            (capture hidden states for P_llm sink mask)
       - audio_tower      (encoder L2 norms for P_prop)
       - visual           (encoder L2 norms for P_prop)
    All hooks fire once during the prompt forward (no generation needed)."""
    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, "av", model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}"

    audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
    if len(audio_pos) == 0 or len(video_pos) == 0:
        return None, "missing_modality"
    S = int(inputs["input_ids"].shape[1])
    n_layers = len(layers)

    # Query masks (CPU; move per device inside hooks)
    video_q_mask_cpu = torch.zeros(S, dtype=torch.bool)
    video_q_mask_cpu[video_pos] = True
    audio_q_mask_cpu = torch.zeros(S, dtype=torch.bool)
    audio_q_mask_cpu[audio_pos] = True

    # Pre-allocated CPU outputs
    attn_from_video = np.zeros((n_layers, S), dtype=np.float64)
    attn_from_audio = np.zeros((n_layers, S), dtype=np.float64)
    per_layer_h = [None] * n_layers
    a_enc_buf, v_enc_buf = [], []

    def make_attn_hook(L_idx):
        def _h(_m, _i, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            aw = out[1]                                       # (1, H, q, kv)
            head_avg = aw[0].float().mean(dim=0)              # (q, kv)
            vqm = video_q_mask_cpu.to(head_avg.device)
            aqm = audio_q_mask_cpu.to(head_avg.device)
            # MEAN over queries in each modality span (per-query average
            # attention received). The classification (a_v >= med_v etc.)
            # is invariant to mean-vs-sum since both sides scale by 1/|Q|,
            # but the mean gives a cleaner per-query interpretation for
            # the reported attention thresholds.
            attn_from_video[L_idx] = head_avg[vqm].mean(dim=0).cpu().numpy()
            attn_from_audio[L_idx] = head_avg[aqm].mean(dim=0).cpu().numpy()
            # Stage 1.2 trick: drop the heavy tensor so nothing retains it.
            return (out[0], None) + tuple(out[2:])
        return _h

    def make_layer_pre_hook(L_idx):
        """PRE-SA hook (2026-06-02 convention switch).
        Captures the INPUT to layer L's forward = residual stream that
        layer L's self-attention reads = `h_input[L]`. This is the state
        ASD evaluates φ on (Eq. 10-11) and the state Stage 5 will patch.
        Previously this hook read `out[0]` (post-SA + post-MLP layer
        output); φ is now evaluated on the pre-SA tensor instead. D_SINK
        and TAU_SINK are unchanged. Note: absolute layer indices are now
        pre-SA, so "sink at L" here ~= post-SA "sink at L-1" (one-layer
        shift; not exact at the front of the network)."""
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:                 # prompt forward only
                per_layer_h[L_idx] = hs[0].detach()
        return _h

    def make_enc_hook(buf):
        def _h(_m, _i, out):
            tok = _extract_tokens(out)
            buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
        return _h

    handles = []
    handles.append(audio_enc.register_forward_hook(make_enc_hook(a_enc_buf)))
    handles.append(visual_enc.register_forward_hook(make_enc_hook(v_enc_buf)))
    for L in range(n_layers):
        # PRE-SA hook for sink classification (captures layer INPUT).
        handles.append(layers[L].register_forward_pre_hook(make_layer_pre_hook(L)))
        # FORWARD hook on self_attn for attention extraction (unchanged).
        handles.append(layers[L].self_attn.register_forward_hook(make_attn_hook(L)))

    try:
        with torch.inference_mode():
            model.thinker(**inputs,
                          use_audio_in_video=use_aiv,
                          output_attentions=True,            # let modules return aw
                          return_dict=True,
                          use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return None, f"fwd:{type(e).__name__}"
    finally:
        for h in handles: h.remove()

    if not a_enc_buf or not v_enc_buf or any(h is None for h in per_layer_h):
        torch.cuda.empty_cache()
        return None, "no_capture"

    # P_llm mask per layer (D_sink + τ, pure RMSNorm)
    p_llm = np.zeros((n_layers, S), dtype=bool)
    for L in range(n_layers):
        h = per_layer_h[L].float()
        rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h / rms).abs()
        d_t = d_sink_t.to(h.device)
        p_llm[L] = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()

    # P_prop mask: encoder L2 norm > 100, aligned to modal positions
    a_enc = np.concatenate(a_enc_buf)
    v_enc = np.concatenate(v_enc_buf)
    a_aligned, a_tag = align_norms(a_enc, len(audio_pos))
    v_aligned, v_tag = align_norms(v_enc, len(video_pos))
    if a_aligned is None or v_aligned is None:
        torch.cuda.empty_cache()
        return None, f"align_failed_a={a_tag}_v={v_tag}"
    p_prop = np.zeros(S, dtype=bool)
    p_prop[audio_pos] = a_aligned > TAU_PROP
    p_prop[video_pos] = v_aligned > TAU_PROP

    torch.cuda.empty_cache()
    return dict(
        attn_from_video=attn_from_video,           # (n_layers, S)
        attn_from_audio=attn_from_audio,           # (n_layers, S)
        p_llm=p_llm,                               # (n_layers, S)
        p_prop=p_prop,                             # (S,)
        audio_pos=audio_pos, video_pos=video_pos,
        S=S, align_tag_a=a_tag, align_tag_v=v_tag,
    ), None


# ----------------------------------------------------------------------
# MDS per clip + aggregation
# ----------------------------------------------------------------------

def per_clip_mds(attn_v, attn_a, sink_mask, video_pos, audio_pos,
                 ref_set="all"):
    """REVISED MDS formulation (per user's correction):

      Per-token signed continuous score, range [-1, +1]:
          mds_i = (a_v(i) - a_a(i)) / (a_v(i) + a_a(i))
        +1 = all attention from video; -1 = all from audio.

      Per-layer SPAN-SPECIFIC median thresholds (scalars per layer):
          mds_v_thresh[L] = median of mds_i over tokens i in VIDEO span at L
          mds_a_thresh[L] = median of mds_i over tokens i in AUDIO span at L
        Interpretation: the typical mds_i of a token sitting in the video
        span / audio span. Video-span tokens are typically video-attended
        (mds_v_thresh > 0); audio-span tokens are typically audio-attended
        (mds_a_thresh < 0).

      Per-token classification:
          uni-video    if mds_i > mds_v_thresh   (more video-leaning than
                                                  the typical video-span token)
          uni-audio    if mds_i < mds_a_thresh   (more audio-leaning than
                                                  the typical audio-span token)
          cross_sink   otherwise

      Per-layer aggregate (signed +1/-1/0 score → mean over sinks):
          MDS_video[L] = mean over sinks IN VIDEO SPAN
          MDS_audio[L] = mean over sinks IN AUDIO SPAN

    Returns:
      mds_v, mds_a              per-layer aggregate MDS (over sinks in span)
      n_v, n_a                  number of sinks in each span at each layer
      n_uv_all, n_ua_all        # sinks (any span) classified uni-V / uni-A
      n_cross_all               # sinks classified cross_sink
      n_total_all               total sinks at each layer
      mds_v_thresh, mds_a_thresh per-layer THRESHOLD scalars (the medians)
    The `ref_set` arg is accepted for parity but no longer affects the threshold
    (the thresholds are span-specific medians now)."""
    n_layers, S = attn_v.shape
    if sink_mask.ndim == 1:
        sink_mask = np.broadcast_to(sink_mask[None, :], (n_layers, S)).copy()

    # Per-token continuous MDS in [-1, +1]; eps guards 0/0 (system/query
    # tokens that receive ~no attention from either span — those rare cells
    # default to mds_i = 0).
    eps = 1e-12
    mds_i = (attn_v - attn_a) / (attn_v + attn_a + eps)

    # Span masks (positions, not per-layer)
    video_idx = np.zeros(S, dtype=bool); video_idx[video_pos] = True
    audio_idx = np.zeros(S, dtype=bool); audio_idx[audio_pos] = True

    # Span-specific median thresholds (scalars per layer)
    mds_v_thresh = np.median(mds_i[:, video_idx], axis=1)
    mds_a_thresh = np.median(mds_i[:, audio_idx], axis=1)

    # Classification (cond_uv and cond_ua are mutually exclusive whenever
    # mds_v_thresh >= mds_a_thresh, which is the normal case)
    cond_uv = mds_i > mds_v_thresh[:, None]
    cond_ua = mds_i < mds_a_thresh[:, None]
    score = cond_uv.astype(np.int8) - cond_ua.astype(np.int8)

    video_sink_at = sink_mask & video_idx[None, :]
    audio_sink_at = sink_mask & audio_idx[None, :]
    n_v = video_sink_at.sum(axis=1)
    n_a = audio_sink_at.sum(axis=1)

    mds_v = np.full(n_layers, np.nan)
    mds_a = np.full(n_layers, np.nan)
    for L in range(n_layers):
        if n_v[L] > 0:
            mds_v[L] = float(score[L][video_sink_at[L]].mean())
        if n_a[L] > 0:
            mds_a[L] = float(score[L][audio_sink_at[L]].mean())

    # Counts among ALL sinks (any span) at each layer
    all_sink_at = sink_mask
    n_uv_all    = (cond_uv & all_sink_at).sum(axis=1)
    n_ua_all    = (cond_ua & all_sink_at).sum(axis=1)
    n_cross_all = all_sink_at.sum(axis=1) - n_uv_all - n_ua_all  # mutually exclusive
    n_total     = all_sink_at.sum(axis=1)
    return (mds_v, mds_a, n_v, n_a,
             n_uv_all, n_ua_all, n_cross_all, n_total,
             mds_v_thresh, mds_a_thresh)


def aggregate_per_layer(rows: list) -> pd.DataFrame:
    """rows: list of per-clip per-layer per-population per-modality records.
    Aggregate by averaging across clips (weighted by sink count)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _wmean(g, col, w_col):
        v = g[col].values; w = g[w_col].values
        mask = np.isfinite(v) & (w > 0)
        if not mask.any():
            return float("nan")
        return float((v[mask] * w[mask]).sum() / w[mask].sum())

    def _agg(g):
        out = dict(
            n_clips=int(len(g)),
            mds_video_llm   = _wmean(g, "mds_video_llm",   "n_video_sinks_llm"),
            mds_audio_llm   = _wmean(g, "mds_audio_llm",   "n_audio_sinks_llm"),
            mds_video_prop  = _wmean(g, "mds_video_prop",  "n_video_sinks_prop"),
            mds_audio_prop  = _wmean(g, "mds_audio_prop",  "n_audio_sinks_prop"),
            # Per-clip min/max envelope for the MDS lines (across clips)
            mds_video_llm_max  = float(g["mds_video_llm"].max()),
            mds_video_llm_min  = float(g["mds_video_llm"].min()),
            mds_audio_llm_max  = float(g["mds_audio_llm"].max()),
            mds_audio_llm_min  = float(g["mds_audio_llm"].min()),
            mds_video_prop_max = float(g["mds_video_prop"].max()),
            mds_video_prop_min = float(g["mds_video_prop"].min()),
            mds_audio_prop_max = float(g["mds_audio_prop"].max()),
            mds_audio_prop_min = float(g["mds_audio_prop"].min()),
            mean_n_video_sinks_llm  = float(g["n_video_sinks_llm"].mean()),
            mean_n_audio_sinks_llm  = float(g["n_audio_sinks_llm"].mean()),
            mean_n_video_sinks_prop = float(g["n_video_sinks_prop"].mean()),
            mean_n_audio_sinks_prop = float(g["n_audio_sinks_prop"].mean()),
            mean_video_llm_sink_fraction = float(g["video_llm_sink_fraction"].mean()),
            mean_audio_llm_sink_fraction = float(g["audio_llm_sink_fraction"].mean()),
            mean_video_prop_sink_fraction = float(g["video_prop_sink_fraction"].mean()),
            mean_audio_prop_sink_fraction = float(g["audio_prop_sink_fraction"].mean()),
            # Span-specific median MDS thresholds (mean across clips of per-clip values)
            mds_v_thresh=float(g["mds_v_thresh"].mean()),
            mds_a_thresh=float(g["mds_a_thresh"].mean()),
            mds_v_thresh_std=float(g["mds_v_thresh"].std()),
            mds_a_thresh_std=float(g["mds_a_thresh"].std()),
            # Classification counts pooled across clips
            sum_n_uv_llm    = int(g["n_uv_llm"].sum()),
            sum_n_ua_llm    = int(g["n_ua_llm"].sum()),
            sum_n_cross_llm = int(g["n_cross_llm"].sum()),
            sum_n_total_llm = int(g["n_total_sinks_llm"].sum()),
            sum_n_uv_prop    = int(g["n_uv_prop"].sum()),
            sum_n_ua_prop    = int(g["n_ua_prop"].sum()),
            sum_n_cross_prop = int(g["n_cross_prop"].sum()),
            sum_n_total_prop = int(g["n_total_sinks_prop"].sum()),
        )
        def _frac(k, tot):
            return float(out[k] / out[tot]) if out[tot] > 0 else float("nan")
        out["frac_uv_llm"]    = _frac("sum_n_uv_llm",    "sum_n_total_llm")
        out["frac_ua_llm"]    = _frac("sum_n_ua_llm",    "sum_n_total_llm")
        out["frac_cross_llm"] = _frac("sum_n_cross_llm", "sum_n_total_llm")
        out["frac_uv_prop"]   = _frac("sum_n_uv_prop",    "sum_n_total_prop")
        out["frac_ua_prop"]   = _frac("sum_n_ua_prop",    "sum_n_total_prop")
        out["frac_cross_prop"]= _frac("sum_n_cross_prop", "sum_n_total_prop")
        return pd.Series(out)

    return df.groupby("layer").apply(_agg).reset_index()


# ----------------------------------------------------------------------
# Plot + decision
# ----------------------------------------------------------------------

def plot_trajectory(agg, out_path, asd_layer=None):
    fig, (ax_mds, ax_sf) = plt.subplots(
        2, 1, figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.20}, sharex=True)
    L = agg["layer"].values

    # Median-based MDS; no fixed comparison band, just the 0 line.
    ax_mds.axhline(0, color="black", lw=0.6, ls=":")

    # Per-clip min/max envelope around each mean line. Envelope shows the
    # full per-clip range at each layer; the mean line is token-weighted.
    for col, color in (("video_llm", "#d62728"), ("audio_llm", "#1f77b4"),
                        ("video_prop", "#ff7f0e"), ("audio_prop", "#2ca02c")):
        ax_mds.fill_between(L, agg[f"mds_{col}_min"], agg[f"mds_{col}_max"],
                             color=color, alpha=0.10, linewidth=0)

    ax_mds.plot(L, agg["mds_video_llm"], marker="o", lw=2, color="#d62728",
                label="MDS_video (LLM-emerged)  mean ± per-clip range")
    ax_mds.plot(L, agg["mds_audio_llm"], marker="o", lw=2, color="#1f77b4",
                label="MDS_audio (LLM-emerged)")
    ax_mds.plot(L, agg["mds_video_prop"], marker="s", lw=2, ls="--",
                color="#ff7f0e", label="MDS_video (propagated)")
    ax_mds.plot(L, agg["mds_audio_prop"], marker="s", lw=2, ls="--",
                color="#2ca02c", label="MDS_audio (propagated)")

    if asd_layer is not None and 0 <= asd_layer < len(L):
        ax_mds.axvline(asd_layer, color="gray", ls=":", alpha=0.7,
                        label=f"ASD reference layer = L{asd_layer}")
    ax_mds.set_ylabel("Modality Detection Score (MDS)", fontsize=11)
    ax_mds.set_ylim(-1.05, 1.05)
    ax_mds.grid(True, ls=":", alpha=0.4)
    ax_mds.legend(loc="lower right", fontsize=8, ncol=1)
    ax_mds.set_title(
        "Stage 3.1 — MDS per layer  (Qwen2.5-Omni, VGGSounder)\n"
        "+1 = uni-video, −1 = uni-audio, 0 = cross-modal  "
        "(median-based classification; no fixed reference)",
        fontsize=11)

    # Sink fractions
    ax_sf.plot(L, agg["mean_video_llm_sink_fraction"], marker="o", lw=1.5,
                color="#d62728", label="P_llm sink-frac in video span")
    ax_sf.plot(L, agg["mean_audio_llm_sink_fraction"], marker="o", lw=1.5,
                color="#1f77b4", label="P_llm sink-frac in audio span")
    ax_sf.plot(L, agg["mean_video_prop_sink_fraction"], marker="s", lw=1.2,
                ls="--", color="#ff7f0e", label="P_prop sink-frac in video span")
    ax_sf.plot(L, agg["mean_audio_prop_sink_fraction"], marker="s", lw=1.2,
                ls="--", color="#2ca02c", label="P_prop sink-frac in audio span")
    ax_sf.axhline(0.5, color="gray", ls=":", alpha=0.7,
                  label="0.5 (saturation onset)")
    ax_sf.set_xlabel("LLM decoder layer L", fontsize=11)
    ax_sf.set_ylabel("sink fraction", fontsize=10)
    ax_sf.set_ylim(0, 1.05)
    ax_sf.grid(True, ls=":", alpha=0.4)
    ax_sf.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_classification_counts(agg, out_path):
    """Per layer, fraction of sinks classified as uni-video (a_v >= med_v),
    uni-audio (a_a <= med_a), or both (the intersection). Two panels: one
    per sink population. Pooled across clips (sum of per-clip counts).
    Note: 'uni-video' and 'uni-audio' here are the independent indicator
    conditions, so they OVERLAP — a sink can be both. 'both' = the
    intersection cell (high-V, low-A; the strongest pure-video signal)."""
    L = agg["layer"].values
    fig, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, suf, title in ((ax_l, "llm", "LLM-emerged (P_llm) sinks"),
                            (ax_p, "prop", "Propagated (P_prop) sinks")):
        ax.plot(L, agg[f"frac_uv_{suf}"],    marker="o", lw=2,
                color="#d62728",
                label="uni-video  (mds_i > mds_v_thresh)")
        ax.plot(L, agg[f"frac_ua_{suf}"],    marker="o", lw=2,
                color="#1f77b4",
                label="uni-audio  (mds_i < mds_a_thresh)")
        ax.plot(L, agg[f"frac_cross_{suf}"], marker="s", lw=1.6, ls="--",
                color="#2ca02c",
                label="cross_sink (between thresholds)")
        ax.set_xlabel("LLM decoder layer L", fontsize=11)
        ax.set_ylabel("fraction of sinks", fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=11)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Stage 3.1 — fraction of sinks classified by each "
                 "median-based condition (pooled across clips)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_decision(agg, out_path, n_clips, ref_set, asd_layer=None):
    """Median-based MDS — no fixed comparison target. Report extreme layers
    (max video MDS, min audio MDS) for each population so the strongest
    signals per modality per population are visible."""

    def _argmax(col):
        v = agg[col].values
        idx = np.nanargmax(v)
        return int(agg.iloc[idx]["layer"]), agg.iloc[idx]

    def _argmin(col):
        v = agg[col].values
        idx = np.nanargmin(v)
        return int(agg.iloc[idx]["layer"]), agg.iloc[idx]

    L_v_llm, r_v_llm = _argmax("mds_video_llm")
    L_a_llm, r_a_llm = _argmin("mds_audio_llm")
    L_v_prop, r_v_prop = _argmax("mds_video_prop")
    L_a_prop, r_a_prop = _argmin("mds_audio_prop")

    with open(out_path, "w") as f:
        f.write("Stage 3.1 — MDS per layer on Qwen2.5-Omni  "
                f"(n_clips={n_clips}, dataset=VGGSounder)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Reference set for the median: {ref_set} "
                "(ASD's exact choice not transcribed; this is the script's "
                "default, flagged.)\n")
        f.write("MDS formula (median-based classification):\n"
                "  uni-video(i) iff a_v(i) >= med_v AND a_a(i) <  med_a\n"
                "  uni-audio(i) iff a_v(i) <  med_v AND a_a(i) >= med_a\n"
                "  score(i) = +1 / -1 / 0\n"
                "  MDS_video[L] = mean score over sinks in VIDEO span\n"
                "  MDS_audio[L] = mean score over sinks in AUDIO span\n"
                "No fixed comparison target — values reported straight from the "
                "median-based classification.\n\n")

        if asd_layer is not None:
            row = agg[agg.layer == asd_layer]
            if not row.empty:
                f.write(f"At reference layer L{asd_layer}:\n")
                f.write(f"  P_llm   MDS_video = {row['mds_video_llm'].values[0]:+.3f}, "
                        f"MDS_audio = {row['mds_audio_llm'].values[0]:+.3f}\n")
                f.write(f"  P_prop  MDS_video = {row['mds_video_prop'].values[0]:+.3f}, "
                        f"MDS_audio = {row['mds_audio_prop'].values[0]:+.3f}\n\n")

        f.write("Extreme layers (per population × modality):\n")
        f.write(f"  P_llm  max MDS_video = {r_v_llm['mds_video_llm']:+.3f} at L{L_v_llm}  "
                f"(audio-sink-fraction here = {r_v_llm['mean_audio_llm_sink_fraction']:.2f})\n")
        f.write(f"  P_llm  min MDS_audio = {r_a_llm['mds_audio_llm']:+.3f} at L{L_a_llm}  "
                f"(audio-sink-fraction here = {r_a_llm['mean_audio_llm_sink_fraction']:.2f})\n")
        f.write(f"  P_prop max MDS_video = {r_v_prop['mds_video_prop']:+.3f} at L{L_v_prop}  "
                f"(prop-frac video = {r_v_prop['mean_video_prop_sink_fraction']:.3f})\n")
        f.write(f"  P_prop min MDS_audio = {r_a_prop['mds_audio_prop']:+.3f} at L{L_a_prop}  "
                f"(prop-frac audio = {r_a_prop['mean_audio_prop_sink_fraction']:.3f})\n\n")

        f.write("Caveats:\n")
        late_sat = agg[(agg.layer >= 20) &
                       (agg["mean_audio_llm_sink_fraction"] >= 0.5)]
        if not late_sat.empty:
            f.write(f"  - P_llm audio saturation flagged at {len(late_sat)} layers "
                    f"(≥ 50% audio tokens are P_llm sinks); P_llm audio MDS at "
                    f"those layers is near-tautological — lean on mid-stack and "
                    f"P_prop.\n")
        f.write("  - P_prop is the sparse, fixed-across-layers population that "
                "feeds Stage 3.2 / Stage 5.\n")
        f.write("  - If reference-set choice flips a number, rerun with "
                "--ref_set non_sink for comparison.\n")
    print(f"wrote {out_path}")
    return (L_v_llm, L_a_llm, L_v_prop, L_a_prop), (r_v_llm, r_a_llm, r_v_prop, r_a_prop)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    eps_norm = _thinker_rms_eps(model)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    audio_enc, visual_enc = _resolve_encoders(model)
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ_sink={TAU_SINK}, "
          f"τ_prop={TAU_PROP}, eps={eps_norm}, ref_set={args.ref_set!r}")

    video_dir = Path(args.video_dir)
    clips_all = sorted(video_dir.glob("*.mp4"))
    if not clips_all:
        raise SystemExit(f"no .mp4 in {video_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed})\n")

    rows = []
    failures: dict = {}
    for clip in tqdm(clips, desc="clips"):
        result, err = process_clip(model, processor, clip, thinker_cfg, layers,
                                    audio_enc, visual_enc, eps_norm, d_sink_t)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {clip.name}: {err}")
            continue
        attn_v = result["attn_from_video"]
        attn_a = result["attn_from_audio"]
        n_v_tok = len(result["video_pos"])
        n_a_tok = len(result["audio_pos"])

        # P_llm (per-layer). The two thresholds (mds_v_thresh, mds_a_thresh)
        # are population-independent (span medians of mds_i), but per_clip_mds
        # returns them; we use the P_llm call's thresholds as the canonical
        # ones for the CSV.
        (mds_v_llm, mds_a_llm, n_vs_llm, n_as_llm,
         n_uv_llm, n_ua_llm, n_cross_llm, n_tot_llm,
         mds_v_thresh, mds_a_thresh) = per_clip_mds(
            attn_v, attn_a, result["p_llm"], result["video_pos"],
            result["audio_pos"], ref_set=args.ref_set)
        # P_prop (fixed; broadcast inside per_clip_mds). Thresholds are the
        # same (computed on all-token mds_i within each span); discard duplicates.
        (mds_v_prop, mds_a_prop, n_vs_prop, n_as_prop,
         n_uv_prop, n_ua_prop, n_cross_prop, n_tot_prop,
         _t1, _t2) = per_clip_mds(
            attn_v, attn_a, result["p_prop"], result["video_pos"],
            result["audio_pos"], ref_set=args.ref_set)

        for L in range(n_layers):
            rows.append(dict(
                clip=clip.name, layer=L,
                mds_video_llm=float(mds_v_llm[L]) if np.isfinite(mds_v_llm[L]) else np.nan,
                mds_audio_llm=float(mds_a_llm[L]) if np.isfinite(mds_a_llm[L]) else np.nan,
                mds_video_prop=float(mds_v_prop[L]) if np.isfinite(mds_v_prop[L]) else np.nan,
                mds_audio_prop=float(mds_a_prop[L]) if np.isfinite(mds_a_prop[L]) else np.nan,
                n_video_sinks_llm=int(n_vs_llm[L]),
                n_audio_sinks_llm=int(n_as_llm[L]),
                n_video_sinks_prop=int(n_vs_prop[L]),
                n_audio_sinks_prop=int(n_as_prop[L]),
                video_llm_sink_fraction=float(n_vs_llm[L] / max(n_v_tok, 1)),
                audio_llm_sink_fraction=float(n_as_llm[L] / max(n_a_tok, 1)),
                video_prop_sink_fraction=float(n_vs_prop[L] / max(n_v_tok, 1)),
                audio_prop_sink_fraction=float(n_as_prop[L] / max(n_a_tok, 1)),
                n_video_tokens=int(n_v_tok), n_audio_tokens=int(n_a_tok),
                # classification counts among ALL sinks (any span) at this layer
                n_uv_llm=int(n_uv_llm[L]),     n_ua_llm=int(n_ua_llm[L]),
                n_cross_llm=int(n_cross_llm[L]), n_total_sinks_llm=int(n_tot_llm[L]),
                n_uv_prop=int(n_uv_prop[L]),   n_ua_prop=int(n_ua_prop[L]),
                n_cross_prop=int(n_cross_prop[L]), n_total_sinks_prop=int(n_tot_prop[L]),
                # Per-layer SPAN-SPECIFIC median thresholds (scalars):
                #   mds_v_thresh = median of per-token mds_i over VIDEO-span tokens
                #   mds_a_thresh = median of per-token mds_i over AUDIO-span tokens
                # where mds_i = (mean_attn_v_to_i - mean_attn_a_to_i)
                #              / (mean_attn_v_to_i + mean_attn_a_to_i)
                mds_v_thresh=float(mds_v_thresh[L]),
                mds_a_thresh=float(mds_a_thresh[L]),
            ))

    if failures:
        print(f"  failures: {failures}")
    if not rows:
        raise SystemExit("no clips processed")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage3_1_per_clip_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage3_1_per_clip_layer.csv'}  "
          f"({len(df)} per-(clip, layer) records)")

    agg = aggregate_per_layer(rows)
    agg.to_csv(out_dir / "stage3_1_per_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage3_1_per_layer.csv'}")

    plot_trajectory(agg, out_dir / "stage3_1_mds_trajectory.png",
                     asd_layer=args.asd_layer)
    plot_classification_counts(agg,
                                out_dir / "stage3_1_classification_counts.png")
    extreme_layers, extreme_rows = write_decision(
        agg, out_dir / "stage3_1_decision.txt",
        n_clips=df["clip"].nunique(), ref_set=args.ref_set,
        asd_layer=args.asd_layer)
    L_v_llm, L_a_llm, L_v_prop, L_a_prop = extreme_layers
    r_v_llm, r_a_llm, r_v_prop, r_a_prop = extreme_rows

    print("\n" + "=" * 80)
    print(f"STAGE 3.1 MDS — per-layer (n_clips = {df['clip'].nunique()}, "
          f"ref_set={args.ref_set!r})")
    print("=" * 80)
    cols = ["layer", "mds_video_llm", "mds_audio_llm",
            "mds_video_prop", "mds_audio_prop",
            "mean_video_llm_sink_fraction", "mean_audio_llm_sink_fraction"]
    print(agg[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)))
    print()
    print("Extreme layers (per population × modality):")
    print(f"  P_llm  max MDS_video = {r_v_llm['mds_video_llm']:+.3f} at L{L_v_llm}  "
          f"(audio-sink-frac there = {r_v_llm['mean_audio_llm_sink_fraction']:.2f})")
    print(f"  P_llm  min MDS_audio = {r_a_llm['mds_audio_llm']:+.3f} at L{L_a_llm}  "
          f"(audio-sink-frac there = {r_a_llm['mean_audio_llm_sink_fraction']:.2f})")
    print(f"  P_prop max MDS_video = {r_v_prop['mds_video_prop']:+.3f} at L{L_v_prop}")
    print(f"  P_prop min MDS_audio = {r_a_prop['mds_audio_prop']:+.3f} at L{L_a_prop}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--n_clips", type=int, default=50,
                   help="Default 50 = smoke / kill-check pass.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref_set", choices=["all", "non_sink"], default="all",
                   help="Reference token set for the median. ASD's exact "
                        "choice not transcribed; 'all' is the default (flagged).")
    p.add_argument("--asd_layer", type=int, default=None,
                   help="If ASD reports a specific layer, pass it here for "
                        "side-by-side reporting. Default: best-match layer is "
                        "auto-selected and printed.")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
