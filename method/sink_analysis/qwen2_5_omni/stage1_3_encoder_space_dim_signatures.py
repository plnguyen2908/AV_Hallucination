"""
stage1_3_encoder_space_dim_signatures.py

Stage 1.3 (encoder-space refinement) — Does the encoder-propagated population
have a distinctive dim in PRE-PROJECTION encoder space, even though the LLM
hidden-state branch of Stage 1.3 (stage1_3_dimension_signatures.py) found none?

Hypothesis: P_prop's high norm originates in the encoder. The distinctive
dimensional signature may live in the encoder's NATIVE feature space (1280-dim
for vision, d_model for audio) and get SMEARED by the projector when mapped to
LLM dim 3584. Sink-or-Not-to-Sink/Darcet identify ViT sinks in encoder space
(feature norm), so checking pre-projection dimensions is the faithful analog.

Hook points (PRE-projection):
- Vision: `visual.blocks[-1]` output (1280-dim per patch, post-blocks/pre-merger).
- Audio:  `audio_tower.avg_pooler` output (d_model-dim, pre-proj).

Classification (within-clip percentiles — robust without needing to know the
absolute scale, which differs per modality and from post-projection space):
- P_prop_pre: top 5% by per-token L2 norm in pre-projection space.
- P_nonprop_pre: bottom 50% by per-token L2 norm in pre-projection space.

Per-dim profile: per-clip median |x_enc_pre[d]|, then median-of-medians across
clips. Distinctiveness = prop_median / max(nonprop_median, eps). A dim is
"distinctive" if distinctiveness > 1.5 AND pop_median > 3 (per-token magnitude
threshold; recalibrated separately from post-proj where activations are larger).

Outputs (--output_dir):
    encoder_space_distinctiveness.csv  per (modality, dim) score + active frac
    encoder_space_profiles.png         per-dim median for prop vs nonprop
    encoder_space_topk.png             top-K bar chart per modality
    encoder_space_profiles.npz         pre-projection medians + thresholds
    stage1_3_encoder_space_decision.txt  verdict + projector-bridge findings

Optional projector bridge (audio only — vision merger is non-linear reshape):
- For the top distinctive audio pre-proj dim d_pre, push e_{d_pre} through the
  audio projector `proj` (Linear d_model→3584). Top dims of |W e_{d_pre}| show
  where the encoder-side signal lands in LLM-dim space after projection.
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


PROMPT_AV = "Describe what you see and hear in detail."
# Default sweep over P_prop percentile thresholds (top 5/10/20/50%). Per-clip
# classification: P_prop_pre = top (100-pctl)% by pre-proj norm.
DEFAULT_PROP_PCTLS = [95.0, 90.0, 80.0, 50.0]
NONPROP_PCTL = 50.0    # bottom 50% by pre-proj norm → P_nonprop_pre (fixed)
DISTINCT_THRESHOLD = 1.5
ACTIVE_FRAC_MIN = 0.30                # user spec: >30% of P_prop tokens active
TOPK = 15
# Per-token "active" threshold (used to compute active_frac per dim). Set low
# so audio's smaller-scale pre-proj activations aren't excluded by an over-large
# absolute floor; video's signal is many orders of magnitude above this anyway.
DEFAULT_TOKEN_ACTIVE_THRESHOLD = 1.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_encoders(model):
    """Return (audio_tower, visual)."""
    thinker = model.thinker
    audio_mod = visual_mod = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr); break
    return audio_mod, visual_mod


def _pre_proj_hook_targets(audio_mod, visual_mod):
    """Find the pre-projection hook points.
    Audio: avg_pooler output (d_model-dim, just before .proj Linear).
    Vision: blocks[-1] output (1280-dim, just before .merger)."""
    audio_pre = getattr(audio_mod, "avg_pooler", None)
    visual_pre = visual_mod.blocks[-1] if hasattr(visual_mod, "blocks") else None
    return audio_pre, visual_pre


def _hook_targets_for_space(audio_mod, visual_mod, space):
    """Pick hook modules and unpackers for a given feature space.
    space='pre': pre-projection (audio_tower.avg_pooler 2D; visual.blocks[-1] 1280).
    space='post': post-projection encoder OUTPUT (audio_tower/visual modules; both
    3584-dim — same as LLM hidden size, fed straight into the LLM input)."""
    if space == "pre":
        a_hook, v_hook = _pre_proj_hook_targets(audio_mod, visual_mod)
        return a_hook, v_hook, _audio_pre_unpack, _vision_pre_unpack
    if space == "post":
        return audio_mod, visual_mod, _post_unpack, _post_unpack
    raise ValueError(f"Unknown --space: {space} (expected 'pre' or 'post')")


def _post_unpack(out):
    """Encoder OUTPUT (post-projection / post-merger). Whatever audio_tower
    and visual return — (n_tokens, 3584) or (1, n, 3584) or a ModelOutput."""
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    return x.contiguous()


def _audio_pre_unpack(out):
    """avg_pooler output. In Qwen2.5-Omni's audio encoder
    (modeling_qwen2_5_omni.py:921), it's called as
        avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
    where each_audio_states is (L, C). So the input to avg_pooler is (C, L)
    (2D, unbatched), and the output is (C, L/2). We need to transpose to
    (L/2, C) so the channel dim is last (H = d_model, the actual hidden size)."""
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.dim() == 3:               # (B, C, L) -> (L, C)
        x = x[0].transpose(0, 1)
    elif x.dim() == 2:             # (C, L) -> (L, C)
        x = x.transpose(0, 1)
    return x.contiguous()


def _vision_pre_unpack(out):
    """blocks[-1] output: typically (n_patches, hidden=1280) or
    (1, n_patches, 1280). Squeeze."""
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.dim() == 3:
        x = x[0]
    return x.contiguous()


# --------------------------------------------------------------------------
# Per-clip processing
# --------------------------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg,
                 audio_hook, visual_hook, audio_unpack, visual_unpack):
    """Forward thinker, capture features at the supplied audio/visual hook
    points (pre- or post-projection, depending on the caller's choice).
    Returns dict {audio: (n_a, H_a), video: (n_v, H_v)} or None."""
    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, "av", model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}:{e}"

    a_buf, v_buf = [], []
    def make_hook(buf, unpack):
        def _h(_m, _i, out):
            x = unpack(out)
            buf.append(x.detach().float().cpu().numpy())
        return _h

    h_a = audio_hook.register_forward_hook(make_hook(a_buf, audio_unpack))
    h_v = visual_hook.register_forward_hook(make_hook(v_buf, visual_unpack))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, output_hidden_states=False,
                          use_audio_in_video=use_aiv,
                          return_dict=True, use_cache=False)
    except Exception as e:
        return None, f"fwd:{type(e).__name__}:{e}"
    finally:
        for h in (h_a, h_v):
            h.remove()
        torch.cuda.empty_cache()

    if not a_buf or not v_buf:
        return None, "no_hooks"
    # Audio pre-proj: avg_pooler may fire multiple times in chunked AV; concat.
    a_pre = np.concatenate(a_buf, axis=0)         # (n_a, d_model)
    v_pre = np.concatenate(v_buf, axis=0)         # (n_v_patches, 1280)
    return {"audio": a_pre, "video": v_pre}, None


# --------------------------------------------------------------------------
# Distinctiveness aggregation + analysis
# --------------------------------------------------------------------------

def _classify_and_stats(features, prop_pctl):
    """Per-clip percentile-based classification.
    features: (n, H). prop_pctl: percentile threshold for P_prop (e.g., 95 →
    top 5%). Returns dict with per-pop median |x|, active fraction, norm thresholds."""
    norms = np.linalg.norm(features, axis=-1)
    if norms.size < 4:
        return None
    tau_high = float(np.percentile(norms, prop_pctl))
    tau_low = float(np.percentile(norms, NONPROP_PCTL))
    prop_mask = norms >= tau_high
    nonprop_mask = norms <= tau_low
    out = {"tau_high": tau_high, "tau_low": tau_low,
           "n_prop": int(prop_mask.sum()), "n_nonprop": int(nonprop_mask.sum())}
    for name, mask in (("prop", prop_mask), ("nonprop", nonprop_mask)):
        if mask.any():
            tok = np.abs(features[mask])
            out[f"{name}_median"] = np.median(tok, axis=0)        # (H,)
            out[f"{name}_active_frac"] = (
                tok > DEFAULT_TOKEN_ACTIVE_THRESHOLD).mean(axis=0)  # (H,)
        else:
            out[f"{name}_median"] = None
            out[f"{name}_active_frac"] = None
    return out


def aggregate(per_clip_stats, H):
    """Median-of-medians + mean-of-active-fractions across clips."""
    out = {}
    for cat in ("prop", "nonprop"):
        meds = [s[f"{cat}_median"] for s in per_clip_stats
                if s and s.get(f"{cat}_median") is not None]
        acts = [s[f"{cat}_active_frac"] for s in per_clip_stats
                if s and s.get(f"{cat}_active_frac") is not None]
        out[f"{cat}_median"] = (np.median(np.stack(meds), axis=0)
                                if meds else np.full(H, np.nan))
        out[f"{cat}_active_frac"] = (np.mean(np.stack(acts), axis=0)
                                     if acts else np.full(H, np.nan))
        out[f"n_{cat}_total"] = sum(int(s[f"n_{cat}"]) for s in per_clip_stats if s)
    out["tau_high_mean"] = float(np.mean(
        [s["tau_high"] for s in per_clip_stats if s]))
    out["tau_low_mean"] = float(np.mean(
        [s["tau_low"] for s in per_clip_stats if s]))
    return out


def distinctiveness(agg):
    """distinctiveness[d] = prop_median / max(nonprop_median, eps).
    Gate per the user spec: distinct > 1.5 AND active_frac > 0.30 (>30% of
    P_prop tokens have |x|>DEFAULT_TOKEN_ACTIVE_THRESHOLD on this dim)."""
    pm = agg["prop_median"]; nm = agg["nonprop_median"]
    af = agg["prop_active_frac"]
    denom = np.maximum(nm, 1e-12)
    dist = np.where(np.isfinite(pm) & np.isfinite(nm), pm / denom, np.nan)
    is_active = np.isfinite(af) & (af > ACTIVE_FRAC_MIN)
    is_distinct = (np.isfinite(dist) & (dist > DISTINCT_THRESHOLD)) & is_active
    return dist, is_active, is_distinct


def report_topk(modality, H, agg, dist, is_active, is_distinct,
                top_k=TOPK, out_rows=None):
    print(f"\n  ({modality}) pre-proj space, H={H}, "
          f"<tau_high>={agg['tau_high_mean']:.2f}, "
          f"<tau_low>={agg['tau_low_mean']:.2f}, "
          f"n_prop_total={agg['n_prop_total']}, "
          f"n_nonprop_total={agg['n_nonprop_total']}")
    print(f"    {'rank':<5}{'dim':<7}{'distinct':<11}"
          f"{'prop_med':<10}{'nonprop_med':<13}{'prop_active%':<14}"
          f"{'distinct?'}")
    order = np.argsort(np.where(np.isfinite(dist), dist, -np.inf))[::-1][:top_k]
    for rank, d in enumerate(order, 1):
        d = int(d)
        ap = float(agg["prop_active_frac"][d]) * 100
        print(f"    {rank:<5}{d:<7}{dist[d]:<11.2f}"
              f"{agg['prop_median'][d]:<10.2f}{agg['nonprop_median'][d]:<13.2f}"
              f"{ap:<14.1f}{str(bool(is_distinct[d]))}")
        if out_rows is not None:
            out_rows.append(dict(
                modality=modality, rank=rank, dim=d,
                distinctiveness=float(dist[d]),
                prop_median=float(agg["prop_median"][d]),
                nonprop_median=float(agg["nonprop_median"][d]),
                prop_active_frac=ap,
                is_active=bool(is_active[d]),
                is_distinctive=bool(is_distinct[d]),
            ))
    n_distinct = int(is_distinct.sum())
    print(f"    → {n_distinct} dim(s) with distinct > {DISTINCT_THRESHOLD} "
          f"AND active_frac > {ACTIVE_FRAC_MIN*100:.0f}%")


SPIKE_THR = 100.0   # prop_median absolute floor for "this is a real spike"


def plot_profiles(per_modality, out_path):
    """Per modality, P_prop vs P_nonprop median per dim. Highlight ONLY actual
    spike dims (prop_median >= SPIKE_THR); fall back to top 3 by prop_med if
    none. Stats listed in an upper-left text box. Log y-scale because the
    spike can be 4+ orders above baseline."""
    mods = list(per_modality.keys())
    fig, axes = plt.subplots(len(mods), 1, figsize=(13, 4.5 * len(mods)),
                             squeeze=False)
    for ri, m in enumerate(mods):
        ax = axes[ri, 0]
        agg = per_modality[m]["agg"]
        pm = agg["prop_median"]; nm = agg["nonprop_median"]
        H = len(pm); dims = np.arange(H)
        spike_dims = np.where(pm >= SPIKE_THR)[0]
        fallback = False
        if spike_dims.size == 0:
            spike_dims = np.argsort(pm)[::-1][:3]
            fallback = True
        spike_dims = sorted(int(s) for s in spike_dims)

        # Bands span their full width even at the edges; x-axis is padded so
        # dim 0's band/star isn't clipped against the left axis.
        BAND_HW = 6
        for sd in spike_dims:
            ax.axvspan(sd - BAND_HW, sd + BAND_HW + 1, color="gold",
                       alpha=0.5, zorder=0)
        ax.plot(dims, nm, color="#1f77b4", lw=0.6,
                label="P_nonprop pre-proj (median)", zorder=2)
        ax.plot(dims, pm, color="#d62728", lw=0.6,
                label="P_prop pre-proj (median)", zorder=2)
        # Star markers — single-dim spikes against a dense 1000+ dim line are
        # easy to miss otherwise (esp. dim 0 at the edge).
        ax.scatter([sd for sd in spike_dims],
                   [pm[sd] for sd in spike_dims],
                   s=110, marker="*", color="black", edgecolor="white",
                   linewidth=1.2, zorder=5, label="spike dim(s)")
        ax.set_yscale("log"); ax.set_xlim(-18, H + 5)
        ax.set_xlabel("pre-proj hidden dim", fontsize=10)
        ax.set_ylabel("median |x_enc[d]|  (log)", fontsize=10)
        note = (f"highlight: dims with prop_med >= {SPIKE_THR:g}"
                if not fallback
                else f"NO dim with prop_med >= {SPIKE_THR:g} — "
                     "fallback: top 3 by prop_med")
        ax.set_title(f"{m} encoder pre-proj — H={H}, "
                     f"n_prop={agg['n_prop_total']}, "
                     f"n_nonprop={agg['n_nonprop_total']}  |  {note}",
                     fontsize=11)
        ax.grid(True, ls=":", alpha=0.35)

        sd_sorted = sorted(spike_dims, key=lambda x: -pm[x])
        med_pm_across = float(np.median(pm))
        text_lines = [
            f"dim {sd:4d}  prop_med = {pm[sd]:8.2f}  "
            f"nonprop_med = {nm[sd]:7.2f}  "
            f"ratio = {pm[sd] / max(nm[sd], 1e-12):6.2f}x  "
            f"vs median(prop_med across dims) = {med_pm_across:.2f}"
            for sd in sd_sorted
        ]
        ax.text(0.012, 0.97, "\n".join(text_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="goldenrod", alpha=0.9))

        ax.legend(handles=[
            plt.Line2D([0], [0], color="#d62728", lw=1.5,
                       label="P_prop pre-proj (median)"),
            plt.Line2D([0], [0], color="#1f77b4", lw=1.5,
                       label="P_nonprop pre-proj (median)"),
            Patch(facecolor="gold", alpha=0.55,
                  label=("spike dim(s) — prop_med >= 100" if not fallback
                         else "spike dim(s) — fallback top 3")),
        ], fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_topk(per_modality, out_path):
    """Per modality, top-K distinctive dims as a horizontal bar chart."""
    mods = list(per_modality.keys())
    fig, axes = plt.subplots(1, len(mods), figsize=(7 * len(mods), 5),
                             squeeze=False)
    for ci, m in enumerate(mods):
        ax = axes[0, ci]
        agg = per_modality[m]["agg"]; dist = per_modality[m]["dist"]
        is_d = per_modality[m]["is_distinct"]
        order = np.argsort(np.where(np.isfinite(dist), dist, -np.inf))[::-1][:TOPK]
        labels = [f"dim {int(d)}" for d in order]
        vals = [float(dist[int(d)]) for d in order]
        colors = ["#d62728" if is_d[int(d)] else "lightgray" for d in order]
        ax.barh(np.arange(len(order))[::-1], vals, color=colors,
                edgecolor="black", linewidth=0.3)
        ax.set_yticks(np.arange(len(order))[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(DISTINCT_THRESHOLD, color="gray", ls="--", lw=1.0,
                   label=f"distinct = {DISTINCT_THRESHOLD}")
        ax.set_xlabel("distinctiveness = prop_med / nonprop_med", fontsize=10)
        ax.set_title(f"{m} encoder — top {TOPK} by distinctiveness", fontsize=11)
        ax.grid(axis="x", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Optional projector bridge (audio only — vision merger is non-linear reshape)
# --------------------------------------------------------------------------

def audio_projector_bridge(model, top_audio_dim, k=10):
    """For top distinctive audio pre-proj dim d, push e_d through audio_tower.proj
    (Linear d_model -> 3584). Top |W e_d| dims show where the encoder signal
    lands in LLM-dim space."""
    audio_mod, _ = _resolve_encoders(model)
    proj = getattr(audio_mod, "proj", None)
    if proj is None or not hasattr(proj, "weight"):
        return None
    W = proj.weight.detach().float().cpu().numpy()      # (3584, d_model)
    col = W[:, top_audio_dim]                            # (3584,)
    mag = np.abs(col)
    top_llm = np.argsort(mag)[::-1][:k]
    total = np.linalg.norm(col)
    frac_top = float(np.linalg.norm(col[top_llm]) / max(total, 1e-12))
    print(f"\n  Audio projector bridge: pre-proj dim {top_audio_dim} -> "
          f"LLM-dim 3584 (via audio_tower.proj weights)")
    print(f"    top {k} LLM dims of |W[:,{top_audio_dim}]|: "
          f"{[(int(d), round(float(mag[d]),4)) for d in top_llm]}")
    print(f"    L2-fraction of mass in top {k}: {frac_top*100:.1f}% "
          f"(rest spread across {3584 - k} other dims) — "
          f"{'concentrated' if frac_top > 0.5 else 'spread / smeared'}.")
    return dict(top_audio_pre_dim=int(top_audio_dim),
                top_llm_dims=list(map(int, top_llm)),
                top_llm_mags=list(map(float, mag[top_llm])),
                frac_top_mass=frac_top)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    if args.output_dir is None:
        args.output_dir = str(
            _REPO / "results/qwen2_5_omni/sink_analysis" /
            ("stage1_3_encoder_space"
             if args.space == "pre"
             else f"stage1_3_encoder_space_{args.space}"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        print(f"  [note] only 1 visible GPU — overriding device_map → 'auto'")
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    audio_mod, visual_mod = _resolve_encoders(model)
    audio_hook, visual_hook, audio_unpack, visual_unpack = \
        _hook_targets_for_space(audio_mod, visual_mod, args.space)
    if audio_hook is None or visual_hook is None:
        raise SystemExit(
            f"Could not resolve {args.space}-projection hook points. "
            f"audio_hook={audio_hook}, visual_hook={visual_hook}")
    thinker_cfg = model.thinker.config
    print(f"  space = {args.space!r}")
    print(f"  audio hook: {type(audio_hook).__name__}")
    print(f"  visual hook: {type(visual_hook).__name__}")

    clip_dir = Path(args.video_dir)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No .mp4 in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips))[: args.n_clips]
    clips = [clips[i] for i in idx]
    print(f"\nVGGSounder pass: {len(clips)} clips")

    pctls = sorted(set(float(p) for p in args.prop_pctls), reverse=True)
    # per_clip[pctl][modality] = list[clip] of stats dict
    per_clip = {p: {"audio": [], "video": []} for p in pctls}
    H_by_mod = {"audio": None, "video": None}
    failures: dict = {}
    succeeded = 0
    for clip in tqdm(clips, desc="VGGSounder"):
        result, err = process_clip(model, processor, clip, thinker_cfg,
                                   audio_hook, visual_hook,
                                   audio_unpack, visual_unpack)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            continue
        for m, feats in result.items():
            if H_by_mod[m] is None:
                H_by_mod[m] = feats.shape[-1]
            for p in pctls:
                stats = _classify_and_stats(feats, p)
                if stats is not None:
                    per_clip[p][m].append(stats)
        succeeded += 1

    if failures:
        print(f"  failures: {failures}")
    if succeeded == 0:
        raise SystemExit("No clips succeeded.")

    print(f"\n  pre-projection encoder dims: "
          f"audio H={H_by_mod['audio']}, video H={H_by_mod['video']}")

    # Per-percentile aggregation + distinctiveness reporting.
    per_pctl: dict = {}     # pctl -> {modality -> dict(agg, dist, is_active, is_distinct)}
    csv_rows: list = []
    for p in pctls:
        print("\n" + "=" * 90)
        print(f"PROP_PCTL = {p:g}  (P_prop_pre = top {100-p:g}% by pre-proj norm; "
              f"P_nonprop_pre = bottom {NONPROP_PCTL:g}%)")
        print(f"  distinctive := distinct > {DISTINCT_THRESHOLD} AND "
              f"active_frac > {ACTIVE_FRAC_MIN*100:.0f}% "
              f"(token-active threshold = {DEFAULT_TOKEN_ACTIVE_THRESHOLD})")
        print("=" * 90)
        per_modality: dict = {}
        for m in ("audio", "video"):
            if not per_clip[p][m]:
                continue
            agg = aggregate(per_clip[p][m], H_by_mod[m])
            dist, is_active, is_distinct = distinctiveness(agg)
            sub_rows: list = []
            report_topk(m, H_by_mod[m], agg, dist, is_active, is_distinct,
                        out_rows=sub_rows)
            for r in sub_rows:
                r["prop_pctl"] = p
            csv_rows.extend(sub_rows)
            per_modality[m] = dict(agg=agg, dist=dist,
                                   is_active=is_active, is_distinct=is_distinct)
        per_pctl[p] = per_modality

    pd.DataFrame(csv_rows).to_csv(
        out_dir / "encoder_space_distinctiveness_by_pctl.csv", index=False)
    print(f"\nwrote {out_dir / 'encoder_space_distinctiveness_by_pctl.csv'}")

    # Save aggregate to npz for replot.
    npz_kwargs = {}
    for p in pctls:
        for m, d in per_pctl[p].items():
            tag = f"p{int(p)}_{m}"
            npz_kwargs[f"{tag}_prop_median"] = d["agg"]["prop_median"]
            npz_kwargs[f"{tag}_nonprop_median"] = d["agg"]["nonprop_median"]
            npz_kwargs[f"{tag}_prop_active_frac"] = d["agg"]["prop_active_frac"]
            npz_kwargs[f"{tag}_nonprop_active_frac"] = d["agg"]["nonprop_active_frac"]
            npz_kwargs[f"{tag}_n_prop"] = np.array([d["agg"]["n_prop_total"]])
            npz_kwargs[f"{tag}_n_nonprop"] = np.array([d["agg"]["n_nonprop_total"]])
            npz_kwargs[f"{tag}_tau_high"] = np.array([d["agg"]["tau_high_mean"]])
            npz_kwargs[f"{tag}_tau_low"] = np.array([d["agg"]["tau_low_mean"]])
            npz_kwargs[f"{tag}_distinct"] = d["dist"]
    npz_kwargs["pctls"] = np.array(pctls)
    np.savez_compressed(out_dir / "encoder_space_profiles_by_pctl.npz",
                        **npz_kwargs)
    print(f"wrote {out_dir / 'encoder_space_profiles_by_pctl.npz'}")

    # Use the strictest percentile (top 5%) for the main figures + bridge
    # (matches the original headline analysis).
    headline_pctl = max(pctls)
    per_modality = per_pctl.get(headline_pctl, {})
    if per_modality:
        plot_profiles(per_modality, out_dir / "encoder_space_profiles.png")
        plot_topk(per_modality, out_dir / "encoder_space_topk.png")

    # Cross-percentile tracking — how do key dims evolve as we loosen P_prop?
    print("\n" + "=" * 90)
    print("CROSS-PERCENTILE TRACKING  "
          "(prop_med >= 100 dims AND top-distinct dim per modality, per pctl)")
    print("=" * 90)
    track_rows: list = []
    for m in ("audio", "video"):
        print(f"\n  {m}: count of dims with prop_median >= 100 at each pctl:")
        print(f"    {'pctl':<7}{'n_prop':<10}{'n>=100':<10}"
              f"{'top dim':<10}{'distinct':<10}{'prop_med':<10}"
              f"{'nonprop_med':<13}{'active%'}")
        for p in pctls:
            if m not in per_pctl[p]:
                continue
            d = per_pctl[p][m]
            agg = d["agg"]; dist = d["dist"]
            pm = agg["prop_median"]; nm = agg["nonprop_median"]
            af = agg["prop_active_frac"]
            n_ge100 = int((pm >= 100).sum())
            top_dim = int(np.nanargmax(dist))
            print(f"    {p:<7}{agg['n_prop_total']:<10}{n_ge100:<10}"
                  f"{top_dim:<10}{dist[top_dim]:<10.2f}{pm[top_dim]:<10.2f}"
                  f"{nm[top_dim]:<13.2f}{af[top_dim]*100:.1f}")
            track_rows.append(dict(
                modality=m, prop_pctl=p, n_prop=int(agg["n_prop_total"]),
                n_dims_prop_med_ge_100=n_ge100,
                top_dim=top_dim, top_distinct=float(dist[top_dim]),
                top_prop_median=float(pm[top_dim]),
                top_nonprop_median=float(nm[top_dim]),
                top_active_frac=float(af[top_dim])))
        # Also: track the headline dims explicitly across pctls.
        key_dim = 849 if m == "video" else 0
        print(f"\n  {m}: tracking dim {key_dim} across pctls:")
        print(f"    {'pctl':<7}{'distinct':<10}{'prop_med':<10}"
              f"{'nonprop_med':<13}{'active%'}")
        for p in pctls:
            if m not in per_pctl[p]:
                continue
            d = per_pctl[p][m]
            pm = d["agg"]["prop_median"][key_dim]
            nm = d["agg"]["nonprop_median"][key_dim]
            af = d["agg"]["prop_active_frac"][key_dim] * 100
            dv = d["dist"][key_dim]
            print(f"    {p:<7}{dv:<10.2f}{pm:<10.2f}{nm:<13.2f}{af:.1f}")
    pd.DataFrame(track_rows).to_csv(
        out_dir / "encoder_space_pctl_tracking.csv", index=False)
    print(f"\nwrote {out_dir / 'encoder_space_pctl_tracking.csv'}")

    # Optional projector bridge — only meaningful for space='pre' (it maps
    # the pre-projection encoder dim to post-projection LLM-dim space).
    bridge_lines = []
    if args.space == "pre" and "audio" in per_modality:
        dist = per_modality["audio"]["dist"]
        if np.isfinite(dist).any():
            top_d = int(np.nanargmax(dist))
            bridge_info = audio_projector_bridge(model, top_d)
            if bridge_info:
                bridge_lines.append(
                    f"audio (p={headline_pctl}) top pre-proj dim = {top_d}; "
                    f"projector mass in top 10 LLM dims = "
                    f"{bridge_info['frac_top_mass']*100:.1f}% "
                    f"({'concentrated' if bridge_info['frac_top_mass']>0.5 else 'spread'})")
    elif args.space == "post":
        bridge_lines.append(
            "projector bridge skipped: space='post' is already in LLM-dim "
            "space (the projector's output side).")

    # Verdict.
    print("\n" + "=" * 90)
    print("STAGE 1.3 ENCODER-SPACE VERDICT")
    print("=" * 90)
    verdict_lines = []
    for m in ("audio", "video"):
        if m not in per_modality:
            continue
        n_d = int(per_modality[m]["is_distinct"].sum())
        dist = per_modality[m]["dist"]
        # Top distinct dim regardless of gate (informative even if it fails).
        top_d = int(np.nanargmax(dist))
        top_score = float(dist[top_d])
        top_pm = float(per_modality[m]["agg"]["prop_median"][top_d])
        top_nm = float(per_modality[m]["agg"]["nonprop_median"][top_d])
        top_af = float(per_modality[m]["agg"]["prop_active_frac"][top_d]) * 100
        if n_d == 0:
            s = (f"  {m} pre-proj: NO dim passes gate — top candidate dim {top_d}: "
                 f"distinct={top_score:.2f}, prop_med={top_pm:.2f}, "
                 f"nonprop_med={top_nm:.2f}, active_frac={top_af:.1f}%.")
        else:
            top_list = sorted(int(d) for d in np.where(per_modality[m]["is_distinct"])[0])[:5]
            s = (f"  {m} pre-proj: {n_d} distinctive dim(s); top by distinctiveness "
                 f"= dim {top_d} (distinct={top_score:.2f}, prop_med={top_pm:.2f}, "
                 f"nonprop_med={top_nm:.2f}, active_frac={top_af:.1f}%). "
                 f"First 5 distinctive dims by index: {top_list}.")
        print(s); verdict_lines.append(s)
    for ln in bridge_lines:
        print(f"  {ln}"); verdict_lines.append(ln)

    # Overall conclusion.
    a_n = int(per_modality["audio"]["is_distinct"].sum()) if "audio" in per_modality else 0
    v_n = int(per_modality["video"]["is_distinct"].sum()) if "video" in per_modality else 0
    if a_n == 0 and v_n == 0:
        concl = ("\n  → REGISTER ABSENT IN BOTH SPACES: the propagated population "
                 "is dimensionless in PRE-projection encoder space too. Stage 1.3 "
                 "result strengthens — Qwen2.5-Omni's encoder-propagated sinks "
                 "lack a dedicated register dimension at every level "
                 "(pre-projection encoder space AND LLM hidden state).")
    else:
        concl = ("\n  → ENCODER-SPACE REGISTER EXISTS: distinctive pre-projection "
                 "dim(s) found. The Stage 1.3 'no register' finding was confined "
                 "to LLM hidden-state space — the projector may smear an encoder-"
                 "side register when mapping to LLM dim. See bridge bullet above.")
    print(concl); verdict_lines.append(concl)

    with open(out_dir / "stage1_3_encoder_space_decision.txt", "w") as f:
        f.write("Stage 1.3 (encoder-space refinement) — "
                "pre-projection distinctiveness\n")
        f.write(f"per-clip top {[100-p for p in pctls]}% = P_prop_pre, "
                f"bottom {NONPROP_PCTL:.0f}% = P_nonprop_pre\n")
        f.write(f"distinctive := distinct > {DISTINCT_THRESHOLD} AND "
                f"active_frac > {ACTIVE_FRAC_MIN*100:.0f}% "
                f"(token-active threshold = {DEFAULT_TOKEN_ACTIVE_THRESHOLD})\n\n")
        for ln in verdict_lines:
            f.write(ln + "\n")
    print(f"\nwrote {out_dir / 'stage1_3_encoder_space_decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",
                   default=str(_REPO / "data/VGGSounder/videos"))
    p.add_argument("--n_clips", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_map", default="balanced_low_0",
                   help="balanced_low_0 only — auto crashes on this model.")
    p.add_argument("--prop_pctls", type=float, nargs="+",
                   default=DEFAULT_PROP_PCTLS,
                   help="P_prop percentile thresholds to sweep (e.g., 95 90 80 50 "
                        "= top 5%%/10%%/20%%/50%%). All run in one model pass.")
    p.add_argument("--space", choices=("pre", "post"), default="pre",
                   help="Feature space: 'pre' = pre-projection encoder "
                        "(audio_tower.avg_pooler 2D; visual.blocks[-1] 1280-dim). "
                        "'post' = post-projection encoder OUTPUT "
                        "(audio_tower / visual modules, both 3584-dim — the "
                        "features fed into the LLM as modal tokens).")
    p.add_argument("--output_dir", default=None,
                   help="Default: stage1_3_encoder_space/ (pre) or "
                        "stage1_3_encoder_space_post/ (post).")
    args = p.parse_args()
    main(args)
