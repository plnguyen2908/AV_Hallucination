"""
stage2_6_logit_lens_audio_sink.py

Stage 2.6 — Per-layer logit-lens probe: do audio LLM-emerged sink tokens
decode to vocab content LESS similar to the clip's GT label than non-sink
audio tokens do?

This is the direct *semantic* version of the content-blindness Stage 2.4
(temporal uniformity) and Stage 2.5 (acoustic correlation = 0.16 R²) showed
only indirectly.

Conventions (exactly match the existing pipeline):
  - Sink criterion is P_llm (LLM-emerged), the same as Stages 2.1 / 2.2 / 2.4:
        pure RMSNorm (NO learned weight) + D_sink={458, 2570} + τ=20.
    Pure RMSNorm = sqrt(mean(x²) + eps); the learned-weight rescale is
    deliberately omitted, exactly as in stage2_1_layer_sink_counts.py.
    This is NOT the propagated/encoder-norm population.
  - Logit lens uses the model's REAL final norm + lm_head (learned-weight
    RMSNorm followed by the unembedding). The norm here is a DIFFERENT
    operation from the sink-detection norm above; they're not interchangeable.
  - Similarity is computed in the model's OWN input-embedding space:
        embed_tokens[idx]  (decoded vocab tokens)
        embed_tokens[label subwords].mean(0)  (label string, mean-pooled)
        L2-normalize both, cosine.
    No external sentence encoder.
  - Two metrics per (clip, layer, token):
        primary  (weighted): top-10 cosine, weighted by probabilities
                              renormalized to sum to 1 over those 10.
        strict   (top-1)   : cosine of the argmax decoded token.
  - Per layer, never averaged across layers. Per-layer CSV reports n_sink
    and n_nonsink.
  - Audio span only; first audio position (pos-0 BOS register, Stage 2.4)
    is dropped by default; --keep_pos0 to override.

How to read the result (interpretation goes in printed output too):
  - The decision quantity is the GAP = mean_sim(non-sink) − mean_sim(sink).
    The absolute lines rise toward the final layers for both groups just
    because the logit lens becomes vocab-aligned late; that's not signal.
  - Read the gap in the MID-STACK (~L10–L22) where n_nonsink is still
    healthy. Late layers are saturation-confounded: by L21 the audio span
    is ~74% sink, by L25 ~96% — so the non-sink comparison group collapses
    and the gap there is uninterpretable. Early layers have ~0 sinks.
  - Layers where mean n_nonsink/clip < MIN_NONSINK_FOR_FLAG are visually
    shaded as saturation-confounded.
  - KILL CONDITION: on the 50-clip smoke, if sink ≈ non-sink across the
    unshaded region, the logit-lens is uninformative for audio tokens
    regardless of sink status and the experiment is NEGATIVE — surface
    that result rather than scaling.

Inputs:
  - data/AudioSet_describe/audios/*.wav (500 clips with multi-label GT in QA.json)
    + data/AudioSet_describe/QA.json (label list per clip)
  - Stage 2.4-describe's clip selection is reproduced via seed=42.

Outputs (--output_dir):
  stage2_6_per_clip_layer.csv     long-form: clip, label, layer, n_sink, n_nonsink,
                                  sink_/nonsink_weighted_mean, sink_/nonsink_top1_mean
  stage2_6_per_layer.csv          per-layer aggregate (weighted by n_*)
  stage2_6_logit_lens_gap.png     two-panel line graph (weighted, top-1)
  stage2_6_decision.txt           one-line verdict for the mid-stack gap
"""

import argparse
import json
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

# --- constants ----------------------------------------------------------------
D_SINK = [458, 2570]              # Stage 2.1/2.4 sink dims
TAU_SINK = 20.0                   # Stage 2.1/2.4 threshold
PROMPT_AUDIO = "Describe what you hear in detail."  # matches Stage 2.4
TOP_K = 10
MID_LO, MID_HI = 10, 22           # mid-stack window for the verdict / kill check
MIN_NONSINK_FOR_FLAG = 30         # below this mean count / clip → shade as confound
SEED = 42

DEFAULT_AUDIO_DIR = _REPO / "data/AudioSet_describe/audios"
DEFAULT_QA = _REPO / "data/AudioSet_describe/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_6_logit_lens"


# --- helpers ------------------------------------------------------------------

def _audio_positions(input_ids, thinker_cfg):
    """Same as Stage 2.4's _audio_positions: indices of audio tokens in the
    LLM seq via audio_token_index match."""
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    return np.where(ids == a_id)[0].astype(np.int64)


def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_token_index"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _thinker_rms_eps(model) -> float:
    cfg = getattr(model.thinker.config, "text_config", model.thinker.config)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def embed_label_string(label_str, processor, embed_layer):
    """Tokenize + mean-pool subword input embeddings of the GT label.
    Returns a (hidden,) tensor on embed_layer.weight.device, unnormalized."""
    ids = processor.tokenizer(label_str, return_tensors="pt",
                              add_special_tokens=False).input_ids
    if ids.numel() == 0:
        return None
    ids = ids[0].to(embed_layer.weight.device)
    with torch.no_grad():
        emb = embed_layer(ids).float()            # (n_sub, hidden)
    return emb.mean(dim=0)                        # (hidden,)


# --- per-clip processing ------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers,
                 label_emb, d_sink_t, eps_norm, skip_pos0=True):
    """For one clip: hook each layer's output, then per layer compute
        - is_sink[t]  via pure RMSNorm + D_sink + τ
        - logits[t]   via the REAL final_norm + lm_head
        - top-K decoded indices/probs (probs renormalized over the K)
        - cosine sims to the clip's label embedding (in input-embed space)
        - per-token weighted (over top-K) and top-1 metrics
    Returns a list of per-layer dicts (or None on failure)."""
    conv = build_conversation(str(clip_path), PROMPT_AUDIO, "a")
    inputs, use_aiv = prepare_inputs(processor, conv, "a", model.device, model.dtype)

    audio_pos = _audio_positions(inputs["input_ids"], thinker_cfg)
    if len(audio_pos) == 0:
        return None
    if skip_pos0 and len(audio_pos) > 1:
        audio_pos = audio_pos[1:]
    a_pos_t = torch.from_numpy(audio_pos)

    n_layers = len(layers)
    per_layer_h = [None] * n_layers

    def make_hook(L_idx):
        def _h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out  # (1, seq, H)
            if hs.shape[1] <= 1:
                return out                                  # generation step (none here)
            ap = a_pos_t.to(hs.device)
            # detach + keep on layer's device until use
            per_layer_h[L_idx] = hs[0, ap].detach()         # (n_audio, hidden)
            return out
        return _h

    handles = [layers[L].register_forward_hook(make_hook(L))
               for L in range(n_layers)]
    try:
        with torch.inference_mode():
            model.thinker(**inputs, output_hidden_states=False,
                          use_audio_in_video=use_aiv, return_dict=True,
                          use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if any(h is None for h in per_layer_h):
        return None

    final_norm = model.thinker.model.norm        # learned-weight RMSNorm
    lm_head = model.thinker.lm_head              # unembedding
    embed_layer = model.thinker.model.embed_tokens
    fn_dev = final_norm.weight.device
    emb_dev = embed_layer.weight.device

    # Pre-normalize the label embedding on the embed device
    label_norm = torch.nn.functional.normalize(
        label_emb.to(emb_dev), dim=-1)            # (hidden,)

    rows = []
    for L in range(n_layers):
        h = per_layer_h[L]                                  # (n_audio, hidden)
        # --- sink detection: pure RMSNorm + D_sink, τ ---
        h_f = h.float()
        rms = torch.sqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_f / rms).abs()
        d_t = d_sink_t.to(h.device)
        is_sink = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
        n_sink = int(is_sink.sum())
        n_nonsink = int((~is_sink).sum())

        # --- logit lens: REAL final_norm + lm_head ---
        h_fn = h.to(fn_dev)
        with torch.no_grad():
            h_normed = final_norm(h_fn)                     # learned-weight RMSNorm
            logits = lm_head(h_normed).float()              # (n_audio, vocab)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_idx = probs.topk(TOP_K, dim=-1)  # (n_audio, K)
            # renormalize the kept K to sum to 1
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            # --- embed decoded tokens via INPUT embedding matrix ---
            topk_idx_e = topk_idx.to(emb_dev)
            dec_emb = embed_layer(topk_idx_e).float()       # (n_audio, K, hidden)
            dec_norm = torch.nn.functional.normalize(dec_emb, dim=-1)

            # cosine = dot of L2-normalized vectors
            cos = (dec_norm * label_norm[None, None, :]).sum(dim=-1)  # (n_audio, K)
            tp_e = topk_probs.to(emb_dev)
            weighted = (cos * tp_e).sum(dim=-1).cpu().numpy()         # (n_audio,)
            top1 = cos[:, 0].cpu().numpy()                            # (n_audio,)

        rows.append(dict(
            layer=L, n_sink=n_sink, n_nonsink=n_nonsink,
            sink_weighted_mean=(float(weighted[is_sink].mean())
                                 if n_sink > 0 else float("nan")),
            nonsink_weighted_mean=(float(weighted[~is_sink].mean())
                                    if n_nonsink > 0 else float("nan")),
            sink_top1_mean=(float(top1[is_sink].mean())
                             if n_sink > 0 else float("nan")),
            nonsink_top1_mean=(float(top1[~is_sink].mean())
                                if n_nonsink > 0 else float("nan")),
        ))
        # release the layer's hidden state ASAP
        per_layer_h[L] = None
    return rows


# --- aggregation + plotting ---------------------------------------------------

def aggregate_per_layer(per_clip_layer_df):
    """Per-layer aggregate. Weights each clip's per-layer mean by that clip's
    n_sink / n_nonsink so clips with no sinks at a given layer contribute
    nothing to the sink mean there (and vice versa)."""

    def _wmean(group, val_col, w_col):
        v = group[val_col].values
        w = group[w_col].values
        mask = np.isfinite(v) & (w > 0)
        if not mask.any():
            return float("nan")
        return float((v[mask] * w[mask]).sum() / w[mask].sum())

    def _agg(g):
        return pd.Series(dict(
            n_clips=int(len(g)),
            mean_n_sink=float(g["n_sink"].mean()),
            mean_n_nonsink=float(g["n_nonsink"].mean()),
            total_n_sink=int(g["n_sink"].sum()),
            total_n_nonsink=int(g["n_nonsink"].sum()),
            sink_weighted_mean=_wmean(g, "sink_weighted_mean", "n_sink"),
            nonsink_weighted_mean=_wmean(g, "nonsink_weighted_mean", "n_nonsink"),
            sink_top1_mean=_wmean(g, "sink_top1_mean", "n_sink"),
            nonsink_top1_mean=_wmean(g, "nonsink_top1_mean", "n_nonsink"),
        ))

    agg = per_clip_layer_df.groupby("layer").apply(_agg).reset_index()
    agg["gap_weighted"] = agg["nonsink_weighted_mean"] - agg["sink_weighted_mean"]
    agg["gap_top1"] = agg["nonsink_top1_mean"] - agg["sink_top1_mean"]
    return agg


def plot_two_panel(agg, out_path, low_n_thr=MIN_NONSINK_FOR_FLAG):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    layers = agg["layer"].values
    low_n_layers = set(agg.loc[agg["mean_n_nonsink"] < low_n_thr, "layer"].tolist())

    for ax, label, sink_col, nonsink_col, gap_col in (
        (ax1, "weighted  (prob-renorm over top-10)",
         "sink_weighted_mean", "nonsink_weighted_mean", "gap_weighted"),
        (ax2, "strict  (top-1)",
         "sink_top1_mean", "nonsink_top1_mean", "gap_top1"),
    ):
        s = agg[sink_col].values; ns = agg[nonsink_col].values
        ax.plot(layers, s, marker="o", lw=2, color="#d62728", label="sink")
        ax.plot(layers, ns, marker="o", lw=2, color="#1f77b4", label="non-sink")
        # Shade saturation-confounded layers
        for L in low_n_layers:
            ax.axvspan(L - 0.5, L + 0.5, color="gray", alpha=0.18, zorder=0)
        # Mid-stack reference band
        ax.axvspan(MID_LO - 0.5, MID_HI + 0.5, fill=False, edgecolor="#2ca02c",
                    linestyle=":", linewidth=1.0, zorder=0)
        ax.set_ylabel(f"mean cosine(decoded, GT label)\n{label}", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper left", fontsize=9)

        ax_g = ax.twinx()
        ax_g.plot(layers, agg[gap_col].values, lw=1.5, ls="--",
                  color="#2ca02c", marker="x", ms=4,
                  label="gap (non-sink − sink)")
        ax_g.axhline(0, color="gray", ls=":", alpha=0.6)
        ax_g.set_ylabel("gap (non-sink − sink)", color="#2ca02c", fontsize=10)
        ax_g.tick_params(axis="y", labelcolor="#2ca02c")
        ax_g.legend(loc="upper right", fontsize=9)

    ax2.set_xlabel("LLM decoder layer L", fontsize=11)
    fig.suptitle(
        f"Stage 2.6 — audio logit-lens semantic gap (cosine to GT label)\n"
        f"grey = mean n_nonsink/clip < {low_n_thr} (saturation-confounded, ignore); "
        f"green box = mid-stack read window L{MID_LO}-L{MID_HI}",
        fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_decision(agg, out_path, n_clips, skip_pos0):
    """One-line verdict + key numbers in the same style as other Stage decisions."""
    mid = agg[(agg["layer"] >= MID_LO) & (agg["layer"] <= MID_HI)]
    g_w_mean = float(mid["gap_weighted"].mean())
    g_w_med = float(mid["gap_weighted"].median())
    g_w_pos = int((mid["gap_weighted"] > 0).sum())
    g_t1_mean = float(mid["gap_top1"].mean())
    g_t1_med = float(mid["gap_top1"].median())
    g_t1_pos = int((mid["gap_top1"] > 0).sum())
    n_layers_mid = int(len(mid))
    safe_layers = int((mid["mean_n_nonsink"] >= MIN_NONSINK_FOR_FLAG).sum())

    # Verdict cutoffs (these are deliberate, not arbitrary):
    #   gap > 0.01 and ≥ 70% of mid-stack layers positive → CONTENT-BLIND
    #   |gap| < 0.005 and direction noisy                 → UNINFORMATIVE (kill)
    #   otherwise                                          → INCONCLUSIVE
    GAP_THR = 0.01
    POS_FRAC_THR = 0.70
    NEUTRAL_THR = 0.005

    def _classify(mean, pos):
        frac = pos / max(n_layers_mid, 1)
        if mean > GAP_THR and frac >= POS_FRAC_THR:
            return "CONTENT-BLIND"
        if abs(mean) < NEUTRAL_THR:
            return "UNINFORMATIVE"
        return "INCONCLUSIVE"

    v_w = _classify(g_w_mean, g_w_pos)
    v_t1 = _classify(g_t1_mean, g_t1_pos)
    combined = v_w if v_w == v_t1 else f"weighted={v_w} / top1={v_t1}"

    with open(out_path, "w") as f:
        f.write(f"Stage 2.6 — audio LLM-emerged sink logit-lens probe  "
                f"(n_clips={n_clips}, D_sink={D_SINK}, τ={TAU_SINK}, "
                f"skip_pos0={skip_pos0}, top_K={TOP_K})\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"Mid-stack window: L{MID_LO}-L{MID_HI}  "
                f"({n_layers_mid} layers, "
                f"{safe_layers} with mean n_nonsink/clip ≥ {MIN_NONSINK_FOR_FLAG})\n\n")
        f.write(f"Primary (weighted, top-{TOP_K}):\n")
        f.write(f"  gap (non-sink − sink) mean   = {g_w_mean:+.4f}\n")
        f.write(f"  gap median                    = {g_w_med:+.4f}\n")
        f.write(f"  layers with gap > 0           = {g_w_pos}/{n_layers_mid}\n\n")
        f.write(f"Strict (top-1):\n")
        f.write(f"  gap mean                      = {g_t1_mean:+.4f}\n")
        f.write(f"  gap median                    = {g_t1_med:+.4f}\n")
        f.write(f"  layers with gap > 0           = {g_t1_pos}/{n_layers_mid}\n\n")
        f.write(f"VERDICT: {combined}\n")
        f.write(f"  CONTENT-BLIND : gap > {GAP_THR} and ≥{POS_FRAC_THR*100:.0f}% of "
                f"mid layers positive (sinks decode to vocab less related to GT label).\n")
        f.write(f"  UNINFORMATIVE : |gap| < {NEUTRAL_THR} (kill — logit-lens "
                f"doesn't differentiate audio token populations).\n")
        f.write(f"  INCONCLUSIVE  : otherwise (mixed direction or small effect).\n")
    print(f"wrote {out_path}")
    return combined, g_w_mean, g_w_med, g_w_pos, g_t1_mean, g_t1_med, g_t1_pos


# --- main ---------------------------------------------------------------------

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
    embed_layer = model.thinker.model.embed_tokens
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ={TAU_SINK}, eps={eps_norm}, "
          f"skip_pos0={args.skip_pos0}, top_K={TOP_K}")

    # Labels
    qa = json.load(open(args.qa))
    labels_by_vid = {q["video_id"]: list(q.get("label", [])) for q in qa}

    # Reproduce Stage 2.4-describe clip order
    audio_dir = Path(args.audio_dir)
    clips_all = sorted(audio_dir.glob("*.wav"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed}); "
          f"audio_dir={audio_dir.name}\n")

    all_rows = []
    failures: dict = {}
    for clip in tqdm(clips, desc="clips"):
        vid = clip.name
        labs = labels_by_vid.get(vid, [])
        if not labs:
            failures["no_label"] = failures.get("no_label", 0) + 1
            continue
        label_str = ", ".join(labs)
        try:
            label_emb = embed_label_string(label_str, processor, embed_layer)
            if label_emb is None:
                failures["empty_label"] = failures.get("empty_label", 0) + 1
                continue
            rows = process_clip(model, processor, clip, thinker_cfg, layers,
                                 label_emb, d_sink_t, eps_norm,
                                 skip_pos0=args.skip_pos0)
        except Exception as e:
            failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
            tqdm.write(f"  [skip] {vid}: {type(e).__name__}: {e}")
            continue
        if rows is None:
            failures["no_audio"] = failures.get("no_audio", 0) + 1
            continue
        for r in rows:
            r["clip"] = vid; r["label"] = label_str
        all_rows.extend(rows)
        torch.cuda.empty_cache()

    if failures:
        print(f"  failures: {failures}")
    if not all_rows:
        raise SystemExit("no clips processed")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "stage2_6_per_clip_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage2_6_per_clip_layer.csv'}  "
          f"({len(df)} (clip, layer) records)")

    agg = aggregate_per_layer(df)
    agg.to_csv(out_dir / "stage2_6_per_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage2_6_per_layer.csv'}")

    plot_two_panel(agg, out_dir / "stage2_6_logit_lens_gap.png")

    combined, gw_mean, gw_med, gw_pos, gt_mean, gt_med, gt_pos = write_decision(
        agg, out_dir / "stage2_6_decision.txt",
        n_clips=df["clip"].nunique(), skip_pos0=args.skip_pos0)

    # Console interpretation block
    print()
    print("=" * 80)
    print(f"STAGE 2.6 — mid-stack gap (L{MID_LO}-L{MID_HI})  "
          f"(n_clips = {df['clip'].nunique()})")
    print("=" * 80)
    mid = agg[(agg["layer"] >= MID_LO) & (agg["layer"] <= MID_HI)]
    print(f"  mean n_sink/clip across L{MID_LO}-L{MID_HI}    = "
          f"{mid['mean_n_sink'].mean():.1f}")
    print(f"  mean n_nonsink/clip across L{MID_LO}-L{MID_HI} = "
          f"{mid['mean_n_nonsink'].mean():.1f}  "
          f"({(mid['mean_n_nonsink'] >= MIN_NONSINK_FOR_FLAG).sum()} of "
          f"{len(mid)} layers above the {MIN_NONSINK_FOR_FLAG}-token confound floor)")
    print(f"  weighted: gap mean = {gw_mean:+.4f}, median = {gw_med:+.4f},  "
          f"{gw_pos}/{len(mid)} layers > 0")
    print(f"  top-1   : gap mean = {gt_mean:+.4f}, median = {gt_med:+.4f},  "
          f"{gt_pos}/{len(mid)} layers > 0")
    print()
    print(f"  VERDICT: {combined}")
    print(f"    CONTENT-BLIND  → gap > 0.01 with ≥70% mid layers positive")
    print(f"    UNINFORMATIVE  → |gap| < 0.005 (kill condition)")
    print(f"    INCONCLUSIVE   → mixed direction or small effect")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--qa", default=str(DEFAULT_QA))
    p.add_argument("--n_clips", type=int, default=50,
                   help="Default 50 = smoke / kill-check pass.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_pos0", action="store_true", default=True,
                   help="Drop the first audio token (pos-0 BOS register).")
    p.add_argument("--keep_pos0", dest="skip_pos0", action="store_false")
    p.add_argument("--device_map", default="balanced_low_0",
                   help="balanced_low_0 — auto crashes Qwen2.5-Omni.")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
