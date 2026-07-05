"""
stage2_6b_logit_lens_lm_head_space.py

Stage 2.6b — Stage 2.6 re-run with the cosine geometry corrected to match the
lens's actual projection space.

Diagnostic finding: Qwen2.5-Omni has tie_word_embeddings = False, and the
row-wise cosine between embed_tokens[v] and lm_head[v] is essentially zero
across the full 152,064-vocab (mean +0.0013, no token above 0.5, 47% negative).
The two matrices live in essentially orthogonal directions of the 3584-d
hidden space.

Stage 2.6 (original) embedded both the decoded vocab token and the GT label
via embed_tokens (input space), while the lens picks tokens by lm_head proximity
(output space). The input-space cosine therefore measured a quantity nearly
orthogonal to what the lens was actually doing — and came back null.

This stage repeats Stage 2.6 with all embeddings via lm_head, so the cosine
lives in the SAME space the lens projects into.

Conventions (unchanged from Stage 2.6 except for the embedding space):
  - Sink criterion is P_llm: pure RMSNorm (no learned weight) + D_sink={458, 2570}
    + τ=20. Same as Stages 2.1 / 2.4 / 2.6 / 2.7.
  - Logit lens uses the model's REAL final RMSNorm (LEARNED weight) + lm_head.
    h_normed = final_norm(h)  is applied BEFORE lm_head; lm_head is then
    applied to the NORMALIZED hidden state — same applies wherever we map
    through lm_head, both for the lens projection and (implicitly) when
    reading the lm_head row corresponding to a vocab id.
  - Audio span only; pos-0 dropped by default.

What changes (vs Stage 2.6):
    decoded-token embedding:   lm_head.weight[topk_idx]   (output space)
    GT-label embedding:        lm_head.weight[label_subword_ids].mean(0)
                                                          (output space)
  L2-normalize both, cosine. Top-10 metric weighted by renormalized top-10
  probabilities. Top-1 metric unchanged.

Same outputs as Stage 2.6, in stage2_6b_logit_lens_lm_head_space/.
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

D_SINK = [458, 2570]
TAU_SINK = 20.0
PROMPT_AUDIO = "Describe what you hear in detail."
TOP_K = 10
MID_LO, MID_HI = 10, 22
MIN_NONSINK_FOR_FLAG = 30
SEED = 42

DEFAULT_AUDIO_DIR = _REPO / "data/AudioSet_describe/audios"
DEFAULT_QA = _REPO / "data/AudioSet_describe/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_6b_logit_lens_lm_head_space"


# --- helpers ------------------------------------------------------------------

def _audio_positions(input_ids, thinker_cfg):
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


def embed_label_lm_head(label_str, tokenizer, lm_head_weight):
    """Tokenize the GT label and mean-pool LM-HEAD rows (output-space embedding).
    Returns a (hidden,) tensor on lm_head_weight.device, unnormalized."""
    ids = tokenizer(label_str, return_tensors="pt",
                    add_special_tokens=False).input_ids
    if ids.numel() == 0:
        return None
    ids = ids[0].to(lm_head_weight.device)
    return lm_head_weight[ids].float().mean(dim=0)         # (hidden,)


# --- per-clip processing ------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers,
                 label_emb_lmhead, d_sink_t, eps_norm, skip_pos0=True):
    """Same per-clip pipeline as Stage 2.6 but:
      - decoded-token embeddings come from lm_head rows
      - label embedding is also a mean-pool of lm_head rows
      - the lens itself is unchanged: final_norm(h) → lm_head → softmax → top-K
    NB: 'normalize before decode' — final_norm is applied to h BEFORE lm_head
    in the lens path, exactly as the model's own forward computes its logits.
    """
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
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] <= 1:
                return out
            ap = a_pos_t.to(hs.device)
            per_layer_h[L_idx] = hs[0, ap].detach()
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

    final_norm = model.thinker.model.norm
    lm_head = model.thinker.lm_head
    fn_dev = final_norm.weight.device

    # Pre-normalize the LM-head-space label on lm_head's device.
    label_norm = torch.nn.functional.normalize(
        label_emb_lmhead.to(fn_dev), dim=-1)         # (hidden,)

    rows = []
    for L in range(n_layers):
        h = per_layer_h[L]
        # Sink detection: pure RMSNorm + D_sink (UNCHANGED from 2.6).
        h_f = h.float()
        rms = torch.sqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_f / rms).abs()
        d_t = d_sink_t.to(h.device)
        is_sink = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
        n_sink = int(is_sink.sum()); n_nonsink = int((~is_sink).sum())

        # Logit lens: NORMALIZE BEFORE DECODE — final_norm THEN lm_head.
        h_fn = h.to(fn_dev)
        with torch.no_grad():
            h_normed = final_norm(h_fn)
            logits = lm_head(h_normed).float()                  # (n_audio, vocab)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_idx = probs.topk(TOP_K, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            # Look up the decoded tokens in lm_head (OUTPUT space) — not embed_tokens.
            dec_emb = lm_head.weight[topk_idx].float()           # (n_audio, K, hidden)
            dec_norm = torch.nn.functional.normalize(dec_emb, dim=-1)
            cos = (dec_norm * label_norm[None, None, :]).sum(dim=-1)   # (n_audio, K)
            weighted = (cos * topk_probs).sum(dim=-1).cpu().numpy()
            top1 = cos[:, 0].cpu().numpy()

        rows.append(dict(
            layer=L, n_sink=n_sink, n_nonsink=n_nonsink,
            sink_weighted_mean=(float(weighted[is_sink].mean()) if n_sink else float("nan")),
            nonsink_weighted_mean=(float(weighted[~is_sink].mean()) if n_nonsink else float("nan")),
            sink_top1_mean=(float(top1[is_sink].mean()) if n_sink else float("nan")),
            nonsink_top1_mean=(float(top1[~is_sink].mean()) if n_nonsink else float("nan")),
        ))
        per_layer_h[L] = None
    return rows


def aggregate_per_layer(df):
    def _wmean(g, val, w):
        v = g[val].values; ww = g[w].values
        mask = np.isfinite(v) & (ww > 0)
        if not mask.any(): return float("nan")
        return float((v[mask] * ww[mask]).sum() / ww[mask].sum())

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

    agg = df.groupby("layer").apply(_agg).reset_index()
    agg["gap_weighted"] = agg["nonsink_weighted_mean"] - agg["sink_weighted_mean"]
    agg["gap_top1"]     = agg["nonsink_top1_mean"]     - agg["sink_top1_mean"]
    return agg


def plot_two_panel(agg, out_path, low_n_thr=MIN_NONSINK_FOR_FLAG):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    layers = agg["layer"].values
    low_n = set(agg.loc[agg["mean_n_nonsink"] < low_n_thr, "layer"].tolist())
    for ax, label, sink_col, non_col, gap_col in (
        (ax1, "weighted  (prob-renorm over top-10)",
         "sink_weighted_mean", "nonsink_weighted_mean", "gap_weighted"),
        (ax2, "strict  (top-1)",
         "sink_top1_mean", "nonsink_top1_mean", "gap_top1"),
    ):
        s = agg[sink_col].values; ns = agg[non_col].values
        ax.plot(layers, s, marker="o", lw=2, color="#d62728", label="sink")
        ax.plot(layers, ns, marker="o", lw=2, color="#1f77b4", label="non-sink")
        for L in low_n:
            ax.axvspan(L - 0.5, L + 0.5, color="gray", alpha=0.18, zorder=0)
        ax.axvspan(MID_LO - 0.5, MID_HI + 0.5, fill=False, edgecolor="#2ca02c",
                    linestyle=":", linewidth=1.0, zorder=0)
        ax.set_ylabel(f"mean cosine(decoded, GT label)\n{label}\n"
                       "(lm_head OUTPUT space)", fontsize=10)
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
    fig.suptitle("Stage 2.6b — audio logit-lens semantic gap, in LM_HEAD space\n"
                 f"grey = mean n_nonsink/clip < {low_n_thr}; "
                 f"green box = mid-stack L{MID_LO}-L{MID_HI}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_decision(agg, out_path, n_clips, skip_pos0):
    mid = agg[(agg["layer"] >= MID_LO) & (agg["layer"] <= MID_HI)]
    g_w_mean = float(mid["gap_weighted"].mean())
    g_w_med  = float(mid["gap_weighted"].median())
    g_w_pos  = int((mid["gap_weighted"] > 0).sum())
    g_t1_mean = float(mid["gap_top1"].mean())
    g_t1_med  = float(mid["gap_top1"].median())
    g_t1_pos  = int((mid["gap_top1"] > 0).sum())
    n_layers_mid = int(len(mid))
    safe_layers  = int((mid["mean_n_nonsink"] >= MIN_NONSINK_FOR_FLAG).sum())

    GAP_THR = 0.01; POS_FRAC_THR = 0.70; NEUTRAL_THR = 0.005

    def _classify(mean, pos):
        frac = pos / max(n_layers_mid, 1)
        if mean > GAP_THR and frac >= POS_FRAC_THR: return "CONTENT-BLIND"
        if abs(mean) < NEUTRAL_THR: return "UNINFORMATIVE"
        return "INCONCLUSIVE"

    v_w = _classify(g_w_mean, g_w_pos)
    v_t1 = _classify(g_t1_mean, g_t1_pos)
    combined = v_w if v_w == v_t1 else f"weighted={v_w} / top1={v_t1}"

    with open(out_path, "w") as f:
        f.write("Stage 2.6b — audio LLM-emerged sink logit-lens probe, "
                "LM-HEAD output-space cosine\n")
        f.write("=" * 95 + "\n\n")
        f.write("Geometry note: lm_head and embed_tokens are UNTIED in "
                "Qwen2.5-Omni; their rows are\nessentially orthogonal "
                "(mean row-wise cosine ≈ 0.001). Stage 2.6 used embed_tokens "
                "(input\nspace); this stage uses lm_head (output space), the "
                "actual space the lens projects into.\n\n")
        f.write(f"n_clips={n_clips}, D_sink={D_SINK}, τ={TAU_SINK}, "
                f"skip_pos0={skip_pos0}, top_K={TOP_K}\n\n")
        f.write(f"Mid-stack window L{MID_LO}-L{MID_HI}  "
                f"({n_layers_mid} layers, {safe_layers} with "
                f"mean n_nonsink/clip ≥ {MIN_NONSINK_FOR_FLAG})\n\n")
        f.write("Primary (weighted, top-10):\n")
        f.write(f"  gap (non-sink − sink) mean   = {g_w_mean:+.4f}\n")
        f.write(f"  gap median                    = {g_w_med:+.4f}\n")
        f.write(f"  layers with gap > 0           = {g_w_pos}/{n_layers_mid}\n\n")
        f.write("Strict (top-1):\n")
        f.write(f"  gap mean                      = {g_t1_mean:+.4f}\n")
        f.write(f"  gap median                    = {g_t1_med:+.4f}\n")
        f.write(f"  layers with gap > 0           = {g_t1_pos}/{n_layers_mid}\n\n")
        f.write(f"VERDICT: {combined}\n")
        f.write(f"  CONTENT-BLIND : gap > {GAP_THR} and ≥{POS_FRAC_THR*100:.0f}% mid layers positive\n")
        f.write(f"  UNINFORMATIVE : |gap| < {NEUTRAL_THR}\n")
        f.write(f"  INCONCLUSIVE  : otherwise\n")
    print(f"wrote {out_path}")
    return combined, g_w_mean, g_w_med, g_w_pos, g_t1_mean, g_t1_med, g_t1_pos


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
    lm_head_w = model.thinker.lm_head.weight
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ={TAU_SINK}, "
          f"skip_pos0={args.skip_pos0}, top_K={TOP_K}, "
          f"vocab_size={int(lm_head_w.shape[0])}")
    print(f"  lm_head device={lm_head_w.device}, dtype={lm_head_w.dtype}  "
          f"(label + decoded-token vectors will live here)")

    qa = json.load(open(args.qa))
    labels_by_vid = {q["video_id"]: list(q.get("label", [])) for q in qa}

    audio_dir = Path(args.audio_dir)
    clips_all = sorted(audio_dir.glob("*.wav"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed})\n")

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
            label_emb = embed_label_lm_head(label_str, processor.tokenizer, lm_head_w)
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
    df.to_csv(out_dir / "stage2_6b_per_clip_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage2_6b_per_clip_layer.csv'}  "
          f"({len(df)} records)")

    agg = aggregate_per_layer(df)
    agg.to_csv(out_dir / "stage2_6b_per_layer.csv", index=False)
    print(f"wrote {out_dir / 'stage2_6b_per_layer.csv'}")

    plot_two_panel(agg, out_dir / "stage2_6b_logit_lens_gap.png")
    combined, gw_mean, gw_med, gw_pos, gt_mean, gt_med, gt_pos = write_decision(
        agg, out_dir / "stage2_6b_decision.txt",
        n_clips=df["clip"].nunique(), skip_pos0=args.skip_pos0)

    print()
    print("=" * 80)
    print(f"STAGE 2.6b — mid-stack gap (L{MID_LO}-L{MID_HI})  "
          f"(n_clips = {df['clip'].nunique()})")
    print("=" * 80)
    mid = agg[(agg["layer"] >= MID_LO) & (agg["layer"] <= MID_HI)]
    print(f"  mean n_sink/clip   = {mid['mean_n_sink'].mean():.1f}")
    print(f"  mean n_nonsink/clip= {mid['mean_n_nonsink'].mean():.1f}  "
          f"({(mid['mean_n_nonsink'] >= MIN_NONSINK_FOR_FLAG).sum()}/{len(mid)} "
          f"layers above the {MIN_NONSINK_FOR_FLAG}-token confound floor)")
    print(f"  weighted: gap mean = {gw_mean:+.4f}, median = {gw_med:+.4f},  "
          f"{gw_pos}/{len(mid)} layers > 0")
    print(f"  top-1   : gap mean = {gt_mean:+.4f}, median = {gt_med:+.4f},  "
          f"{gt_pos}/{len(mid)} layers > 0")
    print(f"\n  VERDICT: {combined}")
    print(f"  (Stage 2.6 originally, in embed_tokens / INPUT space, returned: "
          f"weighted gap mean = -0.0021, top-1 mean = -0.0023, both UNINFORMATIVE.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--qa", default=str(DEFAULT_QA))
    p.add_argument("--n_clips", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_pos0", action="store_true", default=True)
    p.add_argument("--keep_pos0", dest="skip_pos0", action="store_false")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
