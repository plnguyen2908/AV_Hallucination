"""
stage2_7b_vocab_probe_propagated.py

Stage 2.7b — Same per-layer logit-lens vocabulary probe as Stage 2.7, but
with the PROPAGATED sink population (P_prop) instead of the LLM-emerged one.

Stage 2.7 used the LLM-side D_sink criterion (P_llm) and got KILL: both
populations decoded to vocab-tail noise. This swap tests the *other* sink
population — the encoder-norm-defined one Stage 1.1 / 1.2 / Sink-or-Not
study — to see if those few high-encoder-norm audio tokens decode
differently from the rest.

Conventions (matching Stages 1.1 / 1.2 / 2.7):
  - Propagation criterion: encoder L2 norm > 100 (Sink-or-Not τ). Hooked
    from `thinker.audio_tower` and aligned to LLM-side audio positions with
    Stage 1.2's chunked-mean / repeat align_norms helper.
  - Logit lens: model's REAL final RMSNorm + lm_head (learned weights).
  - Audio span only; first audio position (pos-0 BOS, per Stage 2.4) is
    dropped by default. --keep_pos0 to override.

P_prop mask is FIXED across all layers (it's defined at the encoder side,
before the LLM), so the partition is the same at every layer; only the
decoded tokens vary. P_prop is sparse — Stage 1.1 found ~5 propagated
audio tokens / clip with 15% of clips having zero — so total counts are
small even with 50 clips.

Two outputs per layer (same format as Stage 2.7):

  DISPLAY top-15 — most frequent decoded tokens that appear in the top-10
  lists of each population.

  PRIMARY CONTRASTIVE — smoothed log-odds, α=0.01, top-1 per position:
      score(v) = log((prop_count[v]    + α) / (prop_total    + α·V))
               − log((nonprop_count[v] + α) / (nonprop_total + α·V))
    Positive score → token v enriched at PROPAGATED positions.
    Negative score → token v enriched at NON-PROPAGATED positions.
    Drop tokens with (prop_count + nonprop_count) < MIN_TOTAL_COUNT.

Verdict (mid-stack L10-L22):
  CONFIRMED   — propagated positions decode to structural / register tokens
                AND non-propagated decode to content-adjacent tokens.
  KILL        — both populations decode to structural junk: P_prop also
                isn't readable by the lens.
  INCONCLUSIVE — mixed.

(Note: no saturation flag here. P_prop count is fixed across layers — it's
the criterion, not the layer, that defines the partition. But total
counts are small, so list rankings have higher variance than in Stage 2.7.)
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

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

# --- constants ---------------------------------------------------------------
TAU_PROP = 100.0                # Sink-or-Not / Stage 1.1 / 1.2 propagation τ
PROMPT_AUDIO = "Describe what you hear in detail."
TOP_K = 10
ALPHA_SMOOTH = 0.01
MIN_TOTAL_COUNT = 10
DISPLAY_TOP = 15
MID_LO, MID_HI = 10, 22
SEED = 42

DEFAULT_AUDIO_DIR = _REPO / "data/AudioSet_describe/audios"
DEFAULT_QA = _REPO / "data/AudioSet_describe/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_7b_vocab_probe_propagated"

# Structural-token heuristic (same as Stage 2.7, used only for the auto verdict).
_STRUCT_LITERALS = {
    "<|", "|>", "</", "</s>", "<s>", "<bos>", "<eos>", "<pad>", "<unk>",
    "[CLS]", "[SEP]", "[PAD]", "[MASK]",
}
_PUNCT_RE = re.compile(r"^[\s\W_]+$")
_FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "but", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "for", "with",
    "as", "at", "by", "this", "that", "it", "its", "from", "into", "if",
    "then", "than", "so", "such", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "his", "her", "us", "them", "do", "does",
    "did", "will", "would", "could", "should", "can", "may", "might", "no",
    "not", "yes",
}


def _is_structural(decoded: str, raw: str) -> bool:
    s = decoded.strip()
    if not s:
        return True
    if raw in _STRUCT_LITERALS or raw.startswith("<|") or raw.endswith("|>"):
        return True
    if _PUNCT_RE.match(s):
        return True
    if s.lower() in _FUNCTION_WORDS:
        return True
    return False


# --- model helpers -----------------------------------------------------------

def _audio_positions(input_ids, thinker_cfg):
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    return np.where(ids == a_id)[0].astype(np.int64)


def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_token_index"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _resolve_audio_encoder(model):
    """Same lookup as Stage 2.1 / stage2_1_layer_sink_counts.py."""
    thinker = model.thinker
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr):
            return getattr(thinker, attr)
    raise RuntimeError("no audio encoder module on thinker")


def _extract_tokens(out):
    """Same as Stage 2.1's _extract_tokens — handle (output, ..), or HF
    BaseModelOutput, or a plain tensor. Returns the (n_enc, hidden) 2D tensor."""
    x = out
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    return x


def align_norms(enc_norms, n_llm):
    """Stage 1.2's align_norms: chunked-mean down or repeat up."""
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


def _safe_decode(tok_id: int, tokenizer):
    raw = tokenizer.convert_ids_to_tokens(int(tok_id))
    try:
        decoded = tokenizer.decode([int(tok_id)], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    except Exception:
        decoded = ""
    return raw, decoded


# --- per-clip processing -----------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers,
                 audio_enc, skip_pos0=True):
    """For ONE clip:
      - hook the audio encoder, get encoder L2 norms (n_enc,)
      - hook each LLM decoder layer's output, get hidden states for the
        FULL audio span (so encoder→LLM alignment is on the unsliced span)
      - align encoder norms to LLM audio span, threshold > TAU_PROP → is_prop
      - logit-lens decode at each layer, return per-layer (is_prop, top1, topk)
      - if skip_pos0, slice off the first audio position from is_prop AND
        from each layer's captured hidden state before computing the lens
    Returns list[dict] (one per layer) or None on failure."""
    conv = build_conversation(str(clip_path), PROMPT_AUDIO, "a")
    inputs, use_aiv = prepare_inputs(processor, conv, "a",
                                      model.device, model.dtype)
    audio_pos = _audio_positions(inputs["input_ids"], thinker_cfg)
    if len(audio_pos) == 0:
        return None
    a_pos_t = torch.from_numpy(audio_pos)              # full span

    enc_buf: list = []
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    enc_handle = audio_enc.register_forward_hook(enc_hook)

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
        enc_handle.remove()
        for h in handles:
            h.remove()
    if any(h is None for h in per_layer_h) or not enc_buf:
        return None

    # ---- align encoder norms to LLM audio span and threshold ---------------
    enc_norms = np.concatenate(enc_buf)
    n_audio_full = int(len(audio_pos))
    aligned, tag = align_norms(enc_norms, n_audio_full)
    if aligned is None:
        return None
    is_prop_full = aligned > TAU_PROP                       # (n_audio_full,) bool

    if skip_pos0 and n_audio_full > 1:
        keep_slice = slice(1, None)
    else:
        keep_slice = slice(None)
    is_prop = is_prop_full[keep_slice]

    final_norm = model.thinker.model.norm
    lm_head = model.thinker.lm_head
    fn_dev = final_norm.weight.device

    out_per_layer = []
    for L in range(n_layers):
        h = per_layer_h[L][keep_slice]                      # (n_audio_used, hidden)
        h_fn = h.to(fn_dev)
        with torch.no_grad():
            h_normed = final_norm(h_fn)
            logits = lm_head(h_normed).float()
            _, topk_idx = logits.topk(TOP_K, dim=-1)
        topk_np = topk_idx.cpu().numpy()
        top1_np = topk_np[:, 0]
        out_per_layer.append(dict(is_prop=is_prop, top1=top1_np,
                                    topk=topk_np, align_tag=tag))
        per_layer_h[L] = None
    return out_per_layer


# --- contrastive scoring + report --------------------------------------------

def _compute_log_odds(prop_counts, non_counts, prop_total, non_total,
                      alpha=ALPHA_SMOOTH, min_total=MIN_TOTAL_COUNT):
    V = len(prop_counts)
    total = prop_counts + non_counts
    mask = total >= min_total
    denom_p = prop_total + alpha * V
    denom_n = non_total  + alpha * V
    log_p = np.log((prop_counts.astype(np.float64) + alpha) / max(denom_p, 1e-12))
    log_n = np.log((non_counts.astype(np.float64)  + alpha) / max(denom_n, 1e-12))
    return log_p - log_n, mask


def _format_row(tok_id, raw, decoded, *cols):
    raw_show = raw.replace("\n", "\\n").replace("\t", "\\t")
    dec_show = decoded.replace("\n", "\\n").replace("\t", "\\t")
    if len(raw_show) > 22: raw_show = raw_show[:19] + "..."
    if len(dec_show) > 22: dec_show = dec_show[:19] + "..."
    return (f"      {tok_id:>7d}  {raw_show:<22s}  {dec_show:<22s}   "
            + "  ".join(str(c) for c in cols))


def write_report(layer_data, tokenizer, out_dir: Path, args):
    report_path = out_dir / "stage2_7b_report.txt"
    contrastive_rows = []
    summary_rows = []
    mid_prop_struct, mid_non_struct = [], []

    with open(report_path, "w") as rf:
        rf.write("Stage 2.7b — audio PROPAGATED sink logit-lens vocabulary probe\n")
        rf.write("=" * 80 + "\n\n")
        rf.write("Propagation criterion: audio_tower output L2 norm > "
                 f"{TAU_PROP}  (Sink-or-Not / Stage 1.1 / 1.2 τ)\n")
        rf.write("Logit lens:            model's REAL final RMSNorm (learned weight) "
                 "+ lm_head\n")
        rf.write(f"top_K = {TOP_K}, skip_pos0 = {args.skip_pos0}, "
                 f"n_clips = {args.n_clips}, seed = {args.seed}\n\n")

        rf.write("CONTRASTIVE SCORE FORMULA (top-1 counts, α-smoothed log-odds)\n")
        rf.write("-" * 80 + "\n")
        rf.write(f"  score(v) = log((prop_count[v]    + α) / "
                 f"(prop_total    + α·V))\n")
        rf.write(f"           − log((nonprop_count[v] + α) / "
                 f"(nonprop_total + α·V))\n")
        rf.write(f"  α = {ALPHA_SMOOTH}, V = vocab size = {tokenizer.vocab_size}\n")
        rf.write(f"  Filter: tokens with (prop_count + nonprop_count) < "
                 f"{MIN_TOTAL_COUNT} are excluded from the ranking.\n")
        rf.write("  Positive → token enriched at PROPAGATED positions.\n")
        rf.write("  Negative → token enriched at NON-PROPAGATED positions.\n\n")

        rf.write("HOW TO READ THIS REPORT\n")
        rf.write("-" * 80 + "\n")
        rf.write("- Propagated set is SPARSE: Stage 1.1 found ~5 propagated audio "
                 "tokens / clip with 15% of clips at zero, so total prop counts "
                 "here are modest (~250 across 50 clips).\n")
        rf.write("- The P_prop mask is FIXED across all layers (defined at the "
                 "encoder); only the decoded tokens vary by layer.\n")
        rf.write(f"- Mid-stack L{MID_LO}-L{MID_HI} is the interpretable window. "
                 "Late layers are NOT saturation-confounded here (unlike Stage 2.7) "
                 "because the partition isn't layer-dependent.\n")
        rf.write("- Hypothesis CONFIRMED if PROP positions are dominated by "
                 "structural tokens (punctuation, special, function words) and "
                 "NON-PROP positions decode to content words, consistently across "
                 "mid-stack. KILL if both sides are structural junk.\n\n")

        for L in sorted(layer_data.keys()):
            d = layer_data[L]
            n_p = d["n_prop_total"]
            n_n = d["n_nonprop_total"]
            empty_flag = " [LOW-PROP]" if n_p < MIN_TOTAL_COUNT else ""
            rf.write("=" * 80 + "\n")
            rf.write(f"Layer L{L:02d}{empty_flag}  "
                     f"n_prop_total = {n_p:,}, n_nonprop_total = {n_n:,}, "
                     f"mean n_prop/clip = {d['mean_n_prop_per_clip']:.2f}, "
                     f"mean n_nonprop/clip = {d['mean_n_nonprop_per_clip']:.1f}\n")
            rf.write("-" * 80 + "\n")

            summary_rows.append(dict(
                layer=L, n_prop_total=int(n_p), n_nonprop_total=int(n_n),
                mean_n_prop_per_clip=float(d["mean_n_prop_per_clip"]),
                mean_n_nonprop_per_clip=float(d["mean_n_nonprop_per_clip"]),
                n_clips_with_prop=int(d.get("n_clips_with_prop", 0)),
            ))

            # DISPLAY top-15
            def _top_n(counts, n):
                idx = np.argsort(counts)[::-1][:n]
                return [(int(v), int(counts[v])) for v in idx if counts[v] > 0]
            prop_disp    = _top_n(d["prop_topk_counts"],    DISPLAY_TOP)
            nonprop_disp = _top_n(d["nonprop_topk_counts"], DISPLAY_TOP)

            rf.write(f"\n  DISPLAY — top-{DISPLAY_TOP} most frequent decoded "
                     f"tokens (counts over top-{TOP_K} lists)\n")
            rf.write("    PROPAGATED positions:\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  {'decoded':<22s}"
                     f"   topk_count\n")
            for v, c in prop_disp:
                raw, dec = _safe_decode(v, tokenizer)
                rf.write(_format_row(v, raw, dec, c) + "\n")
            rf.write("    NON-PROPAGATED positions:\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  {'decoded':<22s}"
                     f"   topk_count\n")
            for v, c in nonprop_disp:
                raw, dec = _safe_decode(v, tokenizer)
                rf.write(_format_row(v, raw, dec, c) + "\n")

            # CONTRASTIVE — log-odds on TOP-1 counts
            scores, mask = _compute_log_odds(
                d["prop_top1_counts"], d["nonprop_top1_counts"],
                d["n_prop_total"], d["n_nonprop_total"])
            usable = np.where(mask)[0]
            if len(usable) == 0:
                rf.write("\n  CONTRASTIVE — no tokens pass the min-count filter\n\n")
                continue
            s_use = scores[usable]
            ord_high = usable[np.argsort(s_use)[::-1]][:DISPLAY_TOP]
            ord_low  = usable[np.argsort(s_use)][:DISPLAY_TOP]

            rf.write(f"\n  CONTRASTIVE — top-{DISPLAY_TOP} by smoothed log-odds "
                     f"(top-1 counts; total ≥ {MIN_TOTAL_COUNT})\n")
            rf.write("    Enriched at PROPAGATED (positive score):\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  {'decoded':<22s}"
                     f"   score    prop_cnt  non_cnt\n")
            n_struct_prop = 0
            for v in ord_high:
                raw, dec = _safe_decode(v, tokenizer)
                if _is_structural(dec, raw): n_struct_prop += 1
                rf.write(_format_row(
                    v, raw, dec,
                    f"{scores[v]:+.3f}",
                    f"{int(d['prop_top1_counts'][v]):>7d}",
                    f"{int(d['nonprop_top1_counts'][v]):>7d}") + "\n")
            rf.write("    Enriched at NON-PROPAGATED (negative score):\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  {'decoded':<22s}"
                     f"   score    prop_cnt  non_cnt\n")
            n_struct_non = 0
            for v in ord_low:
                raw, dec = _safe_decode(v, tokenizer)
                if _is_structural(dec, raw): n_struct_non += 1
                rf.write(_format_row(
                    v, raw, dec,
                    f"{scores[v]:+.3f}",
                    f"{int(d['prop_top1_counts'][v]):>7d}",
                    f"{int(d['nonprop_top1_counts'][v]):>7d}") + "\n")
            rf.write(f"    [auto-heuristic] structural / function / punct fraction "
                     f"in top-{DISPLAY_TOP}: "
                     f"prop-enriched = {n_struct_prop}/{DISPLAY_TOP} "
                     f"({n_struct_prop/DISPLAY_TOP*100:.0f}%), "
                     f"non-prop-enriched = {n_struct_non}/{DISPLAY_TOP} "
                     f"({n_struct_non/DISPLAY_TOP*100:.0f}%)\n\n")
            if MID_LO <= L <= MID_HI:
                mid_prop_struct.append(n_struct_prop / DISPLAY_TOP)
                mid_non_struct.append(n_struct_non / DISPLAY_TOP)

            ord_all = usable[np.argsort(scores[usable])[::-1]]
            for v in ord_all:
                raw, dec = _safe_decode(v, tokenizer)
                contrastive_rows.append(dict(
                    layer=L, vocab_id=int(v),
                    raw_bpe=raw, decoded=dec,
                    prop_count=int(d["prop_top1_counts"][v]),
                    nonprop_count=int(d["nonprop_top1_counts"][v]),
                    total=int(d["prop_top1_counts"][v]
                              + d["nonprop_top1_counts"][v]),
                    score=float(scores[v]),
                    is_structural=bool(_is_structural(dec, raw)),
                ))

        rf.write("=" * 80 + "\n")
        rf.write(f"VERDICT (mid-stack auto-heuristic, averaged over L{MID_LO}-L{MID_HI})\n")
        rf.write("-" * 80 + "\n")
        if not mid_prop_struct:
            rf.write("  no mid-stack layers qualified\n")
            verdict = "INCONCLUSIVE"
        else:
            p_avg = float(np.mean(mid_prop_struct))
            n_avg = float(np.mean(mid_non_struct))
            rf.write(f"  fraction of structural / function / punct tokens in "
                     f"DISPLAY_TOP={DISPLAY_TOP} contrastive lists:\n")
            rf.write(f"    PROP-enriched     mean = {p_avg*100:.0f}%  "
                     f"(n_mid_layers = {len(mid_prop_struct)})\n")
            rf.write(f"    NON-PROP-enriched mean = {n_avg*100:.0f}%\n")
            if p_avg >= 0.60 and n_avg <= 0.40:
                verdict = ("CONFIRMED (propagated are structural, "
                            "non-propagated are content)")
            elif p_avg >= 0.60 and n_avg >= 0.60:
                verdict = "KILL (both populations decode to structural junk)"
            else:
                verdict = "INCONCLUSIVE (inspect lists; heuristic is English-centric)"
            rf.write(f"\n  VERDICT: {verdict}\n")
            rf.write(f"    CONFIRMED   : prop-struct ≥ 0.60 AND non-prop-struct ≤ 0.40\n")
            rf.write(f"    KILL        : both ≥ 0.60\n")
            rf.write(f"    INCONCLUSIVE: otherwise (manual read of per-layer "
                     f"lists is the real verdict)\n")
    print(f"wrote {report_path}")
    return contrastive_rows, summary_rows, verdict


# --- main --------------------------------------------------------------------

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
    audio_enc = _resolve_audio_encoder(model)
    V = int(model.thinker.lm_head.weight.shape[0])
    print(f"  n_layers={n_layers}, τ_prop={TAU_PROP}, vocab_size={V}, "
          f"skip_pos0={args.skip_pos0}, top_K={TOP_K}")

    audio_dir = Path(args.audio_dir)
    clips_all = sorted(audio_dir.glob("*.wav"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed})\n")

    prop_top1 = np.zeros((n_layers, V), dtype=np.int64)
    non_top1  = np.zeros((n_layers, V), dtype=np.int64)
    prop_topk = np.zeros((n_layers, V), dtype=np.int64)
    non_topk  = np.zeros((n_layers, V), dtype=np.int64)
    n_prop_total    = np.zeros(n_layers, dtype=np.int64)
    n_nonprop_total = np.zeros(n_layers, dtype=np.int64)
    per_clip_prop    = [[] for _ in range(n_layers)]
    per_clip_nonprop = [[] for _ in range(n_layers)]
    n_clips_with_prop = Counter()
    align_tags: Counter = Counter()

    failures: dict = {}
    for clip in tqdm(clips, desc="clips"):
        try:
            layer_data = process_clip(model, processor, clip, thinker_cfg, layers,
                                       audio_enc, skip_pos0=args.skip_pos0)
        except Exception as e:
            failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
            tqdm.write(f"  [skip] {clip.name}: {type(e).__name__}: {e}")
            continue
        if layer_data is None:
            failures["no_audio_or_align"] = failures.get("no_audio_or_align", 0) + 1
            continue
        if layer_data:
            align_tags[layer_data[0]["align_tag"]] += 1
        for L, d in enumerate(layer_data):
            is_prop = d["is_prop"]; top1 = d["top1"]; topk = d["topk"]
            prop_mask = is_prop; non_mask = ~is_prop
            n_p = int(prop_mask.sum()); n_n = int(non_mask.sum())
            n_prop_total[L]    += n_p
            n_nonprop_total[L] += n_n
            per_clip_prop[L].append(n_p)
            per_clip_nonprop[L].append(n_n)
            if n_p > 0: n_clips_with_prop[L] += 1
            if n_p > 0:
                prop_top1[L] += np.bincount(top1[prop_mask], minlength=V).astype(np.int64)
                prop_topk[L] += np.bincount(topk[prop_mask].reshape(-1),
                                              minlength=V).astype(np.int64)
            if n_n > 0:
                non_top1[L] += np.bincount(top1[non_mask], minlength=V).astype(np.int64)
                non_topk[L] += np.bincount(topk[non_mask].reshape(-1),
                                             minlength=V).astype(np.int64)
        torch.cuda.empty_cache()

    if failures:
        print(f"  failures: {failures}")
    print(f"  encoder→LLM alignment tags: {dict(align_tags)}")

    layer_data = {}
    for L in range(n_layers):
        layer_data[L] = dict(
            prop_top1_counts    = prop_top1[L],
            nonprop_top1_counts = non_top1[L],
            prop_topk_counts    = prop_topk[L],
            nonprop_topk_counts = non_topk[L],
            n_prop_total        = int(n_prop_total[L]),
            n_nonprop_total     = int(n_nonprop_total[L]),
            mean_n_prop_per_clip = (float(np.mean(per_clip_prop[L]))
                                     if per_clip_prop[L] else 0.0),
            mean_n_nonprop_per_clip = (float(np.mean(per_clip_nonprop[L]))
                                        if per_clip_nonprop[L] else 0.0),
            n_clips_with_prop = int(n_clips_with_prop[L]),
        )

    contrastive_rows, summary_rows, verdict = write_report(
        layer_data, processor.tokenizer, out_dir, args)

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "stage2_7b_per_layer_summary.csv", index=False)
    print(f"wrote {out_dir / 'stage2_7b_per_layer_summary.csv'}")
    pd.DataFrame(contrastive_rows).to_csv(
        out_dir / "stage2_7b_contrastive.csv", index=False)
    print(f"wrote {out_dir / 'stage2_7b_contrastive.csv'}  "
          f"({len(contrastive_rows)} rows)")
    np.savez_compressed(
        out_dir / "stage2_7b_counters.npz",
        prop_top1=prop_top1, nonprop_top1=non_top1,
        prop_topk=prop_topk, nonprop_topk=non_topk,
        n_prop_total=n_prop_total, n_nonprop_total=n_nonprop_total)
    print(f"wrote {out_dir / 'stage2_7b_counters.npz'}")

    n_proc = len(clips) - sum(failures.values())
    print("\n" + "=" * 80)
    print(f"STAGE 2.7b VERDICT  (n_clips = {n_proc})")
    print("=" * 80)
    print(f"  {verdict}")
    print(f"  mean n_prop/clip across layers: "
          f"{np.mean([layer_data[L]['mean_n_prop_per_clip'] for L in range(n_layers)]):.2f}  "
          f"(P_prop is FIXED across layers; small variation = clips with 0 prop)")
    print(f"  full per-layer report in: {out_dir / 'stage2_7b_report.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--qa", default=str(DEFAULT_QA),
                   help="(unused; kept for invocation parity with Stage 2.7)")
    p.add_argument("--n_clips", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_pos0", action="store_true", default=True)
    p.add_argument("--keep_pos0", dest="skip_pos0", action="store_false")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
