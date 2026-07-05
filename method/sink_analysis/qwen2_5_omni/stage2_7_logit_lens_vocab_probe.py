"""
stage2_7_logit_lens_vocab_probe.py

Stage 2.7 — Per-layer logit-lens VOCABULARY probe for audio LLM-emerged sinks.

Stage 2.6 measured the semantic distance of sink-token decodings to the GT label
in embedding space; the gap was null. This stage opens the box and just looks
at WHAT the logit lens emits, per layer, contrastively for sink vs non-sink
audio positions. Embedding-free and label-free.

Hypothesis: sinks decode to STRUCTURAL/high-frequency junk (register-like
behavior); non-sinks decode to label-adjacent content. Read out the
mid-stack — where the lens has meaning but the non-sink population hasn't
yet collapsed under saturation — and look for the asymmetry.

Conventions, exactly matching Stages 2.1 / 2.4 / 2.6:
  - Sink criterion is P_llm (LLM-emerged):
        pure RMSNorm (NO learned weight) + D_sink={458, 2570} + τ=20.
    Pure RMSNorm = sqrt(mean(x²) + eps); the learned-weight rescale is
    deliberately omitted (matches stage2_1_layer_sink_counts.py).
    This is NOT the propagated / encoder-norm population.
  - Logit lens uses the model's REAL final norm + lm_head (learned-weight
    RMSNorm followed by the unembedding). The norm here is a DIFFERENT
    operation from the sink-detection norm above; don't conflate them.
  - Audio span only; first audio position (pos-0 BOS register per 2.4) is
    dropped by default. --keep_pos0 to override.

Two outputs per layer:

  DISPLAY top-15 — most frequent decoded tokens that appear in the top-10
  lists of each population. Each audio position contributes its top-10 ids
  to its population's counter; "top-15 most frequent" = the 15 vocab ids
  with the highest such count. Useful for human inspection but dominated
  by globally common tokens regardless of sink status, which is why it's
  NOT the primary result.

  PRIMARY CONTRASTIVE — smoothed log-odds, with α=0.01, top-1 per position:
      score(v) = log((sink_count[v]    + α) / (sink_total    + α·V))
               − log((nonsink_count[v] + α) / (nonsink_total + α·V))
    Positive score → token v enriched at SINK positions.
    Negative score → token v enriched at NON-SINK positions.
    Drop tokens with (sink_count + nonsink_count) < MIN_TOTAL_COUNT before
    ranking — the cheap substitute for a prior on rare tokens.

  Counting rule: contrastive uses TOP-1 per position so each position
  contributes mass where it actually points, not 10 near-tied votes.
  Display uses TOP-10 frequency, by design.

Verdict (per layer; combined across mid-stack L10-L22):
  CONFIRMED   — sink-enriched lists are dominated by structural/punctuation/
                special tokens AND non-sink-enriched lists contain content
                words, consistently across mid-stack.
  KILL        — both populations decode to structural junk: lens is
                uninformative for audio tokens; surface clearly.
  INCONCLUSIVE — mixed or borderline.
"""

import argparse
import json
import math
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

# --- constants (matched to Stage 2.6 / 2.4 / 2.1) -----------------------------
D_SINK = [458, 2570]
TAU_SINK = 20.0
PROMPT_AUDIO = "Describe what you hear in detail."
TOP_K = 10                       # decoded list size from the lens
ALPHA_SMOOTH = 0.01              # Laplace alpha for the log-odds
MIN_TOTAL_COUNT = 10             # drop rare tokens (sink+non-sink combined)
DISPLAY_TOP = 15                 # how many tokens to show per side per layer
MID_LO, MID_HI = 10, 22          # mid-stack read window
MIN_NONSINK_FOR_FLAG = 30        # below this mean per-clip → saturation flag
SEED = 42

DEFAULT_AUDIO_DIR = _REPO / "data/AudioSet_describe/audios"
DEFAULT_QA = _REPO / "data/AudioSet_describe/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_7_vocab_probe"

# Coarse heuristic for "is this token structural / function / special?"
# Used only for the printed verdict block — the CSV keeps everything.
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


def _safe_decode(tok_id: int, tokenizer) -> tuple[str, str]:
    """Returns (raw_bpe, decoded). Both for transparency in the report."""
    raw = tokenizer.convert_ids_to_tokens(int(tok_id))
    try:
        decoded = tokenizer.decode([int(tok_id)], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    except Exception:
        decoded = ""
    return raw, decoded


# --- per-clip processing ------------------------------------------------------

def process_clip(model, processor, clip_path, thinker_cfg, layers,
                 d_sink_t, eps_norm, skip_pos0=True):
    """Per layer L, returns dict with:
        is_sink   : (n_audio,) bool
        top1_idx  : (n_audio,)  int64  argmax decoded vocab id
        topk_idx  : (n_audio, K) int64 top-K decoded vocab ids
    All numpy on cpu. Returns None on failure."""
    conv = build_conversation(str(clip_path), PROMPT_AUDIO, "a")
    inputs, use_aiv = prepare_inputs(processor, conv, "a",
                                      model.device, model.dtype)

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

    final_norm = model.thinker.model.norm        # learned-weight RMSNorm
    lm_head = model.thinker.lm_head
    fn_dev = final_norm.weight.device

    out_per_layer = []
    for L in range(n_layers):
        h = per_layer_h[L]                      # (n_audio, hidden) on layer device
        h_f = h.float()
        rms = torch.sqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_f / rms).abs()
        d_t = d_sink_t.to(h.device)
        is_sink = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()

        h_fn = h.to(fn_dev)
        with torch.no_grad():
            h_normed = final_norm(h_fn)
            logits = lm_head(h_normed).float()  # (n_audio, vocab)
            _, topk_idx = logits.topk(TOP_K, dim=-1)
        topk_np = topk_idx.cpu().numpy()
        top1_np = topk_np[:, 0]

        out_per_layer.append(dict(is_sink=is_sink, top1=top1_np, topk=topk_np))
        per_layer_h[L] = None
    return out_per_layer


# --- contrastive scoring + display --------------------------------------------

def _compute_log_odds(sink_counts: np.ndarray, non_counts: np.ndarray,
                      sink_total: int, non_total: int,
                      alpha: float = ALPHA_SMOOTH,
                      min_total: int = MIN_TOTAL_COUNT):
    """Returns (scores, mask) where mask[v]=True iff total >= min_total."""
    V = len(sink_counts)
    total = sink_counts + non_counts
    mask = total >= min_total
    denom_s = sink_total + alpha * V
    denom_n = non_total  + alpha * V
    log_p_sink = np.log((sink_counts.astype(np.float64) + alpha) / max(denom_s, 1e-12))
    log_p_non  = np.log((non_counts.astype(np.float64)  + alpha) / max(denom_n, 1e-12))
    scores = log_p_sink - log_p_non
    return scores, mask


def _format_token_row(tok_id, raw, decoded, *cols) -> str:
    raw_show = raw.replace("\n", "\\n").replace("\t", "\\t")
    dec_show = decoded.replace("\n", "\\n").replace("\t", "\\t")
    if len(raw_show) > 22: raw_show = raw_show[:19] + "..."
    if len(dec_show) > 22: dec_show = dec_show[:19] + "..."
    col_str = "  ".join(str(c) for c in cols)
    return f"      {tok_id:>7d}  {raw_show:<22s}  {dec_show:<22s}   {col_str}"


def write_report(layer_data, layer_n_clips_with_sinks, layer_n_clips_with_nonsinks,
                 tokenizer, out_dir: Path, args):
    """layer_data: dict of dict.
       layer_data[L] = {
         sink_top1_counts, nonsink_top1_counts,         # (V,) np.int64
         sink_topk_counts, nonsink_topk_counts,         # (V,) np.int64
         n_sink_total, n_nonsink_total,                 # int
         mean_n_sink_per_clip, mean_n_nonsink_per_clip, # float
       }"""
    report_path = out_dir / "stage2_7_report.txt"
    contrastive_rows = []
    per_layer_summary = []
    sink_struct_frac_in_mid = []
    nonsink_struct_frac_in_mid = []

    with open(report_path, "w") as rf:
        # ----- header / formula -----
        rf.write("Stage 2.7 — audio LLM-emerged sink logit-lens vocabulary probe\n")
        rf.write("=" * 80 + "\n\n")
        rf.write(f"Sink criterion: pure RMSNorm (no learned weight) + "
                 f"D_sink={D_SINK} + τ={TAU_SINK}\n")
        rf.write(f"Logit lens:     model's REAL final RMSNorm (learned weight) + "
                 f"lm_head\n")
        rf.write(f"top_K = {TOP_K}, skip_pos0 = {args.skip_pos0}, "
                 f"n_clips = {args.n_clips}, seed = {args.seed}\n\n")

        rf.write("CONTRASTIVE SCORE FORMULA (top-1 counts, α-smoothed log-odds)\n")
        rf.write("-" * 80 + "\n")
        rf.write(f"  score(v) = log((sink_count[v]    + α) / "
                 f"(sink_total    + α·V))\n")
        rf.write(f"           − log((nonsink_count[v] + α) / "
                 f"(nonsink_total + α·V))\n")
        rf.write(f"  α = {ALPHA_SMOOTH}, V = vocab size = {tokenizer.vocab_size}\n")
        rf.write(f"  Filter: tokens with (sink_count + nonsink_count) < "
                 f"{MIN_TOTAL_COUNT} are excluded from the ranking.\n")
        rf.write("  Positive → token is enriched at SINK positions.\n")
        rf.write("  Negative → token is enriched at NON-SINK positions.\n\n")

        rf.write("HOW TO READ THIS REPORT\n")
        rf.write("-" * 80 + "\n")
        rf.write("- DISPLAY rows are raw top-15 by frequency in the top-10 decoded "
                 "lists. Use as a sanity\n  check on what the lens emits, but expect "
                 "dominance by globally common tokens regardless\n  of sink status. The "
                 "CONTRASTIVE rows are the real instrument.\n")
        rf.write("- Audio is ~74% sink by L21 and ~96% by L25 (Stage 2.1 / 2.4); when "
                 "mean n_nonsink/clip\n  drops below "
                 f"{MIN_NONSINK_FOR_FLAG} a layer is marked [SAT] and its rankings are "
                 "high-variance.\n")
        rf.write(f"- Mid-stack L{MID_LO}-L{MID_HI} is the interpretable window. Early "
                 "layers have ~0 sinks.\n")
        rf.write("- Hypothesis CONFIRMED if SINK-enriched is dominated by structural "
                 "tokens (punctuation,\n  special, function words) and NON-SINK-enriched "
                 "contains content words, consistently across\n  mid-stack. KILL if both "
                 "sides are structural junk across the mid-stack.\n\n")

        # ----- per layer -----
        for L in sorted(layer_data.keys()):
            d = layer_data[L]
            n_s, n_n = d["n_sink_total"], d["n_nonsink_total"]
            mean_ns_per_clip = d["mean_n_nonsink_per_clip"]
            sat_flag = "[SAT]" if mean_ns_per_clip < MIN_NONSINK_FOR_FLAG else "    "
            empty_flag = (" [EMPTY-SINK]"
                           if d["mean_n_sink_per_clip"] < 5 else "")
            rf.write("=" * 80 + "\n")
            rf.write(f"Layer L{L:02d}  {sat_flag}{empty_flag}  "
                     f"n_sink_total = {n_s:,}, n_nonsink_total = {n_n:,}, "
                     f"mean n_sink/clip = {d['mean_n_sink_per_clip']:.1f}, "
                     f"mean n_nonsink/clip = {mean_ns_per_clip:.1f}\n")
            rf.write("-" * 80 + "\n")

            per_layer_summary.append(dict(
                layer=L, n_sink_total=int(n_s), n_nonsink_total=int(n_n),
                mean_n_sink_per_clip=float(d["mean_n_sink_per_clip"]),
                mean_n_nonsink_per_clip=float(mean_ns_per_clip),
                n_clips_with_sinks=int(layer_n_clips_with_sinks.get(L, 0)),
                n_clips_with_nonsinks=int(layer_n_clips_with_nonsinks.get(L, 0)),
                sat=bool(mean_ns_per_clip < MIN_NONSINK_FOR_FLAG)))

            # DISPLAY top-15 most frequent (in top-10 lists)
            def _top_n_by(counts, n):
                idx = np.argsort(counts)[::-1][:n]
                return [(int(v), int(counts[v])) for v in idx if counts[v] > 0]
            sink_top_disp    = _top_n_by(d["sink_topk_counts"],    DISPLAY_TOP)
            nonsink_top_disp = _top_n_by(d["nonsink_topk_counts"], DISPLAY_TOP)

            rf.write(f"\n  DISPLAY — top-{DISPLAY_TOP} most frequent decoded tokens "
                     f"(counts over top-{TOP_K} lists)\n")
            rf.write("    SINK positions:\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  "
                     f"{'decoded':<22s}   topk_count\n")
            for v, c in sink_top_disp:
                raw, dec = _safe_decode(v, tokenizer)
                rf.write(_format_token_row(v, raw, dec, c) + "\n")
            rf.write("    NON-SINK positions:\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  "
                     f"{'decoded':<22s}   topk_count\n")
            for v, c in nonsink_top_disp:
                raw, dec = _safe_decode(v, tokenizer)
                rf.write(_format_token_row(v, raw, dec, c) + "\n")

            # CONTRASTIVE — log-odds on TOP-1 counts
            scores, mask = _compute_log_odds(
                d["sink_top1_counts"], d["nonsink_top1_counts"],
                d["n_sink_total"], d["n_nonsink_total"],
                alpha=ALPHA_SMOOTH, min_total=MIN_TOTAL_COUNT)
            usable = np.where(mask)[0]
            if len(usable) == 0:
                rf.write("\n  CONTRASTIVE — no tokens pass the min-count filter\n\n")
                continue
            s_in_use = scores[usable]
            ord_high = usable[np.argsort(s_in_use)[::-1]][:DISPLAY_TOP]
            ord_low  = usable[np.argsort(s_in_use)][:DISPLAY_TOP]

            rf.write(f"\n  CONTRASTIVE — top-{DISPLAY_TOP} by smoothed log-odds "
                     f"(top-1 counts; total ≥ {MIN_TOTAL_COUNT})\n")
            rf.write("    Enriched at SINK (positive score):\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  "
                     f"{'decoded':<22s}   score    sink_cnt  non_cnt\n")
            n_struct_sink = 0
            for v in ord_high:
                raw, dec = _safe_decode(v, tokenizer)
                if _is_structural(dec, raw):
                    n_struct_sink += 1
                rf.write(_format_token_row(
                    v, raw, dec,
                    f"{scores[v]:+.3f}",
                    f"{int(d['sink_top1_counts'][v]):>7d}",
                    f"{int(d['nonsink_top1_counts'][v]):>7d}") + "\n")
            rf.write("    Enriched at NON-SINK (negative score):\n")
            rf.write(f"      {'vocab_id':>7s}  {'raw_bpe':<22s}  "
                     f"{'decoded':<22s}   score    sink_cnt  non_cnt\n")
            n_struct_non = 0
            for v in ord_low:
                raw, dec = _safe_decode(v, tokenizer)
                if _is_structural(dec, raw):
                    n_struct_non += 1
                rf.write(_format_token_row(
                    v, raw, dec,
                    f"{scores[v]:+.3f}",
                    f"{int(d['sink_top1_counts'][v]):>7d}",
                    f"{int(d['nonsink_top1_counts'][v]):>7d}") + "\n")
            rf.write(f"    [auto-heuristic] structural / function / punctuation "
                     f"fraction in top-{DISPLAY_TOP}: "
                     f"sink-enriched = {n_struct_sink}/{DISPLAY_TOP} "
                     f"({n_struct_sink/DISPLAY_TOP*100:.0f}%), "
                     f"non-sink-enriched = {n_struct_non}/{DISPLAY_TOP} "
                     f"({n_struct_non/DISPLAY_TOP*100:.0f}%)\n\n")

            if MID_LO <= L <= MID_HI and mean_ns_per_clip >= MIN_NONSINK_FOR_FLAG:
                sink_struct_frac_in_mid.append(n_struct_sink / DISPLAY_TOP)
                nonsink_struct_frac_in_mid.append(n_struct_non / DISPLAY_TOP)

            # FULL ranking for the contrastive CSV (all passes filter, sorted)
            ord_all = usable[np.argsort(scores[usable])[::-1]]
            for v in ord_all:
                raw, dec = _safe_decode(v, tokenizer)
                contrastive_rows.append(dict(
                    layer=L, vocab_id=int(v),
                    raw_bpe=raw, decoded=dec,
                    sink_count=int(d["sink_top1_counts"][v]),
                    nonsink_count=int(d["nonsink_top1_counts"][v]),
                    total=int(d["sink_top1_counts"][v]
                              + d["nonsink_top1_counts"][v]),
                    score=float(scores[v]),
                    is_structural=bool(_is_structural(dec, raw)),
                ))

        # ----- verdict block at the end -----
        rf.write("=" * 80 + "\n")
        rf.write("VERDICT (mid-stack auto-heuristic, "
                 f"averaged over L{MID_LO}-L{MID_HI} layers with "
                 f"mean n_nonsink/clip ≥ {MIN_NONSINK_FOR_FLAG})\n")
        rf.write("-" * 80 + "\n")
        if not sink_struct_frac_in_mid:
            rf.write("  no mid-stack layers qualified — verdict not computable\n")
            verdict = "INCONCLUSIVE"
        else:
            s_avg = float(np.mean(sink_struct_frac_in_mid))
            n_avg = float(np.mean(nonsink_struct_frac_in_mid))
            rf.write(f"  fraction of structural / function / punct tokens in the "
                     f"DISPLAY_TOP={DISPLAY_TOP} contrastive lists:\n")
            rf.write(f"    SINK-enriched     mean = {s_avg*100:.0f}%  "
                     f"(n_mid_layers = {len(sink_struct_frac_in_mid)})\n")
            rf.write(f"    NON-SINK-enriched mean = {n_avg*100:.0f}%\n")
            # Verdict cutoffs (deliberate, not arbitrary):
            #   CONFIRMED   : sink-struct ≥ 0.60 AND non-sink-struct ≤ 0.40
            #   KILL        : both ≥ 0.60 (both sides structural junk)
            #   INCONCLUSIVE: otherwise
            if s_avg >= 0.60 and n_avg <= 0.40:
                verdict = "CONFIRMED (sinks are structural, non-sinks are content)"
            elif s_avg >= 0.60 and n_avg >= 0.60:
                verdict = "KILL (both populations decode to structural junk)"
            else:
                verdict = ("INCONCLUSIVE (structural fractions don't show a clean "
                           "sink-vs-content split; inspect the per-layer lists)")
            rf.write(f"\n  VERDICT: {verdict}\n")
            rf.write(f"    CONFIRMED   : sink-struct ≥ 0.60 AND non-sink-struct ≤ 0.40\n")
            rf.write(f"    KILL        : both ≥ 0.60\n")
            rf.write(f"    INCONCLUSIVE: otherwise\n")

    print(f"wrote {report_path}")
    return contrastive_rows, per_layer_summary, verdict


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
    V = int(model.thinker.lm_head.weight.shape[0])
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ={TAU_SINK}, eps={eps_norm}, "
          f"skip_pos0={args.skip_pos0}, top_K={TOP_K}, vocab_size={V}")

    # Reproduce Stage 2.4-describe clip order
    audio_dir = Path(args.audio_dir)
    clips_all = sorted(audio_dir.glob("*.wav"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed})\n")

    # Per-layer counters
    sink_top1 = np.zeros((n_layers, V), dtype=np.int64)
    non_top1  = np.zeros((n_layers, V), dtype=np.int64)
    sink_topk = np.zeros((n_layers, V), dtype=np.int64)
    non_topk  = np.zeros((n_layers, V), dtype=np.int64)
    n_sink_total    = np.zeros(n_layers, dtype=np.int64)
    n_nonsink_total = np.zeros(n_layers, dtype=np.int64)
    per_clip_sink_counts    = [[] for _ in range(n_layers)]
    per_clip_nonsink_counts = [[] for _ in range(n_layers)]
    n_clips_with_sinks    = Counter()
    n_clips_with_nonsinks = Counter()

    failures: dict = {}
    for clip in tqdm(clips, desc="clips"):
        try:
            layer_data = process_clip(model, processor, clip, thinker_cfg, layers,
                                       d_sink_t, eps_norm,
                                       skip_pos0=args.skip_pos0)
        except Exception as e:
            failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
            tqdm.write(f"  [skip] {clip.name}: {type(e).__name__}: {e}")
            continue
        if layer_data is None:
            failures["no_audio"] = failures.get("no_audio", 0) + 1
            continue
        for L, d in enumerate(layer_data):
            is_sink = d["is_sink"]; top1 = d["top1"]; topk = d["topk"]
            sink_mask = is_sink; non_mask = ~is_sink
            n_s = int(sink_mask.sum()); n_n = int(non_mask.sum())
            n_sink_total[L]    += n_s
            n_nonsink_total[L] += n_n
            per_clip_sink_counts[L].append(n_s)
            per_clip_nonsink_counts[L].append(n_n)
            if n_s > 0: n_clips_with_sinks[L]    += 1
            if n_n > 0: n_clips_with_nonsinks[L] += 1
            # bincount up to V; top1 is (n_audio,), topk is (n_audio, K)
            if n_s > 0:
                sink_top1[L] += np.bincount(top1[sink_mask], minlength=V).astype(np.int64)
                sink_topk[L] += np.bincount(topk[sink_mask].reshape(-1),
                                              minlength=V).astype(np.int64)
            if n_n > 0:
                non_top1[L] += np.bincount(top1[non_mask], minlength=V).astype(np.int64)
                non_topk[L] += np.bincount(topk[non_mask].reshape(-1),
                                             minlength=V).astype(np.int64)
        torch.cuda.empty_cache()

    if failures:
        print(f"  failures: {failures}")

    # Stage layer data for the report
    layer_data = {}
    for L in range(n_layers):
        layer_data[L] = dict(
            sink_top1_counts   = sink_top1[L],
            nonsink_top1_counts = non_top1[L],
            sink_topk_counts   = sink_topk[L],
            nonsink_topk_counts = non_topk[L],
            n_sink_total       = int(n_sink_total[L]),
            n_nonsink_total    = int(n_nonsink_total[L]),
            mean_n_sink_per_clip = (float(np.mean(per_clip_sink_counts[L]))
                                     if per_clip_sink_counts[L] else 0.0),
            mean_n_nonsink_per_clip = (float(np.mean(per_clip_nonsink_counts[L]))
                                        if per_clip_nonsink_counts[L] else 0.0),
        )

    contrastive_rows, summary_rows, verdict = write_report(
        layer_data, n_clips_with_sinks, n_clips_with_nonsinks,
        processor.tokenizer, out_dir, args)

    # Per-layer summary CSV
    pd.DataFrame(summary_rows).to_csv(out_dir / "stage2_7_per_layer_summary.csv",
                                       index=False)
    print(f"wrote {out_dir / 'stage2_7_per_layer_summary.csv'}")

    # Full contrastive CSV (all (layer, vocab) rows that pass the filter)
    pd.DataFrame(contrastive_rows).to_csv(
        out_dir / "stage2_7_contrastive.csv", index=False)
    print(f"wrote {out_dir / 'stage2_7_contrastive.csv'}  "
          f"({len(contrastive_rows)} rows)")

    # Raw counters npz for replot / reanalysis
    np.savez_compressed(
        out_dir / "stage2_7_counters.npz",
        sink_top1=sink_top1, nonsink_top1=non_top1,
        sink_topk=sink_topk, nonsink_topk=non_topk,
        n_sink_total=n_sink_total, n_nonsink_total=n_nonsink_total)
    print(f"wrote {out_dir / 'stage2_7_counters.npz'}")

    print("\n" + "=" * 80)
    print(f"STAGE 2.7 VERDICT  (n_clips = {len(clips) - sum(failures.values())})")
    print("=" * 80)
    print(f"  {verdict}")
    print(f"  full per-layer report in: {out_dir / 'stage2_7_report.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--qa", default=str(DEFAULT_QA),
                   help="Unused here (kept for parity with Stage 2.6 invocation).")
    p.add_argument("--n_clips", type=int, default=50,
                   help="Default 50 = smoke / kill-check pass.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_pos0", action="store_true", default=True)
    p.add_argument("--keep_pos0", dest="skip_pos0", action="store_false")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
