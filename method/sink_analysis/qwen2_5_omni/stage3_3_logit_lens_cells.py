"""
stage3_3_logit_lens_cells.py — ASD content-hub validation of cross-modal
sinks (Stage 3.3).

For every sink at every layer (P_prop AND P_llm; one cell tag per sink)
project the pre-SA hidden state h_input[L] through the model's REAL
final RMSNorm + LM head, take top-K vocab tokens and full-distribution
entropy. Aggregate per (population × cell × layer × lens_state) into:
  - content_rate   = mean(top-1 is a content/alphabetic word piece)
  - on_topic_rate  = mean(top-K contains a substring of the VGGSounder
                          ground-truth label for this clip)
  - mean_entropy   = mean H over the full softmax (152K classes)

Two lens states:
  - "preSA"    : lens on h_input[L]            (matches Stage-5 patch site)
  - "postblk"  : lens on h_output[L] = h_input[L+1] (ASD's lens)

For L = n_layers-1 (= 27), h_output[L] is captured via a separate
register_forward_hook so it isn't missed.

Reuses Stage 3.2 per-clip masks (`stage3_2/per_clip_tokens/*.npz`):
p_llm, p_prop, mds_cell, video_pos, audio_pos, S. The forward pass here
is required only to (re)capture hidden states; sink identity and cell
assignments come from the dump for provable convention match.

D_sink / τ / RMS gate / MDS thresholds are inherited unchanged from
Stage 3.1; this script never touches them.

Outputs (under --output_dir):
  logit_lens_by_cell_layer.csv
  per_clip_lens/<clip_stem>.npz   (per-sink top-K ids/probs + entropy +
                                   cell + pop tags, for the examples doc)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from stage3_1_modality_detection_score import (  # noqa: E402
    _resolve_thinker_cfg,
)
from utils import (  # noqa: E402
    build_conversation, load_omni, prepare_inputs, thinker_layers,
)

PROMPT_AV = "Describe what you see and hear in detail."
TOP_K = 10

DEFAULT_DUMP_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_QA_JSON = _REPO / "data/VGGSounder/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_3"

CELL_UNI_V, CELL_UNI_A, CELL_CROSS = +1, -1, 0
CELL_NAME = {CELL_UNI_V: "uni_video", CELL_UNI_A: "uni_audio", CELL_CROSS: "cross"}

# ----------------------------------------------------------------------
# Token classification helpers (matches Stage 2.7's convention)
# ----------------------------------------------------------------------

_STRUCT_LITERALS = {
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|audio_start|>", "<|audio_end|>", "<|video_pad|>",
    "<|audio_pad|>", "<|image_pad|>", "<|object_ref_start|>",
    "<|object_ref_end|>",
}
_PUNCT_RE = re.compile(r"^[\s\W_]+$")
_DIGIT_RE = re.compile(r"^[\s\d.,_-]+$")
_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")  # any 2+ alpha letters → "content"


def is_content_token(decoded: str, raw: str) -> bool:
    """Content = alphabetic word piece (>=2 alpha letters), NOT a special
    token, NOT pure punctuation/whitespace, NOT pure digits. Function
    words DO count as content here (we measure 'a real word leaked out',
    not 'a topical noun')."""
    s = decoded.strip()
    if not s:
        return False
    if raw in _STRUCT_LITERALS or raw.startswith("<|") or raw.endswith("|>"):
        return False
    if _PUNCT_RE.match(s) or _DIGIT_RE.match(s):
        return False
    return bool(_ALPHA_RE.search(s))


# ----------------------------------------------------------------------
# Ground-truth labels
# ----------------------------------------------------------------------

# words to drop from label phrases so e.g. "playing drum kit" matches
# "drum" but doesn't false-match every clip on "playing".
_LABEL_STOP = {
    "playing", "sound", "sounds", "noise", "noises", "music", "audio",
    "a", "an", "the", "of", "with", "and", "or", "in", "on", "at",
}
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def load_label_lookup(qa_path: Path) -> dict:
    """video_id (basename .mp4) -> set of GT keyword strings (lowercase,
    >=3 alpha letters, after dropping common helper words)."""
    raw = json.loads(Path(qa_path).read_text())
    lookup = {}
    for entry in raw:
        vid = entry.get("video_id")
        if not vid:
            continue
        labels = entry.get("label") or []
        if isinstance(labels, str):
            labels = [labels]
        words = set()
        for lab in labels:
            for w in _WORD_RE.findall(str(lab).lower()):
                if w in _LABEL_STOP:
                    continue
                words.add(w)
        lookup.setdefault(vid, set()).update(words)
    return lookup


# ----------------------------------------------------------------------
# Vocab cache: pre-decode the entire vocab once, mark which ids are content
# ----------------------------------------------------------------------

def build_vocab_cache(tokenizer, vocab_size: int):
    """Returns:
        raw_arr      : list[str]   raw BPE for id (length V)
        decoded_arr  : list[str]   decoded for id (length V), lowercased
        is_content   : np.ndarray  bool (V,)
    Pre-decoded once; downstream lookups are O(1)."""
    raw_arr = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    decoded_arr = []
    is_content = np.zeros(vocab_size, dtype=bool)
    for i in range(vocab_size):
        try:
            d = tokenizer.decode([i], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        except Exception:
            d = ""
        decoded_arr.append(d)
        is_content[i] = is_content_token(d, raw_arr[i])
    return raw_arr, decoded_arr, is_content


# ----------------------------------------------------------------------
# Per-clip forward — capture h_input[L] for all L, plus h_output[27]
# ----------------------------------------------------------------------

def forward_capture_hidden(model, processor, clip_path, n_layers, layers):
    """Run a prompt forward with no generation; capture h_input[L] for
    L=0..n_layers-1 (full S × D) via pre-hooks, plus h_output[n_layers-1]
    via a post-hook on the last layer. Returns:
       per_layer_in[L] : tensor (S, D) on the layer's device, L=0..n_layers-1
       last_out        : tensor (S, D) on the last layer's device
       use_aiv, S      : forward bookkeeping
    Or (None, None, None, None) on failure."""
    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(
            processor, conv, "av", model.device, model.dtype)
    except Exception:
        return None, None, None, None
    S = int(inputs["input_ids"].shape[1])

    per_layer_in = [None] * n_layers
    last_out = [None]                          # boxed so closure can mutate

    def make_pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:               # prompt forward only
                per_layer_in[L_idx] = hs[0].detach()
        return _h

    def last_post_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] > 1:
            last_out[0] = hs[0].detach()
        return out

    handles = []
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(make_pre_hook(L)))
    handles.append(layers[-1].register_forward_hook(last_post_hook))

    try:
        with torch.inference_mode():
            model.thinker(**inputs,
                          use_audio_in_video=use_aiv,
                          output_attentions=False,
                          return_dict=True,
                          use_cache=False)
    except Exception:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return None, None, None, None
    finally:
        for h in handles: h.remove()

    if any(h is None for h in per_layer_in) or last_out[0] is None:
        torch.cuda.empty_cache()
        return None, None, None, None

    return per_layer_in, last_out[0], use_aiv, S


# ----------------------------------------------------------------------
# Logit lens projection
# ----------------------------------------------------------------------

def lens_topk_entropy(h_chunk: torch.Tensor, final_norm, lm_head, K: int):
    """h_chunk: (N, D) on any device. Move to final_norm/lm_head device,
    apply, return top-K ids (int64 N×K), top-K probs (fp32 N×K), and
    full-distribution entropy (fp32 N).
    Heavy logits tensor freed before return."""
    if h_chunk.shape[0] == 0:
        return (np.zeros((0, K), dtype=np.int64),
                np.zeros((0, K), dtype=np.float32),
                np.zeros((0,),    dtype=np.float32))
    dev = final_norm.weight.device
    h = h_chunk.to(dev)
    with torch.no_grad():
        h_normed = final_norm(h)
        logits = lm_head(h_normed).float()             # (N, V)
        probs  = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(K, dim=-1)
        logp   = torch.log_softmax(logits, dim=-1)
        H      = -(probs * logp).sum(dim=-1)
    return (topk_idx.cpu().numpy().astype(np.int64),
            topk_probs.cpu().numpy().astype(np.float32),
            H.cpu().numpy().astype(np.float32))


# ----------------------------------------------------------------------
# Per-clip aggregation
# ----------------------------------------------------------------------

def process_clip(clip_path, dump_path, label_words, model, processor,
                 layers, n_layers, final_norm, lm_head,
                 raw_arr, decoded_arr, is_content_arr, K,
                 per_clip_out_dir: Path):
    """Returns list of (pop, cell, layer, lens_state, n_total, n_content,
    n_on_topic, sum_entropy) records, or None on failure."""
    dump = np.load(dump_path, allow_pickle=True)
    p_llm     = dump["p_llm"]                    # (n_layers, S) bool
    p_prop    = dump["p_prop"]                   # (S,)        bool
    mds_cell  = dump["mds_cell"]                 # (n_layers, S) int8
    dump_S    = int(dump["S"])

    per_layer_in, last_out, use_aiv, S = forward_capture_hidden(
        model, processor, clip_path, n_layers, layers)
    if per_layer_in is None:
        return None
    if S != dump_S:
        # Shouldn't happen with same seed/inputs, but guard.
        print(f"  [warn] S mismatch for {clip_path.name}: dump={dump_S} fwd={S}; skipping")
        return None

    # Pre-compute on-topic mask for the full vocab × this clip
    if label_words:
        on_topic_vocab = np.zeros(len(decoded_arr), dtype=bool)
        # cheap: precompute a single lowercase + strip array isn't needed;
        # iterate once over the vocab and substring-check.
        # 152K × few words is ~0.3s in pure python — acceptable per clip.
        # Strip BPE prefix marker 'Ġ' to space, then lower.
        for vid, d in enumerate(decoded_arr):
            ds = d.strip().lower()
            if len(ds) < 3:
                continue
            for w in label_words:
                if w in ds or ds in w:
                    on_topic_vocab[vid] = True
                    break
    else:
        on_topic_vocab = np.zeros(len(decoded_arr), dtype=bool)

    records = []
    per_clip_examples = []   # for the npz dump used by the examples MD

    for L in range(n_layers):
        # Sinks at this layer (union of populations); we lens each only once
        # then attribute counts to BOTH pops if a position is in both.
        is_prop = p_prop                         # (S,)
        is_llm  = p_llm[L]                        # (S,)
        any_sink = is_prop | is_llm
        idx = np.where(any_sink)[0]
        if idx.size == 0:
            continue
        cells_here = mds_cell[L, idx]            # (n,) int8

        # ---- pre-SA hidden state at L ----
        h_pre = per_layer_in[L][idx]             # (n, D)
        topk_pre, topkp_pre, H_pre = lens_topk_entropy(
            h_pre, final_norm, lm_head, K)
        top1_pre = topk_pre[:, 0]

        # ---- post-block at L = h_input[L+1] (or last_out for the tail) ----
        if L < n_layers - 1:
            h_post = per_layer_in[L + 1][idx]
        else:
            h_post = last_out[idx]
        topk_post, topkp_post, H_post = lens_topk_entropy(
            h_post, final_norm, lm_head, K)
        top1_post = topk_post[:, 0]

        # ---- counts per (pop, cell, lens) ----
        in_prop_mask = is_prop[idx]
        in_llm_mask  = is_llm[idx]
        for cell_val in (CELL_UNI_V, CELL_UNI_A, CELL_CROSS):
            sel_cell = cells_here == cell_val
            for pop_name, sel_pop in (("prop", in_prop_mask),
                                       ("llm",  in_llm_mask)):
                sel = sel_cell & sel_pop
                n = int(sel.sum())
                if n == 0:
                    continue
                # pre
                c1 = int(is_content_arr[top1_pre[sel]].sum())
                onp = int(np.any(on_topic_vocab[topk_pre[sel]], axis=1).sum())
                sH = float(H_pre[sel].sum())
                records.append(("preSA", pop_name, CELL_NAME[cell_val], L,
                                 n, c1, onp, sH))
                # post
                c1b = int(is_content_arr[top1_post[sel]].sum())
                onpb = int(np.any(on_topic_vocab[topk_post[sel]], axis=1).sum())
                sHb = float(H_post[sel].sum())
                records.append(("postblk", pop_name, CELL_NAME[cell_val], L,
                                 n, c1b, onpb, sHb))

        # Per-clip example dump (only at candidate layers, to keep size small)
        if L in (11, 13, 14, 16, 20):
            per_clip_examples.append(dict(
                layer=int(L),
                positions=idx.astype(np.int64),
                in_prop=in_prop_mask.copy(),
                in_llm=in_llm_mask.copy(),
                cell=cells_here.astype(np.int8),
                topk_pre=topk_pre.astype(np.int64),
                topk_pre_prob=topkp_pre.astype(np.float32),
                H_pre=H_pre.astype(np.float32),
                topk_post=topk_post.astype(np.int64),
                topk_post_prob=topkp_post.astype(np.float32),
                H_post=H_post.astype(np.float32),
            ))

    # Save per-clip examples
    out_npz = per_clip_out_dir / f"{clip_path.stem}.npz"
    if per_clip_examples:
        np.savez_compressed(
            out_npz,
            clip=np.array(clip_path.name),
            label_words=np.array(sorted(label_words) if label_words else [],
                                 dtype=object),
            **{f"L{e['layer']}_positions":       e["positions"]      for e in per_clip_examples},
            **{f"L{e['layer']}_in_prop":         e["in_prop"]        for e in per_clip_examples},
            **{f"L{e['layer']}_in_llm":          e["in_llm"]         for e in per_clip_examples},
            **{f"L{e['layer']}_cell":            e["cell"]           for e in per_clip_examples},
            **{f"L{e['layer']}_topk_pre":        e["topk_pre"]       for e in per_clip_examples},
            **{f"L{e['layer']}_topk_pre_prob":   e["topk_pre_prob"]  for e in per_clip_examples},
            **{f"L{e['layer']}_H_pre":           e["H_pre"]          for e in per_clip_examples},
            **{f"L{e['layer']}_topk_post":       e["topk_post"]      for e in per_clip_examples},
            **{f"L{e['layer']}_topk_post_prob":  e["topk_post_prob"] for e in per_clip_examples},
            **{f"L{e['layer']}_H_post":          e["H_post"]         for e in per_clip_examples},
        )

    # free GPU memory from captured states
    del per_layer_in, last_out
    torch.cuda.empty_cache()
    return records


def write_csv(rows: list, out_csv: Path):
    """rows: list of (lens, pop, cell, layer, n, c1, onp, sumH)."""
    import pandas as pd
    df = pd.DataFrame(rows, columns=["lens_state", "pop", "cell", "layer",
                                       "n", "n_top1_content",
                                       "n_topk_on_topic", "sum_entropy"])
    df = (df.groupby(["lens_state", "pop", "cell", "layer"], as_index=False)
            .agg(n=("n", "sum"),
                 n_top1_content=("n_top1_content", "sum"),
                 n_topk_on_topic=("n_topk_on_topic", "sum"),
                 sum_entropy=("sum_entropy", "sum")))
    df["content_rate"]  = df["n_top1_content"] / df["n"].clip(lower=1)
    df["on_topic_rate"] = df["n_topk_on_topic"] / df["n"].clip(lower=1)
    df["mean_entropy"]  = df["sum_entropy"]    / df["n"].clip(lower=1)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows)")


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_clip_dir = out_dir / "per_clip_lens"; per_clip_dir.mkdir(exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    final_norm = model.thinker.model.norm
    lm_head    = model.thinker.lm_head
    vocab_size = int(lm_head.weight.shape[0])
    print(f"  n_layers={n_layers}, vocab_size={vocab_size}, "
          f"final_norm.dev={final_norm.weight.device}, "
          f"lm_head.dev={lm_head.weight.device}")

    print("\nBuilding vocab cache (one-time decode pass) ...")
    raw_arr, decoded_arr, is_content_arr = build_vocab_cache(
        processor.tokenizer, vocab_size)
    print(f"  content tokens in vocab: {int(is_content_arr.sum())}/{vocab_size} "
          f"({100*is_content_arr.mean():.1f}%)")

    print(f"\nLoading VGGSounder labels from {args.qa_json}")
    label_lookup = load_label_lookup(Path(args.qa_json))
    print(f"  {len(label_lookup)} clips with labels")

    # Walk dump dir
    dump_dir = Path(args.dump_dir)
    video_dir = Path(args.video_dir)
    dump_files = sorted(dump_dir.glob("*.npz"))
    if not dump_files:
        raise SystemExit(f"no .npz under {dump_dir}")
    print(f"\nProcessing {len(dump_files)} clips ...\n")

    rows = []
    failures = {}
    for dump_path in tqdm(dump_files, desc="clips"):
        clip_name = dump_path.stem + ".mp4"
        clip_path = video_dir / clip_name
        if not clip_path.exists():
            failures["missing_video"] = failures.get("missing_video", 0) + 1
            continue
        label_words = label_lookup.get(clip_name, set())
        recs = process_clip(clip_path, dump_path, label_words,
                              model, processor, layers, n_layers,
                              final_norm, lm_head,
                              raw_arr, decoded_arr, is_content_arr, TOP_K,
                              per_clip_dir)
        if recs is None:
            failures["fwd_fail"] = failures.get("fwd_fail", 0) + 1
            continue
        rows.extend(recs)

    if failures:
        print(f"  failures: {failures}")
    if not rows:
        raise SystemExit("no records collected")

    csv_path = out_dir / "logit_lens_by_cell_layer.csv"
    write_csv(rows, csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",  default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--dump_dir",   default=str(DEFAULT_DUMP_DIR))
    p.add_argument("--qa_json",    default=str(DEFAULT_QA_JSON))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
