"""librispeech_attribution_eval.py

Stage 1 (eval) for LibriSpeech hallucination-head attribution.

Unlike eval.py (closed-vocab entity matching), LibriSpeech is open-vocab ASR,
so "hallucination" is defined by reference alignment: greedy-decode the
transcription, word-align the model hypothesis to the ground-truth transcript
(Levenshtein backtrace), and label each generated hypothesis word:

    substitution / insertion error -> hallucinated   (model "made it up")
    correct match                  -> non-hallucinated

The output `sampled_entities.json` matches the schema consumed by
`identify_halluc_head.py` (Stage 2 per-head zero-ablation), so the existing
attribution machinery runs unchanged with `--modal_type a`.

Entity strings are the RAW generated word-spans (original casing) because the
attribution matcher does `next_token.strip() in entity` (case-sensitive
substring). task is set to "LibriSpeech Captioning" so the attribution script
uses max_new_tokens=2048 and appends NO describe suffix (not in
DESCRIBE_SUFFIX_BY_TASK).

Generation here mirrors identify_halluc_head.py exactly (greedy, do_sample
False, pad=eos) so the saved `generated_caption` reproduces under Stage 2.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 qwen_venv/bin/python \
        method/qwen2_5_omni/librispeech_attribution_eval.py \
        --split test-other --target_n 200
"""
import argparse
import json
import os
import re
import string
import sys
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

TRANSCRIBE_PROMPT = (
    "Transcribe the spoken English words in this audio, word for word. "
    "Output ONLY the transcription as plain text. Do not add any punctuation, "
    "quotation marks, capitalization beyond normal words, commentary, "
    "speaker labels, or descriptions of non-speech sounds. "
    "Transcribe only the speech and nothing else."
)
TASK = "LibriSpeech Captioning"  # contains "Captioning" -> Stage2 max_new_tokens=2048

_PUNCT = str.maketrans("", "", string.punctuation)


def norm_word(w: str) -> str:
    return w.upper().translate(_PUNCT)


def align_ops(ref, hyp):
    """Levenshtein backtrace over word lists. Returns (ops, edits) where `ops`
    is the same length as `hyp` ('match'|'sub'|'ins' per hypothesis word) and
    `edits` is the total edit distance S+D+I (for the per-utterance WER)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                           dp[i][j - 1] + 1,      # insertion
                           dp[i - 1][j - 1] + cost)
    # Backtrace
    i, j = n, m
    ops = [None] * m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            ops[j - 1] = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops[j - 1] = "ins"
            j -= 1
        else:  # deletion (no hyp word consumed)
            i -= 1
    return ops, dp[n][m]


def load_manifest(data_dir: Path):
    items = []
    for trans in sorted(data_dir.rglob("*.trans.txt")):
        with open(trans) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uid, _, text = line.partition(" ")
                flac = trans.parent / f"{uid}.flac"
                if flac.exists():
                    rel = flac.relative_to(data_dir)
                    items.append((uid, str(rel), text))
    items.sort(key=lambda r: r[0])
    return items


def generate_caption(model, processor, flac_path, max_new_tokens):
    """Mirror identify_halluc_head.py generation so the text reproduces."""
    tokenizer = processor.tokenizer
    conv = utils.build_conversation(flac_path, TRANSCRIBE_PROMPT, "a")
    inputs, use_aiv = utils.prepare_inputs(
        processor, conv, "a", model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(
            **inputs, use_audio_in_video=use_aiv, return_audio=False,
            do_sample=False, thinker_max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq = utils._extract_sequences(out)
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(
        seq[:, prompt_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    return utils.trim_chat_artifacts(text)


def main(args):
    data_dir = _REPO / "data/LibriSpeech" / args.split
    out_dir = _REPO / "results/qwen3_omni" / f"LibriSpeech_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    print(f"{args.split}: {len(manifest)} utterances available", flush=True)

    model, processor = utils.load_omni(args.model_path, device_map=args.device_map)
    tokenizer = processor.tokenizer

    def tok_ids(words):
        ids = []
        for w in words:
            ids.extend(tokenizer.encode(" " + w, add_special_tokens=False))
        return ids

    # Pass 1: transcribe every utterance, label words, compute per-utt WER.
    scan = manifest if args.scan_limit is None else manifest[: args.scan_limit]
    candidates = []
    for uid, rel, gt in tqdm(scan, desc="transcribe"):
        flac = str(data_dir / rel)
        cap = generate_caption(model, processor, flac, args.max_new_tokens)
        raw_words = cap.split()
        if not raw_words:
            continue
        hyp_norm = [norm_word(w) for w in raw_words]
        ref_norm = [norm_word(w) for w in gt.split()]
        ops, edits = align_ops(ref_norm, hyp_norm)
        wer = edits / len(ref_norm) if ref_norm else 0.0

        hal, non_hal = [], []
        for raw, nw, op in zip(raw_words, hyp_norm, ops):
            ent = raw.strip(string.punctuation) or raw  # entity = raw word (case kept)
            if not nw:  # pure punctuation token, never a target
                continue
            (hal if op in ("sub", "ins") else non_hal).append(ent)
        # Dedup; a word both correct and erroneous in one utt -> hallucinated.
        hal = list(dict.fromkeys(hal))
        non_hal = [w for w in dict.fromkeys(non_hal) if w not in hal]

        if args.require_hal and not hal:
            continue  # WER>0 from deletions only -> no hyp token to attribute

        candidates.append({
            "question_id": str(uuid.uuid5(uuid.NAMESPACE_URL, uid)),
            "video": rel,
            "task": TASK,
            "question": TRANSCRIBE_PROMPT,
            "answer": gt,
            "gt_entities": [norm_word(w) for w in gt.split()],
            "generated_caption": cap,
            "wer": round(wer, 4),
            "hallucinated_tokens": tok_ids(hal),
            "non_hallucinated_tokens": tok_ids(non_hal),
            "hallucinated_entities": hal,
            "non_hallucinated_entities": non_hal,
            "generated_entities": raw_words,
        })

    # Pass 2: keep the highest-WER utterances (strongest hallucination signal).
    candidates.sort(key=lambda r: r["wer"], reverse=True)
    records = candidates[: args.target_n]

    # Dump ALL scored candidates so the 300-set can be re-selected/filtered
    # later (e.g. drop refusals / repetition loops) without re-running on GPU.
    with open(out_dir / "all_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

    out_json = out_dir / "sampled_entities.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    n_hal_tok = sum(len(r["hallucinated_entities"]) for r in records)
    n_non = sum(len(r["non_hallucinated_entities"]) for r in records)
    wers = [r["wer"] for r in records]
    print(f"\nscanned {len(scan)} utts -> {len(candidates)} with >=1 hallucination"
          f" -> kept top {len(records)} by WER", flush=True)
    if wers:
        print(f"selected WER range: {min(wers):.3f}-{max(wers):.3f} "
              f"(median {sorted(wers)[len(wers)//2]:.3f})", flush=True)
    print(f"hallucinated entities: {n_hal_tok} | non-hallucinated: {n_non}", flush=True)
    print(f"WROTE {out_json}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test-other")
    p.add_argument("--model_path", default="/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--target_n", type=int, default=300,
                   help="Keep this many highest-WER utterances.")
    p.add_argument("--scan_limit", type=int, default=None,
                   help="Cap how many utterances to transcribe (default: all).")
    p.add_argument("--require_hal", action="store_true", default=True,
                   help="Only keep utts with >=1 hallucinated word.")
    p.add_argument("--no_require_hal", dest="require_hal", action="store_false")
    main(p.parse_args())
