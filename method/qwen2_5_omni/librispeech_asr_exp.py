"""librispeech_asr_exp.py

Qwen2.5-Omni speech-only transcription on LibriSpeech test-clean.

Loads the thinker (audio encoder + LLM) via utils.load_omni and prompts the
model to transcribe ONLY the spoken words — no punctuation, no commentary, no
non-speech sound description — so the output is directly comparable to the
LibriSpeech ground-truth transcripts (uppercase words, no punctuation).

Reports word error rate (WER) with LibriSpeech-style normalization
(uppercase, strip punctuation, collapse whitespace). WER is computed inline
(word-level Levenshtein) so no jiwer dependency is needed.

Data: data/LibriSpeech/test-clean/<spk>/<chapter>/{*.flac, *.trans.txt}
(downloaded from https://www.openslr.org/resources/12/test-clean.tar.gz).

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 qwen_venv/bin/python \
        method/qwen2_5_omni/librispeech_asr_exp.py
    # quick prompt sanity:
    ... librispeech_asr_exp.py --limit 50
"""
import argparse
import csv
import os
import re
import string
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _REPO / "data/LibriSpeech/test-clean"
_DEFAULT_OUT = _REPO / "results/qwen2_5_omni/librispeech_asr"

# Force the model to emit only the spoken words, in a form that matches the
# LibriSpeech ground truth (plain words, no punctuation, no extra text).
TRANSCRIBE_PROMPT = (
    "Transcribe the spoken English words in this audio, word for word. "
    "Output ONLY the transcription as plain text. Do not add any punctuation, "
    "quotation marks, capitalization beyond normal words, commentary, "
    "speaker labels, or descriptions of non-speech sounds. "
    "Transcribe only the speech and nothing else."
)

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """LibriSpeech-style: uppercase, drop punctuation, collapse whitespace."""
    text = text.upper().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def wer_counts(ref_words, hyp_words):
    """Levenshtein edit distance (S+D+I) between two word lists."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return m, 0  # all insertions; ref length 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n


def load_manifest(data_dir: Path):
    """Return [(utt_id, flac_path, gt_text), ...] sorted by utt_id."""
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
                    items.append((uid, flac, text))
    items.sort(key=lambda r: r[0])
    return items


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"LibriSpeech test-clean: {len(manifest)} utterances", flush=True)
    print(f"PROMPT: {TRANSCRIBE_PROMPT}", flush=True)

    model, processor = utils.load_omni(args.model_path, device_map=args.device_map)

    csv_path = out_dir / "librispeech_test_clean_transcriptions.csv"
    tot_edits = tot_ref = 0
    t0 = time.time()
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["utt_id", "n_ref_words", "edits", "wer", "gt", "hyp_raw", "hyp_norm"])
        for i, (uid, flac, gt) in enumerate(manifest):
            conv = utils.build_conversation(str(flac), TRANSCRIBE_PROMPT, "a")
            raw = utils.omni_infer(model, processor, conv, "a",
                                    max_new_tokens=args.max_new_tokens)
            raw = utils.trim_chat_artifacts(raw)
            ref_n = normalize(gt).split()
            hyp_n = normalize(raw).split()
            edits, nref = wer_counts(ref_n, hyp_n)
            tot_edits += edits
            tot_ref += nref
            uwer = edits / nref if nref else 0.0
            w.writerow([uid, nref, edits, f"{uwer:.4f}", normalize(gt),
                        raw.replace("\n", " "), " ".join(hyp_n)])
            if (i + 1) % 25 == 0 or i + 1 == len(manifest):
                run_wer = tot_edits / tot_ref if tot_ref else 0.0
                dt = time.time() - t0
                print(f"[{i+1}/{len(manifest)}] running WER={run_wer*100:.2f}% "
                      f"({dt/60:.1f} min, {dt/(i+1):.2f}s/utt)", flush=True)
                fcsv.flush()

    final_wer = tot_edits / tot_ref if tot_ref else 0.0
    summary = (
        f"# LibriSpeech test-clean — Qwen2.5-Omni speech-only transcription\n\n"
        f"- utterances: {len(manifest)}\n"
        f"- total reference words: {tot_ref}\n"
        f"- total edits (S+D+I): {tot_edits}\n"
        f"- **WER: {final_wer*100:.2f}%**\n"
        f"- prompt: {TRANSCRIBE_PROMPT!r}\n"
        f"- per-utterance CSV: {csv_path.name}\n"
    )
    (out_dir / "librispeech_summary.md").write_text(summary)
    print("\n" + summary, flush=True)
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(_DEFAULT_DATA))
    p.add_argument("--output_dir", default=str(_DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args())
