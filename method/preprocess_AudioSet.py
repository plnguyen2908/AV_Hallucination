"""
preprocess_AudioSet.py

Downloads the AudioSet balanced-train and eval splits from HuggingFace
(agkphysics/AudioSet), extracts audio as WAV files, and writes a QA.json
in one of three formats selected via --variant:

  hallucination (default)
      Per-label binary probe.
      One QA entry per ground-truth label  →  label "Yes"
      Plus --n_negatives sampled distractor labels  →  label "No"
      text: "Does the <label> sound appear in the audio?"

  describe
      One open-ended QA entry per audio. The prompt embeds a small
      per-entry option list — the clip's GT labels plus
      --n_distractors_describe random distractor labels (shuffled
      together) — so the model is constrained to a closed but short
      vocabulary. Previous versions inlined the full ~516-label list,
      which blew up GPU memory during attention-bias analysis.
      text:  "Listen to the audio and describe what you hear. Choose
              only from the following AudioSet sound labels: <list>."
      label: list[str] — the ground-truth AudioSet labels for the clip.
      Also writes the full label vocabulary to
      <--out>/audioset_labels.txt (one label per line) for
      eval.py's output scanning.

  mcq
      One 4-option multiple-choice QA entry per GT label (so a clip
      with k GT labels yields k MCQ entries). The correct option is
      that single GT label, placed at a random letter; the other three
      options are random distractor labels sampled from labels the
      clip does NOT have (so no option is silently also correct).
      The A/B/C/D options are inlined into `text`; `label` is the
      correct letter.

All variants share the same schema so `eval.py` can read any of them
unchanged:
    {video_id, task, text, label, question_id, split}

Re-using clips across variants:
    Pass --reuse_clips_from <existing_QA.json> to restrict to the
    set of audio clips already produced by an earlier run. WAVs are
    not re-decoded when the file already exists.

Output filename is always QA.json. Use --out to send each variant to
its own folder so they don't overwrite each other.

Usage:
    python method/preprocess_AudioSet.py                              # hallucination
    python method/preprocess_AudioSet.py --variant describe \
        --out data/AudioSet_describe \
        --reuse_clips_from data/AudioSet/QA.json
    python method/preprocess_AudioSet.py --variant mcq \
        --out data/AudioSet_mcq \
        --reuse_clips_from data/AudioSet/QA.json

Dependencies:
    pip install pyarrow soundfile huggingface_hub pandas tqdm
"""

import argparse
import io
import json
import os
import random
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent  # method/ -> repo root

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "agkphysics/AudioSet"
TASK_HALLUC = "Video-driven Audio Hallucination"
TASK_DESCRIBE = "AudioSet Captioning"
TASK_MCQ = "AudioSet Multiple-Choice"
SEED = 42

DEFAULT_OUT_NAME = "QA.json"

# Schema shared by all variants. eval.py reads these fields by name:
#   video_id, task, text, label, question_id, split
# Keeping every variant on the same schema lets the existing eval.py
# pipeline consume any variant without changes.

# Labels from the AudioSet "Acoustic environment" ontology category.
# These describe recording conditions rather than sound events, so they are
# excluded from both positive and negative QA entries.
# Source: https://research.google.com/audioset/ontology/acoustic_environment_1.html
ACOUSTIC_ENV_LABELS: set[str] = {
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    "Outside, urban or manmade",
    "Outside, rural or natural",
    "Reverberation",
    "Echo",
}

SPLIT_PREFIX = {
    "bal_train": "data/bal_train/",
    "eval": "data/eval/",
}


def list_parquet_files(split: str) -> list[str]:
    prefix = SPLIT_PREFIX[split]
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(
        f for f in all_files if f.startswith(prefix) and f.endswith(".parquet")
    )


def read_parquet(repo_file: str) -> tuple[pd.DataFrame, str]:
    local = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type="dataset")
    return pq.read_table(local).to_pandas(), local


def decode_audio_to_wav(audio_bytes: bytes, out_path: str) -> None:
    buf = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf)
    sf.write(out_path, data, samplerate, subtype="PCM_16")


def to_label_list(x: object) -> list[str]:
    if x is None:
        return []
    try:
        return list(x)  # type: ignore[arg-type]
    except TypeError:
        return [str(x)]


def load_reuse_video_ids(path: str) -> set[str]:
    with open(path) as f:
        entries = json.load(f)
    ids: set[str] = set()
    for e in entries:
        vid = e["video_id"]
        if vid.endswith(".wav"):
            vid = vid[:-4]
        ids.add(vid)
    return ids


def build_entries_hallucination(
    wav_name: str,
    split: str,
    audio_labels: list[str],
    global_labels_list: list[str],
    n_negatives: int,
) -> list[dict]:
    entries: list[dict] = []
    for lbl in audio_labels:
        entries.append({
            "video_id": wav_name,
            "task": TASK_HALLUC,
            "text": f"Does the {lbl} sound appear in the audio?",
            "label": "Yes",
            "question_id": str(uuid.uuid4()),
            "split": split,
        })
    negative_pool = [l for l in global_labels_list if l not in audio_labels]
    negatives = random.sample(negative_pool, min(n_negatives, len(negative_pool)))
    for lbl in negatives:
        entries.append({
            "video_id": wav_name,
            "task": TASK_HALLUC,
            "text": f"Does the {lbl} sound appear in the audio?",
            "label": "No",
            "question_id": str(uuid.uuid4()),
            "split": split,
        })
    return entries


def build_entry_describe(
    wav_name: str,
    split: str,
    audio_labels: list[str],
    global_labels_list: list[str],
    n_distractors: int,
) -> dict:
    # Per-entry option list: GT labels + a small sample of distractors,
    # shuffled so position carries no signal. Inlining the full ~516-label
    # vocabulary made the prefill OOM on the attention-bias pass.
    distractor_pool = [l for l in global_labels_list if l not in audio_labels]
    n_take = min(n_distractors, len(distractor_pool))
    distractors = random.sample(distractor_pool, n_take)
    options = list(audio_labels) + distractors
    random.shuffle(options)
    return {
        "video_id": wav_name,
        "task": TASK_DESCRIBE,
        "text": (
            "Listen to the audio and describe what you hear. "
            "Choose only from the following AudioSet sound labels: "
            + ", ".join(options) + "."
        ),
        "label": audio_labels,
        "question_id": str(uuid.uuid4()),
        "split": split,
    }


def build_entries_mcq(
    wav_name: str,
    split: str,
    audio_labels: list[str],
    global_labels_list: list[str],
) -> list[dict]:
    """One MCQ entry per GT label: the correct option is that single label;
    the other three options are random distractors drawn from labels the clip
    does NOT have (so no option is silently also correct)."""
    entries: list[dict] = []
    distractor_pool = [l for l in global_labels_list if l not in audio_labels]
    if len(distractor_pool) < 3:
        print(
            f"  Warning: distractor pool ({len(distractor_pool)}) < 3 for "
            f"{wav_name}; skipping MCQ entries for this clip."
        )
        return entries

    for gt_lbl in audio_labels:
        distractors = random.sample(distractor_pool, 3)
        correct = random.choice(["A", "B", "C", "D"])
        opts: dict[str, str] = {}
        d_iter = iter(distractors)
        for L in ["A", "B", "C", "D"]:
            opts[L] = gt_lbl if L == correct else next(d_iter)

        entries.append({
            "video_id": wav_name,
            "task": TASK_MCQ,
            "text": (
                "Which sound do you hear in the audio?\n"
                f"A. {opts['A']}\n"
                f"B. {opts['B']}\n"
                f"C. {opts['C']}\n"
                f"D. {opts['D']}\n"
                "Answer with a single letter (A, B, C, or D)."
            ),
            "label": correct,
            "question_id": str(uuid.uuid4()),
            "split": split,
        })
    return entries


def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    audios_dir = os.path.join(args.out, "audios")
    os.makedirs(audios_dir, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    for s in splits:
        if s not in SPLIT_PREFIX:
            raise ValueError(f"Unknown split '{s}'. Choose from: {list(SPLIT_PREFIX)}")

    # --- Optional clip reuse: restrict to a fixed set of video_ids ---
    reuse_ids: set[str] | None = None
    if args.reuse_clips_from:
        if not os.path.exists(args.reuse_clips_from):
            raise FileNotFoundError(
                f"--reuse_clips_from path does not exist: {args.reuse_clips_from}"
            )
        reuse_ids = load_reuse_video_ids(args.reuse_clips_from)
        print(
            f"Reuse mode: restricting to {len(reuse_ids)} video_ids from "
            f"{args.reuse_clips_from}"
        )

    # --- Download all parquets and accumulate rows ---
    all_dfs: list[pd.DataFrame] = []

    for split in splits:
        parquet_files = list_parquet_files(split)
        print(f"\n[{split}] {len(parquet_files)} parquet files")
        for repo_file in tqdm(parquet_files, desc=f"Downloading {split} parquets"):
            df, _ = read_parquet(repo_file)
            df["_split"] = split
            all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")

    # --- Build global label set from all rows (excluding acoustic env labels) ---
    global_labels: set[str] = set()
    for labels in df_all["human_labels"]:
        global_labels.update(
            l for l in to_label_list(labels) if l not in ACOUSTIC_ENV_LABELS
        )
    global_labels_list = sorted(global_labels)
    print(f"Global label vocabulary: {len(global_labels_list)} unique labels (acoustic env excluded)")

    if args.variant == "describe":
        labels_txt = os.path.join(args.out, "audioset_labels.txt")
        with open(labels_txt, "w") as f:
            f.write("\n".join(global_labels_list) + "\n")
        print(f"Wrote {len(global_labels_list)} labels to {labels_txt}")

    # --- Filter rows ---
    if reuse_ids is not None:
        df_all = df_all[df_all["video_id"].astype(str).isin(reuse_ids)].reset_index(drop=True)
        print(f"After reuse-clips filter: {len(df_all)} rows")
    else:
        df_all = df_all[
            df_all["human_labels"].apply(lambda x: len(to_label_list(x)) >= args.min_labels)
        ].reset_index(drop=True)
        print(f"After min_labels>={args.min_labels} filter: {len(df_all)} rows")

    # --- Sample (or take all matched rows in reuse mode) ---
    if reuse_ids is not None:
        # Keep one row per video_id; preserve all matched clips.
        df_sampled = df_all.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
        print(f"Reuse mode: {len(df_sampled)} unique clips matched")
    else:
        n = min(args.n_audio_samples, len(df_all))
        df_sampled = df_all.sample(n=n, random_state=SEED).reset_index(drop=True)
        print(f"Sampled {n} unique audio clips")

    # --- Build QA entries ---
    qa_path = os.path.join(args.out, args.out_name or DEFAULT_OUT_NAME)
    existing_ids: set[str] = set()
    existing_entries: list[dict] = []
    if args.resume and os.path.exists(qa_path):
        with open(qa_path) as f:
            existing_entries = json.load(f)
        existing_ids = {e["video_id"] for e in existing_entries}
        print(f"Resuming: {len(existing_entries)} entries already in {qa_path}")

    qa_entries: list[dict] = []
    skipped = 0
    failed = 0
    wav_already_present = 0

    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Building QA"):
        video_id = str(row["video_id"])
        wav_name = f"{video_id}.wav"
        wav_path = os.path.join(audios_dir, wav_name)
        split = row["_split"]
        audio_labels = [
            l for l in to_label_list(row["human_labels"])
            if l not in ACOUSTIC_ENV_LABELS
        ]
        if not audio_labels:
            failed += 1
            continue

        if args.resume and wav_name in existing_ids:
            skipped += 1
            continue

        # Write WAV (or skip if already on disk — common in reuse mode).
        if os.path.exists(wav_path):
            wav_already_present += 1
        else:
            audio_field = row["audio"]
            audio_bytes: bytes = (
                audio_field["bytes"]
                if isinstance(audio_field, dict)
                else bytes(audio_field)
            )
            if not audio_bytes:
                failed += 1
                continue
            try:
                decode_audio_to_wav(audio_bytes, wav_path)
            except Exception as e:
                print(f"  Warning: failed to decode {video_id}: {e}")
                failed += 1
                continue

        # Dispatch to the requested variant.
        if args.variant == "hallucination":
            qa_entries.extend(
                build_entries_hallucination(
                    wav_name, split, audio_labels, global_labels_list, args.n_negatives
                )
            )
        elif args.variant == "describe":
            qa_entries.append(
                build_entry_describe(
                    wav_name, split, audio_labels, global_labels_list,
                    args.n_distractors_describe,
                )
            )
        elif args.variant == "mcq":
            qa_entries.extend(
                build_entries_mcq(wav_name, split, audio_labels, global_labels_list)
            )
        else:
            raise ValueError(f"Unknown variant: {args.variant}")

    if skipped:
        print(f"Skipped {skipped} already-existing WAVs (--resume)")
    if wav_already_present:
        print(f"Reused {wav_already_present} WAVs already on disk")
    if failed:
        print(f"Failed (no audio bytes / empty labels / decode error): {failed}")

    all_entries = existing_entries + qa_entries
    with open(qa_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nWrote {len(all_entries)} QA entries to {qa_path}")
    print(f"WAV files in: {audios_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_REPO / "data/AudioSet"))
    parser.add_argument(
        "--splits", default="bal_train,eval", help="Comma-separated: bal_train, eval"
    )
    parser.add_argument(
        "--variant",
        default="hallucination",
        choices=["hallucination", "describe", "mcq"],
        help="Which QA format to produce. See module docstring.",
    )
    parser.add_argument(
        "--reuse_clips_from",
        default=None,
        help=(
            "Path to an existing QA.json. If given, restricts processing to the "
            "video_ids in that file (skips re-decoding WAVs already on disk)."
        ),
    )
    parser.add_argument(
        "--out_name",
        default=None,
        help=(
            "Output JSON filename (under --out). Default: QA.json for every "
            "variant. Use --out to direct variants to different folders."
        ),
    )
    parser.add_argument(
        "--min_labels", type=int, default=5,
        help="Keep only audios with at least this many labels (ignored in reuse mode)",
    )
    parser.add_argument(
        "--n_audio_samples", type=int, default=500,
        help="Number of unique audio clips to sample (ignored in reuse mode)",
    )
    parser.add_argument(
        "--n_negatives", type=int, default=4,
        help="Number of negative (label=No) QA entries per audio (hallucination variant only)",
    )
    parser.add_argument(
        "--n_distractors_describe", type=int, default=10,
        help=(
            "Number of distractor labels added (alongside the GT labels) to "
            "each describe entry's prompt. Keeps prompt length short to avoid "
            "OOM at attention-analysis time. (describe variant only)"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
