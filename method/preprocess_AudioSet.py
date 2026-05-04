"""
preprocess_AudioSet.py

Downloads the AudioSet balanced-train and eval splits from HuggingFace
(agkphysics/AudioSet), extracts audio as WAV files, and writes a QA.json
compatible with the AVHBench/VGGSound format.

Steps:
  1. Download parquet files for each split one at a time.
  2. Keep only rows with >= --min_labels labels, then sample --n_samples total.
  3. Decode embedded FLAC bytes and write a WAV file per row.
  4. Write <out>/QA.json.

QA.json entry:
    {
        "video_id":    "<youtube_id>.wav",
        "task":        "Audio Captioning",
        "text":        "Describe what you hear.",
        "label":       "<space-separated human labels>",
        "question_id": "<uuid>",
        "split":       "bal_train" | "eval"
    }

Usage:
    python method/preprocess_AudioSet.py
    python method/preprocess_AudioSet.py --n_samples 3000 --min_labels 3
    python method/preprocess_AudioSet.py --splits eval --out data/AudioSet
    python method/preprocess_AudioSet.py --resume   # skip already-written WAVs

Dependencies (add to venv if missing):
    pip install pyarrow soundfile huggingface_hub pandas tqdm
"""

import argparse
import io
import json
import os
import uuid

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "agkphysics/AudioSet"
QUESTION = "Describe what you hear."
TASK = "Audio Captioning"

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


def read_parquet(repo_file: str) -> pd.DataFrame:
    local = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type="dataset")
    return pq.read_table(local).to_pandas()


def decode_audio_to_wav(audio_bytes: bytes, out_path: str) -> None:
    buf = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf)
    sf.write(out_path, data, samplerate, subtype="PCM_16")


def to_label_list(x: object) -> list[str]:
    """Coerce any sequence type pyarrow/pandas may produce into a plain list."""
    if x is None:
        return []
    try:
        return list(x)  # type: ignore[arg-type]
    except TypeError:
        return [str(x)]


def filter_and_sample(
    df: pd.DataFrame, min_labels: int, n_samples: int, seed: int = 42
) -> pd.DataFrame:
    df = df[df["human_labels"].apply(lambda x: len(to_label_list(x)) >= min_labels)]
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
    return df.reset_index(drop=True)  # type: ignore[return-value]


def process_split(
    split: str,
    audios_dir: str,
    min_labels: int,
    n_samples: int,
    resume: bool,
) -> list[dict]:
    parquet_files = list_parquet_files(split)
    print(f"\n[{split}] {len(parquet_files)} parquet files")

    all_rows: list[pd.DataFrame] = []
    for repo_file in tqdm(parquet_files, desc=f"Downloading {split} parquets"):
        all_rows.append(read_parquet(repo_file))

    df = pd.concat(all_rows, ignore_index=True)
    print(f"[{split}] Total rows before filter: {len(df)}")

    df = filter_and_sample(df, min_labels=min_labels, n_samples=n_samples)
    print(f"[{split}] After filter (>={min_labels} labels) + sample: {len(df)} rows")

    qa_entries: list[dict] = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {split} WAVs"):
        video_id = str(row["video_id"])
        wav_name = f"{video_id}.wav"
        wav_path = os.path.join(audios_dir, wav_name)

        if resume and os.path.exists(wav_path):
            skipped += 1
        else:
            audio_field = row["audio"]
            audio_bytes: bytes = (
                audio_field["bytes"]
                if isinstance(audio_field, dict)
                else bytes(audio_field)
            )
            if not audio_bytes:
                continue
            try:
                decode_audio_to_wav(audio_bytes, wav_path)
            except Exception as e:
                print(f"  Warning: failed to decode {video_id}: {e}")
                continue

        label_str = " ".join(str(l) for l in to_label_list(row["human_labels"]))

        qa_entries.append(
            {
                "video_id": wav_name,
                "task": TASK,
                "text": QUESTION,
                "label": label_str,
                "question_id": str(uuid.uuid4()),
                "split": split,
            }
        )

    if skipped:
        print(f"[{split}] Skipped {skipped} already-existing WAVs (--resume)")
    return qa_entries


def main(args: argparse.Namespace) -> None:
    audios_dir = os.path.join(args.out, "audios")
    os.makedirs(audios_dir, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    for s in splits:
        if s not in SPLIT_PREFIX:
            raise ValueError(f"Unknown split '{s}'. Choose from: {list(SPLIT_PREFIX)}")

    all_qa: list[dict] = []
    for split in splits:
        entries = process_split(
            split=split,
            audios_dir=audios_dir,
            min_labels=args.min_labels,
            n_samples=args.n_samples,
            resume=args.resume,
        )
        all_qa.extend(entries)

    qa_path = os.path.join(args.out, "QA.json")
    if args.resume and os.path.exists(qa_path):
        with open(qa_path) as f:
            existing: list[dict] = json.load(f)
        existing_ids = {e["video_id"] for e in existing}
        new_entries = [e for e in all_qa if e["video_id"] not in existing_ids]
        all_qa = existing + new_entries
        print(f"Merged {len(new_entries)} new entries with {len(existing)} existing.")

    with open(qa_path, "w") as f:
        json.dump(all_qa, f, indent=2)
    print(f"\nWrote {len(all_qa)} entries to {qa_path}")
    print(f"WAV files in: {audios_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/nobackup3/le/AV_Hallucination/data/AudioSet")
    parser.add_argument(
        "--splits", default="bal_train,eval", help="Comma-separated: bal_train, eval"
    )
    parser.add_argument(
        "--min_labels",
        type=int,
        default=5,
        help="Keep only rows with at least this many labels",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3000,
        help="Max rows to keep per split after filtering",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip WAVs that already exist on disk"
    )
    args = parser.parse_args()
    main(args)
