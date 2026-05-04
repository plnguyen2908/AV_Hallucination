"""
preprocess_VGGSound.py

Downloads VGGSound from HuggingFace (Loie/VGGSound), keeps --per_label videos
per label, and writes a QA.json compatible with the AVCaps/AVHBench format.

Steps:
  1. snapshot_download the entire repo into <out>/raw/  (re-runs if < 20 tars present)
  2. Sample --per_label clips per label from vggsound.csv
  3. If enough videos already on disk per label, skip extraction
  4. Otherwise extract matching videos from each tar.gz into <out>/videos/
  5. Delete <out>/raw/ after all extraction is done
  6. Write <out>/QA.json

QA.json entry:
    {
        "video_id":    "<youtube_id>_<start_sec>.mp4",
        "task":        "AV Captioning",
        "text":        "Describe in detail what you see and hear.",
        "label":       "<label>",
        "question_id": "<uuid>",
        "split":       "train" | "test"
    }

Usage:
    python method/preprocess_VGGSound.py [--out data/VGGSound] [--per_label 100]
"""

import argparse
import json
import os
import shutil
import tarfile
import uuid
import warnings

import pandas as pd
from huggingface_hub import list_repo_files, snapshot_download
from tqdm import tqdm

REPO_ID = "Loie/VGGSound"
QUESTION = "Describe in detail what you hear."


def load_and_sample(csv_path: str, per_label: int) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path, header=None, names=["youtube_id", "start_sec", "label", "split"]
    )
    df = df.groupby(["youtube_id", "start_sec", "split"], as_index=False).agg(
        label=("label", lambda x: " ".join(x.drop_duplicates()))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sampled = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_label), random_state=42))
            .reset_index(drop=True)
        )
    assert isinstance(sampled, pd.DataFrame)
    n_labels = df.groupby("label").ngroups
    print(f"Sampled {len(sampled)} clips across {n_labels} labels")
    return sampled


def build_target_set(sampled: pd.DataFrame) -> dict:
    targets = {}
    for idx, row in sampled.iterrows():
        yt = row["youtube_id"]
        sec = int(row["start_sec"])
        targets[f"{yt}_{sec:06d}"] = idx
        targets[f"{yt}_{sec}"] = idx
    return targets


def extract_tar(tar_path: str, targets: dict, videos_dir: str, extracted: set):
    with tarfile.open(tar_path, "r:gz") as tf:
        while True:
            try:
                member = tf.next()
            except Exception as e:
                print(f"  Warning: corrupt entry in {os.path.basename(tar_path)}: {e}")
                break
            if member is None:
                break
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            stem, ext = os.path.splitext(basename)
            if ext.lower() != ".mp4" or stem not in targets:
                continue
            idx = targets[stem]
            if idx in extracted:
                continue
            member.name = basename
            try:
                tf.extract(member, path=videos_dir)
                extracted.add(idx)
            except Exception as e:
                print(f"  Warning: failed to extract {basename}: {e}")


def build_extracted_set(videos_dir: str, sampled: pd.DataFrame) -> set:
    stem_to_idx = {
        f"{row['youtube_id']}_{int(row['start_sec']):06d}": idx
        for idx, row in sampled.iterrows()
    }
    extracted: set = set()
    for fname in os.listdir(videos_dir):
        stem = os.path.splitext(fname)[0]
        if stem in stem_to_idx:
            extracted.add(stem_to_idx[stem])
    return extracted


def has_enough_per_label(
    videos_dir: str, sampled: pd.DataFrame, per_label: int
) -> bool:
    stem_to_label: dict = {
        f"{row['youtube_id']}_{int(row['start_sec']):06d}": str(row["label"])
        for _, row in sampled.iterrows()
    }
    counts: dict = {}
    for fname in os.listdir(videos_dir):
        label = stem_to_label.get(os.path.splitext(fname)[0])
        if label:
            counts[label] = counts.get(label, 0) + 1
    needed = sampled.groupby("label").size().to_dict()
    return all(counts.get(lbl, 0) >= min(n, per_label) for lbl, n in needed.items())


def write_qa_json(
    out_path: str, extracted: set, sampled: pd.DataFrame, videos_dir: str
):
    qa = []
    for idx in extracted:
        row = sampled.iloc[idx]
        yt = row["youtube_id"]
        sec = int(row["start_sec"])
        for fname in (f"{yt}_{sec:06d}.mp4", f"{yt}_{sec}.mp4"):
            if os.path.exists(os.path.join(videos_dir, fname)):
                qa.append(
                    {
                        "video_id": fname,
                        "task": "AV Captioning",
                        "text": QUESTION,
                        "label": row["label"],
                        "question_id": str(uuid.uuid4()),
                        "split": row["split"],
                    }
                )
                break
    with open(out_path, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"Wrote {len(qa)} entries to {out_path}")


def ensure_downloaded(raw_dir: str):
    expected_tars = sorted(
        f
        for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".tar.gz")
    )
    n_expected = len(expected_tars)
    present = (
        sum(1 for f in os.listdir(raw_dir) if f.endswith(".tar.gz"))
        if os.path.isdir(raw_dir)
        else 0
    )
    if present >= n_expected:
        print(f"All {n_expected} tars already in {raw_dir}, skipping download.")
        return
    print(f"Found {present}/{n_expected} tars. Downloading {REPO_ID} to {raw_dir} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=raw_dir,
        ignore_patterns=["src/*"],
    )
    print("Download complete.")


def main(args):
    raw_dir = os.path.join(args.out, "raw")
    videos_dir = os.path.join(args.out, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Step 1: download (resumes if < 20 tars present)
    ensure_downloaded(raw_dir)

    # Step 2: sample from CSV
    csv_path = os.path.join(raw_dir, "vggsound.csv")
    sampled = load_and_sample(csv_path, args.per_label)

    # Step 3: check if enough videos already on disk per label
    if has_enough_per_label(videos_dir, sampled, args.per_label):
        print("All labels already have enough videos — skipping extraction.")
    else:
        extracted = build_extracted_set(videos_dir, sampled)
        print(f"Already on disk: {len(extracted)}/{len(sampled)}")
        targets = build_target_set(sampled)
        tar_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".tar.gz"))
        if args.start_tar:
            tar_files = [f for f in tar_files if f >= args.start_tar]
            print(
                f"Resuming from {args.start_tar} ({len(tar_files)} archives remaining)"
            )
        for tar_name in tqdm(tar_files, desc="Extracting"):
            extract_tar(os.path.join(raw_dir, tar_name), targets, videos_dir, extracted)
            print(f"  {tar_name}: {len(extracted)}/{len(sampled)} extracted so far")

    # Step 4: delete raw dir now that extraction is complete
    if os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir)
        print(f"Deleted {raw_dir}")

    # Step 5: write QA.json
    extracted = build_extracted_set(videos_dir, sampled)
    write_qa_json(os.path.join(args.out, "QA.json"), extracted, sampled, videos_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/nobackup3/le/AV_Hallucination/data/VGGSound")
    parser.add_argument("--per_label", type=int, default=10)
    parser.add_argument(
        "--start_tar",
        type=str,
        default=None,
        help="Skip archives before this name, e.g. vggsound_08.tar.gz",
    )
    args = parser.parse_args()
    main(args)
