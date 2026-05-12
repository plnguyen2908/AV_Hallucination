"""
preprocess_MSVD.py

Downloads the MSVD dataset from HuggingFace (VLM2Vec/MSVD) via snapshot_download,
reads msvd_train.json for captions, copies videos from raw_videos/, samples 1000
entries, writes QA.json, then deletes the snapshot to save space.

QA.json entry:
    {
        "video_id":    "<clip_name>.mp4",
        "task":        "Video Captioning",
        "text":        "Describe what you see.",
        "label":       "<all captions concatenated by space>",
        "question_id": "<uuid>",
        "split":       "train"
    }

Usage:
    python method/preprocess_MSVD.py
    python method/preprocess_MSVD.py --n_samples 1000 --out data/MSVD
    python method/preprocess_MSVD.py --resume

Dependencies:
    pip install huggingface_hub tqdm
"""

import argparse
import json
import os
import random
import shutil
import uuid

from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO_ID = "VLM2Vec/MSVD"
QUESTION = "Describe what you see."
TASK = "Video Captioning"
SEED = 42


def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    videos_dir = os.path.join(args.out, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print(f"Downloading snapshot of {REPO_ID} ...")
    snapshot_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    print(f"Snapshot at: {snapshot_dir}")

    # Locate the annotation JSON and raw_videos folder
    json_path = os.path.join(snapshot_dir, "msvd_train.json")
    raw_videos_dir = os.path.join(snapshot_dir, "raw_videos")

    if not os.path.exists(json_path):
        # Try to find it anywhere under snapshot
        for root, _, files in os.walk(snapshot_dir):
            for f in files:
                if f.endswith(".json"):
                    print(f"  Found JSON: {os.path.join(root, f)}")
        raise FileNotFoundError(f"msvd_train.json not found under {snapshot_dir}")

    if not os.path.exists(raw_videos_dir):
        # List top-level contents to help debug
        print(f"  Snapshot contents: {os.listdir(snapshot_dir)}")
        raise FileNotFoundError(f"raw_videos/ not found under {snapshot_dir}")

    with open(json_path) as f:
        data = json.load(f)
    print(f"Total entries in msvd_train.json: {len(data)}")
    print(
        f"First entry sample: {json.dumps(data[0] if isinstance(data, list) else next(iter(data.items())), indent=2, default=str)[:500]}"
    )

    raw_videos_files = os.listdir(raw_videos_dir)
    print(
        f"raw_videos/ has {len(raw_videos_files)} files. First 5: {raw_videos_files[:5]}"
    )

    # data may be a list of dicts or a dict keyed by video id
    if isinstance(data, dict):
        entries = [
            {"video_id": k, **v}
            if isinstance(v, dict)
            else {"video_id": k, "captions": v}
            for k, v in data.items()
        ]
    else:
        entries = data

    # Sample
    n = min(args.n_samples, len(entries))
    sampled = random.sample(entries, n)
    print(f"Sampled {n} entries.")

    qa_path = os.path.join(args.out, "QA.json")
    existing_ids: set = set()
    existing_entries: list = []
    if args.resume and os.path.exists(qa_path):
        with open(qa_path) as f:
            existing_entries = json.load(f)
        existing_ids = {e["video_id"] for e in existing_entries}
        print(f"Resuming: {len(existing_entries)} entries already in QA.json")

    qa_entries: list[dict] = []
    failed = 0

    for entry in tqdm(sampled, desc="Copying videos"):
        # Resolve video filename — field may be 'video_id', 'clip_name', 'video', etc.
        vid = (
            entry.get("video_id")
            or entry.get("clip_name")
            or entry.get("video")
            or entry.get("id")
            or ""
        )
        vid = str(vid)
        if not vid.endswith(".mp4"):
            vid_file = vid + ".mp4"
        else:
            vid_file = vid

        if args.resume and vid_file in existing_ids:
            continue

        src = os.path.join(raw_videos_dir, vid_file)
        if not os.path.exists(src):
            # Try without extension
            src_bare = os.path.join(raw_videos_dir, vid)
            if os.path.exists(src_bare):
                src = src_bare
                vid_file = vid
            else:
                print(f"  Warning: video not found: {src}")
                failed += 1
                continue

        dst = os.path.join(videos_dir, vid_file)
        shutil.copy2(src, dst)

        # Build label from captions
        captions = (
            entry.get("captions")
            or entry.get("caption")
            or entry.get("sentences")
            or []
        )
        if isinstance(captions, str):
            label = captions
        elif isinstance(captions, list):
            label = " ".join(str(c) for c in captions if c)
        else:
            label = str(captions)

        qa_entries.append(
            {
                "video_id": vid_file,
                "task": TASK,
                "text": QUESTION,
                "label": label,
                "question_id": str(uuid.uuid4()),
                "split": "train",
            }
        )

    if failed:
        print(f"Failed to copy {failed} videos")

    all_entries = existing_entries + qa_entries
    with open(qa_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nWrote {len(all_entries)} entries to {qa_path}")
    print(f"Video files in: {videos_dir}")

    if not args.keep_cache:
        print(f"\nDeleting snapshot cache: {snapshot_dir}")
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/nobackup3/le/AV_Hallucination/data/MSVD",
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep_cache", action="store_true")
    args = parser.parse_args()
    main(args)
