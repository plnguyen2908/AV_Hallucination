"""
preprocess_AVCaps.py

Extracts the AVCaps dataset from its zip archive and converts it to the
same QA.json format used by AVHBench:

    [
        {
            "video_id":    "<youtube_id>",
            "task":        "AV Captioning",
            "text":        "Describe what you see and hear.",
            "label":       "<sentence1> <sentence2> ...",  # all audio_visual_captions joined
            "question_id": "<uuid>",
            "split":       "train" | "val" | "test"
        },
        ...
    ]

The label concatenates all human-written audio_visual_captions for the video (typically 3–5
sentences), separated by a single space. Falls back to audio_captions if unavailable.

Videos are extracted to  data/AVCaps/videos/<video_id>.mp4
Output JSON is written to data/AVCaps/QA.json

Usage:
    python method/preprocess_AVCaps.py [--zip path/to/14536325.zip] [--out data/AVCaps]
"""

import argparse
import io
import json
import uuid
import zipfile
from pathlib import Path

CAPTION_PROMPT = "Describe in detail what you see and hear."


def extract_captions(zip_handle: zipfile.ZipFile, caption_filename: str) -> dict:
    """Read a captions JSON from the outer zip."""
    with zip_handle.open(caption_filename) as f:
        return json.load(f)


def extract_videos(zip_handle: zipfile.ZipFile, videos_zip_name: str, videos_dir: Path):
    """Extract an inner videos zip into videos_dir, stripping the prefix folder."""
    videos_dir.mkdir(parents=True, exist_ok=True)

    raw = zip_handle.read(videos_zip_name)
    inner = zipfile.ZipFile(io.BytesIO(raw))

    names = inner.namelist()
    print(f"  Extracting {len(names)} videos from {videos_zip_name} …")

    for name in names:
        if not name.endswith(".mp4"):
            continue
        video_id = Path(name).name  # strip "test_videos/" prefix
        dest = videos_dir / video_id
        if dest.exists():
            continue  # skip already extracted
        data = inner.read(name)
        with open(dest, "wb") as fout:
            fout.write(data)

    inner.close()


def captions_to_qa(captions: dict, split: str) -> list:
    """Convert a captions dict to a list of QA entries (one per video).

    The label is formed by joining all human-written audio_visual_captions
    into a single string (typically 3–5 sentences).  Falls back to
    audio_captions when audio_visual_captions is absent.
    """
    entries = []
    for video_id, caps in captions.items():
        human_sentences = (
            caps.get("audio_visual_captions") or caps.get("audio_captions") or []
        )
        label = " ".join(s.strip() for s in human_sentences if s.strip())

        entries.append(
            {
                "video_id": video_id,
                "task": "AV Captioning",
                "text": CAPTION_PROMPT,
                "label": label,
                "question_id": str(uuid.uuid4()),
                "split": split,
            }
        )
    return entries


def print_stats(qa: list):
    """Print dataset statistics."""
    total = len(qa)
    splits = {}
    tasks = {}
    for item in qa:
        s = item["split"]
        splits[s] = splits.get(s, 0) + 1
        t = item["task"]
        tasks[t] = tasks.get(t, 0) + 1

    print("\n=== AVCaps Dataset Statistics ===")
    print(f"Total entries : {total}")
    print("\nBy split:")
    for s in ("train", "val", "test"):
        print(f"  {s:6s}: {splits.get(s, 0)}")
    print("\nBy task:")
    for t, count in sorted(tasks.items()):
        print(f"  {t}: {count}")

    # Label length stats (words)
    word_counts = [len(item["label"].split()) for item in qa]
    print("\nLabel word-count stats:")
    print(f"  min   : {min(word_counts)}")
    print(f"  max   : {max(word_counts)}")
    print(f"  mean  : {sum(word_counts) / len(word_counts):.1f}")

    # Sentence count per label (split on ". ")
    import re

    sent_counts = [
        len(re.split(r"(?<=[.!?])\s+", item["label"].strip())) for item in qa
    ]
    print("\nLabel sentence-count stats:")
    print(f"  min   : {min(sent_counts)}")
    print(f"  max   : {max(sent_counts)}")
    print(f"  mean  : {sum(sent_counts) / len(sent_counts):.1f}")
    print("=================================\n")


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Preprocess AVCaps dataset")
    parser.add_argument(
        "--zip",
        default=str(repo_root / "data" / "AVCaps" / "14536325.zip"),
        help="Path to the AVCaps master zip file",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "data" / "AVCaps"),
        help="Output directory (videos/ and QA.json go here)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process (default: all three)",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video extraction (useful if already done)",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    videos_dir = out_dir / "videos"

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_qa = []

    with zipfile.ZipFile(zip_path) as zf:
        for split in args.splits:
            caption_file = f"{split}_captions.json"
            videos_file = f"{split}_videos.zip"

            print(f"\n--- Processing split: {split} ---")

            # Captions
            print(f"  Reading {caption_file} …")
            captions = extract_captions(zf, caption_file)
            print(f"  Found {len(captions)} videos in {split} captions")

            qa_entries = captions_to_qa(captions, split)
            all_qa.extend(qa_entries)

            # Videos
            if not args.skip_videos:
                extract_videos(zf, videos_file, videos_dir)
            else:
                print(f"  Skipping video extraction for {split}")

    # Write QA.json
    qa_path = out_dir / "QA.json"
    with open(qa_path, "w") as f:
        json.dump(all_qa, f, indent=2)
    print(f"\nWrote {len(all_qa)} entries to {qa_path}")

    # Verify video files if extraction was done
    if not args.skip_videos:
        extracted = list(videos_dir.glob("*.mp4"))
        print(f"Videos in {videos_dir}: {len(extracted)}")

    print_stats(all_qa)


if __name__ == "__main__":
    main()

