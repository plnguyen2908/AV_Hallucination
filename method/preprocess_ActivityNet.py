"""
preprocess_ActivityNet.py

Download ActivityNet-200 via FiftyOne (voxel51) and emit a QA.json that
mirrors the AudioSet **describe** variant — closed-vocabulary captioning
with a per-entry shuffled list of GT + distractor action labels in the
prompt, and a list-of-strings `label` field.

Restricts to **10 samples per class** with **duration ≤ 20 seconds** by
default. ActivityNet-200 has 200 action classes; FiftyOne's zoo loader
pulls source videos from YouTube via yt-dlp when no `source_dir` is given.

QA.json entry shape (matches `preprocess_AudioSet.py --variant describe`):
    {
        "video_id":    "<uuid>.mp4",
        "task":        "ActivityNet Captioning",
        "text":        "Watch the video and identify the action ...
                        Choose only from the following ActivityNet action
                        labels: <shuffled GT + distractors>.",
        "label":       ["<GT class>"],
        "question_id": "<uuid>",
        "split":       "validation" | "train",
    }

Also writes `<--out>/activitynet_labels.txt`: one class per line, the full
200-class vocabulary, ready to be loaded by eval.py's
`find_labels_in_text` matcher.

Usage:
    python method/preprocess_ActivityNet.py
    python method/preprocess_ActivityNet.py \\
        --out data/ActivityNet \\
        --n_per_class 10 \\
        --max_duration 20 \\
        --n_distractors 10 \\
        --splits validation,train
    python method/preprocess_ActivityNet.py --resume

Dependencies:
    pip install fiftyone yt-dlp tqdm
"""

import argparse
import json
import os
import random
import shutil
import uuid
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

# Final QA.json + trimmed video copies land inside the repo (small per-clip
# at ≤20s), but FiftyOne's raw-download cache routes to /nobackup/le since
# /nobackup3 doesn't have the headroom for yt-dlp's working set.
_DEFAULT_OUT = _REPO / "data/ActivityNet"
_DEFAULT_CACHE = Path("/nobackup/le/AV_Hallucination/.fiftyone_zoo_cache")

TASK_DESCRIBE = "ActivityNet Captioning"
SEED = 42


def _safe_import_fiftyone(cache_dir: Path | None = None):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as e:
        raise SystemExit(
            "fiftyone is required. Install it with:\n"
            "    pip install fiftyone yt-dlp\n"
            f"(import error: {e})"
        )
    # Route the zoo cache to the big-disk location BEFORE any download
    # happens. FiftyOne's default is ~/fiftyone/datasets which would fill
    # /nobackup3.
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fo.config.dataset_zoo_dir = str(cache_dir)
    return fo, foz


def discover_classes(splits, source_dir, cache_dir):
    """Load a tiny slice of ActivityNet-200 to read its class taxonomy."""
    fo, foz = _safe_import_fiftyone(cache_dir)
    kwargs = dict(split=splits[0], max_samples=1, shuffle=False)
    if source_dir:
        kwargs["source_dir"] = source_dir
    ds = foz.load_zoo_dataset("activitynet-200", **kwargs)
    classes = list(ds.default_classes or [])
    if not classes and ds.info and "classes" in ds.info:
        classes = list(ds.info["classes"])
    # Drop a leading "background" placeholder if present.
    classes = [c for c in classes if c and c.lower() != "background"]
    fo.delete_dataset(ds.name)
    return classes


def _per_class_split_load(cls, split, need, max_duration, source_dir, cache_dir):
    """Load up to `need` samples for one (class, split) pair."""
    fo, foz = _safe_import_fiftyone(cache_dir)
    kwargs = dict(
        split=split,
        classes=[cls],
        max_samples=need,
        max_duration=max_duration,
        shuffle=True,
        seed=SEED,
    )
    if source_dir:
        kwargs["source_dir"] = source_dir
    return foz.load_zoo_dataset("activitynet-200", **kwargs)


def build_entry_describe(
    video_id: str,
    split: str,
    gt_class: str,
    classes_list: list[str],
    n_distractors: int,
) -> dict:
    """Mirror preprocess_AudioSet.build_entry_describe: GT + distractors,
    shuffled together in the prompt; label is list[str]."""
    distractor_pool = [c for c in classes_list if c != gt_class]
    n_take = min(n_distractors, len(distractor_pool))
    distractors = random.sample(distractor_pool, n_take)
    options = [gt_class] + distractors
    random.shuffle(options)
    return {
        "video_id": video_id,
        "task": TASK_DESCRIBE,
        "text": (
            "Watch the video and describe what you see. "
            "Choose only from the following ActivityNet action labels: "
            + ", ".join(options) + "."
        ),
        "label": [gt_class],
        "question_id": str(uuid.uuid4()),
        "split": split,
    }


def main(args):
    random.seed(SEED)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache_dir)
    print(f"Output dir       : {out_dir}")
    print(f"FiftyOne cache   : {cache_dir}")
    fo, _ = _safe_import_fiftyone(cache_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    source_dir = args.source_dir or None

    print("Discovering ActivityNet-200 class taxonomy ...")
    classes = discover_classes(splits, source_dir, cache_dir)
    print(f"Found {len(classes)} classes.")

    labels_txt = out_dir / "activitynet_labels.txt"
    with open(labels_txt, "w") as f:
        f.write("\n".join(classes) + "\n")
    print(f"Wrote {len(classes)} labels to {labels_txt}")

    qa_path = out_dir / "QA.json"
    entries: list[dict] = []
    seen_classes: set[str] = set()
    if args.resume and qa_path.exists():
        with open(qa_path) as f:
            entries = json.load(f)
        per_class_counts: dict[str, int] = {}
        for e in entries:
            # describe-format entries: GT class lives in label[0].
            if e.get("label") and isinstance(e["label"], list):
                per_class_counts[e["label"][0]] = (
                    per_class_counts.get(e["label"][0], 0) + 1
                )
        seen_classes = {c for c, n in per_class_counts.items() if n >= args.n_per_class}
        print(
            f"Resuming: {len(entries)} entries already in {qa_path}; "
            f"{len(seen_classes)} classes already at quota."
        )

    failed_classes: list[str] = []

    for cls in tqdm(classes, desc="Classes"):
        if cls in seen_classes:
            continue
        current = sum(
            1 for e in entries if e.get("label") and e["label"][0] == cls
        )
        target = args.n_per_class
        new_for_class = 0

        for split in splits:
            need = target - current - new_for_class
            if need <= 0:
                break
            try:
                sub = _per_class_split_load(
                    cls, split, need, args.max_duration, source_dir, cache_dir
                )
            except Exception as e:
                print(f"  [{cls}] split={split}: load failed ({e})")
                continue

            for sample in sub:
                src = sample.filepath
                if not src or not os.path.exists(src):
                    continue
                ext = os.path.splitext(src)[1] or ".mp4"
                video_id = f"{uuid.uuid4().hex[:12]}{ext}"
                dst = videos_dir / video_id
                try:
                    if not dst.exists():
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(f"  [{cls}] copy failed for {src}: {e}")
                    continue

                entries.append(
                    build_entry_describe(
                        video_id=video_id,
                        split=split,
                        gt_class=cls,
                        classes_list=classes,
                        n_distractors=args.n_distractors,
                    )
                )
                new_for_class += 1
                if current + new_for_class >= target:
                    break

            try:
                fo.delete_dataset(sub.name)
            except Exception:
                pass

        if new_for_class == 0 and current == 0:
            failed_classes.append(cls)

        # Save incrementally so a mid-run interruption is recoverable.
        with open(qa_path, "w") as f:
            json.dump(entries, f, indent=2)

    print(f"\nWrote {len(entries)} entries to {qa_path}")
    print(f"Videos in: {videos_dir}")
    if failed_classes:
        print(
            f"{len(failed_classes)} classes produced 0 samples "
            f"(no short videos available / yt-dlp errors):"
        )
        for c in failed_classes:
            print(f"  - {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help=(
            "Where the trimmed video copies and QA.json land. Defaults to "
            "<repo>/data/ActivityNet alongside the other preprocess scripts."
        ),
    )
    parser.add_argument(
        "--cache_dir", default=str(_DEFAULT_CACHE),
        help=(
            "FiftyOne dataset zoo cache. yt-dlp / fiftyone write the raw "
            "downloads here. Defaults to /nobackup/le/AV_Hallucination/"
            ".fiftyone_zoo_cache to keep the heavyweight raw working set off "
            "the small /nobackup3 disk that holds the repo."
        ),
    )
    parser.add_argument(
        "--n_per_class", type=int, default=10,
        help="Target number of samples per ActivityNet-200 class.",
    )
    parser.add_argument(
        "--max_duration", type=float, default=20.0,
        help="Only keep videos no longer than this many seconds.",
    )
    parser.add_argument(
        "--n_distractors", type=int, default=10,
        help=(
            "Number of distractor action labels added (alongside the GT "
            "label) to each describe entry's prompt."
        ),
    )
    parser.add_argument(
        "--splits", default="validation,train",
        help=(
            "Comma-separated splits to draw from. Validation is checked "
            "before train; we stop once a class hits its quota."
        ),
    )
    parser.add_argument(
        "--source_dir", default=None,
        help=(
            "Path to a pre-downloaded ActivityNet raw-video tree. If "
            "omitted, FiftyOne downloads each video via yt-dlp."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
