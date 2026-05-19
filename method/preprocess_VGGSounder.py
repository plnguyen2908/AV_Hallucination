"""
preprocess_VGGSounder.py

Build a describe-variant QA.json from VGGSounder (a re-annotated benchmark
over VGG-Sound, https://github.com/Bizilizi/VGGSounder).

VGGSounder's annotations are multi-label with a per-label modality tag
(A / V / AV). Each underlying clip is 10 seconds from VGG-Sound (sourced
from YouTube). The annotation rows are at
    vggsounder/data/vggsounder.csv
inside the `vggsounder` pip package, with columns:
    video_id, label, modality, background_music, static_image, voice_over

This script:
  1. Loads the CSV via the `vggsounder` pip package.
  2. Filters by --modality (default "AV" to match the QWen "av" pipeline).
  3. For each label, samples up to --n_per_class videos.
  4. Either copies the matching mp4 from --source_dir (if you've already
     downloaded VGG-Sound separately) or fetches the 10-second segment via
     yt-dlp.
  5. Emits QA.json in the describe schema and a vggsounder_labels.txt.

QA.json entry (matches preprocess_{AudioSet,ActivityNet}.py describe):
    {
        "video_id":    "<uuid>.mp4",
        "task":        "VGGSounder Captioning",
        "text":        "Watch the video and describe what you hear and see.
                        Choose only from the following VGGSounder labels:
                        <GT + distractors, shuffled>.",
        "label":       ["<gt label 1>", "<gt label 2>", ...],
        "question_id": "<uuid>",
        "split":       "test"
    }

Dependencies:
    pip install vggsounder yt-dlp pandas tqdm
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

# Final video copies and QA.json land inside the repo; the (potentially large)
# yt-dlp cache routes to /nobackup/le by default, matching the convention
# used by preprocess_ActivityNet.py.
_DEFAULT_OUT = _REPO / "data/VGGSounder"
_DEFAULT_CACHE = Path("/nobackup/le/AV_Hallucination/.vggsounder_yt_cache")

TASK_DESCRIBE = "VGGSounder Captioning"
SEED = 42


def _safe_import_vggsounder():
    try:
        import vggsounder  # noqa: F401
    except ImportError:
        raise SystemExit(
            "vggsounder is required. Install it with:\n"
            "    pip install vggsounder yt-dlp pandas tqdm\n"
        )
    import vggsounder
    return vggsounder


def _locate_csv(args_csv: str | None):
    """Resolve the VGGSounder annotation CSV. Prefer the --annotations_csv
    arg; otherwise look inside the pip package."""
    if args_csv:
        p = Path(args_csv)
        if not p.exists():
            raise SystemExit(f"--annotations_csv not found: {p}")
        return p
    pkg = _safe_import_vggsounder()
    pkg_dir = Path(pkg.__file__).parent
    cand = pkg_dir / "data" / "vggsounder.csv"
    if not cand.exists():
        raise SystemExit(
            f"VGGSounder CSV not found at {cand}. Either reinstall the "
            "`vggsounder` package or pass --annotations_csv explicitly."
        )
    return cand


def load_annotations(csv_path: Path, modality_filter: list[str]):
    """Return ({video_id: [labels]}, all_unique_labels)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if modality_filter:
        wanted = {m.upper() for m in modality_filter}
        df = df[df["modality"].str.upper().isin(wanted)]
    by_video: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        by_video[str(row["video_id"])].append(str(row["label"]))
    all_classes = sorted(df["label"].astype(str).unique())
    return by_video, all_classes


def parse_vggsound_id(video_id: str):
    """VGG-Sound 10-second clip IDs are '{youtube_id}_{start_seconds}'
    (start often zero-padded to 6 digits but not always). Returns
    (youtube_id, start_seconds) or (None, None) if the format is
    unrecognised — in that case --source_dir must be used."""
    if "_" not in video_id:
        return None, None
    yt_id, _, start_str = video_id.rpartition("_")
    try:
        return yt_id, int(start_str)
    except ValueError:
        return None, None


def yt_dlp_clip(youtube_id: str, start_sec: int, duration: int, out_path: Path):
    """Download `duration` seconds of a YouTube video starting at
    `start_sec`. Returns True on success AND if the resulting mp4 has an
    audio stream (Qwen2.5-Omni's "av" path errors out otherwise).

    The format string requires audio: the first preference is a single
    combined stream with audio (acodec!=none); the second is bestvideo +
    bestaudio merged; the third is plain `best` as a last resort."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    end_sec = start_sec + duration
    fmt = (
        "best[height<=720][acodec!=none]"
        "/bestvideo[height<=720]+bestaudio"
        "/best[acodec!=none]"
        "/best"
    )
    cmd = [
        "yt-dlp", url,
        "--download-sections", f"*{start_sec}-{end_sec}",
        "-o", str(out_path),
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--quiet", "--no-warnings",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    if not out_path.exists():
        return False
    # Belt-and-suspenders: drop the file if it lacks an audio stream so the
    # describe-av path doesn't choke on it later. ffprobe is shipped with
    # ffmpeg; if unavailable we just skip the check.
    if not _has_audio_stream(out_path):
        try:
            out_path.unlink()
        except OSError:
            pass
        return False
    return True


def _has_audio_stream(path: Path) -> bool:
    """Return True if the mp4 has at least one audio stream. Treats a missing
    ffprobe binary as 'unknown' → True so we don't drop valid clips."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a", "-show_entries", "stream=index",
                "-of", "csv=p=0", str(path),
            ],
            check=True, timeout=15, capture_output=True, text=True,
        )
        return bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return True


def build_entry_describe(
    video_id_filename: str,
    split: str,
    labels: list[str],
    classes_list: list[str],
    n_distractors: int,
) -> dict:
    """Same shape as preprocess_AudioSet.build_entry_describe: GT labels +
    distractors, shuffled together in the prompt; label is list[str]."""
    distractor_pool = [c for c in classes_list if c not in labels]
    n_take = min(n_distractors, len(distractor_pool))
    distractors = random.sample(distractor_pool, n_take)
    options = list(labels) + distractors
    random.shuffle(options)
    return {
        "video_id": video_id_filename,
        "task": TASK_DESCRIBE,
        "text": (
            "Watch the video and describe what you hear and see. "
            "Choose only from the following VGGSounder labels: "
            + ", ".join(options) + "."
        ),
        "label": list(labels),
        "question_id": str(uuid.uuid4()),
        "split": split,
    }


def cleanup_audioless_entries(qa_path: Path, videos_dir: Path) -> int:
    """Drop entries in QA.json whose backing video has no audio stream.

    Older preprocess runs (or yt-dlp picking a video-only format) sometimes
    produced clips that crash the Qwen 'av' path with
        AssertionError: Video must has audio track when use_audio_in_video=True
    This sweeps QA.json once and removes the offenders so eval doesn't keep
    skipping them. Idempotent — clean runs report 0 dropped.

    Returns the number of entries removed."""
    if not qa_path.exists():
        return 0
    with open(qa_path) as f:
        entries = json.load(f)
    if not entries:
        return 0
    kept: list[dict] = []
    dropped = 0
    for e in tqdm(entries, desc="Cleanup audioless"):
        vid_path = videos_dir / e["video_id"]
        if not vid_path.exists():
            dropped += 1
            continue
        if not _has_audio_stream(vid_path):
            try:
                vid_path.unlink()
            except OSError:
                pass
            dropped += 1
            continue
        kept.append(e)
    if dropped:
        with open(qa_path, "w") as f:
            json.dump(kept, f, indent=2)
        print(f"Cleanup: dropped {dropped} audioless / missing entries; "
              f"{len(kept)} remain in {qa_path}")
    return dropped


def main(args):
    random.seed(SEED)
    out_dir = Path(args.out)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir}")
    print(f"yt-dlp cache: {cache_dir}")

    # Self-heal any stale audioless entries before resuming.
    cleanup_audioless_entries(out_dir / "QA.json", videos_dir)

    modality_filter = [m.strip() for m in args.modality.split(",") if m.strip()]
    print(f"Modality filter: {modality_filter}")

    csv_path = _locate_csv(args.annotations_csv)
    print(f"Annotations CSV: {csv_path}")

    print("Loading VGGSounder annotations ...")
    by_video, all_classes = load_annotations(csv_path, modality_filter)
    print(f"Loaded {len(by_video)} videos, {len(all_classes)} unique labels.")

    labels_txt = out_dir / "vggsounder_labels.txt"
    with open(labels_txt, "w") as f:
        f.write("\n".join(all_classes) + "\n")
    print(f"Wrote {len(all_classes)} labels to {labels_txt}")

    # Per-label index: which videos carry this label.
    by_class: dict[str, list[str]] = defaultdict(list)
    for vid, labels in by_video.items():
        for lbl in labels:
            by_class[lbl].append(vid)

    qa_path = out_dir / "QA.json"
    entries: list[dict] = []
    seen_videos: set[str] = set()
    per_class_count: dict[str, int] = defaultdict(int)
    if args.resume and qa_path.exists():
        with open(qa_path) as f:
            entries = json.load(f)
        # `video_id` is a uuid filename, not the original VGG-Sound id, so
        # build seen_videos from a side mapping stored on disk.
        idx_path = out_dir / ".vggsound_id_map.json"
        if idx_path.exists():
            seen_videos = set(json.load(open(idx_path)))
        for e in entries:
            for lbl in e.get("label", []):
                per_class_count[lbl] += 1
        print(
            f"Resuming: {len(entries)} entries; "
            f"{len(seen_videos)} VGG-Sound clips already processed."
        )

    failed_classes: list[str] = []

    for cls in tqdm(all_classes, desc="Classes"):
        if per_class_count[cls] >= args.n_per_class:
            continue
        candidates = list(by_class[cls])
        random.shuffle(candidates)
        added_for_class = 0
        for vid in candidates:
            if per_class_count[cls] + added_for_class >= args.n_per_class:
                break
            if vid in seen_videos:
                continue

            dst_name = f"{uuid.uuid4().hex[:12]}.mp4"
            final_dst = videos_dir / dst_name

            ok = False
            if args.source_dir:
                src = Path(args.source_dir) / f"{vid}.mp4"
                if src.exists():
                    try:
                        shutil.copy2(src, final_dst)
                        ok = True
                    except Exception as e:
                        print(f"  [{cls}] copy failed for {src}: {e}")
            if not ok:
                yt_id, start = parse_vggsound_id(vid)
                if yt_id is None:
                    continue
                tmp_dst = cache_dir / dst_name
                if yt_dlp_clip(yt_id, start, 10, tmp_dst):
                    try:
                        shutil.copy2(tmp_dst, final_dst)
                        ok = True
                    except Exception as e:
                        print(f"  [{cls}] copy from cache failed: {e}")
            if not ok:
                continue

            labels = by_video[vid]
            entries.append(build_entry_describe(
                video_id_filename=dst_name,
                split="test",
                labels=labels,
                classes_list=all_classes,
                n_distractors=args.n_distractors,
            ))
            for lbl in labels:
                per_class_count[lbl] += 1
            seen_videos.add(vid)
            added_for_class += 1

        if added_for_class == 0 and per_class_count[cls] == 0:
            failed_classes.append(cls)

        # Save incrementally.
        with open(qa_path, "w") as f:
            json.dump(entries, f, indent=2)
        with open(out_dir / ".vggsound_id_map.json", "w") as f:
            json.dump(sorted(seen_videos), f)

    print(f"\nWrote {len(entries)} entries to {qa_path}")
    print(f"Videos in: {videos_dir}")
    if failed_classes:
        print(
            f"{len(failed_classes)} classes produced 0 samples "
            f"(no matching source video / yt-dlp errors):"
        )
        for c in failed_classes[:20]:
            print(f"  - {c}")
        if len(failed_classes) > 20:
            print(f"  ... ({len(failed_classes) - 20} more)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    parser.add_argument(
        "--cache_dir", default=str(_DEFAULT_CACHE),
        help="yt-dlp working directory. Routes to /nobackup/le by default "
             "to keep the heavy raw-download set off /nobackup3.",
    )
    parser.add_argument(
        "--n_per_class", type=int, default=10,
        help="Target number of samples per VGGSounder label.",
    )
    parser.add_argument(
        "--n_distractors", type=int, default=10,
        help="Distractor labels added to each describe prompt alongside the "
             "GT labels.",
    )
    parser.add_argument(
        "--modality", default="AV",
        help="Comma-separated subset of {A,V,AV}. AV by default to match the "
             "Qwen 'av' pipeline; pass 'A,V,AV' to include everything.",
    )
    parser.add_argument(
        "--source_dir", default=None,
        help="Optional pre-downloaded VGG-Sound video directory (one "
             "<video_id>.mp4 per clip). If a video isn't here, the script "
             "falls back to yt-dlp.",
    )
    parser.add_argument(
        "--annotations_csv", default=None,
        help="Override path to vggsounder.csv. Defaults to the file inside "
             "the installed `vggsounder` pip package.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
