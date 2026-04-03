import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

_HERE       = Path(__file__).parent
QA_FILE     = _HERE / "../data/AVHBench/QA.json"
VIDEO_DIR   = _HERE / "../data/AVHBench/videos"
OUTPUT_FILE = _HERE / "../data/AVHBench/sampled_entities.json"

SEED           = 42
N_PER_CATEGORY = 100

TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Matching",
    "AV Captioning",
]


def extract_entity(text: str) -> Optional[str]:
    m = re.search(r"Is the (.+?) making sound in the audio\?", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    m = re.search(r"Is the (.+?) visible in the video\?", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    return None


def get_entity_labels_for_video(video_id: str, all_qa: List[dict]) -> dict:
    result = {
        "present_entities":        [],
        "absent_entities":         [],
        "audio_present_entities":  [],
        "audio_absent_entities":   [],
        "visual_present_entities": [],
        "visual_absent_entities":  [],
        "av_match":                None,
        "gt_caption":              None,
    }

    for entry in all_qa:
        if entry["video_id"] != video_id:
            continue

        task  = entry["task"]
        text  = entry["text"]
        label = entry["label"].strip()

        if task == "Video-driven Audio Hallucination":
            entity = extract_entity(text)
            if entity:
                if label == "Yes":
                    result["audio_present_entities"].append(entity)
                    result["present_entities"].append(entity)
                else:
                    result["audio_absent_entities"].append(entity)
                    result["absent_entities"].append(entity)

        elif task == "Audio-driven Video Hallucination":
            entity = extract_entity(text)
            if entity:
                if label == "Yes":
                    result["visual_present_entities"].append(entity)
                    result["present_entities"].append(entity)
                else:
                    result["visual_absent_entities"].append(entity)
                    result["absent_entities"].append(entity)

        elif task == "AV Matching":
            result["av_match"] = (label == "Yes")

        elif task == "AV Captioning":
            result["gt_caption"] = label

    for key in ["present_entities", "absent_entities",
                "audio_present_entities", "audio_absent_entities",
                "visual_present_entities", "visual_absent_entities"]:
        result[key] = list(dict.fromkeys(result[key]))

    return result


def main():
    random.seed(SEED)

    with open(QA_FILE) as f:
        all_qa: List[dict] = json.load(f)
    print(f"Loaded {len(all_qa)} QA entries across {len(set(d['video_id'] for d in all_qa))} videos")

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for entry in all_qa:
        by_task[entry["task"]].append(entry)

    print("\nTask distribution:")
    for task in TASKS:
        print(f"  {task!r:45s}: {len(by_task[task])} entries")

    sampled_by_task: Dict[str, List[dict]] = {}
    sampled_video_ids: Set[str] = set()

    print(f"\nSampling {N_PER_CATEGORY} questions per task ...")
    for task in TASKS:
        pool = by_task[task]
        k = min(N_PER_CATEGORY, len(pool))
        sampled = random.sample(pool, k)
        sampled_by_task[task] = sampled
        for entry in sampled:
            sampled_video_ids.add(entry["video_id"])
        print(f"  {task!r:45s}: sampled {k}")

    print(f"\n  Unique videos: {len(sampled_video_ids)}")

    qa_by_video: Dict[str, List[dict]] = defaultdict(list)
    for entry in all_qa:
        if entry["video_id"] in sampled_video_ids:
            qa_by_video[entry["video_id"]].append(entry)

    print("\nExtracting entities per video ...")
    results: List[dict] = []

    for video_id in sorted(sampled_video_ids):
        video_entries = qa_by_video[video_id]
        entity_labels = get_entity_labels_for_video(video_id, video_entries)

        record = {
            "video_id":               video_id,
            "video_path":             str(VIDEO_DIR / f"{video_id}.mp4"),
            **entity_labels,
            "generated_caption":      None,
            "hallucinated_tokens":    [],
            "non_hallucinated_tokens": [],
        }

        results.append(record)

        print(f"\n  [{video_id}]")
        print(f"    GT caption  : {entity_labels['gt_caption']}")
        print(f"    AV match    : {entity_labels['av_match']}")
        print(f"    Present     : {entity_labels['present_entities']}")
        print(f"    Absent      : {entity_labels['absent_entities']}")
        print(f"    Audio ✓     : {entity_labels['audio_present_entities']}")
        print(f"    Audio ✗     : {entity_labels['audio_absent_entities']}")
        print(f"    Visual ✓    : {entity_labels['visual_present_entities']}")
        print(f"    Visual ✗    : {entity_labels['visual_absent_entities']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} video records → {OUTPUT_FILE}")

    n_with_absent  = sum(1 for r in results if r["absent_entities"])
    n_with_present = sum(1 for r in results if r["present_entities"])
    n_with_caption = sum(1 for r in results if r["gt_caption"])

    print(f"\nTotal videos processed       : {len(results)}")
    print(f"Videos with absent entities  : {n_with_absent}")
    print(f"Videos with present entities : {n_with_present}")
    print(f"Videos with GT caption       : {n_with_caption}")

    return results


if __name__ == "__main__":
    main()
