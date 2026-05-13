import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import nltk
import tqdm
from nltk import pos_tag, word_tokenize
from torch.utils.data import DataLoader, Dataset
from videollama2 import mm_infer, model_init
from videollama2.utils import disable_torch_init

nltk.download("averaged_perceptron_tagger_eng")


_HERE = Path(__file__).parent
# QA_FILE     = "/nobackup/le/AV_Hallucination/data/AVHBench/QA.json"
OUTPUT_FILE = (
    "/nobackup/le/AV_Hallucination/results/videollama2/AVHBench/sampled_entities.json"
)

SEED = 42

TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Captioning",
    "Audio Captioning",
    "Video Captioning",
    "AudioSet Captioning",        # describe variant (closed-vocab labels)
    "AudioSet Multiple-Choice",   # mcq variant (A/B/C/D)
]

HALLUC_TASKS: Set[str] = {
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
}
MCQ_TASKS: Set[str] = {"AudioSet Multiple-Choice"}
DESCRIBE_TASKS: Set[str] = {"AudioSet Captioning"}
NLTK_CAPTIONING_TASKS: Set[str] = {
    "AV Captioning",
    "Audio Captioning",
    "Video Captioning",
}

VALID_OUTPUTS: Dict[str, tuple] = {
    **{t: ("Yes", "No") for t in HALLUC_TASKS},
    **{t: ("A", "B", "C", "D") for t in MCQ_TASKS},
}
DISCRETE_TASKS: Set[str] = HALLUC_TASKS | MCQ_TASKS


class CustomDataset(Dataset):
    def __init__(self, questions, video_folder, processor, args):
        self.questions = questions
        self.video_folder = video_folder
        self.processor = processor
        self.args = args

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        line = self.questions[index]
        video_path = os.path.join(self.video_folder, line["video"])
        if "Captioning" in line["task"]:
            qs = f"{line['question']}"
        else:
            qs = f"{line['question']}"
        modal = self.args.modal_type

        try:
            if modal == "a":
                audio_video_tensor = self.processor["audio"](video_path)
            else:
                audio_video_tensor = self.processor["video"](
                    video_path, va=True if modal == "av" else False
                )
        except Exception:
            import traceback

            print(f"video read error: {video_path}")
            traceback.print_exc()
            audio_video_tensor = None

        return {
            "audio_video": audio_video_tensor,
            "question": qs,
            "modal": "audio" if modal == "a" else "video",
        }


def collate_fn(batch):
    aud_vid = [x["audio_video"] for x in batch]
    questions = [x["question"] for x in batch]
    modals = [x["modal"] for x in batch]
    return aud_vid, questions, modals


def get_dataloader(questions, args, processor):
    dataset = CustomDataset(questions, args.video_folder, processor, args)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn
    )
    return dataloader


def extract_entity(text: str) -> List[str]:
    res = []
    tokens = word_tokenize(text, language="english", preserve_line=True)
    tags = pos_tag(tokens)
    for i, (token, tag) in enumerate(tags):
        if tag[:2] == "NN" or token.lower() in ["yes", "no"]:
            res.append(token)
    return res


def find_labels_in_text(text: str, labels: List[str]) -> List[str]:
    """Return labels from `labels` that occur as a whole-word substring of
    `text` (case-insensitive). Used for the AudioSet describe variant: the GT
    vocabulary is closed, so we skip NLTK POS tagging and just check which
    AudioSet labels appear in the model output."""
    text_lc = text.lower()
    hits: List[str] = []
    for lbl in labels:
        lbl_lc = lbl.lower()
        # Word-boundary on alphanumeric ends; keeps embedded punctuation
        # (commas, hyphens) inside labels like "Burst, pop" intact.
        pattern = r"(?<![a-z0-9])" + re.escape(lbl_lc) + r"(?![a-z0-9])"
        if re.search(pattern, text_lc):
            hits.append(lbl)
    return hits


def get_entity_labels_for_entry(entry: dict) -> dict:
    result = {"gt_entities": []}

    task = entry["task"]
    label = entry["label"]

    if task in HALLUC_TASKS:
        result["gt_entities"].append(str(label).strip().lower())

    elif task in MCQ_TASKS:
        # Single letter A/B/C/D.
        result["gt_entities"].append(str(label).strip().lower())

    elif task in DESCRIBE_TASKS:
        # `label` is a list[str] of GT AudioSet labels for the clip.
        if isinstance(label, list):
            result["gt_entities"].extend(str(l).strip().lower() for l in label)
        else:
            # Defensive: tolerate a comma-joined string.
            result["gt_entities"].extend(
                s.strip().lower() for s in str(label).split(",") if s.strip()
            )

    elif task in NLTK_CAPTIONING_TASKS:
        for entity in extract_entity(str(label)):
            result["gt_entities"].append(entity.lower())

    return result


def main(args):
    random.seed(SEED)
    disable_torch_init()

    with open(args.QA_FILE) as f:
        all_qa: List[dict] = json.load(f)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for entry in all_qa:
        by_task[entry["task"]].append(entry)

    active_tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else TASKS

    for task in active_tasks:
        print(f"  {task!r:45s}: {len(by_task[task])} entries")

    # Load the AudioSet label vocabulary when any describe entry is present.
    audioset_labels: List[str] = []
    needs_labels = any(t in DESCRIBE_TASKS for t in active_tasks) and any(
        len(by_task[t]) > 0 for t in active_tasks if t in DESCRIBE_TASKS
    )
    if needs_labels:
        labels_path = args.audioset_labels_file or os.path.join(
            os.path.dirname(args.QA_FILE), "audioset_labels.txt"
        )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"AudioSet describe entries are present but the labels file "
                f"was not found at {labels_path}. Generate it via "
                f"`python method/preprocess_AudioSet.py --variant describe` "
                f"or pass --audioset_labels_file."
            )
        with open(labels_path) as f:
            audioset_labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(audioset_labels)} AudioSet labels from {labels_path}")

    sampled_by_task: Dict[str, List[dict]] = {}
    sampled_video_ids: Set[str] = set()

    for task in active_tasks:
        pool = by_task[task]
        k = (
            len(pool)
            if args.n_per_category is None
            else min(args.n_per_category, len(pool))
        )
        # Discrete-answer tasks (Yes/No or A/B/C/D) balance correct vs.
        # incorrect responses at inference time. We must therefore expose the
        # full pool to the loop — sub-sampling here would cap how many of each
        # side we can ever observe.
        if task not in DISCRETE_TASKS:
            sampled = random.sample(pool, k)
        else:
            random.shuffle(pool)
            sampled = pool
        sampled_by_task[task] = sampled
        for entry in sampled:
            sampled_video_ids.add(entry["video_id"])
        print(f"  {task!r:45s}: sampled {len(sampled)}")

    results: List[dict] = []

    for task, samples in sampled_by_task.items():
        for entry in samples:
            video = entry["video_id"]
            entity_labels = get_entity_labels_for_entry(entry)

            record = {
                "question_id": entry["question_id"],
                "video": video if os.path.splitext(video)[1] else f"{video}.mp4",
                "task": task,
                "question": entry["text"],
                "answer": entry["label"],
                **entity_labels,
                "generated_caption": None,
                "hallucinated_tokens": [],
                "non_hallucinated_tokens": [],
                "hallucinated_entities": [],
                "non_hallucinated_entities": [],
            }
            results.append(record)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    ## start videollama2
    model, processor, tokenizer = model_init(args.model_path)

    dataloader = get_dataloader(results, args, processor)

    kept_results: List[dict] = []

    # Per-task counters for non-hallucination tasks
    task_counts: Dict[str, int] = defaultdict(int)
    # Per-task correct/incorrect counters for hallucination tasks
    halluc_correct: Dict[str, int] = defaultdict(int)
    halluc_incorrect: Dict[str, int] = defaultdict(int)

    half: int = (args.n_per_category // 2) if args.n_per_category is not None else 10**9

    def _task_full(task: str) -> bool:
        if args.n_per_category is None:
            return False
        if task in DISCRETE_TASKS:
            return halluc_correct[task] >= half and halluc_incorrect[task] >= half
        return task_counts[task] >= args.n_per_category

    for i, (aud_vid_tensors, questions, modals) in tqdm.tqdm(
        enumerate(dataloader), total=len(results)
    ):
        # Early exit if all tasks are full
        if args.n_per_category is not None and all(_task_full(t) for t in active_tasks):
            break

        if i % 200 == 0:
            for t in active_tasks:
                if t in DISCRETE_TASKS:
                    print(
                        f"  [i={i}] {t!r}: "
                        f"{halluc_correct[t]} correct, {halluc_incorrect[t]} incorrect"
                    )

        task = results[i]["task"]

        # Skip if this task's quota is already met
        if _task_full(task):
            continue

        audio_video_tensor = aud_vid_tensors[0]
        assert audio_video_tensor is not None
        question = questions[0]
        modal = modals[0]

        output = mm_infer(
            audio_video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal=modal,
            do_sample=False,
        )

        # print(f"'{output}'")

        if task in DISCRETE_TASKS:
            valid = VALID_OUTPUTS[task]
            if output not in valid:
                continue

            gt = str(results[i]["answer"]).strip()
            answered_correctly = output == gt

            if answered_correctly and halluc_correct[task] < half:
                halluc_correct[task] += 1
            elif not answered_correctly and halluc_incorrect[task] < half:
                halluc_incorrect[task] += 1
            else:
                # This slot is full; skip
                continue
        else:
            task_counts[task] += 1

        results[i]["generated_caption"] = output

        if task in DISCRETE_TASKS:
            # Single discrete output token is the only "entity".
            output_entities = [output]
        elif task in DESCRIBE_TASKS:
            output_entities = find_labels_in_text(output, audioset_labels)
        else:
            output_entities = extract_entity(output)

        results[i]["generated_entities"] = output_entities

        for entity in output_entities:
            if entity.lower() in results[i]["gt_entities"]:
                results[i]["non_hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
                results[i]["non_hallucinated_tokens"].extend(tokens)
            else:
                results[i]["hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
                results[i]["hallucinated_tokens"].extend(tokens)

        kept_results.append(results[i])

        # Early exit if all tasks are full
        if args.n_per_category is not None and all(_task_full(t) for t in active_tasks):
            break

    for task in active_tasks:
        if task in DISCRETE_TASKS:
            print(
                f"  {task!r}: kept {halluc_correct[task]} correct, "
                f"{halluc_incorrect[task]} incorrect"
            )

    results = kept_results

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to run. Defaults to all tasks.",
    )
    parser.add_argument(
        "--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--modal_type", type=str, default="av")
    parser.add_argument(
        "--video_folder",
        type=str,
        default="/nobackup/le/AV_Hallucination/data/AVHBench/videos",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/nobackup/le/AV_Hallucination/results/videollama2/AVHBench/sampled_entities.json",
    )
    parser.add_argument(
        "--QA_FILE",
        type=str,
        default="/nobackup3/le/AV_Hallucination/data/AVHBench/QA.json",
    )
    parser.add_argument(
        "--audioset_labels_file",
        type=str,
        default=None,
        help=(
            "Path to audioset_labels.txt (one label per line). Used by the "
            "'AudioSet Captioning' describe variant to scan generated outputs "
            "for closed-vocabulary labels. Defaults to "
            "<dirname(--QA_FILE)>/audioset_labels.txt."
        ),
    )
    args = parser.parse_args()
    main(args)
