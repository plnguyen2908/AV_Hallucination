import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import nltk
from nltk import word_tokenize, pos_tag
import argparse

from torch.utils.data import Dataset, DataLoader

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

import os
import tqdm


nltk.download('averaged_perceptron_tagger_eng')


_HERE       = Path(__file__).parent
QA_FILE     = "/nobackup/le/AV_Hallucination/data/AVHBench/QA.json"
OUTPUT_FILE = "/nobackup/le/AV_Hallucination/data/AVHBench/sampled_entities.json"

SEED  = 42

TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Captioning"
]

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
        if line["task"] == "AV Captioning":
            qs = f"{line['question']}. Please describe the video in one full sentence."
        else:
            qs = f"{line['question']}. Start you answer with Yes/No and please provide a detailed explanation after that."
        modal = self.args.modal_type

        preprocess = self.processor['audio' if modal == "a" else "video"]
        try:
            audio_video_tensor = preprocess(video_path, va=True if modal == "av" else False)
        except Exception:
            print(f"video read error: {video_path}")
            audio_video_tensor = None

        return {
            'audio_video': audio_video_tensor,
            'question':    qs,
            'modal':       'audio' if modal == 'a' else "video",
        }


def collate_fn(batch):
    aud_vid   = [x['audio_video'] for x in batch]
    questions = [x['question']    for x in batch]
    modals    = [x['modal']       for x in batch]
    return aud_vid, questions, modals


def get_dataloader(questions, args, processor):
    dataset = CustomDataset(questions, args.video_folder, processor, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                            collate_fn=collate_fn)
    return dataloader

def extract_entity(text: str) -> List[str]:
    res = []
    tokens = word_tokenize(text, language='english', preserve_line=True) 
    tags = pos_tag(tokens)
    for i, (token, tag) in enumerate(tags):
        if tag[:2] == "NN" or token.lower() in ["yes", "no"]:
            res.append(token)
    return res


def get_entity_labels_for_entry(entry: dict) -> dict:
    result = {
        "gt_entities": [],
    }

    task  = entry["task"]
    text  = entry["text"]
    label = entry["label"].strip()

    if task == "Video-driven Audio Hallucination":
        entities = extract_entity(text)
        if entities:
            result["gt_entities"].append(label)
            for entity in entities:
                result["gt_entities"].append(entity)

    elif task == "Audio-driven Video Hallucination":
        entities = extract_entity(text)
        if entities:
            result["gt_entities"].append(label)
            for entity in entities:
                result["gt_entities"].append(entity)

    elif task == "AV Captioning":
        entities = extract_entity(label)
        if entities:
            for entity in entities:
                result["gt_entities"].append(entity)

    return result


def main(args):
    random.seed(SEED)
    disable_torch_init()

    with open(QA_FILE) as f:
        all_qa: List[dict] = json.load(f)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for entry in all_qa:
        by_task[entry["task"]].append(entry)

    for task in TASKS:
        print(f"  {task!r:45s}: {len(by_task[task])} entries")

    sampled_by_task: Dict[str, List[dict]] = {}
    sampled_video_ids: Set[str] = set()

    for task in TASKS:
        pool = by_task[task]
        k = min(args.n_per_category, len(pool))
        sampled = random.sample(pool, k)
        sampled_by_task[task] = sampled
        for entry in sampled:
            sampled_video_ids.add(entry["video_id"])
        # print(f"  {task!r:45s}: sampled {k}")

    results: List[dict] = []

    for task, samples in sampled_by_task.items():
        for entry in samples:
            video = entry["video_id"]
            entity_labels = get_entity_labels_for_entry(entry)

            record = {
                "question_id":            entry["question_id"],
                "video":                  f"{video}.mp4",
                "task":                   task,
                "question":               entry["text"],
                "answer":                  entry["label"],
                **entity_labels,
                "generated_caption":      None,
                "hallucinated_tokens":    [],
                "non_hallucinated_tokens": [],
                "hallucinated_entities":    [],
                "non_hallucinated_entities": [],
            }
            results.append(record)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    ## start videollama2
    model, processor, tokenizer = model_init(args.model_path)

    dataloader = get_dataloader(results, args, processor)

    for i, (aud_vid_tensors, questions, modals) in tqdm.tqdm(enumerate(dataloader), total=len(results)):
        audio_video_tensor = aud_vid_tensors[0]
        question           = questions[0]
        modal              = modals[0]

        try:
            output = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal=modal,
                do_sample=False,
            )
        except Exception:
            import traceback; traceback.print_exc()
            output = "error"

        results[i]["generated_caption"] = output

        output_entities = extract_entity(output)
        results[i]["generated_entities"] = output_entities

        for entity in output_entities:
            if entity in results[i]["gt_entities"]:
                results[i]["non_hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(entity, add_special_tokens=False)
                results[i]["non_hallucinated_tokens"].extend(tokens)
            else:
                results[i]["hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(entity, add_special_tokens=False)
                results[i]["hallucinated_tokens"].extend(tokens)

    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=100)
    parser.add_argument("--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
    parser.add_argument("--modal_type", type=str, default="av")
    parser.add_argument("--video_folder", type=str, default="/nobackup/le/AV_Hallucination/data/AVHBench/videos")
    parser.add_argument("--output_file", type=str, default="/nobackup/le/AV_Hallucination/data/AVHBench/sampled_entities.json")
    args = parser.parse_args()
    main(args)
