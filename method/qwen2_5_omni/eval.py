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
from utils import build_conversation, load_omni, omni_infer, trim_chat_artifacts as _trim_chat_artifacts

nltk.download("averaged_perceptron_tagger_eng")


_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent  # method/qwen2_5_omni/ -> method/ -> repo root

SEED = 42

TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Captioning",
    "Audio Captioning",
    "Video Captioning",
    "AudioSet Captioning",  # describe variant (closed-vocab labels)
    "AudioSet Multiple-Choice",  # mcq variant (A/B/C/D)
]

HALLUC_TASKS: Set[str] = {
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
}
MCQ_TASKS: Set[str] = {"AudioSet Multiple-Choice"}
DESCRIBE_TASKS: Set[str] = {
    "AudioSet Captioning",
    "ActivityNet Captioning",
    "VGGSounder Captioning",
    "YouTubeVOS Captioning",
}
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

# Constrain Qwen on the describe variant: instead of a freeform caption, ask
# for a single sentence built from the provided labels.
# Per-task describe suffix, picked by modality:
#   AudioSet      (a)  → "sounds you hear"
#   ActivityNet   (v)  → "what you see"
#   VGGSounder    (av) → "what you see and hear"   (planned)
DESCRIBE_SUFFIX_BY_TASK: Dict[str, str] = {
    "AudioSet Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match the sounds you hear. No explanations, no other words."
    ),
    "ActivityNet Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match what you see. No explanations, no other words."
    ),
    "VGGSounder Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match what you see and hear. No explanations, no other words."
    ),
    "YouTubeVOS Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match what you see. No explanations, no other words."
    ),
}


# Grammar for shallow noun-phrase chunking: ONE OR MORE noun tags only.
# Adjectives are excluded entirely — both standalone ("loud") and as
# modifiers inside a phrase ("loud engine sound" → "engine sound").
# The isolated-noun problem (NLTK mis-tagging "Music" alone as JJ) is
# handled via a dummy article prefix on short lines below.
_NP_GRAMMAR = r"""
    NP: {<NN.*>+}
"""
_NP_CHUNKER = nltk.RegexpParser(_NP_GRAMMAR)

# Meta words from the question scaffolding that aren't real audio entities;
# the prompt always contains "Answer with …" so it would always sneak into
# gt_entities and pollute the hallucinated/non-hallucinated split.
# `"i"` is the lowercased pronoun; NLTK occasionally tags it as NN once we
# lowercase the input.
_NP_STOPWORDS = {"answer", "i"}


def extract_entity(text: str) -> List[str]:
    """Return noun phrases (and the yes/no answer tokens) from `text`.

    Uses POS-tag-driven NP chunking instead of per-token NN matching, so
    multi-word entities like "background music" stay together and isolated
    NN-misclassified function words ("so", "all") are dropped.

    Each input line is tokenised separately. That naturally breaks noun
    phrases at line boundaries — which is what we want for MCQ option lists
    so the four options don't get glued into one phrase."""
    res: List[str] = []
    seen: set = set()
    # Lowercase before POS tagging so capitalisation doesn't change tags
    # ("Independent" → NNP vs "independent" → JJ would otherwise pull
    # different phrases out of the question (capitalised AudioSet labels)
    # vs the model's lowercase response).
    text = text.lower()
    for raw_line in text.splitlines():
        # Strip an MCQ option label ("a. " / "b. " / "c. " / "d. ") at the
        # very start of the line so the option text becomes its own phrase.
        line = re.sub(r"^\s*[a-d]\.\s+", "", raw_line).strip()
        if not line:
            continue
        # On short lines the tagger has no context and mis-tags isolated
        # nouns as JJ (e.g. "music" by itself). Prepend a dummy article
        # just for tagging on short non-sentence lines, then drop it.
        if len(line.split()) <= 3 and line[-1] not in ".?!,;:":
            tokens = word_tokenize("the " + line, language="english")
            tags = pos_tag(tokens)[1:]
        else:
            tokens = word_tokenize(line, language="english")
            tags = pos_tag(tokens)
        # Yes/No answer tokens.
        for token, _ in tags:
            lo = token.lower()
            if lo in ("yes", "no") and lo not in seen:
                res.append(token)
                seen.add(lo)
        # Noun phrases.
        tree = _NP_CHUNKER.parse(tags)
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            phrase = " ".join(w for w, _ in subtree.leaves())
            lo = phrase.lower()
            if lo in _NP_STOPWORDS or lo in seen:
                continue
            res.append(phrase)
            seen.add(lo)
    return res


def find_labels_in_text(text: str, labels: List[str]) -> List[str]:
    """Whole-word, case-insensitive scan of `text` for entries in `labels`.

    Used for the describe variants (AudioSet / ActivityNet): the GT
    vocabulary is closed, so NLTK POS tagging is unnecessary.

    Longest-match wins: if "Playing beach volleyball" matches, the
    contained "Volleyball" label is suppressed. Otherwise scanning the
    full 200/500-label vocab against the model output produces spurious
    sub-label hits whenever one label is a substring of another."""
    text_lc = text.lower()
    spans = []  # (start, end, lbl)
    for lbl in labels:
        pattern = r"(?<![a-z0-9])" + re.escape(lbl.lower()) + r"(?![a-z0-9])"
        for m in re.finditer(pattern, text_lc):
            spans.append((m.start(), m.end(), lbl))
    # Take longest matches first; any shorter span fully contained inside
    # an already-accepted span is dropped.
    spans.sort(key=lambda s: -(s[1] - s[0]))
    accepted: list = []
    for start, end, lbl in spans:
        contained = any(s <= start and end <= e for s, e, _ in accepted)
        if not contained:
            accepted.append((start, end, lbl))
    # Return in textual order so the debug print reads intuitively.
    accepted.sort(key=lambda s: s[0])
    return [lbl for _, _, lbl in accepted]


# Re-exported from utils so the rest of eval.py can refer to it without the
# alias underscore. utils.trim_chat_artifacts is the canonical implementation.
trim_chat_artifacts = _trim_chat_artifacts


def find_last_valid_answer(output: str, valid: tuple) -> str:
    """Return the valid answer that appears LATEST as a whole word in
    `output` (case-insensitive); '' if none.

    The Qwen prompt asks for chain-of-thought reasoning ending with the
    final Yes/No or A/B/C/D, so we want the last whole-word match — not
    the first — which is what falls out of any reasoning that mentions
    other candidates in passing."""
    if not output:
        return ""
    text_lc = output.lower()
    best = ""
    best_pos = -1
    for v in valid:
        pattern = r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(?![a-z0-9])"
        last_pos = -1
        for m in re.finditer(pattern, text_lc):
            last_pos = m.start()
        if last_pos > best_pos:
            best = v
            best_pos = last_pos
    return best


def get_entity_labels_for_entry(entry: dict) -> dict:
    result = {"gt_entities": []}

    task = entry["task"]
    label = entry["label"]

    def _add(e: str, seen: set):
        e = e.strip().lower()
        if e and e not in seen:
            result["gt_entities"].append(e)
            seen.add(e)

    if task in HALLUC_TASKS or task in MCQ_TASKS:
        # Discrete answer (yes/no or a/b/c/d) + every noun mentioned in the
        # question text — so a CoT response that names the asked-about sound
        # (HALLUC) or any of the four MCQ options doesn't get counted as a
        # hallucinated entity.
        seen: set = set()
        _add(str(label), seen)
        for ent in extract_entity(str(entry.get("text", ""))):
            _add(ent, seen)

    elif task in DESCRIBE_TASKS:
        items: list[str]
        if isinstance(label, list):
            items = [str(l) for l in label]
        else:
            items = str(label).split(",")
        seen: set = set()
        for item in items:
            item_lc = item.strip().lower()
            if not item_lc or item_lc in seen:
                continue
            result["gt_entities"].append(item_lc)
            seen.add(item_lc)
            # Some VGGSounder labels are themselves comma-joined synonyms,
            # e.g. "male speech, man speaking". Split on internal commas so a
            # model output that names just one half ("male speech") still
            # matches the GT.
            if "," in item_lc:
                for piece in item_lc.split(","):
                    piece = piece.strip()
                    if piece and piece not in seen:
                        result["gt_entities"].append(piece)
                        seen.add(piece)

    elif task in NLTK_CAPTIONING_TASKS:
        for entity in extract_entity(str(label)):
            result["gt_entities"].append(entity.lower())

    return result


def main(args):
    random.seed(SEED)

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
        if args.audioset_labels_file:
            labels_path = args.audioset_labels_file
        else:
            # Auto-detect: look for a labels file next to QA.json. AudioSet
            # describe writes audioset_labels.txt; ActivityNet describe
            # writes activitynet_labels.txt.
            qa_dir = os.path.dirname(args.QA_FILE)
            labels_path = None
            for fname in (
                "audioset_labels.txt",
                "activitynet_labels.txt",
                "vggsounder_labels.txt",
                "youtubevos_labels.txt",
            ):
                cand = os.path.join(qa_dir, fname)
                if os.path.exists(cand):
                    labels_path = cand
                    break
        if not labels_path or not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Describe entries are present but no labels file was found "
                f"next to {args.QA_FILE} (looked for audioset_labels.txt / "
                "activitynet_labels.txt). Generate via the matching "
                "preprocess script or pass --audioset_labels_file."
            )
        with open(labels_path) as f:
            audioset_labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(audioset_labels)} describe labels from {labels_path}")

    sampled_by_task: Dict[str, List[dict]] = {}
    sampled_video_ids: Set[str] = set()

    for task in active_tasks:
        pool = by_task[task]
        k = (
            len(pool)
            if args.n_per_category is None
            else min(args.n_per_category, len(pool))
        )
        # Discrete-answer tasks balance correct vs incorrect at inference time,
        # so we expose the full pool to the loop.
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

    # --- start Qwen2.5-Omni ---
    model, processor = load_omni(args.model_path)
    # Tokenizer for the entity-token side-info (same as processor.tokenizer).
    tokenizer = processor.tokenizer

    kept_results: List[dict] = []

    task_counts: Dict[str, int] = defaultdict(int)
    halluc_correct: Dict[str, int] = defaultdict(int)
    halluc_incorrect: Dict[str, int] = defaultdict(int)

    half: int = (args.n_per_category // 2) if args.n_per_category is not None else 10**9

    def _task_full(task: str) -> bool:
        if args.n_per_category is None:
            return False
        if task in DISCRETE_TASKS:
            return halluc_correct[task] >= half and halluc_incorrect[task] >= half
        return task_counts[task] >= args.n_per_category

    for i, record in tqdm.tqdm(enumerate(results), total=len(results)):
        if args.n_per_category is not None and all(_task_full(t) for t in active_tasks):
            break

        if i % 50 == 0:
            for t in active_tasks:
                if t in DISCRETE_TASKS:
                    print(
                        f"  [i={i}] {t!r}: "
                        f"{halluc_correct[t]} correct, {halluc_incorrect[t]} incorrect"
                    )

        task = record["task"]
        if _task_full(task):
            continue

        video_path = os.path.join(args.video_folder, record["video"])
        question = record["question"]
        describe_suffix = DESCRIBE_SUFFIX_BY_TASK.get(task)
        if describe_suffix and describe_suffix not in question:
            question = question + describe_suffix
            # Persist so identify_halluc_head / analyze_attention_bias
            # reconstruct the same prompt that produced `generated_caption`.
            record["question"] = question
        conv = build_conversation(video_path, question, args.modal_type)

        try:
            output = omni_infer(model, processor, conv, args.modal_type)
        except Exception as e:
            import traceback

            print(f"inference error for {record['question_id']} ({video_path}): {e}")
            traceback.print_exc()
            continue

        # Qwen2.5-Omni occasionally bleeds into a fake "What\nHuman: …"
        # chat-template continuation past its own EOS. Clip it.
        output = trim_chat_artifacts(output)

        if task in DISCRETE_TASKS:
            valid = VALID_OUTPUTS[task]
            gt = str(record["answer"]).strip()
            # Relaxed matching for Qwen: it's chatty and CoT-style, so we
            # take the LAST whole-word valid answer in the response — that's
            # the final answer the prompt asks for after the reasoning.
            detected = find_last_valid_answer(output, valid)
            if not detected:
                if i < 5 or i % 50 == 0:
                    print(
                        f"  [unparsable discrete output] task={task!r} raw={output!r}"
                    )
                continue
            answered_correctly = detected.lower() == gt.lower()
            # Keep `output` as the raw model response so the saved
            # generated_caption matches what identify_halluc_head will
            # regenerate. `detected` is used only for entity tracking below.
            if answered_correctly and halluc_correct[task] < half:
                halluc_correct[task] += 1
            elif not answered_correctly and halluc_incorrect[task] < half:
                halluc_incorrect[task] += 1
            else:
                continue
        else:
            task_counts[task] += 1

        record["generated_caption"] = output

        if task in DISCRETE_TASKS:
            # CoT response: pull NLTK nouns from the full reasoning AND ensure
            # the canonical answer token (`detected`) is in the list.
            # extract_entity already special-cases "yes"/"no"; for MCQ letters
            # it ignores single-letter tokens, so we splice `detected` in.
            entities = extract_entity(output)
            if not any(e.lower() == detected.lower() for e in entities):
                entities.append(detected)
            output_entities = entities
        elif task in DESCRIBE_TASKS:
            # Closed-vocab label matches first (canonical case).
            label_matches = find_labels_in_text(output, audioset_labels)
            output_entities = list(label_matches)
            # Plus NLTK noun phrases from the rest of the response, so
            # off-vocab outputs like "Brush brush hair" still surface as
            # entities and get classified (against gt_entities) — typically
            # as hallucinated, since they don't match the GT label string.
            matched_lc = [lm.lower() for lm in label_matches]
            for np in extract_entity(output):
                np_lc = np.lower()
                # Skip NPs already covered by a matched label.
                if any(np_lc in lm for lm in matched_lc):
                    continue
                output_entities.append(np)
        else:
            output_entities = extract_entity(output)

        record["generated_entities"] = output_entities

        # For discrete tasks fall back to word-level overlap when an exact
        # match fails. NLTK's POS tagger is finicky on phrases like
        # "Chirp, tweet sound": the question may yield `["chirp", "tweet"]`
        # while the model output yields `["chirp", "tweet sound"]`. Exact
        # comparison would mis-classify "tweet sound" as hallucinated even
        # though it clearly refers to the asked-about label.
        use_word_overlap = task in DISCRETE_TASKS
        gt_words: set = set()
        if use_word_overlap:
            for ge in record["gt_entities"]:
                gt_words.update(ge.lower().split())

        for entity in output_entities:
            entity_lc = entity.lower()
            matched = entity_lc in record["gt_entities"]
            if not matched and use_word_overlap:
                if set(entity_lc.split()) & gt_words:
                    matched = True
            if matched:
                record["non_hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
                record["non_hallucinated_tokens"].extend(tokens)
            else:
                record["hallucinated_entities"].append(entity)
                tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
                record["hallucinated_tokens"].extend(tokens)

        kept_results.append(record)

        # Diagnostic: dump the first 5 Qwen responses so the CoT format,
        # entity extraction, and hallucinated/non-hallucinated splits can be
        # eyeballed.
        if len(kept_results) <= 5:
            print(f"\n[answer {len(kept_results)}] task={task!r} qid={record['question_id']}")
            print(f"  question   : {record['question']!r}")
            print(f"  raw_output : {output!r}")
            print(f"  entities   : {record['generated_entities']!r}")
            print(f"  gt_entities: {record['gt_entities']!r}")
            print(f"  non_halluc : {record['non_hallucinated_entities']!r}")
            print(f"  halluc     : {record['hallucinated_entities']!r}")
            if task in DISCRETE_TASKS:
                print(f"  gt_answer  : {record['answer']!r}")
                print(f"  correct?   : {detected.lower() == str(record['answer']).strip().lower()}")

        if args.n_per_category is not None and all(_task_full(t) for t in active_tasks):
            break

    for task in active_tasks:
        if task in DISCRETE_TASKS:
            print(
                f"  {task!r}: kept {halluc_correct[task]} correct, "
                f"{halluc_incorrect[task]} incorrect"
            )

    with open(args.output_file, "w") as f:
        json.dump(kept_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to run. Defaults to all tasks.",
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--modal_type", type=str, default="a", choices=["a", "v", "av"])
    parser.add_argument(
        "--video_folder",
        type=str,
        default=str(_REPO / "data/AudioSet/audios"),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(_REPO / "results/qwen2_5_omni/AudioSet/sampled_entities.json"),
    )
    parser.add_argument(
        "--QA_FILE",
        type=str,
        default=str(_REPO / "data/AudioSet/QA.json"),
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
