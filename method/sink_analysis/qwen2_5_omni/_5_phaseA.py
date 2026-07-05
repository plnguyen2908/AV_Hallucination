"""
_5_phaseA.py — Stage 5 Phase A:
  (1) Make DEV / HELD-OUT split (100 per yes/no task stratified, seed=0,
      stratify by clip so the same clip is not in both splits).
  (2) Stage-1 routing accuracy on DEV — text-only LLM zero-shot prompt.
  (3) Baseline yes/no accuracy on DEV (full media forward + generate).
  (4) Baseline yes/no accuracy on HELD-OUT.

AVHBench tasks used (yes/no only):
    Video-driven Audio Hallucination  → ground-truth modality = AUDIO
    Audio-driven Video Hallucination  → ground-truth modality = VISUAL
    AV Matching                       → ground-truth modality = AV

Stops once all four numbers are reported (no grid search).

Outputs (`results/qwen2_5_omni/stage5_intervention/`):
    split.csv              question_id, video_id, task, label, split
    router_dev.csv         question_id, gt_modality, predicted, correct
    baseline_<split>.csv   question_id, task, label, generated, predicted, correct
    phaseA_summary.md      headline numbers
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, omni_infer, OMNI_SYSTEM_PROMPT  # noqa: E402


def text_only_infer(model, processor, prompt, max_new_tokens=8):
    """Generate response from text-only prompt (no media). Used for the
    Stage-1 router."""
    conv = [
        {"role": "system",
         "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        text_ids = model.generate(
            **inputs, return_audio=False, do_sample=False,
            max_new_tokens=max_new_tokens)
    gen_ids = text_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()

DEFAULT_QA = _REPO / "data/AVHBench/QA.json"
DEFAULT_VIDEO_DIR = _REPO / "data/AVHBench/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention"

YES_NO_TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Matching",
]
TASK_TO_GT_MODALITY = {
    "Video-driven Audio Hallucination": "AUDIO",
    "Audio-driven Video Hallucination": "VISUAL",
    "AV Matching":                       "AV",
}

YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."
ROUTER_PROMPT_TPL = (
    "You are classifying questions by the modality they ask about. "
    "Audio if the question is about sounds, hearing, or what is audible. "
    "Visual if the question is about images, objects, or what is visible. "
    "AV if the question requires both audio and visual content together "
    "(e.g. matching, joint description). "
    "Respond with exactly one label: Audio, Visual, or AV.\n\n"
    "Question: {q}\n\nClassification:"
)
# Canonical GT labels are AUDIO/VISUAL/AV; accept several casings/aliases
# in the parser and map them to these canonical labels.
_LABEL_ALIASES = {
    "AUDIO": "AUDIO", "AUDIO ": "AUDIO",
    "VISUAL": "VISUAL", "VIS": "VISUAL",
    "AV": "AV",
    "A": "AUDIO", "V": "VISUAL",
}


def make_split(qa_path, seed=0, n_per_task=100):
    """50/50-by-clip split, but DEV is capped at n_per_task per task."""
    qa = json.load(open(qa_path))
    qa_yn = [x for x in qa if x["task"] in YES_NO_TASKS]
    print(f"  yes/no entries: {len(qa_yn)} "
          f"(of {len(qa)} total)")
    # Step 1: assign each clip to DEV or HELD-OUT (seeded 50/50)
    clips = sorted({x["video_id"] for x in qa_yn})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(clips))
    half = len(clips) // 2
    dev_clips = set(clips[i] for i in perm[:half])
    print(f"  clips split: dev={len(dev_clips)}, held-out={len(clips) - len(dev_clips)}")
    # Step 2: from DEV clips, sample n_per_task per task (stratified)
    rng2 = np.random.default_rng(seed + 1)
    dev_chosen_qids = set()
    for task in YES_NO_TASKS:
        cands = [x["question_id"] for x in qa_yn
                  if x["task"] == task and x["video_id"] in dev_clips]
        if len(cands) <= n_per_task:
            dev_chosen_qids.update(cands)
        else:
            chosen_idx = rng2.permutation(len(cands))[:n_per_task]
            dev_chosen_qids.update(cands[i] for i in chosen_idx)
    rows = []
    for x in qa_yn:
        if x["question_id"] in dev_chosen_qids:
            split = "DEV"
        elif x["video_id"] in dev_clips:
            # video is in dev pool but not selected — drop to avoid leakage
            continue
        else:
            split = "HELDOUT"
        rows.append(dict(question_id=x["question_id"],
                          video_id=x["video_id"],
                          task=x["task"],
                          label=x["label"],
                          text=x["text"],
                          split=split))
    df = pd.DataFrame(rows)
    print(f"  DEV size by task:     {dict(df[df.split=='DEV'].task.value_counts())}")
    print(f"  HELD-OUT size by task:{dict(df[df.split=='HELDOUT'].task.value_counts())}")
    return df


def parse_router_answer(text: str) -> str:
    s = text.upper()
    # Prefer longest match (AUDIO/VISUAL/AV over A/V).
    m = re.search(r"\b(AUDIO|VISUAL|AV|A|V)\b", s)
    if not m:
        return "UNK"
    return _LABEL_ALIASES.get(m.group(1), m.group(1))


def parse_yes_no(text: str) -> str:
    s = text.strip()
    m = re.search(r"\b(yes|no)\b", s, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unk"


def run_router_dev(model, processor, dev_df, out_dir):
    print(f"\n=== Stage-1 router: text-only LLM zero-shot on DEV ===")
    print(f"  n_samples = {len(dev_df)}")
    rows = []
    for _, r in tqdm(dev_df.iterrows(), total=len(dev_df), desc="router"):
        prompt = ROUTER_PROMPT_TPL.format(q=r["text"])
        try:
            out = text_only_infer(model, processor, prompt, max_new_tokens=8)
        except Exception as e:
            out = f"ERROR:{type(e).__name__}"
        pred = parse_router_answer(out)
        gt = TASK_TO_GT_MODALITY[r["task"]]
        rows.append(dict(question_id=r["question_id"], task=r["task"],
                          gt_modality=gt, raw=out[:120],
                          predicted=pred, correct=int(pred == gt)))
    df = pd.DataFrame(rows)
    out_csv = out_dir / "router_dev.csv"
    df.to_csv(out_csv, index=False)
    acc = df["correct"].mean()
    per_task = df.groupby("task")["correct"].mean()
    print(f"  overall routing accuracy: {acc*100:.1f}%")
    print(f"  per task:")
    for t, a in per_task.items():
        print(f"    {t:<40s} {a*100:5.1f}%")
    print(f"  confusion (rows=gt, cols=pred):")
    conf = pd.crosstab(df.gt_modality, df.predicted)
    print(conf.to_string())
    print(f"  → {out_csv}")
    return df, acc


def run_baseline(model, processor, df, split_name, out_dir,
                  video_dir):
    print(f"\n=== Baseline on {split_name} ({len(df)} samples) ===")
    rows = []
    failures = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"baseline-{split_name}"):
        # AVHBench IDs are 5-digit zero-padded strings; CSV may have
        # cast them to int, so re-pad before joining.
        vid = str(r["video_id"]).zfill(5)
        vp = video_dir / f"{vid}.mp4"
        if not vp.exists():
            failures += 1
            continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        try:
            out = omni_infer(model, processor, conv, "av",
                              max_new_tokens=8)
        except Exception as e:
            failures += 1
            tqdm.write(f"  [skip] {r['question_id']}: "
                        f"{type(e).__name__}: {e}")
            continue
        pred = parse_yes_no(out)
        rows.append(dict(question_id=r["question_id"], task=r["task"],
                          label=r["label"], generated=out[:120],
                          predicted=pred,
                          correct=int(pred == r["label"])))
    if failures:
        print(f"  failures: {failures}")
    out_csv = out_dir / f"baseline_{split_name}.csv"
    bdf = pd.DataFrame(rows)
    bdf.to_csv(out_csv, index=False)
    overall = bdf["correct"].mean() if len(bdf) else float("nan")
    per_task = bdf.groupby("task")["correct"].mean()
    print(f"  overall baseline accuracy: {overall*100:.1f}% (n={len(bdf)})")
    print(f"  per task:")
    for t, a in per_task.items():
        n = (bdf.task == t).sum()
        print(f"    {t:<40s} {a*100:5.1f}% (n={n})")
    print(f"  → {out_csv}")
    return bdf, overall


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)

    print(f"=== Building split ===")
    if (out_dir / "split.csv").exists() and not args.force_resplit:
        df = pd.read_csv(out_dir / "split.csv",
                          dtype={"video_id": str})
        # Defensive: re-pad in case the file was written without dtype.
        df["video_id"] = df["video_id"].astype(str).str.zfill(5)
        print(f"  loaded existing split: {len(df)} rows")
    else:
        df = make_split(args.qa, seed=args.seed, n_per_task=args.n_dev_per_task)
        df.to_csv(out_dir / "split.csv", index=False)
        print(f"  wrote {out_dir / 'split.csv'}")
    dev_df = df[df.split == "DEV"].reset_index(drop=True)
    held_df = df[df.split == "HELDOUT"].reset_index(drop=True)
    print(f"  DEV n={len(dev_df)}, HELDOUT n={len(held_df)}")

    if args.split_only:
        return

    print(f"\nLoading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)

    if args.do_router:
        run_router_dev(model, processor, dev_df, out_dir)

    if args.do_dev_baseline:
        run_baseline(model, processor, dev_df, "DEV", out_dir, video_dir)

    if args.do_held_baseline:
        run_baseline(model, processor, held_df, "HELDOUT", out_dir,
                       video_dir)

    print("\nPhase A complete. STOP for confirmation before grid.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default=str(DEFAULT_QA))
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_dev_per_task", type=int, default=100)
    p.add_argument("--force_resplit", action="store_true")
    p.add_argument("--split_only", action="store_true")
    p.add_argument("--do_router", action="store_true")
    p.add_argument("--do_dev_baseline", action="store_true")
    p.add_argument("--do_held_baseline", action="store_true")
    args = p.parse_args()
    main(args)
