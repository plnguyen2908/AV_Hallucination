"""
_5_phaseA_dropped.py — rerun router (A/V/AV prompt) + baseline on the
AVHBench yes/no entries dropped from the original DEV/HELDOUT split.

Combines DEV + HELDOUT + DROPPED baselines for the full 5302-entry
paper-comparable AVHBench number. Also re-runs the Stage-1 router with
the new A/V/AV label prompt (saves over router_dev.csv).

Outputs:
    router_dev.csv         router re-run (A/V/AV prompt)
    baseline_DROPPED.csv   same schema as baseline_<split>.csv
    baseline_FULL.csv      DEV ∪ HELDOUT ∪ DROPPED + per-task accuracy
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

from utils import build_conversation, load_omni, omni_infer  # noqa: E402
import _5_phaseA as _pa  # noqa: E402 — reuse router prompt + text infer

DEFAULT_QA = _REPO / "data/AVHBench/QA.json"
DEFAULT_VIDEO_DIR = _REPO / "data/AVHBench/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention"

YES_NO_TASKS = [
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Matching",
]
YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."


def parse_yes_no(text: str) -> str:
    m = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unk"


def main(args):
    out_dir = Path(args.output_dir)
    video_dir = Path(args.video_dir)

    qa = json.load(open(args.qa))
    qa_yn = [x for x in qa if x["task"] in YES_NO_TASKS]
    all_qids = set(x["question_id"] for x in qa_yn)
    qa_map = {x["question_id"]: x for x in qa_yn}
    print(f"  total AVHBench yes/no entries: {len(qa_yn)}")

    used_qids = set()
    for fname in ("baseline_DEV.csv", "baseline_HELDOUT.csv"):
        p = out_dir / fname
        if not p.exists():
            raise SystemExit(f"Missing {p} — run Phase A first.")
        d = pd.read_csv(p)
        used_qids.update(d["question_id"].astype(str).tolist())
        print(f"  already in {fname}: {len(d)} entries "
              f"(accuracy {d.correct.mean()*100:.1f}%)")

    dropped_qids = all_qids - used_qids
    print(f"  DROPPED entries to run: {len(dropped_qids)}")
    dropped_entries = [qa_map[q] for q in dropped_qids]
    by_task = pd.Series([e["task"] for e in dropped_entries]).value_counts()
    print(f"  by task: {dict(by_task)}")

    print(f"\nLoading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)

    # --- Re-run router with new A/V/AV prompt ---
    print(f"\n=== Re-running router on DEV (A/V/AV prompt) ===")
    split_df = pd.read_csv(out_dir / "split.csv", dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    dev_df = split_df[split_df.split == "DEV"].reset_index(drop=True)
    _pa.run_router_dev(model, processor, dev_df, out_dir)

    print(f"\n=== Baseline on DROPPED ({len(dropped_entries)} samples) ===")
    rows = []
    failures = 0
    for e in tqdm(dropped_entries, desc="baseline-DROPPED"):
        vid = str(e["video_id"]).zfill(5)
        vp = video_dir / f"{vid}.mp4"
        if not vp.exists():
            failures += 1
            continue
        prompt = e["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        try:
            out = omni_infer(model, processor, conv, "av", max_new_tokens=8)
        except Exception as ex:
            failures += 1
            tqdm.write(f"  [skip] {e['question_id']}: "
                        f"{type(ex).__name__}: {ex}")
            continue
        pred = parse_yes_no(out)
        rows.append(dict(question_id=e["question_id"], task=e["task"],
                          label=e["label"], generated=out[:120],
                          predicted=pred,
                          correct=int(pred == e["label"])))
    if failures:
        print(f"  failures: {failures}")
    dropped_df = pd.DataFrame(rows)
    dropped_df.to_csv(out_dir / "baseline_DROPPED.csv", index=False)
    print(f"  DROPPED accuracy = {dropped_df.correct.mean()*100:.1f}% "
          f"(n={len(dropped_df)})")
    print(f"  per task:")
    for t in YES_NO_TASKS:
        sub = dropped_df[dropped_df.task == t]
        if len(sub):
            print(f"    {t:<40s} n={len(sub):4d}  "
                  f"acc={sub.correct.mean()*100:5.1f}%")

    # ---- Combine all three for full-5302 view ----
    dev = pd.read_csv(out_dir / "baseline_DEV.csv")
    held = pd.read_csv(out_dir / "baseline_HELDOUT.csv")
    dev["origin"] = "DEV"; held["origin"] = "HELDOUT"; dropped_df["origin"] = "DROPPED"
    full = pd.concat([dev, held, dropped_df], ignore_index=True)
    full.to_csv(out_dir / "baseline_FULL.csv", index=False)
    print(f"\n=== Full AVHBench yes/no (n={len(full)}) ===")
    print(f"  overall: {full.correct.mean()*100:.2f}%")
    for t in YES_NO_TASKS:
        sub = full[full.task == t]
        print(f"    {t:<40s} n={len(sub):4d}  "
              f"acc={sub.correct.mean()*100:5.2f}%")
    print(f"  Excluding DEV (question-disjoint test, n={len(full)-len(dev)}):")
    nd = full[full.origin != "DEV"]
    print(f"    overall: {nd.correct.mean()*100:.2f}%")
    for t in YES_NO_TASKS:
        sub = nd[nd.task == t]
        print(f"    {t:<40s} n={len(sub):4d}  "
              f"acc={sub.correct.mean()*100:5.2f}%")
    print(f"\nWrote {out_dir / 'baseline_FULL.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default=str(DEFAULT_QA))
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
