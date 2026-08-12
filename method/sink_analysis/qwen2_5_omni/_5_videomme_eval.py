"""Video-MME eval harness for Qwen2.5-Omni — reuses the validated WorldSense
backends (baseline/ours/avcd/asd/mad), whole-video sampling, efficient encoders
and intervention code from `_5_worldsense_eval.py`. Only the data loader, video
path, scoring dimensions are Video-MME-specific.

Video-MME = video(+audio) MCQ, 2700 QA over 900 videos, duration ∈ {short<2min,
medium 4-15min, long 30-60min}. Per the user we run the WorldSense-like subset
(short+medium, ≤~15min ~ WorldSense range) unless --durations overrides.
Answer = single letter A/B/C/D; scored by the official letter extraction.

Sampling = WorldSense v2 (video ≤20s → fps=1; >20s → 20 uniform frames; audio
separate wav ≤90s; use_audio_in_video=False).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_videomme_eval.py \
    --method ours --durations short medium
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
import _5_worldsense_eval as W  # noqa: reuse backends + sampling + scoring

PARQUET = _REPO / "data/VideoMME/videomme/test-00000-of-00001.parquet"
VIDEOS = _REPO / "data/VideoMME/videos"
OUT_DIR = _REPO / "results/qwen2_5_omni/videomme"
DUR_CACHE = _REPO / "data/VideoMME/_durations.json"


def video_seconds(vpath):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(vpath)], capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def load_rows(durations, limit=0):
    df = pd.read_parquet(PARQUET)
    if durations:
        df = df[df["duration"].isin(durations)]
    # cache per-video real duration (ffprobe) for the ≤20s / >20s sampling rule
    cache = json.load(open(DUR_CACHE)) if DUR_CACHE.exists() else {}
    rows = []
    for _, r in df.iterrows():
        vid = r["videoID"]
        vp = VIDEOS / f"{vid}.mp4"
        if not vp.exists():
            continue
        if vid not in cache:
            cache[vid] = video_seconds(vp)
        rows.append(dict(
            index=len(rows), video=vid, vdur=int(round(cache[vid])) or 30,
            question=r["question"], candidates=list(r["options"]),
            answer=r["answer"], duration=r["duration"], domain=r["domain"],
            sub_category=r["sub_category"], task_type=r["task_type"],
            question_id=r["question_id"]))
    DUR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(DUR_CACHE, "w"))
    if limit:
        rows = rows[:limit]
    return rows


def rating(df):
    def m(x):
        x = [v for v in x if v >= 0]
        return round(float(np.mean(x)), 4) if x else None
    out = {"overall": m(df["score"]), "n": int(len(df))}
    for col in ["duration", "domain", "task_type"]:
        out[col] = {k: m(g["score"].tolist()) for k, g in df.groupby(col)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(W.BACKENDS))
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--device_map", default="balanced_low_0")
    ap.add_argument("--max_pixels", type=int, default=360 * 640)
    ap.add_argument("--durations", nargs="+", default=["short", "medium"],
                    help="Video-MME duration buckets to run (default: WorldSense-like short+medium)")
    ap.add_argument("--cd_alpha", type=float, default=2.5)
    ap.add_argument("--heads_csv", default=str(_REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"))
    ap.add_argument("--gamma_schedule_npz", default=str(_REPO / "results/qwen2_5_omni/sink_analysis/gamma_schedules/rev_sched_common508.npz"))
    ap.add_argument("--g_base", type=float, default=5.0)
    ap.add_argument("--sink_mask", default="all")
    ap.add_argument("--mad_gamma", type=float, default=0.5)
    ap.add_argument("--asd_alpha", type=float, default=0.2)
    ap.add_argument("--asd_mds_thr", type=float, default=0.3)
    ap.add_argument("--nframe", type=int, default=0)  # unused (kept for backend compat)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # point the reused WorldSense helpers at Video-MME media; move the audio-wav
    # cache to PERSISTENT disk (the scratch tmpfs is only ~16G and 900 x 90s wavs
    # overflow it -> ENOSPC crash).
    W.VIDEOS = VIDEOS
    W.TRIM_CACHE = _REPO / "data/VideoMME/_audio_cache"

    rows = load_rows(args.durations, args.limit)
    print(f"Video-MME [{','.join(args.durations)}]: {len(rows)} QA (videos present); "
          f"method={args.method}", flush=True)
    backend = W.BACKENDS[args.method](); backend.load(args)
    tag = args.tag or f"{args.method}_{'_'.join(args.durations)}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recs = []; fails = 0; t0 = time.time()
    for row in tqdm(rows, desc=tag):
        try:
            raw = backend.answer(row)
        except RuntimeError as e:
            fails += 1; raw = f"__ERR__{str(e)[:40]}"; torch.cuda.empty_cache()
        pred = W.extract_characters_regex(raw)
        score = -1 if str(raw).startswith("__ERR__") else int(pred == row["answer"])
        recs.append({**{k: row[k] for k in ("question_id", "video", "duration", "domain",
                                            "task_type", "answer")},
                     "prediction": raw, "pred_letter": pred, "score": score})

    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / f"vmme_{tag}.csv", index=False)
    r = rating(df)
    json.dump(r, open(OUT_DIR / f"vmme_{tag}_rating.json", "w"), indent=2)
    valid = df[df["score"] >= 0]
    print(f"\n=== Video-MME {tag} ===")
    print(f"overall: {r['overall']}  n={len(df)} valid={len(valid)} fails={fails}  {(time.time()-t0)/60:.1f} min")
    print("by duration:", r["duration"])
    print("wrote", OUT_DIR / f"vmme_{tag}.csv")


if __name__ == "__main__":
    main()
