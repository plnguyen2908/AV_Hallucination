"""av_fusion_categorize_qwen3.py

Qwen3-Omni head categorization at a percentile threshold (port of
method/qwen2_5_omni/av_fusion_2axis_categorize_exp.py to this run's axis
layout: 2 audio + 2 visual proxy datasets, no AV dataset).

Score per (layer, head, dataset) = contrastive `mean_hal - mean_non_hal`
recomputed from `<attribution>/pth/`. A single threshold
tau = `--percentile`th percentile of |scores| pooled across ALL datasets.
A head is a hallucination head in dataset X iff score_X > tau (strictly
positive => drives hallucination).

  4-axis view: global Inert = below tau in ALL 4 datasets.
  2-axis view: axis A = mean(AudioSet, LibriSpeech), axis V =
               mean(ActivityNet, YouTubeVOS); Audio/Visual/Audiovisual/Inert.

Writes heads.csv (+ per-percentile counts) and prints the numbers.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]

DATASETS = {  # name -> (attribution_dir, modality)
    "AudioSet":    ("results/qwen3_omni/AudioSet_describe/attribution", "audio"),
    "LibriSpeech": ("results/qwen3_omni/LibriSpeech_test-other/attribution", "audio"),
    "ActivityNet": ("results/qwen3_omni/ActivityNet_describe/attribution", "visual"),
    "YouTubeVOS":  ("results/qwen3_omni/YouTubeVOS_describe/attribution", "visual"),
}


def compute_difference(attribution_dir: Path):
    pth_dir = attribution_dir / "pth"
    hal, non = [], []
    L = H = None
    for fn in sorted(os.listdir(pth_dir)):
        if not fn.endswith(".pth"):
            continue
        is_hal = fn.startswith("hal")
        data = torch.load(pth_dir / fn, weights_only=False)
        if not data:
            continue
        for _, v in data.items():
            if L is None:
                L, H = len(v), len(v[0])
            g = torch.zeros(L, H)
            for li in range(L):
                for hi in range(H):
                    g[li][hi] = v[li][hi]["influence"]
            (hal if is_hal else non).append(torch.nan_to_num(g, nan=0.0))
    mean_hal = torch.stack(hal).mean(0).float().numpy()
    mean_non = torch.stack(non).mean(0).float().numpy()
    return mean_hal - mean_non, len(hal), len(non)


def main(a):
    scores, ntargets = {}, {}
    for name, (d, _mod) in DATASETS.items():
        s, nh, nn = compute_difference(_REPO / d)
        scores[name] = s
        ntargets[name] = (nh, nn)
    L, H = next(iter(scores.values())).shape
    names = list(DATASETS)

    pooled_abs = np.concatenate([np.abs(scores[n]).ravel() for n in names])
    taus = {p: float(np.percentile(pooled_abs, p)) for p in (90, 95, 99)}
    tau = taus[a.percentile]

    # membership per dataset
    inH = {n: scores[n] > tau for n in names}

    print(f"grid = {L} x {H} = {L*H} heads")
    print("targets (hal/non-hal):", {n: ntargets[n] for n in names})
    print(f"\ntau at pooled percentiles: "
          + ", ".join(f"{p}th={taus[p]:.4g}" for p in (90, 95, 99)))
    print(f"USING tau_{a.percentile} = {tau:.4g}\n")

    print(f"per-dataset hallucination heads (score > tau_{a.percentile}):")
    for n in names:
        print(f"  {n:12s}: {int(inH[n].sum()):4d} / {L*H}")

    audio_H = inH["AudioSet"] | inH["LibriSpeech"]
    visual_H = inH["ActivityNet"] | inH["YouTubeVOS"]
    audio_both = inH["AudioSet"] & inH["LibriSpeech"]
    visual_both = inH["ActivityNet"] & inH["YouTubeVOS"]

    any_H = np.zeros((L, H), dtype=bool)
    for n in names:
        any_H |= inH[n]
    global_inert = ~any_H

    print(f"\n=== 4-axis view (tau_{a.percentile}) ===")
    print(f"  Audio-core  (halluc in >=1 audio set) : {int(audio_H.sum())}")
    print(f"     (in BOTH audio sets)                : {int(audio_both.sum())}")
    print(f"  Visual-core (halluc in >=1 visual set): {int(visual_H.sum())}")
    print(f"     (in BOTH visual sets)               : {int(visual_both.sum())}")
    print(f"  ** GLOBAL INERT (not halluc in ANY of 4 datasets): {int(global_inert.sum())} / {L*H} **")

    # 2-axis pooled categorization
    scoreA = np.mean([scores["AudioSet"], scores["LibriSpeech"]], 0)
    scoreV = np.mean([scores["ActivityNet"], scores["YouTubeVOS"]], 0)
    pooled2 = np.concatenate([np.abs(scoreA).ravel(), np.abs(scoreV).ravel()])
    tau2 = float(np.percentile(pooled2, a.percentile))
    inA, inV = scoreA > tau2, scoreV > tau2
    print(f"\n=== 2-axis view (pooled-audio vs pooled-visual, tau_{a.percentile}={tau2:.4g}) ===")
    print(f"  Audiovisual (1,1): {int((inA & inV).sum())}")
    print(f"  Audio       (1,0): {int((inA & ~inV).sum())}")
    print(f"  Visual      (0,1): {int((~inA & inV).sum())}")
    print(f"  Inert       (0,0): {int((~inA & ~inV).sum())}")

    # write heads.csv (4-axis membership + 4-axis category)
    out = _REPO / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for li in range(L):
        for hi in range(H):
            cat = "Global inert" if global_inert[li, hi] else (
                "Audiovisual" if (audio_H[li, hi] and visual_H[li, hi]) else (
                    "Audio" if audio_H[li, hi] else "Visual"))
            rows.append({"layer": li, "head": hi, "category": cat,
                         **{f"inH_{n}": int(inH[n][li, hi]) for n in names}})
    pd.DataFrame(rows).to_csv(out / "heads.csv", index=False)
    print(f"\nwrote {out/'heads.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--percentile", type=int, default=90)
    p.add_argument("--output_dir", default="results/qwen3_omni/categorize_exp_4axis")
    main(p.parse_args())
