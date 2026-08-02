"""Generate cleaner global-inert head-sets for the intervention sweep.

The default 1059-inert set uses a SINGLE pooled tau_90 across all 4 proxy
datasets, which lets YouTube-VOS (only 9 hallucination targets -> noisy,
inflated |scores|) dominate the threshold. Cleaner definitions:

  perdataset : inert = score_d <= tau_d (each dataset's OWN 90th pct) for ALL 4
  noytvos    : drop YouTube-VOS; inert = below per-dataset tau for the 3 solid
               datasets (AudioSet, LibriSpeech, ActivityNet)

Writes heads_<name>.csv with column `category` = "Global inert" for inert heads
(so run_avhbench's category=="Global inert" filter picks them up).
"""
import os, numpy as np, torch, pandas as pd
from pathlib import Path

REPO = Path("/nobackup2/le/AV_Hallucination/.claude/worktrees/qwen3-omni-attribution")
DS = {
    "AudioSet": "results/qwen3_omni/AudioSet_describe/attribution",
    "LibriSpeech": "results/qwen3_omni/LibriSpeech_test-other/attribution",
    "ActivityNet": "results/qwen3_omni/ActivityNet_describe/attribution",
    "YouTubeVOS": "results/qwen3_omni/YouTubeVOS_describe/attribution",
}


def diff(d):
    pth = REPO / d / "pth"
    hal, non, L, H = [], [], None, None
    for fn in sorted(os.listdir(pth)):
        if not fn.endswith(".pth"):
            continue
        data = torch.load(pth / fn, weights_only=False)
        if not data:
            continue
        for _, v in data.items():
            if L is None:
                L, H = len(v), len(v[0])
            g = torch.tensor([[v[l][h]["influence"] for h in range(H)] for l in range(L)])
            (hal if fn.startswith("hal") else non).append(torch.nan_to_num(g))
    return (torch.stack(hal).mean(0) - torch.stack(non).mean(0)).numpy(), L, H


S = {}
L = H = None
for n, d in DS.items():
    S[n], L, H = diff(d)


def write(name, datasets):
    inH = np.zeros((L, H), bool)
    for n in datasets:
        tau = np.percentile(np.abs(S[n]), 90)      # per-dataset tau_90
        inH |= (S[n] > tau)                         # hallucination head in dataset n
    inert = ~inH                                    # inert in ALL listed datasets
    rows = [{"layer": l, "head": h,
             "category": "Global inert" if inert[l, h] else "Active"}
            for l in range(L) for h in range(H)]
    out = REPO / f"results/qwen3_omni/categorize_exp_4axis/heads_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{name}: {int(inert.sum())} inert / {L*H}  -> {out.name}")


write("perdataset", ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"])
write("noytvos", ["AudioSet", "LibriSpeech", "ActivityNet"])
write("audioonly", ["AudioSet", "LibriSpeech"])   # strictest: only the 2 solid audio sets
