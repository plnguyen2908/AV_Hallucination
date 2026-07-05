"""head_set_structure.py

Clean, paper-ready STRUCTURE of the hallucination heads (instead of forcing a
2D cluster). Two panels:
  (A) UpSet-style intersection bars: how many heads are flagged as
      hallucination-driving by each *combination* of the 4 proxy datasets.
  (B) sorted binary flag heatmap (heads x 4 datasets), grouped by # flags.
The always-inert core (flagged by NO dataset) is the dominant block; flagged
heads are sparse and split across datasets with little overlap (none in all 4).
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "results/qwen2_5_omni/head_pca_4datasets"
df = pd.read_csv(OUT / "head_pca_4datasets.csv")
DS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
F = df[[f"flag_{d}" for d in DS]].to_numpy().astype(int)   # (784,4)
count = F.sum(1)
n = len(df)

# Intersection patterns (which subset of datasets flags each head)
from collections import Counter
pat = [tuple(row) for row in F]
cnt = Counter(pat)
# order: empty set first (the inert core), then by size desc among flagged
items = sorted(cnt.items(), key=lambda kv: (sum(kv[0]) > 0, -kv[1]))
labels, sizes = [], []
for p, s in items:
    if sum(p) == 0:
        labels.append("(none) — inert core")
    else:
        labels.append("+".join(DS[i] for i in range(4) if p[i]))
    sizes.append(s)

fig = plt.figure(figsize=(15, 6.5))

# Panel A: intersection bars
axA = fig.add_subplot(1, 2, 1)
colors = ["#2c7fb8" if l.startswith("(none)") else "#e6550d" for l in labels]
y = np.arange(len(labels))
axA.barh(y, sizes, color=colors)
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=9)
axA.invert_yaxis()
for i, s in enumerate(sizes):
    axA.text(s + 4, i, str(s), va="center", fontsize=9)
axA.set_xlabel("# heads", fontsize=11)
axA.set_title("Hallucination-head set intersections\n(which proxy datasets flag each head)", fontsize=12)
axA.grid(axis="x", alpha=0.25)

# Panel B: sorted binary flag heatmap
axB = fig.add_subplot(1, 2, 2)
order = np.lexsort((F[:, 3], F[:, 2], F[:, 1], F[:, 0], count))  # group by count then pattern
M = F[order]
axB.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
axB.set_xticks(range(4)); axB.set_xticklabels(DS, rotation=30, ha="right", fontsize=10)
axB.set_ylabel(f"heads (sorted; n={n})", fontsize=11)
# annotate the inert block
n0 = (count == 0).sum()
axB.axhline(n0 - 0.5, color="#2c7fb8", lw=2)
axB.text(3.6, n0/2, f"{n0}\ninert\ncore", color="#2c7fb8", fontsize=11,
         va="center", ha="left", fontweight="bold")
axB.set_title("Per-head proxy-dataset flags (black=flagged)\ngrouped by # datasets", fontsize=12)

fig.suptitle("Structure of hallucination heads across 4 proxy datasets "
             "(no head flagged by all 4; large stable inert core)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "head_set_structure.png", dpi=300)
print(f"wrote {OUT/'head_set_structure.png'}")
print("\nintersection sizes:")
for l, s in zip(labels, sizes):
    print(f"  {l:<40} {s}")
