"""head_cluster_binary.py

Goal: a CLEAN 2-cluster split of heads into non-hallucination vs hallucination,
from the 4 proxy-dataset contrastive scores (no count needed).

Key idea: a head is a hallucination head if it drives hallucination strongly in
AT LEAST ONE proxy -> the separating feature is the per-head MAX z-scored score
("hallucination magnitude"), which is bimodal (dense non-halluc mode + halluc
tail). A 2-component GMM on that magnitude gives a clean, data-driven boundary.

Outputs several views so we can pick the clearest separation.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "results/qwen2_5_omni/head_pca_4datasets"
df = pd.read_csv(OUT / "head_pca_4datasets.csv")
DS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
S = df[[f"score_{d}" for d in DS]].to_numpy()          # (784,4)
Z = StandardScaler().fit_transform(S)
n = len(df)

# Hallucination magnitude = strongest (positive) z-scored score across the 4.
mag = Z.max(axis=1)

# 2-component GMM on the 1-D magnitude -> non-halluc (low mean) vs halluc (high).
gm = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(mag.reshape(-1, 1))
hi = int(np.argmax(gm.means_.ravel()))
lab = (gm.predict(mag.reshape(-1, 1)) == hi).astype(int)   # 1 = halluc, 0 = non-halluc
boundary = None
xs = np.linspace(mag.min(), mag.max(), 2000)
pr = gm.predict(xs.reshape(-1, 1))
flip = np.where(np.diff((pr == hi).astype(int)) != 0)[0]
if len(flip):
    boundary = xs[flip[0]]
print(f"GMM means: {sorted(gm.means_.ravel().round(3))}  boundary≈{boundary:.3f}")
print(f"halluc heads: {lab.sum()}  non-halluc: {(lab==0).sum()}")
sil = silhouette_score(mag.reshape(-1, 1), lab)
print(f"silhouette (1-D magnitude): {sil:.3f}")

COL = {0: "#3182bd", 1: "#e6550d"}   # non-halluc blue, halluc orange
NAME = {0: "non-halluc", 1: "halluc"}

# --- View 1: histogram of magnitude, colored by GMM cluster (THE clean split) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.linspace(mag.min(), mag.max(), 60)
for k in (0, 1):
    ax.hist(mag[lab == k], bins=bins, color=COL[k], alpha=0.75,
            label=f"{NAME[k]} (n={(lab==k).sum()})")
if boundary is not None:
    ax.axvline(boundary, color="k", ls="--", lw=1.2, label=f"GMM boundary={boundary:.2f}")
ax.set_xlabel("hallucination magnitude  =  max over 4 datasets of z-scored contrastive score", fontsize=11)
ax.set_ylabel("# heads"); ax.legend(fontsize=10)
ax.set_title("Heads split into non-halluc vs halluc by hallucination magnitude (2-comp GMM)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "binary_hist_magnitude.png", dpi=300)
print(f"wrote {OUT/'binary_hist_magnitude.png'}")

# --- View 2: 2-D (magnitude vs 2nd-strongest score), colored by cluster ---
mag2 = np.sort(Z, axis=1)[:, -2]    # 2nd largest z-score
fig2, ax2 = plt.subplots(figsize=(8.5, 7))
for k in (0, 1):
    m = lab == k
    ax2.scatter(mag[m], mag2[m], c=COL[k], s=40 if k else 18,
                edgecolors="k" if k else "none", linewidths=0.4,
                alpha=0.9 if k else 0.4, label=f"{NAME[k]} (n={m.sum()})")
if boundary is not None:
    ax2.axvline(boundary, color="k", ls="--", lw=1.2)
ax2.set_xlabel("hallucination magnitude (max z-score)", fontsize=12)
ax2.set_ylabel("2nd-strongest z-score", fontsize=12)
ax2.legend(fontsize=11); ax2.grid(alpha=0.25)
ax2.set_title("Non-halluc vs halluc heads — magnitude vs 2nd-strongest score", fontsize=12)
fig2.tight_layout(); fig2.savefig(OUT / "binary_2d_magnitude.png", dpi=300)
print(f"wrote {OUT/'binary_2d_magnitude.png'}")

# --- View 3: t-SNE of 4-D scores, colored by the binary GMM label ---
T = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, max_iter=1500).fit_transform(Z)
fig3, ax3 = plt.subplots(figsize=(8.5, 7))
for k in (0, 1):
    m = lab == k
    ax3.scatter(T[m, 0], T[m, 1], c=COL[k], s=40 if k else 16,
                edgecolors="k" if k else "none", linewidths=0.4,
                alpha=0.9 if k else 0.4, label=f"{NAME[k]} (n={m.sum()})")
ax3.legend(fontsize=11); ax3.set_xlabel("t-SNE 1"); ax3.set_ylabel("t-SNE 2")
ax3.set_title("t-SNE of 4 proxy scores, colored by non-halluc/halluc (magnitude GMM)", fontsize=12)
ax3.grid(alpha=0.25)
fig3.tight_layout(); fig3.savefig(OUT / "binary_tsne.png", dpi=300)
print(f"wrote {OUT/'binary_tsne.png'}")

# save labels
df["halluc_label"] = lab
df["halluc_magnitude"] = mag
df.to_csv(OUT / "head_binary_labels.csv", index=False)
print(f"wrote {OUT/'head_binary_labels.csv'}")
