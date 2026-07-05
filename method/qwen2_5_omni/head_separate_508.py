"""head_separate_508.py

Separate the 508 always-non-hallucination heads (Inert in all 4 audio×visual
taxonomies) from the 276 hallucination heads, using the 4 proxy-dataset
contrastive scores. The 508/276 label is fixed (from the combination
categorization); we find the projection of the 4 scores that cleanly separates
the two groups (LDA, supervised), plus a magnitude view.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "results/qwen2_5_omni/head_pca_4datasets"

# 4 scores
df = pd.read_csv(OUT / "head_pca_4datasets.csv")
DS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
S = df[[f"score_{d}" for d in DS]].to_numpy()
Z = StandardScaler().fit_transform(S)

# Fixed 508/276 label from the common-508 head set (Inert = non-halluc)
lab_df = pd.read_csv(_REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv")
key = {(int(r.layer), int(r.head)): (0 if r.category == "Inert" else 1)
       for r in lab_df.itertuples()}
y = np.array([key[(int(l), int(h))] for l, h in zip(df["layer"], df["head"])])
print(f"non-halluc (508-core): {(y==0).sum()}   halluc: {(y==1).sum()}")

COL = {0: "#3182bd", 1: "#e6550d"}; NAME = {0: "non-halluc (508)", 1: "halluc (276)"}

# LDA: 2-class -> 1 discriminant axis that maximally separates the groups.
ld = LDA().fit(Z, y)
LD1 = ld.transform(Z)[:, 0]
acc = ld.score(Z, y)
# silhouette of the 2 groups on the LDA axis
sil = silhouette_score(LD1.reshape(-1, 1), y)
# overlap: how separable along LD1 (pick threshold = midpoint of class means)
m0, m1 = LD1[y == 0].mean(), LD1[y == 1].mean()
thr = (m0 + m1) / 2
sign = 1 if m1 > m0 else -1
pred = ((sign * LD1) > (sign * thr)).astype(int)
print(f"LDA train accuracy={acc:.3f}  silhouette(LD1)={sil:.3f}  "
      f"separable@midpoint={ (pred==y).mean():.3f}")
print(f"LDA loadings (on z-scored scores): " +
      ", ".join(f"{DS[i]}={ld.coef_[0][i]:+.2f}" for i in range(4)))

# --- View 1: LD1 histogram by group (the clean separation) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.linspace(LD1.min(), LD1.max(), 60)
for k in (0, 1):
    ax.hist(LD1[y == k], bins=bins, color=COL[k], alpha=0.75, label=f"{NAME[k]} (n={(y==k).sum()})")
ax.axvline(thr, color="k", ls="--", lw=1.2, label="LDA midpoint")
ax.set_xlabel("LDA discriminant axis (LD1) of the 4 proxy-dataset scores", fontsize=11)
ax.set_ylabel("# heads"); ax.legend(fontsize=10)
ax.set_title(f"508 non-halluc vs 276 halluc heads — LDA separation "
             f"(silhouette={sil:.2f}, sep={ (pred==y).mean()*100:.1f}%)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "sep508_lda_hist.png", dpi=300)
print(f"wrote {OUT/'sep508_lda_hist.png'}")

# --- View 2: 2-D scatter (LD1 vs hallucination magnitude) colored by group ---
mag = Z.max(axis=1)
fig2, ax2 = plt.subplots(figsize=(8.5, 7))
for k in (0, 1):
    m = y == k
    ax2.scatter(LD1[m], mag[m], c=COL[k], s=40 if k else 16,
                edgecolors="k" if k else "none", linewidths=0.4,
                alpha=0.9 if k else 0.4, label=f"{NAME[k]} (n={m.sum()})")
ax2.axvline(thr, color="k", ls="--", lw=1.2)
ax2.set_xlabel("LDA discriminant axis (LD1)", fontsize=12)
ax2.set_ylabel("hallucination magnitude (max z-score)", fontsize=12)
ax2.legend(fontsize=11); ax2.grid(alpha=0.25)
ax2.set_title("508 non-halluc vs 276 halluc — separated in the 4-score space", fontsize=12)
fig2.tight_layout(); fig2.savefig(OUT / "sep508_lda_2d.png", dpi=300)
print(f"wrote {OUT/'sep508_lda_2d.png'}")

df["label_508"] = y; df["LD1"] = LD1
df.to_csv(OUT / "head_sep508_labels.csv", index=False)
print(f"wrote {OUT/'head_sep508_labels.csv'}")
