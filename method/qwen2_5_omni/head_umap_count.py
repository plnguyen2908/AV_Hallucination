"""head_umap_count.py

Supervised UMAP of the 4 proxy-dataset attribution scores, colored by the
per-COMBINATION hallucination count (0..4): how many of the 4 audio×visual
taxonomies flag the head as a hallucination head (count=0 -> the 508 inert core,
count=4 -> always-halluc). Reported with multiclass RF CV accuracy as the
held-out validation that the count is separable.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import umap

R = Path("results/qwen2_5_omni")
OUT = R / "head_pca_4datasets"
df = pd.read_csv(OUT / "head_pca_4datasets.csv")
DS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
Z = StandardScaler().fit_transform(df[[f"score_{d}" for d in DS]].to_numpy())

# Per-combination hallucination count: # of the 4 taxonomies flagging the head.
COMBOS = ["categorize_exp_2axis", "categorize_exp_2axis_youtubevos",
          "categorize_exp_2axis_libri_x_activitynet", "categorize_exp_2axis_libri_x_youtubevos"]
flag = np.zeros((len(df), len(COMBOS)), dtype=int)
for j, c in enumerate(COMBOS):
    h = pd.read_csv(R / c / "heads.csv")
    fl = {(int(r.layer), int(r.head)): (0 if r.category == "Inert" else 1) for r in h.itertuples()}
    flag[:, j] = [fl[(int(l), int(hd))] for l, hd in zip(df["layer"], df["head"])]
count = flag.sum(1)   # 0..4
print("per-combination count distribution:")
for k in range(5):
    print(f"  count={k}: {(count==k).sum()} heads")

# Held-out validation: can a classifier recover the 5-way count from the 4 scores?
rf = RandomForestClassifier(400, random_state=0)
acc = cross_val_score(rf, Z, count, cv=5).mean()
print(f"\nRandomForest 5-fold CV accuracy (5-class count): {acc:.3f}")

# Supervised UMAP, target = count
emb = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0,
                target_metric="categorical").fit_transform(Z, y=count)

CMAP = {0: "#d9d9d9", 1: "#4575b4", 2: "#2ca25f", 3: "#fc8d59", 4: "#d73027"}
SIZE = {0: 16, 1: 40, 2: 70, 3: 120, 4: 170}
fig, ax = plt.subplots(figsize=(9, 7.5))
for k in range(5):
    m = count == k
    if not m.any():
        continue
    ax.scatter(emb[m, 0], emb[m, 1], c=CMAP[k], s=SIZE[k],
               edgecolors="k" if k else "none", linewidths=0.4,
               alpha=0.4 if k == 0 else 0.9, label=f"count={k}  (n={m.sum()})", zorder=k + 1)
ax.legend(title="# taxonomies flagging head", fontsize=10, title_fontsize=11,
          loc="best", framealpha=0.95)
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(alpha=0.25)
ax.set_title("Supervised UMAP of 4 proxy scores, colored by per-combination "
             f"hallucination count (0-4)\nRF 5-fold CV accuracy = {acc*100:.1f}%", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "umap_count_supervised.png", dpi=300)
print(f"wrote {OUT/'umap_count_supervised.png'}")
