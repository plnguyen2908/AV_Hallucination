"""head_umap_508.py

The 508 non-halluc vs 276 halluc heads are NONLINEARLY separable in the 4
proxy-dataset score space (RF 5-fold acc 0.994, SVM-rbf 0.939; linear LDA only
0.765). Linear PCA/LDA therefore can't show it. Here: nonlinear 2D embeddings
that do — unsupervised UMAP, kernel PCA (RBF), and supervised UMAP — each
colored by the fixed 508/276 label.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score
import umap

R = Path("results/qwen2_5_omni")
OUT = R / "head_pca_4datasets"
df = pd.read_csv(OUT / "head_pca_4datasets.csv")
DS = ["AudioSet", "LibriSpeech", "ActivityNet", "YouTubeVOS"]
Z = StandardScaler().fit_transform(df[[f"score_{d}" for d in DS]].to_numpy())
lab = pd.read_csv(R / "categorize_exp_2axis_common508/heads.csv")
key = {(int(r.layer), int(r.head)): (0 if r.category == "Inert" else 1) for r in lab.itertuples()}
y = np.array([key[(int(l), int(h))] for l, h in zip(df["layer"], df["head"])])

COL = {0: "#3182bd", 1: "#e6550d"}; NAME = {0: "non-halluc (508)", 1: "halluc (276)"}

def embed_plot(E, title, fname):
    sil = silhouette_score(E, y)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for k in (0, 1):
        m = y == k
        ax.scatter(E[m, 0], E[m, 1], c=COL[k], s=34 if k else 18,
                   edgecolors="k" if k else "none", linewidths=0.3,
                   alpha=0.9 if k else 0.5, label=f"{NAME[k]} (n={m.sum()})")
    ax.legend(fontsize=11); ax.grid(alpha=0.25)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_title(f"{title}\nsilhouette(508 vs 276) = {sil:.2f}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=300)
    print(f"wrote {OUT/fname}   silhouette={sil:.3f}")
    return sil

# 1) Unsupervised UMAP (is the separation real, label-free structure?)
U = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0).fit_transform(Z)
embed_plot(U, "Unsupervised UMAP of 4 proxy scores (colored by 508/276)", "sep508_umap_unsup.png")

# 2) Kernel PCA (RBF) — unsupervised nonlinear
K = KernelPCA(n_components=2, kernel="rbf", gamma=0.5).fit_transform(Z)
embed_plot(K, "Kernel PCA (RBF) of 4 proxy scores (colored by 508/276)", "sep508_kpca.png")

# 3) Supervised UMAP (cleanest view; uses the label to structure the map)
US = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0,
               target_metric="categorical").fit_transform(Z, y=y)
embed_plot(US, "Supervised UMAP of 4 proxy scores (colored by 508/276)", "sep508_umap_sup.png")
