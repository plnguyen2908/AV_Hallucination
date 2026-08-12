"""head_pca_4datasets.py

Each decoder head has a hallucination-contrastive score (mean_hal - mean_non_hal)
in each of the 4 proxy datasets:
    AudioSet, LibriSpeech (audio) ; ActivityNet, YouTube-VOS (visual).

Build the 784x4 score matrix, count for each head how many of the 4 datasets
flag it as a hallucination head (score > that dataset's 90th-percentile of
|score|), PCA the standardized 4D scores to 2D, and scatter colored by that
count (0..4, gradient).
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Paper-figure style, matched to head_umap_508.py / head_heatmap_4datasets.py:
# no in-figure titles (captions carry them), serif text, large type (labels 30pt
# / ticks 25pt), vector PDF + PNG, opaque white canvas.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 25,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 20,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "none",
    "savefig.transparent": False, "savefig.bbox": "tight",
})


def _save(fig, stem):
    """PDF (vector, camera-ready) + PNG, both on an opaque white canvas."""
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400, facecolor="white",
                    edgecolor="none", transparent=False)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from av_fusion_2axis_categorize_exp import compute_difference  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "results/qwen2_5_omni/head_pca_4datasets"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "AudioSet":   "AudioSet_describe",
    "LibriSpeech":"LibriSpeech_test-other",
    "ActivityNet":"ActivityNet_describe",
    "YouTubeVOS": "YouTubeVOS_describe",
}

# 784x4 contrastive-score matrix
cols, names = [], []
for name, d in DATASETS.items():
    diff, L, H = compute_difference(_REPO / "results/qwen2_5_omni" / d / "attribution")
    cols.append(diff.reshape(-1))
    names.append(name)
    print(f"{name:<12} {d}: scores shape {diff.shape}")
S = np.stack(cols, axis=1)              # (784, 4)
n = S.shape[0]

# Per-dataset hallucination flag: score > p90 of |score| (top decile, positive)
flags = np.zeros_like(S, dtype=bool)
for j in range(4):
    tau = np.percentile(np.abs(S[:, j]), 90)
    flags[:, j] = S[:, j] > tau
    print(f"  {names[j]:<12} tau(p90 |score|)={tau:.5f}  flagged={flags[:,j].sum()}")
count = flags.sum(axis=1)               # 0..4 per head
print("\nhead count distribution (# datasets flagging the head):")
for k in range(5):
    print(f"  count={k}: {(count==k).sum()} heads")

# PCA on standardized scores
Z = StandardScaler().fit_transform(S)
pca = PCA(n_components=2)
P = pca.fit_transform(Z)
evr = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={evr[0]:.3f} PC2={evr[1]:.3f} (sum {evr.sum():.3f})")
print("PC loadings (rows=PC, cols=datasets):")
for i in range(2):
    print(f"  PC{i+1}: " + ", ".join(f"{names[j]}={pca.components_[i,j]:+.2f}" for j in range(4)))

# Distinct categorical colors + sizes per count level (0..4)
COLORS = {0: "#d9d9d9", 1: "#4575b4", 2: "#2ca25f", 3: "#fc8d59", 4: "#d73027"}
SIZES  = {0: 16, 1: 42, 2: 70, 3: 120, 4: 170}

def draw(ax):
    for k in range(5):                  # background (0) first, high counts on top
        m = count == k
        if not m.any():
            continue
        ax.scatter(P[m, 0], P[m, 1], c=COLORS[k], s=SIZES[k],
                   edgecolors="k" if k else "none", linewidths=0.4,
                   alpha=0.35 if k == 0 else 0.9,
                   label=f"{k}  (n={m.sum()})", zorder=k + 1)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.grid(alpha=0.25)

# --- Figure 1: single scatter, categorical colors + legend ---
fig, ax = plt.subplots(figsize=(9, 7.5))
draw(ax)
leg = ax.legend(title="# datasets flagging head", title_fontsize=22, frameon=False,
                loc="upper right", framealpha=0.95)
fig.tight_layout()
_save(fig, "head_pca_4datasets")
print(f"\nwrote {OUT/'head_pca_4datasets.png'}")

# --- Figure 2: faceted small-multiples, one panel per count level ---
fig2, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.ravel()
for k in range(5):
    a = axes[k]
    a.scatter(P[:, 0], P[:, 1], c="#e0e0e0", s=12, alpha=0.5, zorder=1)  # all heads gray
    m = count == k
    a.scatter(P[m, 0], P[m, 1], c=COLORS[k], s=SIZES.get(k, 40) if k else 26,
              edgecolors="k", linewidths=0.4, alpha=0.95, zorder=2)
    a.set_title(f"count = {k}   (n={m.sum()} heads)", fontsize=24,
                color=COLORS[k] if k else "black")
    a.grid(alpha=0.25)
axes[5].axis("off")
fig2.supxlabel(f"PC1 ({evr[0]*100:.1f}% var)"); fig2.supylabel(f"PC2 ({evr[1]*100:.1f}% var)")
fig2.tight_layout()
_save(fig2, "head_pca_4datasets_facets")
print(f"wrote {OUT/'head_pca_4datasets_facets.png'}")

# --- Per-modality PCA: 2 audio datasets -> 1 axis, 2 visual -> 1 axis ---
ai = [names.index("AudioSet"), names.index("LibriSpeech")]
vi = [names.index("ActivityNet"), names.index("YouTubeVOS")]

def modality_axis(idx):
    z = StandardScaler().fit_transform(S[:, idx])
    p = PCA(n_components=1).fit(z)
    ax1 = p.transform(z)[:, 0]
    # orient so positive = higher mean (more hallucination-driving)
    if np.corrcoef(ax1, z.mean(1))[0, 1] < 0:
        ax1 = -ax1
        load = -p.components_[0]
    else:
        load = p.components_[0]
    return ax1, p.explained_variance_ratio_[0], load

audio_ax, evr_a, load_a = modality_axis(ai)
visual_ax, evr_v, load_v = modality_axis(vi)
print(f"\nAudio axis  PC1 var={evr_a:.3f} loadings AudioSet={load_a[0]:+.2f} LibriSpeech={load_a[1]:+.2f}")
print(f"Visual axis PC1 var={evr_v:.3f} loadings ActivityNet={load_v[0]:+.2f} YouTubeVOS={load_v[1]:+.2f}")

figm, axm = plt.subplots(figsize=(9, 7.5))
for k in range(5):
    m = count == k
    if not m.any():
        continue
    axm.scatter(audio_ax[m], visual_ax[m], c=COLORS[k], s=SIZES[k],
                edgecolors="k" if k else "none", linewidths=0.4,
                alpha=0.35 if k == 0 else 0.9,
                label=f"{k}  (n={m.sum()})", zorder=k + 1)
axm.axhline(0, color="gray", lw=0.6); axm.axvline(0, color="gray", lw=0.6)
axm.legend(title="# datasets flagging head", title_fontsize=22, frameon=False,
           loc="upper left", framealpha=0.95)
axm.set_xlabel("Audio hallucination axis  (PC1 of AudioSet + LibriSpeech)")
axm.set_ylabel("Visual hallucination axis  (PC1 of ActivityNet + YouTube-VOS)")
axm.grid(alpha=0.25)
figm.tight_layout()
_save(figm, "head_modality_axes")
print(f"wrote {OUT/'head_modality_axes.png'}")

# --- Per-modality MIN of the 2 (z-scored) proxy datasets ---
# z-score so the min isn't dominated by the smaller-scale dataset.
audio_min = np.minimum(Z[:, ai[0]], Z[:, ai[1]])   # high only if BOTH audio datasets high
visual_min = np.minimum(Z[:, vi[0]], Z[:, vi[1]])

fign, axn = plt.subplots(figsize=(9, 7.5))
for k in range(5):
    m = count == k
    if not m.any():
        continue
    axn.scatter(audio_min[m], visual_min[m], c=COLORS[k], s=SIZES[k],
                edgecolors="k" if k else "none", linewidths=0.4,
                alpha=0.35 if k == 0 else 0.9,
                label=f"{k}  (n={m.sum()})", zorder=k + 1)
axn.axhline(0, color="gray", lw=0.6); axn.axvline(0, color="gray", lw=0.6)
axn.legend(title="# datasets flagging head", title_fontsize=22, frameon=False,
           loc="upper left", framealpha=0.95)
axn.set_xlabel("Audio axis  =  min(z[AudioSet], z[LibriSpeech])")
axn.set_ylabel("Visual axis  =  min(z[ActivityNet], z[YouTube-VOS])")
axn.grid(alpha=0.25)
fign.tight_layout()
_save(fign, "head_modality_min")
print(f"wrote {OUT/'head_modality_min.png'}")

# --- t-SNE (nonlinear) on the standardized 4D scores ---
from sklearn.manifold import TSNE
for perp in (15, 30, 50):
    T = TSNE(n_components=2, perplexity=perp, init="pca",
             random_state=0, max_iter=1500).fit_transform(Z)

    figt, axt = plt.subplots(figsize=(9, 7.5))
    for k in range(5):
        m = count == k
        if not m.any():
            continue
        axt.scatter(T[m, 0], T[m, 1], c=COLORS[k], s=SIZES[k],
                    edgecolors="k" if k else "none", linewidths=0.4,
                    alpha=0.35 if k == 0 else 0.9,
                    label=f"{k}  (n={m.sum()})", zorder=k + 1)
    axt.legend(title="# datasets flagging head", title_fontsize=22, frameon=False,
               loc="best", framealpha=0.95)
    axt.set_xlabel("t-SNE 1"); axt.set_ylabel("t-SNE 2")
    axt.grid(alpha=0.25)
    figt.tight_layout()
    _save(figt, f"head_tsne_p{perp}")
    print(f"wrote {OUT/f'head_tsne_p{perp}.png'}")

# Save the per-head table
import csv
layers = np.repeat(np.arange(L), H); heads = np.tile(np.arange(H), L)
with open(OUT / "head_pca_4datasets.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head"] + [f"score_{x}" for x in names] +
               [f"flag_{x}" for x in names] + ["count", "PC1", "PC2"])
    for i in range(n):
        w.writerow([layers[i], heads[i]] + [f"{S[i,j]:.6f}" for j in range(4)] +
                   [int(flags[i,j]) for j in range(4)] + [int(count[i]),
                    f"{P[i,0]:.4f}", f"{P[i,1]:.4f}"])
print(f"wrote {OUT/'head_pca_4datasets.csv'}")
