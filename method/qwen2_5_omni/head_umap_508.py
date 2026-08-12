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

COL = {0: "#3182bd", 1: "#e6550d"}
# Short labels: at 25pt the full words make the legend ~55% of the axes width,
# and with two diagonal clusters no corner is free of points underneath it.
NAME = {0: "non-halluc.", 1: "halluc."}

# Paper-figure style: no in-figure title (the caption carries it), serif text to
# match the body font, recessive axes, vector output. The two hues are validated
# colorblind-safe (worst-case CVD dE 22.0 protan / 31.0 normal vision).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 25,
    "axes.labelsize": 30,      # axis text
    "legend.fontsize": 25,
    "xtick.labelsize": 25,     # tick numbers
    "ytick.labelsize": 25,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,      # embed TrueType, not Type3 (camera-ready requirement)
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.transparent": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def embed_plot(E, axis_label, fname):
    """One embedding panel. Large type (labels 30pt / ticks 25pt) on a 9.5in
    canvas, so the ratio survives being scaled down into a column."""
    sil = silhouette_score(E, y)   # reported to stdout only, not drawn on the figure
    fig, ax = plt.subplots(figsize=(8.5, 7.0))   # ORIGINAL size; big fonts make the type large RELATIVE to the plot
    for k in (0, 1):                       # majority first so the minority sits on top
        m = y == k
        ax.scatter(E[m, 0], E[m, 1], c=COL[k], s=70,
                   edgecolors="white", linewidths=0.7,
                   alpha=1.0, rasterized=True,
                   label=f"{NAME[k]} ({m.sum()})")
    # Axis stays TIGHT to the data (no inflated ylim): padding the axis to make
    # room for the legend left the spine running far past the last data point
    # with no ticks up there, which reads as a broken axis. Instead the legend
    # goes in whichever CORNER is genuinely empty -- with two diagonal clusters
    # one corner always is.
    x0, x1 = E[:, 0].min(), E[:, 0].max()
    y0, y1 = E[:, 1].min(), E[:, 1].max()
    padx, pady = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    # Corner choice measured on the RENDERED legend box, not on a guessed
    # region: at 25pt the legend is ~half the axes wide, so counting points in a
    # small corner square picks a corner the legend does not actually fit in
    # (it landed on a cluster twice). Draw it in each corner, convert its bbox
    # to data coords, count points underneath, keep the best.
    best, best_hits = None, None
    for loc in ("upper left", "upper right", "lower left", "lower right"):
        leg = ax.legend(loc=loc, ncol=1, frameon=False, handletextpad=0.35,
                        labelspacing=0.35, borderaxespad=0.5, markerscale=1.8)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        hits = int(((E[:, 0] > bb.x0) & (E[:, 0] < bb.x1) &
                    (E[:, 1] > bb.y0) & (E[:, 1] < bb.y1)).sum())
        if best_hits is None or hits < best_hits:
            best, best_hits = loc, hits
        if hits == 0:
            break
    # Nudge the legend outward from the chosen corner so its marker column
    # clears the nearest cluster (the upper-right placement was sitting on it).
    ANCHOR = {"upper right": (1.045, 1.0), "lower right": (1.045, 0.0),
              "upper left": (-0.02, 1.0), "lower left": (-0.02, 0.0)}
    leg = ax.legend(loc=best, bbox_to_anchor=ANCHOR[best],
                    bbox_transform=ax.transAxes, ncol=1, frameon=False,
                    handletextpad=0.35, labelspacing=0.35, borderaxespad=0.0,
                    markerscale=1.8)
    for t in leg.get_texts():
        t.set_color("#333333")
    ax.tick_params(labelsize=25, colors="#222222", pad=6, width=1.6, length=6)
    ax.locator_params(nbins=8)   # enough ticks that the spine is never bare
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2", fontsize=25)   # y-label 25 (x stays 30)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    fig.tight_layout()
    for ext in ("pdf", "png"):             # pdf = vector, for the camera-ready
        # facecolor/transparent are explicit: without an opaque canvas rect a PDF
        # viewer in dark mode shows its own page colour through the figure.
        fig.savefig(OUT / f"{fname}.{ext}", dpi=400,
                    facecolor="white", edgecolor="none", transparent=False)
    plt.close(fig)
    print(f"wrote {OUT/fname}.pdf/.png   silhouette={sil:.3f}")
    return sil

# 1) Unsupervised UMAP (is the separation real, label-free structure?)
U = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0).fit_transform(Z)
embed_plot(U, "UMAP", "sep508_umap_unsup")

# 2) Kernel PCA (RBF) — unsupervised nonlinear
K = KernelPCA(n_components=2, kernel="rbf", gamma=0.5).fit_transform(Z)
embed_plot(K, "kPCA", "sep508_kpca")

# 2b) SUPERVISED kernel PCA (Barshan et al. 2011, HSIC formulation).
# Plain kPCA maximizes variance and ignores y. Supervised kPCA instead maximizes
# the Hilbert-Schmidt dependence between the projection and the labels:
#   Q = K H L H K,  K = RBF kernel on X, L = delta kernel on y, H = I - 11'/n
# and takes the leading eigenvectors of Q; the embedding is K @ beta.
# NOTE: like supervised UMAP, this CONSUMES THE LABELS, so its separation is
# partly by construction -- it is not evidence of label-free cluster structure.
def supervised_kpca(Z, y, gamma=0.5, n_components=2):
    n = Z.shape[0]
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-gamma * D2)                       # RBF kernel on the features
    L = (y[:, None] == y[None, :]).astype(float)  # delta kernel on the labels
    H = np.eye(n) - np.ones((n, n)) / n
    Q = K @ H @ L @ H @ K
    Q = (Q + Q.T) / 2                             # symmetrize against fp drift
    w, V = np.linalg.eigh(Q)
    beta = V[:, np.argsort(w)[::-1][:n_components]]
    return K @ beta

SK = supervised_kpca(Z, y)
embed_plot(SK, "sup. kPCA", "sep508_kpca_sup")

# 3) Supervised UMAP (cleanest view; uses the label to structure the map)
US = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0,
               target_metric="categorical").fit_transform(Z, y=y)
embed_plot(US, "UMAP", "sep508_umap_sup")

# ---------------------------------------------------------------------------
# 4) HARDER separation, two ways -- one is decorative, one is evidence.
#
#  (a) Kernel Fisher discriminant (kernel LDA): explicitly maximizes the
#      between-class / within-class scatter ratio, so it separates far harder
#      than supervised kPCA. But it is fit on ALL points, so the separation is
#      IN-SAMPLE and partly circular -- same caveat as supervised UMAP.
#  (b) OUT-OF-FOLD SVM-rbf decision value: each head is scored by a model that
#      never saw it (5-fold CV). If the classes still separate here, that is
#      real generalizing structure, not an artifact of fitting the labels.
# ---------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

Kp = KernelPCA(n_components=60, kernel="rbf", gamma=0.5).fit_transform(Z)

# (a) in-sample kernel Fisher discriminant, plotted against an orthogonal kPCA axis
fisher = LDA(n_components=1).fit(Kp, y).transform(Kp)[:, 0]
resid = Kp[:, 0] - np.polyval(np.polyfit(fisher, Kp[:, 0], 1), fisher)  # decorrelate
embed_plot(np.c_[fisher, resid], "KFD", "sep508_kfda")

# (b) out-of-fold SVM-rbf decision values -- the honest version
cv = StratifiedKFold(5, shuffle=True, random_state=0)
oof = cross_val_predict(SVC(kernel="rbf", C=10, gamma="scale"), Z, y,
                        cv=cv, method="decision_function")
acc = accuracy_score(y, (oof > 0).astype(int)); auc = roc_auc_score(y, oof)
print(f"[out-of-fold SVM-rbf]  acc={acc:.3f}  AUC={auc:.3f}")
resid2 = Kp[:, 0] - np.polyval(np.polyfit(oof, Kp[:, 0], 1), oof)
embed_plot(np.c_[oof, resid2], "OOF SVM margin", "sep508_oof_svm")

# ---------------------------------------------------------------------------
# 5) Second HIGH-separation view, so the claim does not rest on supervised UMAP
#    alone. Both are standard supervised dimensionality reduction:
#      (a) NCA -- Neighbourhood Components Analysis (Goldberger et al. 2004):
#          learns a linear map that maximizes leave-one-out kNN accuracy.
#      (b) t-SNE run in the NCA metric -- a different embedding family (t-SNE,
#          not UMAP) on a supervised metric, so agreement is not method-specific.
#    Both fit the labels in-sample, exactly like supervised UMAP.
# ---------------------------------------------------------------------------
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.manifold import TSNE

nca = NCA(n_components=2, random_state=0, max_iter=300).fit(Z, y)
embed_plot(nca.transform(Z), "NCA", "sep508_nca")

NCA8 = NCA(n_components=min(4, Z.shape[1]), random_state=0, max_iter=300).fit(Z, y).transform(Z)
TS = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(NCA8)
embed_plot(TS, "t-SNE (NCA metric)", "sep508_tsne_nca")

# ---------------------------------------------------------------------------
# 6) Supervised t-SNE (label-weighted metric) -- the t-SNE-family analogue of
#    what supervised UMAP does internally: inflate between-class distances by
#    (1+beta) before embedding. This is the ONLY construction that reaches a
#    supervised-UMAP-like silhouette, and it reaches it the same way: by
#    building the separation into the metric. Reported as a method-agnostic
#    companion to supervised UMAP, NOT as independent evidence.
# ---------------------------------------------------------------------------
from scipy.spatial.distance import squareform, pdist

D = squareform(pdist(Z))
diff = (y[:, None] != y[None, :])
for beta in (2.0, 5.0):
    Db = D * (1.0 + beta * diff)
    np.fill_diagonal(Db, 0.0)
    T = TSNE(n_components=2, metric="precomputed", init="random",
             perplexity=30, random_state=0).fit_transform(Db)
    embed_plot(T, "t-SNE", f"sep508_tsne_sup_b{int(beta)}")

# ---------------------------------------------------------------------------
# 7) BLENDED variants -- less supervision, so the clusters touch instead of
#    sitting as disjoint islands. Both methods expose the same knob:
#      UMAP: target_weight (0 = ignore labels, 1 = labels dominate; default 0.5)
#      t-SNE: beta, the between-class distance inflation
#    Lower values are the MORE honest picture: the geometry is closer to the
#    data's own, less imposed by the labels.
# ---------------------------------------------------------------------------
for tw in (0.15, 0.30):
    Utw = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0,
                    target_metric="categorical", target_weight=tw).fit_transform(Z, y=y)
    embed_plot(Utw, "UMAP", f"sep508_umap_sup_tw{int(tw*100):02d}")

for beta in (0.5, 1.0):
    Db = D * (1.0 + beta * diff)
    np.fill_diagonal(Db, 0.0)
    Tb = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=30, random_state=0).fit_transform(Db)
    embed_plot(Tb, "t-SNE", f"sep508_tsne_sup_b{str(beta).replace('.','p')}")

# UMAP's supervised mode stays fully separated even at target_weight=0.15;
# genuine blending needs a much weaker label pull.
for tw in (0.02, 0.06):
    Utw = umap.UMAP(n_neighbors=20, min_dist=0.05, random_state=0,
                    target_metric="categorical", target_weight=tw).fit_transform(Z, y=y)
    embed_plot(Utw, "UMAP", f"sep508_umap_sup_tw{int(tw*100):02d}")

# ---------------------------------------------------------------------------
# 8) BLENDED supervised UMAP, matched to the blended t-SNE (silhouette ~0.5).
#    target_weight alone cannot get there -- UMAP's supervised mode returns
#    disjoint islands even at tw=0.02 (silhouette 0.777). Loosening the local
#    geometry too (large n_neighbors, large min_dist) lets the clusters spread
#    until they touch: tw=0.01 / nn=60 / min_dist=0.9 -> 0.511, comparable to
#    supervised t-SNE at beta=0.5 (0.556).
# ---------------------------------------------------------------------------
UB = umap.UMAP(n_neighbors=60, min_dist=0.9, random_state=0,
               target_metric="categorical", target_weight=0.01).fit_transform(Z, y=y)
embed_plot(UB, "UMAP", "sep508_umap_sup_blend")
