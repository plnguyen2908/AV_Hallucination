"""stage2_1h_paper_schedule_figs.py — the 3 camera-ready schedule figures.

Everything the per-layer schedule story needs, one figure each, no in-figure
titles (the caption carries them), same paper style as the head-embedding
figures (serif, 30pt axis text / 25pt ticks & legend, vector PDF + PNG).

  fig1_p_m_layer      p_m(L): per-layer share of LLM-emerged sinks, with the
                      4 single-modality proxies POOLED into one curve per
                      modality (audio = AudioSet+LibriSpeech, visual =
                      ActivityNet+YouTube-VOS) plus av = VGGSounder.
  fig2_shape_m_layer  shape_M(L) = 1 - p_M(L)/max p_M  (reverse schedule),
                      for the 508 always-inert core only.
  fig3_gamma_applied  the gamma actually applied on AVHBench FULL for the 508
                      core: mean +/- 10-90% band across clips.

Output: results/qwen2_5_omni/sink_analysis/paper_figs_schedule/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SA = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis")
SCHED = os.path.join(SA, "gamma_schedules")
S5 = os.path.join(REPO, "results/qwen2_5_omni/stage5_intervention")
OUT = os.path.join(SA, "paper_figs_schedule")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 25,
    "axes.labelsize": 30,
    "legend.fontsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "none",
    "savefig.transparent": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Same three hues as the modality curves elsewhere in the paper.
C = {"a": "#1f77b4", "v": "#2ca02c", "av": "#d62728"}
LBL = {"a": "audio", "v": "visual", "av": "audio-visual"}


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#222222", pad=6)
    ax.grid(alpha=0.25, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=300)
    print("saved", os.path.join(OUT, name) + ".{pdf,png}")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1: p_m(L)
Z = np.load(os.path.join(SA, "stage2_1b_sink_layer_proportion/sink_layer_counts.npz"))
prop = lambda n: Z[n].astype(float) / Z[n].sum()
# Pool the 4 single-modality proxies into one curve per modality (this is what
# the 508-core schedule averages over); av has a single proxy, VGGSounder.
P = {"a": (prop("AudioSet") + prop("LibriSpeech")) / 2,
     "v": (prop("ActivityNet") + prop("YouTubeVOS")) / 2,
     "av": prop("VGGSounder")}
nL = len(P["a"])
L = np.arange(nL)

fig, ax = plt.subplots(figsize=(10, 6))
for k in ("a", "v", "av"):
    ax.plot(L, P[k] * 100, "-o", ms=6, lw=2.2, color=C[k], label=LBL[k])
ax.set_xlabel("decoder layer")
ax.set_ylabel(r"$p_m(\ell)$  (% of sinks)")
ax.set_xticks(range(0, nL, 4))
ax.locator_params(axis="y", nbins=6)
despine(ax)
ax.legend(frameon=False, loc="upper left", handletextpad=0.5, labelspacing=0.3)
save(fig, "fig1_p_m_layer")

# ------------------------------------------------------- fig 2: shape_M(L), 508
rev = dict(np.load(os.path.join(SCHED, "rev_sched_common508.npz")))
fig, ax = plt.subplots(figsize=(10, 6))
for k in ("a", "v", "av"):
    ax.plot(L, rev[f"shape_{k}"], "-o", ms=6, lw=2.2, color=C[k], label=LBL[k])
ax.axhline(1.0, ls="--", lw=1.2, color="#888888")
ax.set_xlabel("decoder layer")
ax.set_ylabel(r"shape$_m(\ell)$")
ax.set_xticks(range(0, nL, 4))
ax.set_ylim(-0.04, 1.12)
ax.locator_params(axis="y", nbins=6)
despine(ax)
ax.legend(frameon=False, loc="lower left", handletextpad=0.5, labelspacing=0.3)
save(fig, "fig2_shape_m_layer")

# --------------------------------------------- fig 3: real applied gamma, 508
G_BASE = 5
ONEHOT = {"AUDIO": (1.0, 0.0, 0.0), "VISUAL": (0.0, 1.0, 0.0), "AV": (0.0, 0.0, 1.0)}
dev = pd.read_csv(os.path.join(S5, "router_v2_dev.csv"))
dev_probs = {r.question_id: (r.p_audio, r.p_visual, r.p_av) for r in dev.itertuples()}
df = pd.read_csv(os.path.join(S5, f"interv_sched_rev_c508_g{G_BASE}_FULL_FULL.csv"))
sa, sv, sav = rev["shape_a"], rev["shape_v"], rev["shape_av"]
G = np.empty((len(df), nL))
for i, row in enumerate(df.itertuples()):
    pa, pv, pav = dev_probs.get(row.question_id, ONEHOT[row.routed])
    G[i] = G_BASE * (sa * pa + sv * pv + sav * pav)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(L, np.percentile(G, 10, 0), np.percentile(G, 90, 0),
                color="#756bb1", alpha=0.22, lw=0, label="10-90% across clips")
ax.plot(L, G.mean(0), "-o", ms=6, lw=2.4, color="#54278f", label=r"mean $\gamma(\ell)$")
ax.axhline(G_BASE, ls="--", lw=1.2, color="#888888", label=r"flat $\gamma=5$")
ax.set_xlabel("decoder layer")
ax.set_ylabel(r"applied $\gamma(\ell)$")
ax.set_xticks(range(0, nL, 4))
ax.locator_params(axis="y", nbins=6)
despine(ax)
ax.legend(frameon=False, loc="lower left", handletextpad=0.5, labelspacing=0.3)
save(fig, "fig3_gamma_applied")

print(f"n clips (fig3) = {len(G)}")
