"""Plot the per-layer gamma SHAPE schedules (forward vs reverse) for every pair.

shape_M(L) is the per-layer multiplier applied to G_BASE for modality M
(audio / visual / av). Forward = prop/mean (peaks deep); reverse = 1 - prop/max
(peaks early). Reads the npz files actually used by the interventions.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
OUT = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/stage2_1b_sink_layer_proportion")

# (key, title, best forward g_base, best reverse g_base) from the DEV sweep.
PAIRS = [
    ("AudioSetxActivityNet",     "AudioSet×ActivityNet",            2, 4),
    ("AudioSetxYouTubeVOS",      "AudioSet×YouTube-VOS",            3, 4),
    ("LibriSpeechxActivityNet",  "LibriSpeech×ActivityNet",         4, 5),
    ("LibriSpeechxYouTubeVOS",   "LibriSpeech×YouTube-VOS",         5, 5),
    ("common508",                "508 always-inert core (avg)",     5, 5),
]
COL = {"shape_a": ("tab:blue", "audio  γ_a"),
       "shape_v": ("tab:green", "visual γ_v"),
       "shape_av": ("tab:red", "av     γ_av")}

fig, axes = plt.subplots(len(PAIRS), 2, figsize=(13, 3.0 * len(PAIRS)), sharex=True)
for r, (key, title, g_fwd, g_rev) in enumerate(PAIRS):
    fwd = dict(np.load(os.path.join(SCHED, f"sched_{key}.npz")))
    rev = dict(np.load(os.path.join(SCHED, f"rev_sched_{key}.npz")))
    L = np.arange(fwd["shape_a"].shape[0])
    panels = [(fwd, g_fwd, "FORWARD  (peaks deep)"),
              (rev, g_rev, "REVERSE  (peaks early)")]
    for col, (data, g_base, tag) in enumerate(panels):
        ax = axes[r, col]
        # effective per-layer gamma if this clip were pure-modality: g_base*shape
        for sk, (c, lab) in COL.items():
            ax.plot(L, g_base * data[sk], marker="o", ms=3, color=c, label=lab)
        # flat-gamma reference = g_base (what a non-scheduled run would use)
        ax.axhline(g_base, ls="--", lw=1.0, color="gray",
                   label=f"flat γ = g_base={g_base}")
        ax.set_ylabel("effective γ(L)")
        if r == 0:
            ax.set_title(tag, fontsize=11)
        ax.text(0.02, 0.95, f"{title}  (g_base={g_base})",
                transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")
        if r == len(PAIRS) - 1:
            ax.set_xlabel("decoder layer")
        if r == 0:
            ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)
fig.suptitle("Per-layer EFFECTIVE gamma — forward vs reverse, per proxy pair "
             "(best g_base per config)\n"
             "γ_M(L) = g_base · shape_M(L);  per clip the 3 lines are blended by "
             "router probs p_a,p_v,p_av and applied to ALL sink tokens at layer L",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.975])
out = os.path.join(OUT, "gamma_schedule_fwd_vs_rev.png")
fig.savefig(out, dpi=130)
print("saved", out)
