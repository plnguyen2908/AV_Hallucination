"""Plot the per-layer REVERSE SHAPE schedules (raw g_base multipliers).

shape_M(L) = the per-layer multiplier for modality M (audio/visual/av).
Reverse = 1 - prop/max (peaks early). Shape BEFORE g_base / router blend.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
OUT = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/stage2_1b_sink_layer_proportion")

PAIRS = [
    ("AudioSetxActivityNet",     "AudioSet×ActivityNet"),
    ("AudioSetxYouTubeVOS",      "AudioSet×YouTube-VOS"),
    ("LibriSpeechxActivityNet",  "LibriSpeech×ActivityNet"),
    ("LibriSpeechxYouTubeVOS",   "LibriSpeech×YouTube-VOS"),
    ("common508",                "508 always-inert core (avg)"),
]
COL = {"shape_a": ("tab:blue", "audio  shape_a"),
       "shape_v": ("tab:green", "visual shape_v"),
       "shape_av": ("tab:red", "av     shape_av")}

fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.0 * len(PAIRS), 4.2), sharey=True)
for ax, (key, title) in zip(axes, PAIRS):
    rev = dict(np.load(os.path.join(SCHED, f"rev_sched_{key}.npz")))
    L = np.arange(rev["shape_a"].shape[0])
    for sk, (c, lab) in COL.items():
        ax.plot(L, rev[sk], marker="o", ms=3, color=c, label=lab)
    ax.axhline(1.0, ls="--", lw=0.8, color="gray", label="shape = 1")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("decoder layer")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("shape_M(L)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Per-layer REVERSE shape schedule (1−prop/max, peaks early), per proxy pair\n"
             "γ(clip,L) = g_base · (shape_a·p_a + shape_v·p_v + shape_av·p_av); "
             "shown is shape_M(L) only", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = os.path.join(OUT, "gamma_schedule_shapes_reverse.png")
fig.savefig(out, dpi=130)
print("saved", out)
