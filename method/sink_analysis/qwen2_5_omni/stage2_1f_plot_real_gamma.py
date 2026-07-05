"""Plot the REAL per-layer gamma actually applied on AVHBench FULL.

The intervention uses ONE blended gamma per layer per clip:
    gamma(clip, L) = g_base * (shape_a[L]*p_a + shape_v[L]*p_v + shape_av[L]*p_av)
and that single value scales ALL sink tokens at layer L.

p_(a,v,av) per clip = the routing actually used by the FULL runs:
  - soft router probs for the 300 DEV clips (router_v2_dev.csv),
  - one-hot on the recorded `routed` modality for the rest (the GT-task
    fallback, since the FULL runs were launched with the DEV router csv).

For each config we take every FULL clip's 28-layer gamma vector and show the
MEAN curve + 10-90 percentile band (the real spread across clips).
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
S5 = os.path.join(REPO, "results/qwen2_5_omni/stage5_intervention")
OUT = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/stage2_1b_sink_layer_proportion")

# (abbr, pair, title, g_base) — reverse schedule, best g_base per config
CONFIGS = [
    ("anxan", "AudioSetxActivityNet",     "AudioSet×ActivityNet",     4),
    ("anxyt", "AudioSetxYouTubeVOS",      "AudioSet×YouTube-VOS",     4),
    ("lixan", "LibriSpeechxActivityNet",  "LibriSpeech×ActivityNet",  5),
    ("lixyt", "LibriSpeechxYouTubeVOS",   "LibriSpeech×YouTube-VOS",  5),
    ("c508",  "common508",                "508 always-inert core",    5),
]
ONEHOT = {"AUDIO": (1.0, 0.0, 0.0), "VISUAL": (0.0, 1.0, 0.0), "AV": (0.0, 0.0, 1.0)}

# Soft router probs for the DEV clips (the only ones the FULL runs soft-routed).
dev = pd.read_csv(os.path.join(S5, "router_v2_dev.csv"))
dev_probs = {r.question_id: (r.p_audio, r.p_visual, r.p_av) for r in dev.itertuples()}

fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4.0 * len(CONFIGS), 4.2), sharey=False)
for ax, (abbr, pair, title, g_base) in zip(axes, CONFIGS):
    rev = dict(np.load(os.path.join(SCHED, f"rev_sched_{pair}.npz")))
    sa, sv, sav = rev["shape_a"], rev["shape_v"], rev["shape_av"]
    nL = sa.shape[0]
    df = pd.read_csv(os.path.join(S5, f"interv_sched_rev_{abbr}_g{g_base}_FULL_FULL.csv"))
    G = np.empty((len(df), nL), dtype=float)
    for i, row in enumerate(df.itertuples()):
        pa, pv, pav = dev_probs.get(row.question_id, ONEHOT[row.routed])
        G[i] = g_base * (sa * pa + sv * pv + sav * pav)
    L = np.arange(nL)
    mean = G.mean(0)
    lo, hi = np.percentile(G, 10, 0), np.percentile(G, 90, 0)
    ax.fill_between(L, lo, hi, color="tab:purple", alpha=0.20,
                    label="10–90% across clips")
    ax.plot(L, mean, color="tab:purple", marker="o", ms=3, lw=2,
            label="mean real γ(L)")
    ax.axhline(g_base, ls="--", lw=1.0, color="gray", label=f"flat γ={g_base}")
    ax.set_title(f"{title}\n(reverse, g_base={g_base}, n={len(df)})", fontsize=9)
    ax.set_xlabel("decoder layer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("real applied γ per layer")
fig.suptitle("REAL per-layer gamma applied on AVHBench FULL — one blended value "
             "per layer per clip, scaling all sink tokens\n"
             "γ(clip,L)=g_base·(shape_a·p_a+shape_v·p_v+shape_av·p_av); "
             "mean ± 10–90% band over all FULL clips (router as actually used)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(OUT, "gamma_schedule_real_applied_FULL.png")
fig.savefig(out, dpi=130)
print("saved", out)
# also print the mean curve numbers for the 508 config
print("508 mean real gamma per layer:")
np.set_printoptions(precision=2, suppress=True, linewidth=140)
