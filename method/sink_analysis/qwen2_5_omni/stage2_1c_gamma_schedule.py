"""stage2_1c_gamma_schedule.py

(A) Per-layer gamma schedule for gamma_a / gamma_v, derived from the average
    per-layer LLM-emerged sink proportion across the 2 datasets of each
    modality (audio = AudioSet+LibriSpeech, visual = ActivityNet+YouTube-VOS).
    Intervene more strongly where sinks concentrate -> gamma(L) scaled by the
    layer's sink share.
(B) Redraw the per-layer sink-proportion plot WITHOUT VGGSounder (4 proxies).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[3]
OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_1b_sink_layer_proportion"
Z = np.load(OUT / "sink_layer_counts.npz")
nL = len(Z["AudioSet"])
layers = np.arange(nL)

def prop(name):
    c = Z[name].astype(float)
    return c / c.sum()

# Average proportion per modality (mean of the 2 proxies' proportions)
audio_prop  = (prop("AudioSet") + prop("LibriSpeech")) / 2
visual_prop = (prop("ActivityNet") + prop("YouTubeVOS")) / 2

# ----- (A) gamma schedule -----
# gamma_m(L) = G_BASE * prop_m(L) / mean(prop_m)  -> averages to G_BASE,
# layer-modulated by where that modality's sinks concentrate.
G_BASE = 3.0
gamma_a = G_BASE * audio_prop  / audio_prop.mean()
gamma_v = G_BASE * visual_prop / visual_prop.mean()
print("per-layer gamma schedule (G_BASE=3.0):")
print("layer  audio_prop%  visual_prop%  gamma_a  gamma_v")
for L in range(nL):
    print(f"{L:>4}  {audio_prop[L]*100:8.2f}  {visual_prop[L]*100:9.2f}  "
          f"{gamma_a[L]:7.3f}  {gamma_v[L]:7.3f}")
np.savez(OUT / "gamma_schedule.npz", gamma_a=gamma_a, gamma_v=gamma_v,
         audio_prop=audio_prop, visual_prop=visual_prop, G_BASE=G_BASE)

# plot the schedule
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(layers, gamma_a, "-o", ms=4, color="#1f77b4", label="gamma_a(L)  (audio)")
ax.plot(layers, gamma_v, "-o", ms=4, color="#2ca02c", label="gamma_v(L)  (visual)")
ax.axhline(G_BASE, color="gray", ls="--", lw=1, label=f"G_BASE={G_BASE}")
ax.set_xlabel("decoder layer"); ax.set_ylabel("scheduled gamma")
ax.set_title("Per-layer gamma schedule from avg sink proportion\n"
             "gamma_m(L) = G_BASE * prop_m(L) / mean(prop_m)", fontsize=12)
ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(range(0, nL, 2))
fig.tight_layout(); fig.savefig(OUT / "gamma_schedule.png", dpi=300)
print(f"\nwrote {OUT/'gamma_schedule.png'} and gamma_schedule.npz")

# ----- (B) redraw without VGGSounder -----
DS = [("AudioSet", "audio", "#1f77b4"), ("LibriSpeech", "audio", "#4ba3e3"),
      ("ActivityNet", "visual", "#2ca02c"), ("YouTubeVOS", "visual", "#7fcf7f")]
fig2, ax2 = plt.subplots(figsize=(10, 6))
for name, mod, color in DS:
    ax2.plot(layers, prop(name) * 100, "-o", ms=4, color=color,
             label=f"{name} ({mod})")
# bold modality means
ax2.plot(layers, audio_prop * 100, "-", lw=2.5, color="#08306b", alpha=0.8, label="audio mean")
ax2.plot(layers, visual_prop * 100, "-", lw=2.5, color="#006d2c", alpha=0.8, label="visual mean")
ax2.set_xlabel("decoder layer"); ax2.set_ylabel("% of dataset's LLM-emerged sinks at this layer")
ax2.set_title("Per-layer LLM-emerged sink distribution — 4 proxy datasets (no VGGSounder)\n"
              "(D_sink={458,2570}, tau=20; each curve sums to 100%)", fontsize=12)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_xticks(range(0, nL, 2))
fig2.tight_layout(); fig2.savefig(OUT / "sink_layer_proportion_no_vgg.png", dpi=300)
print(f"wrote {OUT/'sink_layer_proportion_no_vgg.png'}")
