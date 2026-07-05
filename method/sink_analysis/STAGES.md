# Sink Analysis — Stage Log (Qwen2.5-Omni)

Investigates whether **Qwen2.5-Omni-7B** inherits the *encoder → LLM propagated-sink*
phenomenon — high-norm encoder tokens becoming disproportionately-attended "sinks"
inside the LLM during decoding — separately for the **audio** and **video** encoders.

**Reference papers**
- *To Sink or Not to Sink* (arXiv 2510.08510) — ViT high-norm tokens (L2 norm > τ=100, ~3–5/image) get ~7× LLM attention and propagate encoder→LLM. Source of the Figure 3A reproduction and the propagation thresholds.
- *See What You Are Told: Visual Attention Sink / VAR* (arXiv 2503.03321) — sink **dimensions** (fixed high-activation hidden channels). Basis of Stage 1.0.
- *Probing Cross-modal Information Hubs in AV-LLMs / ASD* (arXiv 2605.10815) — AV-LLM sink tokens carry cross-modal info. The fallback framing.

**Project framing decided by Stage 1.2**
- **(A)** both modalities propagate → symmetric two-population story (extends Sink-or-Not to AV).
- **(B)** only video propagates → asymmetric audio-deficit story.
- **(C)** neither propagates clearly → not a Sink-extension; project becomes a refinement of **ASD**.

**Environment** — venv `/nobackup3/le/AV_Hallucination/qwen_omni_venv`. Model loaded
via `utils.load_omni` (thinker only, talker disabled, `attn_implementation="eager"`,
bf16). Attention-capturing stages need all 4 GPUs (`--device_map balanced_low_0`).

## Datasets (per modality)
| Modality | `modal_type` | Dataset | Media | Encoder tokens/clip |
|---|---|---|---|---|
| audio | `a` | AudioSet | `data/AudioSet/audios/*.wav` | ~245 (≈25 tok/s) |
| video | `v` | ActivityNet | `data/ActivityNet/videos/*.mp4` | ~1320 (frames × patches; variable) |
| audio+video | `av` | VGGSounder | `data/VGGSounder/videos/*.mp4` | ~1746 + ~1748 |

300 clips per modality (audio = AudioSet, video = ActivityNet) are used throughout 1.1/1.2.

## Stage map
| Stage | Script | Result folder |
|---|---|---|
| 0.1 — base-LLM sink dims | `identify_sink_dimensions_base.py` + `identify_sink_dimensions.py` | `sink_dimensions/` |
| 0.2 — token bookkeeping | `token_position_bookkeeping.py` | `stage0_2_token_bookkeeping/` |
| 1.0 — LLM sink dimensions | `identify_sink_dimensions.py` | `sink_dimensions/` |
| 1.1 — encoder-norm bimodality | `encoder_norm_bimodality_exp.py` | `stage1_1_encoder_norms/` |
| 1.1 v2 — global-threshold | `encoder_norm_global_threshold_exp.py` | `stage1_1_encoder_norms/` |
| 1.2 — encoder→LLM propagation | `encoder_to_llm_propagation_exp.py` | `stage1_2_propagation/` |
| 1.3 — hidden-state dim signatures | `stage1_3_dimension_signatures.py` (+ `_encoder_space`) | `stage1_3_dim_signatures/` |
| 2.1 — layer-wise sink counts | `stage2_1_layer_sink_counts.py` | `stage2_1_layer_sinks/` |
| 2.4 — temporal sink positions | `stage2_4_temporal_sink_positions.py` | `stage2_4_temporal_sinks/` |

All result folders live under `results/qwen2_5_omni/sink_analysis/`. Each script
defaults its `--output_dir` to its stage folder.

---

## Stage 0.2 — Token-position bookkeeping
**Purpose** Document the absolute index span of each token category (system /
audio / video / query / generated) in the full `prompt + generated` sequence, so
later stages can slice attention over modality spans unambiguously.

**Run** `python token_position_bookkeeping.py` → `stage0_2_token_bookkeeping/token_layout.md`

**Findings** (greedy, 3 clips/dataset)
- **AudioSet**: `system[0,39) → audio[44,294) (~245 tok) → query (7 tok) → generated`.
- **ActivityNet**: `system[0,39) → video[44,2156) (~1320 tok, range 792–2112) → query → generated`.
- **VGGSounder** (`av`): audio and video spans are **interleaved** (`video[44,1880)`, `audio[45,1879)`), ~1746/1748 tokens each.
- Conventions: audio ≈ 25 tokens/sec; video tokens = frames × spatial patches (set by `fps`/`max_pixels` in `build_conversation`). audio-only clips have no video span and vice-versa.

---

## Stage 1.0 — LLM sink dimensions (Kang et al.)
**Purpose** Locate the LLM hidden-state channels that carry abnormally large
activation across most layers — prerequisite for the later "LLM-emerged vs
propagated" sink classification (Stage 1.3).

**Method** Per layer, flag dims with `mean|x| > MEDIAN_MULT × median`; a dim is a
sink dim if flagged in > `LAYER_FRAC` of layers.

**Run** `python identify_sink_dimensions.py` → `sink_dimensions/sink_dimensions.{csv,png}`

**Findings** **3 sink dimensions** out of 3584: **{458, 2570, 3197}**, flagged in
55–83 % of layers, mean magnitudes 16–52. Within the paper's expected 2–5 range.

---

## Stage 1.1 — Encoder-norm bimodality
**Purpose** Does each encoder emit a *bimodal* output-token-norm distribution (the
high-norm-outlier signature of encoder-side sinks)?

**Method** Per-token L2 norm at encoder output; per-clip robust z = (norm − median)/IQR;
count tokens with z > k for k ∈ {3,5,7}. Caches per-token norms to
`encoder_norms.npz` (consumed by 1.1 v2 and 1.2).

**Run** `python encoder_norm_bimodality_exp.py`
→ `stage1_1_encoder_norms/{encoder_norms.npz, encoder_norm_histograms.png, encoder_norm_stats.txt}`

**Findings** (300 clips/modality)
| Modality | tokens | median norm | p99 | max | robust-z verdict |
|---|---|---|---|---|---|
| audio | 74,830 | 48.75 | 112.5 | 171 | Unimodal at all k (median 0 outliers/clip) |
| video | 382,175 | 44.5 | 94.5 | 193 | **Bimodal at k=3** (median 1 outlier/clip); Unimodal at k≥5 |

Caveat: the per-clip robust-z self-suppresses (a clip's own outliers inflate its IQR),
which motivated v2.

---

## Stage 1.1 v2 — Global percentile thresholds
**Purpose** Re-do the bimodality check with a single **global** threshold per
modality (no per-clip self-suppression), including the Sink-or-Not absolute τ=100.

**Run** `python encoder_norm_global_threshold_exp.py`
→ `stage1_1_encoder_norms/encoder_norm_global_threshold.{png,_summary.txt}` (reads `encoder_norms.npz`)

**Findings** — at the paper's **τ=100** criterion both modalities show a high-norm tail:
| Modality | τ95 | τ99 | τ=100: median tokens/clip | % clips with zero | verdict |
|---|---|---|---|---|---|
| audio | 88 | 112 | 5 | 15 % | Propagated-sink signature present |
| video | 77.5 | 94.5 | 5 | 4.3 % | Propagated-sink signature present |

So **both encoders produce high-norm tokens** at the encoder side. Whether those
tokens *propagate* into LLM attention is the Stage 1.2 question.

---

## Stage 1.2 — Encoder → LLM propagation (the framing decision)
**Purpose** Reproduce Sink-or-Not **Figure 3A**: do high-norm encoder tokens receive
disproportionate LLM attention during decoding?

**Method** For 300 clips/modality, greedy-decode 20 tokens with `output_attentions`.
For each modal token, average its received attention over **all decoder layers + all
heads + the 20 generated query positions** (PRIMARY = cross-layer). Bin by encoder L2
norm (fixed width 5; fine bins are essential — coarse bins wash out the heavy-tailed
high-norm region). Raw per-token attention, **no renormalization**.

**Verdict driver** `tail_ratio` = (top-3 reliable high-norm bins, token-weighted
mean_attn) ÷ (mean_attn of norm<50 bins): **≥1.5 present | 1.0–1.5 ambiguous | <1.0 none**.
`high_low_ratio` (norm>p95&reliable ÷ norm<p50) is reported alongside.

**Run**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 qwen_omni_venv/bin/python \
  method/sink_analysis/qwen2_5_omni/encoder_to_llm_propagation_exp.py
# recompute metrics/figures from cached CSVs, no GPU:
qwen_omni_venv/bin/python method/sink_analysis/qwen2_5_omni/encoder_to_llm_propagation_exp.py \
  --from_csv results/qwen2_5_omni/sink_analysis/stage1_2_propagation/propagation_summary.csv
```
**Outputs** (`stage1_2_propagation/`): `figure_3a_crosslayer.png` (PRIMARY),
`figure_3a_layer_trajectory.png` (per-(layer, norm-bin) heatmap), `figure_3a_layers_2_14.png`
(tertiary per-layer), `propagation_{summary,layers,metrics}.csv`, `propagation_decision.txt`.

**Findings — current run** (cross-layer primary)
| Modality | layer | pearson_r | high/low | **tail3/low** | verdict |
|---|---|---|---|---|---|
| audio | all (primary) | −0.60 | 0.95× | **0.51×** | none |
| audio | 2 | −0.53 | 0.81× | 0.63× | none |
| video | **all (primary)** | −0.21 | 0.93× | **1.21×** | **ambiguous** |
| video | 2 | +0.65 | 2.13× | **4.55×** | **present** |
| video | 14 | +0.03 | 1.00× | 1.25× | ambiguous |

- **Audio**: attention concentrates on the **lowest-norm bin** (norm 0–10 attn ≈ 9.5e-4, ~4× the secondary window) — a positional/BOS-style sink, *not* a high-norm propagated sink. No propagation.
- **Video**: propagation is a strong **early-layer** effect (layer 2 tail_ratio **4.55×**, pearson +0.65) that is **diluted to ambiguous (1.21×) when averaged across all 28 layers**, because most mid/late layers don't show it.

**Current framing recommendation: (C)** by the cross-layer primary metric — but note
the layer-2 evidence alone supports **(B)** (video propagates, audio doesn't). The
audio-vs-video asymmetry is robust regardless of metric.

> ⚠ The numbers above are from the run currently on disk and have **not** been
> confirmed against a clean full GPU rerun after the device/OOM fixes. The framing
> is **not** finalized — Stage 1.3 is gated on user confirmation.

---

---

## Stage 0.1 — Base-LLM-inherited sink dimensions (text-only Qwen2.5-7B)
**Script** `identify_sink_dimensions_base.py` (and the corresponding Omni run
`identify_sink_dimensions.py`, both now RMSNorm pure-normalization convention).

**Method** Per-token |RMSNorm(x)[d]| at every hidden state (embedding + each
decoder layer); per-layer flag `> 20× median`; sink dim if flagged in >50% of
layers. The previous Stage 1.0 used raw |x|; this is the corrected version.

**Findings** (100 prompts/100 audio clips, threshold 20×):
- **D_sink_base = {458, 2570}** (text Qwen2.5-7B, 37.9× ratio, flagged 86–93% of layers)
- **D_sink_omni = {458, 2570, 3197}** (Qwen2.5-Omni audio, 22.6× ratio).
- **Inherited from base: {458, 2570}**; **Omni-only candidate: {3197}**.
- Hidden dim matches (3584/3584, 28 decoder layers each).

But after Stage 1.3 distinctiveness analysis, **3197 is NOT a true sink register** —
see below.

---

## Stage 1.3 — Hidden-state dimension signatures (per-modality, distinctiveness)
**Script** `stage1_3_dimension_signatures.py`. 100 VGGSounder AV clips,
RMSNorm pure (matches Stage 0.1). Layers 2 (video peak), 14, 21 (audio peak).
Populations: P_prop (encoder norm >100), P_llm (max over D_sink ≥20), random.

**Final sink set: D_sink = {458, 2570}**. Dim 3197 dropped — distinctiveness
analysis showed it has distinctiveness < 1 to ALL P_prop populations and is
broadly active (highest in `(video, non_sink)` at 12.79). It's a video-content
dim, not a sink register. With 3197 in the gate, ~40% of P_llm_video were
3197-only false positives (P_llm_video L2: 4805 → 3100 = −35.5%; L21:
96922 → 55106 = −43.1%).

**Per-modality dimensional signatures (L2):**
- `P_llm_video`, `P_llm_audio`: distinctively load {458, 2570} (ratios 2.8–3.0×).
- `P_prop_video`: **NO** distinctive dim. Suppressed on {458, 2570} (act ~2.7).
- `P_prop_audio`: 4 weakly distinctive non-sink dims [2030, 2730, 2890, 3070].
- Dim 3197 distinctiveness: P_prop_video 0.83, P_prop_audio 0.97 (NOT distinct).

**Final conclusion (Stage 1.3):** Qwen2.5-Omni's propagated sinks do **NOT**
form a dedicated hidden-state register dimension — unlike LLaVA's
{982, 2494, 3263}. Propagation appears in **attention** (Stage 1.2's video
early-sharp peak) but **not** in a distinctive hidden-state register. The
dimensional separation observed between P_prop_video and P_llm_video is
**separation by absence**: P_prop_video fails to load the inherited register
dims {458, 2570}, rather than activating its own propagated register.

**Outputs** (`stage1_3_dim_signatures/`): `asd_sink_dim_per_modality.png`,
`distinctiveness_top10.csv` / `distinctiveness_verdict.txt`,
`em_inh_trajectory.png` / `.csv`, `dimension_signatures.csv`,
`sink_dim_comparison.png`, `stage1_3_decision.txt`. `profiles.npz` (+ backup
`profiles_dsink_3dims.npz` with the old 3-dim D_sink) supports `--replot_only`
for no-GPU figure tweaks.

---

## Stage 2.1 — Layer-wise LLM sink counts (ASD Fig. 7 reproduction)
**Script** `stage2_1_layer_sink_counts.py`. 300 VGGSounder AV clips, audio +
video both present, D_sink = {458, 2570}, τ=20, RMSNorm pure. Per-clip per-layer
forward hooks (no `output_hidden_states` retention) keep memory small.

**Method** For each decoder layer L, count tokens crossing the LLM-sink
criterion (`max(|RMSNorm(x)[d]|) ≥ 20` over D_sink), separately for audio and
video token positions, separately per clip. Also intersect the per-layer sink
mask with the **propagated** mask (encoder norm > 100) to track
P_prop ∩ P_llm@L — i.e. how many of the encoder-propagated tokens additionally
satisfy the LLM-sink criterion at each layer.

**Findings (n=300 VGGSounder)**
- **Audio LLM-emerged sinks** peak at L26 (~221/clip); at the Stage 1.2
  audio-peak layer **L21 = ~100 sinks/clip** out of ~136 audio LLM tokens.
- **Video LLM-emerged sinks** peak at L26 (~1194/clip); the Stage 1.2 video
  early-sharp layer L2 also shows substantial count (~3100/clip across the run,
  per the S1.3 reference table).
- **~99 % of P_prop tokens become LLM-sinks at deep layers** for both
  modalities — the propagated population is almost entirely absorbed into
  the LLM-sink population by the deep layers.

**Outputs** (`stage2_1_layer_sinks/`): `layer_sink_counts.csv`,
`figure_2_1_llm_emerged.png` (audio/video per-layer means ± std),
`figure_2_1_prop_overlap.png` (P_prop ∩ P_llm @ L with P_prop baseline),
`per_clip_counts.npz` (replot-friendly per-clip per-layer arrays — also
consumed by Stage 2.4's cross-stage proportion check), `stage2_1_decision.txt`.

---

## Stage 2.4 — Temporal position of audio late-spread sinks
**Script** `stage2_4_temporal_sink_positions.py`. 300 AudioSet clips,
audio-only forward (`modal_type="a"`), D_sink = {458, 2570}, τ=20, RMSNorm
pure. Primary layer **L21** (Stage 1.2's audio late-spread peak); also captured
at L27 for the deep-layer saturation comparison.

**Question** Are audio L21 LLM-emerged sinks (i) **positional** (fixed
locations regardless of content), (ii) **content-conditional** (clustered at
acoustic events, varying per clip), or (iii) **distributed** (no temporal
structure)?

**Method** Per clip, hook L21's output; sink mask over the audio span only;
record sink in-span indices + normalized positions in [0, 1). Aggregates:
pooled marginal, per-clip KS vs uniform, per-clip IQR, per-clip sink count
**and per-clip sink PROPORTION = sink_count / audio_span_length** (the cleaner
content-dependence diagnostic — span-length variation that confounds the
absolute count cancels out). Cross-stage: load Stage 2.1's
`per_clip_counts.npz`, compute its L21 audio sink proportion on VGGSounder,
compare to AudioSet's.

**Findings (revised with proportion analysis, n=300 AudioSet)**
| Quantity | Value |
|---|---|
| audio span length | 248.8 ± 8.6 tokens (essentially fixed) |
| per-clip sink count | 183 ± 21 |
| **per-clip sink proportion** | **0.736 ± 0.081** (range 0.52–0.93, 95% ≈ 0.58–0.90) |
| **proportion std/mean** | **0.110** (just above the 0.10 saturation cutoff) |
| pos-0 is a sink | 69 % of clips (contributes <1 % of the marginal) |
| pooled marginal | flat, low (<0.05)=6.1 %, middle (0.4–0.6)=19.7 %, high (>0.95)=4.0 % |
| per-clip KS vs uniform | median 0.062 |
| per-clip IQR | median 0.494 (≈ uniform's 0.5) |
| Stage 2.1 cross-check (L21 VGGSounder) | ~100 sinks/clip — same proportion modulo shorter audio span |

**Interpretation — revised**
- **Within-clip structure:** sinks are uniformly distributed along the audio
  time axis (KS≈0.06, IQR≈0.5). No clustering at clip start / middle / end.
  Position-0 sink frequency (69%) is a small per-clip footnote, not a marginal
  bump.
- **Across-clip structure:** sink **proportion** varies meaningfully across
  clips (0.110 std/mean, range 0.52–0.93). This cross-clip variance reflects
  genuine sink-rate variance, not span variation (span std/mean = 0.034).
- **Cross-dataset consistency:** AudioSet's 0.736 proportion on 249-token spans
  ≡ Stage 2.1's ~100 sinks/clip on VGGSounder's shorter audio spans — the
  100-vs-183 absolute-count gap is span-length, not rate.
- **Verdict:** audio late-spread at L21 is **NOT positional** (rules out the
  "audio BOS register" hypothesis) and **NOT content-localization within
  clips** (rules out "sinks track acoustic events"). It IS a **clip-level
  saturation phenomenon** — a diffuse sink fill that varies in intensity
  across clips. Stage 2.5 (acoustic correlation) is now well-motivated to
  identify what drives the cross-clip variance.

**Outputs** (`stage2_4_temporal_sinks/`): `temporal_sink_distribution.png`
(pooled marginal), `per_clip_distribution_shape.png` (KS + IQR),
`per_clip_count_distribution.png`, `per_clip_proportion_distribution.png`
(sink proportion + span length), `temporal_sink_distribution_no_pos0.png`
(position-0 sensitivity), `temporal_sink_distribution_L27.png` (deep-layer
comparison), `per_clip_temporal_stats.csv` (incl. `sink_proportion` column),
`per_clip_temporal_arrays.npz`, `stage2_4_decision.txt`.

---

## Status & next
- Stages complete: 0.1 / 0.2 / 1.1 / 1.1 v2 / 1.2 / 1.3 / 2.1 / 2.4.
- **Stage 1.2 framing**: per-layer pattern classification (Sharp/Spread/None).
  Video = early-sharp at L2 (top3 5.28×, p95 ~2). Audio = late-spread at L21
  (top3 ~0.9, p95 ~1.7). Framing B (asymmetric mechanism) per the per-layer
  pattern logic — but at the hidden-state register level (Stage 1.3) the
  asymmetry is "separation by absence", not LLaVA-style dedicated registers.
- **Stage 1.3 final**: D_sink = {458, 2570}; no propagated register dim;
  3197 reclassified as a broadly-active video-content channel.
- **Stage 2.1**: deep-layer LLM-sink saturation; ~99 % of P_prop tokens
  absorbed into the LLM-sink population at deep layers (both modalities).
- **Stage 2.4**: audio L21 sinks are temporally **distributed within each
  clip** (KS 0.062, IQR 0.494) but **clip-level saturation rate varies**
  (proportion 0.736 ± 0.081, std/mean 0.110). Diffuse sink fill with
  cross-clip rate variance. Rules out positional ("audio BOS register") and
  within-clip content localization. Cross-stage check with Stage 2.1 confirms
  proportion consistency across AudioSet/VGGSounder.
- **Next — Stage 2.5 (motivated by 2.4)**: acoustic correlation. What
  clip-level properties (energy, spectral content, semantic class,
  speech-vs-noise, etc.) predict the cross-clip variance in audio sink
  proportion? Confirm scope before starting.
