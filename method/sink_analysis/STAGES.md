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
| 0.2 — token bookkeeping | `token_position_bookkeeping.py` | `stage0_2_token_bookkeeping/` |
| 1.0 — LLM sink dimensions | `identify_sink_dimensions.py` | `sink_dimensions/` |
| 1.1 — encoder-norm bimodality | `encoder_norm_bimodality_exp.py` | `stage1_1_encoder_norms/` |
| 1.1 v2 — global-threshold | `encoder_norm_global_threshold_exp.py` | `stage1_1_encoder_norms/` |
| 1.2 — encoder→LLM propagation | `encoder_to_llm_propagation_exp.py` | `stage1_2_propagation/` |

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

## Status & next
- Stages 0.2, 1.0, 1.1, 1.1 v2 complete. Stage 1.2 code complete and run; **framing pending confirmation**.
- Open methodological question surfaced by 1.2: cross-layer averaging vs early-layer-localized signal (video propagates at layer 2 but not on average).
- **Stage 1.3** (LLM-emerged vs propagated sink disentanglement, using the {458, 2570, 3197} sink dims) is **not** started — do not proceed until the framing is confirmed.
