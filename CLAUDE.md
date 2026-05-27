# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This research project studies **attention sinks and hallucination in Audio-Visual large multimodal models**, primarily **Qwen2.5-Omni-7B**. There are two interlocking research lines:

1. **Hallucination-head attribution** — identify which decoder attention heads drive hallucinated tokens, via per-head zero-ablation. Adapted from the [Hallucination-Attribution](Hallucination-Attribution/) paper (ICLR 2025, originally LLaVA image-language models). Run separately on audio-only, video-only, and audio-visual probes, then combined to categorize heads by modality.

2. **Sink analysis** (`method/sink_analysis/`) — test whether Qwen2.5-Omni inherits the *encoder→LLM propagated-sink* phenomenon (high-norm encoder tokens becoming LLM attention sinks) for audio and video. This is grounded in three papers:
   - **To Sink or Not to Sink** (arXiv 2510.08510) — ViT high-norm "sink" tokens (L2 norm > τ=100, ~3–5/image) get ~7× LLM attention and propagate from encoder through the connector into the LLM. Source of the Figure 3A reproduction and the propagation thresholds.
   - **See What You Are Told: Visual Attention Sink / VAR** (arXiv 2503.03321) — sink tokens defined by massive activation in fixed hidden-state "sink dimensions"; basis of `identify_sink_dimensions.py`. The "wasted attention" counter-view.
   - **Probing Cross-modal Information Hubs in AV-LLMs / ASD** (arXiv 2605.10815) — AV-LLM sink tokens carry cross-modal info; Adaptive Sink-Guided Decoding on Qwen2.5-Omni. The closest AV precedent.

> **Earlier work used VideoLLaMA2** on AVHBench; that pipeline still lives in `method/videollama2/` and is documented under "Prior work (VideoLLaMA2)" below. New work targets Qwen2.5-Omni.

## Repository Structure

- **`method/qwen2_5_omni/`** — Primary pipeline (the `qwen_omni_venv` runs these):
  - `utils.py` — `load_omni` (composite `Qwen2_5OmniForConditionalGeneration`, talker disabled so only the **thinker** = encoders + LLM is used; `attn_implementation="eager"`, bfloat16, `device_map="auto"`), `build_conversation`, `prepare_inputs`, `find_modality_spans`, `omni_infer`.
  - `eval.py` — Stage 1: run inference on a dataset, sample entities, label tokens hallucinated vs. non-hallucinated (NLTK POS tagging, GT comparison). → `sampled_entities.json`.
  - `identify_halluc_head.py` — Stage 2: per-head zero-ablation attribution. → per-sample `.pth`, heatmaps, `attribution/heads/attribution_result.json`.
  - `head_attribution.py` — monkey-patches `GenerationMixin._sample` to zero-ablate each thinker head during generation (ports the VideoLLaMA2 version; enumerates `o_proj` via the thinker's decoder layers so encoder o_projs are untouched).
  - `analyze_attention_bias.py` — Stage 3: forward pass with `output_attentions=True`, measure hal vs. non-hal heads' attention to video / audio / text spans (`find_modality_spans`, single thinker forward).
  - `av_fusion_categorize_exp.py` / `av_fusion_scatter_exp.py` / `av_fusion_heatmap_exp.py` — combine the three single/dual-modality attribution runs (A=AudioSet, V=ActivityNet, AV=VGGSounder) to categorize heads by modality (8 disjoint cells from the `(in_H_A, in_H_V, in_H_AV)` signature at a pooled-percentile threshold).
- **`method/sink_analysis/qwen2_5_omni/`** — Sink-analysis stages:
  - `token_position_bookkeeping.py` — **Stage 0.2**: document audio/video/system/query/generated span indices per clip. → `token_layout.md`.
  - `encoder_norm_bimodality_exp.py` — **Stage 1.1**: per-token L2 norms at encoder output; caches `encoder_norms.npz` (`{audio,video}_{flat,offsets,names}`).
  - `encoder_norm_global_threshold_exp.py` — **Stage 1.1 (v2)**: global-percentile thresholding of the cached norms (no per-clip self-suppression).
  - `encoder_to_llm_propagation_exp.py` — **Stage 1.2**: reproduce Sink-or-Not Figure 3A — bin encoder norms (width 5), measure LLM attention to each modal token from generated query positions at layers 2 & 14, decide framing (A/B/C). See "Sink analysis" below.
  - `identify_sink_dimensions.py` — find LLM hidden-state sink dimensions (VAR-paper method). → `sink_dimensions.csv`.
- **`bash_scripts/qwen2_5_omni/{audioset,activitynet,vggsounder}/`** — Shell wrappers (eval / identify_halluc_head / analyze_attention_bias, with `_describe` and `_mcq` variants).
- **`Hallucination-Attribution/`** — Original ICLR 2025 LLaVA code (attribution methodology, baselines VCD/DOLA/HALC/OPERA, interventions ADHH/TFHH).
- **`results/qwen2_5_omni/`** — Outputs: `AudioSet[_describe]/`, `ActivityNet_describe/`, `VGGSounder_describe/` (each with `sampled_entities.json`, `attribution/`, `attention_bias/`), `categorize_exp/`, `sink_analysis/`.
- **`data/`** (gitignored) — see "Datasets" below.
- **`method/videollama2/`, `bash_scripts/videollama2/`, `results/videollama2/`** — Prior work (see bottom).

## Environment Setup

Three dedicated venvs at the repo root **`/nobackup3/le/AV_Hallucination/`**:

- **`qwen_omni_venv`** — primary; runs all `method/qwen2_5_omni/` and `method/sink_analysis/` scripts. (torch 2.6.0 + cu124, recent `transformers` with `Qwen2_5Omni*`, `qwen_omni_utils`.)
- **`activitynet_venv`** — ActivityNet data prep.
- **`videollama2_venv`** — prior VideoLLaMA2 pipeline (torch 2.2.0, transformers 4.42.3).

Invoke the venv Python directly, e.g.:

```bash
/nobackup3/le/AV_Hallucination/qwen_omni_venv/bin/python method/sink_analysis/qwen2_5_omni/encoder_to_llm_propagation_exp.py
```

The 7B model fits on a single 24 GB GPU for attention work; the bash wrappers default to `CUDA_VISIBLE_DEVICES=0,1,2,3` for eval throughput.

## Datasets

Per-modality probes (`data/`, gitignored). Each has `QA.json` + a media dir; `_describe` / `_mcq` variants share the same media and differ only in `QA.json` and the labels file.

| Dataset | `modal_type` | Modality | Media |
|---|---|---|---|
| AudioSet | `a` | audio-only | `data/AudioSet/audios/*.wav` |
| ActivityNet | `v` | video-only | `data/ActivityNet/videos/*.mp4` |
| VGGSounder | `av` | audio-visual (audio extracted from the video via `use_audio_in_video=True`) | `data/VGGSounder/videos/*.mp4` |

Also present: `AVHBench`, `AVCaps`, `MSVD`, `VGGSound` (used by the prior VideoLLaMA2 work and some eval variants).

## Running the Qwen2.5-Omni Pipeline

```bash
# Stage 1 — inference + entity labeling
bash bash_scripts/qwen2_5_omni/audioset/eval_describe.sh
# Stage 2 — per-head zero-ablation attribution
bash bash_scripts/qwen2_5_omni/activitynet/identify_halluc_head_describe.sh
# Stage 3 — attention-bias analysis
bash bash_scripts/qwen2_5_omni/vggsounder/analyze_attention_bias_describe.sh
```

`identify_halluc_head.sh` sets `INFLUENCE_SCORE` ∈ {`prob_diff` (default), `abs_prob_diff`, `log_prob_diff`}.

After running the three single/dual-modality attribution probes, combine them:

```bash
python method/qwen2_5_omni/av_fusion_scatter_exp.py --top_k 40
python method/qwen2_5_omni/av_fusion_heatmap_exp.py --top_k 40
python method/qwen2_5_omni/av_fusion_categorize_exp.py
```

### How head attribution works

`head_attribution.py` patches `GenerationMixin._sample`. At each generation step, if the next token is a target (hallucinated / non-hallucinated):
1. Save the current `DynamicCache` key/value lengths.
2. For every `(layer, head)`, register a `forward_pre_hook` on the thinker decoder layer's `o_proj` that zeroes that head's slice of the input.
3. Forward pass, compute the ablated probability, restore the cache, record the influence score.

The cache-restoration step resets `pkv.key_cache[i]` / `pkv.value_cache[i]` to pre-ablation lengths before each ablation pass — fixing the `DynamicCache`-growth tensor-size mismatch (see `ERROR.md`).

## Sink Analysis (Stages 0.2 → 1.1 → 1.2 → 1.3)

Goal: determine whether Qwen2.5-Omni's audio and video encoders inherit the **encoder→LLM propagated sink** phenomenon (Sink-or-Not, CLIP-ViT ~7×).

**Results are organized per stage** under `results/qwen2_5_omni/sink_analysis/`:
- `stage0_2_token_bookkeeping/` — `token_layout.md`
- `stage1_1_encoder_norms/` — `encoder_norms.npz` (the shared norm cache), `encoder_norm_histograms.png`/`_stats.txt` (bimodality), `encoder_norm_global_threshold.png`/`_summary.txt` (v2)
- `stage1_2_propagation/` — `figure_3a_crosslayer.png` (primary), `figure_3a_layer_trajectory.png`, `figure_3a_layers_2_14.png`, `propagation_summary.csv` (cross-layer per-bin), `propagation_layers.csv` (per-layer per-bin), `propagation_metrics.csv`, `propagation_decision.txt`; `_superseded/` holds old per-layer figures
- `sink_dimensions/` — `sink_dimensions.csv`/`.png`

Each script defaults its `--output_dir` (or `--output_md`) to its stage folder, and Stage 1.2 / 1.1-v2 default `--norms_npz` to `stage1_1_encoder_norms/encoder_norms.npz`.

**Stage 1.2** (`encoder_to_llm_propagation_exp.py`) reproduces Figure 3A on 300 AudioSet + 300 ActivityNet clips. It bins encoder L2 norms (fixed width 5; fine bins are essential — coarse bins wash out the heavy-tailed high-norm tail) and measures raw, head-averaged LLM attention to each modal token from the 20 generated query positions. **Primary metric = attention averaged across ALL decoder layers + heads** (`figure_3a_crosslayer.png`); per-layer views (a trajectory heatmap and a layers-2/14 panel) are supplementary. Verdict per modality, driven by `tail_ratio` (top-3 reliable high-norm bins, token-weighted ÷ mean_attn of norm<50 bins): **≥1.5 present | 1.0–1.5 ambiguous | <1.0 none**. (`high_low_ratio` = norm>p95&reliable ÷ norm<p50 is reported alongside.)

The framing decision the project hinges on:
- **(A)** both modalities propagate → symmetric two-population story (extends Sink-or-Not to AV).
- **(B)** only video propagates → asymmetric audio-deficit story.
- **(C)** neither → not a Sink-extension; project becomes a refinement of **ASD** (2605.10815).

`--from_csv <propagation_summary.csv>` recomputes metrics/verdicts and regenerates all figures with no GPU. Multi-GPU note: pass `--device_map balanced_low_0` (the default) and run with all 4 GPUs visible — plain `auto` packs the model onto GPU 0 and OOMs on `output_attentions`. Do **not** normalize attention to within-modality share — use raw per-token mean attention (small ~1e-4 values are expected). **Do not proceed past Stage 1.2 until the framing is confirmed.**

## Prior work (VideoLLaMA2)

The original effort ran the same attribution idea on **VideoLLaMA2** over AVHBench. Code in `method/videollama2/` (`eval.py`, `identify_halluc_head.py`, `head_attribution.py`, `analyze_attention_bias.py`); wrappers in `bash_scripts/videollama2/`; install the vendored package with `cd VideoLLaMA2 && pip install -e .` into `videollama2_venv` (torch 2.2.0, transformers 4.42.3). The Qwen2.5-Omni scripts are direct ports of these. Notable VideoLLaMA2-specific gotcha preserved in `analyze_attention_bias.py`: eager attention overflows to NaN for long sequences in float16 — run in bfloat16. AVHBench task categories: *Video-driven Audio Hallucination*, *Audio-driven Video Hallucination*, *AV Captioning*.
