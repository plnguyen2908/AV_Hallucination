# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This research project investigates **hallucination attribution in Audio-Visual (AV) large multimodal models**, specifically VideoLLaMA2. The goal is to identify which attention heads are responsible for hallucinated outputs on the AVHBench benchmark, adapting the methodology from the [Hallucination-Attribution](Hallucination-Attribution/) paper (originally for LLaVA image-language models) to AV models.

## Repository Structure

- **`VideoLLaMA2/`** — Vendored copy of the VideoLLaMA2 model codebase (installed as a local package). The `videollama2` Python package is imported from here.
- **`method/videollama2/`** — Core pipeline scripts for this project:
  - `eval.py` — Stage 1: Run VideoLLaMA2 inference on AVHBench, sample entities, classify tokens as hallucinated vs. non-hallucinated using NLTK POS tagging.
  - `identify_halluc_head.py` — Stage 2: Run zero-ablation attribution per attention head for each hallucinated/non-hallucinated token.
  - `head_attribution.py` — Monkey-patches `transformers.GenerationMixin._sample` to perform per-head zero-ablation during generation.
- **`bash_scripts/videollama2/`** — Shell wrappers for running the pipeline.
- **`Hallucination-Attribution/`** — The original paper's code (LLaVA-based), used as reference for attribution methodology, baselines, and analysis scripts.
- **`results/videollama2/AVHBench/`** — Output directory for inference results and attribution `.pth` files + heatmap PNGs.
- **`data/AVHBench/`** — Dataset (gitignored). Expected structure: `QA.json` and `videos/` directory.

## Environment Setup

The project uses a dedicated venv at `/nobackup/le/AV_Hallucination/videollama2_venv` (on the compute cluster). Install VideoLLaMA2 as a local editable package:

```bash
cd VideoLLaMA2
pip install -e .
```

Key dependency pinning (see `VideoLLaMA2/requirements.txt`):
- `torch==2.2.0`, `transformers==4.42.3`, `accelerate==0.26.1`
- `decord`, `librosa`, `pytorchvideo` for AV processing

## Running the Pipeline

### Stage 1 — Inference & Entity Extraction

```bash
bash bash_scripts/videollama2/eval.sh
```

Runs `method/videollama2/eval.py`. Samples `N_PER_CATEGORY` (default 100) entries per AVHBench task category, runs VideoLLaMA2 inference, extracts noun/yes/no entities, and labels them as hallucinated or not by comparing against ground truth. Output: `results/videollama2/AVHBench/sampled_entities.json`.

### Stage 2 — Head Attribution

```bash
bash bash_scripts/videollama2/identify_halluc_head.sh
```

Runs `method/videollama2/identify_halluc_head.py`. For each sample with hallucinated/non-hallucinated tokens, performs zero-ablation of each attention head's output during generation and measures the probability change. Output: per-question `.pth` files and heatmap PNGs in `results/videollama2/AVHBench/attribution/`.

### Influence Score Options

Set `INFLUENCE_SCORE` in `identify_halluc_head.sh`:
- `prob_diff` — signed probability difference (default)
- `abs_prob_diff` — absolute probability difference
- `log_prob_diff` — log-probability difference

## How Head Attribution Works

`head_attribution.py` monkey-patches `GenerationMixin._sample` with `zero_ablation_sample`. At each generation step, if the next token is a target (hallucinated or non-hallucinated), it:
1. Saves the current `DynamicCache` key/value lengths.
2. For every `(layer, head)` pair, registers a `forward_pre_hook` on the corresponding `o_proj` module that zeroes out that head's slice of the input.
3. Runs a forward pass, computes the ablated probability, restores the cache, and records the influence score.

**Known issue (see `ERROR.md`):** A tensor size mismatch (`RuntimeError: size of tensor a (3003) must match tensor b (2219)`) occurs during ablation passes when `DynamicCache` grows beyond the original sequence length. The cache restoration logic in `head_attribution.py` addresses this by resetting `pkv.key_cache[i]` and `pkv.value_cache[i]` to their pre-ablation lengths before each ablation pass.

## AVHBench Task Categories

- `Video-driven Audio Hallucination` — model asked if audio matches video
- `Audio-driven Video Hallucination` — model asked if video matches audio
- `AV Captioning` — model asked to describe the video

Entity labels are extracted via NLTK POS tagging (nouns + yes/no tokens). For hallucination tasks, ground truth label (Yes/No) + response nouns are compared; for captioning, GT caption nouns are used.

## Hallucination-Attribution Reference Code

The `Hallucination-Attribution/` directory contains the original ICLR 2025 paper code for LLaVA models. Useful reference for:
- Attribution methodology: `LLaVA/bash_scripts/attribute.sh`
- Baseline comparisons (VCD, DOLA, HALC, OPERA): `baselines/`
- Intervention methods (training-free ADHH, targeted finetuning TFHH): `baselines/bash_scripts/`
- Analysis scripts (attention bias, inheritance, JS divergence): `LLaVA/bash_scripts/analysis/`

## Attention Bias Analysis (Stage 3 — To Be Adapted)

**Reference script**: `Hallucination-Attribution/LLaVA/eval_scripts/analyze_attention_bias.py`

**Purpose**: After identifying hallucination heads via zero-ablation (Stage 2), this analysis tests the hypothesis that hallucination heads systematically over-attend to visual/audio tokens (image bias) while neglecting text context — and that non-hallucination heads show the opposite pattern. This validates *why* the identified heads cause hallucination.

### What the LLaVA script does

1. **Input**: CHAIR evaluation results JSON (captions with `CHAIRs == 1` are hallucinated), the identified `hal_heads` / `non_hal_heads` from `attribution_result.json`, and the image folder.

2. **For each hallucinated caption**, runs a full forward pass with `output_attentions=True` to capture raw attention weight matrices `[batch, head, seq, seq]` at every layer.

3. **Token segmentation**: For each generated token that is an object word in the hallucinated caption, identifies its position in the sequence and splits the preceding context into:
   - **Image tokens**: a fixed block of 576 tokens starting at `IMAGE_TOKEN_INDEX`
   - **Text tokens**: everything after the image block up to the current token

4. **For each (layer, head) in `hal_heads` and `non_hal_heads`**, reads the attention row of the current token, sums separately over image positions and text positions, and averages across the top-k hal/non-hal heads.

5. **Output**: Saves `attention_statics.pth` (list of `[hal_img_attn, hal_txt_attn, nonhal_img_attn, nonhal_txt_attn]` per hallucinated token), then plots a grouped bar chart comparing image vs. text attention for both head groups.

### Key design choices to adapt for VideoLLaMA2

| LLaVA | VideoLLaMA2 adaptation needed |
|---|---|
| Single image token block, fixed length 576 | **Two modality blocks**: video tokens (variable length, depends on num_frames × spatial patches) and audio tokens (variable length). Must locate each block dynamically from the input token sequence using `DEFAULT_VIDEO_TOKEN` / `DEFAULT_AUDIO_TOKEN` positions and the model's multimodal preprocessing output. |
| `output_attentions=True` with eager attention | VideoLLaMA2 must also use `attn_implementation="eager"` to get attention tensors. **NaN issue**: eager attention overflows for long sequences (2201 tokens) in float16. Workaround: run this analysis in **bfloat16** (`model.to(torch.bfloat16)`) or cast QK products to float32 before softmax. Alternatively, only use this analysis pass on short sequences. |
| COCO CHAIR metric for hallucination labels | Use `hallucinated_entities` from `sampled_entities.json` (already computed in Stage 1). Match each generated token against the entity token list. |
| `mscoco_generated_words_first_token` — takes the second tokenizer token of each word (skips BOS) | Replicate for VideoLLaMA2 tokenizer (Qwen2): `tokenizer(word)['input_ids']` may not have a BOS; use index `[0]` or check tokenizer behavior. |
| Loads `hal_heads` / `non_hal_heads` from `attribution_result.json` | Load from `results/videollama2/AVHBench/attribution/heads/attribution_result.json`. The key names differ: use `hal_heads_contrastive` or `hal_heads_mean` as the hal head list. |
| Processes up to 100 samples | Tune to AVHBench sample count. |

### What a VideoLLaMA2 version should measure

For each hallucinated entity token generated by VideoLLaMA2:
- **Video attention** (sum of attention weights over all video token positions)
- **Audio attention** (sum over all audio token positions)
- **Text attention** (sum over all text/instruction token positions)

Compared separately for hal heads vs. non-hal heads. The expected finding (if the LLaVA result generalizes) is that hal heads over-attend to one modality (video or audio) at the expense of cross-modal grounding.

### Implementation notes for VideoLLaMA2 token layout

After `prepare_inputs_labels_for_multimodal`, the embedding sequence is approximately:
```
[system prompt tokens] [video tokens ~2048] [audio tokens ~variable] [instruction text tokens]
```
The exact boundaries are not returned by the model API — they must be inferred by running `prepare_inputs_labels_for_multimodal` separately and tracking where `None` placeholders (modal tokens) were replaced. Alternatively, count from the known modal token counts: video frames × spatial patch count, audio spectrogram frames × patch count.
