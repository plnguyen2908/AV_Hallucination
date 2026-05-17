# Qwen2.5-Omni Port of the AV Hallucination Pipeline

## Context

The repo currently runs the hallucination-attribution pipeline (inference →
per-head zero-ablation → contrastive head identification → attention-bias
analysis) against **VideoLLaMA2** under `method/videollama2/` and
`bash_scripts/videollama2/`. This document plans the parallel port to
**Qwen2.5-Omni** (`Qwen/Qwen2.5-Omni-7B`) — same datasets, same outputs, same
downstream scripts — so the two models can be compared head-to-head. First
pass targets AudioSet only (3 variants: hallucination, describe, mcq) but the
Python code stays dataset-agnostic so more wrappers can be added later.

Three differences from VideoLLaMA2 drive most of the porting work:

1. **Composite architecture.** `Qwen2_5OmniForConditionalGeneration` wraps
   `thinker` (encoders + LLM), `talker` (audio decoder), and `token2wav`. For
   text-only eval we call `model.disable_talker()` and pass
   `return_audio=False` to `generate`. The LLM we instrument lives at
   `model.thinker.model.layers[*]`.
2. **No `mm_projector` expansion step.** Modal placeholders are already
   embedded in `input_ids` between explicit start/end token IDs
   (`audio_start_token_id=151647`, `audio_end_token_id=151648`,
   `vision_start_token_id=151652`, `vision_end_token_id=151653`). There is
   no `prepare_inputs_labels_for_multimodal` that grows the embedding
   sequence, so `ids_pos == emb_pos` and the awkward `ids_pos_to_emb_pos`
   helper in the existing bias script becomes a no-op.
3. **Chat-template message format.** Multimodal inputs are passed as a
   list-of-content-items conversation; the official helper
   `qwen_omni_utils.process_mm_info(conversation, use_audio_in_video=…)`
   returns `(audios, images, videos)` ready for
   `processor(text=…, audio=…, images=…, videos=…, use_audio_in_video=…)`.

Everything else (per-head o_proj zero-ablation, contrastive scoring,
attention-bias bar plots) is mechanically the same because Qwen2.5-Omni's
thinker is a standard transformers `GenerationMixin` model with
`DynamicCache` and per-layer `self_attn.o_proj`.

## Loader decision

Use the **composite loader** + `disable_talker()`. This is the documented
path, keeps a single loader across eval / attribution / bias, and the hook
target stays at `model.thinker.model.layers[i].self_attn.o_proj`.

## Files to create

### Python — new package `method/qwen2_5_omni/`

| File | Mirrors | Notes |
|------|---------|-------|
| `__init__.py` | — | empty marker |
| `utils.py` | — | Qwen2.5-Omni-specific helpers (see §1) |
| `eval.py` | `method/videollama2/eval.py` | same CLI + schema |
| `head_attribution.py` | `method/videollama2/head_attribution.py` | three model-specific deltas |
| `identify_halluc_head.py` | `method/videollama2/identify_halluc_head.py` | swap inference glue, swap config accessors |
| `analyze_attention_bias.py` | `method/videollama2/analyze_attention_bias.py` | drop projector hooks; use `model.thinker.forward(...)` |

### Bash wrappers — `bash_scripts/qwen2_5_omni/audioset/`

Nine scripts, one-to-one with the existing VideoLLaMA2 set:

- `eval.sh`, `eval_describe.sh`, `eval_mcq.sh`
- `identify_halluc_head.sh`, `identify_halluc_head_describe.sh`,
  `identify_halluc_head_mcq.sh`
- `analyze_attention_bias.sh`, `analyze_attention_bias_describe.sh`,
  `analyze_attention_bias_mcq.sh`

### Environment

- `qwen2_5_omni_requirements.txt` — side-by-side with the existing
  `video_llama2_requirements.txt`. Install into a separate venv (the
  transformers pins diverge).

### Runtime outputs (no plan changes needed)

`results/qwen2_5_omni/AudioSet[/, _describe/, _mcq/]/…` mirroring the
existing `results/videollama2/AudioSet*` tree.

## Design

### 1. Shared utilities — `method/qwen2_5_omni/utils.py`

Single source of truth for the Qwen2.5-Omni glue. Exposes:

```python
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

def load_omni(model_path, attn_implementation="eager", dtype=torch.bfloat16):
    """Composite loader + talker disabled. `attn_implementation='eager'` is
    needed for analyze_attention_bias (output_attentions=True). Set to
    'flash_attention_2' for eval to be faster."""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor

def build_conversation(media_path, question, modal_type):
    """modal_type ∈ {'a','v','av'}. For 'av' the media is a video and we
    pass use_audio_in_video=True; no separate 'audio' content item."""
    user_content = []
    if modal_type == "a":
        user_content.append({"type": "audio", "audio": media_path})
    elif modal_type == "v":
        user_content.append({"type": "video", "video": media_path})
    else:  # "av"
        user_content.append({"type": "video", "video": media_path})
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

def prepare_inputs(processor, conversation, modal_type, device, dtype):
    """Returns the dict accepted by model.generate() and the use_audio_in_video flag."""
    use_aiv = (modal_type == "av")
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_aiv)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_aiv,
    ).to(device).to(dtype)
    return inputs, use_aiv

def omni_infer(model, processor, conversation, modal_type, max_new_tokens=256):
    """One-shot text-only inference; returns the decoded string."""
    inputs, use_aiv = prepare_inputs(
        processor, conversation, modal_type, model.device, model.dtype
    )
    text_ids = model.generate(
        **inputs, use_audio_in_video=use_aiv, return_audio=False,
        do_sample=False, max_new_tokens=max_new_tokens,
    )
    gen_ids = text_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

def find_modality_spans(input_ids, config):
    """Return {'audio': (start_emb, end_emb), 'video': (start, end)}
    (inner range, exclusive of start/end markers). Returns (0,0) when
    a modality is absent. emb_pos == ids_pos for Qwen2.5-Omni."""
    ids = input_ids[0].tolist()
    def _span(start_id, end_id):
        try:
            s = ids.index(start_id); e = ids.index(end_id, s + 1)
            return s + 1, e
        except ValueError:
            return 0, 0
    return {
        "audio": _span(config.audio_start_token_id, config.audio_end_token_id),
        "video": _span(config.vision_start_token_id, config.vision_end_token_id),
    }

def thinker_layers(model):
    """Single accessor for the LLM decoder layers. Composite-loader version."""
    return model.thinker.model.layers
```

`find_modality_spans` replaces both `tokenizer_multimodal_token` and the
`ids_pos_to_emb_pos` arithmetic. `thinker_layers` is the one knob
`head_attribution.py` and `analyze_attention_bias.py` use to reach o_proj.

### 2. `method/qwen2_5_omni/eval.py`

Direct mirror of `method/videollama2/eval.py`. Same schema and CLI flags.
Differences:

- Replace `from videollama2 import mm_infer, model_init` and the
  `CustomDataset`/`mm_infer` glue with calls to `utils.load_omni`,
  `utils.build_conversation`, `utils.omni_infer`.
- No `processor["audio"]/processor["video"]` split — the Qwen2.5-Omni
  processor handles all modalities via `process_mm_info`. The
  `CustomDataset.__getitem__` becomes: build the conversation dict from
  `(video_path, question, modal_type)` and stash it as the dataset item.
  Inference is one-shot `omni_infer(model, processor, conv, modal_type)`.
- Keep the full task registry (`HALLUC_TASKS`, `MCQ_TASKS`,
  `DESCRIBE_TASKS`, `NLTK_CAPTIONING_TASKS`, `VALID_OUTPUTS`,
  `DISCRETE_TASKS`), `find_labels_in_text`, `get_entity_labels_for_entry`,
  the 50/50 balancing loop, and the AudioSet labels-file auto-discovery —
  copied unchanged from the VideoLLaMA2 version.
- CLI flags unchanged: `--n_per_category`, `--tasks`, `--model_path`
  (default `Qwen/Qwen2.5-Omni-7B`), `--modal_type`, `--video_folder`,
  `--output_file`, `--QA_FILE`, `--audioset_labels_file`.
- The output `sampled_entities.json` has the **exact same field set** as
  VideoLLaMA2's, so `identify_halluc_head.py` and
  `analyze_attention_bias.py` read it untouched.

### 3. `method/qwen2_5_omni/head_attribution.py`

Copy the VideoLLaMA2 version; change only the three model-specific things:

1. **o_proj enumeration.** VideoLLaMA2 scans `model.named_modules()` for
   `"o_proj"`. That would also pick up the talker's o_proj. Replace with a
   deterministic comprehension over `utils.thinker_layers(model)`:
   ```python
   o_proj_modules = [
       (i, layer.self_attn.o_proj)
       for i, layer in enumerate(utils.thinker_layers(model))
   ]
   ```
   Read `num_attention_heads` and `head_dim` from `model.thinker.config`.
   Qwen2 uses GQA — `num_attention_heads` is the **query** head count, which
   is what `head_dim = hidden_size // num_attention_heads` indexes against,
   matching the o_proj input shape.

2. **`GenerationMixin._sample` monkey-patch.** Qwen2.5-Omni's thinker
   inherits the same class, so the patch survives. `model.generate(...,
   return_audio=False, ...)` on the composite dispatches into
   `thinker.generate(...)` → `_sample`. Verify once with a debug print that
   the patched function fires; if it doesn't, fall back to loading
   `Qwen2_5OmniThinkerForConditionalGeneration` directly for this script
   only.

3. **DynamicCache restoration.** Same class as VideoLLaMA2. Only change is
   replacing `model.config.num_hidden_layers` with
   `model.thinker.config.num_hidden_layers`.

Everything else (the hook function that zeroes
`ablated[:, :, head_dim*h : head_dim*(h+1)] = 0`, the per-token probability
comparison, the influence accumulator dict shape) copies verbatim. The
exported entry point keeps its name:
`set_zero_ablation_greedy_search(tokenizer, hallucinated_entities,
non_hallucinated_entities, influence_score)`.

### 4. `method/qwen2_5_omni/identify_halluc_head.py`

Almost a verbatim copy. Changes:

- Replace `model_init / mm_infer` imports with
  `from method.qwen2_5_omni import utils`.
- Replace the inference call with
  `utils.omni_infer(model, processor, conv, modal_type)`.
- Replace `model.config.num_hidden_layers / num_attention_heads` with
  `model.thinker.config.…`.
- The token-level loop driving each ablation pass is unchanged.
- Contrastive scoring (`hal_heads_contrastive`, `non_hal_heads_contrastive`)
  is identical post-processing.
- CLI flags unchanged.

### 5. `method/qwen2_5_omni/analyze_attention_bias.py`

Direct port. The bias-analysis pass is a single forward over `prompt +
generated_caption` with `output_attentions=True`. Differences:

- Drop the two projector hooks that capture `n_video` / `n_audio`. Qwen2.5-Omni
  embeds modal tokens inline, so
  `find_modality_spans(input_ids, model.thinker.config)` returns
  `(video_start, video_end, audio_start, audio_end)` directly. Keep the
  `n_modal == 0` assertion as a sanity check.
- `ids_pos_to_emb_pos` collapses to identity. Remove the helper.
- The forward pass:
  ```python
  outputs = model.thinker(
      **inputs, output_attentions=True, return_dict=True, use_cache=False,
  )
  step0_attns = tuple(a.detach().to("cpu") for a in outputs.attentions)
  ```
  Keeps the CPU-offload fix from the OOM episode. Composite `forward` would
  also work but routes through the talker too; calling the thinker directly
  is cleaner. `attn_implementation="eager"` is set by `load_omni`.
- `compute_attn_stats` is unchanged — slices a row out of each layer's
  attention tensor and sums over the modality spans.
- The plot helpers (`plot_result`) are unchanged.
- CLI flags unchanged.

### 6. Bash wrappers — `bash_scripts/qwen2_5_omni/audioset/`

Carbon-copy of the VideoLLaMA2 set with substitutions:

- Python target: `method/videollama2/<name>.py` → `method/qwen2_5_omni/<name>.py`
- Output paths: `results/videollama2/AudioSet*/...` →
  `results/qwen2_5_omni/AudioSet*/...`
- `MODEL_PATH` default: `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` → `Qwen/Qwen2.5-Omni-7B`
- `VIDEO_FOLDER` stays at `$ROOT_DIR/data/AudioSet/audios` (same data)
- `--tasks` strings unchanged (the QA.json is shared)

### 7. `qwen2_5_omni_requirements.txt`

Side-by-side with `video_llama2_requirements.txt`. Key pins (from the
Qwen2.5-Omni README):

```
torch>=2.4
transformers==4.52.3
accelerate>=0.30
qwen-omni-utils[decord]
soundfile
librosa
av
decord
numpy==2.0.0
matplotlib
nltk
tqdm
huggingface_hub>=0.25
pyarrow                # preprocess_AudioSet
pandas
# optional, big speedup for eval (NOT used by analyze_attention_bias):
# flash-attn  --no-build-isolation
```

Notes:

- The `transformers==4.52.3` pin is the GitHub README's recommendation. The
  older preview tag `v4.51.3-Qwen2.5-Omni-preview` also works. Without one
  of these you hit `KeyError: 'qwen2_5_omni'`.
- `huggingface_hub>=0.25` is required by recent `transformers` — same
  constraint that caused the earlier `huggingface_hub==0.23.4` clash. The
  Qwen venv is independent of the VideoLLaMA2 venv, so the two coexist on
  disk.
- `numpy==2.0.0` is called out by the README; later 2.x versions also tend
  to work, but pin to be safe.
- System dep (document in the requirements file's header comment):
  `ffmpeg` must be on `$PATH`.

Install into a fresh venv:

```bash
python -m venv qwen_omni_venv
source qwen_omni_venv/bin/activate
pip install -r qwen2_5_omni_requirements.txt
```

## Out of scope (deliberately deferred)

- AVHBench / AVCaps / MSVD bash wrappers. AudioSet only for this pass; the
  Python is dataset-agnostic so adding more wrappers later is a copy-edit.
- Flash-attention. Eval can opt in via a `FLASH_ATTN=1` env var read in
  `load_omni`, but the analysis pass must stay on eager attention. Not
  required for first pass.
- Sharing a virtualenv with VideoLLaMA2. Pinned `transformers` versions
  diverge, so keep them separate.

## Verification

1. **Environment smoke** (CPU only, before any GPU run):
   ```bash
   python -c "from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; \
              from qwen_omni_utils import process_mm_info; \
              print('imports OK')"
   ```

2. **Token-span helper** (no GPU; verifies utils.find_modality_spans against
   the real processor):
   ```bash
   python - <<'PY'
   from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor
   from qwen_omni_utils import process_mm_info
   from method.qwen2_5_omni.utils import find_modality_spans, build_conversation
   proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
   conv = build_conversation("data/AudioSet/audios/<some>.wav",
                             "Does a dog bark?", "a")
   a, i, v = process_mm_info(conv, use_audio_in_video=False)
   text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
   inp  = proc(text=text, audio=a, images=i, videos=v, return_tensors="pt",
               use_audio_in_video=False)
   cfg = Qwen2_5OmniConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")
   spans = find_modality_spans(inp["input_ids"], cfg.thinker_config)
   print(spans)   # expect 'audio': (s,e) with (e-s) == number of audio tokens
   PY
   ```

3. **Per-task end-to-end on AudioSet hallucination variant**:
   ```bash
   bash bash_scripts/qwen2_5_omni/audioset/eval.sh
   bash bash_scripts/qwen2_5_omni/audioset/identify_halluc_head.sh
   bash bash_scripts/qwen2_5_omni/audioset/analyze_attention_bias.sh
   ```
   Compare the resulting `sampled_entities.json`,
   `attribution_result.json`, and bar plots side-by-side with the
   VideoLLaMA2 outputs under `results/videollama2/AudioSet/`. Field schemas
   must match exactly; counts can differ.

4. **MCQ + describe variants**: repeat step 3 with `_mcq` / `_describe`
   suffixes.

5. **Monkey-patch sanity** (one-time, during head_attribution porting): drop
   a `print("[zero_ablation_sample fired]")` at the top of the patched
   `_sample` and run `identify_halluc_head.sh` for a single sample. Confirm
   the line prints — verifies that the composite `generate()` routes
   through `GenerationMixin._sample` and not a Qwen2.5-Omni-specific
   override.

## Critical files to reference while implementing

- `method/videollama2/eval.py` — copy structure, swap inference glue.
- `method/videollama2/head_attribution.py` — copy monkey-patch; change
  o_proj enumeration + config accessors.
- `method/videollama2/identify_halluc_head.py` — copy verbatim except
  imports + inference call.
- `method/videollama2/analyze_attention_bias.py` — copy; drop projector
  hooks, replace `ids_pos_to_emb_pos`, use thinker forward.
- `bash_scripts/videollama2/audioset/*.sh` — template for the new wrappers.
- `video_llama2_requirements.txt` — pin-style reference for the new
  requirements file.
