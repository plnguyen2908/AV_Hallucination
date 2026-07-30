"""Qwen3-Omni-30B-A3B-specific glue shared by the eval / attribution scripts.

Direct port of ``method/qwen2_5_omni/utils.py``. Deltas vs. Qwen2.5-Omni:

* Model / processor classes are ``Qwen3OmniMoe*``.
* The thinker LLM is a 48-layer, 32-head MoE (128 experts / 8 active). Its
  attention uses an EXPLICIT ``head_dim = 128`` that is NOT ``hidden_size /
  num_attention_heads`` (2048 / 32 = 64). So the o_proj input is 4096-wide
  (32 heads x 128) and any per-head slicing MUST use the config ``head_dim``,
  not the naive quotient. ``thinker_head_dim`` exposes this.
* The default system prompt is not hard-baked into the chat template anymore,
  so we pass a neutral one (talker is disabled -> text-only thinker path).

Everything else (build_conversation / prepare_inputs / omni_infer /
find_modality_spans) is structurally identical; the modal placeholder tokens
(<|audio_start|>/<|vision_start|>/...) and their config ids match Qwen2.5.
"""

from typing import List, Tuple

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# Qwen3-Omni no longer bakes a default persona into its chat template. We only
# ever exercise the text-output thinker path (talker disabled), so a plain
# assistant system prompt is enough; kept as a module constant so the eval /
# attribution scripts share one string and reproduce each other's prompts.
OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def load_omni(
    model_path: str,
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "balanced_low_0",
    max_memory=None,
):
    """Composite loader, talker disabled.

    ``attn_implementation='eager'`` is required for the per-head ablation
    (the o_proj forward-pre-hook needs a real attention forward). For pure
    eval, callers can pass ``'flash_attention_2'``/``'sdpa'`` for speed.

    ``device_map='balanced_low_0'`` shards the 30B MoE across all visible GPUs
    while keeping GPU 0 light for generation overhead — the only placement that
    works reliably for this family (plain ``'auto'`` packs onto GPU 0 and OOMs
    on the long-sequence / ablation forwards).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False; the 30B model would land on "
            f"CPU. torch={torch.__version__}, cuda={torch.version.cuda}"
        )
    print(
        f"[load_omni] CUDA OK: {torch.cuda.device_count()} device(s), "
        f"torch {torch.__version__} / cuda {torch.version.cuda}; "
        f"device_map={device_map!r}, max_memory={max_memory}"
    )
    from_kwargs = dict(
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    if max_memory is not None:
        from_kwargs["max_memory"] = max_memory
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, **from_kwargs
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor


# Default sampling caps for video inputs — same rationale as Qwen2.5: cap the
# visual token count so a single forward fits alongside the prompt.
VIDEO_FPS_DEFAULT = 1.0
VIDEO_MAX_PIXELS_DEFAULT = 360 * 640


def build_conversation(
    media_path: str,
    question: str,
    modal_type: str,
    video_fps: float = VIDEO_FPS_DEFAULT,
    video_max_pixels: int = VIDEO_MAX_PIXELS_DEFAULT,
    resized_height: int = None,
    resized_width: int = None,
) -> list:
    """Build the chat-template message list for one sample.

    modal_type in {"a","v","av"}. For "av" we pass the video and read its audio
    track via use_audio_in_video=True (no separate audio item).
    """
    user_content: list = []
    if modal_type == "a":
        user_content.append({"type": "audio", "audio": media_path})
    elif modal_type in ("v", "av"):
        video_dict = {
            "type": "video",
            "video": media_path,
            "fps": video_fps,
        }
        if resized_height is not None and resized_width is not None:
            video_dict["resized_height"] = resized_height
            video_dict["resized_width"] = resized_width
        else:
            video_dict["max_pixels"] = video_max_pixels
        user_content.append(video_dict)
    else:
        raise ValueError(f"Unknown modal_type: {modal_type!r}")
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def prepare_inputs(processor, conversation, modal_type, device, dtype):
    """Build the kwargs dict accepted by model.generate / model.thinker.

    Returns (inputs, use_audio_in_video). The flag must match between
    process_mm_info, processor, and generate.
    """
    use_aiv = modal_type == "av"
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_aiv)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_aiv,
    )
    inputs = inputs.to(device).to(dtype)
    return inputs, use_aiv


def _extract_sequences(out):
    """Normalize model.generate(...) output to the token-id tensor.

    Qwen3-Omni's composite generate returns a tuple even with
    return_audio=False (the thinker text ids come first, audio/None second),
    unlike Qwen2.5 which returned a bare tensor. Also handle GenerateOutput
    (has .sequences) and the bare-tensor case for robustness.
    """
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "sequences"):
        out = out.sequences
    return out


def omni_infer(
    model,
    processor,
    conversation,
    modal_type: str,
    max_new_tokens: int = 256,
) -> str:
    """One-shot text-only greedy inference; returns the decoded string."""
    inputs, use_aiv = prepare_inputs(
        processor, conversation, modal_type, model.device, model.dtype
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            use_audio_in_video=use_aiv,
            return_audio=False,
            do_sample=False,
            # The composite generate ignores a plain `max_new_tokens` (it
            # pre-seeds thinker_kwargs with thinker_max_new_tokens, default
            # 1024, and only backfills missing keys). Control decode length
            # via thinker_max_new_tokens.
            thinker_max_new_tokens=max_new_tokens,
        )
    seq = _extract_sequences(out)
    gen_ids = seq[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


_CHAT_LEAK_MARKERS = (
    "\nHuman:",
    "\nuser:",
    "\nUser:",
    "\nAssistant:",
    "\nassistant:",
    "Human:",
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
)


def trim_chat_artifacts(text: str) -> str:
    """Truncate the model output at the first chat-template leakage marker.
    (Same helper as Qwen2.5; the composite occasionally continues past its own
    turn into a fake user/assistant block.)"""
    if not text:
        return text
    earliest = len(text)
    for m in _CHAT_LEAK_MARKERS:
        idx = text.find(m)
        if 0 <= idx < earliest:
            earliest = idx
    return text[:earliest].rstrip()


def thinker_layers(model):
    """Decoder layers of the thinker's LLM inside the composite."""
    return model.thinker.model.layers


def thinker_text_config(model):
    """Config exposing num_hidden_layers / num_attention_heads / hidden_size /
    head_dim for the thinker's LLM.

    Qwen3OmniMoeThinkerConfig is composite (text/audio/vision hang off it), so
    the LLM dims live on ``text_config`` (or on the inner text model's own
    config). Prefer the path that actually carries num_hidden_layers.
    """
    thinker = getattr(model, "thinker", model)
    # The text model's own config is what `_sample` sees as self.model.config.
    cfg = getattr(getattr(thinker, "model", thinker), "config", thinker.config)
    if not hasattr(cfg, "num_hidden_layers"):
        cfg = getattr(thinker.config, "text_config", thinker.config)
    return cfg


def thinker_head_dim(text_config):
    """Per-head width of the thinker attention.

    Qwen3 sets an EXPLICIT head_dim (128) that is not hidden_size //
    num_attention_heads (2048 // 32 = 64). The o_proj input concatenates
    num_attention_heads x head_dim = 4096, so per-head ablation must slice with
    this value. Falls back to the quotient only if head_dim is absent.
    """
    hd = getattr(text_config, "head_dim", None)
    if hd is not None:
        return int(hd)
    return text_config.hidden_size // text_config.num_attention_heads


def _config_with(attr: str, *candidates):
    for cfg in candidates:
        if cfg is not None and hasattr(cfg, attr):
            return cfg
    return None


def find_modality_spans(input_ids: torch.Tensor, thinker_config) -> dict:
    """Locate the inline audio / video placeholder block in input_ids.

    Returns {"audio": (start, end), "video": (start, end)} with the inner
    [start, end) range (exclusive of the start/end markers). Missing modality
    returns (0, 0).
    """
    ids = input_ids[0].tolist()

    text_cfg = getattr(thinker_config, "text_config", None)
    cfg = _config_with("audio_start_token_id", thinker_config, text_cfg)
    if cfg is None:
        raise RuntimeError(
            "audio_start_token_id not found on thinker config or text_config."
        )

    def _span(start_id, end_id):
        try:
            s = ids.index(start_id)
            e = ids.index(end_id, s + 1)
            return s + 1, e
        except ValueError:
            return 0, 0

    return {
        "audio": _span(cfg.audio_start_token_id, cfg.audio_end_token_id),
        "video": _span(cfg.vision_start_token_id, cfg.vision_end_token_id),
    }


def find_caption_start_in_ids(
    processor, conversation, modal_type, generated_caption: str
) -> Tuple[torch.Tensor, int]:
    """Return (input_ids_with_caption, caption_start_position)."""
    use_aiv = modal_type == "av"
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_aiv)
    prompt_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    prompt_only = processor(
        text=prompt_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_aiv,
    )
    prompt_with_caption = processor(
        text=prompt_text + generated_caption,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_aiv,
    )
    caption_start = prompt_only["input_ids"].shape[1]
    return prompt_with_caption, caption_start


def list_omni_imports() -> List[str]:
    """Smoke-test helper — verify the env has everything wired up."""
    return [
        "transformers.Qwen3OmniMoeForConditionalGeneration",
        "transformers.Qwen3OmniMoeProcessor",
        "qwen_omni_utils.process_mm_info",
    ]
