"""Qwen2.5-Omni-specific glue shared by the eval / attribution / bias scripts.

Loader + inference helpers wrap the composite `Qwen2_5OmniForConditionalGeneration`
with `disable_talker()` so the pipeline only sees the thinker (encoders + LLM).
Modal placeholders are located by start/end token IDs from
`model.thinker.config`.
"""

from typing import List, Tuple

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# The Qwen2.5-Omni README and cookbook examples all use this exact system
# prompt. The model's RLHF favors it; deviating tends to make outputs lazier.
OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def load_omni(
    model_path: str,
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
):
    """Composite loader, talker disabled.

    `attn_implementation='eager'` is required when this pipeline runs the
    attention-bias pass (output_attentions=True). For pure eval, callers can
    pass `'flash_attention_2'` for a ~2x speedup.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False, so device_map='auto' will "
            "place the model on CPU and the pipeline will crawl. Reinstall "
            "torch / torchvision from the CUDA index, e.g.:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch torchvision\n"
            f"(torch.__version__={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda})"
        )
    print(
        f"[load_omni] CUDA OK: {torch.cuda.device_count()} device(s), "
        f"torch {torch.__version__} / cuda {torch.version.cuda}"
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


# Default sampling caps for video inputs. Qwen2.5-Omni's stock video
# preprocessing samples at 2 fps with no pixel ceiling, so even a 20s clip
# can produce thousands of visual tokens and OOM on a single forward.
# 1 fps + ~230k pixels/frame gives ~300 spatial tokens × 20 frames ≈ 6k
# visual tokens, which fits comfortably alongside the prompt.
VIDEO_FPS_DEFAULT = 1.0
VIDEO_MAX_PIXELS_DEFAULT = 360 * 640


def build_conversation(
    media_path: str,
    question: str,
    modal_type: str,
    video_fps: float = VIDEO_FPS_DEFAULT,
    video_max_pixels: int = VIDEO_MAX_PIXELS_DEFAULT,
) -> list:
    """Build the chat-template message list for one sample.

    modal_type ∈ {"a","v","av"}. For "av" we pass the video and tell the
    processor to read its audio track via use_audio_in_video=True — no
    separate "audio" content item.

    `video_fps` and `video_max_pixels` cap the visual token count for the
    "v" / "av" cases. The qwen-omni-utils processor reads these directly
    off the video content dict.
    """
    user_content: list = []
    if modal_type == "a":
        user_content.append({"type": "audio", "audio": media_path})
    elif modal_type in ("v", "av"):
        user_content.append({
            "type": "video",
            "video": media_path,
            "fps": video_fps,
            "max_pixels": video_max_pixels,
        })
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
    `process_mm_info`, `processor`, and `generate`.
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
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_aiv,
            return_audio=False,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    gen_ids = text_ids[:, inputs["input_ids"].shape[1]:]
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
    """Truncate Qwen2.5-Omni output at the first occurrence of any common
    chat-template leakage marker.

    The composite model sometimes continues past its own turn into a fake
    user / assistant block, e.g. "Hand\\nHuman: What's the most interesting
    thing you've seen in a video?". eval.py applies this to the raw model
    output before saving `generated_caption`; identify_halluc_head.py applies
    it to the regenerated string before comparing against
    `generated_caption` — otherwise the assertion blows up on every
    chatty sample."""
    if not text:
        return text
    earliest = len(text)
    for m in _CHAT_LEAK_MARKERS:
        idx = text.find(m)
        if 0 <= idx < earliest:
            earliest = idx
    return text[:earliest].rstrip()


def thinker_layers(model):
    """Decoder layers of the LLM inside the composite. Mirrors VideoLLaMA2's
    `model.model.layers` access. Use this everywhere instead of indexing into
    the composite directly so swapping to the thinker-only loader is one line."""
    return model.thinker.model.layers


def thinker_text_config(model):
    """Config that exposes num_hidden_layers / num_attention_heads /
    hidden_size for the thinker's LLM.

    `Qwen2_5OmniThinkerConfig` is composite — text/audio/vision configs hang
    off it — so the LLM dims live one level deeper. The text model itself
    carries them, so we prefer that path (matches what `_sample` sees as
    `self.model.config` when `self` is the thinker)."""
    thinker = getattr(model, "thinker", model)
    cfg = getattr(getattr(thinker, "model", thinker), "config", thinker.config)
    if not hasattr(cfg, "num_hidden_layers"):
        cfg = getattr(thinker.config, "text_config", thinker.config)
    return cfg


def _config_with(attr: str, *candidates):
    for cfg in candidates:
        if cfg is not None and hasattr(cfg, attr):
            return cfg
    return None


def find_modality_spans(input_ids: torch.Tensor, thinker_config) -> dict:
    """Locate the inline audio / video placeholder block in `input_ids`.

    Qwen2.5-Omni embeds modal placeholders directly between start/end tokens —
    no expansion step. So embedding-position == input-id-position and the
    awkward `ids_pos_to_emb_pos` arithmetic from the VideoLLaMA2 bias script
    disappears.

    Returns {"audio": (start, end), "video": (start, end)} with the inner
    [start, end) range (exclusive of the start/end markers). Missing modality
    returns (0, 0).
    """
    ids = input_ids[0].tolist()

    # Modal token IDs may live on the composite thinker config or one level
    # deeper on `text_config`, depending on transformers version. Try both.
    text_cfg = getattr(thinker_config, "text_config", None)
    cfg = _config_with("audio_start_token_id", thinker_config, text_cfg)
    if cfg is None:
        raise RuntimeError(
            "audio_start_token_id not found on thinker config or text_config; "
            "modal-span lookup needs it."
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
    """Return (input_ids_with_caption, caption_start_position).

    Used by analyze_attention_bias: re-tokenizes prompt + already-generated
    caption so we can pull attention rows at every caption token position.
    """
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
        "transformers.Qwen2_5OmniForConditionalGeneration",
        "transformers.Qwen2_5OmniProcessor",
        "qwen_omni_utils.process_mm_info",
    ]
