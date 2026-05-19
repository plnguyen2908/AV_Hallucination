"""Attention-bias analysis for the Qwen2.5-Omni thinker.

Ports method/videollama2/analyze_attention_bias.py. Two structural changes:

  1. No `mm_projector` hooks. Qwen2.5-Omni places audio/video placeholders
     inline in `input_ids`, so we locate modality spans via
     `utils.find_modality_spans` and `emb_pos == ids_pos`.
  2. Single thinker forward (`model.thinker(...)`) instead of running
     `generate(max_new_tokens=1)` and pulling step-0 attentions. The full
     prompt + already-generated caption is re-tokenised by the processor.

The CPU-offload fix from the earlier OOM episode is preserved.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

from utils import build_conversation, find_modality_spans, load_omni


_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent

DESCRIBE_TASKS = {
    "AudioSet Captioning",
    "ActivityNet Captioning",
    "VGGSounder Captioning",
}
# Per-task describe suffix; must match eval.py / identify_halluc_head.py.
DESCRIBE_SUFFIX_BY_TASK = {
    "AudioSet Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match the sounds you hear. No explanations, no other words."
    ),
    "ActivityNet Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match what you see. No explanations, no other words."
    ),
    "VGGSounder Captioning": (
        "\nRespond with ONLY a comma-separated list of labels from the list "
        "above that match what you see and hear. No explanations, no other words."
    ),
}


def apply_qwen_prompt_suffix(prompt: str, task: str) -> str:
    suffix = DESCRIBE_SUFFIX_BY_TASK.get(task)
    if suffix and suffix not in prompt:
        return prompt + suffix
    return prompt


def compute_attn_stats(
    step0_attns,
    entity_emb_pos,
    total_emb_len,
    video_start,
    video_end,
    audio_start,
    audio_end,
    hal_heads,
    non_hal_heads,
):
    """Mean video/audio/text attention over hal and non-hal heads for one
    entity token position. Absent modalities return 0. Returns None if pos
    is out of range."""
    if entity_emb_pos is None or entity_emb_pos >= total_emb_len:
        return None

    # Text region = anything after the larger modality block and before the
    # current entity position. Robust to absent modalities (their spans are 0).
    text_start = max(video_end, audio_end)

    def _row_stats(heads):
        vid = aud = txt = 0.0
        for layer_idx, head_idx in heads:
            row = step0_attns[layer_idx][0, head_idx, entity_emb_pos, :].float()
            vid += row[video_start:video_end].sum().item()
            aud += row[audio_start:audio_end].sum().item()
            txt += row[text_start:entity_emb_pos].sum().item()
        n = len(heads) or 1
        return vid / n, aud / n, txt / n

    h_vid, h_aud, h_txt = _row_stats(hal_heads)
    n_vid, n_aud, n_txt = _row_stats(non_hal_heads)
    return {
        "hal_video": h_vid,
        "hal_audio": h_aud,
        "hal_text": h_txt,
        "nhal_video": n_vid,
        "nhal_audio": n_aud,
        "nhal_text": n_txt,
    }


def eval_model(args):
    model, processor = load_omni(args.model_path, attn_implementation="eager")
    tokenizer = processor.tokenizer

    modal_type = args.modal_type  # "av" | "v" | "a"
    if modal_type == "a":
        task_filter = {
            "Audio Captioning",
            "Video-driven Audio Hallucination",
            "AudioSet Captioning",
            "AudioSet Multiple-Choice",
        }
    elif modal_type == "v":
        task_filter = {"Video Captioning", "ActivityNet Captioning"}
    else:  # "av"
        task_filter = {
            "AV Captioning",
            "ActivityNet Captioning",
            "VGGSounder Captioning",
        }

    with open(args.input_file) as f:
        samples = json.load(f)

    samples = [
        s
        for s in samples
        if s.get("task") in task_filter
        and (s.get("hallucinated_entities") or s.get("non_hallucinated_entities"))
        and s.get("generated_caption")
    ]
    print(f"[{modal_type}] Samples to process: {len(samples)}")

    with open(args.attention_head_path) as f:
        head_data = json.load(f)
    hal_heads = head_data["hal_heads_contrastive"][: args.top_k]
    non_hal_heads = head_data["non_hal_heads_contrastive"][: args.top_k]
    print(f"Hal heads ({len(hal_heads)}): {hal_heads[:3]} ...")
    print(f"Non-hal heads ({len(non_hal_heads)}): {non_hal_heads[:3]} ...")

    hal_stats = []
    nonhal_stats = []

    for line in tqdm(samples):
        question_id = line["question_id"]
        video_path = os.path.join(args.video_folder, line["video"])
        task = line["task"]
        prompt = apply_qwen_prompt_suffix(line["question"], task)
        generated_caption = line["generated_caption"].strip()
        hallucinated_entities = line.get("hallucinated_entities", [])
        non_hal_entities = line.get("non_hallucinated_entities", [])

        conv = build_conversation(video_path, prompt, modal_type)
        use_aiv = modal_type == "av"

        try:
            audios, images, videos = process_mm_info(conv, use_audio_in_video=use_aiv)
        except Exception as e:
            print(f"Read error: {video_path}: {e}")
            continue

        prompt_text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        # Qwen2.5-Omni's processor returns ['…'] (batched form) rather than a
        # bare str. Unwrap so string concatenation with generated_caption works.
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0]
        # Teacher-forcing: locate where the generated caption starts in the
        # combined ids.
        prompt_only = processor(
            text=prompt_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_aiv,
        )
        inputs = processor(
            text=prompt_text + generated_caption,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_aiv,
        )
        caption_start_in_ids = prompt_only["input_ids"].shape[1]

        # Modal spans (emb_pos == ids_pos for Qwen2.5-Omni — no expansion).
        # find_modality_spans handles the audio/vision token IDs living on
        # either thinker.config or thinker.config.text_config.
        spans = find_modality_spans(inputs["input_ids"], model.thinker.config)
        video_start, video_end = spans["video"]
        audio_start, audio_end = spans["audio"]
        n_video = video_end - video_start
        n_audio = audio_end - audio_start
        n_modal = n_video + n_audio
        assert n_modal > 0, f"No modal tokens found in {question_id}"

        # Move inputs to model's device/dtype.
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.inference_mode():
            outputs = model.thinker(
                **inputs,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        # CPU-offload immediately — these tensors are huge for long prompts.
        step0_attns = tuple(a.detach().to("cpu") for a in outputs.attentions)
        del outputs
        torch.cuda.empty_cache()

        assert step0_attns[0].ndim == 4, (
            f"Expected 4D attention tensor (batch, heads, q_len, kv_len), "
            f"got shape {step0_attns[0].shape}"
        )
        total_emb_len = step0_attns[0].shape[2]

        # NaN check on first + last layer (catches dtype-overflow at long prompts).
        for check_layer in [0, len(step0_attns) - 1]:
            assert not torch.isnan(step0_attns[check_layer].float()).any(), (
                f"NaN in attention at layer {check_layer} for {question_id}. "
                "Ensure attn_implementation='eager' and bfloat16."
            )

        for pos in range(caption_start_in_ids, inputs["input_ids"].shape[1]):
            token_str = tokenizer.decode([inputs["input_ids"][0, pos].item()]).strip()
            if not token_str:
                continue
            is_hal = any(token_str in ent for ent in hallucinated_entities)
            is_nonhal = any(token_str in ent for ent in non_hal_entities)
            if not is_hal and not is_nonhal:
                continue
            stat = compute_attn_stats(
                step0_attns,
                pos,
                total_emb_len,
                video_start,
                video_end,
                audio_start,
                audio_end,
                hal_heads,
                non_hal_heads,
            )
            if stat is None:
                continue
            if is_hal:
                hal_stats.append(stat)
            if is_nonhal:
                nonhal_stats.append(stat)

    print(
        f"Collected {len(hal_stats)} hal-entity stats, "
        f"{len(nonhal_stats)} non-hal-entity stats"
    )

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(
        {"hal": hal_stats, "nonhal": nonhal_stats, "modal_type": modal_type},
        os.path.join(args.output_path, "attention_statics.pth"),
    )


# --- Plotting (identical to the VideoLLaMA2 version) ---


def plot_result(args):
    data = torch.load(
        os.path.join(args.output_path, "attention_statics.pth"),
        weights_only=False,
    )
    hal_stats = data["hal"]
    nonhal_stats = data["nonhal"]
    modal_type = data.get("modal_type", args.modal_type)

    def col_mean(stats, key):
        vals = [s[key] for s in stats if s is not None]
        return float(np.mean(vals)) if vals else 0.0

    if modal_type == "v":
        modalities = [
            ("Video", "hal_video", "nhal_video"),
            ("Text", "hal_text", "nhal_text"),
        ]
    elif modal_type == "a":
        modalities = [
            ("Audio", "hal_audio", "nhal_audio"),
            ("Text", "hal_text", "nhal_text"),
        ]
    else:  # "av"
        modalities = [
            ("Video", "hal_video", "nhal_video"),
            ("Audio", "hal_audio", "nhal_audio"),
            ("Text", "hal_text", "nhal_text"),
        ]

    n = len(modalities)
    bar_width = 0.2
    group_spacing = 1.0
    mod_colors = ["tab:blue", "tab:orange", "tab:green"][:n]

    hal_center = 0.0
    nhal_center = group_spacing
    offsets = np.linspace(-(n - 1) * bar_width / 2, (n - 1) * bar_width / 2, n)
    hal_xs = hal_center + offsets
    nhal_xs = nhal_center + offsets

    def _draw_single(ax, title, stats):
        for i, (mod_label, hal_key, nhal_key) in enumerate(modalities):
            print(f"[{title}] {mod_label}, hal_heads: {col_mean(stats, hal_key):.4f}")
            ax.bar(
                hal_xs[i],
                col_mean(stats, hal_key),
                bar_width,
                color=mod_colors[i],
                label=mod_label,
            )
            print(f"[{title}] {mod_label}, nhal_heads: {col_mean(stats, nhal_key):.4f}")
            ax.bar(
                nhal_xs[i], col_mean(stats, nhal_key), bar_width, color=mod_colors[i]
            )
        ax.set_title(
            f"[{modal_type.upper()}] {title} (top {args.top_k} heads)", fontsize=13
        )
        ax.set_ylabel("Mean Attention Weight", fontsize=12)
        ax.set_xticks([hal_center, nhal_center])
        ax.set_xticklabels(["Hal Heads", "Non-Hal Heads"], fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(title="Modality", fontsize=11)

    fig, ax = plt.subplots(figsize=(5 * max(n, 2), 6))
    _draw_single(ax, "At Hallucinated Entity Tokens", hal_stats)
    plt.tight_layout()
    out_path = os.path.join(
        args.output_path, f"attention_bias_hal_top_{args.top_k}_heads.png"
    )
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5 * max(n, 2), 6))
    _draw_single(ax, "At Non-Hallucinated Entity Tokens", nonhal_stats)
    plt.tight_layout()
    out_path = os.path.join(
        args.output_path, f"attention_bias_nonhal_top_{args.top_k}_heads.png"
    )
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5 * max(n, 2) * 2, 6))
    _draw_single(axes[0], "At Hallucinated Entity Tokens", hal_stats)
    _draw_single(axes[1], "At Non-Hallucinated Entity Tokens", nonhal_stats)
    plt.tight_layout()
    out_path = os.path.join(
        args.output_path, f"attention_bias_top_{args.top_k}_heads.png"
    )
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(_REPO / "results/qwen2_5_omni/AudioSet/sampled_entities.json"),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=str(_REPO / "data/AudioSet/audios"),
    )
    parser.add_argument(
        "--attention_head_path",
        type=str,
        default=str(
            _REPO
            / "results/qwen2_5_omni/AudioSet/attribution/heads/attribution_result.json"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(_REPO / "results/qwen2_5_omni/AudioSet/attention_bias"),
    )
    parser.add_argument(
        "--modal_type", type=str, default="a", choices=["av", "v", "a"]
    )
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    eval_model(args)
    plot_result(args)
