import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from videollama2 import model_init
from videollama2.constants import (
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    VIDEO_TOKEN_INDEX,
)
from videollama2.mm_utils import tokenizer_multimodal_token
from videollama2.utils import disable_torch_init

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ids_pos_to_emb_pos(pos_in_ids, modal_pos, n_modal):
    """Map a position in input_ids to the corresponding position in the
    embedded sequence after the single modal placeholder is expanded."""
    if pos_in_ids <= modal_pos:
        return pos_in_ids
    return pos_in_ids + n_modal - 1


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
    """Compute mean video/audio/text attention over hal and non-hal heads
    for a single entity token position.

    Absent modalities (video_start == video_end, etc.) yield 0.0.
    Returns a dict or None if position is out of range.
    """
    if entity_emb_pos is None or entity_emb_pos >= total_emb_len:
        return None

    def _row_stats(heads):
        vid = aud = txt = 0.0
        for layer_idx, head_idx in heads:
            row = step0_attns[layer_idx][0, head_idx, entity_emb_pos, :].float()
            vid += row[video_start:video_end].sum().item()
            aud += row[audio_start:audio_end].sum().item()
            txt += row[audio_end:entity_emb_pos].sum().item()
        n = len(heads)
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


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def eval_model(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)
    model = model.to(torch.bfloat16)

    modal_type = args.modal_type  # "av", "v", or "a"

    # --- modal-dependent setup ---
    if modal_type == "a":
        preprocess = processor["audio"]
        modal_token = DEFAULT_AUDIO_TOKEN
        modal_token_idx = AUDIO_TOKEN_INDEX
        task_filter = {"Audio Captioning", "Video-driven Audio Hallucination"}
    elif modal_type == "v":
        preprocess = processor["video"]
        modal_token = DEFAULT_VIDEO_TOKEN
        modal_token_idx = VIDEO_TOKEN_INDEX
        task_filter = {"Video Captioning"}
    else:  # "av"
        preprocess = processor["video"]
        modal_token = DEFAULT_VIDEO_TOKEN
        modal_token_idx = VIDEO_TOKEN_INDEX
        task_filter = {"AV Captioning"}

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

    # --- hooks (only register what the modality needs) ---
    video_n: dict = {}
    audio_n: dict = {}

    hooks = []
    if modal_type in ("av", "v"):
        hooks.append(
            model.model.mm_projector.register_forward_hook(
                lambda m, i, o: video_n.__setitem__(
                    "n", o.shape[1] if o.dim() == 3 else o.shape[0]
                )
            )
        )
    if modal_type in ("av", "a"):
        hooks.append(
            model.model.mm_projector_a.register_forward_hook(
                lambda m, i, o: audio_n.__setitem__(
                    "n", o.shape[1] if o.dim() == 3 else o.shape[0]
                )
            )
        )

    hal_stats = []  # stats at hallucinated entity positions
    nonhal_stats = []  # stats at non-hallucinated entity positions

    for line in tqdm(samples):
        question_id = line["question_id"]
        video_path = os.path.join(args.video_folder, line["video"])
        qs = line["question"]
        generated_caption = line["generated_caption"].strip()
        hallucinated_entities = line.get("hallucinated_entities", [])
        non_hal_entities = line.get("non_hallucinated_entities", [])

        try:
            if modal_type == "a":
                av_tensor = preprocess(video_path)
            else:
                av_tensor = preprocess(video_path, va=(modal_type == "av"))
        except Exception as e:
            print(f"Read error: {video_path}: {e}")
            continue

        if isinstance(av_tensor, dict):
            tensor_cuda = {k: v.to(torch.bfloat16).cuda() for k, v in av_tensor.items()}
        else:
            tensor_cuda = av_tensor.to(torch.bfloat16).cuda()
        images = [(tensor_cuda, "audio" if modal_type == "a" else "video")]

        message = [{"role": "user", "content": modal_token + "\n" + qs}]
        if model.config.model_type in [
            "videollama2",
            "videollama2_mistral",
            "videollama2_mixtral",
        ]:
            system_message = [
                {
                    "role": "system",
                    "content": (
                        "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
                        "Always answer as helpfully as possible, while being safe.  "
                        "Your answers should not include any harmful, unethical, racist, "
                        "sexist, toxic, dangerous, or illegal content. Please ensure that "
                        "your responses are socially unbiased and positive in nature.\n"
                        "If a question does not make any sense, or is not factually coherent, "
                        "explain why instead of answering something not correct. If you don't "
                        "know the answer to a question, please don't share false information."
                        "\n<</SYS>>"
                    ),
                }
            ]
        else:
            system_message = []
        message = system_message + message

        # Teacher-forcing: tokenize with and without caption to find caption start
        prompt_only = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        prompt_with_caption = prompt_only + generated_caption

        input_ids_prompt = (
            tokenizer_multimodal_token(
                prompt_only, tokenizer, modal_token, return_tensors="pt"
            )
            .unsqueeze(0)
            .long()
            .cuda()
        )
        input_ids = (
            tokenizer_multimodal_token(
                prompt_with_caption, tokenizer, modal_token, return_tensors="pt"
            )
            .unsqueeze(0)
            .long()
            .cuda()
        )

        caption_start_in_ids = input_ids_prompt.shape[1]
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        modal_positions = (input_ids[0] == modal_token_idx).nonzero(as_tuple=False)
        if modal_positions.numel() == 0:
            print(f"No modal token in {question_id}, skipping")
            continue
        modal_pos = modal_positions[0][0].item()

        video_n.clear()
        audio_n.clear()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=images,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_video = video_n.get("n", 0)
        n_audio = audio_n.get("n", 0)
        n_modal = n_video + n_audio

        assert n_modal > 0

        # outputs['attentions'][0]: tuple of num_layers tensors (prefill step)
        # Each tensor: (batch, heads, q_len, kv_len)
        step0_attns = outputs["attentions"][0]
        assert isinstance(step0_attns, tuple) and len(step0_attns) > 0, (
            f"Expected tuple of layer attentions, got {type(step0_attns)}"
        )
        assert step0_attns[0].ndim == 4, (
            f"Expected 4D attention tensor (batch, heads, q_len, kv_len), "
            f"got shape {step0_attns[0].shape}"
        )
        total_emb_len = step0_attns[0].shape[2]

        assert total_emb_len == input_ids.shape[1] + n_modal - 1, (
            f"{question_id}: emb_len={total_emb_len}, "
            f"expected={input_ids.shape[1] + n_modal - 1} "
            f"(n_video={n_video}, n_audio={n_audio})"
        )

        # Assert no NaN in attention (first and last layer)
        for check_layer in [0, len(step0_attns) - 1]:
            assert not torch.isnan(step0_attns[check_layer].float()).any(), (
                f"NaN in attention at layer {check_layer} for {question_id}. "
                "Ensure the model is in bfloat16."
            )

        video_start = modal_pos
        video_end = modal_pos + n_video
        audio_start = video_end
        audio_end = video_end + n_audio

        for pos in range(caption_start_in_ids, input_ids.shape[1]):
            token_str = tokenizer.decode([input_ids[0, pos].item()]).strip()
            if not token_str:
                continue
            is_hal = any(token_str in ent for ent in hallucinated_entities)
            is_nonhal = any(token_str in ent for ent in non_hal_entities)
            if not is_hal and not is_nonhal:
                continue
            emb_pos = ids_pos_to_emb_pos(pos, modal_pos, n_modal)
            stat = compute_attn_stats(
                step0_attns,
                emb_pos,
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

        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    print(
        f"Collected {len(hal_stats)} hal-entity stats, "
        f"{len(nonhal_stats)} non-hal-entity stats"
    )

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(
        {"hal": hal_stats, "nonhal": nonhal_stats, "modal_type": modal_type},
        os.path.join(args.output_path, "attention_statics.pth"),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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

    # Modality bars depend on what was recorded
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

    # Graph 1: hallucinated entity tokens only
    fig, ax = plt.subplots(figsize=(5 * max(n, 2), 6))
    _draw_single(ax, "At Hallucinated Entity Tokens", hal_stats)
    plt.tight_layout()
    out_path = os.path.join(
        args.output_path, f"attention_bias_hal_top_{args.top_k}_heads.png"
    )
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

    # Graph 2: non-hallucinated entity tokens only
    fig, ax = plt.subplots(figsize=(5 * max(n, 2), 6))
    _draw_single(ax, "At Non-Hallucinated Entity Tokens", nonhal_stats)
    plt.tight_layout()
    out_path = os.path.join(
        args.output_path, f"attention_bias_nonhal_top_{args.top_k}_heads.png"
    )
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

    # Graph 3: both side by side (original layout)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/nobackup3/le/AV_Hallucination/results/videollama2/AVCaps/sampled_entities.json",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="/nobackup3/le/AV_Hallucination/data/AVCaps/videos",
    )
    parser.add_argument(
        "--attention_head_path",
        type=str,
        default="/nobackup3/le/AV_Hallucination/results/videollama2/AVCaps/attribution/heads/attribution_result.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nobackup3/le/AV_Hallucination/results/videollama2/AVCaps/attention_bias",
    )
    parser.add_argument(
        "--modal_type", type=str, default="av", choices=["av", "v", "a"]
    )
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    eval_model(args)
    plot_result(args)
