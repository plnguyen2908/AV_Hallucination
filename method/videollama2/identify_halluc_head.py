import argparse
import json
import os
import shutil

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from head_attribution import *
from tqdm import tqdm
from videollama2 import model_init
from videollama2.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)
from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token
from videollama2.utils import disable_torch_init


def _heatmap(data, ax, title=None, hal_heads=None, non_hal_heads=None):
    # 1. Automatic Contrast
    # Using a symmetric limit based on the actual data spread
    v_limit = np.percentile(np.abs(data), 99.0)

    # 2. Standard Heatmap Call
    # We remove the custom colorbar logic to keep it "normal"
    sns.heatmap(
        data,
        cmap="coolwarm",
        center=0,
        ax=ax,
        vmin=-v_limit,
        vmax=v_limit,
        cbar_kws={"shrink": 0.8},  # Keeps colorbar from being taller than the plot
    )

    # 3. Drawing Outlines
    # Zip allows us to loop through both sets of heads and colors cleanly
    for heads, color in zip([hal_heads, non_hal_heads], ["red", "blue"]):
        if heads:
            for layer_idx, head_idx in heads:
                # IMPORTANT: Matplotlib patches use (x, y) coordinates.
                # If Layer is Y (rows) and Head is X (cols): use (head_idx, layer_idx)
                # If Layer is X (cols) and Head is Y (rows): use (layer_idx, head_idx)

                # Based on your previous screenshots (Layer=Y, Head=X):
                ax.add_patch(
                    mpatches.Rectangle(
                        (head_idx, layer_idx),
                        1,
                        1,
                        fill=False,
                        edgecolor=color,
                        linewidth=2,
                        zorder=3,
                    )
                )

    # 4. Formatting
    if title:
        ax.set_title(title, pad=20)

    # Adjust these to match your specific data orientation
    ax.set_ylabel("Layer Index")
    ax.set_xlabel("Head Index")


def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Type %s not serializable" % type(obj))


def _dims_from_pth(pth_dir: str):
    """Infer (layer_num, head_num) from a saved influence pth file."""
    for fname in os.listdir(pth_dir):
        if not fname.endswith(".pth"):
            continue
        data = torch.load(os.path.join(pth_dir, fname), weights_only=False)
        if not data:
            continue
        v = next(iter(data.values()))  # list[layer][head] of dicts
        return len(v), len(v[0])
    raise RuntimeError(f"No usable pth files found in {pth_dir}")


def main(args):

    if args.skip:
        return _dims_from_pth(os.path.join(args.output_path, "pth"))

    shutil.rmtree(args.output_path, ignore_errors=True)
    os.makedirs(f"{args.output_path}/images")
    os.makedirs(f"{args.output_path}/pth")
    os.makedirs(f"{args.output_path}/heads")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, processor, tokenizer = model_init(model_path)

    with open(args.input_file, "r") as f:
        samples = json.load(f)

    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads

    for i, line in tqdm(enumerate(samples), total=len(samples)):
        question_id = line["question_id"]
        video_path = os.path.join(args.video_folder, line["video"])
        task = line["task"]
        prompt = line["question"]

        if "Captioning" in task:
            qs = prompt
        else:
            qs = f"{prompt}"

        modal = args.modal_type  # "av", "v", or "a"

        # --- modal-dependent preprocessing ---
        if modal == "a":
            preprocess = processor["audio"]
            modal_token = DEFAULT_AUDIO_TOKEN
            modal_str = "audio"
        elif modal == "v":
            preprocess = processor["video"]
            modal_token = DEFAULT_VIDEO_TOKEN
            modal_str = "video"
        else:  # "av"
            preprocess = processor["video"]
            modal_token = DEFAULT_VIDEO_TOKEN
            modal_str = "video"

        try:
            if modal != "a":
                audio_video_tensor = preprocess(video_path, va=(modal == "av"))
            else:
                audio_video_tensor = preprocess(video_path)
            assert audio_video_tensor is not None
        except Exception:
            print(f"video read error: {video_path}")
            continue

        hallucinated_entities = line.get("hallucinated_entities", [])
        non_hallucinated_entities = line.get("non_hallucinated_entities", [])

        set_zero_ablation_greedy_search(
            tokenizer,
            hallucinated_entities,
            non_hallucinated_entities,
            args.influence_score,
        )

        if isinstance(audio_video_tensor, dict):
            tensor = {
                k: v.to(torch.float16).cuda() for k, v in audio_video_tensor.items()
            }
        else:
            tensor = audio_video_tensor.to(torch.float16).cuda()
        tensor = [(tensor, modal_str)]

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
                        "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
                        "\n"
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
                    ),
                }
            ]
        else:
            system_message = []
        message = system_message + message
        prompt_text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        input_ids = (
            tokenizer_multimodal_token(
                prompt_text, tokenizer, modal_token, return_tensors="pt"
            )
            .unsqueeze(0)
            .long()
            .cuda()
        )
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_new_tokens = 5 if "Captioning" not in task else 2048

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
                output_attentions=False,
            )

            if isinstance(outputs, tuple) and len(outputs) == 3:
                (
                    generate_outputs,
                    hallucination_influences,
                    non_hallucination_influences,
                ) = outputs
            else:
                print(f"Failed to get influences for {question_id}, moving on...")
                continue

            generated_text = tokenizer.batch_decode(
                generate_outputs, skip_special_tokens=True
            )[0].strip()

            expected_text = line.get("generated_caption", "").strip()
            assert generated_text == expected_text, (
                f"Output mismatch for {question_id}:\n"
                f"  expected: {expected_text!r}\n"
                f"  got:      {generated_text!r}"
            )

        torch.save(
            hallucination_influences,
            os.path.join(
                args.output_path, "pth", f"hallucination_influences_{question_id}.pth"
            ),
        )
        torch.save(
            non_hallucination_influences,
            os.path.join(
                args.output_path,
                "pth",
                f"non_hallucination_influences_{question_id}.pth",
            ),
        )
        torch.cuda.empty_cache()

        if hallucination_influences:
            influences = []
            for _, v in hallucination_influences.items():
                influence = torch.zeros(layer_num, head_num)
                for layer_idx in range(layer_num):
                    for head_idx in range(head_num):
                        influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                            "influence"
                        ]
                influences.append(torch.nan_to_num(influence, nan=0.0))
            fig, ax = plt.subplots(figsize=(8, 8))
            _heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(), ax)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            fig.savefig(
                f"{args.output_path}/images/hallucination_influences_{question_id}.png"
            )
            plt.close(fig)

        if non_hallucination_influences:
            influences = []
            for _, v in non_hallucination_influences.items():
                influence = torch.zeros(layer_num, head_num)
                for layer_idx in range(layer_num):
                    for head_idx in range(head_num):
                        influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                            "influence"
                        ]
                influences.append(torch.nan_to_num(influence, nan=0.0))
            fig, ax = plt.subplots(figsize=(8, 8))
            _heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(), ax)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            fig.savefig(
                f"{args.output_path}/images/non_hallucination_influences_{question_id}.png"
            )
            plt.close(fig)

    return layer_num, head_num


def contrastive_score(args, layer_num, head_num):
    files = os.listdir(os.path.join(args.output_path, "pth"))
    hallucination_samples = []
    non_hallucination_samples = []
    for file in files:
        if file.endswith("pth"):
            if file.startswith("hal"):
                hallucination_sample = torch.load(
                    os.path.join(args.output_path, "pth", file)
                )
                if not hallucination_sample:
                    continue
                influences = []
                for _, v in hallucination_sample.items():
                    influence = torch.zeros(layer_num, head_num)
                    for layer_idx in range(layer_num):
                        for head_idx in range(head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                                "influence"
                            ]
                    influences.append(torch.nan_to_num(influence, nan=0.0))
                hallucination_samples += influences
            else:
                non_hallucination_sample = torch.load(
                    os.path.join(args.output_path, "pth", file)
                )
                if not non_hallucination_sample:
                    continue
                influences = []
                for _, v in non_hallucination_sample.items():
                    influence = torch.zeros(layer_num, head_num)
                    for layer_idx in range(layer_num):
                        for head_idx in range(head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                                "influence"
                            ]
                    influences.append(torch.nan_to_num(influence, nan=0.0))
                non_hallucination_samples += influences

    hallucinated_scores = torch.stack(hallucination_samples)
    non_hallucinated_scores = torch.stack(non_hallucination_samples)

    mean_hal = torch.mean(hallucinated_scores, 0).float()
    mean_non_hal = torch.mean(non_hallucinated_scores, 0).float()
    difference = mean_hal - mean_non_hal

    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(mean_hal.cpu().numpy(), ax, "Mean Hallucination Influence")
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    fig.savefig(f"{args.output_path}/images/mean_hallucination_influences.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(mean_non_hal.cpu().numpy(), ax, "Mean Non-Hallucination Influence")
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    fig.savefig(f"{args.output_path}/images/mean_non_hallucination_influences.png")
    plt.close(fig)

    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
    hal_heads_contrastive = [
        [int(fi.numpy() // head_num), int(fi.numpy() % head_num)] for fi in flat_indices
    ]

    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
    non_hal_heads_contrastive = [
        [int(fi.numpy() // head_num), int(fi.numpy() % head_num)] for fi in flat_indices
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(
        difference.cpu().numpy(),
        ax,
        "Contrastive Influence (Hal - Non-Hal)",
        hal_heads=hal_heads_contrastive,
        non_hal_heads=non_hal_heads_contrastive,
    )
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    fig.savefig(f"{args.output_path}/images/contrastive_influences.png")
    plt.close(fig)

    results = {}
    results["hal_heads_contrastive"] = hal_heads_contrastive
    results["non_hal_heads_contrastive"] = non_hal_heads_contrastive

    _, flat_indices = torch.topk(mean_hal.flatten(), args.topk, largest=True)
    indices = [[fi.numpy() // head_num, fi.numpy() % head_num] for fi in flat_indices]
    results["hal_heads_mean"] = indices

    _, flat_indices = torch.topk(mean_hal.flatten(), args.topk, largest=False)
    indices = [[fi.numpy() // head_num, fi.numpy() % head_num] for fi in flat_indices]
    results["non_hal_heads_mean"] = indices

    print(results)
    with open(f"{args.output_path}/heads/attribution_result.json", "w") as file:
        json.dump(results, file, default=json_custom_serializer)
    print(f"Saved: {args.output_path}/heads/attribution_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument(
        "--modal_type", type=str, default="av", choices=["av", "v", "a"]
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nobackup3/le/AV_Hallucination/results/videollama2/AVCaps/attribution",
    )
    parser.add_argument("--influence_score", type=str, default="prob_diff")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    layer_num, head_num = main(args)
    contrastive_score(args, layer_num, head_num)
