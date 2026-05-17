import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from head_attribution import set_zero_ablation_greedy_search
from tqdm import tqdm

from utils import build_conversation, load_omni, prepare_inputs, thinker_text_config


_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent

DESCRIBE_TASKS = {"AudioSet Captioning"}
DESCRIBE_SUFFIX = (
    "\nRespond with ONLY a comma-separated list of labels from the list "
    "above that match the sounds you hear. No explanations, no other words."
)


def apply_qwen_prompt_suffix(prompt: str, task: str) -> str:
    """Mirror eval.py: append the describe-variant format suffix if not
    already present, so the prompt rebuilt here matches what produced the
    saved `generated_caption`."""
    if task in DESCRIBE_TASKS and DESCRIBE_SUFFIX not in prompt:
        return prompt + DESCRIBE_SUFFIX
    return prompt


def _heatmap(data, ax, title=None, hal_heads=None, non_hal_heads=None):
    v_limit = np.percentile(np.abs(data), 99.0)
    sns.heatmap(
        data,
        cmap="coolwarm",
        center=0,
        ax=ax,
        vmin=-v_limit,
        vmax=v_limit,
        cbar_kws={"shrink": 0.8},
    )
    for heads, color in zip([hal_heads, non_hal_heads], ["red", "blue"]):
        if heads:
            for layer_idx, head_idx in heads:
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
    if title:
        ax.set_title(title, pad=20)
    ax.set_ylabel("Layer Index")
    ax.set_xlabel("Head Index")


def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def _dims_from_pth(pth_dir: str):
    """Infer (layer_num, head_num) from any saved influence pth file."""
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

    model, processor = load_omni(args.model_path)
    tokenizer = processor.tokenizer

    with open(args.input_file, "r") as f:
        samples = json.load(f)

    # Composite config -> thinker text-config for LLM dims.
    text_cfg = thinker_text_config(model)
    layer_num = text_cfg.num_hidden_layers
    head_num = text_cfg.num_attention_heads

    for line in tqdm(samples, total=len(samples)):
        question_id = line["question_id"]
        video_path = os.path.join(args.video_folder, line["video"])
        task = line["task"]
        prompt = apply_qwen_prompt_suffix(line["question"], task)

        hallucinated_entities = line.get("hallucinated_entities", [])
        non_hallucinated_entities = line.get("non_hallucinated_entities", [])

        # (Re-)patch GenerationMixin._sample with this sample's entity lists.
        set_zero_ablation_greedy_search(
            tokenizer,
            hallucinated_entities,
            non_hallucinated_entities,
            args.influence_score,
        )

        conv = build_conversation(video_path, prompt, args.modal_type)
        try:
            inputs, use_aiv = prepare_inputs(
                processor, conv, args.modal_type, model.device, model.dtype
            )
        except Exception as e:
            print(f"video read / preprocess error: {video_path}: {e}")
            continue

        max_new_tokens = 5 if "Captioning" not in task else 2048

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                use_audio_in_video=use_aiv,
                return_audio=False,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
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

            # Strip prompt then decode.
            prompt_len = inputs["input_ids"].shape[1]
            if hasattr(generate_outputs, "sequences"):
                full_seq = generate_outputs.sequences
            else:
                full_seq = generate_outputs
            generated_text = processor.batch_decode(
                full_seq[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
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
        if not file.endswith("pth"):
            continue
        if file.startswith("hal"):
            sample = torch.load(os.path.join(args.output_path, "pth", file))
            if not sample:
                continue
            influences = []
            for _, v in sample.items():
                influence = torch.zeros(layer_num, head_num)
                for layer_idx in range(layer_num):
                    for head_idx in range(head_num):
                        influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                            "influence"
                        ]
                influences.append(torch.nan_to_num(influence, nan=0.0))
            hallucination_samples += influences
        else:
            sample = torch.load(os.path.join(args.output_path, "pth", file))
            if not sample:
                continue
            influences = []
            for _, v in sample.items():
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

    results = {
        "hal_heads_contrastive": hal_heads_contrastive,
        "non_hal_heads_contrastive": non_hal_heads_contrastive,
    }

    _, flat_indices = torch.topk(mean_hal.flatten(), args.topk, largest=True)
    results["hal_heads_mean"] = [
        [int(fi.numpy() // head_num), int(fi.numpy() % head_num)] for fi in flat_indices
    ]
    _, flat_indices = torch.topk(mean_hal.flatten(), args.topk, largest=False)
    results["non_hal_heads_mean"] = [
        [int(fi.numpy() // head_num), int(fi.numpy() % head_num)] for fi in flat_indices
    ]

    print(results)
    with open(f"{args.output_path}/heads/attribution_result.json", "w") as f:
        json.dump(results, f, default=json_custom_serializer)
    print(f"Saved: {args.output_path}/heads/attribution_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument(
        "--modal_type", type=str, default="a", choices=["a", "v", "av"]
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(_REPO / "results/qwen2_5_omni/AudioSet/attribution"),
    )
    parser.add_argument("--influence_score", type=str, default="prob_diff")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    layer_num, head_num = main(args)
    contrastive_score(args, layer_num, head_num)
