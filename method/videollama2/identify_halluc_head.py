import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Import the custom monkey patch for head attribution
from head_attribution import set_zero_ablation_greedy_search
from tqdm import tqdm
from videollama2.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)
from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token
from videollama2.utils import disable_torch_init

from videollama2 import model_init


def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Type %s not serializable" % type(obj))


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    model, processor, tokenizer = model_init(model_path, attn_implementation="eager")
    set_zero_ablation_greedy_search(tokenizer)

    with open(args.input_file, "r") as f:
        samples = json.load(f)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(f"{args.output_path}/images", exist_ok=True)
    os.makedirs(f"{args.output_path}/pth", exist_ok=True)
    os.makedirs(f"{args.output_path}/heads", exist_ok=True)
    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads

    for i, line in tqdm(enumerate(samples), total=len(samples)):
        question_id = line["question_id"]
        video_path = os.path.join(args.video_folder, line["video"])
        task = line["task"]
        prompt = line["question"]

        if task == "AV Captioning":
            qs = f"{prompt}. Please describe the video in one full sentence."
        else:
            qs = f"{prompt}. Start you answer with Yes/No and please provide a detailed explanation after that."

        modal = args.modal_type

        preprocess = processor["audio" if modal == "a" else "video"]
        try:
            audio_video_tensor = preprocess(
                video_path, va=True if modal == "av" else False
            )
        except Exception:
            print(f"video read error: {video_path}")
            continue

        hallucinated_tokens = line.get("hallucinated_tokens", [])
        non_hallucinated_tokens = line.get("non_hallucinated_tokens", [])

        if not hallucinated_tokens and not non_hallucinated_tokens:
            print(f"Skipping {question_id} format")
            continue

        modal = "audio" if modal == "a" else "video"
        # Preprocessing from mm_infer
        if modal == "image":
            modal_token = DEFAULT_IMAGE_TOKEN
        elif modal == "video":
            modal_token = DEFAULT_VIDEO_TOKEN
        elif modal == "audio":
            modal_token = DEFAULT_AUDIO_TOKEN
        else:
            modal_token = ""

        if isinstance(audio_video_tensor, dict):
            tensor = {k: v.half().cuda() for k, v in audio_video_tensor.items()}
        else:
            tensor = audio_video_tensor.half().cuda()
        tensor = [(tensor, modal)]

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
                        "<<SYS>>\nYou are a helpful, respectful and honest assistant. ... <</SYS>>"
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

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=tensor,
                do_sample=False,
                max_new_tokens=2048,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
                output_attentions=False,
                return_dict_in_generate=True,
                hallucinated_tokens=hallucinated_tokens,
                non_hallucinated_tokens=non_hallucinated_tokens,
                influence_score=args.influence_score,
            )

            # The custom greedy_search returns the tuple, so outputs should be a tuple length 3
            if isinstance(outputs, tuple) and len(outputs) == 3:
                (
                    generate_outputs,
                    hallucination_influences,
                    non_hallucination_influences,
                ) = outputs
            else:
                print(f"Failed to get influences for {question_id}, moving on...")
                continue

        # Save influence data per question
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
                influences.append(influence)
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(
                torch.mean(torch.stack(influences), 0).cpu().numpy(),
                cmap="coolwarm",
                center=0,
                ax=ax,
            )
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
                influences.append(influence)
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(
                torch.mean(torch.stack(influences), 0).cpu().numpy(),
                cmap="coolwarm",
                center=0,
                ax=ax,
            )
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            fig.savefig(
                f"{args.output_path}/images/non_hallucination_influences_{question_id}.png"
            )
            plt.close(fig)


def contrastive_score(args):

    files = os.listdir(os.path.join(args.output_path, "pth"))
    hallucination_samples = []
    non_hallucination_samples = []
    for file in files:
        if file.endswith("pth"):
            if file.startswith("hal"):
                hallucination_sample = torch.load(os.path.join(args.output_path, file))
                influences = []
                for _, v in hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                                "influence"
                            ]
                    influences.append(influence)
                hallucination_samples += influences
            else:
                non_hallucination_sample = torch.load(
                    os.path.join(args.output_path, file)
                )
                influences = []
                for _, v in non_hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx][
                                "influence"
                            ]
                    influences.append(influence)
                non_hallucination_samples += influences

    hallucinated_scores = torch.stack(hallucination_samples)
    non_hallucinated_scores = torch.stack(non_hallucination_samples)

    plt.figure(figsize=(8, 8))
    difference = (
        torch.mean(hallucinated_scores, 0).float()
        - torch.mean(non_hallucinated_scores, 0).float()
    )
    ax = sns.heatmap(difference.cpu().numpy(), cmap="coolwarm", center=0)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    print(f"{args.output_path}/images/constrastive_influences.png")
    plt.savefig(f"{args.output_path}/images/constrastive_influences.png")

    results = {}
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
    indices = [
        [flat_indice.numpy() // 32, flat_indice.numpy() % 32]
        for flat_indice in flat_indices
    ]
    results.update({"hal_heads": indices})
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
    indices = [
        [flat_indice.numpy() // 32, flat_indice.numpy() % 32]
        for flat_indice in flat_indices
    ]
    results.update({"non_hal_heads": indices})
    print(results)

    print(f"{args.output_path}/heads/attribution_result.json")
    with open(f"{args.output_path}/heads/attribution_result.json", "w") as file:
        json.dump(results, file, default=json_custom_serializer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="/nobackup/le/AV_Hallucination/results/videollama2/AVHBench/sampled_entities.json",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="/nobackup/le/AV_Hallucination/data/AVHBench/videos",
    )
    parser.add_argument(
        "--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--modal_type", type=str, default="av")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nobackup/le/AV_Hallucination/results/videollama2/AVHBench/attribution",
    )
    parser.add_argument("--influence_score", type=str, default="prob_diff")
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()
    main(args)
    contrastive_score(args)
