"""
av_fusion_heatmap_exp.py

Plot the VGGSounder contrastive hallucination-influence heatmap (audio-visual
"av" modality) and overlay the modality-specific hallucination heads that
were identified on the two single-modality probes:

    BLUE   = audio halluc heads — from `AudioSet_describe`
    GREEN  = visual halluc heads — from `ActivityNet_describe`
    ORANGE = head appears in BOTH lists

The single-modality lists are read from the existing `attribution_result.json`
files; the VGGSounder difference heatmap is recomputed on the fly from the
per-sample .pth influence files in
    <vgg_attribution_dir>/pth/
(same data the standard identify_halluc_head.py heatmap is drawn from).

Usage:
    python method/qwen2_5_omni/av_fusion_heatmap_exp.py \\
        --top_k 40
    python method/qwen2_5_omni/av_fusion_heatmap_exp.py \\
        --vgg_attribution_dir results/qwen2_5_omni/VGGSounder_describe/attribution \\
        --audio_heads_json results/qwen2_5_omni/AudioSet_describe/attribution/heads/attribution_result.json \\
        --visual_heads_json results/qwen2_5_omni/ActivityNet_describe/attribution/heads/attribution_result.json \\
        --output_path results/qwen2_5_omni/VGGSounder_describe/av_fusion_heatmap.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

_REPO = Path(__file__).resolve().parent.parent.parent

_DEFAULT_VGG_DIR = _REPO / "results/qwen2_5_omni/VGGSounder_describe/attribution"
_DEFAULT_AUDIO_HEADS = (
    _REPO
    / "results/qwen2_5_omni/AudioSet_describe/attribution/heads/attribution_result.json"
)
_DEFAULT_VISUAL_HEADS = (
    _REPO
    / "results/qwen2_5_omni/ActivityNet_describe/attribution/heads/attribution_result.json"
)
_DEFAULT_OUT = _REPO / "results/qwen2_5_omni/VGGSounder_describe/av_fusion_heatmap.png"


def load_heads(json_path: Path, key: str, top_k: int) -> set:
    """Read `attribution_result.json` and return the first `top_k` entries
    under `key` as a set of (layer, head) tuples."""
    if not json_path.exists():
        raise SystemExit(f"Heads JSON not found: {json_path}")
    with open(json_path) as f:
        d = json.load(f)
    if key not in d:
        raise SystemExit(f"Key {key!r} not found in {json_path}. Available: {list(d)}")
    return {tuple(h) for h in d[key][:top_k]}


def compute_vgg_difference(attribution_dir: Path):
    """Recompute VGGSounder's `mean_hal - mean_non_hal` influence matrix
    from the per-sample .pth files. Mirrors `contrastive_score` in
    identify_halluc_head.py — kept inline so this exp script doesn't import
    the heavy generation dependencies.

    Returns (difference_np, layer_num, head_num).
    """
    pth_dir = attribution_dir / "pth"
    if not pth_dir.is_dir():
        raise SystemExit(f"VGGSounder pth dir not found: {pth_dir}")

    hal_samples: list = []
    non_hal_samples: list = []
    layer_num = head_num = None

    for fname in sorted(os.listdir(pth_dir)):
        if not fname.endswith(".pth"):
            continue
        path = pth_dir / fname
        is_hal = fname.startswith("hal")
        data = torch.load(path, weights_only=False)
        if not data:
            continue
        for _, v in data.items():
            if layer_num is None:
                layer_num = len(v)
                head_num = len(v[0])
            influence = torch.zeros(layer_num, head_num)
            for li in range(layer_num):
                for hi in range(head_num):
                    influence[li][hi] = v[li][hi]["influence"]
            influence = torch.nan_to_num(influence, nan=0.0)
            (hal_samples if is_hal else non_hal_samples).append(influence)

    if not hal_samples or not non_hal_samples:
        raise SystemExit(
            "Need both hallucination and non-hallucination influence samples; "
            f"got {len(hal_samples)} / {len(non_hal_samples)} in {pth_dir}"
        )

    mean_hal = torch.stack(hal_samples).mean(0).float()
    mean_non_hal = torch.stack(non_hal_samples).mean(0).float()
    return (mean_hal - mean_non_hal).cpu().numpy(), layer_num, head_num


# Marker colours — keep here so a future tweak (e.g. swap palette) is one edit.
COLOR_AUDIO = "blue"
COLOR_VISUAL = "green"
COLOR_BOTH = "orange"


def plot(difference, layer_num, head_num, audio_set, visual_set, top_k, output_path):
    """Draw the VGGSounder difference heatmap with three classes of head
    rectangles overlaid: audio-only, visual-only, both."""
    both = audio_set & visual_set
    audio_only = audio_set - both
    visual_only = visual_set - both

    fig, ax = plt.subplots(figsize=(10, 10))
    v_limit = np.percentile(np.abs(difference), 99.9)
    sns.heatmap(
        difference,
        cmap="coolwarm",
        center=0,
        vmin=-v_limit,
        vmax=v_limit,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    def _draw(heads, color):
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

    # Order matters for the visual layering: draw single-modality first so
    # the "both" rectangles, drawn last, sit on top.
    _draw(audio_only, COLOR_AUDIO)
    _draw(visual_only, COLOR_VISUAL)
    _draw(both, COLOR_BOTH)

    legend_handles = [
        mpatches.Patch(
            edgecolor=COLOR_AUDIO,
            facecolor="none",
            linewidth=2,
            label=f"Audio halluc head (AudioSet) — {len(audio_only)}",
        ),
        mpatches.Patch(
            edgecolor=COLOR_VISUAL,
            facecolor="none",
            linewidth=2,
            label=f"Visual halluc head (ActivityNet) — {len(visual_only)}",
        ),
        mpatches.Patch(
            edgecolor=COLOR_BOTH,
            facecolor="none",
            linewidth=2,
            label=f"Both (audio ∩ visual) — {len(both)}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        title=f"Head provenance (top {top_k} each)",
    )

    ax.set_title(
        "VGGSounder hal − non-hal contrastive influence\n"
        "single-modality halluc heads overlaid",
        pad=14,
    )
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main(args):
    audio_set = load_heads(Path(args.audio_heads_json), args.heads_key, args.top_k)
    visual_set = load_heads(Path(args.visual_heads_json), args.heads_key, args.top_k)
    print(
        f"AudioSet  hal heads (top {args.top_k}): {len(audio_set)} -> e.g. {sorted(audio_set)[:5]}"
    )
    print(
        f"ActivityNet hal heads (top {args.top_k}): {len(visual_set)} -> e.g. {sorted(visual_set)[:5]}"
    )
    overlap = audio_set & visual_set
    print(
        f"Overlap (both lists)                  : {len(overlap)} -> {sorted(overlap)}"
    )

    diff, L, H = compute_vgg_difference(Path(args.vgg_attribution_dir))
    print(f"VGGSounder difference matrix: ({L}, {H})")
    plot(diff, L, H, audio_set, visual_set, args.top_k, Path(args.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vgg_attribution_dir",
        default=str(_DEFAULT_VGG_DIR),
        help="VGGSounder attribution directory; expects a `pth/` subdir.",
    )
    parser.add_argument(
        "--audio_heads_json",
        default=str(_DEFAULT_AUDIO_HEADS),
        help="AudioSet attribution_result.json — audio halluc heads source.",
    )
    parser.add_argument(
        "--visual_heads_json",
        default=str(_DEFAULT_VISUAL_HEADS),
        help="ActivityNet attribution_result.json — visual halluc heads source.",
    )
    parser.add_argument(
        "--heads_key",
        default="hal_heads_contrastive",
        choices=[
            "hal_heads_contrastive",
            "non_hal_heads_contrastive",
            "hal_heads_mean",
            "non_hal_heads_mean",
        ],
        help="Which head list to overlay from each modality.",
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--output_path",
        default=str(_DEFAULT_OUT),
        help="Where to write the final PNG.",
    )
    args = parser.parse_args()
    main(args)
