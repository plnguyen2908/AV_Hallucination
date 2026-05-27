"""
av_fusion_scatter_exp.py

For every (layer, head) in the Qwen2.5-Omni thinker, plot:
    x = audio contrastive influence score  (from AudioSet_describe)
    y = visual contrastive influence score (from ActivityNet_describe)

Color by membership in the two single-modality "hal head" top-K lists:
    BLUE   = visual halluc head (ActivityNet) only
    RED    = audio  halluc head (AudioSet) only
    GREEN  = head appears in BOTH top-K lists
    GRAY   = neither

Both axes are forced symmetric around 0; dashed reference lines mark x=0
and y=0. Heads in the top-right quadrant are positive on both modalities
(high hallucination influence on both); top-left = visual-only positive;
bottom-right = audio-only positive; etc.

Usage:
    python method/qwen2_5_omni/av_fusion_scatter_exp.py --top_k 40
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


_REPO = Path(__file__).resolve().parent.parent.parent

_DEFAULT_AUDIO_DIR = _REPO / "results/qwen2_5_omni/AudioSet_describe/attribution"
_DEFAULT_VISUAL_DIR = _REPO / "results/qwen2_5_omni/ActivityNet_describe/attribution"
_DEFAULT_AUDIO_HEADS = (
    _REPO
    / "results/qwen2_5_omni/AudioSet_describe/attribution/heads/attribution_result.json"
)
_DEFAULT_VISUAL_HEADS = (
    _REPO
    / "results/qwen2_5_omni/ActivityNet_describe/attribution/heads/attribution_result.json"
)
_DEFAULT_OUT = (
    _REPO / "results/qwen2_5_omni/av_fusion_scatter.png"
)

# User-requested palette.
COLOR_AUDIO = "red"     # audio halluc head (AudioSet) only
COLOR_VISUAL = "blue"   # visual halluc head (ActivityNet) only
COLOR_BOTH = "green"    # head appears in both top-K lists
COLOR_OTHER = "lightgray"


def compute_difference(attribution_dir: Path):
    """Mirror `contrastive_score` in identify_halluc_head.py: walk the per-
    sample .pth files and return `mean_hal − mean_non_hal` as a numpy array
    shaped (layer_num, head_num)."""
    pth_dir = attribution_dir / "pth"
    if not pth_dir.is_dir():
        raise SystemExit(f"pth dir not found: {pth_dir}")

    hal_samples: list = []
    non_hal_samples: list = []
    layer_num = head_num = None

    for fname in sorted(os.listdir(pth_dir)):
        if not fname.endswith(".pth"):
            continue
        is_hal = fname.startswith("hal")
        data = torch.load(pth_dir / fname, weights_only=False)
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
            f"Need hal + non_hal samples in {pth_dir}; "
            f"got {len(hal_samples)} / {len(non_hal_samples)}"
        )

    mean_hal = torch.stack(hal_samples).mean(0).float()
    mean_non_hal = torch.stack(non_hal_samples).mean(0).float()
    return (mean_hal - mean_non_hal).cpu().numpy(), layer_num, head_num


def load_heads(json_path: Path, key: str, top_k: int) -> set:
    """Return the first `top_k` entries under `key` as a set of (layer, head)
    tuples."""
    if not json_path.exists():
        raise SystemExit(f"Heads JSON not found: {json_path}")
    with open(json_path) as f:
        d = json.load(f)
    if key not in d:
        raise SystemExit(f"Key {key!r} not found in {json_path}. Available: {list(d)}")
    return {tuple(h) for h in d[key][:top_k]}


def main(args):
    print("Computing AudioSet contrastive difference ...")
    audio_diff, La, Ha = compute_difference(Path(args.audio_attribution_dir))
    print(f"  shape: ({La}, {Ha})")

    print("Computing ActivityNet contrastive difference ...")
    visual_diff, Lv, Hv = compute_difference(Path(args.visual_attribution_dir))
    print(f"  shape: ({Lv}, {Hv})")

    if (La, Ha) != (Lv, Hv):
        raise SystemExit(
            f"Layer/head shape mismatch: audio ({La},{Ha}) vs visual ({Lv},{Hv}). "
            "Both attribution runs must use the same Qwen2.5-Omni model."
        )
    L, H = La, Ha

    audio_heads = load_heads(Path(args.audio_heads_json), args.heads_key, args.top_k)
    visual_heads = load_heads(Path(args.visual_heads_json), args.heads_key, args.top_k)
    both_heads = audio_heads & visual_heads
    audio_only = audio_heads - both_heads
    visual_only = visual_heads - both_heads

    print(
        f"top-{args.top_k}: audio={len(audio_heads)}, visual={len(visual_heads)}, "
        f"both={len(both_heads)}"
    )

    # Build per-class point clouds.
    def _xy_for(heads_subset):
        xs, ys = [], []
        for li, hi in heads_subset:
            xs.append(audio_diff[li, hi])
            ys.append(visual_diff[li, hi])
        return xs, ys

    # Heads not in either list — "other".
    other_li, other_hi = [], []
    flagged = audio_heads | visual_heads
    for li in range(L):
        for hi in range(H):
            if (li, hi) in flagged:
                continue
            other_li.append(li)
            other_hi.append(hi)
    other_x = audio_diff[other_li, other_hi] if other_li else np.array([])
    other_y = visual_diff[other_li, other_hi] if other_li else np.array([])

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(9, 9))

    # Background "other" heads first so the highlighted ones sit on top.
    ax.scatter(
        other_x, other_y,
        s=14, c=COLOR_OTHER, alpha=0.5, edgecolors="none",
        label=f"Other ({len(other_li)})",
    )
    xs, ys = _xy_for(audio_only)
    ax.scatter(
        xs, ys,
        s=46, c=COLOR_AUDIO, alpha=0.9, edgecolors="black", linewidths=0.4,
        label=f"Audio halluc only (AudioSet) — {len(audio_only)}",
    )
    xs, ys = _xy_for(visual_only)
    ax.scatter(
        xs, ys,
        s=46, c=COLOR_VISUAL, alpha=0.9, edgecolors="black", linewidths=0.4,
        label=f"Visual halluc only (ActivityNet) — {len(visual_only)}",
    )
    xs, ys = _xy_for(both_heads)
    ax.scatter(
        xs, ys,
        s=80, c=COLOR_BOTH, alpha=0.95, edgecolors="black", linewidths=0.6,
        marker="D",
        label=f"Both (audio ∩ visual) — {len(both_heads)}",
    )

    # Symmetric axis limits centred on 0, with a small margin.
    x_abs = max(np.abs(audio_diff).max(), 1e-12) * 1.05
    y_abs = max(np.abs(visual_diff).max(), 1e-12) * 1.05
    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(-y_abs, y_abs)

    # Reference dashed lines through the origin.
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Audio contrastive score (AudioSet hal − non-hal)", fontsize=12)
    ax.set_ylabel("Visual contrastive score (ActivityNet hal − non-hal)", fontsize=12)
    ax.set_title(
        "Per-head contrastive scores: audio vs visual\n"
        f"top-{args.top_k} halluc heads highlighted",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_attribution_dir", default=str(_DEFAULT_AUDIO_DIR),
        help="AudioSet describe attribution directory (with pth/ inside).",
    )
    parser.add_argument(
        "--visual_attribution_dir", default=str(_DEFAULT_VISUAL_DIR),
        help="ActivityNet describe attribution directory (with pth/ inside).",
    )
    parser.add_argument(
        "--audio_heads_json", default=str(_DEFAULT_AUDIO_HEADS),
        help="AudioSet attribution_result.json — audio halluc heads source.",
    )
    parser.add_argument(
        "--visual_heads_json", default=str(_DEFAULT_VISUAL_HEADS),
        help="ActivityNet attribution_result.json — visual halluc heads source.",
    )
    parser.add_argument(
        "--heads_key", default="hal_heads_contrastive",
        choices=[
            "hal_heads_contrastive",
            "non_hal_heads_contrastive",
            "hal_heads_mean",
            "non_hal_heads_mean",
        ],
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--output_path", default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    main(args)
