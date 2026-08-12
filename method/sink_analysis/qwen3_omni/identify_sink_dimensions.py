"""
identify_sink_dimensions.py

Identify Qwen3-Omni 30B's sink dimensions following Kang et al. (2025).

A sink dimension is a hidden-state channel that, *for any input*, carries an
abnormally large activation magnitude across most layers. Locating these
indices up front is the prerequisite for the "LLM-emerged vs propagated" sink
classification we want to do later — both populations live in the LLM hidden
states, but they activate disjoint subsets of channels.

NORM CONVENTION = RMSNorm (matching identify_sink_dimensions_base.py, so the
Omni+audio D_sink is directly comparable to the base-LLM D_sink_base). RMSNorm
here is pure normalization x / sqrt(mean(x^2) + eps), no learned weight.

Procedure (per the spec):
  1. Run an audio-only forward pass on each of N AudioCaps-style clips with the
     fixed prompt "Describe what you hear in detail."
  2. Cache hidden states at every (layer, token position).
  3. For each layer l, compute mean_abs[l, d] = mean |RMSNorm(x)[d]| over
     (clips, positions).
  4. Per layer, flag dimensions where mean_abs[l, d] > MEDIAN_MULT * median.
  5. Pool: a dimension is a sink dimension if it is flagged in > LAYER_FRAC
     of layers.

Outputs (under --output_dir):
    sink_dimensions.csv     dim, fraction_layers_flagged,
                            mean_magnitude_all_layers, is_sink_dim
    sink_dimensions.png     for each sink dim, line of mean_abs across layers
                            + the per-layer 10× median cutoff

Sanity-checks the result, then asks for confirmation before writing if the
count is implausible (expected 2–5; >10 = threshold too loose).
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reuse the existing Qwen2.5-Omni loader & conversation builder.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen3_omni"))
from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402


SEED = 42
PROMPT = "Describe what you hear in detail."


def _rmsnorm_abs(hs: torch.Tensor, eps: float) -> torch.Tensor:
    """|RMSNorm(x)| per position, pure normalization (no learned weight).
    hs: (seq, H) float -> (seq, H). Matches identify_sink_dimensions_base.py."""
    rms = torch.sqrt(hs.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (hs / rms).abs()


def _thinker_rms_eps(model) -> float:
    """rms_norm_eps of the thinker's LLM (Qwen2 text config)."""
    cfg = model.thinker.config
    cfg = getattr(cfg, "text_config", cfg)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def main(args):
    random.seed(SEED)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir)
    candidates = sorted(audio_dir.glob("*.wav"))
    if not candidates:
        raise SystemExit(f"No .wav files found in {audio_dir}")
    random.shuffle(candidates)
    audio_files = candidates[: args.n_clips]
    print(f"Pool: {len(candidates)} clips in {audio_dir} → using {len(audio_files)}")
    print(f"Prompt: {PROMPT!r}")

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path)
    eps = _thinker_rms_eps(model)
    print(f"  RMSNorm convention (pure, no weight), rms_norm_eps = {eps}")

    # Running accumulators (allocated on the first successful forward when we
    # know L, H — we never store full hidden states across clips).
    sum_per_dim: torch.Tensor | None = None    # (L, H) float64
    count_per_dim: torch.Tensor | None = None  # (L,)  int64

    failed = 0
    succeeded = 0
    for path in tqdm(audio_files, desc="Forward passes"):
        conv = build_conversation(str(path), PROMPT, "a")
        try:
            inputs, _ = prepare_inputs(
                processor, conv, "a", model.device, model.dtype
            )
        except Exception as e:
            print(f"  skip {path.name}: preprocess error: {e}")
            failed += 1
            continue

        try:
            with torch.inference_mode():
                outputs = model.thinker(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
        except Exception as e:
            print(f"  skip {path.name}: forward error: {e}")
            failed += 1
            continue

        # `hidden_states` is a tuple of length (num_decoder_layers + 1):
        #   [0] = output of the embedding layer (pre-block-0)
        #   [1..N] = output after decoder layer 1..N
        # Include them all so we can see whether sink magnitude is present
        # already at the embedding (typical for inherited sink dims) or only
        # builds up across blocks.
        hidden_states = outputs.hidden_states
        L = len(hidden_states)
        H = hidden_states[0].shape[-1]
        if sum_per_dim is None:
            sum_per_dim = torch.zeros(L, H, dtype=torch.float64)
            count_per_dim = torch.zeros(L, dtype=torch.int64)

        for l, hs in enumerate(hidden_states):
            # hs: (batch=1, seq, hidden) bf16 on GPU. Apply RMSNorm per position,
            # then move per-layer to CPU float64 right away so we never hold a
            # multi-layer stack on GPU at once.
            flat = _rmsnorm_abs(hs[0].float(), eps).to(torch.float64).cpu()
            sum_per_dim[l] += flat.sum(0)
            count_per_dim[l] += flat.shape[0]

        del outputs
        torch.cuda.empty_cache()
        succeeded += 1

    if sum_per_dim is None:
        raise SystemExit("No successful forward passes — bailing.")
    if failed:
        print(f"({failed} clips failed; {succeeded} aggregated)")

    mean_abs = (sum_per_dim / count_per_dim[:, None]).numpy()    # (L, H)
    L, H = mean_abs.shape

    # Per-layer cutoff.
    medians = np.median(mean_abs, axis=1)                         # (L,)
    cutoff_per_layer = args.median_mult * medians                 # (L,)
    flagged_per_layer = mean_abs > cutoff_per_layer[:, None]      # (L, H) bool

    fraction_flagged = flagged_per_layer.mean(axis=0)             # (H,)
    sink_dims = np.where(fraction_flagged > args.layer_frac)[0]   # (|D_sink|,)

    # ----------------------------- sanity --------------------------------
    print("=" * 70)
    print("Sanity checks (no files written yet)")
    print("=" * 70)
    print(f"  layers (incl. embedding output): {L}")
    print(f"  hidden dim                     : {H}")
    print(f"  median_mult                    : {args.median_mult}")
    print(f"  layer_frac threshold           : {args.layer_frac}")
    print(f"  sink dimensions identified     : {len(sink_dims)}")
    print(f"    -> {list(sink_dims)}")

    if len(sink_dims) > 0:
        mean_over_layers = mean_abs.mean(axis=0)
        flagged_mag = float(mean_over_layers[sink_dims].mean())
        mask = np.ones(H, dtype=bool)
        mask[sink_dims] = False
        other_mag = float(mean_over_layers[mask].mean())
        ratio = flagged_mag / max(other_mag, 1e-12)
        print(f"  Mean |RMSNorm(x)| flagged dims : {flagged_mag:.4g}")
        print(f"  Mean |RMSNorm(x)| other   dims : {other_mag:.4g}")
        print(f"  Ratio                          : {ratio:.1f}× (target ≥10×)")

        consistency = fraction_flagged[sink_dims]
        print(
            f"  Flagged-in-layer fraction      : "
            f"min={consistency.min():.2f} mean={consistency.mean():.2f} "
            f"max={consistency.max():.2f}"
        )

    plausible = 2 <= len(sink_dims) <= 10
    if not plausible:
        print(
            f"\n  WARNING: count {len(sink_dims)} is outside the expected 2–10 "
            f"range. Consider tweaking --median_mult "
            f"(current {args.median_mult}) or --layer_frac "
            f"(current {args.layer_frac})."
        )

    if not args.force and not plausible:
        try:
            ans = input(
                "Counts look implausible. Proceed and write outputs anyway? "
                "[y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != "y":
            print("Aborted. Re-run with adjusted thresholds.")
            return

    # ----------------------------- outputs -------------------------------
    df = pd.DataFrame({
        "dim": np.arange(H),
        "fraction_layers_flagged": fraction_flagged,
        "mean_magnitude_all_layers": mean_abs.mean(axis=0),
        "is_sink_dim": np.isin(np.arange(H), sink_dims),
    })
    csv_path = out_dir / "sink_dimensions.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(L)
    if len(sink_dims) > 0:
        # Pick a colour cycle that's readable for up to ~10 dims.
        cmap = plt.get_cmap("tab10")
        for i, d in enumerate(sink_dims):
            ax.plot(
                layers, mean_abs[:, d],
                marker="o", markersize=4, linewidth=1.6,
                color=cmap(i % 10),
                label=f"dim {d}",
            )
    ax.plot(
        layers, cutoff_per_layer,
        color="gray", linestyle="--", linewidth=1.5,
        label=f"{args.median_mult:g}× per-layer median (cutoff)",
    )
    # Also plot the per-layer median as a faint reference.
    ax.plot(
        layers, medians,
        color="lightgray", linestyle=":", linewidth=1.0,
        label="per-layer median (all dims)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("LLM hidden-state index (0 = embedding output, then layers 1..N)")
    ax.set_ylabel("Mean |RMSNorm(x)[dim]| across tokens & clips")
    ax.set_title(
        f"Qwen3-Omni 30B sink dimensions — RMSNorm  "
        f"(N={succeeded} clips, |D_sink|={len(sink_dims)})"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    png_path = out_dir / "sink_dimensions.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695")
    p.add_argument(
        "--audio_dir",
        default=str(_REPO / "data/AudioSet/audios"),
        help="Directory of *.wav clips to sample from. AudioSet is fine "
             "(AudioCaps-equivalent for this audio-only forward).",
    )
    p.add_argument("--n_clips", type=int, default=100)
    p.add_argument(
        "--median_mult", type=float, default=20.0,
        help="Per-layer cutoff multiplier: dim flagged if "
             "mean_abs[l, d] > median_mult * median(mean_abs[l, :]).",
    )
    p.add_argument(
        "--layer_frac", type=float, default=0.5,
        help="Fraction of layers a dim must be flagged in to count as a sink dim.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen3_omni/sink_analysis/sink_dimensions"),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Skip the confirmation prompt when sanity check warns.",
    )
    args = p.parse_args()
    main(args)
