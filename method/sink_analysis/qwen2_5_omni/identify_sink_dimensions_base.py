"""
identify_sink_dimensions_base.py

Stage 0.1 (base-LLM variant) — Identify the BASE LLM's inherited sink
dimensions (D_sink_base) by running the text-only Qwen2.5-7B base model on
generic text prompts.

This replaces the earlier Stage 0.1 run (identify_sink_dimensions.py), which used
Qwen2.5-Omni with AUDIO input and therefore conflated base-LLM-inherited sink
dimensions with multimodal-training-induced ones. Qwen2.5-Omni's *thinker* is
initialized from this base LLM, so the dims found here are the inherited
component; the Omni run's D_sink should be a (near-)superset = Inherited +
anything that emerged during multimodal training.

Procedure (mirrors the previous Stage 0.1; NORM CONVENTION = RMSNorm):
  1. For each prompt, one forward pass; cache the hidden state x_i^l at every
     (token position i, layer l) via output_hidden_states.
  2. Per layer l, mean_abs[l, d] = mean over (prompts, positions) of
     |RMSNorm(x_i^l)[d]|, where RMSNorm(x) = x / sqrt(mean(x^2) + eps)
     (pure normalization, no learned weight — well-defined without choosing a
     layer's gamma; the learned-weight variant is a deliberately-omitted option).
  3. Per layer, flag dims where mean_abs[l, d] > MEDIAN_MULT * median_d(mean_abs[l, :]).
  4. Pool: D_sink_base = dims flagged in > LAYER_FRAC of layers.

Outputs (--output_dir):
    D_sink_base.json   {"dims": [...], "hidden_dim": H, "n_layers": L, ...}
    D_sink_base.csv    dim, fraction_layers_flagged, mean_magnitude_all_layers, is_sink_dim
    D_sink_base.png    for each D_sink_base dim, mean_abs across layers + cutoff

Prints sanity checks (count, flagged-vs-unflagged magnitude, hidden-dim match
vs Omni thinker, overlap with the previous Omni D_sink) and stops. Confirm the
result before running the Omni-text-only / Omni-multimodal comparisons.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

SEED = 42

# Built-in generic prompts (topics x templates) so the script runs offline
# without the `datasets` package. The distribution doesn't matter — we're after
# intrinsic model properties.
_TEMPLATES = [
    "What is {}?",
    "Explain how {} works.",
    "Describe {} in a few sentences.",
    "Write a short paragraph about {}.",
    "Why is {} important?",
    "Give an example involving {}.",
    "Summarize the history of {}.",
    "How does {} affect everyday life?",
    "Compare {} with something similar.",
    "List a few facts about {}.",
]
_TOPICS = [
    "the water cycle", "photosynthesis", "the French Revolution", "gravity",
    "machine learning", "the stock market", "volcanoes", "the human heart",
    "democracy", "climate change", "the internet", "ancient Rome",
    "quantum mechanics", "coffee", "the immune system", "black holes",
    "the printing press", "ocean currents", "electricity", "the brain",
    "antibiotics", "the solar system", "language", "evolution", "music theory",
]


def build_prompts(n: int) -> list:
    """n distinct generic prompts from topics x templates (seeded shuffle)."""
    combos = [t.format(topic) for topic, t in
              itertools.product(_TOPICS, _TEMPLATES)]
    rng = random.Random(SEED)
    rng.shuffle(combos)
    if n > len(combos):
        # Pad by cycling if more are requested than the grid provides.
        combos = (combos * (n // len(combos) + 1))
    return combos[:n]


def _rmsnorm_abs(hs: torch.Tensor, eps: float) -> torch.Tensor:
    """|RMSNorm(x)| per position, pure normalization (no learned weight).
    hs: (seq, H) float32 -> (seq, H) float32."""
    rms = torch.sqrt(hs.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (hs / rms).abs()


def _omni_thinker_hidden_dim() -> int:
    """Qwen2.5-Omni thinker hidden size, for the dimension-comparability check.
    Returns -1 if it can't be resolved without a download."""
    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        tcfg = getattr(cfg, "thinker_config", cfg)
        tcfg = getattr(tcfg, "text_config", tcfg)
        return int(tcfg.hidden_size)
    except Exception:
        return -1


def _previous_d_sink(csv_path: Path):
    """Load the previous Omni D_sink (raw-|x| convention) for comparison.
    Returns (dims_list, hidden_dim) or (None, None)."""
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    dims = df.loc[df["is_sink_dim"], "dim"].astype(int).tolist()
    return dims, len(df)


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- prompts -----
    if args.prompts_file:
        prompts = [l.strip() for l in Path(args.prompts_file).read_text().splitlines()
                   if l.strip()][: args.n_prompts]
        print(f"Prompts: {len(prompts)} from {args.prompts_file}")
    else:
        prompts = build_prompts(args.n_prompts)
        print(f"Prompts: {len(prompts)} built-in generic (topics x templates)")

    # ----- model -----
    print(f"Loading text-only base LLM: {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
    H_cfg = int(model.config.hidden_size)
    print(f"  hidden_size = {H_cfg}, rms_norm_eps = {eps}, "
          f"num_hidden_layers = {model.config.num_hidden_layers}")

    # ----- forward passes with running accumulators (never store full HS) -----
    sum_per_dim = None      # (L, H) float64
    count_per_layer = None  # (L,) int64
    failed = succeeded = 0
    for prompt in tqdm(prompts, desc="Forward passes"):
        try:
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(**enc, output_hidden_states=True,
                                use_cache=False, return_dict=True)
        except Exception as e:
            print(f"  skip prompt {prompt!r}: {e}")
            failed += 1
            continue

        hidden_states = outputs.hidden_states  # tuple len (L+1): emb, then layers
        L = len(hidden_states)
        H = hidden_states[0].shape[-1]
        if sum_per_dim is None:
            sum_per_dim = torch.zeros(L, H, dtype=torch.float64)
            count_per_layer = torch.zeros(L, dtype=torch.int64)

        for l, hs in enumerate(hidden_states):
            normed = _rmsnorm_abs(hs[0].float(), eps).to(torch.float64).cpu()
            sum_per_dim[l] += normed.sum(0)
            count_per_layer[l] += normed.shape[0]

        del outputs
        torch.cuda.empty_cache()
        succeeded += 1

    if sum_per_dim is None:
        raise SystemExit("No successful forward passes — bailing.")
    if failed:
        print(f"({failed} prompts failed; {succeeded} aggregated)")

    mean_abs = (sum_per_dim / count_per_layer[:, None]).numpy()   # (L, H)
    L, H = mean_abs.shape

    medians = np.median(mean_abs, axis=1)                          # (L,)
    cutoff_per_layer = args.median_mult * medians                  # (L,)
    flagged_per_layer = mean_abs > cutoff_per_layer[:, None]       # (L, H)
    fraction_flagged = flagged_per_layer.mean(axis=0)              # (H,)
    sink_dims = np.where(fraction_flagged > args.layer_frac)[0]
    mean_over_layers = mean_abs.mean(axis=0)                       # (H,)

    # --------------------------- sanity checks ---------------------------
    print("\n" + "=" * 72)
    print("Sanity checks (no files written yet)  —  D_sink_base, RMSNorm convention")
    print("=" * 72)
    print(f"  layers (incl. embedding output): {L}")
    print(f"  hidden dim (base model)        : {H}")
    print(f"  median_mult / layer_frac       : {args.median_mult} / {args.layer_frac}")
    print(f"  D_sink_base count              : {len(sink_dims)}")
    print(f"    -> {list(map(int, sink_dims))}")
    if not (2 <= len(sink_dims) <= 10):
        print(f"  WARNING: count {len(sink_dims)} outside expected 2–10 — consider "
              f"adjusting --median_mult / --layer_frac.")

    if len(sink_dims) > 0:
        flagged_mag = float(mean_over_layers[sink_dims].mean())
        mask = np.ones(H, dtype=bool); mask[sink_dims] = False
        other_mag = float(mean_over_layers[mask].mean())
        ratio = flagged_mag / max(other_mag, 1e-12)
        print(f"  mean |RMSNorm(x)| flagged dims : {flagged_mag:.4g}")
        print(f"  mean |RMSNorm(x)| other   dims : {other_mag:.4g}")
        print(f"  ratio                          : {ratio:.1f}× (target ≥10×)")
        cons = fraction_flagged[sink_dims]
        print(f"  flagged-in-layer fraction      : min={cons.min():.2f} "
              f"mean={cons.mean():.2f} max={cons.max():.2f}")

    # Hidden-dim comparability vs Omni thinker.
    H_omni = _omni_thinker_hidden_dim()
    prev_csv = Path(args.prev_sink_csv)
    prev_dims, prev_H = _previous_d_sink(prev_csv)
    if H_omni < 0 and prev_H is not None:
        H_omni = prev_H
    print(f"\n  hidden dim — base LLM          : {H}")
    print(f"  hidden dim — Omni thinker      : "
          f"{H_omni if H_omni > 0 else 'unresolved'}")
    if H_omni > 0 and H_omni != H:
        print("  ⚠ hidden dims DIFFER — dimension indices are NOT comparable "
              "across the two models.")
    elif H_omni > 0:
        print("  ✓ hidden dims match — dimension indices are comparable.")

    # Overlap with the previous (Omni+audio) D_sink. NOTE convention mismatch.
    print("\n  Comparison to previous Stage 0.1 D_sink (Omni + audio):")
    if prev_dims is None:
        print(f"    (not found at {prev_csv})")
    else:
        base_set, prev_set = set(map(int, sink_dims)), set(prev_dims)
        print(f"    Omni D_sink                    : {sorted(prev_set)}")
        print(f"    overlap (in both)              : {sorted(base_set & prev_set)}")
        print(f"    base-only (not in Omni)        : {sorted(base_set - prev_set)}")
        print(f"    Omni-only (multimodal-emerged) : {sorted(prev_set - base_set)}")
        print("    NOTE: valid only if the Omni sink_dimensions.csv was generated "
              "with the SAME convention (RMSNorm) & threshold as this run. Confirm "
              "before interpreting the base-only / Omni-only split.")

    # ----------------------------- outputs -------------------------------
    df = pd.DataFrame({
        "dim": np.arange(H),
        "fraction_layers_flagged": fraction_flagged,
        "mean_magnitude_all_layers": mean_over_layers,
        "is_sink_dim": np.isin(np.arange(H), sink_dims),
    })
    csv_path = out_dir / "D_sink_base.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    json_path = out_dir / "D_sink_base.json"
    with open(json_path, "w") as f:
        json.dump({
            "dims": list(map(int, sink_dims)),
            "hidden_dim": H,
            "n_layers": L,
            "model_path": args.model_path,
            "norm": "rmsnorm_no_weight",
            "median_mult": args.median_mult,
            "layer_frac": args.layer_frac,
            "n_prompts": succeeded,
        }, f, indent=2)
    print(f"wrote {json_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(L)
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(sink_dims):
        ax.plot(layers, mean_abs[:, d], marker="o", markersize=4, linewidth=1.6,
                color=cmap(i % 10), label=f"dim {d}")
    ax.plot(layers, cutoff_per_layer, color="gray", linestyle="--", linewidth=1.5,
            label=f"{args.median_mult:g}× per-layer median (cutoff)")
    ax.plot(layers, medians, color="lightgray", linestyle=":", linewidth=1.0,
            label="per-layer median (all dims)")
    ax.set_yscale("log")
    ax.set_xlabel("hidden-state index (0 = embedding output, then layers 1..N)")
    ax.set_ylabel("mean |RMSNorm(x)[dim]| across tokens & prompts")
    ax.set_title(f"Qwen2.5-7B base — D_sink_base "
                 f"(N={succeeded} prompts, |D_sink_base|={len(sink_dims)})")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    png_path = out_dir / "D_sink_base.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"wrote {png_path}")

    print("\nDone — D_sink_base identified. Confirm it looks reasonable before "
          "running the Omni-text-only / Omni-multimodal comparisons.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path", default="Qwen/Qwen2.5-7B",
        help="Text-only base LLM. Default 'Qwen/Qwen2.5-7B' (base, non-instruct, "
             "which Qwen2.5-Omni's thinker is initialized from). Use "
             "'Qwen/Qwen2.5-7B-Instruct' if the base isn't available.",
    )
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument(
        "--prompts_file", default=None,
        help="Optional text file, one prompt per line (overrides the built-in set).",
    )
    p.add_argument("--median_mult", type=float, default=20.0)
    p.add_argument("--layer_frac", type=float, default=0.5)
    p.add_argument(
        "--prev_sink_csv",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                    "sink_dimensions/sink_dimensions.csv"),
        help="Previous (Omni+audio) sink_dimensions.csv for the overlap check.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/sink_dimensions_base"),
    )
    args = p.parse_args()
    main(args)
