"""
encoder_norm_bimodality_exp.py

Stage 1.1 — Probe whether Qwen2.5-Omni 7B's audio / video encoders emit a
**bimodal** output-token-norm distribution, the Sink-or-Not-to-Sink (Kang et
al.) signature of *propagated* sinks at the encoder side.

Per-clip metric (robust z-score, no global cutoff):
  1. Per-token L2 norm at the encoder output.
  2. Per-clip median and IQR (p75 − p25) of those norms.
  3. normalized = (norm − median) / IQR.
  4. For each k ∈ {3, 5, 7}, count tokens with normalized > k.
  5. Record the count per clip, and keep the outlier tokens' normalized
     magnitudes for a cross-clip consistency check.

Aggregate per modality:
  - distribution of per-clip outlier counts at each k (mean, median, variance,
    histogram).
  - pooled outlier-magnitude distribution: do high-norm tokens carry a
    consistent robust z across clips, or are they all over the place?
  - the classic global-pooled raw-norm histogram (Sink-or-Not-to-Sink Fig 3A
    format) for the secondary view.

Outputs (--output_dir, default results/qwen2_5_omni/sink_analysis/stage1_1_encoder_norms):
    encoder_norm_histograms.png   2 rows (modalities) × 2 cols:
                                  (a) per-clip count histogram (overlay of k)
                                  (b) global-pooled norm histogram (log y)
    encoder_norm_stats.txt        per-modality, per-k counts + outlier-z
                                  statistics + per-k verdict

Stop after this — no downstream analysis.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402


SEED = 42
K_VALUES = (3.0, 5.0, 7.0)
PROMPT_BY_MODAL = {
    "a":  "Describe what you hear in detail.",
    "v":  "Describe what you see in detail.",
    "av": "Describe what you see and hear in detail.",
}


# --------------------------------------------------------------------------
# Per-clip norm caching (flat + offsets so we can store ragged lists in npz)
# --------------------------------------------------------------------------

def _pack_per_clip(per_clip):
    """Flatten a list of 1D arrays into (flat, offsets) for npz storage."""
    if not per_clip:
        return np.array([], dtype=np.float32), np.zeros(1, dtype=np.int64)
    flat = np.concatenate([n.astype(np.float32) for n in per_clip])
    offsets = np.zeros(len(per_clip) + 1, dtype=np.int64)
    for i, n in enumerate(per_clip):
        offsets[i + 1] = offsets[i] + len(n)
    return flat, offsets


def _save_norms_npz(
    out_path,
    audio_norms, audio_names,
    video_norms, video_names,
):
    audio_flat, audio_offsets = _pack_per_clip(audio_norms)
    video_flat, video_offsets = _pack_per_clip(video_norms)
    np.savez_compressed(
        out_path,
        audio_flat=audio_flat,
        audio_offsets=audio_offsets,
        audio_names=np.array(audio_names, dtype=object),
        video_flat=video_flat,
        video_offsets=video_offsets,
        video_names=np.array(video_names, dtype=object),
    )


# --------------------------------------------------------------------------
# Encoder discovery + output unpacking
# --------------------------------------------------------------------------

def _resolve_encoders(model):
    thinker = model.thinker
    audio_mod = None
    visual_mod = None
    for attr in ("audio_tower", "audio_encoder", "audio_model"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr)
            break
    for attr in ("visual", "visual_tower", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr)
            break
    if audio_mod is None or visual_mod is None:
        present = [a for a in dir(thinker) if not a.startswith("_")]
        raise SystemExit(
            f"Encoder lookup failed. audio_mod={audio_mod}, "
            f"visual_mod={visual_mod}. thinker attrs: {present}"
        )
    print(f"  Audio encoder : {type(audio_mod).__name__}")
    print(f"  Vision encoder: {type(visual_mod).__name__}")
    return audio_mod, visual_mod


def _extract_tokens(output) -> torch.Tensor:
    x = output
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    if x.dim() == 3:
        x = x[0]
    elif x.dim() != 2:
        raise RuntimeError(
            f"unexpected encoder output rank {x.dim()}: shape {tuple(x.shape)}"
        )
    return x


# --------------------------------------------------------------------------
# Per-clip norm collection
# --------------------------------------------------------------------------

def collect_norms(model, processor, encoder_module, clips, modal_type, label):
    """Return (per_clip_norms, per_clip_names). `per_clip_names[i]` is the
    file name of the clip that produced `per_clip_norms[i]`. Failed clips
    are absent from both lists; indices stay aligned."""
    per_clip_norms: list[np.ndarray] = []
    per_clip_names: list[str] = []
    buffer: list[np.ndarray] = []

    def hook(module, inp, out):
        tokens = _extract_tokens(out)
        norms = tokens.detach().norm(dim=-1).float().cpu().numpy()
        buffer.append(norms)

    h = encoder_module.register_forward_hook(hook)
    failed = 0
    prompt = PROMPT_BY_MODAL[modal_type]
    try:
        for clip in tqdm(clips, desc=label):
            buffer.clear()
            conv = build_conversation(str(clip), prompt, modal_type)
            try:
                inputs, _ = prepare_inputs(
                    processor, conv, modal_type, model.device, model.dtype
                )
            except Exception as e:
                print(f"  skip {clip.name}: preprocess error: {e}")
                failed += 1
                continue
            try:
                with torch.inference_mode():
                    model.thinker(
                        **inputs,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
            except Exception as e:
                print(f"  skip {clip.name}: forward error: {e}")
                failed += 1
                continue
            torch.cuda.empty_cache()
            if buffer:
                per_clip_norms.append(np.concatenate(buffer))
                per_clip_names.append(clip.name)
    finally:
        h.remove()
    if failed:
        print(f"  ({failed} clips failed; {len(per_clip_norms)} succeeded)")
    return per_clip_norms, per_clip_names


# --------------------------------------------------------------------------
# Per-clip robust-z metric + aggregation
# --------------------------------------------------------------------------

def per_clip_robust_z(norms: np.ndarray):
    """Return (median, iqr, normalized) for one clip's token-norm array."""
    if norms.size == 0:
        return 0.0, 0.0, np.array([])
    median = float(np.median(norms))
    p25, p75 = np.percentile(norms, [25, 75])
    iqr = float(p75 - p25)
    # If IQR collapses (all tokens nearly equal), avoid div-by-zero. A tiny
    # epsilon keeps the per-clip relative geometry intact while flagging the
    # degenerate case.
    safe_iqr = iqr if iqr > 0 else 1e-12
    normalized = (norms - median) / safe_iqr
    return median, iqr, normalized


def aggregate_per_clip(per_clip_norms: list[np.ndarray], k_values=K_VALUES):
    """Return (counts_per_k, outliers_per_k):
       counts_per_k[k]   = np.ndarray of length n_clips, per-clip outlier count
       outliers_per_k[k] = np.ndarray, pooled normalized magnitudes of all
                           outlier tokens (across all clips)
    """
    counts: dict[float, list[int]] = {k: [] for k in k_values}
    outs: dict[float, list[np.ndarray]] = {k: [] for k in k_values}
    for norms in per_clip_norms:
        _, _, normalized = per_clip_robust_z(norms)
        for k in k_values:
            mask = normalized > k
            counts[k].append(int(mask.sum()))
            outs[k].append(normalized[mask])
    return (
        {k: np.array(counts[k]) for k in k_values},
        {k: (np.concatenate(outs[k]) if outs[k] else np.array([])) for k in k_values},
    )


# --------------------------------------------------------------------------
# Report + plot
# --------------------------------------------------------------------------

def verdict_for_k(median_count: float, bimodal_min: float, bimodal_max: float) -> str:
    return "Bimodal" if bimodal_min <= median_count <= bimodal_max else "Unimodal"


def build_report(stats: dict, args) -> list[str]:
    blocks: list[str] = []
    for name, s in stats.items():
        block = [
            f"\n## {name} encoder\n",
            f"  clips                 : {s['n_clips']}\n",
            f"  total tokens          : {s['total_tokens']}\n",
            f"  pooled median norm    : {s['pooled_median']:.4g}\n",
            f"  pooled p99 norm       : {s['pooled_p99']:.4g}\n",
            f"  pooled max norm       : {s['pooled_max']:.4g}\n",
            f"\n  Per-clip robust z   = (norm − per-clip median) / per-clip IQR\n",
            f"  {'k':>3}  {'mean':>6}  {'median':>6}  {'var':>7}   "
            f"{'#outliers':>9}   {'mean(z)':>7}  {'std(z)':>7}   verdict\n",
        ]
        for k in K_VALUES:
            counts = s["counts_per_k"][k]
            outs = s["outliers_per_k"][k]
            mean_c = float(counts.mean())
            median_c = float(np.median(counts))
            var_c = float(counts.var())
            n_outliers = int(outs.size)
            mean_z = float(outs.mean()) if n_outliers else float("nan")
            std_z = float(outs.std()) if n_outliers else float("nan")
            verdict = verdict_for_k(median_c, args.bimodal_min, args.bimodal_max)
            block.append(
                f"  {int(k):>3}  {mean_c:>6.2f}  {median_c:>6.0f}  {var_c:>7.2f}   "
                f"{n_outliers:>9}   {mean_z:>7.2f}  {std_z:>7.2f}   {verdict}\n"
            )
        blocks.append("".join(block))
    return blocks


_MODAL_COLOR = {"audio": "#1f77b4", "video": "#d62728"}
_K_COLORS = {3.0: "#bdbdbd", 5.0: "#737373", 7.0: "#252525"}


def plot(stats: dict, png_path: Path):
    n_modalities = len(stats)
    fig, axes = plt.subplots(n_modalities, 2, figsize=(15, 4.5 * n_modalities))
    if n_modalities == 1:
        axes = np.array([axes])

    for row, (name, s) in enumerate(stats.items()):
        modality_color = _MODAL_COLOR.get(name, "#888888")

        # ---- (a) per-clip outlier-count histogram, overlay of k -----------
        ax = axes[row, 0]
        max_count = 0
        for k in K_VALUES:
            c = s["counts_per_k"][k]
            if c.size:
                max_count = max(max_count, int(c.max()))
        bins = np.arange(0, max(max_count + 2, 10))
        for k in K_VALUES:
            counts = s["counts_per_k"][k]
            ax.hist(
                counts, bins=bins, alpha=0.65, color=_K_COLORS[k],
                edgecolor="white", linewidth=0.6,
                label=f"k={int(k)}  median={np.median(counts):.0f}",
            )
        ax.set_xlabel("# high-norm tokens per clip")
        ax.set_ylabel("# clips")
        ax.set_title(
            f"({chr(ord('a'))})  {name.capitalize()} encoder — per-clip "
            f"outlier count across {s['n_clips']} clips"
        )
        ax.legend(loc="upper right", title="threshold k (robust z)")
        ax.grid(True, linestyle=":", alpha=0.4)

        # ---- (b) global-pooled raw-norm histogram, log y ------------------
        ax2 = axes[row, 1]
        ax2.hist(s["pooled_norms"], bins=80, color=modality_color, alpha=0.78)
        ax2.axvline(
            s["pooled_median"], color="gray", linestyle=":", linewidth=1.2,
            label=f"pooled median = {s['pooled_median']:.2g}",
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("L2 norm of encoder-output token")
        ax2.set_ylabel("count (log)")
        ax2.set_title(
            f"({chr(ord('b'))})  {name.capitalize()} encoder — global-pooled "
            f"norm histogram ({s['total_tokens']:,} tokens)"
        )
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    random.seed(SEED)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path)
    audio_mod, visual_mod = _resolve_encoders(model)

    # --- audio pass ---
    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise SystemExit(f"No .wav files in {audio_dir}")
    random.shuffle(audio_files)
    audio_clips = audio_files[: args.n_clips]
    print(f"\nAudio pass: {len(audio_clips)} clips from {audio_dir}")
    audio_norms, audio_names = collect_norms(
        model, processor, audio_mod, audio_clips, "a", "AudioSet (audio)"
    )

    # --- video pass ---
    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        raise SystemExit(f"No .mp4 files in {video_dir}")
    random.shuffle(video_files)
    video_clips = video_files[: args.n_clips]
    print(f"\nVideo pass: {len(video_clips)} clips from {video_dir}")
    video_norms, video_names = collect_norms(
        model, processor, visual_mod, video_clips, "v", "ActivityNet (video)"
    )

    # --- cache per-clip norms to disk so downstream analyses (e.g.
    # Stage 1.1 v2 with global percentile thresholds) can reuse them
    # without recomputing forward passes. ---
    cache_path = out_dir / "encoder_norms.npz"
    _save_norms_npz(
        cache_path,
        audio_norms=audio_norms, audio_names=audio_names,
        video_norms=video_norms, video_names=video_names,
    )
    print(f"\ncached per-clip norms to {cache_path}")

    # --- aggregate per modality ---
    stats: dict[str, dict] = {}
    for name, per_clip in (("audio", audio_norms), ("video", video_norms)):
        if not per_clip:
            print(f"[{name}] no clips processed — skipping.")
            continue
        counts_per_k, outliers_per_k = aggregate_per_clip(per_clip, K_VALUES)
        pooled = np.concatenate(per_clip)
        stats[name] = {
            "n_clips": len(per_clip),
            "total_tokens": int(pooled.size),
            "pooled_norms": pooled,
            "pooled_median": float(np.median(pooled)),
            "pooled_p99": float(np.percentile(pooled, 99)),
            "pooled_max": float(pooled.max()),
            "counts_per_k": counts_per_k,
            "outliers_per_k": outliers_per_k,
        }

    if not stats:
        raise SystemExit("Nothing aggregated — bailing.")

    # --- printed + saved report ---
    print("\n" + "=" * 78)
    print(
        f"Per-clip robust-z bimodality  (k ∈ {tuple(int(k) for k in K_VALUES)},  "
        f"bimodal band on median count: [{args.bimodal_min}, {args.bimodal_max}])"
    )
    print("=" * 78)
    blocks = build_report(stats, args)
    for b in blocks:
        print(b, end="")

    stats_path = out_dir / "encoder_norm_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Encoder-side bimodality — per-clip robust z-score\n")
        f.write("=" * 78 + "\n")
        f.write(f"n_clips_per_modality = {args.n_clips}\n")
        f.write(f"k_values             = {tuple(int(k) for k in K_VALUES)}\n")
        f.write(
            f"bimodal_band         = [{args.bimodal_min}, {args.bimodal_max}]  "
            f"(applied to median per-clip outlier count)\n"
        )
        f.write(f"audio_dir            = {audio_dir}\n")
        f.write(f"video_dir            = {video_dir}\n")
        for b in blocks:
            f.write(b)
    print(f"\nwrote {stats_path}")

    # --- plot ---
    png_path = out_dir / "encoder_norm_histograms.png"
    plot(stats, png_path)
    print(f"wrote {png_path}")
    print("\nDone — Stage 1.1 complete. No further analysis run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument(
        "--audio_dir", default=str(_REPO / "data/AudioSet/audios"),
        help="Directory of *.wav clips for the audio-encoder pass.",
    )
    p.add_argument(
        "--video_dir", default=str(_REPO / "data/ActivityNet/videos"),
        help="Directory of *.mp4 clips for the vision-encoder pass.",
    )
    p.add_argument("--n_clips", type=int, default=300)
    p.add_argument(
        "--bimodal_min", type=float, default=1.0,
        help="Lower bound on median per-clip outlier count for a 'Bimodal' verdict.",
    )
    p.add_argument(
        "--bimodal_max", type=float, default=5.0,
        help="Upper bound on median per-clip outlier count for a 'Bimodal' verdict.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/stage1_1_encoder_norms"),
    )
    args = p.parse_args()
    main(args)
