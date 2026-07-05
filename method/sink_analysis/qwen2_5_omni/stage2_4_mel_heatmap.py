"""
stage2_4_mel_heatmap.py

Mel-spectrogram-aggregated temporal heatmap for Stage 2.4 audio sinks.

Per Qwen2.5-Omni audio encoder config: 128 mel bins, 16 kHz, hop=160 (10 ms),
n_fft=400 (25 ms window). The encoder + avg_pooler downsample 4× in time, so
ONE LLM audio token corresponds to FOUR consecutive mel frames (= 40 ms).

This script aggregates ACROSS CLIPS, conditioning on sink-status at the
per-token level:

  M_sink[m, t]    = mean (over clip, mel-frames in token t) of log-mel
                    for ONLY clip-token pairs where token t is a sink.
  M_nonsink[m, t] = same for non-sink pairs.
  Diff[m, t]      = M_sink[m, t] - M_nonsink[m, t].

Three stacked heatmaps over (mel_bin × audio-token-position). The Diff panel
reveals what spectral content distinguishes sink-locations from non-sink
locations after aggregating away all the per-clip noise.

Inputs:
  - per_clip_temporal_arrays.npz from Stage 2.4 / 2.4-describe (sink masks).
  - The matching 500 (or 300) AudioSet .wav files in --audio_dir, drawn via
    the SAME seed=42 sorted-glob permutation Stage 2.4 used (verified by
    matching per-clip n_audio against npz).

Output (--output_dir, default alongside the npz):
  mel_heatmap_L<L>.png
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

# Match Qwen2.5-Omni's Whisper-style audio config.
SR = 16000
HOP = 160
N_FFT = 400
N_MELS = 128
# Encoder + avg_pooler downsample 4× → 1 LLM audio token = 4 mel frames.
MEL_PER_TOKEN = 4

DEFAULT_NPZ_DESCRIBE = (_REPO / "results/qwen2_5_omni/sink_analysis/"
                        "stage2_4_temporal_sinks_describe/"
                        "per_clip_temporal_arrays.npz")
DEFAULT_AUDIO_DIR_DESCRIBE = _REPO / "data/AudioSet_describe/audios"
DEFAULT_AUDIO_DIR_ORIG = _REPO / "data/AudioSet/audios"

AUDIO_TOK_PER_SEC = 25.0
N_AUDIO_TOL = 2


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Whisper-style log-mel spectrogram: (n_mels, n_frames)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        center=True, power=2.0)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    # Whisper's normalization: clamp the dynamic range to 8 dB below max, then
    # scale to roughly [-1, 1]. Exact constants aren't critical for relative
    # comparison; this just keeps the heatmap on a sensible scale.
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def token_mel_means(log_mel: np.ndarray, n_tokens: int) -> np.ndarray:
    """Reduce a (128, n_mel_frames) log-mel to (128, n_tokens) by averaging
    every MEL_PER_TOKEN consecutive frames. Pads/truncates as needed."""
    needed = n_tokens * MEL_PER_TOKEN
    if log_mel.shape[1] < needed:
        pad = needed - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="edge")
    elif log_mel.shape[1] > needed:
        log_mel = log_mel[:, :needed]
    return log_mel.reshape(N_MELS, n_tokens, MEL_PER_TOKEN).mean(axis=2)


def reconstruct_clip_order(audio_dir: Path, n_clips: int, seed: int = 42):
    clips_all = sorted(audio_dir.glob("*.wav"))
    if not clips_all:
        raise SystemExit(f"no .wav in {audio_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips_all))[:n_clips]
    return [clips_all[i] for i in idx]


def aggregate(npz_path: Path, audio_dir: Path, layer: int):
    """Returns (M_sink, M_nonsink, sink_counts, nonsink_counts, max_n)
    where each M is (N_MELS, max_n) of mean log-mel and counts are per-token-
    position scalar arrays."""
    import librosa  # noqa: F401  (early import for clear error)
    z = np.load(npz_path, allow_pickle=True)
    flat = z[f"L{layer}_sink_idx_flat"]
    offs = z[f"L{layer}_sink_idx_offsets"]
    n_audio_arr = z[f"L{layer}_n_audio"]
    n_clips = len(n_audio_arr)
    max_n = int(n_audio_arr.max())
    clips = reconstruct_clip_order(audio_dir, n_clips=n_clips)

    sink_sum    = np.zeros((N_MELS, max_n), dtype=np.float64)
    nonsink_sum = np.zeros((N_MELS, max_n), dtype=np.float64)
    sink_cnt    = np.zeros(max_n, dtype=np.int64)
    nonsink_cnt = np.zeros(max_n, dtype=np.int64)

    n_aligned = 0; n_misaligned = 0
    for c in tqdm(range(n_clips), desc="mel aggregate"):
        n_a = int(n_audio_arr[c])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav, _ = librosa.load(str(clips[c]), sr=SR, mono=True)
        exp_n = int(round(len(wav) / SR * AUDIO_TOK_PER_SEC))
        if abs(exp_n - n_a) > N_AUDIO_TOL:
            n_misaligned += 1
            continue
        n_aligned += 1
        log_mel = log_mel_spectrogram(wav)                  # (128, n_mel_frames)
        tok_mel = token_mel_means(log_mel, n_a)             # (128, n_a)

        # Per-token sink mask
        a, b = int(offs[c]), int(offs[c + 1])
        sink_idx = flat[a:b].astype(np.int64)
        sink_idx = sink_idx[(sink_idx >= 0) & (sink_idx < n_a)]
        is_sink = np.zeros(n_a, dtype=bool); is_sink[sink_idx] = True

        # Accumulate column-wise
        sink_sum[:, :n_a]    += tok_mel * is_sink[None, :]
        nonsink_sum[:, :n_a] += tok_mel * (~is_sink)[None, :]
        sink_cnt[:n_a]       += is_sink.astype(np.int64)
        nonsink_cnt[:n_a]    += (~is_sink).astype(np.int64)
    print(f"  aligned: {n_aligned}/{n_clips}  (skipped {n_misaligned} on "
          f"|expected - npz n_audio| > {N_AUDIO_TOL})")

    M_sink    = np.where(sink_cnt    > 0, sink_sum    / np.clip(sink_cnt,    1, None),    np.nan)
    M_nonsink = np.where(nonsink_cnt > 0, nonsink_sum / np.clip(nonsink_cnt, 1, None), np.nan)
    return M_sink, M_nonsink, sink_cnt, nonsink_cnt, max_n


def plot_mel_heatmap(M_sink, M_nonsink, sink_cnt, nonsink_cnt, max_n,
                     layer, n_clips, out_path):
    diff = M_sink - M_nonsink
    # Shared color scale for the two means, symmetric for the diff.
    vmin = float(np.nanmin([M_sink, M_nonsink]))
    vmax = float(np.nanmax([M_sink, M_nonsink]))
    dlim = float(np.nanmax(np.abs(diff)))

    fig, (ax_s, ax_n, ax_d, ax_m) = plt.subplots(
        4, 1, figsize=(13, 11),
        gridspec_kw={"height_ratios": [3, 3, 3, 1], "hspace": 0.25},
        sharex=True)

    extent = [0, max_n, 0, N_MELS]
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#222222")
    im_s = ax_s.imshow(M_sink, aspect="auto", origin="lower", cmap=cmap,
                        vmin=vmin, vmax=vmax, extent=extent,
                        interpolation="nearest")
    plt.colorbar(im_s, ax=ax_s, fraction=0.025, pad=0.01,
                  label="mean log-mel")
    ax_s.set_ylabel("mel bin (0=low f → 128=high f)", fontsize=10)
    ax_s.set_title(f"SINK-conditional mean mel-spectrogram  (L{layer})", fontsize=11)

    im_n = ax_n.imshow(M_nonsink, aspect="auto", origin="lower", cmap=cmap,
                        vmin=vmin, vmax=vmax, extent=extent,
                        interpolation="nearest")
    plt.colorbar(im_n, ax=ax_n, fraction=0.025, pad=0.01,
                  label="mean log-mel")
    ax_n.set_ylabel("mel bin", fontsize=10)
    ax_n.set_title("NON-SINK-conditional mean mel-spectrogram", fontsize=11)

    cmap_d = plt.get_cmap("RdBu_r").copy(); cmap_d.set_bad("#222222")
    im_d = ax_d.imshow(diff, aspect="auto", origin="lower", cmap=cmap_d,
                        vmin=-dlim, vmax=dlim, extent=extent,
                        interpolation="nearest")
    plt.colorbar(im_d, ax=ax_d, fraction=0.025, pad=0.01,
                  label="sink − non-sink (log-mel)")
    ax_d.set_ylabel("mel bin", fontsize=10)
    ax_d.set_title("DIFFERENCE  (red = louder at sink, blue = louder at non-sink)",
                    fontsize=11)

    # Bottom: per-token sink rate across clips
    total = sink_cnt + nonsink_cnt
    sink_rate = np.where(total > 0, sink_cnt / np.clip(total, 1, None), np.nan)
    ax_m.bar(np.arange(max_n) + 0.5, sink_rate, width=1.0, color="#1f77b4",
             edgecolor="black", linewidth=0.15, alpha=0.85)
    ax_m.set_ylabel("P(sink)\nover valid\nclips", fontsize=9)
    ax_m.set_xlabel(f"audio token position (1 token = 40 ms = 4 mel frames; "
                    f"max = {max_n})", fontsize=11)
    ax_m.set_xlim(0, max_n)
    ax_m.set_ylim(0, 1)
    ax_m.grid(True, ls=":", alpha=0.4, axis="y")

    fig.suptitle(f"Stage 2.4 — clip-aggregated mel heatmap conditional on sink "
                 f"status, layer L{layer}  (n_clips = {n_clips})",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(args):
    npz_path = Path(args.npz)
    audio_dir = Path(args.audio_dir)
    if not npz_path.is_file():
        raise SystemExit(f"not found: {npz_path}")
    out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    layers = sorted({int(k[1:].split("_", 1)[0])
                      for k in z.files
                      if k.startswith("L") and "_sink_idx_flat" in k})
    if args.layers:
        layers = [int(L) for L in args.layers if int(L) in layers]
    print(f"loaded {npz_path}; layers found: {layers}")
    print(f"audio_dir: {audio_dir}  →  reconstructing clip order via seed=42")

    n_clips_total = len(z[f"L{layers[0]}_n_audio"])
    for L in layers:
        print(f"\nlayer L{L}:")
        M_sink, M_nonsink, sink_cnt, nonsink_cnt, max_n = aggregate(
            npz_path, audio_dir, L)
        plot_mel_heatmap(M_sink, M_nonsink, sink_cnt, nonsink_cnt, max_n,
                         L, n_clips_total,
                         out_dir / f"mel_heatmap_L{L}.png")
        # Save the raw aggregates so future replots are free.
        np.savez_compressed(
            out_dir / f"mel_heatmap_L{L}.npz",
            M_sink=M_sink, M_nonsink=M_nonsink,
            sink_count=sink_cnt, nonsink_count=nonsink_cnt,
            max_n_audio=max_n)
        print(f"wrote {out_dir / f'mel_heatmap_L{L}.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ_DESCRIBE),
                   help="Stage 2.4 per_clip_temporal_arrays.npz (sink masks).")
    p.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR_DESCRIBE),
                   help="Source .wav directory (must match the one Stage 2.4 used).")
    p.add_argument("--output_dir", default="",
                   help="Where to save figures (default: alongside the npz).")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Which layers to plot (default = all in the npz).")
    args = p.parse_args()
    main(args)
