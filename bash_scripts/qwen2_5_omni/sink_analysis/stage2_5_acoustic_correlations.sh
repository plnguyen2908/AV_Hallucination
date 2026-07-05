#!/bin/bash
# Stage 2.5 — acoustic correlations on per-clip audio sink proportion @ L21.
#
# CPU-only (librosa + numpy + scipy + matplotlib). No GPUs needed; runtime is
# dominated by librosa's STFT/RMS on ~300 wav clips — minutes, not hours.
#
# Reads Stage 2.4's per_clip_temporal_stats.csv for sink_proportion @ L21 and
# reconstructs Stage 2.4's clip ordering by replaying the same seed=42
# permutation on a sorted glob of $AUDIO_DIR/*.wav. Alignment is sanity-checked
# row-by-row against the CSV's n_audio (round(duration × 25) ±2 tokens); the
# script aborts loudly if >5% of rows disagree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PY="$ROOT_DIR/qwen_venv/bin/python"

# Override if your env lives elsewhere (or just set PYTHON in the shell).
PYTHON="${PYTHON:-$VENV_PY}"
if [ ! -x "$PYTHON" ]; then
    echo "[stage2_5] python not found at $PYTHON" >&2
    echo "[stage2_5] set PYTHON=/path/to/python to override" >&2
    exit 1
fi

AUDIO_DIR="$ROOT_DIR/data/AudioSet/audios"
S24_CSV="$ROOT_DIR/results/qwen2_5_omni/sink_analysis/stage2_4_temporal_sinks/per_clip_temporal_stats.csv"
OUTPUT_DIR="$ROOT_DIR/results/qwen2_5_omni/sink_analysis/stage2_5_acoustic_correlations"
N_CLIPS=300        # must match Stage 2.4 for ordering alignment
SEED=42            # must match Stage 2.4

# Optional AudioSet metadata for the per-label box plot. Leave empty to skip.
# Drop the AudioSet *_segments.csv (header: 3 comment lines, then
# YTID,start,end,positive_labels) and the class_labels_indices.csv anywhere
# and point these vars at them.
LABELS_CSV=""
CLASS_LABELS_CSV=""

mkdir -p "$OUTPUT_DIR"

# Argument assembly — only forward the label flags when they're set, so a
# bad/empty path doesn't reach the script.
ARGS=(
    --audio_dir   "$AUDIO_DIR"
    --s24_csv     "$S24_CSV"
    --n_clips     "$N_CLIPS"
    --seed        "$SEED"
    --output_dir  "$OUTPUT_DIR"
)
if [ -n "$LABELS_CSV" ]; then
    ARGS+=(--labels_csv "$LABELS_CSV")
fi
if [ -n "$CLASS_LABELS_CSV" ]; then
    ARGS+=(--class_labels_csv "$CLASS_LABELS_CSV")
fi

echo "[stage2_5] python:     $PYTHON"
echo "[stage2_5] audio_dir:  $AUDIO_DIR"
echo "[stage2_5] s24_csv:    $S24_CSV"
echo "[stage2_5] output_dir: $OUTPUT_DIR"
echo "[stage2_5] n_clips=$N_CLIPS  seed=$SEED"
if [ -n "$LABELS_CSV" ]; then
    echo "[stage2_5] labels_csv:       $LABELS_CSV"
    echo "[stage2_5] class_labels_csv: $CLASS_LABELS_CSV"
else
    echo "[stage2_5] labels_csv not set — skipping per-label box plot"
fi
echo

exec "$PYTHON" \
    "$ROOT_DIR/method/sink_analysis/qwen2_5_omni/stage2_5_acoustic_correlations.py" \
    "${ARGS[@]}" "$@"
