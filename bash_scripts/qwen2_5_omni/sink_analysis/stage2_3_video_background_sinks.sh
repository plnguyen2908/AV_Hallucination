#!/bin/bash
# Stage 2.3 — video L2 sink ↔ SAM 3 background association.
#
# Runs in the sam3btr conda env (Python 3.12 + torch 2.7 + CUDA 12.6 + sam3).
# qwen_venv torch 2.6 doesn't satisfy sam3's requirements.
#
# Defaults to a full run over all 299 clips in the Stage 2.2 npz; pass
# --dry-run to smoke-test the plumbing without frames or SAM (synthetic
# top=background bias), or --limit N to subsample for speed.

set -u
ROOT=/nobackup2/le/AV_Hallucination
PY=${PYTHON:-/nobackup2/jaden/miniconda3/envs/sam3btr/bin/python}
SCRIPT=$ROOT/method/sink_analysis/qwen2_5_omni/stage2_3_video_background_sinks.py
LOG_DIR=$ROOT/results/qwen2_5_omni/sink_analysis/_logs
mkdir -p $LOG_DIR

if [ ! -x "$PY" ]; then
    echo "[stage2_3] python not found at $PY  (set PYTHON=/path/to/python to override)" >&2
    exit 1
fi

# SAM 3 expects to import from its own repo root (relative paths in
# pkg_resources.resource_filename etc.)
export PYTHONPATH="/nobackup2/jaden/sam3btr${PYTHONPATH:+:$PYTHONPATH}"
# avoid HF_TRANSFER spew if it's pre-set
unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

echo "[stage2_3] python:     $PY"
echo "[stage2_3] script:     $SCRIPT"
echo "[stage2_3] args:       $@"
echo

exec "$PY" "$SCRIPT" "$@"
