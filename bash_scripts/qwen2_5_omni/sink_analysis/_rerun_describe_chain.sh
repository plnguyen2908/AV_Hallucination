#!/bin/bash
# One-shot re-run chain on the labeled AudioSet_describe subset (500 wavs).
# Waits for Stage B (video spatial) to finish, then runs 2.4 → 2.5 → 2.5b →
# Stage A in sequence. Each step is gated on the previous step's success;
# logs to one file per step under results/.../_logs/.
#
# Emits a final CHAIN_COMPLETE or CHAIN_FAILED marker line that an external
# watcher can grep for.

set -u
PY=/nobackup2/le/AV_Hallucination/qwen_venv/bin/python
ROOT=/nobackup2/le/AV_Hallucination
LOG_DIR=$ROOT/results/qwen2_5_omni/sink_analysis/_logs
TS=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=$LOG_DIR/_chain_describe_${TS}.log

OUT_24=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_4_temporal_sinks_describe
OUT_25=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5_acoustic_correlations_describe
OUT_25B=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5b_log_transforms_describe
OUT_A=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5_supp_audioset_describe

AUDIO_DIR=$ROOT/data/AudioSet_describe/audios   # 500 labeled wavs
QA_JSON=$ROOT/data/AudioSet_describe/QA.json    # multi-label per clip
WAIT_PID=${1:-}                                  # optional: PID of stage B to wait on

log()  { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }
fail() { log "CHAIN_FAILED at step: $*"; exit 1; }

log "chain started; logs → $CHAIN_LOG"
log "audio_dir=$AUDIO_DIR  qa_json=$QA_JSON  wait_pid=$WAIT_PID"

# ----- wait for stage B -----
if [ -n "$WAIT_PID" ] && ps -p "$WAIT_PID" > /dev/null 2>&1; then
    log "waiting for stage B (PID $WAIT_PID) to exit ..."
    while ps -p "$WAIT_PID" > /dev/null 2>&1; do sleep 30; done
    log "stage B exited."
    # Brief settle for filesystem.
    sleep 5
    if [ ! -f "$ROOT/results/qwen2_5_omni/sink_analysis/stage2_2_spatial/decision.txt" ]; then
        log "WARNING: stage B exited but no decision.txt — proceeding anyway."
    else
        log "stage B decision.txt present ✓"
    fi
else
    log "no live stage B to wait on; proceeding immediately."
fi

# ----- Stage 2.4 (GPU, ~5-10 min) -----
log "STEP 1/4 — Stage 2.4 (audio temporal sinks) on AudioSet_describe …"
LOG_24=$LOG_DIR/stage2_4_describe_${TS}.log
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" \
    "$ROOT/method/sink_analysis/qwen2_5_omni/stage2_4_temporal_sink_positions.py" \
    --audio_dir "$AUDIO_DIR" \
    --n_clips 500 \
    --output_dir "$OUT_24" \
    > "$LOG_24" 2>&1
rc=$?
log "  Stage 2.4 exit $rc; log → $LOG_24"
[ $rc -eq 0 ] && [ -f "$OUT_24/per_clip_temporal_stats.csv" ] || fail "stage 2.4"

# ----- Stage 2.5 (CPU, ~1 min — librosa over 500 wavs) -----
log "STEP 2/4 — Stage 2.5 (acoustic correlations) …"
LOG_25=$LOG_DIR/stage2_5_describe_${TS}.log
CUDA_VISIBLE_DEVICES="" "$PY" \
    "$ROOT/method/sink_analysis/qwen2_5_omni/stage2_5_acoustic_correlations.py" \
    --audio_dir "$AUDIO_DIR" \
    --s24_csv "$OUT_24/per_clip_temporal_stats.csv" \
    --n_clips 500 \
    --output_dir "$OUT_25" \
    > "$LOG_25" 2>&1
rc=$?
log "  Stage 2.5 exit $rc; log → $LOG_25"
[ $rc -eq 0 ] && [ -f "$OUT_25/acoustic_features_per_clip.csv" ] || fail "stage 2.5"

# ----- Stage 2.5b (CPU, ~5 s — cache-based) -----
log "STEP 3/4 — Stage 2.5b (log transforms) …"
LOG_25B=$LOG_DIR/stage2_5b_describe_${TS}.log
CUDA_VISIBLE_DEVICES="" "$PY" \
    "$ROOT/method/sink_analysis/qwen2_5_omni/stage2_5b_log_transforms.py" \
    --features_csv "$OUT_25/acoustic_features_per_clip.csv" \
    --output_dir "$OUT_25B" \
    > "$LOG_25B" 2>&1
rc=$?
log "  Stage 2.5b exit $rc; log → $LOG_25B"
[ $rc -eq 0 ] || fail "stage 2.5b"

# ----- Stage A (CPU, ~30 s — categorical with full label coverage) -----
log "STEP 4/4 — Stage A (categorical) on full label coverage (500/500) …"
LOG_A=$LOG_DIR/stage_A_describe_${TS}.log
CUDA_VISIBLE_DEVICES="" "$PY" \
    "$ROOT/method/sink_analysis/qwen2_5_omni/stage2_5_supp_audioset_categorical.py" \
    --audio_dir "$AUDIO_DIR" \
    --qa_json "$QA_JSON" \
    --s24_csv "$OUT_24/per_clip_temporal_stats.csv" \
    --n_clips 500 \
    --output_dir "$OUT_A" \
    > "$LOG_A" 2>&1
rc=$?
log "  Stage A exit $rc; log → $LOG_A"
[ $rc -eq 0 ] && [ -f "$OUT_A/decision.txt" ] || fail "stage A"

log "CHAIN_COMPLETE — all 4 steps done"
log "  Stage 2.4 → $OUT_24/decision.txt + per_clip_temporal_stats.csv"
log "  Stage 2.5 → $OUT_25/stage2_5_decision.txt + correlation_table.csv"
log "  Stage 2.5b → $OUT_25B/stage2_5b_decision.txt + log_correlation_table.csv"
log "  Stage A   → $OUT_A/decision.txt + label_root_summary.csv + label_finegrained_summary.csv"
