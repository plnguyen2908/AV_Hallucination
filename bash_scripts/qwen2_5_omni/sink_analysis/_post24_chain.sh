#!/bin/bash
# Waits for Stage 2.4-describe (given PID), then runs 2.5 → 2.5b → A on CPU.
# Each step gated on the previous step's success.
# Emits CHAIN_COMPLETE or CHAIN_FAILED to its own log.

set -u
PY=/nobackup2/le/AV_Hallucination/qwen_venv/bin/python
ROOT=/nobackup2/le/AV_Hallucination
LOG_DIR=$ROOT/results/qwen2_5_omni/sink_analysis/_logs
TS=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=$LOG_DIR/_chain_post24_${TS}.log

OUT_24=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_4_temporal_sinks_describe
OUT_25=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5_acoustic_correlations_describe
OUT_25B=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5b_log_transforms_describe
OUT_A=$ROOT/results/qwen2_5_omni/sink_analysis/stage2_5_supp_audioset_describe

AUDIO_DIR=$ROOT/data/AudioSet_describe/audios
QA_JSON=$ROOT/data/AudioSet_describe/QA.json
WAIT_PID=${1:?usage: $0 <stage24_describe_pid>}

log()  { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }
fail() { log "CHAIN_FAILED at step: $*"; exit 1; }

log "post-24 chain started; logs → $CHAIN_LOG"
log "waiting for stage 2.4-describe (PID $WAIT_PID) to exit ..."
while ps -p "$WAIT_PID" > /dev/null 2>&1; do sleep 30; done
log "stage 2.4-describe exited."
sleep 5
[ -f "$OUT_24/per_clip_temporal_stats.csv" ] || fail "stage 2.4-describe (no per_clip_temporal_stats.csv)"

log "STEP 1/3 — Stage 2.5 (acoustic correlations) on 500 wavs …"
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

log "STEP 2/3 — Stage 2.5b (log transforms) …"
LOG_25B=$LOG_DIR/stage2_5b_describe_${TS}.log
CUDA_VISIBLE_DEVICES="" "$PY" \
    "$ROOT/method/sink_analysis/qwen2_5_omni/stage2_5b_log_transforms.py" \
    --features_csv "$OUT_25/acoustic_features_per_clip.csv" \
    --output_dir "$OUT_25B" \
    > "$LOG_25B" 2>&1
rc=$?
log "  Stage 2.5b exit $rc; log → $LOG_25B"
[ $rc -eq 0 ] || fail "stage 2.5b"

log "STEP 3/3 — Stage A (categorical) with full 500/500 label coverage …"
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

log "CHAIN_COMPLETE — all 3 steps done"
log "  Stage 2.5  → $OUT_25/"
log "  Stage 2.5b → $OUT_25B/"
log "  Stage A    → $OUT_A/decision.txt"
