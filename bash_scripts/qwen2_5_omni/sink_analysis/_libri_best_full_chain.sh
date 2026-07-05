#!/usr/bin/env bash
# Generic AVHBench-FULL -> AVS chain for a LibriSpeech-axis taxonomy.
# Args via env: GPUS, HEADS, GA, GV, GAV, TAG
#   GPUS  e.g. "0,1,2,3"
#   HEADS path to heads.csv
#   GA/GV/GAV  per-modality gammas
#   TAG   short tag prefix (e.g. libxan_a3v3av3)
set -euo pipefail
cd /nobackup2/le/AV_Hallucination
export CUDA_VISIBLE_DEVICES="${GPUS}"
PY=qwen_venv/bin/python

echo "===== [1/2] AVHBench FULL — ${TAG} (heads=${HEADS}) ====="
$PY method/sink_analysis/qwen2_5_omni/_5_explore.py \
  --variant boost_inert_content --gamma 3.0 \
  --mad_soft --gamma_a "${GA}" --gamma_v "${GV}" --gamma_av "${GAV}" --sink_mask all \
  --heads_csv "${HEADS}" \
  --split FULL \
  --tag "${TAG}_FULL" \
  --device_map balanced_low_0 \
  --note "Best LibriSpeech-axis config ${TAG} on AVHBench FULL (5302)"

echo "===== [2/2] AV-SpeakerBench — ${TAG} ====="
$PY method/sink_analysis/qwen2_5_omni/_5_avspeaker_eval.py \
  --variant boost_inert_content --gamma 3.0 \
  --gamma_a "${GA}" --gamma_v "${GV}" --gamma_av "${GAV}" --sink_mask all \
  --heads_csv "${HEADS}" \
  --tag "${TAG}_avs" \
  --device_map balanced_low_0 \
  --video_fps 1.0 --video_resized_hw 280 --max_duration_s 15 \
  --note "Best LibriSpeech-axis config ${TAG} on AV-SpeakerBench <=15s fps1 280"

echo "===== CHAIN DONE ${TAG} ====="
