#!/usr/bin/env bash
# GPU 0-3 sequential: run the best YT-VOS x AudioSet config (a3v3av5, DEV 76.67%)
# the same way the promoted configs ran — on AVHBench FULL (5302) — then on
# AV-SpeakerBench (AVS). Both use the youtubevos head taxonomy.
set -euo pipefail
cd /nobackup2/le/AV_Hallucination
export CUDA_VISIBLE_DEVICES=0,1,2,3
PY=qwen_venv/bin/python
HEADS=results/qwen2_5_omni/categorize_exp_2axis_youtubevos/heads.csv

echo "===== [1/2] AVHBench FULL — ytvos a3v3av5 ====="
$PY method/sink_analysis/qwen2_5_omni/_5_explore.py \
  --variant boost_inert_content --gamma 3.0 \
  --mad_soft --gamma_a 3.0 --gamma_v 3.0 --gamma_av 5.0 --sink_mask all \
  --heads_csv "$HEADS" \
  --split FULL \
  --tag ytx_a3v3av5_FULL \
  --device_map balanced_low_0 \
  --note "Best YT-VOS x AudioSet config (a3v3av5, DEV 76.67%) on AVHBench FULL (5302)"

echo "===== [2/2] AV-SpeakerBench — ytvos a3v3av5 ====="
$PY method/sink_analysis/qwen2_5_omni/_5_avspeaker_eval.py \
  --variant boost_inert_content --gamma 3.0 \
  --gamma_a 3.0 --gamma_v 3.0 --gamma_av 5.0 --sink_mask all \
  --heads_csv "$HEADS" \
  --tag ytx_a3v3av5_avs \
  --device_map balanced_low_0 \
  --video_fps 1.0 --video_resized_hw 280 --max_duration_s 15 \
  --note "Best YT-VOS x AudioSet config (a3v3av5) on AV-SpeakerBench <=15s, fps1 280"

echo "===== CHAIN DONE ====="
