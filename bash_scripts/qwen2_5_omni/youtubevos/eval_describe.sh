#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="v"
VIDEO_FOLDER="$ROOT_DIR/data/YouTubeVOS/videos"
OUTPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/YouTubeVOS_describe/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/YouTubeVOS/QA.json"
LABELS_FILE="$ROOT_DIR/data/YouTubeVOS/youtubevos_labels.txt"

PY="${PY:-/nobackup2/le/AV_Hallucination/qwen_venv/bin/python}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} "$PY" "$ROOT_DIR/method/qwen2_5_omni/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE" \
    --audioset_labels_file "$LABELS_FILE" \
    --n_per_category 300 \
    --tasks "YouTubeVOS Captioning"
