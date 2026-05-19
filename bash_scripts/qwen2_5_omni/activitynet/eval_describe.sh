#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="v"
VIDEO_FOLDER="$ROOT_DIR/data/ActivityNet/videos"
OUTPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/ActivityNet/QA.json"
# Flag is named --audioset_labels_file for legacy reasons; eval.py also
# accepts activitynet_labels.txt via this flag.
LABELS_FILE="$ROOT_DIR/data/ActivityNet/activitynet_labels.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/qwen2_5_omni/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE" \
    --audioset_labels_file "$LABELS_FILE" \
    --n_per_category 200 \
    --tasks "ActivityNet Captioning"
