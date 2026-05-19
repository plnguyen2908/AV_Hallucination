#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="av"   # VGGSounder is audio+visual
VIDEO_FOLDER="$ROOT_DIR/data/VGGSounder/videos"
OUTPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/VGGSounder/QA.json"
LABELS_FILE="$ROOT_DIR/data/VGGSounder/vggsounder_labels.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/qwen2_5_omni/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE" \
    --audioset_labels_file "$LABELS_FILE" \
    --n_per_category 200 \
    --tasks "VGGSounder Captioning"
