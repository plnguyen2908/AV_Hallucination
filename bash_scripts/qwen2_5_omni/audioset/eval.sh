#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="a"
VIDEO_FOLDER="$ROOT_DIR/data/AudioSet/audios"
OUTPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/AudioSet/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/AudioSet/QA.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/qwen2_5_omni/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE" \
    --n_per_category 200 \
    --tasks "Video-driven Audio Hallucination"
