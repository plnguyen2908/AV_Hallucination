#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="a"
VIDEO_FOLDER="$ROOT_DIR/data/AudioSet/audios"
INPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/AudioSet/sampled_entities.json"
OUTPUT_PATH="$ROOT_DIR/results/qwen2_5_omni/AudioSet/attribution"
INFLUENCE_SCORE="prob_diff" # options: prob_diff, abs_prob_diff, log_prob_diff

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/qwen2_5_omni/identify_halluc_head.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --input_file "$INPUT_FILE" \
    --output_path "$OUTPUT_PATH" \
    --influence_score "$INFLUENCE_SCORE" \
    --topk 40
# --skip
