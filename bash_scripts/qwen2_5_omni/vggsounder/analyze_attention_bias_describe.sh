#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
MODAL_TYPE="av"
VIDEO_FOLDER="$ROOT_DIR/data/VGGSounder/videos"
INPUT_FILE="$ROOT_DIR/results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json"
ATTENTION_HEAD_PATH="$ROOT_DIR/results/qwen2_5_omni/VGGSounder_describe/attribution/heads/attribution_result.json"
OUTPUT_PATH="$ROOT_DIR/results/qwen2_5_omni/VGGSounder_describe/attention_bias"
TOP_K=20

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/qwen2_5_omni/analyze_attention_bias.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --input_file "$INPUT_FILE" \
    --attention_head_path "$ATTENTION_HEAD_PATH" \
    --output_path "$OUTPUT_PATH" \
    --top_k "$TOP_K"
