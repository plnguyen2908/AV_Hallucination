#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
MODAL_TYPE="a"
VIDEO_FOLDER="$ROOT_DIR/data/AudioSet/audios"
INPUT_FILE="$ROOT_DIR/results/videollama2/AudioSet/sampled_entities.json"
ATTENTION_HEAD_PATH="$ROOT_DIR/results/videollama2/AudioSet/attribution/heads/attribution_result.json"
OUTPUT_PATH="$ROOT_DIR/results/videollama2/AudioSet/attention_bias"
TOP_K=20

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/videollama2/analyze_attention_bias.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --input_file "$INPUT_FILE" \
    --attention_head_path "$ATTENTION_HEAD_PATH" \
    --output_path "$OUTPUT_PATH" \
    --top_k "$TOP_K"