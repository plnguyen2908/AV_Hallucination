#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
MODAL_TYPE="av"
VIDEO_FOLDER="$ROOT_DIR/data/AVHBench/videos"
INPUT_FILE="$ROOT_DIR/results/videollama2/AVHBench/sampled_entities.json"
OUTPUT_PATH="$ROOT_DIR/results/videollama2/AVHBench/attribution"
INFLUENCE_SCORE="prob_diff"   # options: prob_diff, abs_prob_diff, log_prob_diff

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/videollama2/identify_halluc_head.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --input_file "$INPUT_FILE" \
    --output_path "$OUTPUT_PATH" \
    --influence_score "$INFLUENCE_SCORE"
