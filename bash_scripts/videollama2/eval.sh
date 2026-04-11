#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
MODAL_TYPE="av"
N_PER_CATEGORY=100
VIDEO_FOLDER="$ROOT_DIR/data/AVHBench/videos"
OUTPUT_FILE="$ROOT_DIR/results/videollama2/AVHBench/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/AVHBench/QA.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/videollama2/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --n_per_category "$N_PER_CATEGORY" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE"
