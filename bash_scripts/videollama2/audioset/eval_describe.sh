#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
MODAL_TYPE="a"
# WAVs live in the original AudioSet location; only QA.json / labels differ per variant.
VIDEO_FOLDER="$ROOT_DIR/data/AudioSet/audios"
OUTPUT_FILE="$ROOT_DIR/results/videollama2/AudioSet_describe/sampled_entities.json"
QA_FILE="$ROOT_DIR/data/AudioSet_describe/QA.json"
AUDIOSET_LABELS_FILE="$ROOT_DIR/data/AudioSet_describe/audioset_labels.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$ROOT_DIR/method/videollama2/eval.py" \
    --model_path "$MODEL_PATH" \
    --modal_type "$MODAL_TYPE" \
    --video_folder "$VIDEO_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --QA_FILE "$QA_FILE" \
    --audioset_labels_file "$AUDIOSET_LABELS_FILE" \
    --n_per_category 500 \
    --tasks "AudioSet Captioning"
