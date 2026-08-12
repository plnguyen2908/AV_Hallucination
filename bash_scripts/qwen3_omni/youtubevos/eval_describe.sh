#!/bin/bash
# Auto-generated Qwen3-Omni Stage-1 eval wrapper (port of qwen2_5_omni).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
PY="${PY:-/nobackup2/le/AV_Hallucination/qwen3_venv/bin/python}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} "$PY" "$ROOT_DIR/method/qwen3_omni/eval.py" \
    --model_path "$MODEL_PATH" --modal_type v \
    --video_folder "$ROOT_DIR/data/YouTubeVOS/videos" \
    --output_file "$ROOT_DIR/results/qwen3_omni/YouTubeVOS_describe/sampled_entities.json" \
    --QA_FILE "$ROOT_DIR/data/YouTubeVOS/QA.json" \
    --audioset_labels_file "$ROOT_DIR/data/YouTubeVOS/youtubevos_labels.txt" \
    --n_per_category "${N:-300}" --tasks "YouTubeVOS Captioning"
