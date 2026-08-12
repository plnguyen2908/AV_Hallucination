#!/bin/bash
# Auto-generated Qwen3-Omni Stage-2 attribution wrapper (data-parallel).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
PY="${PY:-/nobackup2/le/AV_Hallucination/qwen3_venv/bin/python}"
GPU_GROUPS="${GPU_GROUPS:-0,1,2,3;4,5,6,7}"
"$PY" "$ROOT_DIR/method/qwen3_omni/run_attribution_parallel.py" \
    --input_file "$ROOT_DIR/results/qwen3_omni/AudioSet_describe/sampled_entities.json" \
    --video_folder "$ROOT_DIR/data/AudioSet/audios" \
    --model_path "$MODEL_PATH" --modal_type a \
    --output_path "$ROOT_DIR/results/qwen3_omni/AudioSet_describe/attribution" \
    --gpu_groups "$GPU_GROUPS" 
