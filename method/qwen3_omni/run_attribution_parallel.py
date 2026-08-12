"""Data-parallel driver for Stage-2 per-head zero-ablation (Qwen3-Omni).

The 30B MoE needs ~4x24GB GPUs per replica, so on an 8-GPU box we run 2
replicas. This orchestrator:

  1. splits sampled_entities.json into one shard per GPU group (round-robin),
  2. launches identify_halluc_head.py --no_score for each shard with its own
     CUDA_VISIBLE_DEVICES and output_path (so the per-shard shutil.rmtree +
     pth writes never collide),
  3. waits, then merges every shard's pth/*.pth into <output_path>/pth and runs
     contrastive_score ONCE on the merged set (mean/contrastive heatmaps +
     heads/attribution_result.json).

Usage:
  qwen3_venv/bin/python method/qwen3_omni/run_attribution_parallel.py \
      --input_file results/qwen3_omni/AudioSet_describe/sampled_entities.json \
      --video_folder data/AudioSet/audios --modal_type a \
      --output_path results/qwen3_omni/AudioSet_describe/attribution \
      --gpu_groups "0,1,2,3;4,5,6,7"
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# identify_halluc_head lives beside this file; reuse its helpers for the merge.
sys.path.insert(0, str(_HERE))
import identify_halluc_head as ihh  # noqa: E402


def split_shards(input_file, n, tmp_dir):
    with open(input_file) as f:
        samples = json.load(f)
    shards = [[] for _ in range(n)]
    for i, s in enumerate(samples):
        shards[i % n].append(s)
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for k, shard in enumerate(shards):
        p = os.path.join(tmp_dir, f"shard_{k}.json")
        with open(p, "w") as f:
            json.dump(shard, f)
        paths.append(p)
    return paths, [len(s) for s in shards]


def main(a):
    groups = [g.strip() for g in a.gpu_groups.split(";") if g.strip()]
    n = len(groups)
    print(f"[parallel] {n} replica(s), GPU groups: {groups}", flush=True)

    shard_root = os.path.join(a.output_path, "_shards")
    shutil.rmtree(shard_root, ignore_errors=True)
    shard_paths, counts = split_shards(a.input_file, n, shard_root)
    print(f"[parallel] shard sizes: {counts}", flush=True)

    procs = []
    for k, (grp, sp) in enumerate(zip(groups, shard_paths)):
        out_k = os.path.join(shard_root, f"out_{k}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=grp)
        cmd = [
            sys.executable, str(_HERE / "identify_halluc_head.py"),
            "--input_file", sp,
            "--video_folder", a.video_folder,
            "--model_path", a.model_path,
            "--modal_type", a.modal_type,
            "--output_path", out_k,
            "--influence_score", a.influence_score,
            "--device_map", a.device_map,
            "--no_score",
        ]
        if a.max_new_tokens is not None:
            cmd += ["--max_new_tokens", str(a.max_new_tokens)]
        log = open(os.path.join(shard_root, f"shard_{k}.log"), "w")
        print(f"[parallel] launch shard {k} on GPUs {grp} -> {out_k}", flush=True)
        procs.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    failed = []
    for k, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"[parallel] shard {k} exited rc={rc}", flush=True)
        if rc != 0:
            failed.append(k)
    if failed:
        print(f"[parallel] WARNING shards failed: {failed} "
              f"(see {shard_root}/shard_*.log); merging survivors.", flush=True)

    # --- merge pth into final output_path/pth, then score once ---
    for sub in ("pth", "images", "heads"):
        os.makedirs(os.path.join(a.output_path, sub), exist_ok=True)
    merged_pth = os.path.join(a.output_path, "pth")
    n_copied = 0
    for k, _, _ in procs:
        src = os.path.join(shard_root, f"out_{k}", "pth")
        if not os.path.isdir(src):
            continue
        for fn in os.listdir(src):
            if fn.endswith(".pth"):
                shutil.copy2(os.path.join(src, fn), os.path.join(merged_pth, fn))
                n_copied += 1
    print(f"[parallel] merged {n_copied} pth files into {merged_pth}", flush=True)
    if n_copied == 0:
        print("[parallel] ERROR no pth produced; aborting scoring.", flush=True)
        sys.exit(1)

    layer_num, head_num = ihh._dims_from_pth(merged_pth)
    print(f"[parallel] grid = {layer_num} layers x {head_num} heads", flush=True)
    score_args = argparse.Namespace(output_path=a.output_path, topk=a.topk)
    ihh.contrastive_score(score_args, layer_num, head_num)
    print("[parallel] DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--video_folder", required=True)
    p.add_argument("--model_path", default="/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695")
    p.add_argument("--modal_type", default="a", choices=["a", "v", "av"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--influence_score", default="prob_diff")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--gpu_groups", default="0,1,2,3;4,5,6,7",
                   help="Semicolon-separated CUDA_VISIBLE_DEVICES groups, one "
                        "per replica, e.g. '0,1,2,3;4,5,6,7'.")
    p.add_argument("--device_map", default="balanced",
                   help="HF device_map per replica. Default 'balanced' (even "
                        "split, no GPU0 reservation) suits a 4-GPU replica.")
    main(p.parse_args())
