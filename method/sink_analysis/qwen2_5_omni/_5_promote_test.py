"""Promote a DEV-winning config to a TEST run (full 5302 entries).

Parses _5_queue.py to find the config tuple matching a tag, then launches
_5_explore.py with --split FULL.

Usage:
    python _5_promote_test.py --tag 06g_boost_inert_g20

Outputs:
    interv_<tag>_FULL.csv  (in stage5_intervention/)
    Auto-appended row to method_log.md (the explore script does this).
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
_PY = str(_REPO / "qwen_venv/bin/python")
_SCRIPT = str(_HERE / "_5_explore.py")


def find_config(tag):
    """Import _5_queue and search QUEUE for the matching tag."""
    sys.path.insert(0, str(_HERE))
    import importlib
    import _5_queue
    importlib.reload(_5_queue)
    for item in _5_queue.QUEUE:
        if len(item) == 5:
            v, g, t, n, lb = item
        else:
            v, g, t, n = item
            lb = "full"
        if t == tag:
            return dict(variant=v, gamma=g, tag=t, note=n, layer_band=lb)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True,
                   help="Tag of the DEV config to promote (must exist in _5_queue.QUEUE).")
    p.add_argument("--split", default="FULL",
                   choices=["DEV", "HELDOUT", "TEST", "FULL"],
                   help="Which split to evaluate on (default: FULL = all 5302).")
    p.add_argument("--gpus", default="0,1,2,3",
                   help="CUDA_VISIBLE_DEVICES (default: 0,1,2,3).")
    args = p.parse_args()

    cfg = find_config(args.tag)
    if cfg is None:
        print(f"Config '{args.tag}' not found in _5_queue.QUEUE")
        sys.exit(1)

    print(f"Promoting {args.tag} to --split {args.split}:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    cmd = [
        _PY, _SCRIPT,
        "--variant", cfg["variant"],
        "--gamma", str(cfg["gamma"]),
        "--tag", cfg["tag"],
        "--split", args.split,
        "--device_map", "balanced_low_0",
        "--layer_band", cfg["layer_band"],
        "--note", f"TEST PROMOTION of DEV winner. {cfg['note']}",
    ]
    log_path = f"/tmp/explore_{args.tag}_{args.split}.log"
    with open(log_path, "w") as f:
        print(f"Launching: {' '.join(cmd)}")
        print(f"Log: {log_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"PID: {proc.pid}")


if __name__ == "__main__":
    main()
