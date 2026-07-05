"""YT-VOS x AudioSet head taxonomy — SMALL-gamma exploration (AVHBench DEV).

Follow-up to _5_queue_ytvos.py. That sweep (γ in 2-12) was flat 74.3-76.7%,
best ytx_a3v3av5 = 76.67%, with the clear trend "lower av helps". This probes
SMALLER per-category γ across the board.

Two 4-GPU slots, AVHBench DEV (n=300), heads pinned to
categorize_exp_2axis_youtubevos/heads.csv.
"""
import subprocess
import time
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni",
                       "_5_explore.py")
HEADS_CSV = os.path.join(
    REPO, "results", "qwen2_5_omni",
    "categorize_exp_2axis_youtubevos", "heads.csv")

SLOTS = [
    dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
    dict(gpus="4,5,6,7", proc=None, tag=None, t0=None),
]

def mad(a, v, av):
    return ["--mad_soft", "--gamma_a", str(a), "--gamma_v", str(v),
            "--gamma_av", str(av), "--sink_mask", "all"]

QUEUE = [
    # uniform small
    dict(tag="ytx_a1v1av1", variant="boost_inert_content", gamma=1.0,
         note="YT-VOS heads small-gamma: a1v1av1 (uniform tiny boost).",
         extras=mad(1.0, 1.0, 1.0)),
    dict(tag="ytx_a2v2av2", variant="boost_inert_content", gamma=2.0,
         note="YT-VOS heads small-gamma: a2v2av2.",
         extras=mad(2.0, 2.0, 2.0)),
    dict(tag="ytx_a3v3av3", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads small-gamma: a3v3av3 (av down from best a3v3av5).",
         extras=mad(3.0, 3.0, 3.0)),
    # av pulled to/below 5 with low a,v
    dict(tag="ytx_a2v2av3", variant="boost_inert_content", gamma=2.0,
         note="YT-VOS heads small-gamma: a2v2av3.",
         extras=mad(2.0, 2.0, 3.0)),
    dict(tag="ytx_a2v2av5", variant="boost_inert_content", gamma=2.0,
         note="YT-VOS heads small-gamma: a2v2av5 (low a/v, best-region av).",
         extras=mad(2.0, 2.0, 5.0)),
    dict(tag="ytx_a1v1av3", variant="boost_inert_content", gamma=1.0,
         note="YT-VOS heads small-gamma: a1v1av3.",
         extras=mad(1.0, 1.0, 3.0)),
    # slight visual lean at small scale
    dict(tag="ytx_a2v3av3", variant="boost_inert_content", gamma=2.0,
         note="YT-VOS heads small-gamma: a2v3av3 (mild visual lean).",
         extras=mad(2.0, 3.0, 3.0)),
    dict(tag="ytx_a3v3av4", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads small-gamma: a3v3av4 (between best a3v3av5 and a3v3av3).",
         extras=mad(3.0, 3.0, 4.0)),
]


def launch(slot, cfg):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    log_path = f"/tmp/explore_{cfg['tag']}.log"
    log_file = open(log_path, "w")
    cmd = [
        PY, SCRIPT,
        "--variant", cfg["variant"],
        "--gamma", str(cfg["gamma"]),
        "--tag", cfg["tag"],
        "--split", "DEV",
        "--device_map", "balanced_low_0",
        "--heads_csv", HEADS_CSV,
        "--note", cfg["note"],
    ] + cfg.get("extras", [])
    print(f"[{cfg['tag']}] LAUNCH on GPUs {slot['gpus']}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    slot["proc"] = proc
    slot["tag"] = cfg["tag"]
    slot["t0"] = time.time()
    slot["log_file"] = log_file


def poll(slot):
    if slot["proc"] is None:
        return False
    rc = slot["proc"].poll()
    if rc is None:
        return True
    dt = time.time() - slot["t0"]
    print(f"[{slot['tag']}] DONE rc={rc} ({dt/60:.1f} min)", flush=True)
    slot["log_file"].close()
    slot["proc"] = None
    slot["tag"] = None
    slot["t0"] = None
    return False


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} configs on slots: "
          f"{[s['gpus'] for s in SLOTS]}", flush=True)
    print(f"HEADS_CSV = {HEADS_CSV}", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q:
                cfg = q.pop(0)
                launch(slot, cfg)
        time.sleep(10)
        for slot in SLOTS:
            poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
