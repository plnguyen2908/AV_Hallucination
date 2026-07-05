"""YT-VOS x AudioSet head-taxonomy exploration queue (AVHBench DEV).

Mirrors _5_queue_round17.py but forces --heads_csv to the
categorize_exp_2axis_youtubevos/heads.csv taxonomy (Audio 40 / Visual 89 /
AV 7 / Inert 648) instead of the default ActivityNet x AudioSet set.

Anchor already on disk: ytvos_mad_a3v3av8_wide = DEV 75.67%
(ActivityNet-heads same config = 79.00%).

Two 4-GPU slots. Each item = MAD-soft per-category gamma (a/v/av) sweep,
plus a couple of variant-diversifying configs. All on AVHBench DEV (n=300).
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

# MAD-soft per-category gamma helper.
def mad(a, v, av):
    return ["--mad_soft", "--gamma_a", str(a), "--gamma_v", str(v),
            "--gamma_av", str(av), "--sink_mask", "all"]

QUEUE = [
    # --- av sweep around the a3v3av8 anchor (75.67%) ---
    dict(tag="ytx_a3v3av5", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a3v3av5 (lower av than anchor).",
         extras=mad(3.0, 3.0, 5.0)),
    dict(tag="ytx_a3v3av12", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a3v3av12 (higher av; AV set only 7 heads).",
         extras=mad(3.0, 3.0, 12.0)),
    # --- stronger visual: YT-VOS Visual set is the bulk (89 heads) ---
    dict(tag="ytx_a3v5av8", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a3v5av8 (stronger visual routing).",
         extras=mad(3.0, 5.0, 8.0)),
    dict(tag="ytx_a3v8av8", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a3v8av8 (much stronger visual).",
         extras=mad(3.0, 8.0, 8.0)),
    # --- audio lever ---
    dict(tag="ytx_a5v3av8", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a5v3av8 (stronger audio).",
         extras=mad(5.0, 3.0, 8.0)),
    dict(tag="ytx_a2v4av8", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a2v4av8 (15d-style asymmetric).",
         extras=mad(2.0, 4.0, 8.0)),
    # --- balanced moderate ---
    dict(tag="ytx_a3v5av5", variant="boost_inert_content", gamma=3.0,
         note="YT-VOS heads: MAD-soft a3v5av5 (moderate, visual-leaning).",
         extras=mad(3.0, 5.0, 5.0)),
    # --- variant diversification: boost the union of halluc heads ---
    dict(tag="ytx_allhalluc_g5", variant="boost_all_halluc_content", gamma=5.0,
         note="YT-VOS heads: boost union of all halluc heads (40+89+7=136), gamma=5, ignores routing.",
         extras=["--sink_mask", "all"]),
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
