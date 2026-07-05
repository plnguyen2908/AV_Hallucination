"""Round 17 queue runner — fine sweep around MAD a3v3av8 winner (DEV 79.00%).

Adds support for arbitrary extra CLI flags per config so we can mix
--mad_soft / --asd / per-modality gammas / --sink_mask in the same queue.

Strategy:
  - Best 2 DEV configs: 15b (MAD a3v3av8 wide, 79.00%) and 15d (MAD a2v4av8 wide, 79.00%).
  - Need ≥ 80.00% to promote to TEST.
  - Diversify on: tau_sink (sink count), ASD adaptive scaling combo,
    finer (a,v,av) coordinates around winners.
  - Each config 20–25 min × 8 configs on 2 slots ≈ 80–100 min.

Slot policy: slot A = 0-3, slot B = 4-7. Slot A may already be busy
(AVS baseline through ~03:30); the runner just sees the slot as busy
and uses slot B. Once A frees, configs reflow.

GPU 4-7 currently freed by YTVOS eval completion.
"""
import subprocess
import time
import os

REPO = "/nobackup2/le/AV_Hallucination"
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni",
                       "_5_explore.py")

# Slot B (4-7) only at first; once AVS baseline finishes (~03:20-03:30) slot A
# will be free for these configs. We DON'T block A — the queue just runs on B.
SLOTS = [
    dict(gpus="4,5,6,7", proc=None, tag=None, t0=None),  # available now
    # dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),  # AVS baseline busy
]

# Config schema:
#  {tag: str, variant: str, gamma: float, note: str, extras: list[str]}
QUEUE = [
    # --- Fine MAD sweep around 15b (a3v3av8) and 15d (a2v4av8) ---
    dict(tag="17a_mad_a3v3av7_wide", variant="boost_inert_content", gamma=3.0,
         note="Round 17 fine — MAD-soft a3v3av7 (one notch below 15b av=8).",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "3.0",
                  "--gamma_av", "7.0", "--sink_mask", "all"]),
    dict(tag="17b_mad_a3v3av9_wide", variant="boost_inert_content", gamma=3.0,
         note="Round 17 fine — MAD-soft a3v3av9 (one notch above 15b av=8).",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "3.0",
                  "--gamma_av", "9.0", "--sink_mask", "all"]),
    dict(tag="17c_mad_a2v5av8_wide", variant="boost_inert_content", gamma=3.0,
         note="Round 17 fine — MAD-soft asymmetric: weaker audio (2), stronger video (5), av=8. Closes 15d/15b gap on V.",
         extras=["--mad_soft", "--gamma_a", "2.0", "--gamma_v", "5.0",
                  "--gamma_av", "8.0", "--sink_mask", "all"]),
    dict(tag="17d_mad_a3v5av8_wide", variant="boost_inert_content", gamma=3.0,
         note="Round 17 fine — MAD-soft a3v5av8 (symmetric a, stronger V than 15b).",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "5.0",
                  "--gamma_av", "8.0", "--sink_mask", "all"]),

    # --- NEW DIM: MAD + ASD adaptive (cross-modal sink filter) ---
    dict(tag="17e_mad_a3v3av8_asd03",
         variant="boost_inert_content", gamma=3.0,
         note="Round 17 NEW DIM — MAD-soft + ASD adaptive (mds_thresh=0.3) on cross-modal sinks. Combines top-DEV MAD with selective scaling.",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "3.0",
                  "--gamma_av", "8.0", "--sink_mask", "all",
                  "--asd", "--asd_mds_threshold", "0.3",
                  "--asd_strength_clamp", "0.7"]),
    dict(tag="17f_mad_a2v4av8_asd02",
         variant="boost_inert_content", gamma=3.0,
         note="Round 17 NEW DIM — MAD-soft (asymm) + ASD tighter (mds=0.2). Stricter cross-modal filter on the 15d winner.",
         extras=["--mad_soft", "--gamma_a", "2.0", "--gamma_v", "4.0",
                  "--gamma_av", "8.0", "--sink_mask", "all",
                  "--asd", "--asd_mds_threshold", "0.2",
                  "--asd_strength_clamp", "0.7"]),

    # --- NEW DIM: tau_sink variation (wider/narrower sink set) ---
    dict(tag="17g_mad_a3v3av8_tau15",
         variant="boost_inert_content", gamma=3.0,
         note="Round 17 NEW DIM — MAD-soft a3v3av8 with tau_sink=15 (wider sink set, more positions boosted). 13a-c sweep showed tau effect.",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "3.0",
                  "--gamma_av", "8.0", "--sink_mask", "all",
                  "--tau_sink", "15"]),
    dict(tag="17h_mad_a3v3av8_tau25",
         variant="boost_inert_content", gamma=3.0,
         note="Round 17 NEW DIM — MAD-soft a3v3av8 with tau_sink=25 (narrower, only strongest sinks).",
         extras=["--mad_soft", "--gamma_a", "3.0", "--gamma_v", "3.0",
                  "--gamma_av", "8.0", "--sink_mask", "all",
                  "--tau_sink", "25"]),
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
