"""Continuous queue runner for Stage 5 exploration.

Two slots (each 4 GPUs). Pops configs from QUEUE and launches them; as
soon as a slot's child exits, the next pending config starts. Doesn't
wait for the batch to finish; minimizes idle time.

Each item in QUEUE = (variant, gamma, tag, note).
"""
import subprocess
import time
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni",
                       "_5_explore.py")

SLOTS = [
    dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
    dict(gpus="4,5,6,7", proc=None, tag=None, t0=None),
]

QUEUE = [
    # Round 3 — extend γ + new diversification dimensions
    # 06j/k/l already done; skip those
    # New head-set dimension: union of all halluc categories (ignore router)
    ("boost_all_halluc_content", 2.0, "07a_boost_all_halluc_g20",
     "Round 3 NEW — boost x3.0 on union of all 130 halluc heads (107A+17V+6AV) ignoring router. Tests if halluc-head boost compounds across modalities."),
    ("boost_all_halluc_content", 5.0, "07b_boost_all_halluc_g50",
     "Round 3 NEW — boost x6.0 on union of all 130 halluc heads."),
    # Broadest head-set: ALL 784 heads
    ("boost_all_heads_content", 2.0, "07c_boost_all_heads_g20",
     "Round 3 NEW — boost x3.0 on ALL 784 heads. Maximum broad intervention; tests if all-head content-routing wins."),
    ("boost_all_heads_content", 5.0, "07d_boost_all_heads_g50",
     "Round 3 NEW — boost x6.0 on ALL 784 heads. Extreme broad intervention."),

    # Round 4 — value-zero variant (different mechanism)
    ("value_zero_inert_sink", 1.0, "08a_vzero_inert",
     "Round 4 — VALUE ZERO at sink positions for INERT heads (γ unused). Cleaner mechanism than suppress: leaves attn distribution untouched but zeroes sink V contribution to output."),
    ("value_zero_all_heads_sink", 1.0, "08b_vzero_all_heads",
     "Round 4 — VALUE ZERO at sinks for ALL 784 heads."),
    # Round 4 — layer-band restriction of best Round 2 winner (06g γ=2.0 boost_inert)
    # 5-tuple includes layer_band
    ("boost_inert_content", 2.0, "09a_boost_inert_g20_early", "Round 4 — γ=2.0 boost_inert restricted to EARLY layers (L0-9, 226 heads).", "early"),
    ("boost_inert_content", 2.0, "09b_boost_inert_g20_mid",   "Round 4 — γ=2.0 boost_inert restricted to MID layers (L10-18, 223 heads).",   "mid"),
    ("boost_inert_content", 2.0, "09c_boost_inert_g20_late",  "Round 4 — γ=2.0 boost_inert restricted to LATE layers (L19-27, 205 heads).", "late"),
]


def launch(slot, variant, gamma, tag, note, layer_band="full"):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    log_path = f"/tmp/explore_{tag}.log"
    log_file = open(log_path, "w")
    cmd = [
        PY, SCRIPT,
        "--variant", variant,
        "--gamma", str(gamma),
        "--tag", tag,
        "--split", "DEV",
        "--device_map", "balanced_low_0",
        "--layer_band", layer_band,
        "--note", note,
    ]
    print(f"[{tag}] LAUNCH on GPUs {slot['gpus']}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    slot["proc"] = proc
    slot["tag"] = tag
    slot["t0"] = time.time()
    slot["log_file"] = log_file


def poll(slot):
    if slot["proc"] is None:
        return False  # idle
    rc = slot["proc"].poll()
    if rc is None:
        return True  # running
    # done
    dt = time.time() - slot["t0"]
    print(f"[{slot['tag']}] DONE rc={rc} ({dt/60:.1f} min)", flush=True)
    slot["log_file"].close()
    slot["proc"] = None
    slot["tag"] = None
    slot["t0"] = None
    return False


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        # Launch on free slots
        for slot in SLOTS:
            if slot["proc"] is None and q:
                item = q.pop(0)
                # Backwards-compatible: 4-tuple (variant, γ, tag, note) or
                # 5-tuple with layer_band.
                if len(item) == 5:
                    v, g, t, n, lb = item
                else:
                    v, g, t, n = item
                    lb = "full"
                launch(slot, v, g, t, n, lb)
        # Wait briefly
        time.sleep(10)
        # Poll
        for slot in SLOTS:
            poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
