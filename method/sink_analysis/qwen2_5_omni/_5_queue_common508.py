"""Sweep for the 508 always-inert heads (Inert in all 4 audio×visual
taxonomies) as the boosted set. boost_inert_content + MAD-soft per-category γ,
AVHBench DEV (n=300). Goal: find a config on par with 79% (the AudioSet×
ActivityNet DEV best). Two 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")
HEADS = os.path.join(REPO, "results", "qwen2_5_omni",
                     "categorize_exp_2axis_common508", "heads.csv")

GRID = [  # (a, v, av) — ROUND 2: higher uniform gamma + balanced low-av
    (6,6,6), (7,7,7), (8,8,8), (10,10,10), (6,6,3), (6,6,4),
    (5,5,3), (5,5,4), (7,7,5), (8,8,5), (6,4,5), (4,6,5),
    (6,6,8), (8,8,4), (5,4,4), (4,5,5),
]

QUEUE = [dict(tag=f"c508_a{a}v{v}av{av}", a=a, v=v, av=av) for a,v,av in GRID]

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, cfg):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"/tmp/explore_{cfg['tag']}.log", "w")
    cmd = [PY, SCRIPT, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--mad_soft", "--gamma_a", str(float(cfg["a"])), "--gamma_v", str(float(cfg["v"])),
           "--gamma_av", str(float(cfg["av"])), "--sink_mask", "all",
           "--heads_csv", HEADS, "--split", "DEV", "--device_map", "balanced_low_0",
           "--tag", cfg["tag"],
           "--note", f"508 always-inert core boosted, MAD-soft {cfg['tag']}"]
    print(f"[{cfg['tag']}] LAUNCH on {slot['gpus']}", flush=True)
    slot.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=cfg["tag"], t0=time.time(), log_file=lf)


def poll(slot):
    if slot["proc"] is None: return False
    if slot["proc"].poll() is None: return True
    print(f"[{slot['tag']}] DONE ({(time.time()-slot['t0'])/60:.1f} min)", flush=True)
    slot["log_file"].close(); slot.update(proc=None, tag=None, t0=None); return False


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} c508 configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
