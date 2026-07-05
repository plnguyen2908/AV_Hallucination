"""LibriSpeech-audio-axis intervention sweep (AVHBench DEV).

Explore the best MAD-soft per-category gamma for head taxonomies built from the
LibriSpeech audio axis x each of the 2 visual datasets:
  libxan = LibriSpeech x ActivityNet   (Audio 21 / Visual 48 / AV 5 / Inert 710)
  libxyt = LibriSpeech x YouTube-VOS    (Audio 6 / Visual 138 / AV 0 / Inert 640)

Mirrors _5_queue_ytvos.py; pins --heads_csv per taxonomy. 6 gamma configs each,
anchored on the proven a3v3av5 region. AVHBench DEV (n=300), 2x 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")

def heads(tax):
    return os.path.join(REPO, "results", "qwen2_5_omni",
                        f"categorize_exp_2axis_{tax}", "heads.csv")

TAXONOMIES = [
    ("libxan", heads("libri_x_activitynet")),
    ("libxyt", heads("libri_x_youtubevos")),
]

def mad(a, v, av):
    return ["--mad_soft", "--gamma_a", str(a), "--gamma_v", str(v),
            "--gamma_av", str(av), "--sink_mask", "all"]

# (suffix, gamma_a, gamma_v, gamma_av)
GRID = [
    ("a3v3av5", 3.0, 3.0, 5.0),   # YT-VOS best anchor
    ("a3v3av3", 3.0, 3.0, 3.0),
    ("a3v5av5", 3.0, 5.0, 5.0),
    ("a5v3av5", 5.0, 3.0, 5.0),
    ("a3v5av8", 3.0, 5.0, 8.0),
    ("a2v2av3", 2.0, 2.0, 3.0),
]

QUEUE = []
for tax_tag, hcsv in TAXONOMIES:
    for suf, ga, gv, gav in GRID:
        QUEUE.append(dict(
            tag=f"{tax_tag}_{suf}", variant="boost_inert_content", gamma=3.0,
            heads_csv=hcsv,
            note=f"LibriSpeech-audio sweep: {tax_tag} MAD-soft {suf}",
            extras=mad(ga, gv, gav)))

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, cfg):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    lf = open(f"/tmp/explore_{cfg['tag']}.log", "w")
    cmd = [PY, SCRIPT, "--variant", cfg["variant"], "--gamma", str(cfg["gamma"]),
           "--tag", cfg["tag"], "--split", "DEV", "--device_map", "balanced_low_0",
           "--heads_csv", cfg["heads_csv"], "--note", cfg["note"]] + cfg["extras"]
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
    print(f"Queue: {len(q)} configs (2 taxonomies x {len(GRID)} gammas)", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
