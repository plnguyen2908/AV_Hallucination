"""REVERSE per-layer scheduled-gamma sweep on AVHBench DEV.

Same as _5_queue_sched.py but uses the REVERSE schedule
(gamma high where sinks are SPARSE -> early layers; ~0 at the deep sink-peak),
i.e. shape = 1 - prop/max(prop). 5 head-sets x g_base{2,3,4,5}. Single 4-GPU
slot (GPUs 0-2 busy with other work). Tags: sched_rev_*.
"""
import subprocess, time, os, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")
SCHED = os.path.join(REPO, "results", "qwen2_5_omni", "sink_analysis", "gamma_schedules")
TMP = "/nobackup2/le/tmp_claude"

PAIRS = [
    ("anxan", "AudioSetxActivityNet",      "categorize_exp_2axis"),
    ("anxyt", "AudioSetxYouTubeVOS",       "categorize_exp_2axis_youtubevos"),
    ("lixan", "LibriSpeechxActivityNet",   "categorize_exp_2axis_libri_x_activitynet"),
    ("lixyt", "LibriSpeechxYouTubeVOS",    "categorize_exp_2axis_libri_x_youtubevos"),
    ("c508",  "common508",                 "categorize_exp_2axis_common508"),
]
G_BASES = [2.0, 3.0, 4.0, 5.0]

_MLOG = os.path.join(REPO, "results/qwen2_5_omni/stage5_intervention/method_log.md")
_DONE = set(re.findall(r'^## (sched_rev_\w+)', open(_MLOG).read(), re.M)) if os.path.exists(_MLOG) else set()

QUEUE = []
for abbr, pair, taxo in PAIRS:
    for g in G_BASES:
        tag = f"sched_rev_{abbr}_g{int(g)}"
        if tag in _DONE:
            continue
        QUEUE.append(dict(
            tag=tag,
            heads=os.path.join(REPO, "results", "qwen2_5_omni", taxo, "heads.csv"),
            sched=os.path.join(SCHED, f"rev_sched_{pair}.npz"),
            g=g, note=f"REVERSE per-layer scheduled gamma, pair={pair}, g_base={g}"))

SLOT = dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)


def launch(cfg):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = SLOT["gpus"]
    lf = open(f"{TMP}/explore_{cfg['tag']}.log", "w")
    cmd = [PY, SCRIPT, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--mad_soft", "--sink_mask", "all", "--heads_csv", cfg["heads"],
           "--gamma_schedule_npz", cfg["sched"], "--g_base", str(cfg["g"]),
           "--split", "DEV", "--device_map", "balanced_low_0",
           "--tag", cfg["tag"], "--note", cfg["note"]]
    print(f"[{cfg['tag']}] LAUNCH", flush=True)
    SLOT.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=cfg["tag"], t0=time.time(), log_file=lf)


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} REVERSE scheduled-gamma configs (slot {SLOT['gpus']})", flush=True)
    while q or SLOT["proc"] is not None:
        if SLOT["proc"] is None and q:
            launch(q.pop(0))
        time.sleep(10)
        if SLOT["proc"] is not None and SLOT["proc"].poll() is not None:
            print(f"[{SLOT['tag']}] DONE ({(time.time()-SLOT['t0'])/60:.1f} min)", flush=True)
            SLOT["log_file"].close(); SLOT.update(proc=None, tag=None, t0=None)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
