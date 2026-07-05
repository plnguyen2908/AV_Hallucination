"""Per-layer scheduled-gamma intervention sweep on AVHBench DEV.

For each audio×visual PAIR, the gamma schedule (gamma_a/gamma_v/gamma_av per
layer) comes from that pair's synthetic-AV sink decomposition (stage2_1d).
Per clip per layer: gamma(L) = g_base * (shape_a*p_a + shape_v*p_v + shape_av*p_av).
Sweep g_base to find the best per pair. SINGLE 4-GPU slot (GPUs 0-2 are busy
with other work; 2-GPU placement OOMs).
"""
import subprocess, time, os

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
    ("c508",  "common508",                 "categorize_exp_2axis_common508"),  # 508 always-inert core
]
G_BASES = [2.0, 3.0, 4.0, 5.0]

# Skip configs already in method_log (so relaunching only runs the missing ones).
_MLOG = os.path.join(REPO, "results/qwen2_5_omni/stage5_intervention/method_log.md")
_DONE = set()
if os.path.exists(_MLOG):
    import re as _re
    _DONE = set(_re.findall(r'^## (sched_\w+)', open(_MLOG).read(), _re.M))

QUEUE = []
for abbr, pair, taxo in PAIRS:
    for g in G_BASES:
        tag = f"sched_{abbr}_g{int(g)}"
        if tag in _DONE:
            continue
        QUEUE.append(dict(
            tag=tag,
            heads=os.path.join(REPO, "results", "qwen2_5_omni", taxo, "heads.csv"),
            sched=os.path.join(SCHED, f"sched_{pair}.npz"),
            g=g, note=f"per-layer scheduled gamma, pair={pair}, g_base={g}"))

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
    print(f"Queue: {len(q)} scheduled-gamma configs (single slot {SLOT['gpus']})", flush=True)
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
