"""Finish the remaining reverse-best AVS runs on BOTH 4-GPU slots (0-3 freed up).

anxan AVS is already running (orphaned on 4-7); this handles the other 4
(anxyt, lixan, lixyt, c508). Two slots: 0,1,2,3 and 4,5,6,7. A slot only
launches when all its GPUs are free (<2.5 GB used) — so 4-7 is picked up
only after anxan finishes. Skips tags already in avspeaker_log.
"""
import subprocess, time, os, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
AVS = os.path.join(REPO, "method/sink_analysis/qwen2_5_omni/_5_avspeaker_eval.py")
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
TMP = "/nobackup2/le/tmp_claude"
R = os.path.join(REPO, "results/qwen2_5_omni")
ALOG = os.path.join(R, "stage5_intervention/avspeaker/avspeaker_log.md")

# (abbr, pair_for_sched, taxo_dir, best_g) — anxan excluded (already running on 4-7)
PENDING = [
    ("anxyt", "AudioSetxYouTubeVOS",       "categorize_exp_2axis_youtubevos",         4),
    ("lixan", "LibriSpeechxActivityNet",   "categorize_exp_2axis_libri_x_activitynet", 5),
    ("lixyt", "LibriSpeechxYouTubeVOS",    "categorize_exp_2axis_libri_x_youtubevos",  5),
    ("c508",  "common508",                 "categorize_exp_2axis_common508",          5),
]

SLOTS = [
    dict(gpus="0,1,2,3", proc=None, tag=None, t0=None, log_file=None),
    dict(gpus="4,5,6,7", proc=None, tag=None, t0=None, log_file=None),
]


def gpu_used():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode()
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}


def slot_free(slot, used):
    return all(used.get(int(g), 99999) < 2500 for g in slot["gpus"].split(","))


def already_done(tag):
    return os.path.exists(ALOG) and f"## {tag}" in open(ALOG).read()


def launch(slot, item):
    abbr, pair, taxo, g = item
    tag = f"sched_rev_{abbr}_g{g}_avs_ofmt"
    heads = os.path.join(R, taxo, "heads.csv")
    sched = os.path.join(SCHED, f"rev_sched_{pair}.npz")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"{TMP}/{tag}.log", "w")
    cmd = [PY, AVS, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--sink_mask", "all", "--heads_csv", heads,
           "--gamma_schedule_npz", sched, "--g_base", str(float(g)),
           "--tag", tag, "--device_map", "balanced_low_0",
           "--video_fps", "1.0", "--max_duration_s", "15",
           "--note", f"reverse-schedule best on AVS (official, 2slot): {pair} g{g}"]
    print(f"[{tag}] LAUNCH on GPUs {slot['gpus']}", flush=True)
    slot.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=tag, t0=time.time(), log_file=lf)


def main():
    q = [it for it in PENDING if not already_done(f"sched_rev_{it[0]}_g{it[3]}_avs_ofmt")]
    print(f"2-slot AVS finisher: {len(q)} pending {[it[0] for it in q]}", flush=True)
    while q or any(s["proc"] is not None for s in SLOTS):
        used = gpu_used()
        for slot in SLOTS:
            if slot["proc"] is None and q and slot_free(slot, used):
                # re-check log at launch time (another slot may have just done it)
                while q and already_done(f"sched_rev_{q[0][0]}_g{q[0][3]}_avs_ofmt"):
                    q.pop(0)
                if q:
                    launch(slot, q.pop(0))
                    used = gpu_used()  # refresh so the other slot sees the new usage
        time.sleep(20)
        for slot in SLOTS:
            if slot["proc"] is not None and slot["proc"].poll() is not None:
                print(f"[{slot['tag']}] DONE ({(time.time()-slot['t0'])/60:.1f} min) "
                      f"on GPUs {slot['gpus']}", flush=True)
                slot["log_file"].close()
                slot.update(proc=None, tag=None, t0=None, log_file=None)
    print("ALL AVS RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
