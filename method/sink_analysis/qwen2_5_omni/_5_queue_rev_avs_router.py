"""Re-run the 5 reverse-schedule AVS configs with the AVHBench ROUTER routing
(--route_from router): soft (p_a,p_v,p_av) blend the schedule, argmax sets
routed_mod — identical mechanism to _5_explore.py on AVHBench.

New tags `sched_rev_<abbr>_g<g>_avs_router` (do NOT clobber the earlier
category-routed `_avs_ofmt` results). Two 4-GPU slots gated on GPU-free
(<2.5 GB), so it uses whatever is idle now (0-3) and grabs 4-7 when it frees.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
AVS = os.path.join(REPO, "method/sink_analysis/qwen2_5_omni/_5_avspeaker_eval.py")
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
TMP = "/nobackup2/le/tmp_claude"
R = os.path.join(REPO, "results/qwen2_5_omni")
ALOG = os.path.join(R, "stage5_intervention/avspeaker/avspeaker_log.md")

# (abbr, pair_for_sched, taxo_dir, best_g)
# anxan = done (logged); anxyt = running orphaned on 1,2,4,7. Remaining 3 below
# fan out across both 4-GPU slots now that 0,3,5,6 are free.
CONFIGS = [
    ("lixan", "LibriSpeechxActivityNet",   "categorize_exp_2axis_libri_x_activitynet", 5),
    ("lixyt", "LibriSpeechxYouTubeVOS",    "categorize_exp_2axis_libri_x_youtubevos",  5),
    ("c508",  "common508",                 "categorize_exp_2axis_common508",           5),
]

# Two 4-GPU slots. 0,3,5,6 free now; 1,2,4,7 busy with the anxyt orphan and
# gated on GPU-free, so it's picked up only after anxyt finishes.
SLOTS = [
    dict(gpus="0,3,5,6", proc=None, tag=None, t0=None, log_file=None),
    dict(gpus="1,2,4,7", proc=None, tag=None, t0=None, log_file=None),
]


def gpu_used():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"]).decode()
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}


def slot_free(slot, used):
    return all(used.get(int(g), 99999) < 2500 for g in slot["gpus"].split(","))


def done(abbr, g):
    tag = f"sched_rev_{abbr}_g{g}_avs_router"
    return os.path.exists(ALOG) and f"## {tag}" in open(ALOG).read()


def launch(slot, item):
    abbr, pair, taxo, g = item
    tag = f"sched_rev_{abbr}_g{g}_avs_router"
    heads = os.path.join(R, taxo, "heads.csv")
    sched = os.path.join(SCHED, f"rev_sched_{pair}.npz")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"{TMP}/{tag}.log", "w")
    cmd = [PY, AVS, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--sink_mask", "all", "--heads_csv", heads,
           "--gamma_schedule_npz", sched, "--g_base", str(float(g)),
           "--route_from", "router",
           "--tag", tag, "--device_map", "balanced_low_0",
           "--video_fps", "1.0", "--max_duration_s", "15",
           "--note", f"reverse-schedule on AVS, AVHBench-router routing: {pair} g{g}"]
    print(f"[{tag}] LAUNCH on GPUs {slot['gpus']}", flush=True)
    slot.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=tag, t0=time.time(), log_file=lf)


def main():
    q = [c for c in CONFIGS if not done(c[0], c[3])]
    print(f"router-routing AVS re-run: {len(q)} pending {[c[0] for c in q]}", flush=True)
    while q or any(s["proc"] is not None for s in SLOTS):
        used = gpu_used()
        for slot in SLOTS:
            if slot["proc"] is None and q and slot_free(slot, used):
                while q and done(q[0][0], q[0][3]):
                    q.pop(0)
                if q:
                    launch(slot, q.pop(0))
                    used = gpu_used()
        time.sleep(20)
        for slot in SLOTS:
            if slot["proc"] is not None and slot["proc"].poll() is not None:
                print(f"[{slot['tag']}] DONE ({(time.time()-slot['t0'])/60:.1f} min) "
                      f"on GPUs {slot['gpus']}", flush=True)
                slot["log_file"].close()
                slot.update(proc=None, tag=None, t0=None, log_file=None)
    print("ALL ROUTER-AVS RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
