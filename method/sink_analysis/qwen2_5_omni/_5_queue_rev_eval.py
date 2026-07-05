"""Run the 5 reverse-schedule BEST configs on AVHBench FULL + AVS (official).

Best reverse g_base per head-set (from the DEV sweep):
  anxan g4, anxyt g4, lixan g5, lixyt g5, c508 g5.
10 runs (5 FULL + 5 AVS) on a single 4-GPU slot (GPUs 0-2 busy). FULL first
(more discriminating), then AVS. Skips runs already in the logs.
"""
import subprocess, time, os, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
EXPLORE = os.path.join(REPO, "method/sink_analysis/qwen2_5_omni/_5_explore.py")
AVS = os.path.join(REPO, "method/sink_analysis/qwen2_5_omni/_5_avspeaker_eval.py")
SCHED = os.path.join(REPO, "results/qwen2_5_omni/sink_analysis/gamma_schedules")
TMP = "/nobackup2/le/tmp_claude"
R = os.path.join(REPO, "results/qwen2_5_omni")
MLOG = os.path.join(R, "stage5_intervention/method_log.md")
ALOG = os.path.join(R, "stage5_intervention/avspeaker/avspeaker_log.md")

# (abbr, pair_for_sched, taxo_dir, best_g)
BEST = [
    ("anxan", "AudioSetxActivityNet",      "categorize_exp_2axis",                    4),
    ("anxyt", "AudioSetxYouTubeVOS",       "categorize_exp_2axis_youtubevos",         4),
    ("lixan", "LibriSpeechxActivityNet",   "categorize_exp_2axis_libri_x_activitynet", 5),
    ("lixyt", "LibriSpeechxYouTubeVOS",    "categorize_exp_2axis_libri_x_youtubevos",  5),
    ("c508",  "common508",                 "categorize_exp_2axis_common508",          5),
]

mtxt = open(MLOG).read() if os.path.exists(MLOG) else ""
atxt = open(ALOG).read() if os.path.exists(ALOG) else ""

QUEUE = []
for abbr, pair, taxo, g in BEST:  # FULL first
    tag = f"sched_rev_{abbr}_g{g}_FULL"
    if f"## {tag}" in mtxt:
        continue
    QUEUE.append(("FULL", tag, taxo, pair, g))
for abbr, pair, taxo, g in BEST:  # then AVS
    tag = f"sched_rev_{abbr}_g{g}_avs_ofmt"
    if f"## {tag}" in atxt:
        continue
    QUEUE.append(("AVS", tag, taxo, pair, g))

SLOT = dict(proc=None, tag=None, t0=None)


def launch(item):
    kind, tag, taxo, pair, g = item
    heads = os.path.join(R, taxo, "heads.csv")
    sched = os.path.join(SCHED, f"rev_sched_{pair}.npz")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    lf = open(f"{TMP}/{tag}.log", "w")
    if kind == "FULL":
        cmd = [PY, EXPLORE, "--variant", "boost_inert_content", "--gamma", "3.0",
               "--mad_soft", "--sink_mask", "all", "--heads_csv", heads,
               "--gamma_schedule_npz", sched, "--g_base", str(float(g)),
               "--split", "FULL", "--tag", tag, "--device_map", "balanced_low_0",
               "--note", f"reverse-schedule best on AVHBench FULL: {pair} g{g}"]
    else:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [PY, AVS, "--variant", "boost_inert_content", "--gamma", "3.0",
               "--sink_mask", "all", "--heads_csv", heads,
               "--gamma_schedule_npz", sched, "--g_base", str(float(g)),
               "--tag", tag, "--device_map", "balanced_low_0",
               "--video_fps", "1.0", "--max_duration_s", "15",
               "--note", f"reverse-schedule best on AVS (official): {pair} g{g}"]
    print(f"[{tag}] LAUNCH ({kind})", flush=True)
    SLOT.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=tag, t0=time.time(), log_file=lf)


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} runs ({sum(1 for x in q if x[0]=='FULL')} FULL + "
          f"{sum(1 for x in q if x[0]=='AVS')} AVS) on slot 4,5,6,7", flush=True)
    while q or SLOT["proc"] is not None:
        if SLOT["proc"] is None and q:
            launch(q.pop(0))
        time.sleep(15)
        if SLOT["proc"] is not None and SLOT["proc"].poll() is not None:
            print(f"[{SLOT['tag']}] DONE ({(time.time()-SLOT['t0'])/60:.1f} min)", flush=True)
            SLOT["log_file"].close(); SLOT.update(proc=None, tag=None, t0=None)
    print("ALL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
