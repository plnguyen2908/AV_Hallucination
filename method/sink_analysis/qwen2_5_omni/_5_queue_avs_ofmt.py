"""Re-run the 3 non-ActivityNet×AudioSet best configs on AVS in the OFFICIAL
setup (1fps, NO resize, official prompt) so they are comparable to the best
ActivityNet×AudioSet AVS result (ours_13d a3v5av5 = 43.00%, _ofmt).

The official prompt is always on (build_question). "1fps without resizing" =
--video_fps 1.0 with no --video_resized_hw / --video_max_pixels override.
≤15s subset (n=2065). Two 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_avspeaker_eval.py")

def heads(tax):
    return os.path.join(REPO, "results", "qwen2_5_omni",
                        f"categorize_exp_2axis{tax}", "heads.csv")

# (tag, heads_csv, gamma_a, gamma_v, gamma_av)
QUEUE = [
    ("ytx_a3v3av5_avs_ofmt",   heads("_youtubevos"),         3.0, 3.0, 5.0),
    ("libxan_a3v3av3_avs_ofmt", heads("_libri_x_activitynet"), 3.0, 3.0, 3.0),
    ("libxyt_a3v5av8_avs_ofmt", heads("_libri_x_youtubevos"),  3.0, 5.0, 8.0),
]

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, cfg):
    tag, hcsv, ga, gv, gav = cfg
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    # No-resize clips produce long video sequences; eager attention (required
    # by the intervention) materializes a large (B,H,Q,K) matrix. The 4-GPU
    # balanced_low_0 placement over-packs one GPU, leaving ~3.7GB free + ~3.1GB
    # reserved-but-unallocated. expandable_segments reclaims the fragmented
    # reserve so the attention alloc fits without OOM.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"/tmp/avs_ofmt_{tag}.log", "w")
    cmd = [PY, SCRIPT,
           "--variant", "boost_inert_content", "--gamma", "3.0",
           "--gamma_a", str(ga), "--gamma_v", str(gv), "--gamma_av", str(gav),
           "--sink_mask", "all", "--heads_csv", hcsv, "--tag", tag,
           "--device_map", "balanced_low_0",
           "--video_fps", "1.0", "--max_duration_s", "15",
           "--note", f"{tag}: official setup (1fps, no resize, official prompt) for fair vs ActivityNet×AudioSet 43%"]
    print(f"[{tag}] LAUNCH on {slot['gpus']}", flush=True)
    slot.update(proc=subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                tag=tag, t0=time.time(), log_file=lf)


def poll(slot):
    if slot["proc"] is None: return False
    if slot["proc"].poll() is None: return True
    print(f"[{slot['tag']}] DONE ({(time.time()-slot['t0'])/60:.1f} min)", flush=True)
    slot["log_file"].close(); slot.update(proc=None, tag=None, t0=None); return False


def main():
    q = list(QUEUE)
    print(f"Queue: {len(q)} AVS-ofmt configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
