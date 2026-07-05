"""Round 4 for the 508 always-inert set. R3 found the lever: lower tau_sink +
sink_mask=llm_emerged (a3v3av3_emg_tau15 = 78.33%). Push tau_sink lower (5-12)
with llm_emerged across the best gamma coords. AVHBench DEV, 2x 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")
HEADS = os.path.join(REPO, "results", "qwen2_5_omni",
                     "categorize_exp_2axis_common508", "heads.csv")

def cfg(a, v, av, mask, tau):
    return ["--mad_soft", "--gamma_a", str(float(a)), "--gamma_v", str(float(v)),
            "--gamma_av", str(float(av)), "--sink_mask", mask, "--tau_sink", str(tau)]

QUEUE = [
    ("c508_a3v3av3_emg_tau10", cfg(3,3,3,"llm_emerged",10)),
    ("c508_a3v3av3_emg_tau12", cfg(3,3,3,"llm_emerged",12)),
    ("c508_a3v3av3_emg_tau8",  cfg(3,3,3,"llm_emerged",8)),
    ("c508_a3v3av3_emg_tau5",  cfg(3,3,3,"llm_emerged",5)),
    ("c508_a3v5av5_emg_tau15", cfg(3,5,5,"llm_emerged",15)),
    ("c508_a3v5av5_emg_tau10", cfg(3,5,5,"llm_emerged",10)),
    ("c508_a5v5av5_emg_tau10", cfg(5,5,5,"llm_emerged",10)),
    ("c508_a3v3av5_emg_tau10", cfg(3,3,5,"llm_emerged",10)),
    ("c508_a3v3av5_emg_tau15", cfg(3,3,5,"llm_emerged",15)),
    ("c508_a2v2av2_emg_tau15", cfg(2,2,2,"llm_emerged",15)),
    ("c508_a2v2av2_emg_tau10", cfg(2,2,2,"llm_emerged",10)),
    ("c508_a4v4av4_emg_tau10", cfg(4,4,4,"llm_emerged",10)),
    ("c508_a3v3av3_all_tau10", cfg(3,3,3,"all",10)),
    ("c508_a3v3av3_all_tau8",  cfg(3,3,3,"all",8)),
]

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, c):
    tag, extras = c
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"/tmp/explore_{tag}.log", "w")
    cmd = [PY, SCRIPT, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--heads_csv", HEADS, "--split", "DEV", "--device_map", "balanced_low_0",
           "--tag", tag, "--note", f"508-inert R4 tau-sink push {tag}"] + extras
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
    print(f"Queue: {len(q)} c508-R4 configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
