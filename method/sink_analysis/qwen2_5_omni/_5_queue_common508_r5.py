"""Round 5 for the 508 always-inert set. Plateau at 78.33%
(a3v3av3 + llm_emerged + tau15). Remaining structured levers: fine tau around
15, route_weighted, asd, modality_complement, and a different variant, all on
the best base. AVHBench DEV, 2x 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")
HEADS = os.path.join(REPO, "results", "qwen2_5_omni",
                     "categorize_exp_2axis_common508", "heads.csv")

def base(a, v, av, tau=15, mask="llm_emerged", *extra):
    return ["--mad_soft", "--gamma_a", str(float(a)), "--gamma_v", str(float(v)),
            "--gamma_av", str(float(av)), "--sink_mask", mask,
            "--tau_sink", str(tau)] + list(extra)

# (tag, variant, extras)
QUEUE = [
    ("c508_a3v3av3_emg_tau13", "boost_inert_content", base(3,3,3,13)),
    ("c508_a3v3av3_emg_tau14", "boost_inert_content", base(3,3,3,14)),
    ("c508_a3v3av3_emg_tau16", "boost_inert_content", base(3,3,3,16)),
    ("c508_a3v3av3_emg_tau17", "boost_inert_content", base(3,3,3,17)),
    ("c508_a3v3av3_emg_t15_routew", "boost_inert_content", base(3,3,3,15,"llm_emerged","--route_weighted")),
    ("c508_a3v3av3_emg_t15_asd",    "boost_inert_content", base(3,3,3,15,"llm_emerged","--asd")),
    ("c508_a3v3av3_emg_t15_modcmp", "boost_inert_content", base(3,3,3,15,"llm_emerged","--modality_complement")),
    ("c508_a3v3av3_emg_t15_textsk", "boost_inert_content", base(3,3,3,15,"llm_emerged","--include_text_sinks")),
    ("c508_a3v3av4_emg_tau15", "boost_inert_content", base(3,3,4,15)),
    ("c508_a3v4av4_emg_tau15", "boost_inert_content", base(3,4,4,15)),
    ("c508_a2v3av3_emg_tau15", "boost_inert_content", base(2,3,3,15)),
    ("c508_a4v4av4_emg_tau15", "boost_inert_content", base(4,4,4,15)),
    ("c508_a3v3av3_emg_t15_iandh", "boost_inert_and_halluc_content", base(3,3,3,15)),
    ("c508_a3v3av3_emg_t13_routew", "boost_inert_content", base(3,3,3,13,"llm_emerged","--route_weighted")),
]

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, c):
    tag, variant, extras = c
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"/tmp/explore_{tag}.log", "w")
    cmd = [PY, SCRIPT, "--variant", variant, "--gamma", "3.0",
           "--heads_csv", HEADS, "--split", "DEV", "--device_map", "balanced_low_0",
           "--tag", tag, "--note", f"508-inert R5 structured levers {tag}"] + extras
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
    print(f"Queue: {len(q)} c508-R5 configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
