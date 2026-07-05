"""Round 3 for the 508 always-inert set. Gamma plateaued at 77.67% (R1+R2,
32 configs). Switch dimension: vary sink_mask / layer_band / tau_sink on the
two best gamma coords (a3v3av3, a5v5av5). AVHBench DEV, 2x 4-GPU slots.
"""
import subprocess, time, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PY = os.path.join(REPO, "qwen_venv", "bin", "python")
SCRIPT = os.path.join(REPO, "method", "sink_analysis", "qwen2_5_omni", "_5_explore.py")
HEADS = os.path.join(REPO, "results", "qwen2_5_omni",
                     "categorize_exp_2axis_common508", "heads.csv")

def mad(a, v, av, *extra):
    return ["--mad_soft", "--gamma_a", str(float(a)), "--gamma_v", str(float(v)),
            "--gamma_av", str(float(av))] + list(extra)

# (tag, extras)  — base γ = a3v3av3 or a5v5av5
QUEUE = [
    ("c508_a3v3av3_emerged",   mad(3,3,3,"--sink_mask","llm_emerged")),
    ("c508_a5v5av5_emerged",   mad(5,5,5,"--sink_mask","llm_emerged")),
    ("c508_a3v3av3_early",     mad(3,3,3,"--sink_mask","all","--layer_band","early")),
    ("c508_a3v3av3_mid",       mad(3,3,3,"--sink_mask","all","--layer_band","mid")),
    ("c508_a3v3av3_late",      mad(3,3,3,"--sink_mask","all","--layer_band","late")),
    ("c508_a5v5av5_early",     mad(5,5,5,"--sink_mask","all","--layer_band","early")),
    ("c508_a5v5av5_mid",       mad(5,5,5,"--sink_mask","all","--layer_band","mid")),
    ("c508_a5v5av5_late",      mad(5,5,5,"--sink_mask","all","--layer_band","late")),
    ("c508_a3v3av3_tau15",     mad(3,3,3,"--sink_mask","all","--tau_sink","15")),
    ("c508_a3v3av3_tau25",     mad(3,3,3,"--sink_mask","all","--tau_sink","25")),
    ("c508_a5v5av5_tau15",     mad(5,5,5,"--sink_mask","all","--tau_sink","15")),
    ("c508_a3v3av3_emg_tau15", mad(3,3,3,"--sink_mask","llm_emerged","--tau_sink","15")),
    ("c508_a5v5av5_emg_late",  mad(5,5,5,"--sink_mask","llm_emerged","--layer_band","late")),
    ("c508_a3v3av3_textsink",  mad(3,3,3,"--sink_mask","all","--include_text_sinks")),
]

SLOTS = [dict(gpus="0,1,2,3", proc=None, tag=None, t0=None),
         dict(gpus="4,5,6,7", proc=None, tag=None, t0=None)]


def launch(slot, cfg):
    tag, extras = cfg
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = slot["gpus"]
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lf = open(f"/tmp/explore_{tag}.log", "w")
    cmd = [PY, SCRIPT, "--variant", "boost_inert_content", "--gamma", "3.0",
           "--heads_csv", HEADS, "--split", "DEV", "--device_map", "balanced_low_0",
           "--tag", tag, "--note", f"508-inert R3 lever sweep {tag}"] + extras
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
    print(f"Queue: {len(q)} c508-R3 configs", flush=True)
    while q or any(s["proc"] for s in SLOTS):
        for slot in SLOTS:
            if slot["proc"] is None and q: launch(slot, q.pop(0))
        time.sleep(10)
        for slot in SLOTS: poll(slot)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
