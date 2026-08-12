"""WorldSense video-length OOM probe for the sink-boost intervention.

Question: at the OFFICIAL VLMEvalKit Qwen2.5-Omni-ForVideo settings
(fps=2, max_pixels=768*28*28, use_audio_in_video=True), how long a video
can *our* method process before the eager-attention prefill OOMs a 24 GB GPU?

Our intervention forces attn_implementation="eager", which materialises the
full (heads, S, S) attention per layer at prefill. Baseline Qwen uses SDPA
and never does, so this ceiling is specific to our method.

Trims one long WorldSense video (zwkbEZwG, 499 s) to a duration sweep, runs
prefill + 1 decode step with the intervention patched & active, and logs
audio/video/total token counts + peak per-GPU memory + OK/OOM.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))

from utils import load_omni, build_conversation, prepare_inputs, find_modality_spans  # noqa
from _5_intervene import patch_qwen_attention, set_intervention, clear_intervention  # noqa
from _5_efficient_encoders import patch_efficient_encoders  # noqa

def use_sdpa_encoders(model):
    """Force the vision & audio ENCODERS onto SDPA attention while leaving the
    thinker decoder eager (so the intervention's eager attn-weight patch still
    fires). Without this, load_omni's global eager forces the ViT to
    materialise a full patch x patch score matrix -> OOM on any real video.
    Handles accelerate's per-instance `_old_forward` device hook."""
    import types
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniVisionAttention, Qwen2_5OmniVisionSdpaAttention,
        Qwen2_5OmniAudioAttention, Qwen2_5OmniAudioSdpaAttention)
    swaps = [(Qwen2_5OmniVisionAttention, Qwen2_5OmniVisionSdpaAttention.forward),
             (Qwen2_5OmniAudioAttention, Qwen2_5OmniAudioSdpaAttention.forward)]
    n = 0
    for _, mod in model.named_modules():
        for cls, fwd in swaps:
            if type(mod) is cls:  # exact eager class, not the sdpa subclass
                if hasattr(mod, "_old_forward"):
                    mod._old_forward = types.MethodType(fwd, mod)
                else:
                    mod.forward = types.MethodType(fwd, mod)
                n += 1
    print(f"swapped {n} encoder attn modules to SDPA", flush=True)


VID = "zwkbEZwG"
SRC = _REPO / "data/WorldSense/videos" / f"{VID}.mp4"
TMP = Path("/tmp/claude-10526/-nobackup2-le-AV-Hallucination/3c6c1667-33a0-48f1-9699-6431bef97b63/scratchpad/wsprobe")
OUT = _REPO / "results/qwen2_5_omni/worldsense/lenprobe.md"

# Official VLMEvalKit Qwen2.5-Omni-ForVideo settings (overridable via CLI).
FPS = 2.0
MAX_PIXELS = 768 * 28 * 28

# A real WorldSense-style MCQ prompt (exact official template).
SYS = ("Carefully watch this video and pay attention to every detail. Based on "
       "your observations, select the best option that accurately addresses the question.")
TMPL = ("These are the frames of a video and the corresponding audio. Select the "
        "best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.")
Q = ("Question: What instrument is being played in the video?\n"
     "A. Guitar\nB. Piano\nC. Guzheng\nD. Drums\nAnswer: ")
PROMPT = SYS + "\n" + TMPL + "\n" + Q


def trim(dur):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{VID}_{dur}s.mp4"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-t", str(dur), "-i", str(SRC),
             "-c", "copy", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def peak_mem_gb():
    return max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def run_one(model, processor, video_path, dur, active):
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.empty_cache()
    conv = build_conversation(str(video_path), PROMPT, modal_type="av",
                              video_fps=FPS, video_max_pixels=MAX_PIXELS)
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "av", model.device, model.dtype)
        n_tot = inputs["input_ids"].shape[1]
        spans = find_modality_spans(inputs["input_ids"], model.thinker.config)
        n_aud = spans.get("audio", (0, 0))[1] - spans.get("audio", (0, 0))[0]
        n_vid = spans.get("video", (0, 0))[1] - spans.get("video", (0, 0))[0]
        if active:
            # memory-faithful config: eager attn materialised, boost mode,
            # empty masks => no numeric change but same peak as a real run.
            set_intervention(dict(mode="boost", gamma=3.0, layer_to_heads={},
                                  layer_to_key_mask={}))
        else:
            clear_intervention()
        t0 = time.time()
        with torch.inference_mode():
            model.generate(**inputs, use_audio_in_video=use_aiv, return_audio=False,
                           do_sample=False, max_new_tokens=1)
        dt = time.time() - t0
        clear_intervention()
        return dict(dur=dur, n_tot=n_tot, n_aud=n_aud, n_vid=n_vid,
                    peak=peak_mem_gb(), dt=dt, status="OK")
    except RuntimeError as e:
        clear_intervention()
        import traceback
        tb = traceback.format_exc()
        is_oom = "out of memory" in str(e).lower()
        print(f"  [FAIL @ {dur}s] {'OOM' if is_oom else 'RuntimeError'}\n{tb[-800:]}", flush=True)
        msg = "OOM" if is_oom else f"ERR:{str(e)[:60]}"
        torch.cuda.empty_cache()
        return dict(dur=dur, n_tot=locals().get("n_tot", -1),
                    n_aud=locals().get("n_aud", -1), n_vid=locals().get("n_vid", -1),
                    peak=peak_mem_gb(), dt=0.0, status=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", type=int, nargs="+",
                    default=[15, 30, 45, 60, 90, 120, 180, 240])
    ap.add_argument("--device_map", default="balanced_low_0")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--eager_encoders", action="store_true",
                    help="leave encoders eager (repro the ViT OOM)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--spread_gb", type=float, default=0.0,
                    help="if >0, use max_memory={i: spread_gb} across all visible GPUs")
    args = ap.parse_args()
    global FPS, MAX_PIXELS
    FPS, MAX_PIXELS = args.fps, args.max_pixels

    print("loading model (eager) ...", flush=True)
    if args.spread_gb > 0:
        mm = {i: f"{args.spread_gb:.0f}GiB" for i in range(torch.cuda.device_count())}
        model, processor = load_omni(args.model_path, device_map="auto", max_memory=mm)
    else:
        model, processor = load_omni(args.model_path, device_map=args.device_map)
    patch_qwen_attention(model)
    if not args.eager_encoders:
        patch_efficient_encoders(model)
    n_gpu = torch.cuda.device_count()
    permb = [torch.cuda.memory_allocated(i) / 1e9 for i in range(n_gpu)]
    print(f"after load, {n_gpu} GPUs, per-GPU weights (GB): "
          + " ".join(f"{m:.1f}" for m in permb), flush=True)

    rows = []
    for dur in args.durations:
        vp = trim(dur)
        r = run_one(model, processor, vp, dur, active=True)
        rows.append(r)
        print(f"[{dur:4d}s] tot={r['n_tot']:6d} aud={r['n_aud']:5d} vid={r['n_vid']:6d} "
              f"peak={r['peak']:5.1f}GB {r['dt']:5.1f}s {r['status']}", flush=True)
        if r["status"] != "OK":
            print("  -> stopping sweep at first failure", flush=True)
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# WorldSense length probe — sink-boost (eager) intervention\n\n")
        f.write(f"Video `{VID}` (499 s) trimmed; fps={FPS}, "
                f"max_pixels={MAX_PIXELS}, use_audio_in_video=True; device_map={args.device_map}.\n\n")
        f.write("| dur (s) | total tok | audio tok | video tok | peak mem (GB) | prefill (s) | status |\n")
        f.write("|--:|--:|--:|--:|--:|--:|:--|\n")
        for r in rows:
            f.write(f"| {r['dur']} | {r['n_tot']} | {r['n_aud']} | {r['n_vid']} | "
                    f"{r['peak']:.1f} | {r['dt']:.1f} | {r['status']} |\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
