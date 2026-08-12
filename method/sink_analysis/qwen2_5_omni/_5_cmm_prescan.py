"""_5_cmm_prescan.py — find CMM videos whose decode HANGS, and repair them.

Some CMM mp4s make decord error out; qwen_omni_utils then falls back to
torchvision, which spins forever in native code on those files. Measured on
reorg_raw_files/.../visual-audio-language/6t3hJb.mp4: fine at fps=1 (14 frames),
hangs at fps=2 and at nframes=16. Because the spin is inside a C extension, a
SIGALRM in the harness never gets delivered -- the only safe place to time it
out is a SEPARATE PROCESS. Hence this pre-flight rather than an in-harness guard.

An ffmpeg re-encode (libx264, yuv420p) fixes them: the same file then decodes in
0.0 s at nframes=16.

Writes data/CMM/_fixed/<stem>.mp4 for each bad file plus _fixed/map.json
{original_abs_path: repaired_abs_path}, which run_gen_cmm.py consults.

  python method/sink_analysis/qwen2_5_omni/_5_cmm_prescan.py --workers 8
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
DATA = _REPO / "data/CMM"
FIXED = DATA / "_fixed"
MAP = FIXED / "map.json"

# Probe EVERY sampling spec the harness can choose, not just one: a clip whose
# duration*fps <= cap stays on the "fps" path and never sees "nframes", so a
# nframes-only probe misses files that hang at fps=2 (measured: vPf81e.mp4,
# 4.9s, fine at nframes=16, hangs at fps=2).
_SPECS = [{"nframes": 16}, {"fps": 2.0}, {"fps": 1.0}]

_PROBE = r'''
import sys, json
sys.path.insert(0, "%s")
from qwen_omni_utils import process_mm_info
f, specs = sys.argv[1], json.loads(sys.argv[2])
for extra in specs:
    conv = [{"role": "user", "content": [
        {"type": "video", "video": f, "max_pixels": %d, **extra},
        {"type": "text", "text": "x"}]}]
    process_mm_info(conv, use_audio_in_video=False)
print("OK")
''' % (_REPO / "method/qwen3_omni", 360 * 640)


def probe(path, timeout):
    """Decode in a subprocess. -> True if it completes in time."""
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE, path,
                            json.dumps(_SPECS)],
                           capture_output=True, text=True, timeout=timeout)
        return "OK" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def reencode(path):
    FIXED.mkdir(parents=True, exist_ok=True)
    out = FIXED / (Path(path).stem + ".mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True)
    return str(out)


def work(args):
    path, timeout = args
    if probe(path, timeout):
        return (path, None)
    try:
        fixed = reencode(path)
    except Exception as e:
        return (path, f"REENCODE_FAILED:{type(e).__name__}")
    return (path, fixed if probe(fixed, timeout) else "STILL_BAD")


def main(a):
    data = json.load(open(DATA / "all_data_final_reorg.json"))
    vids = sorted({str(DATA / r["video_path"].replace("./reorg_raw_files",
                                                      "reorg_raw_files"))
                   for r in data if r.get("video_path")})
    vids = [v for v in vids if os.path.exists(v)]
    print(f"probing {len(vids)} unique CMM videos "
          f"(timeout {a.timeout}s, {a.workers} workers)", flush=True)
    m, bad = {}, []
    with mp.Pool(a.workers) as pool:
        for i, (path, res) in enumerate(
                pool.imap_unordered(work, [(v, a.timeout) for v in vids])):
            if res is None:
                pass
            elif res.startswith(("REENCODE_FAILED", "STILL_BAD")):
                bad.append((path, res))
                print(f"  UNFIXABLE {os.path.basename(path)} {res}", flush=True)
            else:
                m[path] = res
                print(f"  repaired {os.path.basename(path)}", flush=True)
            if (i + 1) % 250 == 0:
                print(f"  [{i+1}/{len(vids)}] repaired={len(m)} unfixable={len(bad)}",
                      flush=True)
    FIXED.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(MAP, "w"), indent=1)
    print(f"\n{len(vids)} probed | {len(m)} repaired | {len(bad)} unfixable")
    print(f"wrote {MAP}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
