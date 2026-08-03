"""run_mad_avhbench.py — MAD on AVHBench (yes/no) for Qwen3-Omni. SDPA (no attn patch)."""
import argparse, re, sys, time
from pathlib import Path
import pandas as pd, torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "method/qwen3_omni")); import utils  # noqa
sys.path.insert(0, str(_HERE)); import mad as M  # noqa

MODEL = "/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
SPLIT_CSV = _REPO / "results/qwen2_5_omni/stage5_intervention/split.csv"
VIDEO_DIR = _REPO / "data/AVHBench/videos"
SUF = " Answer with only 'Yes' or 'No'."


def yn_ids(tok):
    y, n = [], []
    for s in ["Yes", "yes", " Yes", " yes"]:
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1: y.append(t[0])
    for s in ["No", "no", " No", " no"]:
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1: n.append(t[0])
    return list(set(y)), list(set(n))


def main(a):
    df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(5)
    rows = df if a.split == "FULL" else df[df.split == a.split]
    if a.limit: rows = rows.iloc[: a.limit]
    print(f"MAD AVHBench {a.split}: {len(rows)} Q  gamma={a.gamma}", flush=True)
    model, processor = utils.load_omni(MODEL, attn_implementation="sdpa")
    yes_ids, no_ids = yn_ids(processor.tokenizer)
    cor = {}; tot = {}
    t0 = time.time()
    for i, (_, r) in enumerate(rows.iterrows()):
        vp = VIDEO_DIR / f"{r.video_id}.mp4"
        if not vp.exists(): continue
        try:
            logits, info = M.mad_answer_logits(model, processor, str(vp), r.text + SUF, gamma=a.gamma)
        except Exception as e:
            print(f"  err {r.video_id}: {e}", flush=True); continue
        l = logits.float()
        pred = "Yes" if l[yes_ids].max() > l[no_ids].max() else "No"
        ok = int(pred == r.label)
        cor[r.task] = cor.get(r.task, 0) + ok; tot[r.task] = tot.get(r.task, 0) + 1
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            N = sum(tot.values()); C = sum(cor.values())
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s overall={C/max(N,1):.3f}", flush=True)
    N = sum(tot.values()); C = sum(cor.values())
    print(f"\n=== MAD {a.split} n={N}: {C/N:.4f} ===")
    for t in sorted(tot):
        print(f"  {t:36s}: {cor[t]/tot[t]:.4f} (n={tot[t]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="DEV")
    p.add_argument("--gamma", type=float, default=2.5)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
