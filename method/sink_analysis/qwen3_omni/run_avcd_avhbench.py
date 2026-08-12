"""run_avcd_avhbench.py — AVCD baseline on AVHBench (yes/no) for Qwen3-Omni.

Same split / prompt / config as run_avhbench.py's baseline (fps=1, 360x640,
greedy). Generation is replaced by avcd_answer_logits restricted to Yes/No.
"""
import argparse, re, sys, time
from pathlib import Path
import pandas as pd, torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "method/qwen3_omni")); import utils  # noqa
sys.path.insert(0, str(_HERE)); import avcd as A  # noqa

MODEL = "/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
SPLIT_CSV = _REPO / "results/qwen2_5_omni/stage5_intervention/split.csv"
VIDEO_DIR = _REPO / "data/AVHBench/videos"
SUF = " Answer with only 'Yes' or 'No'."


def yn_ids(tok):
    yes, no = [], []
    for s in ["Yes", "yes", " Yes", " yes"]:
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1: yes.append(t[0])
    for s in ["No", "no", " No", " no"]:
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1: no.append(t[0])
    return list(set(yes)), list(set(no))


def main(a):
    df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(5)
    rows = df if a.split == "FULL" else df[df.split == a.split]
    if a.limit: rows = rows.iloc[: a.limit]
    print(f"AVCD AVHBench {a.split}: {len(rows)} Q  cd_alpha={a.cd_alpha}", flush=True)
    model, processor = utils.load_omni(MODEL)   # eager
    A.patch(model)
    yes_ids, no_ids = yn_ids(processor.tokenizer)
    print(f"  yes_ids={yes_ids} no_ids={no_ids}", flush=True)

    cor = {}; tot = {}; gated = 0
    t0 = time.time()
    for i, (_, r) in enumerate(rows.iterrows()):
        vp = VIDEO_DIR / f"{r.video_id}.mp4"
        if not vp.exists(): continue
        conv = utils.build_conversation(str(vp), r.text + SUF, "av")
        try:
            inputs, use_aiv = utils.prepare_inputs(processor, conv, "av", model.device, model.dtype)
        except Exception as e:
            print(f"  skip {r.video_id}: {e}", flush=True); continue
        A.set_spans_from_ids(inputs["input_ids"])
        try:
            logits, info = A.avcd_answer_logits(model, inputs, use_aiv, cd_alpha=a.cd_alpha)
        except Exception as e:
            print(f"  err {r.video_id}: {e}", flush=True); continue
        gated += int(info.get("gated", False))
        l = logits[0].float()
        pred = "Yes" if l[yes_ids].max() > l[no_ids].max() else "No"
        ok = int(pred == r.label)
        cor[r.task] = cor.get(r.task, 0) + ok; tot[r.task] = tot.get(r.task, 0) + 1
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            N = sum(tot.values()); C = sum(cor.values())
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s overall={C/max(N,1):.3f} gated={gated}", flush=True)
    N = sum(tot.values()); C = sum(cor.values())
    print(f"\n=== AVCD {a.split} n={N}: {C/N:.4f}  (gated {gated}) ===")
    for t in sorted(tot):
        print(f"  {t:36s}: {cor[t]/tot[t]:.4f} (n={tot[t]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="DEV")
    p.add_argument("--cd_alpha", type=float, default=2.5)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
