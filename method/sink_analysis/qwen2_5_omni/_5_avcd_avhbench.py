"""_5_avcd_avhbench.py — AVCD baseline on AVHBench (yes/no).

Reuses the exact split, prompt (YES_NO_SUFFIX), conversation, and scoring as
_5_explore.py's baseline path; only the generation step is replaced by the
AVCD contrastive forward (avcd_answer_logits), restricted to Yes/No.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_avcd_avhbench.py \
      --split FULL --cd_alpha 2.5 --tag avcd_avhbench
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))

from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402
import _5_explore as EX  # noqa: E402
from _5_avcd_qwen import patch_thinker_avcd, set_spans, avcd_answer_logits  # noqa: E402
from _5_efficient_encoders import patch_efficient_encoders  # noqa: E402

OUT_DIR = _REPO / "results/qwen2_5_omni/stage5_intervention"
DEFAULT_SPLIT = OUT_DIR / "split.csv"
DEFAULT_VIDEO_DIR = _REPO / "data/AVHBench/videos"


def yes_no_from_logits(logits, tok):
    """Restrict next-token logits to the Yes/No families, return 'Yes'/'No'."""
    yes_ids = [tok(" Yes"), tok("Yes"), tok(" yes"), tok("yes")]
    no_ids = [tok(" No"), tok("No"), tok(" no"), tok("no")]
    ly = max(logits[0, i].item() for i in yes_ids if i is not None)
    ln = max(logits[0, i].item() for i in no_ids if i is not None)
    return "Yes" if ly >= ln else "No"


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_omni(args.model_path, attn_implementation="eager",
                                 device_map=args.device_map)
    patch_thinker_avcd(model)
    patch_efficient_encoders(model)
    cfg = model.thinker.config
    vid_id, aud_id = cfg.video_token_id, cfg.audio_token_id

    def tok(s):
        ids = processor.tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    split_df = pd.read_csv(args.split_csv, dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    if args.split.upper() == "FULL":
        df = split_df.reset_index(drop=True)
    elif args.split.upper() == "DEV":
        df = split_df[split_df.split == "DEV"].reset_index(drop=True)
    else:
        df = split_df[split_df.split != "DEV"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"AVCD AVHBench: split={args.split} n={len(df)}", flush=True)

    rows = []
    fail = 0
    t0 = time.time()
    for _, r in tqdm(df.iterrows(), total=len(df), desc="avcd-avhbench"):
        vid = str(r["video_id"]).zfill(5)
        vp = Path(args.video_dir) / f"{vid}.mp4"
        if not vp.exists():
            fail += 1
            continue
        prompt = r["text"] + EX.YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        try:
            inputs, _ = prepare_inputs(processor, conv, "av", model.device, model.dtype)
            ids = inputs["input_ids"][0]
            vidx = torch.where(ids == vid_id)[0].tolist()
            aidx = torch.where(ids == aud_id)[0].tolist()
            set_spans(vidx, aidx, ids.shape[0])
            logits, _ = avcd_answer_logits(model, inputs, True, cd_alpha=args.cd_alpha)
            pred = yes_no_from_logits(logits, tok)
        except Exception as e:
            print(f"  [skip] {vid}: {type(e).__name__}: {e}", flush=True)
            fail += 1
            continue
        rows.append(dict(question_id=r["question_id"], task=r["task"],
                         label=r["label"], predicted=pred,
                         correct=int(pred == r["label"])))
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.tag}_{args.split}.csv", index=False)
            acc = pd.DataFrame(rows).correct.mean()
            print(f"  {len(rows)}/{len(df)}  acc={acc*100:.2f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"{args.tag}_{args.split}.csv", index=False)
    print(f"\n=== AVCD AVHBench {args.split}  n={len(out)}  failures={fail}")
    print(f"  overall = {out.correct.mean()*100:.2f}%")
    for t, a in out.groupby("task").correct.mean().items():
        print(f"    {t:<40s} {a*100:.2f}%")
    print(f"saved: {OUT_DIR / (args.tag + '_' + args.split + '.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="FULL")
    p.add_argument("--split_csv", default=str(DEFAULT_SPLIT))
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--cd_alpha", type=float, default=2.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default="avcd_avhbench")
    main(p.parse_args())
