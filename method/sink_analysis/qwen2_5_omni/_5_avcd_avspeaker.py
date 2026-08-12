"""_5_avcd_avspeaker.py — AVCD baseline on AV-SpeakerBench (A/B/C/D).

Reuses _5_avspeaker_eval's official prompt (build_question), the n=2065
duration filter (≤15s), and letter scoring; only generation is replaced by
the AVCD contrastive forward, scored over the option-letter token ids.

Official config: --video_fps 1.0 (no resize), --max_duration_s 15.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_avcd_avspeaker.py \
      --max_duration_s 15 --video_fps 1.0 --cd_alpha 2.5 --tag avcd
"""
import argparse
import re
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
import _5_avspeaker_eval as AVS  # noqa: E402
from _5_avcd_qwen import patch_thinker_avcd, set_spans, avcd_answer_logits  # noqa: E402
from _5_efficient_encoders import patch_efficient_encoders  # noqa: E402

DEFAULT_DATA = _REPO / "data/AV_SpeakerBench"
OUT_DIR = _REPO / "results/qwen2_5_omni/stage5_intervention/avspeaker"


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_omni(args.model_path, attn_implementation="eager",
                                 device_map=args.device_map)
    patch_thinker_avcd(model)
    patch_efficient_encoders(model)
    cfg = model.thinker.config
    vid_id, aud_id = cfg.video_token_id, cfg.audio_token_id

    df = pd.read_csv(args.test_csv)
    if args.max_duration_s and args.max_duration_s > 0:
        def _dur(p):
            m = re.search(r'_(\d+)_(\d+)\.mp4$', p or "")
            return int(m.group(2)) - int(m.group(1)) if m else None
        df["dur_s"] = df["audio_visual_path"].apply(_dur)
        n0 = len(df)
        df = df[df["dur_s"].notna() & (df["dur_s"] <= args.max_duration_s)].reset_index(drop=True)
        print(f"[duration filter] kept {len(df)}/{n0} clips ≤ {args.max_duration_s}s", flush=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"AVCD AV-SpeakerBench: n={len(df)}", flush=True)

    rows = []
    fail = 0
    t0 = time.time()
    for _, r in tqdm(df.iterrows(), total=len(df), desc="avcd-avs"):
        vp = Path(args.av_dir) / r["audio_visual_path"]
        if not vp.exists():
            fail += 1
            continue
        prompt = AVS.build_question(r)
        conv = build_conversation(str(vp), prompt, "av", video_fps=args.video_fps)
        try:
            inputs, _ = prepare_inputs(processor, conv, "av", model.device, model.dtype)
            ids = inputs["input_ids"][0]
            vidx = torch.where(ids == vid_id)[0].tolist()
            aidx = torch.where(ids == aud_id)[0].tolist()
            set_spans(vidx, aidx, ids.shape[0])
            logits, _ = avcd_answer_logits(model, inputs, True, cd_alpha=args.cd_alpha)
            # Official scoring: decode AVCD's argmax token, parse the letter,
            # score 0 if no valid A/B/C/D letter is emitted (matches baseline).
            tok = int(logits[0].argmax().item())
            pred = AVS.parse_letter(processor.tokenizer.decode([tok]))
        except Exception as e:
            print(f"  [skip] {r['question_id']}: {type(e).__name__}: {e}", flush=True)
            fail += 1
            continue
        rows.append(dict(question_id=r["question_id"], category=r["category"],
                         sub_category=r["sub_category"], answer=r["answer"],
                         predicted=pred, correct=int(pred == str(r["answer"]).strip())))
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"avspeaker_{args.tag}.csv", index=False)
            print(f"  {len(rows)}/{len(df)}  acc={pd.DataFrame(rows).correct.mean()*100:.2f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"avspeaker_{args.tag}.csv", index=False)
    print(f"\n=== AVCD AV-SpeakerBench  n={len(out)}  failures={fail}")
    print(f"  overall = {out.correct.mean()*100:.2f}%")
    for c, a in out.groupby("category").correct.mean().items():
        print(f"    {c:<18s} {a*100:.2f}%")
    print(f"saved: {OUT_DIR / ('avspeaker_' + args.tag + '.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default=str(DEFAULT_DATA / "test.csv"))
    p.add_argument("--av_dir", default=str(DEFAULT_DATA))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--cd_alpha", type=float, default=2.5)
    p.add_argument("--video_fps", type=float, default=1.0)
    p.add_argument("--max_duration_s", type=int, default=15)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default="avcd")
    main(p.parse_args())
