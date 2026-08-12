"""_5_videohallucer_eval.py — VideoHallucer (video-only hallucination) eval.

VideoHallucer (patrick-tssn): 5 subtasks, each item = a `basic` yes/no Q and a
`hallucination` yes/no Q on the same video. Official metric: a sample is correct
only if BOTH basic AND hallucination are answered correctly ("Overall"). Also
report basic_acc, halluc_acc, and yes/no bias.

VIDEO-ONLY (no audio). The AV cross-modal methods therefore degenerate:
  - asd / mad: no audio -> no cross-modal contrast -> fall back to baseline.
  - avcd: partial video-vs-language contrast only.
  - ours: routes VISUAL, boosts video-sink attention (meaningful, like ActivityNet).
So the meaningful comparison here is baseline vs ours; avcd/asd/mad are reported
for completeness and largely reproduce baseline.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_videohallucer_eval.py --method ours
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))

from utils import load_omni, OMNI_SYSTEM_PROMPT  # noqa: E402
import _5_explore as EX  # noqa: E402
import _5_intervene as IV  # noqa: E402

DATA = _REPO / "data/VideoHallucer"
OUT_DIR = _REPO / "results/qwen2_5_omni/videohallucer"
SUFFIX = " Answer with yes or no."
HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"

SUBTASKS = {
    "object_relation": "object_relation/object_relation.json",
    "temporal": "temporal/temporal.json",
    "semantic_detail": "semantic_detail/semantic_detail.json",
    "external_factual": "external_factual/external_factual.json",
    "external_nonfactual": "external_nonfactual/external_nonfactual.json",
}
VID_SUBDIR = {  # where each subtask's videos live (dir next to the json)
    k: str(DATA / Path(v).parent / "videos") for k, v in SUBTASKS.items()
}


_DUR_CACHE = {}


def _duration(video_path):
    if video_path not in _DUR_CACHE:
        import subprocess
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", video_path], capture_output=True, text=True)
        try:
            _DUR_CACHE[video_path] = float(r.stdout.strip())
        except Exception:
            _DUR_CACHE[video_path] = 20.0
    return _DUR_CACHE[video_path]


def build_conv(video_path, question, max_pixels, max_frames=16):
    # Whole-video frame cap: long clips (median 133s) at fps=1 blow past the
    # eager-attention token ceiling. Sample <= max_frames uniformly so ours
    # (eager) fits and generation is fast. dur<=max_frames -> fps=1.
    dur = _duration(video_path)
    fps = 1.0 if dur <= max_frames else max_frames / dur
    return [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": question + SUFFIX},
        ]},
    ]


def make_inputs(processor, model, conv):
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    return inputs.to(model.device).to(model.dtype)


def gen_text(model, processor, inputs, max_new=12):
    with torch.inference_mode():
        ids = model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                             do_sample=False, max_new_tokens=max_new)
    gen = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def find_video(subtask, fname):
    for cand in [Path(VID_SUBDIR[subtask]) / fname,
                 DATA / Path(SUBTASKS[subtask]).parent / fname,
                 DATA / fname]:
        if cand.exists():
            return str(cand)
    return None


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = args.subtasks.split(",") if args.subtasks else list(SUBTASKS)

    eager = args.method in ("ours", "avcd", "asd", "mad")
    model, processor = load_omni(args.model_path,
                                 attn_implementation="eager" if eager else "sdpa",
                                 device_map=args.device_map)
    from _5_efficient_encoders import patch_efficient_encoders
    patch_efficient_encoders(model)
    cfg = model.thinker.config
    vid_id, aud_id = cfg.video_token_id, cfg.audio_token_id

    if args.method == "ours":
        IV.patch_qwen_attention(model); IV.clear_intervention()
        layers = EX.thinker_layers(model)
        eps_norm = EX._thinker_rms_eps(model)
        audio_enc, visual_enc = EX._resolve_encoders(model)
        d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)
        l2h = EX.heads_by_layer(EX.load_head_sets(HEADS)["Inert"], "full")
    elif args.method == "avcd":
        from _5_avcd_qwen import patch_thinker_avcd
        patch_thinker_avcd(model)
    elif args.method == "asd":
        import _5_asd_qwen as ASD
        ASD.patch_thinker_asd(model); ASD._ASD["alpha"] = args.asd_alpha

    def answer_one(video_path, question):
        conv = build_conv(video_path, question, args.max_pixels)
        inputs = make_inputs(processor, model, conv)
        if args.method == "baseline":
            return gen_text(model, processor, inputs)
        if args.method == "ours":
            masks, S = EX.compute_per_layer_sink_masks(
                model, processor, conv, False, layers, audio_enc, visual_enc,
                eps_norm, d_sink_t, "VISUAL")
            IV.set_intervention(dict(layer_to_heads=l2h, layer_to_key_mask=masks,
                                     mode="boost", gamma=args.g_base))
            try:
                return gen_text(model, processor, inputs)
            finally:
                IV.clear_intervention()
        if args.method == "avcd":
            from _5_avcd_qwen import set_spans, avcd_answer_logits
            ids = inputs["input_ids"][0]
            vidx = torch.where(ids == vid_id)[0].tolist()
            aidx = torch.where(ids == aud_id)[0].tolist()
            set_spans(vidx, aidx, ids.shape[0])
            logits, _ = avcd_answer_logits(model, inputs, False, cd_alpha=args.cd_alpha)
            return processor.tokenizer.decode([int(logits[0].argmax())]).strip()
        if args.method == "asd":
            import _5_asd_qwen as ASD
            ids = inputs["input_ids"][0]
            vidx = torch.where(ids == vid_id)[0].tolist()
            aidx = torch.where(ids == aud_id)[0].tolist()
            if not vidx or not aidx:  # video-only -> no cross-modal sinks -> baseline
                return gen_text(model, processor, inputs)
            ASD.clear(); ASD._ASD["sink_cols"] = None; ASD._ASD["cross"] = None
            with torch.inference_mode():
                out = model.thinker(**inputs, use_audio_in_video=False,
                                    use_cache=False, output_hidden_states=True)
            sinks = ASD.sinks_from_hidden(out.hidden_states); del out
            if not sinks:
                return gen_text(model, processor, inputs)
            ASD._ASD["sink_cols"] = sinks; ASD.reset("collect")
            with torch.inference_mode():
                model.thinker(**inputs, use_audio_in_video=False, use_cache=False)
            cross = ASD.compute_cross_modal_sinks(sinks, vidx, aidx, args.asd_mds_thr)
            ASD.clear(); ASD._ASD["cross"] = cross if cross else None; ASD.reset("boost")
            try:
                return gen_text(model, processor, inputs)
            finally:
                ASD.clear(); ASD._ASD["cross"] = None; ASD._ASD["sink_cols"] = None
        if args.method == "mad":
            # MAD needs audio -> video-only -> baseline generation.
            return gen_text(model, processor, inputs)
        raise SystemExit(args.method)

    def yn(text):
        t = text.strip().lower()
        return "yes" if "yes" in t[:5] else ("no" if "no" in t[:5] else "?")

    rows = []
    fail = 0
    t0 = time.time()
    for st in subs:
        items = json.load(open(DATA / SUBTASKS[st]))
        if args.limit:
            items = items[: args.limit]
        for it in tqdm(items, desc=f"vh-{args.method}-{st}"):
            try:
                bv = find_video(st, it["basic"]["video"])
                hv = find_video(st, it["hallucination"]["video"])
                if not bv or not hv:
                    fail += 1; continue
                bp = yn(answer_one(bv, it["basic"]["question"]))
                hp = yn(answer_one(hv, it["hallucination"]["question"]))
            except Exception as e:
                print(f"  [skip] {st}: {type(e).__name__}: {e}", flush=True)
                fail += 1; continue
            b_ok = int(bp == it["basic"]["answer"].strip().lower())
            h_ok = int(hp == it["hallucination"]["answer"].strip().lower())
            rows.append(dict(subtask=st, basic_ans=it["basic"]["answer"],
                             halluc_ans=it["hallucination"]["answer"],
                             basic_pred=bp, halluc_pred=hp,
                             basic_ok=b_ok, halluc_ok=h_ok, both_ok=int(b_ok and h_ok)))
            if len(rows) % 100 == 0:
                pd.DataFrame(rows).to_csv(OUT_DIR / f"vh_{args.tag or args.method}.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / f"vh_{args.tag or args.method}.csv", index=False)
    print(f"\n=== VideoHallucer {args.method}  n={len(d)} pairs  failures={fail}  ({(time.time()-t0)/60:.1f} min)")
    print(f"  basic_acc = {d.basic_ok.mean()*100:.2f}   halluc_acc = {d.halluc_ok.mean()*100:.2f}")
    print(f"  OVERALL (both correct) = {d.both_ok.mean()*100:.2f}")
    print(f"  yes-bias: basic preds {(d.basic_pred=='yes').mean()*100:.0f}% yes | halluc {(d.halluc_pred=='yes').mean()*100:.0f}% yes")
    print("  per subtask (overall):")
    for st, g in d.groupby("subtask"):
        print(f"    {st:22s} {g.both_ok.mean()*100:.2f}  (basic {g.basic_ok.mean()*100:.1f}/halluc {g.halluc_ok.mean()*100:.1f})")
    print(f"saved: {OUT_DIR / ('vh_' + (args.tag or args.method) + '.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["baseline", "ours", "avcd", "asd", "mad"])
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--max_pixels", type=int, default=360 * 420)
    p.add_argument("--g_base", type=float, default=3.0)
    p.add_argument("--cd_alpha", type=float, default=2.5)
    p.add_argument("--asd_alpha", type=float, default=0.2)
    p.add_argument("--asd_mds_thr", type=float, default=0.3)
    p.add_argument("--subtasks", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default=None)
    main(p.parse_args())
