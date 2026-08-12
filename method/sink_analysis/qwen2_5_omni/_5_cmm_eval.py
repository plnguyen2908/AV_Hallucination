"""_5_cmm_eval.py — CMM (Curse of Multi-Modalities) hallucination eval.

2400 yes/no probing questions (1200 yes / 1200 no) over 6 sub-categories.
Modality per item: visual (mp4), audio (wav), visual+audio (mp4 + separate wav).

Official scoring (calculate_score.py, verbatim): generate free text, check
first 5 chars of the prediction:
  PA (Perception Accuracy)      = correct on "yes" questions
  HR (Hallucination Resistance) = correct on "no" questions
  overall accuracy              = answer in pred[:5]
  reported score                = mean(PA, HR)

Methods: baseline | ours | avcd | asd | mad  (--method).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_cmm_eval.py --method ours
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

DATA = _REPO / "data/CMM"
JSON = DATA / "all_data_final_reorg.json"
OUT_DIR = _REPO / "results/qwen2_5_omni/cmm"
SUFFIX = " Answer with yes or no."
HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"

MOD_MAP = {"visual": "v", "audio": "a", "visual+audio": "av"}
ROUTE = {"v": "VISUAL", "a": "AUDIO", "av": "AV"}


def fix_path(p):
    if not p:
        return None
    return str(DATA / p.replace("./reorg_raw_files", "reorg_raw_files"))


_MUX_DIR = DATA / "_muxed"


def mux_av(video_mp4, audio_wav):
    """CMM AV mp4s carry no audio track (audio is in a separate wav). MAD's
    av branch reads audio from the video, so produce a muxed mp4 (cached)."""
    import subprocess
    _MUX_DIR.mkdir(exist_ok=True)
    out = _MUX_DIR / (Path(video_mp4).stem + "_mux.mp4")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-i", video_mp4, "-i", audio_wav,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)


def build_conv(row, max_pixels):
    mt = MOD_MAP[row["modality"]]
    media = []
    if mt in ("v", "av"):
        media.append({"type": "video", "video": fix_path(row["video_path"]),
                      "fps": 1.0, "max_pixels": max_pixels})
    if mt in ("a", "av"):
        media.append({"type": "audio", "audio": fix_path(row["audio_path"])})
    content = media + [{"type": "text", "text": row["question"] + SUFFIX}]
    return [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ], mt


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


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.load(open(args.json or JSON))
    if args.limit:
        data = data[: args.limit]

    eager = args.method in ("ours", "avcd", "asd", "mad")
    model, processor = load_omni(args.model_path,
                                 attn_implementation="eager" if eager else "sdpa",
                                 device_map=args.device_map)
    from _5_efficient_encoders import patch_efficient_encoders
    patch_efficient_encoders(model)

    cfg = model.thinker.config
    vid_id, aud_id = cfg.video_token_id, cfg.audio_token_id

    # method-specific patching
    if args.method == "ours":
        IV.patch_qwen_attention(model)
        IV.clear_intervention()
        layers = EX.thinker_layers(model)
        eps_norm = EX._thinker_rms_eps(model)
        audio_enc, visual_enc = EX._resolve_encoders(model)
        d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)
        inert = EX.load_head_sets(HEADS)["Inert"]
        l2h = EX.heads_by_layer(inert, "full")
        # per-modality gamma (default: flat g_base for all); flat per-layer
        # schedule kept identical across all configs.
        gmap = {"a": args.gamma_a if args.gamma_a is not None else args.g_base,
                "v": args.gamma_v if args.gamma_v is not None else args.g_base,
                "av": args.gamma_av if args.gamma_av is not None else args.g_base}
        # tau_sink override (AVHBench winner used 15; CMM default was 20).
        EX.TAU_SINK = args.tau_sink
        # sink_mask type (llm_emerged default; 'all' = wider llm|prop).
        EX._RUNTIME_ARGS = type("A", (), dict(sink_mask=args.sink_mask))()
        # gamma SCHEDULING: per-layer gamma = g_base * shape[routed_modality]
        # (matches _5_explore's scheduled path). The paper's method is scheduled.
        sched = None
        if args.gamma_schedule_npz:
            import numpy as np
            z = np.load(args.gamma_schedule_npz)
            sched = {"a": z["shape_a"], "v": z["shape_v"], "av": z["shape_av"]}
        print(f"[ours config] tau_sink={args.tau_sink} sink_mask={args.sink_mask} "
              f"gamma a/v/av={gmap['a']}/{gmap['v']}/{gmap['av']} "
              f"schedule={'ON:'+Path(args.gamma_schedule_npz).name if sched else 'OFF(flat)'}",
              flush=True)
    elif args.method == "avcd":
        from _5_avcd_qwen import patch_thinker_avcd
        patch_thinker_avcd(model)
    elif args.method == "asd":
        import _5_asd_qwen as ASD
        ASD.patch_thinker_asd(model)
        ASD._ASD["alpha"] = args.asd_alpha

    rows = []
    fail = 0
    t0 = time.time()
    for i, r in enumerate(tqdm(data, desc=f"cmm-{args.method}")):
        conv, mt = build_conv(r, args.max_pixels)
        try:
            inputs = make_inputs(processor, model, conv)
            if args.method == "baseline":
                pred = gen_text(model, processor, inputs)
            elif args.method == "ours":
                masks, S = EX.compute_per_layer_sink_masks(
                    model, processor, conv, False, layers, audio_enc, visual_enc,
                    eps_norm, d_sink_t, ROUTE[mt])
                iv_cfg = dict(layer_to_heads=l2h, layer_to_key_mask=masks,
                              mode="boost", gamma=gmap[mt])
                if sched is not None:
                    # per-layer gamma = g_base * shape[routed modality]
                    iv_cfg["gamma_schedule"] = (args.g_base * sched[mt]).tolist()
                IV.set_intervention(iv_cfg)
                try:
                    pred = gen_text(model, processor, inputs)
                finally:
                    IV.clear_intervention()
            elif args.method == "avcd":
                from _5_avcd_qwen import set_spans, avcd_answer_logits
                ids = inputs["input_ids"][0]
                vidx = torch.where(ids == vid_id)[0].tolist()
                aidx = torch.where(ids == aud_id)[0].tolist()
                set_spans(vidx, aidx, ids.shape[0])
                logits, _ = avcd_answer_logits(model, inputs, False, cd_alpha=args.cd_alpha)
                tok = int(logits[0].argmax().item())
                pred = processor.tokenizer.decode([tok]).strip()
            elif args.method == "asd":
                import _5_asd_qwen as ASD
                ids = inputs["input_ids"][0]
                vidx = torch.where(ids == vid_id)[0].tolist()
                aidx = torch.where(ids == aud_id)[0].tolist()
                if not vidx or not aidx:
                    # ASD needs BOTH modalities for cross-modal sinks; on
                    # single-modality items it reduces to baseline generation.
                    pred = gen_text(model, processor, inputs)
                else:
                    ASD.clear(); ASD._ASD["sink_cols"] = None; ASD._ASD["cross"] = None
                    with torch.inference_mode():
                        out = model.thinker(**inputs, use_audio_in_video=False,
                                            use_cache=False, output_hidden_states=True)
                    sinks = ASD.sinks_from_hidden(out.hidden_states); del out
                    if not sinks:
                        pred = gen_text(model, processor, inputs)
                    else:
                        ASD._ASD["sink_cols"] = sinks; ASD.reset("collect")
                        with torch.inference_mode():
                            model.thinker(**inputs, use_audio_in_video=False, use_cache=False)
                        cross = ASD.compute_cross_modal_sinks(sinks, vidx, aidx, args.asd_mds_thr)
                        ASD.clear(); ASD._ASD["cross"] = cross if cross else None
                        ASD.reset("boost")
                        try:
                            pred = gen_text(model, processor, inputs)
                        finally:
                            ASD.clear(); ASD._ASD["cross"] = None; ASD._ASD["sink_cols"] = None
            elif args.method == "mad":
                if r["modality"] != "visual+audio":
                    # MAD's multi-branch contrast needs both modalities; on
                    # single-modality items it reduces to baseline generation.
                    pred = gen_text(model, processor, inputs)
                else:
                    from _5_avspeaker_mad import mad_decode
                    # CMM stores audio in a separate wav (mp4 has no audio track);
                    # MAD's av branch needs audio IN the video, so mux on the fly.
                    muxed = mux_av(fix_path(r["video_path"]), fix_path(r["audio_path"]))
                    pred = mad_decode(model, processor, r["question"] + SUFFIX,
                                      av_path=muxed,
                                      audio_path=fix_path(r["audio_path"]),
                                      visual_path=fix_path(r["video_path"]),
                                      gamma=args.mad_gamma, max_new_tokens=1,  # official
                                      # (eval_batch_cmm_mad.py); 2 tokens made
                                      # MAD emit degenerate "NoNo" on 48 items
                                      max_pixels=args.max_pixels)
            else:
                raise SystemExit(f"method {args.method} not wired")
        except Exception as e:
            print(f"  [skip] {r['question'][:30]}: {type(e).__name__}: {e}", flush=True)
            fail += 1
            continue
        p5 = pred.strip().lower()[:5]
        ans = r["answer"].strip().lower()
        # idx = position in all_data_final_reorg.json. Without it, runs that
        # skip different items (e.g. OOM on long clips at a smaller GPU count)
        # cannot be aligned row-by-row for a matched comparison.
        rows.append(dict(idx=i, sub_category=r["sub_category"], modality=r["modality"],
                         answer=ans, pred=pred, correct=int(ans in p5),
                         is_yes=int("yes" in ans[:5]),
                         pred_yes=int("yes" in p5), pred_no=int("no" in p5)))
        if len(rows) % 200 == 0:
            _save(rows, args)
            print(f"  {len(rows)}/{len(data)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    _save(rows, args)
    _report(rows, args, fail, time.time() - t0)


def _save(rows, args):
    pd.DataFrame(rows).to_csv(OUT_DIR / f"cmm_{args.tag or args.method}.csv", index=False)


def _report(rows, args, fail, dt):
    d = pd.DataFrame(rows)
    yes = d[d.is_yes == 1]; no = d[d.is_yes == 0]
    pa = yes.pred_yes.mean() if len(yes) else float("nan")
    hr = no.pred_no.mean() if len(no) else float("nan")
    print(f"\n=== CMM {args.method}  n={len(d)}  failures={fail}  ({dt/60:.1f} min)")
    print(f"  PA (perception)          = {pa*100:.2f}%")
    print(f"  HR (halluc. resistance)  = {hr*100:.2f}%")
    print(f"  score = mean(PA,HR)      = {(pa+hr)/2*100:.2f}%")
    print("  per sub_category (PA / HR):")
    for sc, g in d.groupby("sub_category"):
        y = g[g.is_yes == 1]; n = g[g.is_yes == 0]
        print(f"    {sc:32s} {y.pred_yes.mean()*100:6.1f} / {n.pred_no.mean()*100:6.1f}")
    print(f"saved: {OUT_DIR / ('cmm_' + (args.tag or args.method) + '.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["baseline", "ours", "avcd", "asd", "mad"])
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--max_pixels", type=int, default=360 * 420)
    p.add_argument("--g_base", type=float, default=3.0)
    p.add_argument("--gamma_a", type=float, default=None)
    p.add_argument("--gamma_v", type=float, default=None)
    p.add_argument("--gamma_av", type=float, default=None)
    p.add_argument("--tau_sink", type=float, default=20.0)
    p.add_argument("--sink_mask", default="llm_emerged", choices=["llm_emerged", "all", "prop"])
    p.add_argument("--gamma_schedule_npz", default=None,
                   help="per-layer gamma = g_base * shape[routed_modality] (the paper's scheduled method)")
    p.add_argument("--cd_alpha", type=float, default=2.5)
    p.add_argument("--asd_alpha", type=float, default=0.2)
    p.add_argument("--asd_mds_thr", type=float, default=0.3)
    p.add_argument("--mad_gamma", type=float, default=2.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--tag", default=None)
    main(p.parse_args())
