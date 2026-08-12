"""Unified WorldSense evaluation harness for Qwen2.5-Omni.

Runs a chosen method over WorldSense MCQ (official VLMEvalKit protocol: single
A/B/C/D letter, exact-match scoring, get_dimension_rating) and writes per-row
predictions + a dimension-rating summary.

Methods (backends):
  baseline : plain Qwen2.5-Omni greedy (SDPA — no length ceiling)
  ours     : sink-boost intervention (eager thinker + efficient encoders)
  avcd     : AVCD reimplemented for Qwen (_5_avcd_qwen)
  asd,mad  : TODO (official crossmodal-hub / top-yun implementations)

Token budget: video frames bounded by --nframe (constant regardless of length);
audio bounded by --trim_s (trim the clip to its first trim_s seconds; 0 = full).
This keeps every clip under the eager ~5-10k-token ceiling
(see results/qwen2_5_omni/worldsense/LENGTH_FINDINGS.md).

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_worldsense_eval.py \
    --method avcd --nframe 16 --trim_s 60 --limit 50
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))
from utils import load_omni, prepare_inputs, OMNI_SYSTEM_PROMPT, find_modality_spans  # noqa
from qwen_omni_utils import process_mm_info  # noqa
from _ws_official_score import extract_characters_regex, get_dimension_rating  # official

QA_JSON = ("/nobackup2/le/.cache/hub/datasets--honglyhly--WorldSense/snapshots/"
           "49df5aa9d62c4900bca7e86c3a9d476122e47fcc/worldsense_qa.json")
VIDEOS = _REPO / "data/WorldSense/videos"
TRIM_CACHE = Path("/tmp/claude-10526/-nobackup2-le-AV-Hallucination/"
                  "3c6c1667-33a0-48f1-9699-6431bef97b63/scratchpad/ws_trim")
OUT_DIR = _REPO / "results/qwen2_5_omni/worldsense"

# Official WorldSense prompt strings (VLMEvalKit worldsense.py, verbatim).
BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'
FRAMES_TMPL_AUDIO = ("\nThese are the frames of a video and the corresponding audio. "
                     "Select the best answer to the following multiple-choice question "
                     "based on the video. Respond with only the letter (A, B, C, or D) "
                     "of the correct option.\n")


# ----------------------------- data ---------------------------------------
# --- sampling rule (user-specified) ---------------------------------------
#   video: <=FPS_MAX_DUR s  -> fps=1 (full coverage)
#          > FPS_MAX_DUR s   -> UNIFORM_NFRAMES frames uniformly over the WHOLE clip
#   audio: separate wav (official WorldSense passes audio as a separate item),
#          capped at AUDIO_CAP_S s to keep total tokens under the eager ceiling.
#   use_audio_in_video is always False (audio is a separate modality item).
import re as _re
FPS_MAX_DUR = 20
UNIFORM_NFRAMES = 20
AUDIO_CAP_S = 90


def load_rows(limit=0, only_present=True):
    d = json.load(open(QA_JSON))
    rows = []
    for vid, e in d.items():
        vdur = int(_re.match(r"(\d+)", str(e["video_duration"])).group(1))
        for tk in [k for k in e if k.startswith("task")]:
            t = e[tk]
            rows.append(dict(
                index=len(rows), video=vid, duration=e["duration"], vdur=vdur,
                domain=e["domain"], sub_category=e["sub_category"],
                audio_class=str(e["audio_class"]), task_domain=t["task_domain"],
                task_type=t["task_type"], question=t["question"],
                candidates=t["candidates"], answer=t["answer"]))
    if only_present:
        rows = [r for r in rows if (VIDEOS / f"{r['video']}.mp4").exists()]
    if limit:
        rows = rows[:limit]
    return rows


def _video_dict(row, max_pixels):
    vp = VIDEOS / f"{row['video']}.mp4"
    d = {"type": "video", "video": str(vp), "max_pixels": max_pixels}
    if row["vdur"] <= FPS_MAX_DUR:
        d["fps"] = 1.0
    else:
        d["nframes"] = UNIFORM_NFRAMES
    return d


def audio_wav(row):
    """Separate audio wav, capped at min(vdur, AUDIO_CAP_S) s."""
    cap = min(int(row["vdur"]), AUDIO_CAP_S)
    TRIM_CACHE.mkdir(parents=True, exist_ok=True)
    out = TRIM_CACHE / f"{row['video']}_a{cap}s.wav"
    if not out.exists():
        src = VIDEOS / f"{row['video']}.mp4"
        subprocess.run(["ffmpeg", "-y", "-t", str(cap), "-i", str(src), "-vn",
                        "-ac", "1", "-ar", "16000", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def build_conv(row, max_pixels, modality="av"):
    """Official WorldSense prompt ordering: SYS text -> video -> audio ->
    FRAMES_TMPL_AUDIO -> 'Question: {q}\\n{cands}\\nAnswer: '. modality selects the
    media items (av/v/a/t) for MAD; default 'av' = video + separate audio wav."""
    q = row["question"] + "\n" + "\n".join(row["candidates"])
    qa = "Question: {}\nAnswer: ".format(q)
    media = []
    if modality in ("av", "v"):
        media.append(_video_dict(row, max_pixels))
    if modality in ("av", "a"):
        media.append({"type": "audio", "audio": str(audio_wav(row))})
    content = [{"type": "text", "text": SYS}] + media + \
              [{"type": "text", "text": FRAMES_TMPL_AUDIO}, {"type": "text", "text": qa}]
    return [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def make_inputs(processor, model, row, max_pixels, modality="av"):
    conv = build_conv(row, max_pixels, modality)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    return inputs.to(model.device).to(model.dtype)


# scoring uses the official extract_characters_regex + get_dimension_rating
# (imported from _ws_official_score, verbatim from VLMEvalKit).


# --------------------------- backends -------------------------------------
class Backend:
    needs_eager = False

    def load(self, args):
        impl = "eager" if self.needs_eager else "sdpa"
        self.model, self.processor = load_omni(args.model_path, attn_implementation=impl,
                                               device_map=args.device_map)
        self.args = args

    def answer(self, row):
        raise NotImplementedError


class BaselineBackend(Backend):
    needs_eager = False  # sdpa thinker -> no length ceiling

    def load(self, args):
        super().load(args)
        # audio/vision SDPA encoders build a dense (N,N) mask -> OOM on long
        # audio; block-SDPA fixes it so baseline can run full-length clips.
        from _5_efficient_encoders import patch_efficient_encoders
        patch_efficient_encoders(self.model)

    def answer(self, row):
        inputs = make_inputs(self.processor, self.model, row, self.args.max_pixels)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                                      do_sample=False, max_new_tokens=8)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        txt = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return txt


class AVCDBackend(Backend):
    needs_eager = True

    def load(self, args):
        super().load(args)
        from _5_efficient_encoders import patch_efficient_encoders
        from _5_avcd_qwen import patch_thinker_avcd
        patch_thinker_avcd(self.model)
        patch_efficient_encoders(self.model)
        cfg = self.model.thinker.config
        self.vid_id, self.aud_id = cfg.video_token_id, cfg.audio_token_id

    def answer(self, row):
        from _5_avcd_qwen import set_spans, avcd_answer_logits
        inputs = make_inputs(self.processor, self.model, row, self.args.max_pixels)
        ids = inputs["input_ids"][0]
        vidx = torch.where(ids == self.vid_id)[0].tolist()
        aidx = torch.where(ids == self.aud_id)[0].tolist()
        set_spans(vidx, aidx, ids.shape[0])
        logits, _ = avcd_answer_logits(self.model, inputs, False, cd_alpha=self.args.cd_alpha)
        tok = int(logits.argmax(dim=-1).item())
        return self.processor.tokenizer.decode([tok]).strip()


class OursBackend(Backend):
    """Sink-boost intervention = the AVS/AVH reproduce config: boost_inert_content,
    508-inert head set, per-clip llm+prop sink masks (sink_mask=all), reverse
    per-layer gamma schedule blended by router probs, g_base=5. Reuses the
    validated _5_explore helpers. Router: WorldSense is all-AV, so default routed
    modality = AV (one-hot) -> schedule uses the av shape. Eager thinker."""
    needs_eager = True

    def load(self, args):
        super().load(args)
        import numpy as np
        import _5_explore as E
        import _5_intervene as IV
        from _5_efficient_encoders import patch_efficient_encoders
        self.E, self.IV = E, IV
        IV.patch_qwen_attention(self.model)
        patch_efficient_encoders(self.model)
        E._RUNTIME_ARGS = type("A", (), dict(
            sink_mask=args.sink_mask, asd=False, random_mask=False,
            include_text_sinks=False))()
        self.layers = E.thinker_layers(self.model)
        self.eps = E._thinker_rms_eps(self.model)
        self.aenc, self.venc = E._resolve_encoders(self.model)
        self.d_sink = torch.tensor(E.D_SINK, dtype=torch.long)
        self.head_sets = E.load_head_sets(Path(args.heads_csv))
        z = np.load(args.gamma_schedule_npz)
        self.sched = (z["shape_a"], z["shape_v"], z["shape_av"])
        self.g_base = args.g_base
        print(f"[ours] heads Inert={len(self.head_sets['Inert'])} "
              f"g_base={self.g_base} sink_mask={args.sink_mask}", flush=True)

    def answer(self, row):
        conv = build_conv(row, self.args.max_pixels)
        routed, pa, pv, pav = "AV", 0.0, 0.0, 1.0  # all-AV benchmark
        l2k, S = self.E.compute_per_layer_sink_masks(
            self.model, self.processor, conv, False, self.layers,
            self.aenc, self.venc, self.eps, self.d_sink, routed)
        heads = self.E.heads_for_variant_routing(self.head_sets, "boost_inert_content", routed)
        l2h = self.E.heads_by_layer(heads, "full")
        sa, sv, sav = self.sched
        cfg = dict(layer_to_heads=l2h, layer_to_key_mask=l2k, mode="boost", gamma=3.0)
        cfg["gamma_schedule"] = (self.g_base * (sa * pa + sv * pv + sav * pav)).tolist()
        self.IV.set_intervention(cfg)
        try:
            inputs = make_inputs(self.processor, self.model, row, self.args.max_pixels)
            with torch.inference_mode():
                ids = self.model.generate(**inputs, use_audio_in_video=False,
                                          return_audio=False, do_sample=False, max_new_tokens=8)
            gen = ids[:, inputs["input_ids"].shape[1]:]
            txt = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        finally:
            self.IV.clear_intervention()
        return txt


class MADBackend(Backend):
    """MAD (top-yun) multimodal contrastive decoding, official algorithm ported
    to the WorldSense prompt. Per question: modality probs from the av-context
    audio/video/both token logits, then combine last-token logits of the 4
    modality contexts (av/v/a/t) with MAD's adaptive alpha weights. Logit-space
    -> sdpa thinker (no length ceiling)."""
    needs_eager = False

    def load(self, args):
        super().load(args)
        from _5_efficient_encoders import patch_efficient_encoders
        patch_efficient_encoders(self.model)
        tok = self.processor.tokenizer
        self.idx_a = tok.encode("audio")[0]
        self.idx_v = tok.encode("video")[0]
        self.idx_b = tok.encode("both")[0]
        self.gamma = args.mad_gamma

    def answer(self, row):
        step = {}
        for key in ("av", "v", "a", "t"):
            inp = make_inputs(self.processor, self.model, row, self.args.max_pixels, modality=key)
            with torch.inference_mode():
                out = self.model.thinker(**inp, use_audio_in_video=False, use_cache=False)
            step[key] = out.logits[:, -1, :].detach()
            if key == "av":
                last = out.logits[0, -1, :]
                probs = torch.softmax(torch.tensor(
                    [last[self.idx_a].item(), last[self.idx_v].item(), last[self.idx_b].item()]), dim=0)
                p_a, p_v, p_b = probs.tolist()
            del out
        g = self.gamma
        w = {"av": 2 + 2 * g * p_b, "v": 1 - (p_b - p_v) * g,
             "a": 1 - (p_b - p_a) * g, "t": -(p_v + p_a) * g}
        combined = sum(w[k] * step[k].squeeze(0) for k in step)
        tok = int(torch.argmax(combined).item())
        return self.processor.tokenizer.decode([tok]).strip()


class ASDBackend(Backend):
    """Official ASD (crossmodal-hub) reconstruction: boost attention to
    cross-modal sink tokens by A + alpha|A| (alpha=0.2). 3 eager forwards per Q:
    hidden->sinks, attn->MDS/cross-modal, boost->generate."""
    needs_eager = True

    def load(self, args):
        super().load(args)
        import _5_asd_qwen as ASD
        from _5_efficient_encoders import patch_efficient_encoders
        self.ASD = ASD
        ASD.patch_thinker_asd(self.model)
        patch_efficient_encoders(self.model)
        cfg = self.model.thinker.config
        self.vid_id, self.aud_id = cfg.video_token_id, cfg.audio_token_id
        self.mds_thr = args.asd_mds_thr
        ASD._ASD["alpha"] = args.asd_alpha

    def _greedy(self, inputs):
        with torch.inference_mode():
            gids = self.model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                                       do_sample=False, max_new_tokens=8)
        gen = gids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def answer(self, row):
        ASD = self.ASD
        inputs = make_inputs(self.processor, self.model, row, self.args.max_pixels)
        ids = inputs["input_ids"][0]
        vidx = torch.where(ids == self.vid_id)[0].tolist()
        aidx = torch.where(ids == self.aud_id)[0].tolist()
        ASD.clear(); ASD._ASD["sink_cols"] = None; ASD._ASD["cross"] = None
        with torch.inference_mode():
            out = self.model.thinker(**inputs, use_audio_in_video=False, use_cache=False,
                                     output_hidden_states=True)
        sinks = ASD.sinks_from_hidden(out.hidden_states); del out
        if not sinks:
            return self._greedy(inputs)
        ASD._ASD["sink_cols"] = sinks
        ASD.reset("collect")
        with torch.inference_mode():
            self.model.thinker(**inputs, use_audio_in_video=False, use_cache=False)
        cross = ASD.compute_cross_modal_sinks(sinks, vidx, aidx, self.mds_thr)
        ASD.clear()
        ASD._ASD["cross"] = cross if cross else None
        ASD.reset("boost")
        try:
            txt = self._greedy(inputs)
        finally:
            ASD.clear(); ASD._ASD["cross"] = None; ASD._ASD["sink_cols"] = None
        return txt


BACKENDS = {"baseline": BaselineBackend, "avcd": AVCDBackend, "ours": OursBackend,
            "mad": MADBackend, "asd": ASDBackend}


# ----------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(BACKENDS))
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--device_map", default="balanced_low_0")
    # AVS-matched preprocessing: fps=1 (nframe=0), max_pixels=360x640, trim to 20s.
    ap.add_argument("--nframe", type=int, default=0, help="0 => fps=1 sampling (AVS-style)")
    ap.add_argument("--max_pixels", type=int, default=360 * 640)
    ap.add_argument("--trim_s", type=int, default=20, help="trim clip to first N s (like AVS <=15-20s)")
    ap.add_argument("--cd_alpha", type=float, default=2.5)
    # ours (sink-boost) config — AVS/AVH reproduce defaults (c508 reverse-sched g5)
    ap.add_argument("--heads_csv", default=str(_REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"))
    ap.add_argument("--gamma_schedule_npz", default=str(_REPO / "results/qwen2_5_omni/sink_analysis/gamma_schedules/rev_sched_common508.npz"))
    ap.add_argument("--g_base", type=float, default=5.0)
    ap.add_argument("--sink_mask", default="all")
    ap.add_argument("--mad_gamma", type=float, default=0.5)  # MAD official default
    ap.add_argument("--asd_alpha", type=float, default=0.2)  # ASD official default
    ap.add_argument("--asd_mds_thr", type=float, default=0.3)  # |MDS|<=thr => cross-modal
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    rows = load_rows(limit=args.limit)
    print(f"{len(rows)} questions (videos present); method={args.method} "
          f"nframe={args.nframe} trim_s={args.trim_s}", flush=True)
    backend = BACKENDS[args.method]()
    backend.load(args)

    tag = args.tag or f"{args.method}_nf{args.nframe}_trim{args.trim_s}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.time()
    fails = 0
    for row in tqdm(rows, desc=tag):
        try:
            raw = backend.answer(row)
        except RuntimeError as e:
            fails += 1
            raw = f"__ERR__{str(e)[:40]}"
            torch.cuda.empty_cache()
        pred = extract_characters_regex(raw)
        # official exact-match: 0/1 by letter; generation failures -> -1 (excluded)
        if str(raw).startswith("__ERR__"):
            score = -1
        else:
            score = int(pred == row["answer"])
        recs.append({**row, "prediction": raw, "pred_letter": pred, "score": score})

    df = pd.DataFrame(recs)
    csv = OUT_DIR / f"ws_{tag}.csv"
    df.to_csv(csv, index=False)
    rating = get_dimension_rating(df)  # official VLMEvalKit scoring
    json.dump(rating, open(OUT_DIR / f"ws_{tag}_rating.json", "w"), indent=2)
    valid = df[df["score"] >= 0]
    print(f"\n=== {tag} ===")
    print(f"overall: {rating['overall']['overall']}  | n={len(df)} "
          f"valid={len(valid)} failed={fails} | {(time.time()-t0)/60:.1f} min")
    print("by task_domain:", rating['overall']['task_domain'])
    print("by audio_class:", rating['overall']['audio_class'])
    print("wrote", csv)


if __name__ == "__main__":
    main()
