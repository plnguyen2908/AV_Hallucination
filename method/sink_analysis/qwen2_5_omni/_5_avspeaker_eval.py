"""AV-SpeakerBench eval driver for Stage 5 interventions.

Loads `data/AV_SpeakerBench/test.csv`, evaluates Qwen2.5-Omni in MCQ
mode (A/B/C/D) under the same intervention pipeline as `_5_explore.py`.

Routing rule (from category column):
    Audio-centric    -> AUDIO
    Visual-centric   -> VISUAL
    Speaker-centric  -> AV

Writes:
    <output_dir>/avspeaker_<tag>.csv
    appends to <output_dir>/avspeaker_log.md
"""
import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/sink_analysis/qwen2_5_omni"))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

# Reuse infra from the AVHBench explore script
import _5_explore as EX
import _5_intervene as IV
import _5_phaseB_prep as PB  # AVHBench text-only constrained-logit router
from utils import build_conversation, load_omni, thinker_layers, OMNI_SYSTEM_PROMPT  # noqa: E402

DEFAULT_DATA = _REPO / "data/AV_SpeakerBench"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention/avspeaker"

CATEGORY_TO_MODALITY = {
    "Audio-centric":   "AUDIO",
    "Visual-centric":  "VISUAL",
    "Speaker-centric": "AV",
}

_SCHED = None  # per-layer gamma schedule (shape_a/shape_v/shape_av), loaded in main()
_ROUTER_LABEL_IDS = None  # [id(" Audio"), id(" Visual"), id(" AV")], set in main()

# Modality order used everywhere below: index 0=AUDIO, 1=VISUAL, 2=AV.
_ROUTE_ORDER = ["AUDIO", "VISUAL", "AV"]


def precompute_router_label_ids(processor):
    """Resolve the single-token ids for the 3 router labels at the assistant
    position — identical to _5_phaseB_prep.run_router_constrained."""
    tok = processor.tokenizer
    conv = [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text",
            "text": PB.ROUTER_PROMPT_TPL_V2.format(q="Does X exist?")}]},
    ]
    chat = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(chat, list):
        chat = chat[0]
    base = tok(chat, add_special_tokens=False).input_ids
    ids = []
    for label in PB.ROUTER_LABELS:
        full = tok(chat + label, add_special_tokens=False).input_ids
        if len(full) - len(base) != 1:
            raise SystemExit(f"router label {label!r} is not single-token; abort")
        ids.append(full[-1])
    print(f"[router] label token ids {dict(zip(PB.ROUTER_LABELS, ids))}", flush=True)
    return ids


def route_probs(model, processor, question_text, label_ids):
    """AVHBench-style TEXT-ONLY constrained-logit router. Returns
    (p_audio, p_visual, p_av) numpy floats summing to 1."""
    prompt = PB.ROUTER_PROMPT_TPL_V2.format(q=question_text)
    conv = [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        out = model.thinker(**inputs, output_attentions=False,
                             return_dict=True, use_cache=False)
    last = out.logits[0, -1, :].float()
    probs = torch.softmax(last[label_ids], dim=-1)
    return probs.cpu().numpy()

# Match the official AV-SpeakerBench eval (plnguyen2908/AV-SpeakerBench).
# Prompt: "Select the best answer ... The best answer is:" — Qwen-Omni
# completes that pattern naturally with just a letter. Extract via the
# same `extract_characters_regex` they use.
ANSWER_PREFIXES = [
    "The best answer is", "The correct answer is", "The answer is",
    "The answer", "The best option is", "The correct option is",
    "Best answer:", "Best option:", "Answer:", "Option:",
    "The correct answer", "The correct option", "Based",
    "Correct answer", "☞", "<|im_end|>",
]
PUNCT_RE = re.compile(r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]")


def parse_letter(text):
    if not text or text is None: return "?"
    s = text.strip()
    for pref in ANSWER_PREFIXES:
        s = s.replace(pref, "")
    s = PUNCT_RE.sub(" ", s)
    for tok in s.split():
        if tok in {"A", "B", "C", "D", "E"}:
            return tok[0]
    return "?"


def build_question(row) -> str:
    """Match the official AV-SpeakerBench prompt template."""
    choices = row["choices"]
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    q = row["question"].strip()
    choices_str = "\n".join(choices)
    return (
        "Select the best answer to the following multiple-choice question "
        "based on the video. Respond with only the letter (A, B, C, or D) "
        "of the correct option.\n"
        f"{q}\n{choices_str}\nThe best answer is:")


def run_eval(args, model, processor, df, head_sets,
              layers, audio_enc, visual_enc, eps_norm, d_sink_t):
    rows = []
    failures = 0
    t0 = time.time()
    for _, r in tqdm(df.iterrows(), total=len(df),
                       desc=f"avspeaker-{args.tag}"):
        vp = Path(args.av_dir) / r["audio_visual_path"]
        if not vp.exists():
            failures += 1
            continue
        prompt = build_question(r)
        if args.qwen_video_defaults:
            # Match MAD's video preprocessing: no fps / max_pixels override
            # so qwen_omni_utils uses FPS=2.0 and per-frame max≈602112.
            conv = [
                {"role": "system",
                 "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [
                     {"type": "video", "video": str(vp)},
                     {"type": "text", "text": prompt}]}]
        else:
            if args.video_resized_hw is not None:
                conv = build_conversation(
                    str(vp), prompt, "av",
                    video_fps=args.video_fps,
                    resized_height=args.video_resized_hw,
                    resized_width=args.video_resized_hw)
            elif args.video_max_pixels is not None:
                conv = build_conversation(
                    str(vp), prompt, "av",
                    video_fps=args.video_fps,
                    video_max_pixels=args.video_max_pixels)
            else:
                conv = build_conversation(str(vp), prompt, "av",
                                           video_fps=args.video_fps)
        # Routing — same mechanism as AVHBench (_5_explore): a text-only
        # constrained-logit router over the question. Soft probs (p_a,p_v,p_av)
        # drive the schedule blend; argmax sets routed_mod (heads/sink-mask/γ).
        # `--route_from category` keeps the legacy hard category route.
        if args.route_from == "router":
            pa, pv, pav = (float(x) for x in
                           route_probs(model, processor, r["question"],
                                       _ROUTER_LABEL_IDS))
            routed_mod = _ROUTE_ORDER[int(np.argmax([pa, pv, pav]))]
        else:
            routed_mod = CATEGORY_TO_MODALITY.get(r["category"], "AV")
            pa, pv, pav = (1.0, 0.0, 0.0) if routed_mod == "AUDIO" else \
                          (0.0, 1.0, 0.0) if routed_mod == "VISUAL" else \
                          (0.0, 0.0, 1.0)

        if args.variant == "none":
            IV.clear_intervention()
            out = EX.generate_with_intervention(model, processor, conv,
                                                  use_aiv=True,
                                                  max_new_tokens=32)
        else:
            layer_to_key_mask, _S = EX.compute_per_layer_sink_masks(
                model, processor, conv, True,
                layers, audio_enc, visual_enc, eps_norm, d_sink_t,
                routed_mod)
            heads_list = EX.heads_for_variant_routing(
                head_sets, args.variant, routed_mod)
            l2h = EX.heads_by_layer(heads_list, args.layer_band)
            effective_gamma = args.gamma
            per_mod = {"AUDIO": args.gamma_a, "VISUAL": args.gamma_v,
                         "AV": args.gamma_av}
            if per_mod.get(routed_mod) is not None:
                effective_gamma = per_mod[routed_mod]
            # ASD adaptive γ
            if getattr(args, "asd", False):
                aw = EX._RUNTIME_STATE.get("asd_adaptive_weight", 1.0)
                clamp = getattr(args, "asd_strength_clamp", 0.6)
                effective_gamma = min(clamp, effective_gamma * aw)
            if "value_zero" in args.variant:
                mode = "value_zero"
            elif "boost" in args.variant:
                mode = "boost"
            else:
                mode = "suppress"
            # MAD-soft γ (flat): blend the 3 per-mod gammas by router probs,
            # same as _5_explore. Only applies when per-mod gammas are set.
            if args.mad_soft and (args.gamma_a is not None
                                  or args.gamma_v is not None
                                  or args.gamma_av is not None):
                g_a = args.gamma_a if args.gamma_a is not None else args.gamma
                g_v = args.gamma_v if args.gamma_v is not None else args.gamma
                g_av = args.gamma_av if args.gamma_av is not None else args.gamma
                effective_gamma = g_a * pa + g_v * pv + g_av * pav
            iv = dict(layer_to_heads=l2h, layer_to_key_mask=layer_to_key_mask,
                      mode=mode, gamma=effective_gamma)
            # Per-layer scheduled γ: router-weighted blend of the 3 shapes,
            # scaled by g_base — IDENTICAL to _5_explore.py:624.
            if _SCHED is not None:
                iv["gamma_schedule"] = (args.g_base * (
                    _SCHED["shape_a"] * pa + _SCHED["shape_v"] * pv
                    + _SCHED["shape_av"] * pav)).tolist()
            IV.set_intervention(iv)
            try:
                out = EX.generate_with_intervention(model, processor, conv,
                                                      use_aiv=True)
            finally:
                IV.clear_intervention()
        pred = parse_letter(out)
        rows.append(dict(
            question_id=r["question_id"], category=r["category"],
            sub_category=r["sub_category"],
            label=r["answer"], routed=routed_mod,
            p_audio=round(pa, 4), p_visual=round(pv, 4), p_av=round(pav, 4),
            generated=out[:120], predicted=pred,
            correct=int(pred == r["answer"])))
    dt = time.time() - t0
    return pd.DataFrame(rows), dt, failures


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    global _SCHED, _ROUTER_LABEL_IDS
    if args.gamma_schedule_npz:
        _SCHED = dict(np.load(args.gamma_schedule_npz))
        print(f"[schedule] loaded {args.gamma_schedule_npz}, g_base={args.g_base}", flush=True)

    print("Loading model + patching attention ...", flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    if args.route_from == "router":
        _ROUTER_LABEL_IDS = precompute_router_label_ids(processor)
        print("[router] using AVHBench text-only constrained-logit router "
              "(soft blend for schedule, argmax for routed_mod)", flush=True)
    layers = thinker_layers(model)
    eps_norm = EX._thinker_rms_eps(model)
    audio_enc, visual_enc = EX._resolve_encoders(model)
    d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)
    IV.patch_qwen_attention(model)
    IV.clear_intervention()

    # Propagate sink_mask + tau to _5_explore globals
    EX._RUNTIME_ARGS = args
    if args.tau_sink is not None:
        EX.TAU_SINK = float(args.tau_sink)
        print(f"[tau_sink override] TAU_SINK = {EX.TAU_SINK}")

    head_sets = EX.load_head_sets(args.heads_csv)
    print("Head-set sizes:")
    for k, v in head_sets.items():
        print(f"  {k:<20s} {len(v):4d}")

    # Load AV-SpeakerBench test.csv
    csv_path = Path(args.test_csv)
    if not csv_path.exists():
        # try to download via HF
        from huggingface_hub import hf_hub_download
        csv_path = Path(hf_hub_download(
            "plnguyen2908/AV-SpeakerBench", "test.csv", repo_type="dataset"))
    df = pd.read_csv(csv_path)
    if args.max_duration_s is not None and args.max_duration_s > 0:
        import re as _re
        def _dur(p):
            m = _re.search(r'_(\d+)_(\d+)\.mp4$', p or "")
            return int(m.group(2)) - int(m.group(1)) if m else None
        df["dur_s"] = df["audio_visual_path"].apply(_dur)
        n0 = len(df)
        df = df[df["dur_s"].notna() & (df["dur_s"] <= args.max_duration_s)
                ].reset_index(drop=True)
        print(f"[duration filter] kept {len(df)}/{n0} clips with dur ≤ "
              f"{args.max_duration_s}s", flush=True)
    if args.limit is not None and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)),
                        random_state=42).reset_index(drop=True)
    print(f"Eval n={len(df)} (variant={args.variant}, tag={args.tag})")

    out_df, dt, fail = run_eval(
        args, model, processor, df, head_sets,
        layers, audio_enc, visual_enc, eps_norm, d_sink_t)
    fname = f"avspeaker_{args.tag}.csv"
    out_df.to_csv(out_dir / fname, index=False)
    overall = out_df.correct.mean() * 100
    per_cat = out_df.groupby("category")["correct"].mean() * 100
    per_sub = out_df.groupby("sub_category")["correct"].mean() * 100

    print(f"\n=== Result tag={args.tag} ===")
    print(f"  n={len(out_df)}, failures={fail}, elapsed={dt/60:.1f} min")
    print(f"  overall = {overall:.2f}%")
    for c, a in per_cat.items():
        print(f"    {c:<20s} {a:5.2f}%")

    log = out_dir / "avspeaker_log.md"
    if not log.exists():
        log.write_text("# AV-SpeakerBench eval log\n\n"
                        "MCQ A/B/C/D over 3212 clips. Routing by category.\n\n")
    with log.open("a") as f:
        f.write("---\n\n")
        f.write(f"## {args.tag}\n\n")
        if args.note:
            f.write(f"**Method:** {args.note}\n\n")
        f.write(f"**Config:** variant=`{args.variant}` gamma=`{args.gamma}` "
                  f"per_mod=(a={args.gamma_a}, v={args.gamma_v}, av={args.gamma_av}) "
                  f"layer_band=`{args.layer_band}` sink_mask=`{args.sink_mask}` "
                  f"tau_sink=`{args.tau_sink or 20.0}` n={len(out_df)}\n\n")
        f.write(f"**Result:** overall **{overall:.2f}%**\n\n")
        f.write("| category | n | accuracy |\n|---|---:|---:|\n")
        for c, sub in out_df.groupby("category"):
            f.write(f"| {c} | {len(sub)} | {sub.correct.mean()*100:.2f}% |\n")
        f.write("\n| sub_category | n | accuracy |\n|---|---:|---:|\n")
        for c, sub in out_df.groupby("sub_category"):
            f.write(f"| {c} | {len(sub)} | {sub.correct.mean()*100:.2f}% |\n")
        f.write(f"\nElapsed: {dt/60:.1f} min, failures: {fail}\n\n")
    print(f"  -> appended to {log}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv",
                   default=str(DEFAULT_DATA / "test.csv"))
    p.add_argument("--av_dir", default=str(DEFAULT_DATA),
                   help="Directory containing audiovisual/* files.")
    p.add_argument("--heads_csv",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--variant", required=True,
                   choices=["none",
                              "suppress_halluc_sink",
                              "suppress_inert_sink",
                              "boost_halluc_content",
                              "boost_inert_content",
                              "boost_all_halluc_content",
                              "boost_inert_and_halluc_content",
                              "boost_all_heads_content",
                              "suppress_all_heads_sink",
                              "suppress_inert_and_halluc_sink",
                              "value_zero_inert_sink",
                              "value_zero_halluc_sink",
                              "value_zero_all_heads_sink"])
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--layer_band", default="full",
                   choices=["full", "early", "mid", "late"])
    p.add_argument("--sink_mask", default="llm_emerged",
                   choices=["llm_emerged", "all", "prop"])
    p.add_argument("--gamma_a", type=float, default=None)
    p.add_argument("--gamma_v", type=float, default=None)
    p.add_argument("--gamma_av", type=float, default=None)
    p.add_argument("--gamma_schedule_npz", type=str, default=None,
                   help="npz shape_a/shape_v/shape_av (per-layer). Per-layer "
                        "gamma = g_base*(shape_a*p_a+shape_v*p_v+shape_av*p_av) "
                        "via router probs — identical to _5_explore.py.")
    p.add_argument("--g_base", type=float, default=3.0)
    p.add_argument("--route_from", default="router",
                   choices=["router", "category"],
                   help="router = AVHBench text-only constrained-logit router "
                        "(soft probs blend the schedule, argmax sets routed_mod); "
                        "category = legacy hard route from the dataset label.")
    p.add_argument("--mad_soft", action="store_true",
                   help="Blend per-mod gammas by router probs for the FLAT γ "
                        "(matches _5_explore); ignored under scheduling.")
    p.add_argument("--tau_sink", type=float, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Random-sampled subset (for fast probing).")
    p.add_argument("--max_duration_s", type=int, default=None,
                   help="Filter to clips with duration ≤ this many seconds.")
    p.add_argument("--video_max_pixels", type=int, default=None,
                   help="Override per-frame max pixels in build_conversation "
                         "(default 360*640=230400). E.g. 78400 = 280×280.")
    p.add_argument("--video_resized_hw", type=int, default=None,
                   help="Force each frame to be resized to N×N (bypasses "
                         "the qwen_omni_utils min_pixels assertion).")
    p.add_argument("--video_fps", type=float, default=1.0,
                   help="Video sampling fps (default 1.0).")
    p.add_argument("--asd", action="store_true",
                   help="ASD-inspired: restrict sink mask to cross-modal "
                         "sinks (low |MDS|) + adaptive γ scaling.")
    p.add_argument("--asd_mds_threshold", type=float, default=0.3)
    p.add_argument("--asd_strength_clamp", type=float, default=0.6)
    p.add_argument("--qwen_video_defaults", action="store_true",
                   help="Use qwen_omni_utils factory video preprocessing "
                         "(fps=2.0, max_pixels≈602112) instead of our "
                         "project defaults (fps=1.0, max_pixels=230400). "
                         "Match MAD's video tokenization for fair comparison.")
    args = p.parse_args()
    main(args)
