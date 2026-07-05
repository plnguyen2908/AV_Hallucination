"""AV-SpeakerBench eval driver using MAD (multimodal contrast decoding).

Adapted from `top-yun/MAD` qwen-omni/utils.py:
  https://github.com/top-yun/MAD/blob/main/qwen-omni/utils.py

Key differences vs. AVHBench MAD:
  * AV-SpeakerBench provides separate audio_only/, visual_only/, and
    audiovisual/ media files; we use those three plus the text-only branch.
  * Output is one of A/B/C/D (max_new_tokens = 1).

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

from utils import load_omni  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

DEFAULT_DATA = _REPO / "data/AV_SpeakerBench"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention/avspeaker"

# MAD's official prompts (verbatim from utils.py in the repo).
MODALITY_QUERY_PROMPT = (
    "To answer this question, which modality is needed "
    "(audio, video, or both): ")
# Empty — the official AV-SpeakerBench template ends with "The best
# answer is:" inside build_question itself, no separate suffix.
ANSWER_QUERY_PROMPT = ""

SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, "
    "Alibaba Group, capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech.")

# Official AV-SpeakerBench answer-extraction (plnguyen2908/AV-SpeakerBench).
ANSWER_PREFIXES = [
    "The best answer is", "The correct answer is", "The answer is",
    "The answer", "The best option is", "The correct option is",
    "Best answer:", "Best option:", "Answer:", "Option:",
    "The correct answer", "The correct option", "Based",
    "Correct answer", "☞", "<|im_end|>",
]
PUNCT_RE = re.compile(r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]")


def parse_letter(text: str) -> str:
    if not text: return "?"
    s = text.strip()
    for pref in ANSWER_PREFIXES:
        s = s.replace(pref, "")
    s = PUNCT_RE.sub(" ", s)
    for tok in s.split():
        if tok in {"A", "B", "C", "D", "E"}:
            return tok[0]
    return "?"


def build_question(row) -> str:
    """Official AV-SpeakerBench MCQ template."""
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


def mad_decode(model, processor, question, av_path, audio_path, visual_path,
                gamma=2.5, max_new_tokens=1, add_prompt=MODALITY_QUERY_PROMPT,
                fps=1.0, max_pixels=360 * 640, _override_max_pixels=None):
    if _override_max_pixels is not None:
        max_pixels = int(_override_max_pixels)
    """MAD multi-branch contrastive decoding.

    Returns the decoded string (length max_new_tokens or up to EOS).
    """
    # ---- 4 branch conversations ------------------------------------------
    def vid(path):
        return {"type": "video", "video": path,
                  "fps": fps, "max_pixels": max_pixels}

    # av branch (= head; also used for routing)
    head_conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            vid(av_path),
            {"type": "text", "text": "Question: " + question + "\n" + add_prompt},
        ]}
    ]
    conv_av = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            vid(av_path),
            {"type": "text", "text": question},
        ]}
    ]
    conv_v = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            vid(visual_path or av_path),
            {"type": "text", "text": question},
        ]}
    ]
    conv_a = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": question},
        ]}
    ]
    conv_t = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]

    # ---- Routing: get p_audio, p_video, p_both from head conv -----------
    head_text = processor.apply_chat_template(
        head_conv, add_generation_prompt=True, tokenize=False)
    h_audios, h_images, h_videos = process_mm_info(
        head_conv, use_audio_in_video=True)
    head_inputs = processor(
        text=head_text, audio=h_audios, images=h_images, videos=h_videos,
        return_tensors="pt", padding=True, use_audio_in_video=True)
    head_inputs = {k: (v.to(model.device).to(model.dtype)
                            if torch.is_floating_point(v) else v.to(model.device))
                       if hasattr(v, "to") else v
                   for k, v in head_inputs.items()}
    with torch.inference_mode():
        head_out = model.thinker.forward(
            **head_inputs, use_audio_in_video=True)
    audio_token_idx = processor.tokenizer.encode("audio")[0]
    video_token_idx = processor.tokenizer.encode("video")[0]
    both_token_idx = processor.tokenizer.encode("both")[0]
    head_logits = head_out.logits[0, -1, :]
    a_logit = head_logits[audio_token_idx].item()
    v_logit = head_logits[video_token_idx].item()
    b_logit = head_logits[both_token_idx].item()
    av_probs = torch.softmax(
        torch.tensor([a_logit, v_logit, b_logit]), dim=0).tolist()
    audio_prob, video_prob, both_prob = av_probs

    alpha_av = 2 + 2 * gamma * both_prob
    alpha_v = 1 - (both_prob - video_prob) * gamma
    alpha_a = 1 - (both_prob - audio_prob) * gamma
    alpha_t = -(video_prob + audio_prob) * gamma
    weights = {"av": alpha_av, "v": alpha_v, "a": alpha_a, "t": alpha_t}

    convs = {"av": (conv_av, True), "v": (conv_v, False),
              "a": (conv_a, True), "t": (conv_t, False)}

    # ---- Pre-fill each branch -------------------------------------------
    branches = {}
    step_logits = {}
    for key, (conv, use_aiv) in convs.items():
        text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(
            conv, use_audio_in_video=use_aiv)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
            use_audio_in_video=use_aiv)
        inputs = {k: (v.to(model.device).to(model.dtype)
                           if torch.is_floating_point(v) else v.to(model.device))
                       if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.thinker(
                **inputs, use_audio_in_video=use_aiv, use_cache=True)
        branches[key] = out.past_key_values
        step_logits[key] = out.logits[:, -1, :].detach()

    # ---- Decode loop -----------------------------------------------------
    eos = processor.tokenizer.eos_token_id
    generated = []
    for _ in range(max_new_tokens):
        logits_mat = torch.stack(
            [(weights[k] * step_logits[k]).squeeze(0) for k in convs.keys()],
            dim=0)
        avg_logits = logits_mat.sum(dim=0)
        next_token = int(torch.argmax(avg_logits, dim=-1))
        generated.append(next_token)
        if next_token == eos:
            break
        next_tok = torch.tensor([[next_token]], device=model.device,
                                  dtype=torch.long)
        for key, (_, use_aiv) in convs.items():
            with torch.inference_mode():
                out = model.thinker(
                    next_tok, use_audio_in_video=use_aiv, use_cache=True,
                    past_key_values=branches[key])
            branches[key] = out.past_key_values
            step_logits[key] = out.logits[:, -1, :].detach()
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def run_eval(args, model, processor, df):
    rows = []
    failures = 0
    t0 = time.time()
    av_root = Path(args.av_dir)
    for _, r in tqdm(df.iterrows(), total=len(df),
                       desc=f"avspeaker-mad-{args.tag}"):
        av_path = av_root / r["audio_visual_path"]
        a_path = av_root / r["audio_path"]
        v_path = av_root / r["visual_path"]
        if not av_path.exists() or not a_path.exists():
            failures += 1; continue
        body = build_question(r)
        prompt = body + ANSWER_QUERY_PROMPT
        out = mad_decode(
            model, processor, prompt, str(av_path),
            str(a_path), str(v_path) if v_path.exists() else None,
            gamma=args.gamma, max_new_tokens=args.max_new_tokens,
            _override_max_pixels=args.video_max_pixels)
        pred = parse_letter(out)
        rows.append(dict(
            question_id=r["question_id"], category=r["category"],
            sub_category=r["sub_category"],
            label=r["answer"], generated=out[:120], predicted=pred,
            correct=int(pred == r["answer"])))
    return pd.DataFrame(rows), time.time() - t0, failures


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model (sdpa — MAD doesn't touch attn weights; "
          "memory-efficient kernel avoids full N×N alloc) ...", flush=True)
    model, processor = load_omni(
        args.model_path, device_map=args.device_map,
        attn_implementation="sdpa")

    csv_path = Path(args.test_csv)
    if not csv_path.exists():
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
    print(f"Eval n={len(df)} (tag={args.tag}, γ={args.gamma})", flush=True)

    out_df, dt, fail = run_eval(args, model, processor, df)
    fname = f"avspeaker_{args.tag}.csv"
    out_df.to_csv(out_dir / fname, index=False)
    overall = out_df.correct.mean() * 100 if len(out_df) else 0
    per_cat = out_df.groupby("category")["correct"].mean() * 100

    print(f"\n=== Result tag={args.tag} ===")
    print(f"  n={len(out_df)}, failures={fail}, elapsed={dt/60:.1f} min")
    print(f"  overall = {overall:.2f}%")
    for c, a in per_cat.items():
        print(f"    {c:<20s} {a:5.2f}%")

    log = out_dir / "avspeaker_log.md"
    if not log.exists():
        log.write_text("# AV-SpeakerBench eval log\n\n")
    with log.open("a") as f:
        f.write("---\n\n")
        f.write(f"## {args.tag}\n\n")
        if args.note:
            f.write(f"**Method:** {args.note}\n\n")
        f.write(f"**Config:** method=`MAD` gamma=`{args.gamma}` "
                  f"max_new_tokens=`{args.max_new_tokens}` n={len(out_df)}\n\n")
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
    p.add_argument("--av_dir", default=str(DEFAULT_DATA))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--tag", required=True)
    p.add_argument("--note", default="MAD multi-branch contrast decoding.")
    p.add_argument("--gamma", type=float, default=2.5,
                   help="MAD γ (default 2.5, official setting).")
    p.add_argument("--max_new_tokens", type=int, default=1,
                   help="Letter MCQ — 1 token suffices.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_duration_s", type=int, default=None,
                   help="Filter to clips with duration ≤ this many seconds "
                         "(parsed from path stem _<start>_<end>).")
    p.add_argument("--video_max_pixels", type=int, default=None,
                   help="Override per-frame max_pixels (default 230400).")
    args = p.parse_args()
    main(args)
