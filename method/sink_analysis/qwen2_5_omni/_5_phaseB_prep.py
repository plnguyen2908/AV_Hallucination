"""
_5_phaseB_prep.py — Stage 5 Phase B prep.

Runs the two things the spec says must precede the exploration loop:

  (a) Per-clip sink-count sanity check. Picks a small number of clips
      (default 4: 2 from AVHBench + 2 from VGGSounder where we have
      offline per_clip_tokens), runs the live in-process sink
      computation (RMSNorm-normed activation on D_sink={458,2570} ≥
      τ_sink=20 AND NOT p_prop), and reports per-layer sink counts.
      On the VGGSounder pair, also compares against the offline
      stage3_2 per_clip_tokens/<clip>.npz to confirm the live code
      matches the offline 4.1b sink definition exactly.

  (b) Constrained-logit router (the spec's preferred design): read the
      logits of EXACTLY the three label tokens (" Audio"/" Visual"/
      " AV") at the answer position, softmax over those 3 only,
      argmax = routed modality. Single forward pass per question
      (no generation). Run on DEV (already split) and the rest of
      AVHBench (the "TEST" pool = HELDOUT ∪ DROPPED). Report routing
      accuracy overall + per-category, on each split and on the
      cross-modal AV-Matching subset specifically.

Outputs (in `results/qwen2_5_omni/stage5_intervention/`):
    sink_sanity.md / sink_sanity.csv
    router_v2_dev.csv / router_v2_test.csv
    routing_per_category.md
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, thinker_layers, OMNI_SYSTEM_PROMPT  # noqa: E402

# Same constants as stage3_1
D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0

AVHBENCH_VIDEOS = _REPO / "data/AVHBench/videos"
VGGSOUNDER_VIDEOS = _REPO / "data/VGGSounder/videos"
VGG_OFFLINE_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
OUT_DIR = _REPO / "results/qwen2_5_omni/stage5_intervention"

ROUTER_PROMPT_TPL_V2 = (
    "You are classifying questions by the modality they ask about. "
    "Audio if the question is about sounds, hearing, or what is audible. "
    "Visual if the question is about images, objects, or what is visible. "
    "AV if answering requires comparing or matching what is heard against "
    "what is seen (cross-modal consistency, joint description). "
    "Respond with exactly one word: Audio, Visual, or AV.\n\n"
    "Question: {q}\n\nClassification:"
)
TASK_TO_GT_MODALITY = {
    "Video-driven Audio Hallucination": "AUDIO",
    "Audio-driven Video Hallucination": "VISUAL",
    "AV Matching":                       "AV",
}
ROUTER_LABELS = [" Audio", " Visual", " AV"]   # leading-space surface forms
ROUTER_LABEL_TO_MODALITY = {" Audio": "AUDIO", " Visual": "VISUAL",
                              " AV": "AV"}


# ---------------------------------------------------------------------
# Sink sanity check
# ---------------------------------------------------------------------

def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_token_index"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _thinker_rms_eps(model) -> float:
    cfg = getattr(model.thinker.config, "text_config", model.thinker.config)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def _resolve_encoders(model):
    thinker = model.thinker
    audio_mod = visual_mod = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr); break
    return audio_mod, visual_mod


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def _align_norms(enc_norms, n_llm):
    n_enc = len(enc_norms)
    if n_enc == n_llm: return enc_norms
    if n_enc > n_llm and n_enc % n_llm == 0:
        return enc_norms.reshape(n_llm, n_enc // n_llm).mean(axis=1)
    if n_llm > n_enc and n_llm % n_enc == 0:
        return np.repeat(enc_norms, n_llm // n_enc)
    return None


def _modal_positions(input_ids, thinker_cfg):
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


def live_sink_compute(model, processor, video_path, layers, audio_enc,
                       visual_enc, eps_norm, d_sink_t, use_aiv=True,
                       question="Describe what you see and hear."):
    """Run a single thinker forward with hooks to capture pre-SA hidden
    states and encoder norms; return per-layer sink counts and masks."""
    conv = build_conversation(str(video_path), question, "av")
    from qwen_omni_utils import process_mm_info
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_aiv)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list): text = text[0]
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True,
                        use_audio_in_video=use_aiv)
    inputs = inputs.to(model.device).to(model.dtype)
    thinker_cfg = _resolve_thinker_cfg(model)
    audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
    S = int(inputs["input_ids"].shape[1])
    n_layers = len(layers)

    per_layer_h = [None] * n_layers
    a_enc_buf, v_enc_buf = [], []

    def make_pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach().cpu()
        return _h

    def make_enc_hook(buf):
        def _h(_m, _i, out):
            tok = _extract_tokens(out)
            buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
        return _h

    handles = []
    if audio_enc is not None:
        handles.append(audio_enc.register_forward_hook(make_enc_hook(a_enc_buf)))
    if visual_enc is not None:
        handles.append(visual_enc.register_forward_hook(make_enc_hook(v_enc_buf)))
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(make_pre_hook(L)))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        for h in handles: h.remove()

    # p_llm per layer
    p_llm = np.zeros((n_layers, S), dtype=bool)
    for L in range(n_layers):
        h_L = per_layer_h[L].float()
        rms = torch.sqrt(h_L.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_L / rms).abs()
        p_llm[L] = (normed_abs[:, d_sink_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
    # p_prop
    p_prop = np.zeros(S, dtype=bool)
    if a_enc_buf and len(audio_pos) > 0:
        a_aligned = _align_norms(np.concatenate(a_enc_buf), len(audio_pos))
        if a_aligned is not None:
            p_prop[audio_pos] = a_aligned > TAU_PROP
    if v_enc_buf and len(video_pos) > 0:
        v_aligned = _align_norms(np.concatenate(v_enc_buf), len(video_pos))
        if v_aligned is not None:
            p_prop[video_pos] = v_aligned > TAU_PROP

    audio_mask = np.zeros(S, dtype=bool); audio_mask[audio_pos] = True
    video_mask = np.zeros(S, dtype=bool); video_mask[video_pos] = True
    return dict(p_llm=p_llm, p_prop=p_prop,
                  audio_mask=audio_mask, video_mask=video_mask,
                  S=S, n_audio=len(audio_pos), n_video=len(video_pos))


def run_sink_sanity(model, processor, layers, audio_enc, visual_enc,
                      eps_norm, out_dir):
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    print(f"\n=== (a) Per-clip sink-count sanity check ===")
    rows = []
    sanity_lines = ["# Sink sanity check", ""]

    # AVHBench clips (no offline data — internal-consistency only)
    avh_clips = sorted([p for p in AVHBENCH_VIDEOS.glob("*.mp4")])[:2]
    for vp in avh_clips:
        print(f"\n  AVHBench {vp.name}")
        r = live_sink_compute(model, processor, vp, layers, audio_enc,
                                visual_enc, eps_norm, d_sink_t)
        llm_mask = r["p_llm"] & ~r["p_prop"]
        audio_sinks_per_L = (llm_mask & r["audio_mask"]).sum(axis=1)
        video_sinks_per_L = (llm_mask & r["video_mask"]).sum(axis=1)
        print(f"    n_audio={r['n_audio']}, n_video={r['n_video']}, S={r['S']}")
        print(f"    audio sinks/layer (head/mid/tail): "
              f"{audio_sinks_per_L[5]:3d} / {audio_sinks_per_L[14]:3d} / "
              f"{audio_sinks_per_L[-1]:3d}")
        print(f"    video sinks/layer (head/mid/tail): "
              f"{video_sinks_per_L[5]:3d} / {video_sinks_per_L[14]:3d} / "
              f"{video_sinks_per_L[-1]:3d}")
        sanity_lines.append(f"## AVHBench `{vp.name}`")
        sanity_lines.append("")
        sanity_lines.append(f"- S={r['S']}, n_audio={r['n_audio']}, "
                              f"n_video={r['n_video']}")
        for L in range(0, len(layers), 4):
            sanity_lines.append(f"  - L{L}: a_sinks={audio_sinks_per_L[L]}, "
                                  f"v_sinks={video_sinks_per_L[L]}")
        for L in range(len(layers)):
            rows.append(dict(source="AVHBench", clip=vp.name, layer=L,
                              n_audio=r["n_audio"], n_video=r["n_video"],
                              audio_sinks=int(audio_sinks_per_L[L]),
                              video_sinks=int(video_sinks_per_L[L]),
                              match_offline=None))

    # VGGSounder clips — compare to offline stage3_2/per_clip_tokens
    vgg_files = sorted(VGG_OFFLINE_DIR.glob("*.npz"))[:2]
    for npz_p in vgg_files:
        clip_stem = npz_p.stem
        vp = VGGSOUNDER_VIDEOS / f"{clip_stem}.mp4"
        if not vp.exists():
            print(f"\n  VGGSounder {clip_stem}: video missing")
            continue
        print(f"\n  VGGSounder {clip_stem} (with offline comparison)")
        r = live_sink_compute(model, processor, vp, layers, audio_enc,
                                visual_enc, eps_norm, d_sink_t)
        live_llm = r["p_llm"] & ~r["p_prop"]
        live_a = (live_llm & r["audio_mask"]).sum(axis=1)
        live_v = (live_llm & r["video_mask"]).sum(axis=1)
        d = np.load(npz_p, allow_pickle=True)
        off_p_llm = d["p_llm"]; off_p_prop = d["p_prop"]
        off_audio_pos = d["audio_pos"]; off_video_pos = d["video_pos"]
        off_S = int(d["S"])
        off_audio_mask = np.zeros(off_S, dtype=bool)
        off_audio_mask[off_audio_pos] = True
        off_video_mask = np.zeros(off_S, dtype=bool)
        off_video_mask[off_video_pos] = True
        off_llm = off_p_llm & ~off_p_prop
        off_a = (off_llm & off_audio_mask).sum(axis=1)
        off_v = (off_llm & off_video_mask).sum(axis=1)

        match_S = (r["S"] == off_S)
        match_audio = bool(np.array_equal(live_a, off_a))
        match_video = bool(np.array_equal(live_v, off_v))
        verdict = ("MATCH" if (match_S and match_audio and match_video)
                   else "MISMATCH")
        print(f"    S live={r['S']} offline={off_S} (same={match_S})")
        print(f"    live audio sinks/L head/mid/tail: "
              f"{live_a[5]} / {live_a[14]} / {live_a[-1]}")
        print(f"    offl audio sinks/L head/mid/tail: "
              f"{off_a[5]} / {off_a[14]} / {off_a[-1]}")
        print(f"    live video sinks/L head/mid/tail: "
              f"{live_v[5]} / {live_v[14]} / {live_v[-1]}")
        print(f"    offl video sinks/L head/mid/tail: "
              f"{off_v[5]} / {off_v[14]} / {off_v[-1]}")
        print(f"    -> {verdict} (audio={match_audio}, video={match_video})")
        sanity_lines.append(f"## VGGSounder `{clip_stem}` "
                              f"(live vs offline 4.1b/3.2): "
                              f"**{verdict}**")
        sanity_lines.append("")
        sanity_lines.append(f"- S live={r['S']}, offline={off_S}; "
                              f"audio match={match_audio}, "
                              f"video match={match_video}")
        for L in range(len(layers)):
            rows.append(dict(source="VGGSounder", clip=vp.name, layer=L,
                              n_audio=r["n_audio"], n_video=r["n_video"],
                              audio_sinks=int(live_a[L]),
                              video_sinks=int(live_v[L]),
                              offline_audio_sinks=int(off_a[L]),
                              offline_video_sinks=int(off_v[L]),
                              match_offline=(int(live_a[L])==int(off_a[L])
                                              and int(live_v[L])==int(off_v[L]))))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sink_sanity.csv", index=False)
    (out_dir / "sink_sanity.md").write_text("\n".join(sanity_lines))
    print(f"\n  -> {out_dir / 'sink_sanity.csv'}")
    print(f"  -> {out_dir / 'sink_sanity.md'}")


# ---------------------------------------------------------------------
# Constrained-logit router
# ---------------------------------------------------------------------

def verify_single_token_labels(processor, prompt_text):
    """For each candidate label, verify that appending it to the prompt
    produces exactly one additional token. Return (ok, info_string)."""
    base_ids = processor.tokenizer(prompt_text,
                                     add_special_tokens=False).input_ids
    info = []
    ok = True
    for label in ROUTER_LABELS:
        full_ids = processor.tokenizer(prompt_text + label,
                                         add_special_tokens=False).input_ids
        diff = len(full_ids) - len(base_ids)
        token_id = full_ids[-1] if diff == 1 else None
        decoded = (processor.tokenizer.decode([token_id])
                    if token_id is not None else "?")
        info.append(f"    {label!r}: +{diff} tokens"
                     + (f" id={token_id} decode={decoded!r}" if token_id
                         else ""))
        if diff != 1:
            ok = False
    return ok, info


def run_router_constrained(model, processor, df, split_name, out_dir):
    """Constrained-logit router on a dataframe with text/question_id/task."""
    print(f"\n=== Router (constrained-logit) on {split_name} "
          f"(n={len(df)}) ===")
    tok = processor.tokenizer

    # Resolve label token ids by running on a sample prompt to be sure
    sample_q = df.iloc[0]["text"] if len(df) else "Does X exist?"
    sample_prompt = ROUTER_PROMPT_TPL_V2.format(q=sample_q)
    # Build full chat text + assistant generation prefix
    conv = [
        {"role": "system",
         "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "text", "text": sample_prompt}]},
    ]
    chat_text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False)
    if isinstance(chat_text, list): chat_text = chat_text[0]

    print("  Single-token verification of router labels at the assistant "
          "position:")
    ok, info = verify_single_token_labels(processor, chat_text)
    for line in info: print(line)
    print(f"  all single-token: {ok}")
    label_token_ids = []
    for label in ROUTER_LABELS:
        full_ids = tok(chat_text + label, add_special_tokens=False).input_ids
        base_ids = tok(chat_text, add_special_tokens=False).input_ids
        if len(full_ids) - len(base_ids) != 1:
            print(f"  FATAL: {label!r} not single token; aborting router.")
            return None
        label_token_ids.append(full_ids[-1])
    print(f"  label token ids: {dict(zip(ROUTER_LABELS, label_token_ids))}")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"router-{split_name}"):
        prompt = ROUTER_PROMPT_TPL_V2.format(q=r["text"])
        conv = [
            {"role": "system",
             "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [{"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)
        if isinstance(text, list): text = text[0]
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            out = model.thinker(**inputs, output_attentions=False,
                                  return_dict=True, use_cache=False)
        # Logits for next token at the last position
        last_logits = out.logits[0, -1, :].float()
        sel = last_logits[label_token_ids]
        probs3 = torch.softmax(sel, dim=-1)
        pred_idx = int(torch.argmax(probs3).item())
        pred_modality = ROUTER_LABEL_TO_MODALITY[ROUTER_LABELS[pred_idx]]
        probs_np = probs3.cpu().numpy().tolist()
        gt = TASK_TO_GT_MODALITY[r["task"]]
        rows.append(dict(
            question_id=r["question_id"], task=r["task"], text=r["text"],
            gt_modality=gt, predicted=pred_modality,
            correct=int(pred_modality == gt),
            p_audio=probs_np[0], p_visual=probs_np[1], p_av=probs_np[2],
        ))
    df_out = pd.DataFrame(rows)
    out_csv = out_dir / f"router_v2_{split_name.lower()}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")
    return df_out


def report_routing(df, name, out_lines):
    out_lines.append(f"## Routing on {name} (n={len(df)})")
    out_lines.append("")
    overall = df["correct"].mean() * 100
    out_lines.append(f"- Overall accuracy: **{overall:.1f}%**")
    out_lines.append("")
    out_lines.append("| task | n | accuracy |")
    out_lines.append("|---|---:|---:|")
    for t, sub in df.groupby("task"):
        out_lines.append(f"| {t} | {len(sub)} | {sub.correct.mean()*100:.1f}% |")
    out_lines.append("")
    # Cross-modal AV-Matching subset (the spec's "adversarial" one)
    if "AV Matching" in df.task.values:
        av = df[df.task == "AV Matching"]
        out_lines.append(f"- AV Matching (cross-modal) accuracy: "
                          f"**{av.correct.mean()*100:.1f}%** (n={len(av)})")
        out_lines.append("")
    out_lines.append("Confusion (rows=gt, cols=pred):")
    conf = pd.crosstab(df.gt_modality, df.predicted, margins=True)
    out_lines.append("```")
    out_lines.append(conf.to_string())
    out_lines.append("```")
    out_lines.append("")


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    eps_norm = _thinker_rms_eps(model)
    audio_enc, visual_enc = _resolve_encoders(model)

    if args.do_sanity:
        run_sink_sanity(model, processor, layers, audio_enc, visual_enc,
                          eps_norm, out_dir)

    if args.do_router:
        split_df = pd.read_csv(out_dir / "split.csv", dtype={"video_id": str})
        split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
        dev_df = split_df[split_df.split == "DEV"].reset_index(drop=True)
        test_df = split_df[split_df.split != "DEV"].reset_index(drop=True)
        dev_out = run_router_constrained(model, processor, dev_df, "DEV", out_dir)
        test_out = run_router_constrained(model, processor, test_df, "TEST", out_dir)
        lines = ["# Phase B prep — routing accuracy (constrained-logit, "
                 f"Audio/Visual/AV)", ""]
        if dev_out is not None:
            report_routing(dev_out, "DEV (n=300)", lines)
        if test_out is not None:
            report_routing(test_out,
                            f"TEST (HELDOUT + DROPPED, n={len(test_df)})",
                            lines)
        (out_dir / "routing_per_category.md").write_text("\n".join(lines))
        print(f"\nwrote {out_dir / 'routing_per_category.md'}")

    print("\nPhase B prep complete. STOP for confirmation before "
          "exploration loop.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(OUT_DIR))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--do_sanity", action="store_true", default=True)
    p.add_argument("--no_sanity", dest="do_sanity", action="store_false")
    p.add_argument("--do_router", action="store_true", default=True)
    p.add_argument("--no_router", dest="do_router", action="store_false")
    args = p.parse_args()
    main(args)
