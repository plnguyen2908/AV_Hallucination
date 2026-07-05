"""
stage5_step0_checkB_video_decode.py — CHECK B (Sink-or-Not / DIYSink-style).

For each clip do three forwards with a custom attention mask that
ISOLATES one video sink class at a time (suppress attention TO all other
video keys); decode the L27 hidden state at the kept positions via the
model's `final_norm + lm_head`, and check whether the decode aligns with
the clip's GT visual labels.

Three classes:
  - P_prop-video      (visual encoder L2 norm > 100, fixed across layers)
  - P_llm-only-video  (P_llm[L27] ∩ video span, EXCLUDING P_prop)
  - non-sink-video    (video positions that are neither P_prop nor P_llm[L27])

Pass: P_prop-video decodes to category-aligned content
(GT-label-word in top-K) at a rate ABOVE P_llm-only and ABOVE non-sink.
Fail: all three classes similar OR P_prop-video at non-sink baseline.

The spec's DIYSink procedure isolates ONE sink per forward; we use a
GROUP-level simplification (one forward per class, keeping all sinks
of that class visible while masking all other video). This costs
3 forwards per clip instead of ~50 (number of video sinks), and gives
the same group-level claim ("class X decodes to category-aligned
content"). The per-sink attribution claim is the only thing the
simplification gives up.

Outputs (`stage5_step0/checkB/`):
  checkB_per_class_decode.csv    per (clip, class) aggregate
  checkB_per_token_decode.csv    per (clip, class, position) top-K + flags
  checkB_word_distributions.md   qualitative top decoded tokens per class
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
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402

DEFAULT_QA   = _REPO / "results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json"
DEFAULT_VID  = _REPO / "data/VGGSounder/videos"
_MODALITY = "av"
_MEDIA_DIR = str(DEFAULT_VID)
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_step0/checkB"

D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0
AUDIO_TOKEN_ID = 151646
VIDEO_TOKEN_ID = 151656
TOP_K = 10
_LABEL_STOP = {"playing","sound","sounds","noise","noises","music","audio",
                "a","an","the","of","with","and","or","in","on","at"}
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def label_words(label_list) -> set:
    out = set()
    for lab in (label_list or []):
        for w in _WORD_RE.findall(str(lab).lower()):
            if w in _LABEL_STOP: continue
            out.add(w)
    return out


def is_on_topic(decoded: str, words: set) -> bool:
    s = decoded.strip().lower()
    if len(s) < 3: return False
    return any((w in s) or (s in w) for w in words)


def _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm):
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    masks = {}
    for L in range(n_layers):
        h = per_layer_h[L].float()
        rms = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h / rms).abs()
        d_t = d_sink_t.to(h.device)
        masks[L] = (normed_abs[:, d_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()
    return masks


def _resolve_encoders(model):
    thinker = model.thinker
    audio = visual = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr): audio = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr): visual = getattr(thinker, attr); break
    return audio, visual


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


def make_video_mask_pre_hook(mask_keys: np.ndarray, S: int):
    """Suppress attention TO `mask_keys` across all queries and heads
    (set to -inf in the attention_mask). Used per-class to keep only one
    video sink class visible."""
    pos_t = torch.as_tensor(mask_keys, dtype=torch.long)
    def _hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None or am.shape[-1] != S: return args, kwargs
        new_am = am.clone()
        idx = pos_t.to(new_am.device)
        new_am[..., :, idx] = torch.finfo(new_am.dtype).min
        kwargs["attention_mask"] = new_am
        return args, kwargs
    return _hook


def forward_capture_last_hidden(model, inputs, use_aiv, layers, n_layers,
                                  S_target):
    """Forward and capture `h_output[L27]` (post-block residual). Returns
    tensor of shape (S, D) or None."""
    last_out = [None]
    def post_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == S_target:
            last_out[0] = hs[0].detach()
        return out
    handle = layers[n_layers - 1].register_forward_hook(post_hook)
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        handle.remove()
    return last_out[0]


def process_clip(d, model, processor, layers, n_layers, visual_enc,
                  eps_norm, final_norm, lm_head, tokenizer,
                  out_records, out_per_token, max_nonsink_keep=None):
    video_path = Path(_MEDIA_DIR) / d["video"]
    if not video_path.exists(): return "missing_video"

    conv = build_conversation(str(video_path), d["question"], _MODALITY)
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, _MODALITY,
                                           model.device, model.dtype)
    except Exception as e:
        return f"prep:{type(e).__name__}"
    prompt_S = int(inputs["input_ids"].shape[1])
    full_S = prompt_S
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0].astype(np.int64)
    if video_pos.size == 0: return "no_video_tokens"

    # ----- Baseline forward: capture h_input[L] for P_llm and video encoder norms for P_prop. -----
    per_layer_h = [None] * n_layers
    v_enc_buf = []

    def pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach()
        return _h
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        v_enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())

    handles = []
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(pre_hook(L)))
    if visual_enc is not None:
        handles.append(visual_enc.register_forward_hook(enc_hook))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return f"fwd_base:{type(e).__name__}"
    finally:
        for h in handles: h.remove()
    if any(h is None for h in per_layer_h) or not v_enc_buf:
        return "no_capture"

    # P_llm[L27]
    p_llm_masks = _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm)
    p_llm_l27 = p_llm_masks[n_layers - 1]
    per_layer_h = [None] * n_layers
    torch.cuda.empty_cache()

    # P_prop video from visual encoder norms
    v_enc = np.concatenate(v_enc_buf)
    aligned = _align_norms(v_enc, len(video_pos))
    p_prop_video = np.zeros(prompt_S, dtype=bool)
    if aligned is not None:
        p_prop_video[video_pos] = aligned > TAU_PROP

    # Build the three class position sets
    video_set = set(video_pos.tolist())
    pos_prop = np.array(sorted(i for i in video_pos
                                  if p_prop_video[int(i)]), dtype=np.int64)
    pos_llm_only = np.array(sorted(i for i in video_pos
                                       if p_llm_l27[int(i)]
                                       and not p_prop_video[int(i)]),
                              dtype=np.int64)
    pos_nonsink = np.array(sorted(i for i in video_pos
                                      if not p_prop_video[int(i)]
                                      and not p_llm_l27[int(i)]),
                              dtype=np.int64)

    # GT label words for category alignment
    gt_words = label_words(d.get("answer") or d.get("gt_entities") or [])

    classes = [
        ("p_prop_video",     pos_prop),
        ("p_llm_only_video", pos_llm_only),
        ("non_sink_video",   pos_nonsink),
    ]

    # ----- For each class, forward with attention masked on all OTHER video keys -----
    for cls_name, kept in classes:
        if kept.size == 0:
            out_records.append((d["video"], cls_name, 0, 0, 0, 0.0, 0.0))
            continue
        # If non-sink and too many positions, randomly subsample to keep per-clip work fair
        kept_use = kept
        if max_nonsink_keep is not None and cls_name == "non_sink_video" \
                and kept.size > max_nonsink_keep:
            rng = np.random.default_rng(hash(d["video"]) % (2**31))
            kept_use = np.sort(rng.choice(kept, size=max_nonsink_keep, replace=False))

        masked = np.array(sorted(set(video_pos.tolist()) - set(kept_use.tolist())),
                            dtype=np.int64)
        hook = make_video_mask_pre_hook(masked, full_S)
        handles = [layers[L].register_forward_pre_hook(hook, with_kwargs=True)
                    for L in range(n_layers)]
        try:
            h27 = forward_capture_last_hidden(model, inputs, use_aiv, layers,
                                                 n_layers, full_S)
        except Exception as e:
            for h in handles: h.remove()
            torch.cuda.empty_cache()
            return f"fwd_{cls_name}:{type(e).__name__}"
        finally:
            for h in handles: h.remove()
        if h27 is None:
            out_records.append((d["video"], cls_name, int(kept_use.size),
                                  0, 0, float("nan"), float("nan")))
            continue

        # Decode at kept positions via final_norm + lm_head
        dev_lm = lm_head.weight.device
        dt_lm  = lm_head.weight.dtype
        h_sel = h27[kept_use].to(device=dev_lm, dtype=dt_lm)
        with torch.no_grad():
            h_n = final_norm(h_sel)
            logits = lm_head(h_n).float()
            top_vals, top_idx = logits.topk(TOP_K, dim=-1)
        ti = top_idx.cpu().numpy(); tv = top_vals.cpu().numpy()
        n_on_topic = 0
        for j, pos in enumerate(kept_use):
            top_strs = [tokenizer.decode([int(t)], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                          for t in ti[j]]
            hit = any(is_on_topic(s, gt_words) for s in top_strs)
            if hit: n_on_topic += 1
            out_per_token.append((d["video"], cls_name, int(pos),
                                    int(ti[j][0]), top_strs[0],
                                    float(tv[j][0]), int(hit),
                                    "|".join(top_strs)))
        on_topic_rate = float(n_on_topic / kept_use.size)
        out_records.append((d["video"], cls_name, int(kept_use.size),
                              int(kept.size), int(n_on_topic),
                              on_topic_rate, float(gt_words and 1.0 or 0.0)))
    torch.cuda.empty_cache()
    return None


def main(args):
    global _MODALITY, _MEDIA_DIR
    _MODALITY = args.modality
    if args.media_dir:
        _MEDIA_DIR = args.media_dir
    print(f"  modality={_MODALITY}; media_dir={_MEDIA_DIR}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)
    eps_norm = float(getattr(
        getattr(model.thinker.config, "text_config", model.thinker.config),
        "rms_norm_eps", 1e-6))
    final_norm = model.thinker.model.norm
    lm_head    = model.thinker.lm_head
    _, visual_enc = _resolve_encoders(model)
    print(f"n_layers={n_layers}; visual_enc={'OK' if visual_enc else 'MISSING'}")

    data = json.load(open(args.sampled_entities_json))
    # CHECK B is a content-decode test on video sinks; it does NOT need
    # paired hal/non labels. Use all clips that have a generated_caption
    # (so a forward will produce a usable L27 state).
    paired = [d for d in data if d.get("generated_caption")]
    print(f"{len(paired)} clips with caption (no paired-hal/non filter needed for CHECK B)")

    records, per_token = [], []
    failures = {}
    for d in tqdm(paired, desc="clips"):
        err = process_clip(d, model, processor, layers, n_layers, visual_enc,
                            eps_norm, final_norm, lm_head, processor.tokenizer,
                            records, per_token,
                            max_nonsink_keep=30)         # cap non-sink subset for time
        if err:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
    if failures: print(f"failures: {failures}")
    print(f"records={len(records)}; per_token={len(per_token)}")

    df = pd.DataFrame(records, columns=[
        "clip", "sink_class", "n_kept_decoded", "n_in_class_total",
        "n_on_topic", "on_topic_rate", "has_gt_label"])
    df.to_csv(out_dir / "checkB_per_class_decode.csv", index=False)
    per_df = pd.DataFrame(per_token, columns=[
        "clip", "sink_class", "position", "top1_id", "top1_decoded",
        "top1_logit", "top1_on_topic", "topk_decoded_pipe"])
    per_df.to_csv(out_dir / "checkB_per_token_decode.csv", index=False)

    agg = (df.groupby("sink_class", as_index=False)
              .agg(n_clips_with_class=("clip", "nunique"),
                   n_tokens_total=("n_kept_decoded", "sum"),
                   sum_on_topic=("n_on_topic", "sum"),
                   mean_on_topic_rate=("on_topic_rate", "mean")))
    agg["overall_on_topic_rate"] = agg["sum_on_topic"] / agg["n_tokens_total"].clip(lower=1)
    print("\n=== aggregate by sink class (CHECK B) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Top decoded tokens per class
    md = ["# Stage 5 Step 0 — CHECK B: video sink masked decode (DIYSink style)\n",
          "For each clip, isolate one video sink class at a time (mask attention to all "
          "other video keys), forward through all layers, decode the L27 hidden state at "
          "the kept positions via `final_norm + lm_head`. `on_topic_rate` = fraction of "
          "positions whose top-10 decode contains a GT label word (≥3 letters, "
          "case-insensitive substring).\n",
          "## Aggregate\n",
          "| sink_class | n_clips | n_positions | overall on-topic rate | mean per-clip rate |",
          "|---|---:|---:|---:|---:|"]
    for _, r in agg.iterrows():
        md.append(f"| {r['sink_class']} | {int(r['n_clips_with_class'])} | "
                  f"{int(r['n_tokens_total'])} | "
                  f"**{r['overall_on_topic_rate']:.3f}** | "
                  f"{r['mean_on_topic_rate']:.3f} |")

    for cls in ("p_prop_video", "p_llm_only_video", "non_sink_video"):
        sub = per_df[per_df["sink_class"] == cls]
        if sub.empty:
            md.append(f"\n## `{cls}` — 0 positions\n")
            continue
        vc = sub["top1_decoded"].value_counts().head(20)
        md.append(f"\n## Top-20 top-1 decoded tokens — `{cls}` (n={len(sub)})\n")
        md.append("```")
        for tok, cnt in vc.items():
            md.append(f"  {tok!r:<28s}  {cnt}")
        md.append("```\n")
    (out_dir / "checkB_word_distributions.md").write_text("\n".join(md))
    print(f"wrote {out_dir / 'checkB_word_distributions.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--modality", default="av", choices=["a", "v", "av"],
                   help="Modal type passed to build_conversation/prepare_inputs.")
    p.add_argument("--media_dir", default=None,
                   help="Override the default media directory (audio/video files).")
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
