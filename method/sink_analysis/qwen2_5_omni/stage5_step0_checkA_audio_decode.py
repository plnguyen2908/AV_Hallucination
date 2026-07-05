"""
stage5_step0_checkA_audio_decode.py — CHECK A.

Re-confirm the decodability split on the exact audio sink sets that
Stage 5 (if built) would target: at L27 (the last decoder layer where
the logit lens is near-exact), do P_prop-audio sinks decode to coherent
auditory descriptors while P_llm-audio sinks decode to BPE noise?

Method:
- For each of the 37 paired sampled_entities clips:
  * Build the prompt + teacher-force the generated caption.
  * Forward thinker; capture h_input[L] for L=0..n_layers-1 via pre-hooks
    (compute P_llm masks per layer using D_sink + τ_sink = 20).
  * Capture h_output[27] via a forward hook on layer 27 (the residual
    stream just before final_norm — the canonical logit-lens state).
  * Compute encoder L2 norms → P_prop mask on audio span (Sink-or-Not
    τ_prop = 100).
  * For each audio position p classify into:
      - P_prop only
      - P_llm only (at L27)
      - both
      - non-sink
  * Logit-lens at L27 (final_norm + lm_head) → top-K vocab.
- Aggregate per category: top-K vocabulary token counts; fraction of
  clips with at least one "auditory descriptor" in top-K (curated word
  list, EN + ZH).

Pass: P_prop-audio decode coherently to auditory descriptors (≥ baseline
non-sink rate); P_llm-audio decode is dominated by BPE noise.

Outputs:
  checkA_audio_decode.csv     per (clip, sink_class) audio-descriptor flag
                                + top-1 decoded
  checkA_audio_decode.md      qualitative summary + aggregate rates
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
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_step0"

D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0
AUDIO_TOKEN_ID = 151646
VIDEO_TOKEN_ID = 151656
TOP_K = 10

# Curated auditory-descriptor word list (EN core + ZH 声音/音频; lowercase ASCII match for EN).
AUDITORY_WORDS_EN = {
    "sound", "noise", "music", "audio", "voice", "voices", "song", "songs",
    "speech", "speak", "speaks", "speaking", "talk", "talking",
    "ring", "rings", "ringing", "bell", "bells", "chime",
    "drum", "drums", "drumming", "beat", "beats", "rhythm",
    "tone", "tones", "hum", "humming", "buzz", "buzzing", "hiss", "hissing",
    "whistle", "whistling", "clang", "click", "clicking",
    "bark", "barks", "barking", "bowwow", "howl", "howling",
    "meow", "meowing", "purr", "purring",
    "chirp", "chirping", "tweet", "tweeting", "song",
    "engine", "honk", "honking", "siren",
    "loud", "quiet", "silent",
    "sing", "singing", "play", "playing",
    "audio", "acoustic", "auditory", "sonic", "phonic",
    "听", "声音", "音频", "音乐", "嗓音", "嘈杂", "响",     # ZH
}
_TOK_RE = re.compile(r"^[A-Za-z]+$")


def is_auditory_descriptor(decoded: str) -> bool:
    s = decoded.strip().lower()
    if not s: return False
    # English: at-least-one auditory-related token (after lower + strip)
    if s in AUDITORY_WORDS_EN: return True
    # Chinese / multi-char tokens: check ZH words
    for w in AUDITORY_WORDS_EN:
        if not _TOK_RE.match(w):     # non-ASCII (ZH) words
            if w in s: return True
    return False


def is_content_token(decoded: str) -> bool:
    """Loose content check: alphabetic word piece (>=2 letters), not pure
    punct/digits. Matches Stage 3.3 convention."""
    s = decoded.strip()
    if not s: return False
    if s.startswith("<|") and s.endswith("|>"): return False
    if not re.search(r"[A-Za-z一-鿿]", s): return False
    if len(re.sub(r"[^A-Za-z一-鿿]", "", s)) < 2: return False
    return True


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


def process_clip(d, model, processor, layers, n_layers,
                  audio_enc, eps_norm, final_norm, lm_head, tokenizer,
                  out_records, out_per_token):
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists(): return "missing_video"

    conv = build_conversation(str(video_path), d["question"], "av")
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "av",
                                           model.device, model.dtype)
    except Exception as e:
        return f"prep:{type(e).__name__}"
    prompt_S = int(inputs["input_ids"].shape[1])

    # Forward with hooks: capture h_input[L] (pre-SA) + h_output[27] (post-block)
    # + encoder L2 norms for P_prop.
    per_layer_h = [None] * n_layers
    last_out = [None]
    a_enc_buf = []

    def pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach()
        return _h
    def last_post_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] > 1:
            last_out[0] = hs[0].detach()
        return out
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        a_enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())

    handles = []
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(pre_hook(L)))
    handles.append(layers[n_layers - 1].register_forward_hook(last_post_hook))
    if audio_enc is not None:
        handles.append(audio_enc.register_forward_hook(enc_hook))

    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return f"fwd:{type(e).__name__}"
    finally:
        for h in handles: h.remove()

    if any(h is None for h in per_layer_h) or last_out[0] is None:
        return "no_capture"

    # P_llm masks
    p_llm_masks = _build_p_llm_per_layer(per_layer_h, n_layers, eps_norm)
    per_layer_h = [None] * n_layers
    torch.cuda.empty_cache()

    # Modal positions in the prompt
    ids = inputs["input_ids"][0].cpu().numpy()[:prompt_S]
    audio_pos = np.where(ids == AUDIO_TOKEN_ID)[0].astype(np.int64)
    if audio_pos.size == 0: return "no_audio_tokens"

    # P_prop on audio span
    p_prop_audio = np.zeros(prompt_S, dtype=bool)
    if a_enc_buf:
        a_enc = np.concatenate(a_enc_buf)
        aligned = _align_norms(a_enc, len(audio_pos))
        if aligned is not None:
            p_prop_audio[audio_pos] = aligned > TAU_PROP

    # Logit lens at L27 (post-block residual)
    h27 = last_out[0]                           # (S, D)
    dev_lm = lm_head.weight.device
    dt_lm  = lm_head.weight.dtype
    h27 = h27.to(device=dev_lm, dtype=dt_lm)
    with torch.no_grad():
        h_normed = final_norm(h27)
        logits = lm_head(h_normed).float()      # (S, V)
        top_idx = logits.topk(TOP_K, dim=-1).indices.cpu().numpy()

    # Per audio position: classify and decode top-K
    last_L = n_layers - 1
    p_llm_audio = p_llm_masks[last_L]            # bool length prompt_S
    cls_summary = dict(p_prop_only=0, p_llm_only=0, both=0, non_sink=0)
    aud_desc_summary = dict(p_prop_only=0, p_llm_only=0, both=0, non_sink=0)
    content_summary = dict(p_prop_only=0, p_llm_only=0, both=0, non_sink=0)
    for p in audio_pos:
        is_prop = bool(p_prop_audio[int(p)])
        is_llm  = bool(p_llm_audio[int(p)])
        cls = ("both" if (is_prop and is_llm)
               else "p_prop_only" if is_prop
               else "p_llm_only" if is_llm
               else "non_sink")
        cls_summary[cls] += 1
        # Decode top-K
        top_ids = top_idx[int(p)]
        top_str = [tokenizer.decode([int(t)], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
                    for t in top_ids]
        is_aud_desc = any(is_auditory_descriptor(s) for s in top_str)
        is_content  = any(is_content_token(s) for s in top_str)
        if is_aud_desc: aud_desc_summary[cls] += 1
        if is_content:  content_summary[cls] += 1
        out_per_token.append((d["video"], int(p), cls, int(top_ids[0]),
                               top_str[0], int(is_aud_desc), int(is_content)))
    for cls, n in cls_summary.items():
        if n == 0: continue
        out_records.append((d["video"], cls, int(n),
                              int(aud_desc_summary[cls]),
                              int(content_summary[cls]),
                              float(aud_desc_summary[cls] / n),
                              float(content_summary[cls] / n)))
    return None


def main(args):
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
    audio_enc, _ = _resolve_encoders(model)
    print(f"n_layers={n_layers}; audio_enc={'OK' if audio_enc else 'MISSING'}; "
          f"vocab={lm_head.weight.shape[0]}")

    data = json.load(open(args.sampled_entities_json))
    paired = [d for d in data
              if d.get("hallucinated_tokens") and d.get("non_hallucinated_tokens")]
    print(f"{len(paired)} paired clips")

    records, per_token = [], []
    failures = {}
    for d in tqdm(paired, desc="clips"):
        err = process_clip(d, model, processor, layers, n_layers,
                            audio_enc, eps_norm, final_norm, lm_head,
                            processor.tokenizer, records, per_token)
        if err:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {d['video']}: {err}")
    if failures: print(f"failures: {failures}")
    print(f"records={len(records)}; per_token={len(per_token)}")

    df = pd.DataFrame(records, columns=[
        "clip", "sink_class", "n_tokens",
        "n_auditory_descriptor", "n_content", "auditory_rate", "content_rate"])
    df.to_csv(out_dir / "checkA_audio_decode.csv", index=False)
    per_df = pd.DataFrame(per_token, columns=[
        "clip", "position", "sink_class", "top1_id", "top1_decoded",
        "is_auditory", "is_content"])
    per_df.to_csv(out_dir / "checkA_audio_decode_per_token.csv", index=False)

    # Aggregate
    agg = (df.groupby("sink_class", as_index=False)
              .agg(n_tokens_total=("n_tokens", "sum"),
                   n_clips_with_class=("clip", "nunique"),
                   mean_auditory_rate=("auditory_rate", "mean"),
                   mean_content_rate=("content_rate", "mean"),
                   sum_auditory=("n_auditory_descriptor", "sum"),
                   sum_content=("n_content", "sum")))
    agg["overall_auditory_rate"] = agg["sum_auditory"] / agg["n_tokens_total"].clip(lower=1)
    agg["overall_content_rate"]  = agg["sum_content"]  / agg["n_tokens_total"].clip(lower=1)
    print("\n=== aggregate by sink class ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Top decoded tokens per class
    print("\n=== top decoded tokens per sink class (top-1, audio positions) ===")
    md = ["# Stage 5 Step 0 — CHECK A: audio sink decodability at L27\n",
          "Per-clip + aggregate rates of `auditory_descriptor` and `content_token` "
          "in the top-10 L27 logit-lens decode of audio-span positions.\n",
          "## Aggregate\n",
          agg.to_markdown(index=False, floatfmt=".3f")]
    for cls in ("p_prop_only", "p_llm_only", "both", "non_sink"):
        sub = per_df[per_df["sink_class"] == cls]
        if sub.empty: continue
        vc = sub["top1_decoded"].value_counts().head(20)
        md.append(f"\n## Top-20 top-1 decoded tokens — `{cls}` (n={len(sub)} positions)\n")
        md.append("```")
        for tok, cnt in vc.items():
            md.append(f"  {tok!r:<28s}  {cnt}")
        md.append("```")
        print(f"\n--- {cls} (n={len(sub)}) ---")
        print(vc.head(8).to_string())
    (out_dir / "checkA_audio_decode.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--sampled_entities_json", default=str(DEFAULT_QA))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
