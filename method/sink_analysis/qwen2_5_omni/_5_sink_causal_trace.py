"""ASD-style causal tracing for Qwen2.5-Omni sinks.

For K AVHBench DEV clips, measure the causal effect of each sink type
(a-sink / v-sink / av-sink) on the next-token distribution at the
generated query position by *patching*: zero out the hidden state at
all positions of one sink type at a target layer, then measure the
shift in next-token probability.

For each sink type, we report:
  - Top-K tokens whose probability DROPS the most under patching
    (i.e. the most likely "meaningful" tokens that sink contributes to).
  - Per-clip and aggregate statistics.

This is the ASD adapter for Qwen2.5-Omni — distinct from logit-lens
which is run by `_5_sink_logit_lens.py`. The two together give a
complementary picture of what each sink type encodes.
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/sink_analysis/qwen2_5_omni"))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
import _5_explore as EX
from utils import build_conversation, load_omni, thinker_layers  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

DEFAULT_OUT = _REPO / "results/qwen2_5_omni/stage5_intervention/sink_readout"
D_SINK = [458, 2570]
TAU_SINK = 20.0


def _modal_positions(input_ids, thinker_cfg):
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading model (eager) ...", flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map,
                                    attn_implementation="eager")
    thinker_cfg = model.thinker.config
    if not hasattr(thinker_cfg, "audio_token_index"):
        thinker_cfg = getattr(thinker_cfg, "text_config", thinker_cfg)
    tok = processor.tokenizer
    layers = thinker_layers(model)
    n_layers = len(layers)
    layer_targets = list(range(args.layer_lo, args.layer_hi + 1))
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    eps_norm = EX._thinker_rms_eps(model)

    if args.dataset == "vggsounder":
        import json as _json
        qa = _json.load(open(_REPO / "data/VGGSounder/QA.json"))
        rows = []
        for q in qa[:args.k_clips]:
            vid = q["video_id"]
            if isinstance(vid, str) and not vid.endswith(".mp4"):
                vid = vid + ".mp4"
            rows.append(dict(video_id=vid, text=q["text"], label=q.get("label")))
        dev = pd.DataFrame(rows)
        # Match the eval.py DESCRIBE_SUFFIX_BY_TASK["VGGSounder Captioning"]
        prompt_suffix = ("\nRespond with ONLY a comma-separated list of "
                          "labels from the list above that match what you "
                          "see and hear. No explanations, no other words.")
        media_dir = _REPO / "data/VGGSounder/videos"
    else:
        split_df = pd.read_csv(_REPO /
            "results/qwen2_5_omni/stage5_intervention/split.csv",
            dtype={"video_id": str})
        split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
        split_df["video_id"] = split_df["video_id"].astype(str) + ".mp4"
        dev = split_df[split_df.split == "DEV"].reset_index(drop=True).head(args.k_clips)
        prompt_suffix = " Answer with only 'Yes' or 'No'."
        media_dir = _REPO / "data/AVHBench/videos"
    YES_NO_SUFFIX = prompt_suffix

    bucket_drop = {"a-sink": Counter(), "v-sink": Counter(),
                    "av-sink": Counter()}
    bucket_total_count = {"a-sink": 0, "v-sink": 0, "av-sink": 0}

    for _, r in tqdm(dev.iterrows(), total=len(dev), desc="causal-trace"):
        vid = str(r["video_id"])
        vp = media_dir / vid
        if not vp.exists(): continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        text = processor.apply_chat_template(conv, add_generation_prompt=True,
                                                tokenize=False)
        if isinstance(text, list): text = text[0]
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                            return_tensors="pt", padding=True,
                            use_audio_in_video=True)
        inputs = {k: (v.to(model.device).to(model.dtype)
                           if torch.is_floating_point(v) else v.to(model.device))
                       if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        audio_pos, video_pos = _modal_positions(inputs["input_ids"], thinker_cfg)
        audio_set = set(audio_pos.tolist())
        video_set = set(video_pos.tolist())
        S = inputs["input_ids"].shape[1]

        # ---- Pass 1: original forward to identify sinks + record P_orig ----
        with torch.inference_mode():
            out = model.thinker(**inputs, use_audio_in_video=True,
                                  output_attentions=True,
                                  output_hidden_states=True, use_cache=False)
        attentions = out.attentions
        hiddens = out.hidden_states
        next_logits_orig = out.logits[0, -1, :].float()
        P_orig = torch.softmax(next_logits_orig, dim=-1)
        del out

        # ---- categorize sinks per layer ----
        sinks_by_bucket = {"a-sink": set(), "v-sink": set(), "av-sink": set()}
        for L in layer_targets:
            if L >= len(hiddens) - 1: break
            h_L = hiddens[L][0].float()
            rms = torch.sqrt(h_L.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
            normed_abs = (h_L / rms).abs()
            p_llm = (normed_abs[:, d_sink_t].amax(dim=-1) >= TAU_SINK)
            sinks = torch.where(p_llm)[0].cpu().tolist()
            if not sinks: continue
            attn_L = attentions[L][0].float()
            audio_q_mask = torch.zeros(S, dtype=torch.bool, device=attn_L.device)
            audio_q_mask[torch.tensor(audio_pos, device=attn_L.device)] = True
            video_q_mask = torch.zeros(S, dtype=torch.bool, device=attn_L.device)
            video_q_mask[torch.tensor(video_pos, device=attn_L.device)] = True
            attn_a = attn_L[:, audio_q_mask, :].mean(dim=(0, 1)) \
                if audio_q_mask.any() else torch.zeros(S, device=attn_L.device)
            attn_v = attn_L[:, video_q_mask, :].mean(dim=(0, 1)) \
                if video_q_mask.any() else torch.zeros(S, device=attn_L.device)
            for s in sinks:
                a = float(attn_a[s]); v = float(attn_v[s])
                denom = a + v + 1e-12
                mds = abs(a - v) / denom
                if mds < args.mds_threshold:
                    sinks_by_bucket["av-sink"].add(s)
                elif s in audio_set and a > v:
                    sinks_by_bucket["a-sink"].add(s)
                elif s in video_set and v > a:
                    sinks_by_bucket["v-sink"].add(s)

        # Free attention memory
        del hiddens, attentions
        torch.cuda.empty_cache()

        # ---- For each non-empty bucket, patch (zero) at the patch_layer ----
        patch_layer = args.patch_layer
        decoder_layer = layers[patch_layer]
        for bucket, positions in sinks_by_bucket.items():
            if not positions: continue
            pos_idx = torch.tensor(sorted(positions), dtype=torch.long,
                                     device=model.device)
            bucket_total_count[bucket] += len(positions)

            def zero_patch(_m, _input, output):
                # output is either Tensor or tuple — Qwen layer returns
                # (hidden_states, ...) for some versions.
                if isinstance(output, tuple):
                    hs = output[0]
                    hs[..., pos_idx, :] = 0.0
                    return (hs,) + output[1:]
                else:
                    output[..., pos_idx, :] = 0.0
                    return output

            handle = decoder_layer.register_forward_hook(zero_patch)
            try:
                with torch.inference_mode():
                    out_p = model.thinker(**inputs, use_audio_in_video=True,
                                            use_cache=False)
                next_logits_patched = out_p.logits[0, -1, :].float()
                P_patched = torch.softmax(next_logits_patched, dim=-1)
            finally:
                handle.remove()

            dP = (P_orig - P_patched).cpu()
            # Tokens whose probability DROPPED under patching (= caused by sink)
            top_drop = torch.topk(dP, k=args.top_k).indices.cpu().tolist()
            for tid in top_drop:
                tok_str = tok.decode([tid])
                bucket_drop[bucket][tok_str] += 1

    summary = {
        "n_clips": int(len(dev)),
        "layer_targets": layer_targets,
        "patch_layer": args.patch_layer,
        "mds_threshold": args.mds_threshold,
        "tau_sink": TAU_SINK,
        "per_bucket_total_positions": bucket_total_count,
        "per_bucket_top_dropped_tokens": {},
    }
    for bucket in ("a-sink", "v-sink", "av-sink"):
        summary["per_bucket_top_dropped_tokens"][bucket] = (
            bucket_drop[bucket].most_common(args.top_k_report))

    (out_dir / "sink_causal_trace.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# Causal-trace per sink type (zero-patch at layer "
          f"{args.patch_layer})\n",
          f"K clips: {summary['n_clips']}; layers probed for sink ID: "
          f"{layer_targets[0]}–{layer_targets[-1]}; |MDS| τ={args.mds_threshold}\n"]
    for bucket in ("a-sink", "v-sink", "av-sink"):
        md.append(f"\n## {bucket}\n")
        md.append(f"Total sink positions patched (sum over clips): "
                  f"{summary['per_bucket_total_positions'][bucket]}\n\n")
        md.append("| rank | token | clips affected |\n|---:|---|---:|\n")
        for r, (t, c) in enumerate(
                summary["per_bucket_top_dropped_tokens"][bucket], 1):
            md.append(f"| {r} | `{t!r}` | {c} |\n")
    (out_dir / "sink_causal_trace.md").write_text("".join(md))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--k_clips", type=int, default=20)
    p.add_argument("--layer_lo", type=int, default=14)
    p.add_argument("--layer_hi", type=int, default=23)
    p.add_argument("--patch_layer", type=int, default=18,
                   help="Layer at which to zero-patch hidden states for causal trace.")
    p.add_argument("--mds_threshold", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=15,
                   help="top-K tokens with largest P drop per (clip, bucket).")
    p.add_argument("--top_k_report", type=int, default=40)
    p.add_argument("--dataset", default="avhbench",
                   choices=["avhbench", "vggsounder"])
    args = p.parse_args()
    main(args)
