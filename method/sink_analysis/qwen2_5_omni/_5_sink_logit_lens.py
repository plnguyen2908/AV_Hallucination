"""ASD-inspired sink classification + logit-lens decoding for Qwen2.5-Omni.

For K AVHBench DEV clips:
  1. Identify sinks per layer via D_sink = {458, 2570}, RMSNorm > τ = 20.
  2. Categorize each sink by token-position modality (audio_idx / video_idx / other).
  3. Compute per-sink attention-symmetry MDS using attention from audio-query
     positions vs video-query positions (cross-modal vs uni-modal).
  4. Refine into 3 ASD-style buckets:
       a-sink:  audio-positioned sink, |MDS| > τ_mds, audio-attn dominant   (uni-modal audio)
       v-sink:  video-positioned sink, |MDS| > τ_mds, video-attn dominant   (uni-modal video)
       av-sink: ANY-positioned sink with |MDS| ≤ τ_mds (cross-modal)
  5. For each sink position at each target layer, project the hidden state
     through final_norm → lm_head → softmax, take top-k tokens.
  6. Aggregate across (clip × layer × sink position) per bucket → return
     the *distribution of meaningful tokens* per sink type.

Output:
  results/qwen2_5_omni/stage5_intervention/sink_readout/sink_logit_lens.json
  results/qwen2_5_omni/stage5_intervention/sink_readout/sink_logit_lens.md
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
    print("Loading model (eager — need output_attentions + hidden_states) ...",
          flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map,
                                    attn_implementation="eager")
    thinker_cfg = model.thinker.config
    if not hasattr(thinker_cfg, "audio_token_index"):
        thinker_cfg = getattr(thinker_cfg, "text_config", thinker_cfg)
    # num_attention_heads / hidden_size live on the text_config inside the
    # composite Qwen2_5OmniThinkerConfig, not at the top.
    text_cfg = thinker_cfg
    if not hasattr(text_cfg, "num_attention_heads"):
        text_cfg = getattr(thinker_cfg, "text_config", thinker_cfg)
    H = text_cfg.num_attention_heads
    D = text_cfg.hidden_size
    Hd = D // H
    final_norm = model.thinker.model.norm
    lm_head = model.thinker.lm_head
    tok = processor.tokenizer
    layers_to_probe = list(range(args.layer_lo, args.layer_hi + 1))
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    eps_norm = EX._thinker_rms_eps(model)

    split_df = pd.read_csv(_REPO /
        None  # placeholder,
        dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    dev = split_df[split_df.split == "DEV"].reset_index(drop=True).head(args.k_clips)
    YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."

    bucket_counter = {"a-sink": Counter(), "v-sink": Counter(),
                       "av-sink": Counter()}
    per_clip_bucket_count = {"a-sink": [], "v-sink": [], "av-sink": []}

    for _, r in tqdm(dev.iterrows(), total=len(dev), desc="sink-logit-lens"):
        vid = str(r["video_id"]).zfill(5)
        vp = _REPO / "data/AVHBench/videos" / f"{vid}.mp4"
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
        audio_pos_t = torch.tensor(audio_pos, dtype=torch.long, device=model.device)
        video_pos_t = torch.tensor(video_pos, dtype=torch.long, device=model.device)
        audio_set = set(audio_pos.tolist())
        video_set = set(video_pos.tolist())

        with torch.inference_mode():
            out = model.thinker(**inputs, use_audio_in_video=True,
                                  output_attentions=True,
                                  output_hidden_states=True, use_cache=False)
        attentions = out.attentions     # tuple (num_layers, B, H, Q, K)
        hiddens = out.hidden_states     # tuple (num_layers+1, B, S, D)

        bucket_count = {"a-sink": 0, "v-sink": 0, "av-sink": 0}
        for L in layers_to_probe:
            if L >= len(hiddens) - 1: break
            h_L = hiddens[L][0].float()  # (S, D)
            S = h_L.shape[0]
            # p_llm: max(|RMSNorm[h]_D_sink|) > τ
            rms = torch.sqrt(h_L.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
            normed_abs = (h_L / rms).abs()
            p_llm = (normed_abs[:, d_sink_t].amax(dim=-1) >= TAU_SINK)
            sinks = torch.where(p_llm)[0].cpu().tolist()
            if not sinks: continue
            # attention at this layer
            attn_L = attentions[L][0].float()  # (H, Q, K)
            audio_q_mask = torch.zeros(S, dtype=torch.bool, device=attn_L.device)
            audio_q_mask[audio_pos_t] = True
            video_q_mask = torch.zeros(S, dtype=torch.bool, device=attn_L.device)
            video_q_mask[video_pos_t] = True
            attn_audio_to_key = attn_L[:, audio_q_mask, :].mean(dim=(0, 1)) \
                if audio_q_mask.any() else torch.zeros(S, device=attn_L.device)
            attn_video_to_key = attn_L[:, video_q_mask, :].mean(dim=(0, 1)) \
                if video_q_mask.any() else torch.zeros(S, device=attn_L.device)
            # decode each sink via lm_head logit-lens
            for s in sinks:
                in_audio = s in audio_set
                in_video = s in video_set
                a_attn = float(attn_audio_to_key[s])
                v_attn = float(attn_video_to_key[s])
                denom = a_attn + v_attn + 1e-12
                mds = abs(a_attn - v_attn) / denom
                # classify
                if mds < args.mds_threshold:
                    bucket = "av-sink"
                elif in_audio and a_attn > v_attn:
                    bucket = "a-sink"
                elif in_video and v_attn > a_attn:
                    bucket = "v-sink"
                else:
                    continue  # other / text sinks
                # logit-lens
                hs = h_L[s].unsqueeze(0)
                hs_n = final_norm(hs.to(model.dtype)).float()
                logits = lm_head(hs_n.to(model.dtype)).float()[0]
                topk_idx = torch.topk(logits, k=args.top_k).indices.cpu().tolist()
                tokens = [tok.decode([i]) for i in topk_idx]
                bucket_counter[bucket].update(tokens)
                bucket_count[bucket] += 1
        for k in bucket_count: per_clip_bucket_count[k].append(bucket_count[k])

    # Output
    out_summary = {
        "n_clips": int(len(dev)),
        "layers_probed": layers_to_probe,
        "tau_sink": TAU_SINK,
        "mds_threshold": args.mds_threshold,
        "top_k_per_sink": args.top_k,
        "per_bucket_top_tokens": {},
        "per_bucket_avg_count_per_clip": {},
    }
    for bucket in ("a-sink", "v-sink", "av-sink"):
        top = bucket_counter[bucket].most_common(args.top_k_report)
        out_summary["per_bucket_top_tokens"][bucket] = top
        cnts = per_clip_bucket_count[bucket]
        out_summary["per_bucket_avg_count_per_clip"][bucket] = (
            float(sum(cnts) / max(1, len(cnts))) if cnts else 0.0)

    (out_dir / "sink_logit_lens.json").write_text(
        json.dumps(out_summary, indent=2, ensure_ascii=False))
    md_lines = ["# Sink-type logit-lens decoding (ASD-inspired)\n",
                 f"K clips: {out_summary['n_clips']}; layers {layers_to_probe[0]}–"
                 f"{layers_to_probe[-1]}; |MDS| threshold {args.mds_threshold}\n"]
    for bucket in ("a-sink", "v-sink", "av-sink"):
        md_lines.append(f"\n## {bucket}\n")
        md_lines.append(
            f"Avg sinks/clip in bucket: "
            f"{out_summary['per_bucket_avg_count_per_clip'][bucket]:.2f}\n\n")
        md_lines.append("| rank | token | freq |\n|---:|---|---:|\n")
        for rank, (tok_str, cnt) in enumerate(
                out_summary["per_bucket_top_tokens"][bucket], 1):
            md_lines.append(f"| {rank} | `{tok_str!r}` | {cnt} |\n")
    (out_dir / "sink_logit_lens.md").write_text("".join(md_lines))
    print(json.dumps(out_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--k_clips", type=int, default=30)
    p.add_argument("--layer_lo", type=int, default=14)
    p.add_argument("--layer_hi", type=int, default=23)
    p.add_argument("--mds_threshold", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=10,
                   help="top-K tokens per sink position via lm_head logit-lens.")
    p.add_argument("--top_k_report", type=int, default=40,
                   help="top-K tokens to report per bucket after aggregation.")
    args = p.parse_args()
    main(args)
