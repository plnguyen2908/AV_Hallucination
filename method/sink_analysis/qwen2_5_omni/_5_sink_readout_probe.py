"""Probe B — does what an Inert head READS from a LLM-emerged sink position
decode to question-relevant content?

For each of K AVHBench DEV clips:
  1. Build conversation, prefill the thinker, run a single forward with
     output_attentions=True at the generated-token query position.
  2. Compute LLM-emerged sink positions per layer (re-using _5_explore
     compute_per_layer_sink_masks).
  3. For every Inert head (layer L, head h) at a chosen layer band:
     * sink_attn_h = attn[L, h, q_pos, :] * sink_mask_L  (zero non-sinks)
     * V_h         = value_states[L, h, :, :]           (S, head_dim)
     * sink_out_h  = sink_attn_h @ V_h                  (head_dim,)
     * head_out_h  = W_o_slice[:, h, :] @ sink_out_h    (hidden_size,)
       where W_o_slice is the o_proj's per-head block.
     * logit-lens: top-k tokens of lm_head @ head_out_h
  4. Aggregate top tokens across all Inert heads in the band → see if
     they include "Yes"/"No"/the question-keyword for the clip.

Writes: results/qwen2_5_omni/stage5_intervention/sink_readout/probe_b.csv,
        + a markdown summary report.
"""
import argparse, ast, os, re, sys, time
from pathlib import Path
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


def get_decoder_layer(model, layer_idx):
    thinker = model.thinker
    text_model = getattr(thinker, "model", thinker)
    return text_model.layers[layer_idx]


def load_heads(heads_csv):
    df = pd.read_csv(heads_csv)
    df = df[df["category"] == "Inert"]
    return [(int(r["layer"]), int(r["head"])) for _, r in df.iterrows()]


def run_one_clip(model, processor, conv, head_set_per_layer, layers_to_probe,
                  d_sink_t, eps_norm, audio_enc, visual_enc, routed_mod,
                  top_k=10):
    """Returns:
        top_tokens_per_head: dict[(L,h)] -> list[(token, prob)]
        per_layer_avg_top: dict[L] -> list[(token, prob)] aggregated over heads
        sink_mask_per_layer: dict[L] -> np.bool array of sink positions
    """
    # 1. compute sink masks (this also internally runs a forward).
    sink_mask_per_layer, S = EX.compute_per_layer_sink_masks(
        model, processor, conv, True,
        thinker_layers(model), audio_enc, visual_enc, eps_norm, d_sink_t,
        routed_mod)

    # 2. build inputs for a forward with output_attentions=True
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

    with torch.inference_mode():
        out = model.thinker(**inputs, use_audio_in_video=True,
                              output_attentions=True, use_cache=False)
    attentions = out.attentions  # tuple of len(num_layers), each (B,H,Q,K)
    q_pos = -1  # last token's row = the position about to generate

    # We also need V at each requested layer + h. Run a custom forward path
    # that returns hidden states post-RMS so we can recompute V.
    # Easiest: use the model layers' self_attn.v_proj on the pre-attention
    # hidden state we can get from output_hidden_states.
    with torch.inference_mode():
        out2 = model.thinker(**inputs, use_audio_in_video=True,
                                output_hidden_states=True, use_cache=False)
    hidden_states = out2.hidden_states  # tuple len num_layers+1

    # lm_head + the right RMSNorm before it.
    text_cfg = model.thinker.config
    if not hasattr(text_cfg, "num_attention_heads"):
        text_cfg = getattr(text_cfg, "text_config", text_cfg)
    H = text_cfg.num_attention_heads
    D = text_cfg.hidden_size
    Hd = D // H

    # lm_head is on the outer composite. It expects hidden_size -> vocab.
    lm_head = model.thinker.lm_head
    # Final norm
    final_norm = model.thinker.model.norm

    results = []  # list of dicts
    aggregate_per_layer = {}

    for L in layers_to_probe:
        if L not in sink_mask_per_layer:
            continue
        sink_mask = sink_mask_per_layer[L].to(attentions[L].device)
        # attention weights for layer L
        attn_L = attentions[L][0, :, q_pos, :].float()  # (H, S)
        # Get V at layer L using v_proj on hidden_states[L]
        layer = get_decoder_layer(model, L)
        # Pre-attn LN normalized hidden states are not directly returned;
        # input_layernorm is the rmsnorm applied before self_attn. Use it.
        ln_in = layer.input_layernorm(hidden_states[L])
        v_proj = layer.self_attn.v_proj
        v = v_proj(ln_in)  # (B, S_total, num_kv_heads*head_dim)
        # Qwen uses GQA; v has num_kv_heads * Hd channels. We need to expand
        # to H heads matching the attention shape.
        num_kv = v.shape[-1] // Hd
        # reshape and broadcast for GQA: each kv head shared by H/num_kv attn
        v = v.view(v.shape[0], v.shape[1], num_kv, Hd)  # (B,S,Hkv,Hd)
        group = H // num_kv
        v = v.repeat_interleave(group, dim=2)  # (B,S,H,Hd)
        v = v[0].transpose(0, 1)  # (H, S, Hd)
        v = v.float()

        # o_proj weight: (D, D) — view as per-head blocks
        o_proj_w = layer.self_attn.o_proj.weight.data.float()  # (D, D)
        o_proj_w_per_head = o_proj_w.view(D, H, Hd)  # output is along dim0
        # contribution to residual stream = sum_h W_o[:, h, :] @ (attn_h * V_h * mask)

        per_layer_top = []
        for (Lh, h) in head_set_per_layer.get(L, []):
            if Lh != L: continue
            attn_h = attn_L[h]  # (S,)
            sink_attn_h = attn_h * sink_mask.float()  # zero out non-sinks
            v_h = v[h]  # (S, Hd)
            sink_out_h = sink_attn_h @ v_h  # (Hd,)
            head_out_h = o_proj_w_per_head[:, h, :] @ sink_out_h  # (D,)
            # Apply final norm so it's in the same space as the residual at end.
            head_out_h_norm = final_norm(head_out_h.unsqueeze(0).to(model.dtype)
                                            ).squeeze(0).float()
            logits = lm_head(head_out_h_norm.to(model.dtype)).float()
            # top-k tokens
            topk = torch.topk(logits.softmax(dim=-1), k=top_k)
            top_tokens = [(processor.tokenizer.decode([int(idx)]),
                              float(topk.values[i]))
                          for i, idx in enumerate(topk.indices)]
            results.append(dict(layer=L, head=h,
                                  top_tokens=top_tokens))
            per_layer_top.append(top_tokens[0][0])
        aggregate_per_layer[L] = per_layer_top
    return results, aggregate_per_layer, sink_mask_per_layer


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading model (eager — need output_attentions) ...", flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map,
                                    attn_implementation="eager")
    eps_norm = EX._thinker_rms_eps(model)
    audio_enc, visual_enc = EX._resolve_encoders(model)
    d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)

    heads = load_heads(args.heads_csv)
    head_set_per_layer = {}
    for L, h in heads:
        head_set_per_layer.setdefault(L, []).append((L, h))

    # Layer band to probe
    layers_to_probe = list(range(args.layer_lo, args.layer_hi + 1))

    # Load DEV split
    split_df = pd.read_csv(_REPO /
        "results/qwen2_5_omni/stage5_intervention/split.csv",
        dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    dev_df = split_df[split_df.split == "DEV"].reset_index(drop=True)
    dev_df = dev_df.head(args.k_clips)

    router_csv = _REPO / "results/qwen2_5_omni/stage5_intervention/router_v2_dev.csv"
    routed = dict(zip(*[pd.read_csv(router_csv)[c].values
                          for c in ["question_id", "predicted"]]))
    TASK_TO_GT = {
        "Video-driven Audio Hallucination": "AUDIO",
        "Audio-driven Video Hallucination": "VISUAL",
        "AV Matching":                       "AV",
    }

    YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."
    rows = []
    md_lines = ["# Sink-Readout Probe B\n",
                 "For each clip, late-layer Inert heads' attention-output "
                 "(restricted to LLM-emerged sink positions) projected "
                 "through lm_head.\n\n"]
    for _, r in tqdm(dev_df.iterrows(), total=len(dev_df), desc="probe-B"):
        vid = str(r["video_id"]).zfill(5)
        vp = _REPO / "data/AVHBench/videos" / f"{vid}.mp4"
        if not vp.exists(): continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        routed_mod = routed.get(r["question_id"], None) or \
                       TASK_TO_GT[r["task"]]

        results, agg, _ = run_one_clip(
            model, processor, conv, head_set_per_layer,
            layers_to_probe, d_sink_t, eps_norm, audio_enc, visual_enc,
            routed_mod, top_k=args.top_k)

        # Save row + format markdown
        md_lines.append(f"## clip {vid} — {r['task']} (routed={routed_mod})\n\n")
        md_lines.append(f"**Q:** {r['text']}  **GT:** {r['label']}\n\n")
        for L in layers_to_probe:
            tops = agg.get(L, [])
            if tops:
                from collections import Counter
                c = Counter(tops)
                md_lines.append(f"- L{L:02d}: head-top-token counts: "
                                  f"{dict(c.most_common(6))}\n")
        md_lines.append("\n")
        for res in results:
            rows.append(dict(video_id=vid, question_id=r["question_id"],
                              task=r["task"], routed=routed_mod,
                              gt=r["label"],
                              layer=res["layer"], head=res["head"],
                              top1_token=res["top_tokens"][0][0],
                              top1_prob=res["top_tokens"][0][1],
                              all_top=str(res["top_tokens"])))
    pd.DataFrame(rows).to_csv(out_dir / "probe_b.csv", index=False)
    (out_dir / "probe_b.md").write_text("".join(md_lines))
    print(f"-> {out_dir / 'probe_b.csv'}")
    print(f"-> {out_dir / 'probe_b.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--heads_csv",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--k_clips", type=int, default=20,
                   help="Number of DEV clips to probe.")
    p.add_argument("--layer_lo", type=int, default=20,
                   help="Lowest layer index to probe (late band).")
    p.add_argument("--layer_hi", type=int, default=27)
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()
    main(args)
