"""CONTROL 2(a) — project the sink-restricted attention output of every
Inert head onto the (Yes − No) logit direction.

For each DEV clip and the late-layer Inert heads:
  attn_out_sink_h = (W_o[:, h, :]) @ ( Σ_k∈sinks attn_h[k] · V_h[k] )
  ⇒ final_norm ⇒ lm_head ⇒ pick logits at tok("Yes") and tok("No")

Aggregate Δlogit_yes_minus_no across (clip, layer, head).
Output a JSON summary + a CSV.
"""
import argparse, sys
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


def get_layer(model, L):
    return model.thinker.model.layers[L]


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading model (eager) ...", flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map,
                                    attn_implementation="eager")
    eps_norm = EX._thinker_rms_eps(model)
    audio_enc, visual_enc = EX._resolve_encoders(model)
    d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)
    EX._RUNTIME_ARGS = type("A", (), {})()
    EX._RUNTIME_ARGS.sink_mask = args.sink_mask
    EX._RUNTIME_ARGS.random_mask = False
    EX._RUNTIME_ARGS.include_text_sinks = False
    EX._RUNTIME_ARGS.modality_complement = False

    head_sets = EX.load_head_sets(args.heads_csv)
    inert = head_sets["Inert"]
    inert_by_layer = {}
    for L, h in inert:
        inert_by_layer.setdefault(L, []).append(h)

    yes_tok = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_tok = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"tok('Yes')={yes_tok}, tok('No')={no_tok}", flush=True)

    text_cfg = model.thinker.config
    if not hasattr(text_cfg, "num_attention_heads"):
        text_cfg = getattr(text_cfg, "text_config", text_cfg)
    H = text_cfg.num_attention_heads
    D = text_cfg.hidden_size
    Hd = D // H
    final_norm = model.thinker.model.norm
    lm_head = model.thinker.lm_head

    layers_to_probe = list(range(args.layer_lo, args.layer_hi + 1))

    split_df = pd.read_csv(_REPO /
        "results/qwen2_5_omni/stage5_intervention/split.csv",
        dtype={"video_id": str})
    split_df["video_id"] = split_df["video_id"].astype(str).str.zfill(5)
    dev = split_df[split_df.split == "DEV"].reset_index(drop=True).head(args.k_clips)

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
    n_done = 0
    for _, r in tqdm(dev.iterrows(), total=len(dev), desc="probe-C2a"):
        vid = str(r["video_id"]).zfill(5)
        vp = _REPO / "data/AVHBench/videos" / f"{vid}.mp4"
        if not vp.exists(): continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        routed_mod = routed.get(r["question_id"], None) or TASK_TO_GT[r["task"]]

        # Sink masks
        sink_mask_per_layer, _S = EX.compute_per_layer_sink_masks(
            model, processor, conv, True,
            thinker_layers(model), audio_enc, visual_enc, eps_norm, d_sink_t,
            routed_mod)

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
                                  output_attentions=True,
                                  output_hidden_states=True, use_cache=False)
        attentions = out.attentions
        hiddens = out.hidden_states

        # baseline-attention (unboosted) — compute the constant "head sink
        # contribution" per Inert head at q_pos = -1.
        q_pos = -1
        for L in layers_to_probe:
            if L not in sink_mask_per_layer or L not in inert_by_layer:
                continue
            sink_mask = sink_mask_per_layer[L].to(attentions[L].device)
            layer = get_layer(model, L)
            ln_in = layer.input_layernorm(hiddens[L])
            v_proj_w = layer.self_attn.v_proj
            v = v_proj_w(ln_in)
            num_kv = v.shape[-1] // Hd
            v = v.view(v.shape[0], v.shape[1], num_kv, Hd)
            group = H // num_kv
            v = v.repeat_interleave(group, dim=2)[0].transpose(0, 1).float()
            o_proj_w = layer.self_attn.o_proj.weight.data.float().view(D, H, Hd)
            attn_L = attentions[L][0, :, q_pos, :].float()

            for h in inert_by_layer[L]:
                attn_h = attn_L[h]
                sink_attn_h = attn_h * sink_mask.float()
                v_h = v[h]
                sink_out_h = sink_attn_h @ v_h
                # head's contribution to residual through W_o block
                head_out = o_proj_w[:, h, :] @ sink_out_h  # (D,)
                head_out_n = final_norm(
                    head_out.unsqueeze(0).to(model.dtype)).squeeze(0).float()
                logits = lm_head(head_out_n.to(model.dtype)).float()
                yes_logit = float(logits[yes_tok])
                no_logit = float(logits[no_tok])
                rows.append(dict(
                    video_id=vid, question_id=r["question_id"], task=r["task"],
                    label=r["label"], routed=routed_mod, layer=L, head=h,
                    yes_logit=yes_logit, no_logit=no_logit,
                    yes_minus_no=yes_logit - no_logit))
        n_done += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_c2a.csv", index=False)
    summary = {
        "n_clips": n_done,
        "n_inert_head_observations": len(df),
        "yes_minus_no_mean": float(df["yes_minus_no"].mean()),
        "yes_minus_no_std":  float(df["yes_minus_no"].std()),
        "yes_minus_no_p50":  float(df["yes_minus_no"].quantile(0.5)),
        "yes_minus_no_p10":  float(df["yes_minus_no"].quantile(0.1)),
        "yes_minus_no_p90":  float(df["yes_minus_no"].quantile(0.9)),
        "per_layer_mean": df.groupby("layer")["yes_minus_no"].mean().to_dict(),
        "yes_label_mean": float(df[df["label"]=="Yes"]["yes_minus_no"].mean()),
        "no_label_mean":  float(df[df["label"]=="No"]["yes_minus_no"].mean()),
    }
    import json
    (out_dir / "probe_c2a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--heads_csv",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--sink_mask", default="all",
                   choices=["llm_emerged", "all", "prop"])
    p.add_argument("--k_clips", type=int, default=30)
    p.add_argument("--layer_lo", type=int, default=18)
    p.add_argument("--layer_hi", type=int, default=27)
    args = p.parse_args()
    main(args)
