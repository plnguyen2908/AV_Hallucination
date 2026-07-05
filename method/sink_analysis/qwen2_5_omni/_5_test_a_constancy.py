"""Test A — sink-V vs content-V constancy for the 11a-targeted Inert heads.

For each (layer, head) in the Inert set at the intervened layers, compute:
  v̄_S(clip)  = mean V over sink positions
  v̄_C(clip)  = mean V over non-sink (modality content) positions

Aggregate across K clips:
  Var_clip(v̄_S) / Var_clip(v̄_C)   — silencer premise predicts ≪ 1
  mean cross-clip cosine(v̄_S)      — predicts ≈ 1 (clustered)
  mean cross-clip cosine(v̄_C)      — predicts spread

Writes summary JSON.
"""
import argparse, sys, json
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


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading model (eager) ...", flush=True)
    model, processor = load_omni(args.model_path, device_map=args.device_map,
                                    attn_implementation="eager")
    eps_norm = EX._thinker_rms_eps(model)
    audio_enc, visual_enc = EX._resolve_encoders(model)
    d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)
    EX._RUNTIME_ARGS = type("A", (), {})()
    EX._RUNTIME_ARGS.sink_mask = "all"
    EX._RUNTIME_ARGS.random_mask = False
    EX._RUNTIME_ARGS.include_text_sinks = False
    EX._RUNTIME_ARGS.modality_complement = False

    head_sets = EX.load_head_sets(args.heads_csv)
    inert = head_sets["Inert"]
    inert_by_layer = {}
    for L, h in inert:
        inert_by_layer.setdefault(L, []).append(h)

    text_cfg = model.thinker.config
    if not hasattr(text_cfg, "num_attention_heads"):
        text_cfg = getattr(text_cfg, "text_config", text_cfg)
    H = text_cfg.num_attention_heads
    D = text_cfg.hidden_size
    Hd = D // H

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

    # For each (layer, head), accumulate per-clip v̄_S and v̄_C.
    per_lh = {}  # (L, h) -> dict with lists vs / vc
    n_done = 0

    for _, r in tqdm(dev.iterrows(), total=len(dev), desc="test-A"):
        vid = str(r["video_id"]).zfill(5)
        vp = _REPO / "data/AVHBench/videos" / f"{vid}.mp4"
        if not vp.exists(): continue
        prompt = r["text"] + YES_NO_SUFFIX
        conv = build_conversation(str(vp), prompt, "av")
        routed_mod = routed.get(r["question_id"], None) or TASK_TO_GT[r["task"]]

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
                                  output_hidden_states=True, use_cache=False)
        hiddens = out.hidden_states

        for L in layers_to_probe:
            if L not in sink_mask_per_layer or L not in inert_by_layer:
                continue
            sink_mask = sink_mask_per_layer[L]
            cur_len = hiddens[L].shape[1]
            if sink_mask.numel() > cur_len:
                sink_mask = sink_mask[:cur_len]
            sink_mask_dev = sink_mask.to(hiddens[L].device)
            # Compute V at this layer (no grad)
            with torch.no_grad():
                layer = model.thinker.model.layers[L]
                ln_in = layer.input_layernorm(hiddens[L])
                v = layer.self_attn.v_proj(ln_in)
                num_kv = v.shape[-1] // Hd
                v = v.view(v.shape[0], v.shape[1], num_kv, Hd)
                group = H // num_kv
                v = v.repeat_interleave(group, dim=2)[0].transpose(0, 1).float()
                v = v.detach()
            # Modality mask = audio + video positions (any non-text, non-sys)
            # The non-sink content within modality = inverse of sink mask AND modality.
            # Simpler: content = ~sink positions (which includes text, but
            # text spans are small; OK for ratio purposes).
            sink_mask_dev = sink_mask_dev.to(torch.bool)
            non_sink = ~sink_mask_dev
            if not sink_mask_dev.any() or not non_sink.any():
                continue

            for h in inert_by_layer[L]:
                v_h = v[h]   # (S, Hd)
                v_S = v_h[sink_mask_dev].mean(dim=0).cpu().numpy()
                v_C = v_h[non_sink].mean(dim=0).cpu().numpy()
                key = (int(L), int(h))
                if key not in per_lh:
                    per_lh[key] = {"vs": [], "vc": []}
                per_lh[key]["vs"].append(v_S)
                per_lh[key]["vc"].append(v_C)
        n_done += 1

    # Aggregate
    def coscos(stack):
        # mean pairwise cosine
        stack = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-12)
        sim = stack @ stack.T
        N = sim.shape[0]
        idx = np.triu_indices(N, k=1)
        return float(sim[idx].mean())

    rows = []
    for (L, h), d in per_lh.items():
        vs = np.stack(d["vs"])  # (n_clips, Hd)
        vc = np.stack(d["vc"])
        if vs.shape[0] < 3: continue
        var_S = vs.var(axis=0).mean()
        var_C = vc.var(axis=0).mean()
        rows.append(dict(
            layer=L, head=h,
            var_S=float(var_S), var_C=float(var_C),
            ratio=float(var_S/(var_C+1e-12)),
            cos_S=coscos(vs),
            cos_C=coscos(vc),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_a.csv", index=False)
    summary = {
        "n_clips": n_done,
        "n_heads": len(df),
        "ratio_mean":      float(df["ratio"].mean()),
        "ratio_p50":       float(df["ratio"].quantile(0.5)),
        "ratio_p10":       float(df["ratio"].quantile(0.1)),
        "ratio_p90":       float(df["ratio"].quantile(0.9)),
        "cos_S_mean":      float(df["cos_S"].mean()),
        "cos_C_mean":      float(df["cos_C"].mean()),
        "interpretation":  ("Silencer premise predicts ratio ≪ 1 and "
                             "cos_S ≈ 1 (sink V tightly clustered cross clips), "
                             "cos_C ≈ 0 (content V varies clip-to-clip)."),
    }
    (out_dir / "test_a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--heads_csv",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/categorize_exp_2axis/heads.csv"))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--k_clips", type=int, default=30)
    p.add_argument("--layer_lo", type=int, default=18)
    p.add_argument("--layer_hi", type=int, default=27)
    args = p.parse_args()
    main(args)
