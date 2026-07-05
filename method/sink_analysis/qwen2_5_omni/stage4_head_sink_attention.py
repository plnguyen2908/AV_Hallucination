"""
stage4_head_sink_attention.py — head × sink-type per-token attention matrix.

For each head h at its own layer L, capture full attention weights
A_h[q, k] during a single forward pass per clip, then bin keys k by
(population × sink class) and reference bins (nonsink audio / video /
text-and-BOS). Reduce IN-HOOK to per-bin scalars so memory stays bounded
(28 layers × 28 heads × per-bin scalar instead of 28 × 28 × 2040²).

Operationalization (matches Stage 3.1 attn-extraction convention):
  inflow_h[k] = mean over query positions q of A_h[q, k]
  per_token[bin] = (sum over k in bin of inflow_h[k]) / n_tokens(bin)
  total_mass[bin] = sum over k in bin of inflow_h[k]

Sink masks (per layer L, per clip) come from
`stage3_2/per_clip_tokens/*.npz`:
  P_prop sinks      = p_prop                              (S,)  fixed
  P_llm  sinks at L = p_llm[L]                            (S,)
  MDS cell at L     = mds_cell[L]   ∈ {+1, -1, 0}

Bins emitted per (clip, layer, head):
  p_prop_cross     p_prop_uni_video     p_prop_uni_audio
  p_llm_cross      p_llm_uni_video      p_llm_uni_audio
  nonsink_audio    nonsink_video        text_and_bos

Output:
  stage4/head_sink_attention.csv
  cols: clip, layer, head, bin, n_tokens, per_token_inflow, total_inflow
"""
import argparse
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

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402

PROMPT_AV = "Describe what you see and hear in detail."

DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage4"


def _build_layer_bins(p_prop, p_llm_L, mds_L, video_pos, audio_pos, S):
    """Return dict {bin_name: (bool_mask shape (S,), n_tokens int)}.
    Bin definitions documented in module docstring."""
    in_video = np.zeros(S, dtype=bool); in_video[video_pos] = True
    in_audio = np.zeros(S, dtype=bool); in_audio[audio_pos] = True
    in_av = in_video | in_audio
    sink = (p_prop | p_llm_L) & in_av  # sinks live in AV spans only

    bins = {
        "p_prop_cross":     p_prop & in_av & (mds_L == 0),
        "p_prop_uni_video": p_prop & in_av & (mds_L == +1),
        "p_prop_uni_audio": p_prop & in_av & (mds_L == -1),
        "p_llm_cross":      p_llm_L & in_av & (mds_L == 0),
        "p_llm_uni_video":  p_llm_L & in_av & (mds_L == +1),
        "p_llm_uni_audio":  p_llm_L & in_av & (mds_L == -1),
        "nonsink_audio":    in_audio & ~sink,
        "nonsink_video":    in_video & ~sink,
        "text_and_bos":     ~in_av,
    }
    return {k: (m.copy(), int(m.sum())) for k, m in bins.items()}


def process_clip(model, processor, clip_path, dump_path, layers, n_layers):
    """Per-clip forward; returns a list of per-(layer, head, bin) dicts,
    or None on failure."""
    dump = np.load(dump_path, allow_pickle=True)
    p_prop = dump["p_prop"]
    p_llm  = dump["p_llm"]
    mds    = dump["mds_cell"]
    video_pos = dump["video_pos"]
    audio_pos = dump["audio_pos"]
    dump_S = int(dump["S"])

    conv = build_conversation(str(clip_path), PROMPT_AV, "av")
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "av",
                                           model.device, model.dtype)
    except Exception:
        return None
    S = int(inputs["input_ids"].shape[1])
    if S != dump_S:
        return None

    # Build per-layer masks (CPU bool arrays) up front
    per_layer_bins = [_build_layer_bins(p_prop, p_llm[L], mds[L],
                                          video_pos, audio_pos, S)
                       for L in range(n_layers)]
    # Convert to torch tensors lazily (we'll move to hook's device on first use)
    per_layer_masks_t = [None] * n_layers
    for L in range(n_layers):
        d = {}
        for name, (mask, n) in per_layer_bins[L].items():
            d[name] = (torch.from_numpy(mask), n)
        per_layer_masks_t[L] = d

    records: list = []

    def make_hook(L_idx):
        def _h(_m, _i, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            aw = out[1]                              # (1, H, q, k)
            a = aw[0].float()                        # (H, q, k)
            inflow = a.mean(dim=1)                   # (H, k) per-head mean-over-q inflow
            H = inflow.shape[0]
            masks = per_layer_masks_t[L_idx]
            # Move masks to attention device once per layer
            for name, (mask_cpu, n) in masks.items():
                if n == 0:
                    continue
                mask = mask_cpu.to(inflow.device)
                total = inflow[:, mask].sum(dim=1)   # (H,)
                per_tok = (total / max(n, 1)).cpu().numpy()
                total_np = total.cpu().numpy()
                for h_idx in range(H):
                    records.append((L_idx, h_idx, name, n,
                                     float(per_tok[h_idx]),
                                     float(total_np[h_idx])))
            # Drop heavy tensor (Stage 1.2 trick)
            return (out[0], None) + tuple(out[2:])
        return _h

    handles = [layers[L].self_attn.register_forward_hook(make_hook(L))
               for L in range(n_layers)]
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=True, return_dict=True,
                          use_cache=False, output_hidden_states=False)
    except Exception:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return None
    finally:
        for h in handles: h.remove()

    torch.cuda.empty_cache()
    return records


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    layers = thinker_layers(model)
    n_layers = len(layers)
    print(f"  n_layers={n_layers}; n_heads_per_layer={layers[0].self_attn.q_proj.out_features // (layers[0].self_attn.head_dim if hasattr(layers[0].self_attn,'head_dim') else 128)}")

    dump_dir = Path(args.dump_dir)
    video_dir = Path(args.video_dir)
    dumps = sorted(dump_dir.glob("*.npz"))
    if not dumps:
        raise SystemExit(f"no .npz under {dump_dir}")
    print(f"\nProcessing {len(dumps)} clips ...\n")

    rows = []
    failures = 0
    for dump_path in tqdm(dumps, desc="clips"):
        clip_name = dump_path.stem + ".mp4"
        clip_path = video_dir / clip_name
        if not clip_path.exists():
            failures += 1; continue
        recs = process_clip(model, processor, clip_path, dump_path,
                              layers, n_layers)
        if recs is None:
            failures += 1; continue
        for (L, h, name, n, pt, tot) in recs:
            rows.append(dict(clip=clip_name, layer=L, head=h, bin=name,
                              n_tokens=n, per_token_inflow=pt,
                              total_inflow=tot))
    if failures:
        print(f"  failures: {failures}")
    if not rows:
        raise SystemExit("no records collected")

    df = pd.DataFrame(rows)
    out_csv = out_dir / "head_sink_attention.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",  default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--dump_dir",   default=str(DEFAULT_DUMP))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
