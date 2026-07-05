"""
stage4_head_sink_attention_bos.py — same forward as
stage4_head_sink_attention.py, with one surgical change to the
in-hook key-token binning:

    `text_and_bos`  →  `bos` + `text_sys`

where `bos` = the single first-position key (position 0 only) and
`text_sys` = all remaining non-modal, non-sink positions that used to
fall into `text_and_bos`. All other bins (`p_prop_*`, `p_llm_*`,
`nonsink_*`) are byte-identical to the prior run. The regression gate
(reproducibility of sink-bin values) is enforced by the contrast-side
post-processing.

For documentation: the token id and decoded string at position 0 of the
first successfully processed clip are printed to stdout (logged in
SUMMARY).
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

DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1_rerun/per_clip_tokens"
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_1_rerun"


def _build_layer_bins(p_prop, p_llm_L, mds_L, video_pos, audio_pos, S):
    """Bin builder with the BOS / text_sys split. All sink + nonsink
    bins are identical to the prior run (same boolean expressions).
    Only the trailing `text_and_bos` bin is split into `bos` (mask
    [True, False, False, ...]) and `text_sys` (the rest of the non-AV,
    non-sink residue)."""
    in_video = np.zeros(S, dtype=bool); in_video[video_pos] = True
    in_audio = np.zeros(S, dtype=bool); in_audio[audio_pos] = True
    in_av = in_video | in_audio
    sink = (p_prop | p_llm_L) & in_av                  # sinks live in AV spans

    text_residue = ~in_av                              # what was text_and_bos
    bos = np.zeros(S, dtype=bool); bos[0] = True       # exactly position 0
    text_sys = text_residue & ~bos                     # everything else non-AV

    bins = {
        "p_prop_cross":     p_prop & in_av & (mds_L == 0),
        "p_prop_uni_video": p_prop & in_av & (mds_L == +1),
        "p_prop_uni_audio": p_prop & in_av & (mds_L == -1),
        "p_llm_cross":      p_llm_L & in_av & (mds_L == 0),
        "p_llm_uni_video":  p_llm_L & in_av & (mds_L == +1),
        "p_llm_uni_audio":  p_llm_L & in_av & (mds_L == -1),
        "nonsink_audio":    in_audio & ~sink,
        "nonsink_video":    in_video & ~sink,
        "bos":              bos,
        "text_sys":         text_sys,
    }
    return {k: (m.copy(), int(m.sum())) for k, m in bins.items()}


def process_clip(model, processor, clip_path, dump_path, layers, n_layers,
                   bos_log: dict | None = None, tokenizer=None):
    """Same as the prior stage4 process_clip, except (a) uses the
    BOS-split bin builder and (b) populates bos_log on the first success
    with the token id/string at position 0."""
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

    # Log position-0 token once
    if bos_log is not None and "token_id" not in bos_log and tokenizer is not None:
        tid = int(inputs["input_ids"][0, 0].item())
        raw = tokenizer.convert_ids_to_tokens(tid)
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        except Exception:
            decoded = ""
        bos_log["token_id"] = tid
        bos_log["raw_bpe"] = str(raw)
        bos_log["decoded"] = str(decoded)
        bos_log["clip"] = clip_path.name

    per_layer_bins = [_build_layer_bins(p_prop, p_llm[L], mds[L],
                                          video_pos, audio_pos, S)
                       for L in range(n_layers)]
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
            aw = out[1]
            a = aw[0].float()
            inflow = a.mean(dim=1)
            H = inflow.shape[0]
            masks = per_layer_masks_t[L_idx]
            for name, (mask_cpu, n) in masks.items():
                if n == 0:
                    continue
                mask = mask_cpu.to(inflow.device)
                total = inflow[:, mask].sum(dim=1)
                per_tok = (total / max(n, 1)).cpu().numpy()
                total_np = total.cpu().numpy()
                for h_idx in range(H):
                    records.append((L_idx, h_idx, name, n,
                                     float(per_tok[h_idx]),
                                     float(total_np[h_idx])))
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

    dump_dir = Path(args.dump_dir)
    video_dir = Path(args.video_dir)
    dumps = sorted(dump_dir.glob("*.npz"))
    if not dumps:
        raise SystemExit(f"no .npz under {dump_dir}")
    print(f"\nProcessing {len(dumps)} clips ...\n")

    bos_log: dict = {}
    rows = []
    failures = 0
    for dump_path in tqdm(dumps, desc="clips"):
        clip_name = dump_path.stem + ".mp4"
        clip_path = video_dir / clip_name
        if not clip_path.exists():
            failures += 1; continue
        recs = process_clip(model, processor, clip_path, dump_path,
                              layers, n_layers,
                              bos_log=bos_log, tokenizer=processor.tokenizer)
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

    if bos_log:
        bos_path = out_dir / "bos_identification.txt"
        with open(bos_path, "w") as f:
            f.write(f"clip            : {bos_log.get('clip','')}\n")
            f.write(f"token_id @pos 0 : {bos_log.get('token_id','')}\n")
            f.write(f"raw_bpe         : {bos_log.get('raw_bpe','')}\n")
            f.write(f"decoded         : {bos_log.get('decoded','')!r}\n")
        print(f"wrote {bos_path}  (BOS = pos 0 token: {bos_log.get('raw_bpe')})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir",  default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--dump_dir",   default=str(DEFAULT_DUMP))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--device_map", default="balanced_low_0")
    args = p.parse_args()
    main(args)
