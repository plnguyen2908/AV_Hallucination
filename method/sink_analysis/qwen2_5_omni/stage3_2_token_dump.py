"""
stage3_2_token_dump.py — Part 2 of Stage 3.2 (per-token sink + cell dump).

Reuses the Stage 3.1 forward path (process_clip) verbatim — the pre-SA
hook, D_sink={458,2570}, τ=20, RMSNorm gate are all inherited unmodified.
Adds per-clip .npz writes containing token-level info:
  p_llm     (n_layers, S) bool        — LLM-emerged sink mask per layer
  p_prop    (S,)         bool        — propagated sink mask (fixed)
  mds_cell  (n_layers, S) int8       — +1 uni_video, -1 uni_audio, 0 cross
  video_pos (n_video,)   int64       — positions of video tokens in S
  audio_pos (n_audio,)   int64       — positions of audio tokens in S
  S         int64                    — sequence length
  clip      <S len>      str         — clip basename
  mds_v_thresh (n_layers,) float64   — span-median thresholds (for audit)
  mds_a_thresh (n_layers,) float64

All cells of cell-classification are computed here with the SAME formula
Stage 3.1 used (per per_clip_mds): span-median mds_i thresholds; uni-video
iff mds_i > mds_v_thresh[L]; uni-audio iff mds_i < mds_a_thresh[L]; else
cross. Stage 3.1's own outputs are NOT touched.

Runtime: ~2 min for 50 VGGSounder clips on 4 GPUs (same as Stage 3.1).

Output: <output_dir>/per_clip_tokens/<clip_stem>.npz  (50 files)
       + <output_dir>/per_clip_tokens/_manifest.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

# Reuse Stage 3.1 helpers (do NOT re-implement; identical sink/φ logic).
from stage3_1_modality_detection_score import (  # noqa: E402
    D_SINK, TAU_SINK, TAU_PROP,
    _resolve_thinker_cfg, _thinker_rms_eps, _resolve_encoders,
    process_clip, DEFAULT_VIDEO_DIR,
)
from utils import load_omni, thinker_layers  # noqa: E402

DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"


def cells_from_attn(attn_v, attn_a, video_pos, audio_pos):
    """Classify every (layer, token) into +1 / -1 / 0 with the Stage 3.1
    formula. Same code path as per_clip_mds — replicated here so that
    Stage 3.1's script remains untouched. Returns:
        mds_cell  (n_layers, S) int8
        mds_v_th  (n_layers,)   float64
        mds_a_th  (n_layers,)   float64
    """
    eps = 1e-12
    mds_i = (attn_v - attn_a) / (attn_v + attn_a + eps)
    n_layers, S = mds_i.shape
    video_idx = np.zeros(S, dtype=bool); video_idx[video_pos] = True
    audio_idx = np.zeros(S, dtype=bool); audio_idx[audio_pos] = True
    mds_v_th = np.median(mds_i[:, video_idx], axis=1)
    mds_a_th = np.median(mds_i[:, audio_idx], axis=1)
    cond_uv = mds_i > mds_v_th[:, None]
    cond_ua = mds_i < mds_a_th[:, None]
    cell = cond_uv.astype(np.int8) - cond_ua.astype(np.int8)
    return cell, mds_v_th, mds_a_th


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Qwen2.5-Omni ...")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 1 and args.device_map != "auto":
        args.device_map = "auto"
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    eps_norm = _thinker_rms_eps(model)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    audio_enc, visual_enc = _resolve_encoders(model)
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ_sink={TAU_SINK}, "
          f"τ_prop={TAU_PROP}, eps={eps_norm}")

    video_dir = Path(args.video_dir)
    clips_all = sorted(video_dir.glob("*.mp4"))
    if not clips_all:
        raise SystemExit(f"no .mp4 in {video_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"\n{len(clips)} clips selected (seed={args.seed})\n")

    manifest_lines = []
    failures: dict = {}
    for clip in tqdm(clips, desc="clips"):
        result, err = process_clip(model, processor, clip, thinker_cfg, layers,
                                    audio_enc, visual_enc, eps_norm, d_sink_t)
        if result is None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {clip.name}: {err}")
            manifest_lines.append(f"SKIP\t{clip.name}\t{err}")
            continue

        cell, mds_v_th, mds_a_th = cells_from_attn(
            result["attn_from_video"], result["attn_from_audio"],
            result["video_pos"], result["audio_pos"])

        out_path = out_dir / f"{clip.stem}.npz"
        np.savez_compressed(
            out_path,
            p_llm        = result["p_llm"].astype(bool),
            p_prop       = result["p_prop"].astype(bool),
            mds_cell     = cell.astype(np.int8),
            video_pos    = np.asarray(result["video_pos"], dtype=np.int64),
            audio_pos    = np.asarray(result["audio_pos"], dtype=np.int64),
            S            = np.int64(result["S"]),
            mds_v_thresh = mds_v_th,
            mds_a_thresh = mds_a_th,
            clip         = np.array(clip.name),
        )
        manifest_lines.append(f"OK\t{clip.name}\tS={result['S']}\t"
                              f"n_v={len(result['video_pos'])}\t"
                              f"n_a={len(result['audio_pos'])}")

    if failures:
        print(f"  failures: {failures}")
    manifest = out_dir / "_manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nwrote {len([l for l in manifest_lines if l.startswith('OK')])} per-clip .npz to {out_dir}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--n_clips", type=int, default=50,
                   help="Match Stage 3.1 (50 = smoke pass).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
