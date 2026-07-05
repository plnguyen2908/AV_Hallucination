"""
stage5_supp_step0_sanity.py — STEP 0 sanity for the relevance-maps
supplementary task.

Three gate checks:
  (1) Grid mapping. Load a video clip, prepare inputs, derive the
      (frame, row, col) grid from video_grid_thw, round-trip a known
      token index → (t, r, c) → back, confirm match.
  (2) Temporal merging. Report whether the vision tower bundles
      multiple frames into one token (temporal_patch_size > 1).
      If yes, document the per-token frame coverage.
  (3) P_prop / non-sink populations on a single clip; iterate
      ActivityNet clips with seed=0 until one has P_prop_video ≥ 6
      AND ≥ 4 distinct frames.

STOP after this script — report numbers and wait for confirmation
before PART 1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
TAU_PROP = 100.0
VIDEO_TOKEN_ID = 151656


def _resolve_visual(model):
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model.thinker, attr):
            return getattr(model.thinker, attr)
    return None


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


def probe_clip(d, model, processor, visual_enc, max_T=4):
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists():
        return None, "missing_video"
    conv = build_conversation(str(video_path), d["question"], "v")
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, "v",
                                           model.device, model.dtype)
    except Exception as e:
        return None, f"prep:{type(e).__name__}"
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0].astype(np.int64)
    if video_pos.size == 0:
        return None, "no_video_tokens"

    # Look for video_grid_thw in inputs (Qwen2.5-VL convention)
    grid = None
    for k in ("video_grid_thw", "videos_grid_thw", "video_grid"):
        if k in inputs:
            grid = inputs[k]
            break
    grid_info = None
    if grid is not None:
        grid_cpu = grid.cpu().numpy() if hasattr(grid, "cpu") else np.asarray(grid)
        grid_info = grid_cpu
        # OOM-guard: skip clips with T > max_T BEFORE forward
        if grid_info.ndim >= 2 and int(grid_info[0, 0]) > max_T:
            return dict(skip=True, T=int(grid_info[0, 0]),
                         H=int(grid_info[0, 1]), W=int(grid_info[0, 2]),
                         n_video_tokens_llm=int(video_pos.size),
                         video_grid_thw=grid_info,
                         encoder_norms=None, n_encoder_tokens=0,
                         clip=d["video"], prompt_S=prompt_S,
                         video_pos=video_pos), None

    # Capture visual encoder norms
    enc_buf = []
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    h = visual_enc.register_forward_hook(enc_hook) if visual_enc is not None else None
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    except torch.cuda.OutOfMemoryError:
        if h: h.remove()
        torch.cuda.empty_cache()
        return None, "oom"
    finally:
        if h: h.remove()
    enc_norms = np.concatenate(enc_buf) if enc_buf else None
    torch.cuda.empty_cache()

    return dict(
        clip=d["video"],
        prompt_S=prompt_S,
        video_pos=video_pos,
        n_video_tokens_llm=int(video_pos.size),
        video_grid_thw=grid_info,
        encoder_norms=enc_norms,
        n_encoder_tokens=int(enc_norms.size) if enc_norms is not None else 0,
    ), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_search", type=int, default=20,
                   help="Max clips to scan for a qualifying one.")
    p.add_argument("--max_T", type=int, default=4,
                   help="Skip clips with T_grid > max_T (vision tower OOM risk).")
    p.add_argument("--min_p_prop", type=int, default=6,
                   help="Minimum P_prop count for clip to qualify.")
    args = p.parse_args()

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B", device_map="balanced_low_0")
    layers = thinker_layers(model)
    visual_enc = _resolve_visual(model)
    print(f"  visual encoder type: {type(visual_enc).__name__}")
    # Vision tower config
    vt_cfg = visual_enc.config if hasattr(visual_enc, "config") else None
    if vt_cfg is not None:
        for k in ("patch_size", "temporal_patch_size", "spatial_merge_size",
                  "in_channels", "depth", "num_heads", "hidden_size",
                  "fullatt_block_indexes", "window_size"):
            if hasattr(vt_cfg, k):
                print(f"    cfg.{k} = {getattr(vt_cfg, k)}")

    data = json.load(open(DEFAULT_QA))
    rng = np.random.default_rng(0)
    # iterate in dataset order; pick the first clip satisfying the gate
    # (also returns counts for the FIRST processable clip as a sanity print)
    qualifying = []
    all_scored = []
    for i, d in enumerate(data):
        if i >= args.n_search:
            break
        info, err = probe_clip(d, model, processor, visual_enc, max_T=args.max_T)
        if err:
            print(f"  [skip {i}] {d['video']}: {err}")
            continue
        if info.get("skip"):
            print(f"  [skip {i}] {d['video']}: T={info['T']} > max_T (would OOM)")
            continue
        n_llm = info["n_video_tokens_llm"]
        n_enc = info["n_encoder_tokens"]
        norms_enc = info["encoder_norms"]
        # P_prop on encoder
        if norms_enc is None:
            print(f"  [skip {i}] {d['video']}: no encoder norms")
            continue
        p_prop_enc = int((norms_enc > TAU_PROP).sum())
        # Try to align to LLM video token count
        aligned = _align_norms(norms_enc, n_llm)
        p_prop_llm = int((aligned > TAU_PROP).sum()) if aligned is not None else None
        # Grid info
        grid = info["video_grid_thw"]
        n_frames = int(grid[0, 0]) if grid is not None and grid.ndim >= 2 else None
        T_grid = grid[0, 0] if grid is not None and grid.ndim >= 2 else None
        H_grid = grid[0, 1] if grid is not None and grid.ndim >= 2 else None
        W_grid = grid[0, 2] if grid is not None and grid.ndim >= 2 else None
        print(f"\n  clip [{i}]: {d['video']}")
        print(f"    grid_thw (raw patches): T={T_grid}, H={H_grid}, W={W_grid}")
        print(f"    n_encoder_tokens={n_enc}, n_video_tokens_llm={n_llm}, ratio={n_enc/max(n_llm,1):.2f}")
        print(f"    P_prop (encoder space): {p_prop_enc} of {n_enc} ({100*p_prop_enc/max(n_enc,1):.1f}%)")
        print(f"    P_prop (LLM space, aligned): {p_prop_llm} of {n_llm}")
        # Quick spatial-merge inference
        if T_grid is not None and grid is not None:
            patches_total = int(T_grid * H_grid * W_grid)
            merge_ratio = patches_total / max(n_llm, 1)
            print(f"    raw patches T*H*W = {patches_total}; "
                  f"LLM tokens = {n_llm}; merge ratio = {merge_ratio:.2f} "
                  f"(2D merge=4, 2D×temporal=8, etc.)")
        # Round-trip a known token: midpoint of LLM video span.
        # Empirical: merge ratio is exactly 4 → spatial 2×2 merge only, no
        # temporal merge. Token layout is (T_post=T_raw, H_post=H/2, W_post=W/2).
        if grid is not None and aligned is not None and p_prop_llm is not None:
            mid_llm = int(n_llm // 2)
            sm = getattr(vt_cfg, "spatial_merge_size", 2) if vt_cfg else 2
            T_post = int(T_grid)                       # no temporal merge
            H_post = int(H_grid) // sm
            W_post = int(W_grid) // sm
            if T_post * H_post * W_post == n_llm:
                t_idx = mid_llm // (H_post * W_post)
                r_idx = (mid_llm % (H_post * W_post)) // W_post
                c_idx = mid_llm % W_post
                back = t_idx * H_post * W_post + r_idx * W_post + c_idx
                print(f"    grid layout: T_post={T_post}, H_post={H_post}, W_post={W_post} "
                      f"(spatial merge {sm}×{sm} only, no temporal merge)")
                print(f"    round-trip token {mid_llm} → (t={t_idx}, r={r_idx}, c={c_idx}) → back={back}  "
                      f"{'OK' if back == mid_llm else 'MISMATCH'}")
                print(f"    each token covers 1 frame × {sm}×{sm} patches; no scatter needed")
            else:
                print(f"    grid layout candidate: T_post*H_post*W_post = "
                      f"{T_post*H_post*W_post} ≠ n_llm={n_llm}; layout mismatch")
        # Gate check
        n_frames_eff = int(T_grid) if T_grid is not None else 0
        # Skip oversize-T clips to avoid vision tower OOM
        if n_frames_eff > args.max_T:
            print(f"    [skip T={n_frames_eff} > max_T={args.max_T}]")
            continue
        gate_pass = (p_prop_llm is not None and p_prop_llm >= args.min_p_prop
                     and n_frames_eff >= 4)
        all_scored.append(dict(idx=i, video=d["video"], p_prop=p_prop_llm,
                                 n_llm=n_llm, T=int(T_grid),
                                 H=int(H_grid), W=int(W_grid),
                                 question=d.get("question", "")))
        if gate_pass:
            qualifying.append(all_scored[-1])
            print(f"    GATE PASS: P_prop_llm={p_prop_llm} ≥ {args.min_p_prop}, T={n_frames_eff} ≥ 4")
            if len(qualifying) >= 2:
                break
        else:
            print(f"    gate: P_prop_llm={p_prop_llm}, T={n_frames_eff}")

    print("\n=== QUALIFYING CLIPS ===")
    if not qualifying:
        print(f"  none with P_prop ≥ {args.min_p_prop} found in first {args.n_search} scanned.")
        if all_scored:
            best = sorted(all_scored, key=lambda x: -x["p_prop"])[:3]
            print(f"  top-3 by P_prop in scanned set:")
            for q in best:
                print(f"    [{q['idx']}] {q['video']} — P_prop={q['p_prop']}, T×H×W = {q['T']}×{q['H']}×{q['W']}")
    for q in qualifying[:2]:
        print(f"  [{q['idx']}] {q['video']} — P_prop={q['p_prop']}, T×H×W = {q['T']}×{q['H']}×{q['W']}")


if __name__ == "__main__":
    main()
