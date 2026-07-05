"""
stage5_supp_part1_relevance.py — PART 1: relevance maps as per-frame
videos (set-level), sink_or_not Fig-4 replication on Qwen2.5-Omni.

For each clip (primary 85: 021bd34fe8ff, replicate 83: 00e77d8995bd):
  1. Monkey-patch Qwen2_5OmniVisionAttention.forward to expose
     post-softmax attention weights per block. (No batch dim — vision
     attention is shape (n_heads, seq_len, seq_len) at raw-patch
     granularity.)
  2. Forward thinker; capture per-block attention + final encoder norms.
  3. Identify P_prop-video sinks: encoder L2 norm > 100 (post-merger,
     LLM-aligned). Map each post-merger sink → its 4 raw patches.
  4. Define non-sink-video: norm < 25th percentile (post-merger), mapped
     to 4 raw patches each (matched-pool baseline).
  5. Per full-attention block × head: aggregate the attention column
     (mean over sink raw indices) → rel_sink (raw-patch shape), same for
     non-sink → rel_nonsink. Reshape to (T_raw, H_raw, W_raw).
  6. Head identification: rank heads at each full-attn block by spatial
     SPARSITY of sink-set relevance (top-10% mass share, frame-mean).
     "Foreground-like" = sparse / concentrated. "Background-like" =
     diffuse / high-entropy. Report top fg head and top bg head, the
     dispersion across heads, and whether the two distinct types exist.
  7. Render per (clip, block, head, set): one heatmap per frame, GLOBAL
     color scale (one cbar over the whole clip), saved as a single
     stacked-frames PNG and an animated GIF.

Outputs (`stage5_supp/clip_<clip>/...`):
  per_block_head_stats.csv     — entropy / sparsity per (block, head, set)
  selected_heads.csv           — fg & bg head per clip + criterion
  fig_<clip>_blk<L>_h<H>_<set>.png   stacked-frames heatmap
  fig_<clip>_blk<L>_h<H>_<set>.gif   animated
  README.md                    — clip metadata + verdict
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None        # GIF rendering will be skipped; PNG still produced
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs, thinker_layers  # noqa: E402
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # noqa: E402
    Qwen2_5OmniVisionAttention, apply_rotary_pos_emb_vision,
)

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp"
TAU_PROP = 100.0
VIDEO_TOKEN_ID = 151656
PRIMARY_CLIP = "021bd34fe8ff.mp4"
REPLICATE_CLIP = "00e77d8995bd.mp4"

# Per-clip in-hook callback and stash
_attn_callback = None


def patched_vision_attn_forward(self, hidden_states, cu_seqlens,
                                 rotary_pos_emb=None):
    """Drop-in replacement for Qwen2_5OmniVisionAttention.forward that
    additionally invokes _attn_callback(self, attn_weights, cu_seqlens)
    on the post-softmax attention tensor. Same dtype/device/values as
    the stock forward — verified by reading source."""
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    attention_mask = torch.full(
        [1, seq_length, seq_length], torch.finfo(q.dtype).min,
        device=q.device, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i],
                       cu_seqlens[i-1]:cu_seqlens[i]] = 0
    q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                           dtype=torch.float32).to(q.dtype)
    if _attn_callback is not None:
        _attn_callback(self, attn_weights, cu_seqlens)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


def install_patch():
    Qwen2_5OmniVisionAttention.forward = patched_vision_attn_forward


# Apply the patch immediately at import time
install_patch()


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def post_to_raw_indices(k_post: int, T_raw: int, H_raw: int, W_raw: int, sm: int = 2):
    """Map a post-merger token index k in flattened (T_raw, H_post=H/sm, W_post=W/sm)
    layout back to the sm*sm raw-patch indices in flattened (T_raw, H_raw, W_raw)."""
    H_post = H_raw // sm
    W_post = W_raw // sm
    t = k_post // (H_post * W_post)
    R = (k_post % (H_post * W_post)) // W_post
    C = k_post % W_post
    raws = []
    for dR in range(sm):
        for dC in range(sm):
            r = sm * R + dR
            c = sm * C + dC
            raws.append(t * H_raw * W_raw + r * W_raw + c)
    return raws


def render_overlay(rel_3d: np.ndarray, frames: list, title: str,
                    png_path: Path, mp4_path: Path, playback_fps: float = 1.0):
    """rel_3d shape (T_raw, H_raw, W_raw). frames: list of raw frames
    (numpy arrays HxWx3, RGB) sampled at the SAME 1 fps the vision tower
    saw. Global normalization across all frames. Writes a stacked-frames
    PNG and a running MP4 (playback_fps frames/sec = 1 fps default to
    match Qwen sampling)."""
    T_raw = rel_3d.shape[0]
    # Global normalization (across all T frames)
    vmin = float(rel_3d.min())
    vmax = float(rel_3d.max())
    eps = max(vmax - vmin, 1e-12)
    norm = (rel_3d - vmin) / eps                 # → [0, 1]
    cmap = plt.get_cmap("turbo")
    composited = []                              # list of HxWx3 uint8
    for t in range(T_raw):
        h_map = norm[t]
        if frames is not None and t < len(frames):
            base = frames[t]
            from PIL import Image as _Im
            fh, fw = base.shape[:2]
            map_pil = _Im.fromarray((h_map * 255).astype(np.uint8)).resize(
                (fw, fh), _Im.BILINEAR)
            h_up = np.asarray(map_pil).astype(np.float32) / 255.0
            heat = (cmap(h_up)[..., :3] * 255).astype(np.uint8)
            overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
        else:
            heat = (cmap(h_map)[..., :3] * 255).astype(np.uint8)
            overlay = heat
        composited.append(overlay)
    # PNG: stack horizontally
    full = np.concatenate(composited, axis=1)
    fig_w = max(8, T_raw * 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3), constrained_layout=True)
    ax.imshow(full)
    ax.set_xticks([(i + 0.5) * full.shape[1] / T_raw for i in range(T_raw)])
    ax.set_xticklabels([f"t={t}s" for t in range(T_raw)], fontsize=9)
    ax.set_yticks([])
    ax.set_title(title + f"  | rel global range [{vmin:.4f}, {vmax:.4f}]  "
                 f"| 1 fps Qwen-aligned",
                 fontsize=9)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # MP4: write frames to a temp dir then ffmpeg-encode at 1 fps playback
    # (each frame held for 1 second, matching Qwen's 1-fps sampling rate)
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Ensure even dimensions (libx264 requires even H, W)
        for t, im in enumerate(composited):
            h, w = im.shape[:2]
            if h % 2 == 1 or w % 2 == 1:
                im = im[: (h // 2) * 2, : (w // 2) * 2]
                composited[t] = im
            plt.imsave(td_p / f"f_{t:04d}.png", im)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(playback_fps),
            "-i", str(td_p / "f_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"fps={playback_fps}",
            str(mp4_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"  [mp4 warn] {mp4_path.name}: {e}")


def load_raw_frames(video_path: Path, T_raw: int, qwen_fps: float = 1.0):
    """Extract T_raw frames at the SAME 1-fps spacing Qwen's vision tower
    used. Frame t (in {0,1,...,T_raw-1}) is at video timestamp t/qwen_fps
    seconds = t*native_fps/qwen_fps frames into the source. Indices are
    clamped to [0, n_total-1] (short clips reuse the last frame). Returns
    (frames, native_fps)."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path), num_threads=1)
        n_total = len(vr)
        native_fps = float(vr.get_avg_fps())
        # 1 frame per video-second at qwen_fps=1.0 → step = native_fps frames
        step = native_fps / max(qwen_fps, 1e-6)
        idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
        frames = [vr[i].asnumpy() for i in idxs]
        print(f"  decord: native_fps={native_fps:.2f}, n_total={n_total}; "
              f"extracted frame idxs = {idxs} for T_raw={T_raw} at qwen_fps={qwen_fps}")
        return frames, native_fps
    except Exception as e:
        print(f"  [decord warn] {video_path}: {e}")
        return None, None


def process_clip(d, model, processor, visual_enc, out_root: Path,
                  full_attn_block_indexes=None):
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists(): return f"missing_video"

    conv = build_conversation(str(video_path), d["question"], "v")
    inputs, use_aiv = prepare_inputs(processor, conv, "v",
                                        model.device, model.dtype)
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0].astype(np.int64)
    n_llm = int(video_pos.size)

    grid = inputs["video_grid_thw"].cpu().numpy()
    T_raw = int(grid[0, 0]); H_raw = int(grid[0, 1]); W_raw = int(grid[0, 2])
    sm = 2
    T_post, H_post, W_post = T_raw, H_raw // sm, W_raw // sm
    assert T_post * H_post * W_post == n_llm, (T_post, H_post, W_post, n_llm)
    n_raw = T_raw * H_raw * W_raw
    print(f"  clip {d['video']}: T_raw={T_raw}, H_raw={H_raw}, W_raw={W_raw}, "
          f"n_raw={n_raw}, n_llm={n_llm}")

    # Identify vision blocks and tag each block.attn with its index
    visual = visual_enc
    blocks = visual.blocks
    n_blocks = len(blocks)
    print(f"  vision blocks: {n_blocks}; full-attn idxs (config): {full_attn_block_indexes}")
    for i, blk in enumerate(blocks):
        blk.attn._blk_idx = i

    # We'll collect per-block per-head (rel_sink, rel_nonsink) raw-vector
    # immediately in the callback to avoid storing the full attention.
    # Need sink/non-sink raw indices BEFORE forward → first compute
    # encoder norms via a one-shot forward... but we'll do it in this
    # same forward via the encoder hook and a TWO-PASS strategy:
    # Pass 1: forward to get encoder norms (capture only the final
    #         post-merger output via enc_hook on visual_enc). Skip attn.
    # Pass 2: with sink masks now known, forward AGAIN with the
    #         callback active.
    # Cost = 2 vision forwards per clip (~2-5s each); acceptable.

    # ---- PASS 1: get encoder norms (final post-merger output) ----
    enc_buf = []
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    h = visual_enc.register_forward_hook(enc_hook)
    global _attn_callback
    _attn_callback = None
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        h.remove()
    enc_norms = np.concatenate(enc_buf)
    sinks_post = np.where(enc_norms > TAU_PROP)[0]
    p25 = float(np.percentile(enc_norms, 25))
    nonsinks_post = np.where(enc_norms < p25)[0]
    print(f"  enc_norms: n={enc_norms.size}; P_prop (>{TAU_PROP}) = {sinks_post.size}; "
          f"non-sink (< p25={p25:.2f}) = {nonsinks_post.size}")

    # Map post-merger sink positions → raw-patch indices
    sink_raw = []
    for k in sinks_post: sink_raw.extend(post_to_raw_indices(int(k), T_raw, H_raw, W_raw, sm))
    nonsink_raw = []
    for k in nonsinks_post: nonsink_raw.extend(post_to_raw_indices(int(k), T_raw, H_raw, W_raw, sm))
    sink_raw = np.array(sink_raw, dtype=np.int64)
    nonsink_raw = np.array(nonsink_raw, dtype=np.int64)
    print(f"  raw patch counts: sink = {sink_raw.size}; non-sink = {nonsink_raw.size}")

    # ---- PASS 2: capture per-block per-head aggregated relevance ----
    # rel_store[block_idx] = dict(sink=(H, n_raw), non=(H, n_raw))
    rel_store = {}

    def cb(self_mod, attn_weights, cu_seqlens):
        # attn_weights: (n_heads, seq, seq); seq should == n_raw at full-attn block
        bi = getattr(self_mod, "_blk_idx", None)
        if bi is None: return
        # Subset columns to sink_raw and nonsink_raw, mean over those columns
        # attn_weights[head, q, k_in_set].mean over k → (n_heads, seq)
        H = attn_weights.shape[0]
        if attn_weights.shape[-1] != n_raw:
            # window-attention block where seq != n_raw — capture nothing meaningful
            rel_store[bi] = None
            return
        sink_idx = torch.as_tensor(sink_raw, device=attn_weights.device, dtype=torch.long)
        non_idx  = torch.as_tensor(nonsink_raw, device=attn_weights.device, dtype=torch.long)
        rel_sink = attn_weights.index_select(2, sink_idx).mean(dim=2).float().cpu().numpy()
        rel_non  = attn_weights.index_select(2, non_idx).mean(dim=2).float().cpu().numpy()
        rel_store[bi] = dict(sink=rel_sink, non=rel_non)

    _attn_callback = cb
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        _attn_callback = None

    full_attn_blocks = [bi for bi, v in rel_store.items()
                          if v is not None and v["sink"].shape[1] == n_raw]
    print(f"  full-attention blocks captured: {full_attn_blocks}")
    if not full_attn_blocks:
        return "no_full_attn_blocks_captured"

    # ---- Head identification: per block × head compute sparsity of mean-frame sink map ----
    H_heads = rel_store[full_attn_blocks[0]]["sink"].shape[0]
    head_stats_rows = []
    for bi in full_attn_blocks:
        rel = rel_store[bi]
        for h_idx in range(H_heads):
            rel_sink_3d = rel["sink"][h_idx].reshape(T_raw, H_raw, W_raw)
            rel_non_3d  = rel["non"][h_idx].reshape(T_raw, H_raw, W_raw)
            # Spatial sparsity (per frame avg): top-10% mass share over total
            frame_mean = rel_sink_3d.mean(axis=0)             # (H_raw, W_raw)
            flat = frame_mean.flatten()
            k = max(int(0.10 * flat.size), 1)
            top_share = float(np.sort(flat)[-k:].sum() / max(flat.sum(), 1e-12))
            # Spatial entropy
            p = flat / max(flat.sum(), 1e-12)
            p = np.clip(p, 1e-12, 1.0)
            ent = float(-(p * np.log(p)).sum())
            # Temporal aggregation: variance across frames at peak patch (proxy
            # for "sink pulls from all frames" vs "only its own frame")
            frame_peaks = rel_sink_3d.reshape(T_raw, -1).max(axis=1)
            tvar = float(frame_peaks.std())
            # Same for non-sink for comparison
            frame_mean_non = rel_non_3d.mean(axis=0)
            flat_non = frame_mean_non.flatten()
            top_share_non = float(np.sort(flat_non)[-k:].sum() / max(flat_non.sum(), 1e-12))
            head_stats_rows.append(dict(
                clip=d["video"], block=bi, head=h_idx,
                sink_top10_share=top_share,
                sink_entropy=ent,
                sink_temp_peakstd=tvar,
                non_top10_share=top_share_non,
                sink_minus_non_top_share=top_share - top_share_non,
            ))

    import pandas as pd
    stats_df = pd.DataFrame(head_stats_rows)
    out_clip_dir = out_root / f"clip_{Path(d['video']).stem}"
    out_clip_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_clip_dir / "per_block_head_stats.csv", index=False)

    # Pick fg/bg heads from the LAST captured full-attn block
    pick_block = full_attn_blocks[-1]
    sub = stats_df[stats_df["block"] == pick_block].copy()
    # fg = highest sparsity (top-10% share); bg = lowest
    fg = sub.loc[sub["sink_top10_share"].idxmax()]
    bg = sub.loc[sub["sink_top10_share"].idxmin()]
    print(f"  block {pick_block}:  fg-like head = {int(fg['head'])} "
          f"(top10 share = {fg['sink_top10_share']:.3f})  ; "
          f"bg-like head = {int(bg['head'])} (top10 share = {bg['sink_top10_share']:.3f})")
    sub.sort_values("sink_top10_share", ascending=False).to_csv(
        out_clip_dir / "selected_heads.csv", index=False)

    # Discrimination: do the two head types actually differ?
    spread = sub["sink_top10_share"].max() - sub["sink_top10_share"].min()
    print(f"  sparsity spread across heads at block {pick_block} = {spread:.3f}")
    fg_bg_distinct = bool(spread > 0.10)             # arbitrary practical threshold
    print(f"  fg/bg distinction: {'YES' if fg_bg_distinct else 'NO (heads ~uniform)'}")

    # ---- Render 4 videos: (fg_head × sink/non) + (bg_head × sink/non) ----
    # Extract frames at Qwen's 1-fps spacing so the heatmap at "frame t"
    # aligns with the actual frame the ViT saw at t seconds into the clip.
    frames, native_fps = load_raw_frames(video_path, T_raw, qwen_fps=1.0)
    selected = [(int(fg["head"]), "fg"), (int(bg["head"]), "bg")]
    for h_idx, tag in selected:
        rel = rel_store[pick_block]
        for set_name in ("sink", "non"):
            rel_3d = rel[set_name][h_idx].reshape(T_raw, H_raw, W_raw)
            png = out_clip_dir / f"fig_blk{pick_block}_h{h_idx}_{tag}_{set_name}.png"
            mp4 = out_clip_dir / f"fig_blk{pick_block}_h{h_idx}_{tag}_{set_name}.mp4"
            title = (f"{d['video']}  blk={pick_block}  head={h_idx} ({tag})  "
                     f"set={set_name}  | T={T_raw} frames @ 1 fps")
            render_overlay(rel_3d, frames, title, png, mp4, playback_fps=1.0)
            print(f"  wrote {png.name} + {mp4.name}")

    # README
    md = [f"# Stage 5 supplementary PART 1 — clip {d['video']}\n",
          f"- T_raw × H_raw × W_raw = {T_raw} × {H_raw} × {W_raw}",
          f"- n_video_tokens (LLM-aligned) = {n_llm}; n_raw = {n_raw}",
          f"- P_prop sinks (>{TAU_PROP}) = {sinks_post.size}; non-sink pool (<p25) = {nonsinks_post.size}",
          f"- vision blocks total = {n_blocks}; full-attn blocks captured = {full_attn_blocks}",
          f"- fg-like head (block {pick_block}): h{int(fg['head'])} "
          f"(top-10% mass share = {fg['sink_top10_share']:.3f})",
          f"- bg-like head (block {pick_block}): h{int(bg['head'])} "
          f"(top-10% mass share = {bg['sink_top10_share']:.3f})",
          f"- across-head sparsity spread at block {pick_block}: {spread:.3f}",
          f"- fg/bg distinction holds: **{'yes' if fg_bg_distinct else 'no (heads ~uniform)'}**\n",
          ]
    (out_clip_dir / "README.md").write_text("\n".join(md) + "\n")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+",
                   default=[PRIMARY_CLIP, REPLICATE_CLIP])
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                                    device_map="balanced_low_0")
    visual_enc = model.thinker.visual
    vt_cfg = visual_enc.config if hasattr(visual_enc, "config") else None
    full_attn = getattr(vt_cfg, "fullatt_block_indexes", None) if vt_cfg else None
    n_blocks = len(visual_enc.blocks)
    n_heads = getattr(vt_cfg, "num_heads", None) if vt_cfg else None
    print(f"  vision tower: {n_blocks} blocks, {n_heads} heads, "
          f"full-attn block idxs = {full_attn}")

    data = json.load(open(DEFAULT_QA))
    # Find clip dicts matching the names
    by_name = {d["video"]: d for d in data}
    for cname in args.clips:
        if cname not in by_name:
            print(f"  [skip] {cname} not in sampled_entities")
            continue
        d = by_name[cname]
        print(f"\n=== {cname} ===")
        err = process_clip(d, model, processor, visual_enc, out_root,
                            full_attn_block_indexes=full_attn)
        if err: print(f"  err: {err}")


if __name__ == "__main__":
    main()
