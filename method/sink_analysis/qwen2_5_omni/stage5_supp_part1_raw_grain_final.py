"""
stage5_supp_part1_raw_grain_final.py — PART 1 redone at strict raw-patch
grain, NO averaging of multiple tokens.

Methodology (per user spec):
  1. Pick the column corresponding to ONE raw key in the ViT self-attention
     at block 31: col = A[head, :, raw_k]  (length n_raw, one attention
     value per query patch).
  2. Reshape that column to (T_raw, H_raw, W_raw) — each frame is a 2D
     map of "fraction of attention from each query patch toward raw_k".
  3. Bilinear-resize each frame map to the native frame resolution and
     overlay on the actual frame.
  4. Normalize per figure (one shared scale across the figure's panels)
     so values read as "fraction of attention looking at this raw key";
     the title annotates the global max as a percentage and the per-raw
     col_mean for context.
  5. Do this for ALL raw cells of:
       - top-3 P_prop post-merger sinks (norm > 100), by encoder norm
       - 3 random post-merger non-sinks (norm < p25 of all-token norm)
     One composite figure per post-merger token × head: rows = 4 raw
     cells (no mean), cols = T_raw frames. Each row is ONE single raw
     key column.

Heads inspected: h9 (fg-like) and h7 (bg-like) at ViT block 31.

Outputs:
  stage5_supp/clip_<id>/per_token/raw_final_<rank>_<kind>_k<post>_h<h>_<tag>.png
  stage5_supp/clip_<id>/per_token/raw_final_summary.csv
"""
import json, math, sys, subprocess, tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # noqa: E402
    Qwen2_5OmniVisionAttention, apply_rotary_pos_emb_vision,
)

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
OUT_ROOT = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp"
VIDEO_TOKEN_ID = 151656
TAU_PROP = 100.0
PICK_BLOCK = 31
HEADS = {9: "fg", 7: "bg"}
CLIPS = ["021bd34fe8ff.mp4", "00e77d8995bd.mp4"]

_attn_callback = None


def patched_vision_attn_forward(self, hidden_states, cu_seqlens,
                                 rotary_pos_emb=None):
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


Qwen2_5OmniVisionAttention.forward = patched_vision_attn_forward


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def post_to_raw_indices(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    t = k_post // (H_post * W_post)
    R = (k_post % (H_post * W_post)) // W_post
    C = k_post % W_post
    out = []
    for dR in range(sm):
        for dC in range(sm):
            raw_idx = t * H_raw * W_raw + (sm*R + dR) * W_raw + (sm*C + dC)
            out.append(dict(idx=raw_idx, t=t, r=sm*R+dR, c=sm*C+dC,
                             dR=dR, dC=dC))
    return out


def post_to_frc(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    return (k_post // (H_post * W_post),
            (k_post % (H_post * W_post)) // W_post,
            k_post % W_post)


def select_tokens(enc_norms, seed=0):
    """top-3 P_prop sinks (by encoder norm) + 3 random non-sinks (<p25)."""
    rng = np.random.default_rng(seed)
    sinks = np.where(enc_norms > TAU_PROP)[0]
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    top3 = sinks[np.argsort(-enc_norms[sinks])[:3]].tolist()
    rand_non = rng.choice(nonsinks, size=min(3, len(nonsinks)),
                            replace=False).tolist()
    selected = [("top_norm_sink", int(k)) for k in top3] \
               + [("random_nonsink", int(k)) for k in rand_non]
    return selected


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    n_total = len(vr); native_fps = float(vr.get_avg_fps())
    step = native_fps / max(qwen_fps, 1e-6)
    idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
    return [vr[i].asnumpy() for i in idxs]


def render_token_at_head(raw_cols, frames, raw_meta, T_raw, H_raw, W_raw,
                          k_post, kind, norm_val, head_idx, tag, png_path):
    """raw_cols shape (4, n_raw). Reshape each to (T_raw, H_raw, W_raw),
    upsample, overlay. Rows=4 raw cells, cols=T_raw frames. Global
    normalization across the 4*T panels of this single figure."""
    cube = raw_cols.reshape(4, T_raw, H_raw, W_raw)
    per_raw_mean = raw_cols.mean(axis=1)
    vmin = 0.0
    vmax = float(cube.max())
    eps = max(vmax - vmin, 1e-12)
    norm = (cube - vmin) / eps
    cmap = plt.get_cmap("turbo")
    fig, axes = plt.subplots(4, T_raw,
                              figsize=(3 * T_raw, 12),
                              constrained_layout=True)
    if T_raw == 1: axes = axes.reshape(4, 1)
    for ri in range(4):
        rm = raw_meta[ri]
        for ti in range(T_raw):
            ax = axes[ri, ti]
            h_map = norm[ri, ti]
            base = frames[ti]
            from PIL import Image as _Im
            fh, fw = base.shape[:2]
            map_pil = _Im.fromarray((h_map * 255).astype(np.uint8)).resize(
                (fw, fh), _Im.BILINEAR)
            h_up = np.asarray(map_pil).astype(np.float32) / 255.0
            heat = (cmap(h_up)[..., :3] * 255).astype(np.uint8)
            overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
            ax.imshow(overlay)
            ax.set_xticks([]); ax.set_yticks([])
            if ti == 0:
                ax.set_ylabel(
                    f"raw {ri}: idx={rm['idx']}\n"
                    f"(r={rm['r']},c={rm['c']})\n"
                    f"col_mean={per_raw_mean[ri]:.3e}",
                    fontsize=8)
            if ri == 0:
                ax.set_title(f"t={ti}s", fontsize=10)
    pct_max = vmax * 100.0
    fig.suptitle(
        f"k_post={k_post} ({kind}, norm={norm_val:.1f})  |  blk{PICK_BLOCK}  "
        f"head h{head_idx} ({tag})\n"
        f"rows = 4 raw key columns (no averaging); cols = T_raw frames.  "
        f"Global vmax = {vmax:.3e} (={pct_max:.3f}% of any one query's attention).",
        fontsize=10)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return per_raw_mean, vmax


def process_clip(clip_name, data, model, processor, visual_enc):
    d = next(x for x in data if x["video"] == clip_name)
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists():
        print(f"  [skip] missing {video_path}"); return

    conv = build_conversation(str(video_path), d["question"], "v")
    inputs, use_aiv = prepare_inputs(processor, conv, "v", model.device,
                                       model.dtype)
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0]
    n_llm = int(video_pos.size)
    grid = inputs["video_grid_thw"].cpu().numpy()
    T_raw = int(grid[0, 0]); H_raw = int(grid[0, 1]); W_raw = int(grid[0, 2])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm
    n_raw = T_raw * H_raw * W_raw
    assert T_raw * H_post * W_post == n_llm
    print(f"\n  clip {clip_name}: T_raw={T_raw} H_raw={H_raw} W_raw={W_raw} "
          f"n_raw={n_raw} n_llm={n_llm}")

    # Pass 1: encoder (post-merger) norms — sink classification
    enc_buf = []
    def enc_hook(_m, _i, out):
        tok = _extract_tokens(out)
        enc_buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
    h = visual_enc.register_forward_hook(enc_hook)
    global _attn_callback
    _attn_callback = None
    with torch.inference_mode():
        model.thinker(**inputs, use_audio_in_video=use_aiv,
                      output_attentions=False, return_dict=True,
                      use_cache=False)
    h.remove()
    enc_norms = np.concatenate(enc_buf)
    if enc_norms.size != n_llm:
        print(f"    enc_norms size mismatch: {enc_norms.size} vs n_llm={n_llm}")
    sinks = np.where(enc_norms > TAU_PROP)[0]
    print(f"    P_prop sinks (norm>{TAU_PROP}): {sinks.size} / {n_llm}")

    selected = select_tokens(enc_norms, seed=0)
    print("    selected tokens:")
    for kind, k in selected:
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        print(f"      {kind:<16s}  k={k:4d}  norm={enc_norms[k]:6.2f}  "
              f"(t={t},r={r},c={c})")

    # Pass 2: ViT attention at block 31 — capture raw key columns separately
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i
    # raw_cols_per_token[k_post][head] = (4, n_raw) array (one row per raw cell)
    raw_cols_per_token = {k: {h: np.zeros((4, n_raw), dtype=np.float32)
                                for h in HEADS}
                            for _, k in selected}

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi != PICK_BLOCK: return
        if attn_weights.shape[-1] != n_raw: return
        for kind, k in selected:
            raws = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
            for ri, rm in enumerate(raws):
                for h in HEADS:
                    raw_cols_per_token[k][h][ri] = (
                        attn_weights[h, :, rm["idx"]].float().cpu().numpy())

    _attn_callback = cb
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        _attn_callback = None
    torch.cuda.empty_cache()

    frames = load_raw_frames(video_path, T_raw, qwen_fps=1.0)
    out_dir = OUT_ROOT / f"clip_{video_path.stem}" / "per_token"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = ["rank,kind,k_post,t,r,c,norm,head,tag,"
                 "col_mean_raw0,col_mean_raw1,col_mean_raw2,col_mean_raw3,"
                 "max_raw,vmax,vmax_pct"]
    print("    rendering...")
    for rank, (kind, k) in enumerate(selected):
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        raw_meta = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
        for h, tag in HEADS.items():
            raw_cols = raw_cols_per_token[k][h]
            fname = (f"raw_final_{rank:02d}_{kind}_k{k}_"
                     f"t{t}r{r}c{c}_h{h}_{tag}.png")
            png_path = out_dir / fname
            per_raw_mean, vmax = render_token_at_head(
                raw_cols, frames, raw_meta, T_raw, H_raw, W_raw,
                k, kind, float(enc_norms[k]), h, tag, png_path)
            max_raw = int(np.argmax(per_raw_mean))
            csv_rows.append(
                f"{rank},{kind},{k},{t},{r},{c},{enc_norms[k]:.3f},"
                f"{h},{tag},"
                f"{per_raw_mean[0]:.6e},{per_raw_mean[1]:.6e},"
                f"{per_raw_mean[2]:.6e},{per_raw_mean[3]:.6e},"
                f"{max_raw},{vmax:.6e},{vmax*100:.4f}")
            print(f"      h{h} {tag}: k={k} {kind}  "
                  f"col_means={[f'{m:.2e}' for m in per_raw_mean]}  "
                  f"vmax={vmax*100:.3f}%  strongest=raw {max_raw}")
    (out_dir / "raw_final_summary.csv").write_text("\n".join(csv_rows))
    print(f"    -> {out_dir.relative_to(_REPO)}")


def main():
    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                                   device_map="balanced_low_0")
    visual_enc = None
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model.thinker, attr):
            visual_enc = getattr(model.thinker, attr); break

    data = json.load(open(DEFAULT_QA))
    for clip in CLIPS:
        process_clip(clip, data, model, processor, visual_enc)


if __name__ == "__main__":
    main()
