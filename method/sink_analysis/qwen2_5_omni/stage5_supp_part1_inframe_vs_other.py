"""
stage5_supp_part1_inframe_vs_other.py — same-frame vs other-frame
diagnostic for ViT sink attention.

The strict raw-grain figures showed Qwen video sinks are within-frame
attention attractors (e.g. k=826 raw 1's bright spots are concentrated
in its own frame t=3, dark elsewhere). This script makes that
quantitative: for each token of interest, slice its column A[h, :, raw_k]
into a same-frame block and an other-frame block, render the 2 frames
side by side per raw cell, and report:

  in_mean  = mean over queries IN the sink's own frame
  out_mean = mean over queries in the chosen OTHER frame
  ratio    = in_mean / out_mean    (high if within-frame attractor)

NO averaging across tokens. Each row of each figure is ONE raw key
column; rendering for that column is split between 2 frames worth of
queries. Each panel normalized to its own [0, max] for display so the
spatial structure within each frame is visible; numerical comparison
stays in the title.

Tokens (per clip): top-3 P_prop sinks (by enc norm) + 3 random non-sinks
(< p25). Heads: h9 (fg), h7 (bg) at ViT block 31.

Output: stage5_supp/clip_<id>/per_token/inframe_<rank>_<kind>_k<post>_h<h>_<tag>.png
        stage5_supp/clip_<id>/per_token/inframe_summary.csv
"""
import json, math, sys
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
    rng = np.random.default_rng(seed)
    sinks = np.where(enc_norms > TAU_PROP)[0]
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    top3 = sinks[np.argsort(-enc_norms[sinks])[:3]].tolist()
    rand_non = rng.choice(nonsinks, size=min(3, len(nonsinks)),
                            replace=False).tolist()
    return [("top_norm_sink", int(k)) for k in top3] \
            + [("random_nonsink", int(k)) for k in rand_non]


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    n_total = len(vr); native_fps = float(vr.get_avg_fps())
    step = native_fps / max(qwen_fps, 1e-6)
    idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
    return [vr[i].asnumpy() for i in idxs]


def render_inframe_vs_other(raw_cols, frames, raw_meta, T_raw, H_raw,
                              W_raw, k_post, kind, norm_val, head_idx,
                              tag, own_t, other_t, png_path):
    """raw_cols shape (4, n_raw). For each of the 4 raw cells, render
    2 frame panels (own frame, other frame). Per-panel normalization so
    the spatial structure within each frame is visible. Numbers in title.
    Layout: rows = 4 raw cells; cols = [own_t panel, other_t panel].
    """
    cube = raw_cols.reshape(4, T_raw, H_raw, W_raw)

    # per-raw: mean over queries in own-frame block vs other-frame block
    in_means  = cube[:, own_t].reshape(4, -1).mean(axis=1)
    out_means = cube[:, other_t].reshape(4, -1).mean(axis=1)
    ratios = in_means / np.maximum(out_means, 1e-12)

    cmap = plt.get_cmap("turbo")
    fig, axes = plt.subplots(4, 2, figsize=(8, 12), constrained_layout=True)

    for ri in range(4):
        rm = raw_meta[ri]
        for col_i, (frame_t, label) in enumerate(
            [(own_t, f"OWN frame t={own_t}s"),
             (other_t, f"OTHER frame t={other_t}s")]):
            ax = axes[ri, col_i]
            h_map = cube[ri, frame_t]
            # per-panel normalization
            pmin, pmax = float(h_map.min()), float(h_map.max())
            eps = max(pmax - pmin, 1e-12)
            norm = (h_map - pmin) / eps
            base = frames[frame_t]
            from PIL import Image as _Im
            fh, fw = base.shape[:2]
            map_pil = _Im.fromarray((norm * 255).astype(np.uint8)).resize(
                (fw, fh), _Im.BILINEAR)
            h_up = np.asarray(map_pil).astype(np.float32) / 255.0
            heat = (cmap(h_up)[..., :3] * 255).astype(np.uint8)
            overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
            ax.imshow(overlay)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(label, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(
                    f"raw {ri}: idx={rm['idx']} (r={rm['r']},c={rm['c']})\n"
                    f"in_mean={in_means[ri]:.3e}\n"
                    f"out_mean={out_means[ri]:.3e}\n"
                    f"ratio={ratios[ri]:.2f}×",
                    fontsize=8)
            # annotate per-panel max as %
            ax.text(0.02, 0.98, f"panel max = {pmax*100:.2f}%",
                     transform=ax.transAxes, fontsize=8, color="white",
                     verticalalignment="top",
                     bbox=dict(facecolor="black", alpha=0.5, pad=2))

    # Aggregate ratio using the strongest raw's in_mean/out_mean
    strongest_raw = int(np.argmax(in_means))
    fig.suptitle(
        f"k_post={k_post} ({kind}, norm={norm_val:.1f})  |  blk{PICK_BLOCK}  "
        f"head h{head_idx} ({tag})\n"
        f"own frame = t={own_t}s | other frame = t={other_t}s. "
        f"Strongest raw = raw {strongest_raw} (in/out ratio "
        f"{ratios[strongest_raw]:.2f}×).  Per-panel normalization.",
        fontsize=10)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return in_means, out_means, ratios


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

    # Pass 1: encoder norms
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

    selected = select_tokens(enc_norms, seed=0)
    print("    selected tokens:")
    for kind, k in selected:
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        print(f"      {kind:<16s}  k={k:4d}  norm={enc_norms[k]:6.2f}  "
              f"(own_frame=t={t})")

    # Pass 2: capture raw key columns at block 31
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i
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

    csv_rows = ["rank,kind,k_post,own_t,other_t,head,tag,"
                 "in_mean_raw0,in_mean_raw1,in_mean_raw2,in_mean_raw3,"
                 "out_mean_raw0,out_mean_raw1,out_mean_raw2,out_mean_raw3,"
                 "ratio_raw0,ratio_raw1,ratio_raw2,ratio_raw3,"
                 "strongest_raw,strongest_ratio"]
    print("    rendering...")
    for rank, (kind, k) in enumerate(selected):
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        raw_meta = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
        own_t = t
        # Pick the frame farthest from own_t (ties broken toward t=0)
        other_t = max(range(T_raw), key=lambda x: (abs(x - own_t), -x))
        for h, tag in HEADS.items():
            raw_cols = raw_cols_per_token[k][h]
            fname = (f"inframe_{rank:02d}_{kind}_k{k}_"
                     f"t{t}r{r}c{c}_h{h}_{tag}.png")
            png_path = out_dir / fname
            in_means, out_means, ratios = render_inframe_vs_other(
                raw_cols, frames, raw_meta, T_raw, H_raw, W_raw,
                k, kind, float(enc_norms[k]), h, tag,
                own_t, other_t, png_path)
            strongest = int(np.argmax(in_means))
            csv_rows.append(
                f"{rank},{kind},{k},{own_t},{other_t},{h},{tag},"
                + ",".join(f"{x:.6e}" for x in in_means) + ","
                + ",".join(f"{x:.6e}" for x in out_means) + ","
                + ",".join(f"{x:.4f}" for x in ratios) + ","
                + f"{strongest},{ratios[strongest]:.4f}")
            print(f"      h{h} {tag}: k={k} {kind}  "
                  f"in/out ratio per raw: "
                  f"{[f'{r:.2f}x' for r in ratios]}  "
                  f"(strongest raw {strongest}: {ratios[strongest]:.2f}x)")
    (out_dir / "inframe_summary.csv").write_text("\n".join(csv_rows))
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
