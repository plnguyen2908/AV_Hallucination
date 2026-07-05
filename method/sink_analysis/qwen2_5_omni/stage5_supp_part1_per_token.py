"""
stage5_supp_part1_per_token.py — PART 1 REDO (per-single-token relevance
maps) + PART 2 (word distributions on the same 7 tokens).

Definition (sink_or_not Fig 4): the relevance map of ONE token is its
vertical key column in ViT self-attention: A[head, :, k] for queries q,
key k = the token of interest. ONE map per ONE token (no set averaging).

Per clip we render 7 tokens:
  - top-3 P_prop-video sinks BY ENCODER NORM
  - 3 random P_prop-video sinks (seed=0)
  - 1 random NON-sink video token (seed=0; norm < p25)

For each token × {fg head, bg head} from PART 1's head identification
(h9 fg / h7 bg at ViT block 31), we render a 1-fps Qwen-aligned MP4 and
a stacked-frames PNG; values are RAW attention numbers (no per-frame /
per-panel renorm; one global color scale per token-video).

ORIENTATION ASSERTION (printed): per head, mean(col) for sink tokens
should be >> mean(col) for the non-sink token. Reported, not enforced.

PART 2: for each of the 7 tokens, mask attention TO all video keys
EXCEPT this one token, forward through all layers, decode `final_norm +
lm_head` at the kept token's residual → top-15 vocab tokens. Per-token
word lists, sink vs non-sink, illustrative on 1 clip + 1 replicate.

Multi-frame tokens under temporal merging: N/A — Qwen2.5-Omni has NO
temporal merging (STEP 0). Each token covers 1 frame × 2×2 raw patches.

Outputs (`stage5_supp/clip_<clip>/per_token/`):
  fig_<rank>_t<post>_h<head>_<tag>.mp4  +  .png
  word_distributions.md
  per_token_summary.csv  (token_idx, frame, row, col, norm, kind, head,
                          col_mean_sink, col_mean_non, ...)
"""
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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

# Heads selected by PART 1 set-level head ID (h9 fg / h7 bg at block 31).
FG_HEAD = 9
BG_HEAD = 7
PICK_BLOCK = 31

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
    raws = [t * H_raw * W_raw + (sm*R + dR) * W_raw + (sm*C + dC)
             for dR in range(sm) for dC in range(sm)]
    return raws


def post_to_frc(k_post, T_raw, H_raw, W_raw, sm=2):
    H_post = H_raw // sm; W_post = W_raw // sm
    return (k_post // (H_post * W_post),
            (k_post % (H_post * W_post)) // W_post,
            k_post % W_post)


def select_seven_tokens(enc_norms, T_post, H_post, W_post, seed=0):
    """7 tokens: top-3 by norm (among P_prop sinks) + 3 random sinks
    + 1 random non-sink. Returns list of (kind, k_post)."""
    rng = np.random.default_rng(seed)
    sinks = np.where(enc_norms > TAU_PROP)[0]
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    if sinks.size < 6:
        # Fall back: still take what we have for sinks.
        print(f"  WARNING: only {sinks.size} sinks; top/random may overlap")
    sink_norms = enc_norms[sinks]
    top3_idx_local = np.argsort(-sink_norms)[:3]
    top3 = sinks[top3_idx_local].tolist()
    remaining_sinks = [int(s) for s in sinks if s not in top3]
    rand3 = (rng.choice(remaining_sinks, size=min(3, len(remaining_sinks)),
                         replace=False).tolist()
              if remaining_sinks else [])
    # If too few remaining for 3 random, draw with replacement from all sinks
    while len(rand3) < 3 and sinks.size:
        extra = int(rng.choice(sinks))
        if extra not in top3 and extra not in rand3:
            rand3.append(extra)
    rand_non = int(rng.choice(nonsinks)) if nonsinks.size else None
    selected = [("top_norm", k) for k in top3] \
                + [("random_sink", k) for k in rand3]
    if rand_non is not None:
        selected.append(("random_nonsink", rand_non))
    return selected


def render_overlay(rel_3d, frames, title, png_path, mp4_path,
                    playback_fps=1.0):
    """rel_3d shape (T_raw, H_raw, W_raw). Global normalization across
    all frames (single color scale per token-video). frames are
    1-fps-aligned via decord."""
    T_raw = rel_3d.shape[0]
    vmin = float(rel_3d.min())
    vmax = float(rel_3d.max())
    eps = max(vmax - vmin, 1e-12)
    norm = (rel_3d - vmin) / eps
    cmap = plt.get_cmap("turbo")
    composited = []
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
    full = np.concatenate(composited, axis=1)
    fig_w = max(8, T_raw * 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2), constrained_layout=True)
    ax.imshow(full)
    ax.set_xticks([(i + 0.5) * full.shape[1] / T_raw for i in range(T_raw)])
    ax.set_xticklabels([f"t={t}s" for t in range(T_raw)], fontsize=9)
    ax.set_yticks([])
    ax.set_title(title + f"  | raw attn range [{vmin:.4e}, {vmax:.4e}]",
                 fontsize=9)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    # MP4 (ffmpeg, 1 fps playback)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for t, im in enumerate(composited):
            h, w = im.shape[:2]
            if h % 2 == 1 or w % 2 == 1:
                im = im[: (h // 2) * 2, : (w // 2) * 2]
            plt.imsave(td_p / f"f_{t:04d}.png", im)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(playback_fps),
                "-i", str(td_p / "f_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", f"fps={playback_fps}",
                str(mp4_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"  [mp4 warn] {mp4_path.name}: {e}")


def load_raw_frames(video_path, T_raw, qwen_fps=1.0):
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path), num_threads=1)
        n_total = len(vr)
        native_fps = float(vr.get_avg_fps())
        step = native_fps / max(qwen_fps, 1e-6)
        idxs = [min(int(round(t * step)), n_total - 1) for t in range(T_raw)]
        frames = [vr[i].asnumpy() for i in idxs]
        return frames, native_fps, idxs
    except Exception as e:
        print(f"  [decord warn] {video_path}: {e}")
        return None, None, None


def isolate_and_decode(model, processor, inputs, use_aiv, layers,
                        prompt_S, video_pos, keep_k_post, T_raw, H_raw,
                        W_raw, sm, n_layers, final_norm, lm_head,
                        tokenizer, top_k=15):
    """sink_or_not isolation: mask attention TO ALL video keys except the
    4 raw-patch indices comprising token keep_k_post (i.e., the single
    post-merger token).

    For Qwen2.5-Omni the video tokens in the LLM sequence are post-merger
    (1 per video token); we keep just the LLM position corresponding to
    keep_k_post (= video_pos[keep_k_post]) and mask attention to all
    OTHER positions in `video_pos`.

    Apply the mask via a layer pre-hook (with_kwargs) on every decoder
    layer; capture h_output[27] via a post-hook on the last layer;
    decode through final_norm + lm_head at the kept position; return
    top-K (id, prob, decoded) list.
    """
    keep_llm_pos = int(video_pos[keep_k_post])     # position in LLM input
    masked_video_keys = np.array(
        [p for i, p in enumerate(video_pos) if i != keep_k_post],
        dtype=np.int64)

    S = int(inputs["input_ids"].shape[1])
    pos_t = torch.as_tensor(masked_video_keys, dtype=torch.long)

    def pre_hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if am is None or am.shape[-1] != S: return args, kwargs
        new_am = am.clone()
        idx = pos_t.to(new_am.device)
        new_am[..., :, idx] = torch.finfo(new_am.dtype).min
        kwargs["attention_mask"] = new_am
        return args, kwargs

    last_out = [None]
    def post_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == S:
            last_out[0] = hs[0].detach()
        return out

    handles = [layers[L].register_forward_pre_hook(pre_hook, with_kwargs=True)
                for L in range(n_layers)]
    handles.append(layers[n_layers - 1].register_forward_hook(post_hook))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        for h in handles: h.remove()

    if last_out[0] is None:
        return None

    h27 = last_out[0]
    dev_lm = lm_head.weight.device
    h_sel = h27[keep_llm_pos:keep_llm_pos+1].to(device=dev_lm,
                                                   dtype=lm_head.weight.dtype)
    with torch.no_grad():
        h_n = final_norm(h_sel)
        logits = lm_head(h_n).float()                            # (1, V)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(top_k, dim=-1)
    tv = top_vals[0].cpu().numpy(); ti = top_idx[0].cpu().numpy()
    decoded = [tokenizer.decode([int(t)], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
                for t in ti]
    return [(int(ti[k]), float(tv[k]), decoded[k]) for k in range(top_k)]


def process_clip(d, model, processor, visual_enc, layers, n_layers,
                  final_norm, lm_head, tokenizer, out_root):
    video_path = Path(DEFAULT_VID) / d["video"]
    if not video_path.exists(): return "missing_video"

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
    assert T_post * H_post * W_post == n_llm
    n_raw = T_raw * H_raw * W_raw
    print(f"\n  clip {d['video']}: T_raw={T_raw}, H_raw={H_raw}, W_raw={W_raw} "
          f"(post {T_post}×{H_post}×{W_post}={n_llm}); n_raw={n_raw}")

    # ---- Pass 1: encoder norms ----
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
    sinks = np.where(enc_norms > TAU_PROP)[0]
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    print(f"  enc_norms n={enc_norms.size}; sinks={sinks.size}, nonsinks(<p25)={nonsinks.size}")

    selected = select_seven_tokens(enc_norms, T_post, H_post, W_post, seed=0)
    print(f"  selected 7 tokens (kind, k_post, norm, (t,r,c)):")
    for kind, k in selected:
        t, r, c = post_to_frc(int(k), T_raw, H_raw, W_raw, sm)
        print(f"    {kind:<15s}  k={int(k):4d}  norm={enc_norms[int(k)]:.2f}  (t={t},r={r},c={c})")

    # ---- Pass 2: per-token relevance at block 31 for fg + bg heads ----
    # Tag attn modules with block idx
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i
    # rel_store[k_post] = dict(head_idx -> (n_raw,))
    rel_store = {k: {} for _, k in selected}

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi != PICK_BLOCK: return
        if attn_weights.shape[-1] != n_raw: return
        for kind, k in selected:
            raws = post_to_raw_indices(int(k), T_raw, H_raw, W_raw, sm)
            raw_t = torch.as_tensor(raws, device=attn_weights.device,
                                       dtype=torch.long)
            # rel[head, q] = mean over the 4 raw patches forming this 1 sink
            col = attn_weights.index_select(2, raw_t).mean(dim=2)  # (H, n_raw)
            rel_store[k][None] = col.float().cpu().numpy()         # all-heads
    _attn_callback = cb
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    finally:
        _attn_callback = None

    # ---- Orientation assertion: col_mean(sink) >> col_mean(nonsink) ----
    print(f"  ORIENTATION (per head, col mean = mean attention RECEIVED):")
    sink_keys = [k for kind, k in selected if kind != "random_nonsink"]
    non_keys  = [k for kind, k in selected if kind == "random_nonsink"]
    for head_label, hh in (("fg(h9)", FG_HEAD), ("bg(h7)", BG_HEAD)):
        if not sink_keys or not non_keys: continue
        sink_mean = float(np.mean([rel_store[k][None][hh].mean() for k in sink_keys]))
        non_mean  = float(np.mean([rel_store[k][None][hh].mean() for k in non_keys]))
        print(f"    block {PICK_BLOCK} {head_label}: mean col_mean over sinks = "
              f"{sink_mean:.4e}; non-sink = {non_mean:.4e}; ratio = "
              f"{sink_mean / max(non_mean, 1e-12):.2f}×")

    # ---- Render per-token MP4 + PNG for fg + bg heads ----
    out_clip = out_root / f"clip_{Path(d['video']).stem}" / "per_token"
    out_clip.mkdir(parents=True, exist_ok=True)
    frames, native_fps, frame_idxs = load_raw_frames(video_path, T_raw, qwen_fps=1.0)
    print(f"  decord: native_fps={native_fps:.2f}; frame idxs = {frame_idxs}")

    rank = 0
    for kind, k in selected:
        for head_label, hh in (("fg", FG_HEAD), ("bg", BG_HEAD)):
            rel = rel_store[k][None][hh]                # (n_raw,)
            rel_3d = rel.reshape(T_raw, H_raw, W_raw)
            t_, r_, c_ = post_to_frc(int(k), T_raw, H_raw, W_raw, sm)
            tag = f"{kind}_k{int(k)}_t{t_}r{r_}c{c_}_h{hh}_{head_label}"
            png = out_clip / f"fig_{rank:02d}_{tag}.png"
            mp4 = out_clip / f"fig_{rank:02d}_{tag}.mp4"
            title = (f"{d['video']}  blk={PICK_BLOCK} head={hh} ({head_label})  "
                     f"token=k{int(k)} ({kind}) at (t={t_},r={r_},c={c_})  "
                     f"norm={enc_norms[int(k)]:.1f}")
            render_overlay(rel_3d, frames, title, png, mp4, playback_fps=1.0)
        rank += 1
    print(f"  wrote {rank} tokens × 2 heads = {rank*2} renders to {out_clip}")

    # ---- PART 2: per-token isolated-forward word distributions ----
    md = [f"# Stage 5 supp PART 2 — clip {d['video']}",
          f"\n7 tokens (top-3 by encoder norm + 3 random sinks + 1 random non-sink, seed=0). "
          f"For each, mask LLM attention TO all other video keys, forward all 28 LLM layers, "
          f"decode the kept video token's L27 hidden state via final_norm + lm_head. Top-15 "
          f"vocab tokens per token. Illustrative on 1 clip — per-class aggregate (300 images) "
          f"is the quantitative follow-up.\n"]
    print(f"  PART 2: per-token isolated-forward decode ...")
    for rank_, (kind, k) in enumerate(selected):
        t_, r_, c_ = post_to_frc(int(k), T_raw, H_raw, W_raw, sm)
        top = isolate_and_decode(model, processor, inputs, use_aiv, layers,
                                    prompt_S, video_pos, int(k), T_raw, H_raw,
                                    W_raw, sm, n_layers, final_norm, lm_head,
                                    tokenizer, top_k=15)
        md.append(f"\n## {rank_:02d}  `{kind}`  k_post={int(k)}  "
                   f"(t={t_}, r={r_}, c={c_})  norm={enc_norms[int(k)]:.1f}\n")
        if top is None:
            md.append("(decode failed)")
            continue
        md.append("| rank | vocab_id | prob | decoded |")
        md.append("|---:|---:|---:|---|")
        for r_idx, (vid, pr, s) in enumerate(top):
            md.append(f"| {r_idx} | {vid} | {pr:.4f} | {s!r} |")
        print(f"    [{rank_:02d}] {kind:<15s} k={int(k):4d}  top-3: "
              + " | ".join(f"{s!r}({pr:.3f})" for _, pr, s in top[:3]))
    (out_clip / "word_distributions.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {out_clip / 'word_distributions.md'}")
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
    layers = thinker_layers(model)
    n_layers = len(layers)
    final_norm = model.thinker.model.norm
    lm_head    = model.thinker.lm_head
    print(f"  ViT: {len(visual_enc.blocks)} blocks, "
          f"{visual_enc.config.num_heads} heads, "
          f"full-attn idxs = {visual_enc.config.fullatt_block_indexes}")
    print(f"  LLM: {n_layers} decoder layers; fg/bg heads at ViT blk "
          f"{PICK_BLOCK}: h{FG_HEAD} (fg) / h{BG_HEAD} (bg) [from set-level PART 1]")

    data = json.load(open(DEFAULT_QA))
    by_name = {d["video"]: d for d in data}
    for cname in args.clips:
        if cname not in by_name:
            print(f"  [skip] {cname} not in sampled_entities")
            continue
        d = by_name[cname]
        print(f"\n=== {cname} ===")
        err = process_clip(d, model, processor, visual_enc, layers, n_layers,
                            final_norm, lm_head, processor.tokenizer, out_root)
        if err: print(f"  err: {err}")


if __name__ == "__main__":
    main()
