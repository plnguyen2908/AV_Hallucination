"""Stage 5 supp v2 — Fix 3 + Fix 4: block sweep + head ranking.

For the user-selected clip, at blocks {29, 30, 31}, rank all 16 ViT
heads by anchored-Gaussian fg-vs-bg sink alignment:

  For each (block, head):
    - For each top-3 sink (post-merger, by encoder norm):
        - Get strongest raw cell's column A[h, :, raw_k] within sink's
          OWN frame (cross-frame attn is structurally 0 in this ViT).
        - Build fg_mask = 2D Gaussian centered at sink's (r, c) with
          sigma = max(H_raw, W_raw) / 8.
        - fg_share = sum(rel * fg_mask) / sum(rel)         (fraction of
                                                            relevance mass
                                                            inside the fg
                                                            Gaussian)
    - Aggregate: mean over 3 sinks.

Top fg-type head per block = argmax fg_share. Top bg-type head per
block = argmin fg_share (relevance most spread out / least anchored to
sink).

Block selection (Fix 3): "least diffuse" = block with HIGHEST mean top-3-sink
peak attention (vmax of strongest raw's own-frame column). Pick that
block for the final figure.

Outputs:
  results/.../stage5_supp_v2/clip_<id>/
    head_block_ranking.csv      (all 3 blocks * 16 heads = 48 rows)
    head_block_ranking_summary.md
"""
import argparse
import json
import math
import sys
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
OUT_ROOT = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp_v2"
VIDEO_TOKEN_ID = 151656
TAU_PROP = 100.0
BLOCKS = [29, 30, 31]

_attn_callback = None


def patched_vision_attn_forward(self, hidden_states, cu_seqlens,
                                 rotary_pos_emb=None):
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    am = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min,
                     device=q.device, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        am[..., cu_seqlens[i-1]:cu_seqlens[i],
            cu_seqlens[i-1]:cu_seqlens[i]] = 0
    q = q.transpose(0,1); k = k.transpose(0,1); v = v.transpose(0,1)
    aw = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.head_dim)
    aw = aw + am
    aw = nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    if _attn_callback is not None:
        _attn_callback(self, aw, cu_seqlens)
    ao = torch.matmul(aw, v).transpose(0,1).reshape(seq_length, -1)
    return self.proj(ao)


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


def gaussian_mask(H, W, cr, cc, sigma):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True,
                   help="ActivityNet clip filename (e.g. 'foo.mp4').")
    args = p.parse_args()
    clip = args.clip

    print(f"Loading Qwen2.5-Omni ...")
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                                   device_map="balanced_low_0")
    visual_enc = None
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model.thinker, attr):
            visual_enc = getattr(model.thinker, attr); break

    data = json.load(open(DEFAULT_QA))
    qa_map = {x["video"]: x for x in data}
    question = qa_map[clip]["question"] if clip in qa_map else "Describe this video."
    video_path = Path(DEFAULT_VID) / clip
    conv = build_conversation(str(video_path), question, "v")
    inputs, use_aiv = prepare_inputs(processor, conv, "v",
                                       model.device, model.dtype)
    prompt_S = int(inputs["input_ids"].shape[1])
    ids_np = inputs["input_ids"][0].cpu().numpy()
    video_pos = np.where(ids_np[:prompt_S] == VIDEO_TOKEN_ID)[0]
    n_llm = int(video_pos.size)
    grid = inputs["video_grid_thw"].cpu().numpy()
    T_raw = int(grid[0,0]); H_raw = int(grid[0,1]); W_raw = int(grid[0,2])
    sm = 2
    H_post, W_post = H_raw // sm, W_raw // sm
    n_raw = T_raw * H_raw * W_raw
    assert T_raw * H_post * W_post == n_llm

    print(f"  clip {clip}: T_raw={T_raw} H_raw={H_raw} W_raw={W_raw} "
          f"n_raw={n_raw} n_llm={n_llm}")

    # Pass 1: encoder (post-merger) norms
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
    print(f"    P_prop sinks (norm>{TAU_PROP}): {sinks.size} / {n_llm}")

    # Top-3 sinks by norm + 3 random non-sinks (seed=0) for cross-check
    rng = np.random.default_rng(0)
    nonsinks = np.where(enc_norms < float(np.percentile(enc_norms, 25)))[0]
    top3 = sinks[np.argsort(-enc_norms[sinks])[:3]].tolist()
    rand_non = rng.choice(nonsinks, size=min(3, len(nonsinks)),
                            replace=False).tolist()
    print("    top-3 sinks:")
    sink_meta = []
    for k in top3:
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        raws = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
        sink_meta.append(dict(k=k, kind="sink", t=t, r=r, c=c,
                                norm=float(enc_norms[k]), raws=raws))
        print(f"      [S] k={k:4d}  norm={enc_norms[k]:.2f}  "
              f"(t={t},R_post={r},C_post={c})  raws={[rm['idx'] for rm in raws]}")
    print("    3 random non-sinks:")
    non_meta = []
    for k in rand_non:
        k = int(k)
        t, r, c = post_to_frc(k, T_raw, H_raw, W_raw, sm)
        raws = post_to_raw_indices(k, T_raw, H_raw, W_raw, sm)
        non_meta.append(dict(k=k, kind="nonsink", t=t, r=r, c=c,
                                norm=float(enc_norms[k]), raws=raws))
        print(f"      [N] k={k:4d}  norm={enc_norms[k]:.2f}  "
              f"(t={t},R_post={r},C_post={c})  raws={[rm['idx'] for rm in raws]}")
    all_meta = sink_meta + non_meta

    # Pass 2: capture attention at each of BLOCKS for each head, for the
    # 4 raw cells of each of {top-3 sinks, 3 random non-sinks}. Memory:
    # 6 tokens * 4 raws * 16 heads * 3 blocks * n_raw float32 ~ 16-20 MB.
    for i, blk in enumerate(visual_enc.blocks):
        blk.attn._blk_idx = i
    n_heads = visual_enc.blocks[0].attn.num_heads
    print(f"    n_heads = {n_heads}")
    all_keys = [m["k"] for m in all_meta]
    # store[block][k_post] shape (4, n_heads, n_raw)
    store = {b: {k: np.zeros((4, n_heads, n_raw), dtype=np.float32)
                  for k in all_keys}
               for b in BLOCKS}

    def cb(self_mod, attn_weights, cu_seqlens):
        bi = getattr(self_mod, "_blk_idx", None)
        if bi not in BLOCKS: return
        if attn_weights.shape[-1] != n_raw: return
        for m_ in all_meta:
            for ri, rm in enumerate(m_["raws"]):
                for h in range(n_heads):
                    store[bi][m_["k"]][ri, h] = (
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

    # Compute per-(block, head, sink) metrics: vmax of strongest raw's
    # own-frame column, and anchored-Gaussian fg_share.
    sigma = max(H_raw, W_raw) / 8.0
    print(f"\nMETRICS  (Gaussian sigma = {sigma:.1f} for fg_share)")

    csv_rows = ["block,head,token_kind,token_k,strongest_raw,"
                 "vmax,fg_share,col_mean_own_frame"]
    # block_stats[b][h] -> dict of aggregated metrics over top-3 sinks
    # block_stats[b][h]['mean_sink_colmean'] and ['mean_non_colmean']
    # let us also report sink-vs-non-sink ratio per (block, head).
    block_stats = {b: {} for b in BLOCKS}
    for b in BLOCKS:
        for h in range(n_heads):
            sinks_metrics = []; non_metrics = []
            for m_ in all_meta:
                k = m_["k"]; own_t = m_["t"]
                arr = store[b][k][:, h]   # (4, n_raw)
                cubes = arr.reshape(4, T_raw, H_raw, W_raw)
                own_means = cubes[:, own_t].reshape(4, -1).mean(axis=1)
                ri = int(np.argmax(own_means))
                relevance = cubes[ri, own_t]   # (H_raw, W_raw)
                tot = float(relevance.sum())
                rm_strong = m_["raws"][ri]
                cr, cc = rm_strong["r"], rm_strong["c"]
                g = gaussian_mask(H_raw, W_raw, cr, cc, sigma)
                fg_share = (float((relevance * g).sum())
                             / max(tot, 1e-30))
                vmax = float(relevance.max())
                col_mean = float(relevance.mean())
                csv_rows.append(
                    f"{b},{h},{m_['kind']},{k},{ri},{vmax:.6e},"
                    f"{fg_share:.6f},{col_mean:.6e}")
                d_ = dict(vmax=vmax, fg_share=fg_share, col_mean=col_mean,
                            tot=tot, strongest_raw=ri)
                if m_["kind"] == "sink":
                    sinks_metrics.append(d_)
                else:
                    non_metrics.append(d_)
            mv  = float(np.mean([s["vmax"] for s in sinks_metrics]))
            mfg = float(np.mean([s["fg_share"] for s in sinks_metrics]))
            sink_cm = float(np.mean([s["col_mean"] for s in sinks_metrics]))
            non_cm  = float(np.mean([s["col_mean"] for s in non_metrics]))
            non_fg  = float(np.mean([s["fg_share"] for s in non_metrics]))
            block_stats[b][h] = dict(
                mean_vmax=mv,
                mean_sink_fg_share=mfg,
                mean_non_fg_share=non_fg,
                fg_share_gap=mfg - non_fg,
                mean_sink_colmean=sink_cm,
                mean_non_colmean=non_cm,
                sink_over_non_ratio=sink_cm / max(non_cm, 1e-30))

    out_dir = OUT_ROOT / f"clip_{Path(clip).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "head_block_ranking.csv").write_text("\n".join(csv_rows))

    # Print ranking per block
    summary = []
    summary.append(f"# Stage 5 supp v2 — head + block ranking\n")
    summary.append(f"Clip: `{clip}`  "
                    f"(T_raw={T_raw} H_raw={H_raw} W_raw={W_raw}, "
                    f"n_llm={n_llm}, P_prop sinks={sinks.size})\n")
    summary.append("Top-3 sinks (by encoder norm):\n")
    for sm_ in sink_meta:
        summary.append(f"- k={sm_['k']}  norm={sm_['norm']:.2f}  "
                        f"(t={sm_['t']}, R_post={sm_['r']}, "
                        f"C_post={sm_['c']})")
    summary.append("\nHead ranking method: anchored-Gaussian fg_share = "
                    f"fraction of own-frame relevance mass within a "
                    f"Gaussian (σ={sigma:.1f}) centered at sink's "
                    f"strongest raw position. Aggregated over top-3 "
                    f"sinks.\n")

    print("\n=== HEAD RANKING PER BLOCK ===")
    block_choice = {}
    for b in BLOCKS:
        print(f"\n  Block {b}:")
        summary.append(f"\n## Block {b}\n")
        # Order heads by sink fg_share desc
        ranking = sorted(range(n_heads),
                          key=lambda h: -block_stats[b][h]["mean_sink_fg_share"])
        summary.append("| rank | head | sink fg_share | non fg_share | "
                        "fg gap | sink/non col_mean ratio | mean vmax |")
        summary.append("|---:|---:|---:|---:|---:|---:|---:|")
        for rank, h in enumerate(ranking):
            s = block_stats[b][h]
            line = (f"    rank {rank:2d}  h{h:2d}  "
                    f"sink_fg={s['mean_sink_fg_share']:.3f}  "
                    f"non_fg={s['mean_non_fg_share']:.3f}  "
                    f"gap={s['fg_share_gap']:+.3f}  "
                    f"sink/non={s['sink_over_non_ratio']:.3f}  "
                    f"vmax={s['mean_vmax']*100:.2f}%")
            print(line)
            summary.append(f"| {rank} | h{h} | "
                            f"{s['mean_sink_fg_share']:.3f} | "
                            f"{s['mean_non_fg_share']:.3f} | "
                            f"{s['fg_share_gap']:+.3f} | "
                            f"{s['sink_over_non_ratio']:.3f} | "
                            f"{s['mean_vmax']*100:.2f}% |")
        top_fg = ranking[0]
        top_bg = ranking[-1]
        # Block "diffuseness" headline number
        best_head_vmax = max(block_stats[b][h]["mean_vmax"]
                              for h in range(n_heads))
        gap = (block_stats[b][top_fg]["mean_sink_fg_share"]
                - block_stats[b][top_bg]["mean_sink_fg_share"])
        # Find best head by sink/non ratio (orthogonal metric)
        sink_over_non = [(h, block_stats[b][h]["sink_over_non_ratio"])
                          for h in range(n_heads)]
        sink_over_non.sort(key=lambda x: -x[1])
        n_dominant = sum(1 for h, r in sink_over_non if r > 1.0)
        best_dominant = sink_over_non[0]
        block_choice[b] = dict(
            top_fg=top_fg, top_bg=top_bg,
            best_head_vmax=best_head_vmax,
            fg_bg_gap=gap,
            n_heads_sink_dominant=n_dominant,
            top_sink_dominant_head=best_dominant[0],
            top_sink_dominant_ratio=best_dominant[1])
        summary.append(
            f"\n→ top fg-type head (max sink_fg_share): **h{top_fg}** "
            f"(sink_fg={block_stats[b][top_fg]['mean_sink_fg_share']:.3f}, "
            f"non_fg={block_stats[b][top_fg]['mean_non_fg_share']:.3f})")
        summary.append(
            f"→ top bg-type head (min sink_fg_share): **h{top_bg}** "
            f"(sink_fg={block_stats[b][top_bg]['mean_sink_fg_share']:.3f})")
        summary.append(f"→ best-head mean_vmax at this block: "
                        f"{best_head_vmax*100:.2f}%")
        summary.append(f"→ heads where sink > non-sink (ratio > 1): "
                        f"{n_dominant}/{n_heads}; "
                        f"top dominant: h{best_dominant[0]} "
                        f"(ratio={best_dominant[1]:.3f})")
        print(f"    → top fg head: h{top_fg} "
              f"(sink_fg={block_stats[b][top_fg]['mean_sink_fg_share']:.3f}, "
              f"non_fg={block_stats[b][top_fg]['mean_non_fg_share']:.3f})")
        print(f"    → top bg head: h{top_bg} "
              f"(sink_fg={block_stats[b][top_bg]['mean_sink_fg_share']:.3f})")
        print(f"    → best-head mean_vmax: {best_head_vmax*100:.2f}%")
        print(f"    → sink_dominant heads: {n_dominant}/{n_heads}, "
              f"top: h{best_dominant[0]} ratio={best_dominant[1]:.3f}")

    # Block selection: "least diffuse / clearest sink structure" =
    # the block with the highest best-head mean_vmax (i.e., the block
    # at which a head exists that gives sinks a concentrated column).
    print("\n=== BLOCK SELECTION ===")
    summary.append("\n## Block selection\n")
    chosen_block = max(BLOCKS,
                        key=lambda b: block_choice[b]["best_head_vmax"])
    summary.append(f"Criterion: highest best-head mean_vmax across the "
                    f"16 heads (Fix 3: 'least diffuse / clearest sink "
                    f"structure').\n")
    for b in BLOCKS:
        ch = block_choice[b]
        marker = "  ← CHOSEN" if b == chosen_block else ""
        line = (f"- Block {b}: best-head vmax = "
                f"{ch['best_head_vmax']*100:.2f}%  "
                f"fg-bg gap = {ch['fg_bg_gap']:.4f}  "
                f"top fg = h{ch['top_fg']}, top bg = h{ch['top_bg']}"
                f"{marker}")
        print(line)
        summary.append(line)

    summary.append(f"\n## Chosen for final figure\n")
    summary.append(f"- Block: **{chosen_block}**")
    summary.append(f"- fg-type head: **h{block_choice[chosen_block]['top_fg']}**")
    summary.append(f"- bg-type head: **h{block_choice[chosen_block]['top_bg']}**")
    summary.append(f"- fg-bg gap (sink anchored Gaussian fg_share): "
                    f"{block_choice[chosen_block]['fg_bg_gap']:.4f}")
    summary.append(f"- heads where sink > non-sink at chosen block: "
                    f"{block_choice[chosen_block]['n_heads_sink_dominant']}/{n_heads}")
    if block_choice[chosen_block]["fg_bg_gap"] < 0.05:
        summary.append(f"\n**Note:** fg-bg gap < 0.05 — the fg-vs-bg "
                        f"head split is weak in this clip. If the final "
                        f"render also shows sink ≈ non-sink, that "
                        f"matches the 'no clean H10/H12 split in Qwen' "
                        f"branch the spec anticipates.")

    # Save the captured store to disk so the final render reuses it
    np.savez_compressed(
        out_dir / "head_block_captures.npz",
        T_raw=T_raw, H_raw=H_raw, W_raw=W_raw, n_raw=n_raw,
        sink_keys=np.array([m["k"] for m in sink_meta]),
        non_keys=np.array([m["k"] for m in non_meta]),
        sink_norms=np.array([m["norm"] for m in sink_meta]),
        non_norms=np.array([m["norm"] for m in non_meta]),
        sink_own_t=np.array([m["t"] for m in sink_meta]),
        non_own_t=np.array([m["t"] for m in non_meta]),
        sink_own_r=np.array([m["r"] for m in sink_meta]),
        non_own_r=np.array([m["r"] for m in non_meta]),
        sink_own_c=np.array([m["c"] for m in sink_meta]),
        non_own_c=np.array([m["c"] for m in non_meta]),
        # store_<block>_<k>: (4, n_heads, n_raw)
        **{f"store_b{b}_k{m['k']}": store[b][m["k"]]
            for b in BLOCKS for m in all_meta},
        chosen_block=chosen_block,
        top_fg_head=block_choice[chosen_block]["top_fg"],
        top_bg_head=block_choice[chosen_block]["top_bg"])

    (out_dir / "head_block_ranking_summary.md").write_text(
        "\n".join(summary))
    print(f"\nSummary -> {out_dir / 'head_block_ranking_summary.md'}")
    print(f"Captures saved to {out_dir / 'head_block_captures.npz'}")


if __name__ == "__main__":
    main()
