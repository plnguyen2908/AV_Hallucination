"""Fast clip scanner for Stage 5 supp v2.

Uses cached encoder_norms.npz (Stage 1.1, 300 ActivityNet clips) to
filter by P_prop count without model forward. Then runs `prepare_inputs`
on the top candidates to get T_raw, and saves frame-0 thumbnails for the
first N clips passing T_raw <= 4. Output suitable for visual selection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
NORMS_NPZ = _REPO / "results/qwen2_5_omni/sink_analysis/stage1_1_encoder_norms/encoder_norms.npz"
OUT_DIR = _REPO / "results/qwen2_5_omni/sink_analysis/stage5_supp_v2/candidates"
TAU_PROP = 100.0
N_WANT = 15        # target N candidates
MAX_TRY = 100      # max candidates to try (sorted by P_prop count desc)
MAX_T = 4


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached encoder norms ...")
    d = np.load(str(NORMS_NPZ), allow_pickle=True)
    names = d["video_names"]
    offsets = d["video_offsets"]
    flat = d["video_flat"]

    # Per-clip P_prop count
    cand = []
    for i, name in enumerate(names):
        norms = flat[offsets[i]:offsets[i+1]]
        p_prop = int((norms > TAU_PROP).sum())
        n_tokens = int(norms.size)
        if p_prop >= 6:
            cand.append((p_prop, n_tokens, str(name)))
    cand.sort(reverse=True)
    print(f"  cached clips: {len(names)}")
    print(f"  P_prop>=6 in cache: {len(cand)}")
    print(f"  top-10 by P_prop count:")
    for p_prop, n_tokens, name in cand[:10]:
        print(f"    P_prop={p_prop:3d}  n_tokens={n_tokens:5d}  {name}")

    # We need T_raw <= 4 for OOM safety in the analysis. Use prepare_inputs
    # (decord + processor) to get the actual T_raw per candidate without
    # model forward.
    print("\nLoading processor (no model forward needed)...")
    # We need the processor; load_omni gives both, model load takes ~30s.
    # For speed, we still call load_omni but only use the processor.
    from transformers import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("Processor loaded.")

    data = json.load(open(DEFAULT_QA))
    qa_map = {d["video"]: d for d in data}
    DEFAULT_Q = "Describe this video."

    passing = []
    for i, (p_prop, n_tokens, name) in enumerate(cand[:MAX_TRY]):
        if len(passing) >= N_WANT:
            break
        # Use QA question if available, else a generic prompt
        # (encoder/ViT analysis doesn't depend on the question).
        question = qa_map[name]["question"] if name in qa_map else DEFAULT_Q
        vp = DEFAULT_VID / name
        if not vp.exists():
            print(f"  [{i}] {name}: video missing, skip")
            continue
        conv = build_conversation(str(vp), question, "v")
        # prepare_inputs needs a model.device/dtype; use cpu/float32 just
        # to derive T_raw (we discard the tensors).
        try:
            inputs, _ = prepare_inputs(
                processor, conv, "v",
                torch.device("cpu"), torch.float32)
            grid = inputs["video_grid_thw"].cpu().numpy()
        except Exception as e:
            print(f"  [{i}] {name}: prep error {type(e).__name__}: {e}")
            continue
        T = int(grid[0, 0]); H = int(grid[0, 1]); W = int(grid[0, 2])
        if T > MAX_T:
            print(f"  [{i}] {name}: T={T} > {MAX_T} (would OOM), skip "
                  f"(P_prop={p_prop})")
            continue
        # Save frame-0 thumbnail
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(vp), num_threads=1)
            n_total = len(vr); native_fps = float(vr.get_avg_fps())
            # 1 fps Qwen-aligned: T frames at t=0, 1, ..., T-1 sec
            idxs = [min(int(round(t * native_fps)), n_total - 1)
                     for t in range(T)]
            frames = [vr[i].asnumpy() for i in idxs]
        except Exception as e:
            print(f"  [{i}] {name}: decord error {type(e).__name__}: {e}")
            continue

        # Composite all T frames into one strip
        import matplotlib.pyplot as plt
        fig_w = max(8, T * 3)
        fig, ax = plt.subplots(figsize=(fig_w, 3),
                                 constrained_layout=True)
        if T == 1:
            ax.imshow(frames[0])
        else:
            full = np.concatenate(frames, axis=1)
            ax.imshow(full)
        ax.set_title(f"{name}  P_prop={p_prop} T={T} H={H} W={W}",
                     fontsize=10)
        ax.set_xticks([(i + 0.5) * (frames[0].shape[1])
                       for i in range(T)])
        ax.set_xticklabels([f"t={t}s" for t in range(T)], fontsize=9)
        ax.set_yticks([])
        rank_str = f"{len(passing):02d}"
        png_path = OUT_DIR / f"cand_{rank_str}_p{p_prop:03d}_T{T}_{Path(name).stem}.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        passing.append(dict(name=name, p_prop=p_prop, T=T, H=H, W=W,
                              png=str(png_path)))
        print(f"  [{i}] {name}: P_prop={p_prop} T={T} -> saved "
              f"{png_path.name}")

    # Write a manifest
    manifest = OUT_DIR / "candidates_manifest.json"
    manifest.write_text(json.dumps(passing, indent=2))
    print(f"\nManifest: {manifest}")
    print(f"Saved {len(passing)} candidate thumbnails to {OUT_DIR}")


if __name__ == "__main__":
    main()
