"""Quick probe: print cu_seqlens at each ViT block for a video input
to determine whether attention is cross-frame or per-frame at the
full-attn blocks (idxs [7, 15, 23, 31])."""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, prepare_inputs
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniVisionAttention, apply_rotary_pos_emb_vision,
)

DEFAULT_QA = _REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json"
DEFAULT_VID = _REPO / "data/ActivityNet/videos"
VIDEO_TOKEN_ID = 151656

_cu_log = {}

def patched(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    bi = getattr(self, "_blk_idx", -1)
    if bi not in _cu_log:
        _cu_log[bi] = cu_seqlens.detach().cpu().numpy().copy()
    # delegate to a vanilla implementation just for forward correctness
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    am = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min,
                     device=q.device, dtype=q.dtype)
    for i in range(1, len(cu_seqlens)):
        am[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = 0
    q = q.transpose(0,1); k = k.transpose(0,1); v = v.transpose(0,1)
    aw = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.head_dim)
    aw = aw + am
    aw = nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    ao = torch.matmul(aw, v).transpose(0,1).reshape(seq_length, -1)
    return self.proj(ao)

Qwen2_5OmniVisionAttention.forward = patched

print("Loading...")
model, processor = load_omni("Qwen/Qwen2.5-Omni-7B",
                               device_map="balanced_low_0")
visual_enc = model.thinker.visual
for i, blk in enumerate(visual_enc.blocks):
    blk.attn._blk_idx = i

data = json.load(open(DEFAULT_QA))
d = next(x for x in data if x["video"] == "021bd34fe8ff.mp4")
conv = build_conversation(str(Path(DEFAULT_VID) / d["video"]), d["question"], "v")
inputs, use_aiv = prepare_inputs(processor, conv, "v",
                                    model.device, model.dtype)
grid = inputs["video_grid_thw"].cpu().numpy()
T, H, W = int(grid[0,0]), int(grid[0,1]), int(grid[0,2])
print(f"grid T,H,W = {T},{H},{W}, n_raw = {T*H*W}, per_frame = {H*W}")

with torch.inference_mode():
    model.thinker(**inputs, use_audio_in_video=use_aiv,
                  output_attentions=False, return_dict=True,
                  use_cache=False)

print("\nFull-attn block idxs are [7, 15, 23, 31].\n")
fullatt = [7, 15, 23, 31]
for bi in sorted(_cu_log.keys()):
    cs = _cu_log[bi]
    is_full = "FULL" if bi in fullatt else "window"
    print(f"block {bi:2d} [{is_full}]: cu_seqlens len={len(cs)}, "
          f"first 10 = {cs[:10].tolist()}{'...' if len(cs)>10 else ''}, "
          f"last = {cs[-1]}")
    # Block sizes:
    sizes = np.diff(cs)
    print(f"             block sizes: min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f} n_blocks={len(sizes)}")
