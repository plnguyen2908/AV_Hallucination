"""Memory-efficient vision/audio ENCODER attention for Qwen2.5-Omni.

Problem: our intervention loads the model with attn_implementation="eager"
(required so the thinker decoder materialises attention weights we can patch).
But eager ALSO forces the vision & audio encoders to build a dense
(seq x seq) block-diagonal mask and materialise the full score matrix
(modeling_qwen2_5_omni.py lines 1017-1024 vision, 632-640 audio). For a real
WorldSense video that is ~20k+ patches -> tens of GB -> OOM even at 15 s.
The encoders' attention is block-diagonal per frame/chunk (via cu_seqlens),
so we replace their forward with a per-block SDPA loop that never materialises
the full matrix. flash_attn is not installed, so we can't use the flash path.

Only the ENCODERS are touched; the thinker decoder (Qwen2_5OmniAttention)
keeps its eager patched forward, so the intervention is unaffected.

Usage:
    from _5_efficient_encoders import patch_efficient_encoders
    model, processor = load_omni(...)          # eager
    patch_qwen_attention(model)                # thinker decoder patch
    patch_efficient_encoders(model)            # encoders -> block SDPA
"""
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as _mq


def _vision_block_sdpa(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = _mq.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = _mq.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    cs = cu_seqlens.tolist()
    out = torch.empty_like(q)  # (seq, heads, hd)
    for i in range(1, len(cs)):
        a, b = cs[i - 1], cs[i]
        if b <= a:
            continue
        qi = q[a:b].transpose(0, 1).unsqueeze(0)  # (1, heads, n, hd)
        ki = k[a:b].transpose(0, 1).unsqueeze(0)
        vi = v[a:b].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        out[a:b] = oi.squeeze(0).transpose(0, 1)  # (n, heads, hd)
    out = out.reshape(seq_length, -1)
    return self.proj(out)


def _audio_block_sdpa(self, hidden_states, cu_seqlens=None):
    seq_length, _ = hidden_states.size()
    q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    cs = cu_seqlens.tolist()
    out = torch.empty_like(q)  # (seq, heads, hd)
    for i in range(1, len(cs)):
        a, b = cs[i - 1], cs[i]
        if b <= a:
            continue
        qi = q[a:b].transpose(0, 1).unsqueeze(0)
        ki = k[a:b].transpose(0, 1).unsqueeze(0)
        vi = v[a:b].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        out[a:b] = oi.squeeze(0).transpose(0, 1)
    out = out.reshape(seq_length, self.embed_dim)
    return self.out_proj(out)


def patch_efficient_encoders(model):
    """Install block-SDPA forwards on every vision/audio encoder attention
    module — eager AND sdpa variants (both build a dense (N,N) cu_seqlens mask
    that OOMs on long video/audio). The thinker decoder is a different class and
    is untouched. Handles accelerate's per-instance `_old_forward` hook."""
    vis_classes = tuple(c for c in (getattr(_mq, "Qwen2_5OmniVisionAttention", None),
                                    getattr(_mq, "Qwen2_5OmniVisionSdpaAttention", None))
                        if c is not None)
    aud_classes = tuple(c for c in (getattr(_mq, "Qwen2_5OmniAudioAttention", None),
                                    getattr(_mq, "Qwen2_5OmniAudioSdpaAttention", None))
                        if c is not None)
    # class-level (covers instances without an accelerate wrapper)
    for c in vis_classes:
        c.forward = _vision_block_sdpa
    for c in aud_classes:
        c.forward = _audio_block_sdpa
    n = 0
    for _, mod in model.named_modules():
        fwd = None
        if isinstance(mod, vis_classes):
            fwd = _vision_block_sdpa
        elif isinstance(mod, aud_classes):
            fwd = _audio_block_sdpa
        if fwd is None:
            continue
        if hasattr(mod, "_old_forward"):
            mod._old_forward = types.MethodType(fwd, mod)
        else:
            mod.forward = types.MethodType(fwd, mod)
        n += 1
    print(f"[efficient_encoders] block-SDPA installed on {n} encoder attn modules",
          flush=True)
    return n
