"""Per-head zero-ablation for Qwen3-Omni-30B-A3B's thinker.

Port of method/qwen2_5_omni/head_attribution.py. Deltas:

1. o_proj enumeration goes through the thinker's decoder layers (no substring
   scan, so the audio/vision encoders' o_projs are never touched). The MoE
   MLP does not participate — only the attention o_proj is hooked.
2. head_dim comes from the config's EXPLICIT `head_dim` (128), not
   hidden_size // num_attention_heads (2048 // 32 = 64). The o_proj input is
   head_num x head_dim = 4096-wide, so the per-head slice must use 128.
3. Cache restoration logic is unchanged; Qwen3-Omni uses the standard
   `transformers.cache_utils.DynamicCache`.
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)

# Module-level state set by set_zero_ablation_greedy_search().
global_tokenizer = None
hallucinated_entities = []
non_hallucinated_entities = []
inf_score_type = []

# Per-sample cap on the number of ablated target tokens. A degenerate
# repetition loop ("once" x1000) or a long confabulation can otherwise trigger
# thousands of 784-head ablation passes inline during generation and dominate
# the whole run. Set HALLUC_MAX_TARGETS to bound it (None = unlimited).
import os as _os
_MAX_TARGETS = (int(_os.environ["HALLUC_MAX_TARGETS"])
                if _os.environ.get("HALLUC_MAX_TARGETS") else None)


def set_tokenizer(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer


def set_entity_list(hal, non_hal, score_type):
    global hallucinated_entities, non_hallucinated_entities, inf_score_type
    hallucinated_entities = hal
    non_hallucinated_entities = non_hal
    inf_score_type = score_type


def _collect_o_proj_modules(self_model):
    """Return the list of o_proj modules from the thinker's decoder layers.

    `self` inside `_sample` is the thinker (since the composite's `generate`
    eventually calls `thinker.generate()`, which calls `self._sample(...)`).
    The thinker exposes its LLM as `self.model.layers`."""
    layers = self_model.model.layers  # standard Qwen2-style decoder layers
    return [layer.self_attn.o_proj for layer in layers]


# --- transformers>=4.54 DynamicCache helpers ---------------------------------
# The cache was refactored: no more `pkv.key_cache[i]` / `pkv.value_cache[i]`
# lists. State now lives on `pkv.layers[i].keys` / `.values` (DynamicLayer),
# and `update()` does torch.cat (a fresh tensor), so it never mutates the
# stored tensor in place. That lets us snapshot by cloning ONCE and restore by
# plain assignment — the next forward cats into a new tensor, leaving the
# snapshot intact.
def _cache_seq_len(pkv):
    try:
        return pkv.get_seq_length()
    except Exception:
        return 0


def _snapshot_cache(pkv):
    """Return a per-layer list of (keys, values) clones (None for empty layers)."""
    snap = []
    for layer in pkv.layers:
        if getattr(layer, "is_initialized", False) and layer.keys is not None \
                and layer.keys.numel() > 0:
            snap.append((layer.keys.clone(), layer.values.clone()))
        else:
            snap.append(None)
    return snap


def _restore_cache(pkv, snap, trim_last=False):
    """Overwrite each layer's keys/values from a snapshot; optionally drop the
    last cached token (trim_last, for the first-token single-decode context).
    No clone: the snapshot tensors are never written in place by the forward."""
    for layer, kv in zip(pkv.layers, snap):
        if kv is None:
            continue
        k, v = kv
        if trim_last:
            k = k[:, :, :-1, :]
            v = v[:, :, :-1, :]
        layer.keys = k
        layer.values = v


# --- batched-head ablation --------------------------------------------------
# Instead of 1 forward per (layer, head) — layer_num*head_num single-token
# forwards per target — ablate all `head_num` heads of ONE layer in a single
# batch-`head_num` forward: batch-row b zeroes head b at that layer, so the
# batch dim enumerates heads. That is head_num-fewer forward launches (48 vs
# 1536 for Qwen3-Omni), an exact reformulation (0/1 masking == zeroing), not an
# approximation. Toggle off with HALLUC_BATCH_HEADS=0 to fall back to per-head.
_BATCH_HEADS = _os.environ.get("HALLUC_BATCH_HEADS", "1") != "0"
# Heads ablated per forward. 32 (=head_num) is the max speedup (48 forwards/
# target); smaller chunks trade speed for closer bf16 agreement with the
# batch-1 per-head path (batch-size-dependent GEMM rounding). Clamped to
# head_num at use.
_HEAD_CHUNK = int(_os.environ.get("HALLUC_HEAD_CHUNK", "32"))

_HEAD_MASK_CACHE = {}


def _get_head_mask(head_num, head_dim, device, dtype):
    """Full [head_num, 1, head_num*head_dim] mask: row i zeros head i, else 1.
    A chunk [c0:c0+b] of this masks a contiguous head range. Cached per
    (device, dtype) since o_proj inputs live on different shards."""
    key = (head_num, head_dim, device, dtype)
    m = _HEAD_MASK_CACHE.get(key)
    if m is None:
        o_in = head_num * head_dim
        m = torch.ones(head_num, 1, o_in, device=device, dtype=dtype)
        for b in range(head_num):
            m[b, :, head_dim * b : head_dim * (b + 1)] = 0
        _HEAD_MASK_CACHE[key] = m
    return m


def _restore_cache_batched(pkv, snap, B, trim_last=False):
    """Restore per-layer keys/values from a batch-1 snapshot, tiled to batch B.
    Uses repeat (a contiguous copy, NOT expand): a stride-0 batch view aliases
    all rows and interacts badly with the GQA repeat_kv reshape / attention
    kernels, silently corrupting per-row results."""
    for layer, kv in zip(pkv.layers, snap):
        if kv is None:
            continue
        k, v = kv
        if trim_last:
            k = k[:, :, :-1, :]
            v = v[:, :, :-1, :]
        reps = (B,) + (1,) * (k.dim() - 1)
        layer.keys = k.repeat(*reps)
        layer.values = v.repeat(*reps)


def _expand_inputs_to_batch(inputs, B):
    """Return a shallow copy of the single-token ablation inputs tiled to batch
    B (contiguous repeat, not expand). cache_position is shared (not batched).
    Handles mrope position_ids ([3, batch, seq]) and plain ([batch, seq])."""
    out = dict(inputs)
    for key in ("input_ids", "inputs_embeds", "attention_mask"):
        t = out.get(key)
        if torch.is_tensor(t) and t.shape[0] == 1:
            out[key] = t.repeat(B, *([1] * (t.dim() - 1)))
    pos = out.get("position_ids")
    if torch.is_tensor(pos):
        if pos.dim() == 3 and pos.shape[1] == 1:      # mrope [3, 1, S]
            out["position_ids"] = pos.repeat(1, B, 1)
        elif pos.dim() == 2 and pos.shape[0] == 1:    # plain [1, S]
            out["position_ids"] = pos.repeat(B, 1)
    return out


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """Patched `GenerationMixin._sample` that performs per-head zero-ablation
    at every token whose decoded substring matches a hallucinated /
    non-hallucinated entity. Returns
    `(sequences_or_output, hallucination_influences, non_hallucination_influences)`.

    NOTE: copy of the VideoLLaMA2 version; only the o_proj collection
    changed (to the deterministic decoder-layer enumeration above)."""
    # init values from generation_config
    pad_token_id = generation_config.pad_token_id
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a "
            f"`LogitsProcessorList` instance (it is {logits_warper})."
        )

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    global hallucinated_entities
    global non_hallucinated_entities
    global inf_score_type

    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    # transformers 4.52+ signature: (seq_length, device, model_kwargs).
    # Older releases (which VideoLLaMA2's port was written for) accepted
    # (input_ids, model_kwargs).
    try:
        model_kwargs = self._get_initial_cache_position(
            input_ids.shape[1], input_ids.device, model_kwargs
        )
    except TypeError:
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    count = 0
    hallucination_influences = {}
    non_hallucination_influences = {}

    o_proj_modules = _collect_o_proj_modules(self)
    layer_num = len(o_proj_modules)

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Clone the cache BEFORE the main forward so ablation passes can
        # reset to this exact sequence length. DynamicCache.update() uses
        # torch.cat (creates a new tensor) so these clones won't be mutated
        # by subsequent forwards.
        pkv = model_inputs.get("past_key_values")
        is_dynamic_cache = isinstance(pkv, DynamicCache)
        initial_cache_empty = is_dynamic_cache and _cache_seq_len(pkv) == 0
        if is_dynamic_cache and not initial_cache_empty:
            before_snap = _snapshot_cache(pkv)
        else:
            before_snap = None

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Snapshot the cache AFTER the main forward — even when the cache was
        # initially empty, the main forward populated it and ablation passes
        # must not corrupt it.
        if is_dynamic_cache and _cache_seq_len(pkv) > 0:
            after_snap = _snapshot_cache(pkv)
        else:
            after_snap = None

        if synced_gpus and this_peer_finished:
            continue

        # Match transformers 4.52's upstream _sample: copy + cast to fp32 +
        # move to input_ids.device. Without this, on multi-GPU device_map the
        # logits sit on the LM-head's shard and the greedy argmax silently
        # diverges from eval's (which goes through the upstream `_sample`).
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # device_map='auto' often places the LM head on the last GPU shard,
        # so next_tokens can end up on a different device than input_ids /
        # unfinished_sequences. Align before any subsequent arithmetic.
        if next_tokens.device != input_ids.device:
            next_tokens = next_tokens.to(input_ids.device)

        next_word = (
            global_tokenizer.decode(next_tokens[0])
            if global_tokenizer
            else str(next_tokens[0].item())
        )

        original_probs = F.softmax(next_token_logits.float(), dim=-1)
        original_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)

        # Qwen3-Omni's thinker.config is composite (text/audio/vision); the
        # LLM dims live on text_config or on the inner text-model's config.
        text_cfg = self.model.config
        if not hasattr(text_cfg, "num_attention_heads"):
            text_cfg = getattr(self.config, "text_config", self.config)
        head_num = text_cfg.num_attention_heads
        # CRITICAL: Qwen3 sets an explicit head_dim (128) that is NOT
        # hidden_size // num_attention_heads (2048 // 32 = 64). The o_proj
        # input concatenates head_num x head_dim = 4096, so slicing must use
        # the config head_dim; the naive quotient would zero the wrong columns.
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // head_num

        def attach_custom_hook(layer_idx, head_idx):
            def hook_fn(module, input):
                ablated = input[0].clone()
                ablated[:, :, head_dim * head_idx : head_dim * (head_idx + 1)] = 0
                return (ablated,)

            return hook_fn

        word = next_word.strip()
        matched_hal = word and any(word in ent for ent in hallucinated_entities)
        matched_non_hal = word and any(word in ent for ent in non_hallucinated_entities)
        is_target = matched_hal or matched_non_hal

        # Bound per-sample ablation work (see _MAX_TARGETS above): once the cap
        # is hit, stop ablating further targets in this sample (generation
        # continues normally, the token is just not attributed).
        if is_target and _MAX_TARGETS is not None and count >= _MAX_TARGETS:
            is_target = matched_hal = matched_non_hal = False

        if is_target:
            influences = [[None for _ in range(head_num)] for _ in range(layer_num)]

            # For the first token (initial_cache_empty), after_snap holds the
            # full prefill KV (positions 0..N-1). Build a single-token decode
            # context: past_kv = positions 0..N-2, input = last prompt token.
            # This keeps the first-token ablation identical in cost to all
            # later tokens (1-token forward instead of full N-token prefill).
            if initial_cache_empty and is_dynamic_cache and after_snap is not None:
                ablation_inputs = dict(model_inputs)
                # Qwen2.5-Omni accepts these multimodal kwargs on the FIRST
                # forward only; on the cached follow-up they must be dropped
                # so the model can't try to re-encode them.
                for k in (
                    "input_features",
                    "feature_attention_mask",
                    "audio_feature_lengths",
                    "feature_lens",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "video_second_per_grid",
                    "second_per_grid_ts",
                    "audio_token_index",
                ):
                    ablation_inputs.pop(k, None)
                # VideoLLaMA2 also pops "images"; harmless if absent here.
                ablation_inputs.pop("images", None)
                if "inputs_embeds" in model_inputs and model_inputs["inputs_embeds"] is not None:
                    ablation_inputs["inputs_embeds"] = model_inputs["inputs_embeds"][:, -1:, :]
                    ablation_inputs.pop("input_ids", None)
                elif "input_ids" in model_inputs and model_inputs["input_ids"] is not None:
                    ablation_inputs["input_ids"] = model_inputs["input_ids"][:, -1:]
                if "position_ids" in ablation_inputs and ablation_inputs["position_ids"] is not None:
                    ablation_inputs["position_ids"] = ablation_inputs["position_ids"][:, -1:]
                if "cache_position" in ablation_inputs and ablation_inputs["cache_position"] is not None:
                    ablation_inputs["cache_position"] = ablation_inputs["cache_position"][-1:]
            else:
                ablation_inputs = model_inputs

            tgt = next_tokens[0]
            orig_p = original_probs[0, tgt]
            orig_lp = original_log_probs[0, tgt]

            def _influence(ap, alp):
                if inf_score_type == "abs_prob_diff":
                    return (orig_p - ap).abs().item()
                if inf_score_type == "log_prob_diff":
                    return (orig_lp - alp).item()
                return (orig_p - ap).item()  # prob_diff (default)

            if _BATCH_HEADS:
                # Ablate heads in chunks of C: one batch-C forward per (layer,
                # chunk); row j ablates head c0+j. C=head_num => 48 forwards/
                # target. Inputs cached per batch size.
                C = max(1, min(_HEAD_CHUNK, head_num))
                _debug_noablate = _os.environ.get("HALLUC_DEBUG_NOABLATE") == "1"
                _inputs_by_b = {}

                def _chunk_inputs(b):
                    if b not in _inputs_by_b:
                        _inputs_by_b[b] = _expand_inputs_to_batch(ablation_inputs, b)
                    return _inputs_by_b[b]

                for layer_idx in range(layer_num):
                    o_proj_module = o_proj_modules[layer_idx]
                    for c0 in range(0, head_num, C):
                        heads = list(range(c0, min(c0 + C, head_num)))
                        b = len(heads)

                        def _hook(module, inp, c0=c0, b=b):
                            x = inp[0]
                            if _debug_noablate:
                                return (x,)
                            m = _get_head_mask(head_num, head_dim, x.device, x.dtype)
                            return (x * m[c0 : c0 + b],)

                        if initial_cache_empty and is_dynamic_cache and after_snap is not None:
                            _restore_cache_batched(pkv, after_snap, b, trim_last=True)
                        elif before_snap is not None:
                            _restore_cache_batched(pkv, before_snap, b)

                        hook_handle = o_proj_module.register_forward_pre_hook(_hook)
                        outputs_ablated = self(
                            **_chunk_inputs(b),
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                        hook_handle.remove()

                        logits_b = outputs_ablated.logits[:, -1, :].to(
                            copy=True, dtype=torch.float32, device=input_ids.device
                        )
                        ablated_probs = F.softmax(logits_b, dim=-1)
                        ablated_log_probs = F.log_softmax(logits_b, dim=-1)
                        for j, head_idx in enumerate(heads):
                            ap = ablated_probs[j, tgt]
                            alp = ablated_log_probs[j, tgt]
                            influences[layer_idx][head_idx] = {
                                "original_prob": orig_p.item(),
                                "perturbed_prob": ap.item(),
                                "original_log_prob": orig_lp.item(),
                                "perturbed_log_prob": alp.item(),
                                "influence": _influence(ap, alp),
                            }
            else:
                for layer_idx in range(layer_num):
                    o_proj_module = o_proj_modules[layer_idx]
                    for head_idx in range(head_num):
                        # Restore pkv to the pre-forward state before each pass.
                        if initial_cache_empty and is_dynamic_cache and after_snap is not None:
                            _restore_cache(pkv, after_snap, trim_last=True)
                        elif before_snap is not None:
                            _restore_cache(pkv, before_snap)

                        hook_handle = o_proj_module.register_forward_pre_hook(
                            attach_custom_hook(layer_idx, head_idx)
                        )
                        outputs_ablated = self(
                            **ablation_inputs,
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                        hook_handle.remove()

                        logits_1 = outputs_ablated.logits[:, -1, :].to(
                            copy=True, dtype=torch.float32, device=input_ids.device
                        )
                        ablated_probs = F.softmax(logits_1, dim=-1)
                        ablated_log_probs = F.log_softmax(logits_1, dim=-1)
                        ap = ablated_probs[0, tgt]
                        alp = ablated_log_probs[0, tgt]
                        influences[layer_idx][head_idx] = {
                            "original_prob": orig_p.item(),
                            "perturbed_prob": ap.item(),
                            "original_log_prob": orig_lp.item(),
                            "perturbed_log_prob": alp.item(),
                            "influence": _influence(ap, alp),
                        }

            key = f"{word}_{count}"
            if matched_hal:
                hallucination_influences[key] = influences
            if matched_non_hal:
                non_hallucination_influences[key] = influences

            # Restore "after" cache so generation continues from the right state.
            if after_snap is not None:
                _restore_cache(pkv, after_snap)

            # empty_cache() forces a GPU sync and dominates per-target time
            # (~19s/target on the 3-GPU-sharded model). The caching allocator
            # reuses freed single-token-forward memory anyway, so call it only
            # periodically to bound fragmentation instead of every target.
            if count % 16 == 0:
                torch.cuda.empty_cache()

        count += 1

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0

        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return (
                GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                ),
                hallucination_influences,
                non_hallucination_influences,
            )
    else:
        return input_ids, hallucination_influences, non_hallucination_influences


def set_zero_ablation_greedy_search(
    tokenizer, hallucinated_entities, non_hallucinated_entities, influence_score
):
    """Patches `transformers.generation.utils.GenerationMixin._sample` so that
    every subsequent `model.generate(...)` call returns a three-tuple
    `(output, hal_influences, non_hal_influences)`."""
    set_tokenizer(tokenizer)
    set_entity_list(hallucinated_entities, non_hallucinated_entities, influence_score)
    transformers.generation.utils.GenerationMixin._sample = _sample
