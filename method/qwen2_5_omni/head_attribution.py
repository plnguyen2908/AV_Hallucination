"""Per-head zero-ablation for Qwen2.5-Omni's thinker.

Mirrors method/videollama2/head_attribution.py with three deltas:

1. o_proj enumeration goes through the thinker's decoder layers (no substring
   scan, so the audio/vision encoders' o_projs are never touched).
2. Config accessors (num_hidden_layers / num_attention_heads / hidden_size)
   come off `self.config` — which, when `_sample` is called via the composite
   `model.generate(...)` → `thinker.generate(...)` path, IS the thinker's
   config. So no change to the access pattern itself.
3. Cache restoration logic is unchanged; Qwen2.5-Omni uses the standard
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


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
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
        initial_cache_empty = is_dynamic_cache and len(pkv.key_cache) == 0
        if is_dynamic_cache and not initial_cache_empty:
            before_keys = [pkv.key_cache[i].clone() for i in range(len(pkv.key_cache))]
            before_values = [
                pkv.value_cache[i].clone() for i in range(len(pkv.value_cache))
            ]
        else:
            before_keys = before_values = None

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Clone the cache AFTER the main forward — even when the cache was
        # initially empty, the main forward populated it and ablation passes
        # must not corrupt it.
        if is_dynamic_cache and len(pkv.key_cache) > 0:
            after_keys = [pkv.key_cache[i].clone() for i in range(len(pkv.key_cache))]
            after_values = [
                pkv.value_cache[i].clone() for i in range(len(pkv.value_cache))
            ]
        else:
            after_keys = after_values = None

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

        # Qwen2.5-Omni's thinker.config is composite (text/audio/vision); the
        # LLM dims live on text_config or on the inner text-model's config.
        text_cfg = self.model.config
        if not hasattr(text_cfg, "num_attention_heads"):
            text_cfg = getattr(self.config, "text_config", self.config)
        head_num = text_cfg.num_attention_heads
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

        if is_target:
            influences = [[None for _ in range(head_num)] for _ in range(layer_num)]

            # For the first token (initial_cache_empty), after_keys holds the
            # full prefill KV (positions 0..N-1). Build a single-token decode
            # context: past_kv = positions 0..N-2, input = last prompt token.
            # This keeps the first-token ablation identical in cost to all
            # later tokens (1-token forward instead of full N-token prefill).
            if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                ablation_inputs = dict(model_inputs)
                # Qwen2.5-Omni accepts these multimodal kwargs on the FIRST
                # forward only; on the cached follow-up they must be dropped
                # so the model can't try to re-encode them.
                for k in (
                    "input_features",
                    "feature_attention_mask",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "video_second_per_grid",
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

            for layer_idx in range(layer_num):
                o_proj_module = o_proj_modules[layer_idx]
                for head_idx in range(head_num):
                    # Restore pkv to the pre-forward state before each pass.
                    if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                        for i in range(len(after_keys)):
                            if i < len(pkv.key_cache):
                                pkv.key_cache[i] = after_keys[i][:, :, :-1, :].clone()
                                pkv.value_cache[i] = after_values[i][:, :, :-1, :].clone()
                            else:
                                pkv.key_cache.append(after_keys[i][:, :, :-1, :].clone())
                                pkv.value_cache.append(after_values[i][:, :, :-1, :].clone())
                        pkv._seen_tokens = after_keys[0].shape[2] - 1
                    elif before_keys is not None:
                        for i in range(len(before_keys)):
                            pkv.key_cache[i] = before_keys[i]
                            pkv.value_cache[i] = before_values[i]

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

                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :].to(
                        copy=True, dtype=torch.float32, device=input_ids.device
                    )
                    ablated_probs = F.softmax(next_token_logits_ablated, dim=-1)
                    ablated_log_probs = F.log_softmax(next_token_logits_ablated, dim=-1)

                    if inf_score_type == "prob_diff":
                        influence = (
                            original_probs[0, next_tokens[0]]
                            - ablated_probs[0, next_tokens[0]]
                        ).item()
                    elif inf_score_type == "abs_prob_diff":
                        influence = (
                            (
                                original_probs[0, next_tokens[0]]
                                - ablated_probs[0, next_tokens[0]]
                            )
                            .abs()
                            .item()
                        )
                    elif inf_score_type == "log_prob_diff":
                        influence = (
                            original_log_probs[0, next_tokens[0]]
                            - ablated_log_probs[0, next_tokens[0]]
                        ).item()
                    else:
                        influence = (
                            original_probs[0, next_tokens[0]]
                            - ablated_probs[0, next_tokens[0]]
                        ).item()

                    influences[layer_idx][head_idx] = {
                        "original_prob": original_probs[0, next_tokens[0]].item(),
                        "perturbed_prob": ablated_probs[0, next_tokens[0]].item(),
                        "original_log_prob": original_log_probs[
                            0, next_tokens[0]
                        ].item(),
                        "perturbed_log_prob": ablated_log_probs[
                            0, next_tokens[0]
                        ].item(),
                        "influence": influence,
                    }

            key = f"{word}_{count}"
            if matched_hal:
                hallucination_influences[key] = influences
            if matched_non_hal:
                non_hallucination_influences[key] = influences

            # Restore "after" cache so generation continues from the right state.
            if after_keys is not None:
                for i in range(len(after_keys)):
                    pkv.key_cache[i] = after_keys[i]
                    pkv.value_cache[i] = after_values[i]

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
