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

# Global variable to hold the tokenizer
global_tokenizer = None
hallucinated_entities = []
non_hallucinated_entities = []
inf_score_type = []


def set_tokenizer(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer


def set_entity_list(
    hallucinated_entities_1, non_hallucinated_entities_1, inf_score_type_1
):
    global hallucinated_entities, non_hallucinated_entities, inf_score_type
    hallucinated_entities = hallucinated_entities_1
    non_hallucinated_entities = non_hallucinated_entities_1
    inf_score_type = inf_score_type_1


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
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
            `generation_config`)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
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
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
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

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    count = 0
    hallucination_influences = {}
    non_hallucination_influences = {}
    o_proj_modules = []

    # Collect o_proj modules from Qwen2 layers inside VideoLLaMA2
    for name, module in self.model.named_modules():
        if "o_proj" in name:
            o_proj_modules.append(module)

    layer_num = len(o_proj_modules)

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Clone the cache BEFORE the main forward so ablation passes can reset to
        # this exact sequence length.  DynamicCache.update() uses torch.cat (creates
        # a new tensor) so these clones won't be mutated by subsequent forwards.
        pkv = model_inputs.get("past_key_values")
        is_dynamic_cache = isinstance(pkv, DynamicCache)
        # Track whether the cache was empty before this step (first generation step:
        # the full prompt is in input_ids, no KV cache yet).
        initial_cache_empty = is_dynamic_cache and len(pkv.key_cache) == 0
        if is_dynamic_cache and not initial_cache_empty:
            before_keys = [pkv.key_cache[i].clone() for i in range(len(pkv.key_cache))]
            before_values = [
                pkv.value_cache[i].clone() for i in range(len(pkv.value_cache))
            ]
        else:
            before_keys = before_values = None

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Clone the cache AFTER the main forward so we can restore the correct
        # post-forward state once all ablation passes are done.
        # Always capture after_keys — even when the cache was initially empty, the
        # main forward populated it and ablation passes must not corrupt it.
        if is_dynamic_cache and len(pkv.key_cache) > 0:
            after_keys = [pkv.key_cache[i].clone() for i in range(len(pkv.key_cache))]
            after_values = [
                pkv.value_cache[i].clone() for i in range(len(pkv.value_cache))
            ]
        else:
            after_keys = after_values = None

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
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

        # token selection
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        next_word = (
            global_tokenizer.decode(next_tokens[0])
            if global_tokenizer
            else str(next_tokens[0].item())
        )

        original_probs = F.softmax(next_token_logits.float(), dim=-1)
        original_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def attach_custom_hook(layer_idx, head_idx):
            def hook_fn(module, input):
                ablated = input[0].clone()
                ablated[:, :, head_dim * head_idx : head_dim * (head_idx + 1)] = 0
                return (ablated,)

            return hook_fn

        # Run zero-ablation when the decoded token is a substring of any entity
        word = next_word.strip()
        matched_hal = word and any(word in ent for ent in hallucinated_entities)
        matched_non_hal = word and any(word in ent for ent in non_hallucinated_entities)
        is_target = matched_hal or matched_non_hal

        if is_target:
            influences = [[None for _ in range(head_num)] for _ in range(layer_num)]

            # For the first token (initial_cache_empty), after_keys holds the full
            # prefill KV (positions 0..N-1).  Build a single-token decode context:
            # past_kv = positions 0..N-2, input = last prompt token.
            # This makes the first-token ablation identical in cost to all later
            # tokens (1 token forward pass instead of a full N-token prefill).
            # Build ablation_inputs for the first-token case: single-token decode
            # using pkv populated with positions 0..N-2 (truncated prefill).
            # This is the same structure as the before_keys path — pkv is modified
            # in place before each ablation pass, and ablation_inputs slices the
            # full-prompt model_inputs down to just the last token.
            if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                ablation_inputs = dict(model_inputs)
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
                    # Restore pkv to the pre-forward state before each ablation pass.
                    if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                        # Populate pkv with positions 0..N-2 from the prefill cache.
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

                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :].float()
                    ablated_probs = F.softmax(next_token_logits_ablated.float(), dim=-1)
                    ablated_log_probs = F.log_softmax(
                        next_token_logits_ablated.float(), dim=-1
                    )

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

            # Restore "after" cache so generation continues from the correct state.
            if after_keys is not None:
                for i in range(len(after_keys)):
                    pkv.key_cache[i] = after_keys[i]
                    pkv.value_cache[i] = after_values[i]

            torch.cuda.empty_cache()

        count += 1

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
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

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
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


def zero_ablation_sample(
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

    # init values from generation_config (mirrors _sample)
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

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
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

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    print(
        f"[_sample DEBUG] input_ids shape: {input_ids.shape}, first tokens: {input_ids[0, :5].tolist()}"
    )
    print(f"[_sample DEBUG] model_kwargs keys: {list(model_kwargs.keys())}")
    if "inputs_embeds" in model_kwargs:
        print(
            f"[_sample DEBUG] inputs_embeds shape: {model_kwargs['inputs_embeds'].shape}"
        )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    # Pop custom kwargs before they pollute prepare_inputs_for_generation
    hallucinated_entities = model_kwargs.pop("hallucinated_entities", [])
    non_hallucinated_entities = model_kwargs.pop("non_hallucinated_entities", [])
    inf_score_type = model_kwargs.pop("influence_score", "prob_diff")

    count = 0
    hallucination_influences = {}
    non_hallucination_influences = {}
    o_proj_modules = []

    # Collect o_proj modules from Qwen2 layers inside VideoLLaMA2
    for name, module in self.model.named_modules():
        if "o_proj" in name:
            o_proj_modules.append(module)

    layer_num = len(o_proj_modules)

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Clone the cache BEFORE the main forward so ablation passes can reset to
        # this exact sequence length.  DynamicCache.update() uses torch.cat (creates
        # a new tensor) so these clones won't be mutated by subsequent forwards.
        pkv = model_inputs.get("past_key_values")
        is_dynamic_cache = isinstance(pkv, DynamicCache)
        # Track whether the cache was empty before this step (first generation step:
        # the full prompt is in input_ids, no KV cache yet).
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

        # Clone the cache AFTER the main forward so we can restore the correct
        # post-forward state once all ablation passes are done.
        # Always capture after_keys — even when the cache was initially empty, the
        # main forward populated it and ablation passes must not corrupt it.
        if is_dynamic_cache and len(pkv.key_cache) > 0:
            after_keys = [pkv.key_cache[i].clone() for i in range(len(pkv.key_cache))]
            after_values = [
                pkv.value_cache[i].clone() for i in range(len(pkv.value_cache))
            ]
        else:
            after_keys = after_values = None

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].clone()
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores / attentions / hidden states
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

        # Token selection
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        next_word = (
            global_tokenizer.decode(next_tokens[0])
            if global_tokenizer
            else str(next_tokens[0].item())
        )

        original_probs = F.softmax(next_token_logits.float(), dim=-1)
        original_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def attach_custom_hook(layer_idx, head_idx):
            def hook_fn(module, input):
                ablated = input[0].clone()
                ablated[:, :, head_dim * head_idx : head_dim * (head_idx + 1)] = 0
                return (ablated,)

            return hook_fn

        # Run zero-ablation when the decoded token is a substring of any entity
        word = next_word.strip()
        matched_hal = word and any(word in ent for ent in hallucinated_entities)
        matched_non_hal = word and any(word in ent for ent in non_hallucinated_entities)
        is_target = matched_hal or matched_non_hal

        if is_target:
            influences = [[None for _ in range(head_num)] for _ in range(layer_num)]

            # For the first token (initial_cache_empty), after_keys holds the full
            # prefill KV (positions 0..N-1).  Build a single-token decode context:
            # past_kv = positions 0..N-2, input = last prompt token.
            # This makes the first-token ablation identical in cost to all later
            # tokens (1 token forward pass instead of a full N-token prefill).
            # Build ablation_inputs for the first-token case: single-token decode
            # using pkv populated with positions 0..N-2 (truncated prefill).
            # This is the same structure as the before_keys path — pkv is modified
            # in place before each ablation pass, and ablation_inputs slices the
            # full-prompt model_inputs down to just the last token.
            if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                ablation_inputs = dict(model_inputs)
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
                    # Restore pkv to the pre-forward state before each ablation pass.
                    if initial_cache_empty and is_dynamic_cache and after_keys is not None:
                        # Populate pkv with positions 0..N-2 from the prefill cache.
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

                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :].float()
                    ablated_probs = F.softmax(next_token_logits_ablated.float(), dim=-1)
                    ablated_log_probs = F.log_softmax(
                        next_token_logits_ablated.float(), dim=-1
                    )

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

            # Restore "after" cache so generation continues from the correct state.
            if after_keys is not None:
                for i in range(len(after_keys)):
                    pkv.key_cache[i] = after_keys[i]
                    pkv.value_cache[i] = after_values[i]

            torch.cuda.empty_cache()

        count += 1

        # Finish sequences that hit eos
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
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
        return input_ids


def set_zero_ablation_greedy_search(
    tokenizer, hallucinated_entities, non_hallucinated_entities, influence_score
):
    set_tokenizer(tokenizer)
    set_entity_list(hallucinated_entities, non_hallucinated_entities, influence_score)
    transformers.generation.utils.GenerationMixin._sample = _sample
