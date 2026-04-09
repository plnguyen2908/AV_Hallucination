import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import List, Optional, Union

import transformers
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput

# Global variable to hold the tokenizer
global_tokenizer = None

def set_tokenizer(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer

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
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    count = 0
    hallucination_influences = {}
    non_hallucination_influences = {}
    o_proj_modules = []

    # Collect o_proj modules from Qwen2 layers inside VideoLLaMA2
    for name, module in self.model.named_modules():
        if 'o_proj' in name:
            o_proj_modules.append(module)

    layer_num = len(o_proj_modules)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

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
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
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

        next_word = global_tokenizer.decode(next_tokens[0]) if global_tokenizer else str(next_tokens[0].item())

        original_probs = F.softmax(next_token_logits, dim=-1)
        original_log_probs = F.log_softmax(next_token_logits, dim=-1)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        def attach_custom_hook(layer_idx, head_idx):
            def hook_fn(module, input):
                ablated = input[0].clone()
                ablated[:, :, head_dim * head_idx: head_dim * (head_idx + 1)] = 0
                return (ablated,)
            return hook_fn

        # Run zero-ablation only for target tokens
        hallucinated_tokens = model_kwargs.get('hallucinated_tokens', [])
        non_hallucinated_tokens = model_kwargs.get('non_hallucinated_tokens', [])
        target_tokens = hallucinated_tokens + non_hallucinated_tokens
        is_target = next_word in target_tokens or next_tokens[0].item() in target_tokens

        if is_target:
            # DynamicCache is mutable — each ablation forward appends to it in-place.
            # Save the current sequence lengths and restore before every ablation pass.
            pkv = model_inputs.get('past_key_values')
            if isinstance(pkv, DynamicCache) and len(pkv.key_cache) > 0:
                orig_lengths = [pkv.key_cache[i].shape[2] for i in range(len(pkv.key_cache))]
            else:
                orig_lengths = None

            influences = [[None for _ in range(head_num)] for _ in range(layer_num)]
            for layer_idx in range(layer_num):
                o_proj_module = o_proj_modules[layer_idx]
                for head_idx in range(head_num):
                    if orig_lengths is not None:
                        for i, L in enumerate(orig_lengths):
                            pkv.key_cache[i] = pkv.key_cache[i][:, :, :L, :]
                            pkv.value_cache[i] = pkv.value_cache[i][:, :, :L, :]

                    hook_handle = o_proj_module.register_forward_pre_hook(
                        attach_custom_hook(layer_idx, head_idx)
                    )
                    outputs_ablated = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    hook_handle.remove()

                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :]
                    ablated_probs = F.softmax(next_token_logits_ablated, dim=-1)
                    ablated_log_probs = F.log_softmax(next_token_logits_ablated, dim=-1)

                    inf_score_type = model_kwargs.get('influence_score', 'prob_diff')
                    if inf_score_type == 'prob_diff':
                        influence = (original_probs[0, next_tokens[0]] - ablated_probs[0, next_tokens[0]]).item()
                    elif inf_score_type == 'abs_prob_diff':
                        influence = (original_probs[0, next_tokens[0]] - ablated_probs[0, next_tokens[0]]).abs().item()
                    elif inf_score_type == 'log_prob_diff':
                        influence = (original_log_probs[0, next_tokens[0]] - ablated_log_probs[0, next_tokens[0]]).item()
                    else:
                        influence = (original_probs[0, next_tokens[0]] - ablated_probs[0, next_tokens[0]]).item()

                    influences[layer_idx][head_idx] = {
                        'original_prob':      original_probs[0, next_tokens[0]].item(),
                        'perturbed_prob':     ablated_probs[0, next_tokens[0]].item(),
                        'original_log_prob':  original_log_probs[0, next_tokens[0]].item(),
                        'perturbed_log_prob': ablated_log_probs[0, next_tokens[0]].item(),
                        'influence':          influence,
                    }

            key = f'{next_word}_{count}'
            if next_word in hallucinated_tokens or next_tokens[0].item() in hallucinated_tokens:
                hallucination_influences[key] = influences
            elif next_word in non_hallucinated_tokens or next_tokens[0].item() in non_hallucinated_tokens:
                non_hallucination_influences[key] = influences

            # Final restore after all ablation passes
            if orig_lengths is not None:
                for i, L in enumerate(orig_lengths):
                    pkv.key_cache[i] = pkv.key_cache[i][:, :, :L, :]
                    pkv.value_cache[i] = pkv.value_cache[i][:, :, :L, :]

            torch.cuda.empty_cache()

        count += 1

        # Finish sequences that hit eos
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
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
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            ), hallucination_influences, non_hallucination_influences
    else:
        return input_ids


def set_zero_ablation_greedy_search(tokenizer):
    set_tokenizer(tokenizer)
    transformers.generation.utils.GenerationMixin._sample = zero_ablation_sample
