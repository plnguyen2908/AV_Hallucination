Traceback (most recent call last):
  File "/nobackup/le/AV_Hallucination/method/videollama2/identify_halluc_head.py", line 167, in <module>
    main(args)
  File "/nobackup/le/AV_Hallucination/method/videollama2/identify_halluc_head.py", line 100, in main
    outputs = model.generate(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/VideoLLaMA2/videollama2/model/videollama2_qwen2.py", line 135, in generate
    return super().generate(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/generation/utils.py", line 1914, in generate
    result = self._sample(
  File "/nobackup/le/AV_Hallucination/method/videollama2/head_attribution.py", line 76, in zero_ablation_sample
    outputs = self(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/accelerate/hooks.py", line 165, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/VideoLLaMA2/videollama2/model/videollama2_qwen2.py", line 93, in forward
    return super().forward(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 1221, in forward
    outputs = self.model(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 1023, in forward
    layer_outputs = decoder_layer(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/accelerate/hooks.py", line 165, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 763, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/accelerate/hooks.py", line 165, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 639, in forward
    return super().forward(
  File "/nobackup/le/AV_Hallucination/videollama2_venv/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 295, in forward
    attn_weights = attn_weights + causal_mask
RuntimeError: The size of tensor a (3003) must match the size of tensor b (2219) at non-singleton dimension 3