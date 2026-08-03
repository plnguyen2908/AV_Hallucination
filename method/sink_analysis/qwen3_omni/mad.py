"""mad.py — MAD (multimodal contrast decoding) for Qwen3-Omni.

Port of method/sink_analysis/qwen2_5_omni/_5_avspeaker_mad.py::mad_decode, SAME
standard config (gamma=2.5). No attention patching -> runs SDPA. For AVHBench
the 4 branches are built from the single mp4: av (use_aiv), video-only (use_aiv
False), audio (mp4 as audio dict), text-only. A routing head reads
p_audio/p_video/p_both from the modality-query prompt; branch weights:
  a_av = 2 + 2g*p_both ; a_v = 1-(p_both-p_video)g ; a_a = 1-(p_both-p_audio)g ;
  a_t = -(p_video+p_audio)g. combined logits = sum_k a_k * logits_k.
"""
import sys
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method/qwen3_omni"))
import utils  # noqa

MODALITY_QUERY = ("To answer this question, which modality is needed "
                  "(audio, video, or both): ")
SYS = utils.OMNI_SYSTEM_PROMPT
FPS, MAXPIX = 1.0, 360 * 640


def _vid(p):
    return {"type": "video", "video": p, "fps": FPS, "max_pixels": MAXPIX}


def _prep(processor, model, conv, use_aiv):
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    au, im, vi = process_mm_info(conv, use_audio_in_video=use_aiv)
    inp = processor(text=text, audio=au, images=im, videos=vi, return_tensors="pt",
                    padding=True, use_audio_in_video=use_aiv).to(model.device).to(model.dtype)
    return inp


def mad_answer_logits(model, processor, video_path, question, gamma=2.5):
    """Return combined next-token logits for the yes/no decision."""
    def sysmsg(): return {"role": "system", "content": [{"type": "text", "text": SYS}]}
    head = [sysmsg(), {"role": "user", "content": [_vid(video_path),
            {"type": "text", "text": "Question: " + question + "\n" + MODALITY_QUERY}]}]
    conv_av = [sysmsg(), {"role": "user", "content": [_vid(video_path), {"type": "text", "text": question}]}]
    conv_v = [sysmsg(), {"role": "user", "content": [_vid(video_path), {"type": "text", "text": question}]}]
    conv_a = [sysmsg(), {"role": "user", "content": [{"type": "audio", "audio": video_path}, {"type": "text", "text": question}]}]
    conv_t = [sysmsg(), {"role": "user", "content": [{"type": "text", "text": question}]}]

    # routing head
    hi = _prep(processor, model, head, True)
    with torch.inference_mode():
        ho = model.thinker(**hi, use_cache=False)
    hl = ho.logits[0, -1, :]
    tok = processor.tokenizer
    ai, vi_, bi = tok.encode("audio")[0], tok.encode("video")[0], tok.encode("both")[0]
    pa, pv, pb = torch.softmax(torch.tensor([hl[ai].item(), hl[vi_].item(), hl[bi].item()]), 0).tolist()
    w = {"av": 2 + 2 * gamma * pb, "v": 1 - (pb - pv) * gamma,
         "a": 1 - (pb - pa) * gamma, "t": -(pv + pa) * gamma}

    convs = {"av": (conv_av, True), "v": (conv_v, False), "a": (conv_a, True), "t": (conv_t, False)}
    combined = None
    for kkey, (conv, use_aiv) in convs.items():
        inp = _prep(processor, model, conv, use_aiv)
        with torch.inference_mode():
            out = model.thinker(**inp, use_cache=False)
        lg = out.logits[:, -1, :].squeeze(0).float()
        combined = w[kkey] * lg if combined is None else combined + w[kkey] * lg
    return combined, dict(p_audio=pa, p_video=pv, p_both=pb)
