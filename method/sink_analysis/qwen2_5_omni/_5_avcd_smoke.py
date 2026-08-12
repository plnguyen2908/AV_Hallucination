"""Smoke test: AVCD-for-Qwen on one short WorldSense clip vs baseline."""
import subprocess, sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni")); sys.path.insert(0, str(_HERE))
from utils import load_omni, build_conversation, prepare_inputs  # noqa
from _5_efficient_encoders import patch_efficient_encoders  # noqa
from _5_avcd_qwen import patch_thinker_avcd, set_spans, avcd_answer_logits  # noqa

VID = "zwkbEZwG"
SRC = _REPO / "data/WorldSense/videos" / f"{VID}.mp4"
TMP = Path("/tmp/claude-10526/-nobackup2-le-AV-Hallucination/3c6c1667-33a0-48f1-9699-6431bef97b63/scratchpad/wsprobe")
PROMPT = ("Carefully watch this video and pay attention to every detail. Based on your "
          "observations, select the best option that accurately addresses the question.\n"
          "These are the frames of a video and the corresponding audio. Select the best "
          "answer to the following multiple-choice question based on the video. Respond "
          "with only the letter (A, B, C, or D) of the correct option.\n"
          "Question: What instrument is being played in the video?\n"
          "A. Guitar\nB. Piano\nC. Guzheng\nD. Drums\nAnswer: ")


def trim(dur):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{VID}_{dur}s.mp4"
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-ss", "0", "-t", str(dur), "-i", str(SRC),
                        "-c", "copy", str(out)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def main():
    model, processor = load_omni("Qwen/Qwen2.5-Omni-7B", device_map="balanced_low_0")
    patch_thinker_avcd(model)
    patch_efficient_encoders(model)
    cfg = model.thinker.config
    vid_id, aud_id = cfg.video_token_id, cfg.audio_token_id
    print("video_token_id", vid_id, "audio_token_id", aud_id, flush=True)

    conv = build_conversation(str(trim(15)), PROMPT, modal_type="av",
                              video_fps=1.0, video_max_pixels=360 * 640)
    inputs, use_aiv = prepare_inputs(processor, conv, "av", model.device, model.dtype)
    ids = inputs["input_ids"][0]
    S = ids.shape[0]
    vidx = torch.where(ids == vid_id)[0].tolist()
    aidx = torch.where(ids == aud_id)[0].tolist()
    print(f"seq_len={S} video_tok={len(vidx)} audio_tok={len(aidx)} "
          f"lang_tok={S - len(vidx) - len(aidx)}", flush=True)
    set_spans(vidx, aidx, S, model.device)

    logits, info = avcd_answer_logits(model, inputs, use_aiv, cd_alpha=2.5)
    tok = int(logits.argmax(dim=-1).item())
    ans = processor.tokenizer.decode([tok]).strip()
    print(f"[AVCD] dominant={info['dominant']} gated={info['gated']} "
          f"thr={info['threshold']:.2e} -> answer token '{ans}'", flush=True)

    # baseline greedy (thinker eager, no masking): re-run full forward argmax
    from _5_avcd_qwen import reset_collect, clear_avcd
    with torch.inference_mode():
        out = model.thinker(**inputs, use_audio_in_video=use_aiv, use_cache=False)
    clear_avcd()
    btok = int(out.logits[0, -1, :].argmax().item())
    bans = processor.tokenizer.decode([btok]).strip()
    print(f"[BASE] answer token '{bans}'  (correct=C)", flush=True)


if __name__ == "__main__":
    main()
