import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse

def inference(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    # Audio-visual Inference
    audio_video_path = "/nobackup/le/AV_Hallucination/data/AVHBench/videos/00000.mp4"
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    if args.modal_type == "a":
        audio_video_tensor = preprocess(audio_video_path)
    else:
        audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    question = f"Is the man making sound in the audio?"

    # # Audio Inference
    # audio_video_path = "assets/bird-twitter-car.wav"
    # preprocess = processor['audio' if args.modal_type == "a" else "video"]
    # if args.modal_type == "a":
    #     audio_video_tensor = preprocess(audio_video_path)
    # else:
    #     audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    # question = f"Please describe the audio."

    # # Video Inference
    # audio_video_path = "assets/output_v_1jgsRbGzCls.mp4"
    # preprocess = processor['audio' if args.modal_type == "a" else "video"]
    # if args.modal_type == "a":
    #     audio_video_tensor = preprocess(audio_video_path)
    # else:
    #     audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    # question = f"What activity are the people practicing in the video?"

    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if args.modal_type == "a" else "video",
        do_sample=False,
    )

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=False, default='DAMO-NLP-SG/VideoLLaMA2.1-7B-AV')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', required=True)
    args = parser.parse_args()

    inference(args)
