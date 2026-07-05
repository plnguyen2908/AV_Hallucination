"""stage2_1b_sink_layer_proportion.py

For each proxy dataset, the PER-LAYER PROPORTION of LLM-emerged sinks
(distribution of sinks over the 28 decoder layers; sums to 100% per dataset).
LLM-emerged sink = token with max_d|RMSNorm(x)[d]| >= 20 over D_sink={458,2570}
(same criterion as Stage 2.1). Counted on the dataset's modality tokens:

  audio       : AudioSet, LibriSpeech     -> audio-token sinks
  visual      : ActivityNet, YouTube-VOS  -> video-token sinks
  audiovisual : VGGSounder                -> audio+video token sinks

Plot: one line per dataset (colored by modality), x=layer, y=% of that dataset's
sinks at that layer. Similar to Stage 2.1's figure but normalized to a per-layer
proportion.
"""
import argparse, os, random, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import build_conversation, load_omni, prepare_inputs, thinker_layers
OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_1b_sink_layer_proportion"
OUT.mkdir(parents=True, exist_ok=True)
D_SINK = [458, 2570]; TAU_SINK = 20.0
PROMPT = {"a": "Describe what you hear in detail.",
          "v": "Describe what you see in detail.",
          "av": "Describe what you see and hear in detail."}

# (name, media glob, modal_type, target modality, plot color)
DATASETS = [
    ("AudioSet",    "data/AudioSet/audios/*.wav",            "a",  "audio",       "#1f77b4"),
    ("LibriSpeech", "data/LibriSpeech/test-other/**/*.flac", "a",  "audio",       "#4ba3e3"),
    ("ActivityNet", "data/ActivityNet/videos/*.mp4",         "v",  "visual",      "#2ca02c"),
    ("YouTubeVOS",  "data/YouTubeVOS/videos/*.mp4",          "v",  "visual",      "#7fcf7f"),
    ("VGGSounder",  "data/VGGSounder/videos/*.mp4",          "av", "audiovisual", "#d62728"),
]


def modal_positions(input_ids, cfg):
    ids = input_ids[0].cpu().numpy()
    a = int(getattr(cfg, "audio_token_index", 151646))
    v = int(getattr(cfg, "video_token_index", 151656))
    return np.where(ids == a)[0], np.where(ids == v)[0]


def process_clip(model, processor, path, modal_type, target, layers, eps, d_sink_t, cfg):
    conv = build_conversation(str(path), PROMPT[modal_type], modal_type)
    try:
        inputs, use_aiv = prepare_inputs(processor, conv, modal_type, model.device, model.dtype)
    except Exception:
        return None
    apos, vpos = modal_positions(inputs["input_ids"], cfg)
    pos = {"audio": apos, "visual": vpos, "audiovisual": np.concatenate([apos, vpos])}[target]
    if len(pos) == 0:
        return None
    nL = len(layers); counts = np.zeros(nL, dtype=np.int64)
    pos_t = torch.from_numpy(pos.astype(np.int64))

    def hook(L):
        def _h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            x = hs[0].float()
            dev = x.device
            d = d_sink_t.to(dev); p = pos_t.to(dev)
            p = p[(p >= 0) & (p < x.shape[0])]   # guard: positions in hidden-seq range
            if p.numel() == 0:
                return
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            sink_act = x.index_select(1, d).div(rms).abs().amax(-1)   # (seq,)
            mask = sink_act >= TAU_SINK
            counts[L] = int(mask.index_select(0, p).sum().item())
        return _h

    handles = [layers[L].register_forward_hook(hook(L)) for L in range(nL)]
    try:
        with torch.inference_mode():
            model.thinker(**inputs, output_hidden_states=False,
                          use_audio_in_video=use_aiv, return_dict=True, use_cache=False)
    except Exception:
        counts = None
    finally:
        for h in handles:
            h.remove()
    torch.cuda.empty_cache()
    return counts


def main(args):
    random.seed(42)
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    cfg = model.thinker.config.get_text_config() if hasattr(model.thinker.config, "get_text_config") else model.thinker.config
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    layers = thinker_layers(model); nL = len(layers)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    print(f"n_layers={nL}, eps={eps}, D_sink={D_SINK}, tau={TAU_SINK}", flush=True)

    results = {}  # name -> (per_layer_total_counts, n_clips, color, modality)
    for name, glob, mt, target, color in DATASETS:
        files = sorted((_REPO).glob(glob))
        random.shuffle(files); files = files[: args.n_clips]
        total = np.zeros(nL, dtype=np.int64); ok = 0
        for f in tqdm(files, desc=name):
            c = process_clip(model, processor, f, mt, target, layers, eps, d_sink_t, cfg)
            if c is not None:
                total += c; ok += 1
        results[name] = (total, ok, color, target)
        prop = total / total.sum() if total.sum() else total
        print(f"{name}: {ok} clips, total sinks={total.sum()}, "
              f"peak layer={int(prop.argmax())} ({prop.max()*100:.1f}%)", flush=True)
        np.savez(OUT / "sink_layer_counts.npz",
                 **{n: r[0] for n, r in results.items()},
                 **{f"{n}_nclips": r[1] for n, r in results.items()})

    # Plot per-layer proportion
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (total, ok, color, target) in results.items():
        prop = total / total.sum() * 100 if total.sum() else total
        ax.plot(range(nL), prop, marker="o", ms=4, color=color,
                label=f"{name} ({target}, n={ok})")
    ax.set_xlabel("decoder layer", fontsize=12)
    ax.set_ylabel("% of dataset's LLM-emerged sinks at this layer", fontsize=12)
    ax.set_title("Per-layer distribution of LLM-emerged sinks by proxy dataset\n"
                 f"(D_sink={{458,2570}}, tau=20; each curve sums to 100%)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xticks(range(0, nL, 2))
    fig.tight_layout(); fig.savefig(OUT / "sink_layer_proportion.png", dpi=300)
    print(f"wrote {OUT/'sink_layer_proportion.png'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--n_clips", type=int, default=300)
    main(p.parse_args())
