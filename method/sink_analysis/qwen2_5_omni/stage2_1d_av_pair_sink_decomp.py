"""stage2_1d_av_pair_sink_decomp.py

For each PAIR (one audio proxy + one visual proxy) fed as a synthetic AV input,
the per-layer proportion of three sink types (ASD classification):
  audio sink (uni-audio), visual sink (uni-video), audiovisual sink (cross-modal).
Each of the 3 curves is normalized to sum to 100% across the 28 layers.

Sink = LLM-emerged (max_d|RMSNorm(x)[d]|>=20 over D_sink={458,2570}). Each sink
classified via Stage 3.1's MDS: mds_i=(a_v-a_a)/(a_v+a_a); per-layer span-median
thresholds -> uni-video / uni-audio / cross-modal (= ASD audiovisual sink).
"""
import argparse, random, sys, warnings
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_REPO / "method/sink_analysis/qwen2_5_omni"))
import utils
from utils import load_omni, thinker_layers, OMNI_SYSTEM_PROMPT
from qwen_omni_utils import process_mm_info
from stage3_1_modality_detection_score import per_clip_mds, _modal_positions, D_SINK, TAU_SINK

OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_1d_av_pair_decomp"
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "Describe what you see and hear in detail."
FPS, MAXPIX = 1.0, 230400

AUDIO = {"AudioSet": "data/AudioSet/audios/*.wav",
         "LibriSpeech": "data/LibriSpeech/test-other/**/*.flac"}
VISUAL = {"ActivityNet": "data/ActivityNet/videos/*.mp4",
          "YouTubeVOS": "data/YouTubeVOS/videos/*.mp4"}


def build_inputs(model, processor, apath, vpath):
    conv = [{"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": str(apath)},
                {"type": "video", "video": str(vpath), "fps": FPS, "max_pixels": MAXPIX},
                {"type": "text", "text": PROMPT}]}]
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    return inputs.to(model.device).to(model.dtype)


def process(model, processor, apath, vpath, layers, eps, d_sink_t, cfg):
    try:
        inputs = build_inputs(model, processor, apath, vpath)
    except Exception:
        return None
    apos, vpos = _modal_positions(inputs["input_ids"], cfg)
    if len(apos) == 0 or len(vpos) == 0:
        return None
    S = int(inputs["input_ids"].shape[1]); nL = len(layers)
    aqm = torch.zeros(S, dtype=torch.bool); aqm[apos] = True
    vqm = torch.zeros(S, dtype=torch.bool); vqm[vpos] = True
    attn_a = np.zeros((nL, S)); attn_v = np.zeros((nL, S))
    h_pre = [None] * nL

    def attn_hook(L):
        def _h(_m, _i, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            ha = out[1][0].float().mean(0)            # (q, kv) head-avg
            attn_v[L] = ha[vqm.to(ha.device)].mean(0).cpu().numpy()
            attn_a[L] = ha[aqm.to(ha.device)].mean(0).cpu().numpy()
            return (out[0], None) + tuple(out[2:])
        return _h

    def pre_hook(L):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                h_pre[L] = hs[0].detach()
        return _h

    handles = []
    for L in range(nL):
        handles.append(layers[L].register_forward_pre_hook(pre_hook(L)))
        handles.append(layers[L].self_attn.register_forward_hook(attn_hook(L)))
    try:
        with torch.inference_mode():
            model.thinker(**inputs, use_audio_in_video=False,
                          output_attentions=True, return_dict=True, use_cache=False)
    except Exception:
        for h in handles: h.remove()
        torch.cuda.empty_cache(); return None
    finally:
        for h in handles: h.remove()
    if any(h is None for h in h_pre):
        torch.cuda.empty_cache(); return None

    p_llm = np.zeros((nL, S), dtype=bool)
    for L in range(nL):
        h = h_pre[L].float()
        rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        dt = d_sink_t.to(h.device)
        p_llm[L] = ((h / rms).abs().index_select(1, dt).amax(-1) >= TAU_SINK).cpu().numpy()
    torch.cuda.empty_cache()

    r = per_clip_mds(attn_v, attn_a, p_llm, vpos, apos)
    # tuple: (mds_v, mds_a, n_v, n_a, n_uv_all, n_ua_all, n_cross_all, n_total, ...)
    return dict(n_ua=r[5], n_uv=r[4], n_cross=r[6])   # audio / visual / audiovisual


def main(args):
    random.seed(42)
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    cfg = model.thinker.config.get_text_config() if hasattr(model.thinker.config, "get_text_config") else model.thinker.config
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    layers = thinker_layers(model); nL = len(layers)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)

    pairs = [(a, v) for a in AUDIO for v in VISUAL]
    results = {}
    for aname, vname in pairs:
        afiles = sorted(_REPO.glob(AUDIO[aname])); random.shuffle(afiles)
        vfiles = sorted(_REPO.glob(VISUAL[vname])); random.shuffle(vfiles)
        n = min(args.n_clips, len(afiles), len(vfiles))
        tot_ua = np.zeros(nL); tot_uv = np.zeros(nL); tot_cr = np.zeros(nL); ok = 0
        for ap, vp in tqdm(list(zip(afiles[:n], vfiles[:n])), desc=f"{aname}x{vname}"):
            r = process(model, processor, ap, vp, layers, eps, d_sink_t, cfg)
            if r is not None:
                tot_ua += r["n_ua"]; tot_uv += r["n_uv"]; tot_cr += r["n_cross"]; ok += 1
        results[f"{aname}x{vname}"] = (tot_ua, tot_uv, tot_cr, ok)
        print(f"{aname}x{vname}: {ok} clips  "
              f"sinks: audio={tot_ua.sum():.0f} visual={tot_uv.sum():.0f} "
              f"audiovisual={tot_cr.sum():.0f}", flush=True)
        np.savez(OUT / "av_pair_counts.npz",
                 **{f"{k}_ua": v[0] for k, v in results.items()},
                 **{f"{k}_uv": v[1] for k, v in results.items()},
                 **{f"{k}_cr": v[2] for k, v in results.items()})

    # Plot: 2x2 panels, one per pair, 3 lines each (normalized to 100% over layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    def nz(x): return x / x.sum() * 100 if x.sum() else x
    for i, (pair, (ua, uv, cr, ok)) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(range(nL), nz(ua), "-o", ms=3, color="#1f77b4", label="audio (uni-audio)")
        ax.plot(range(nL), nz(uv), "-o", ms=3, color="#2ca02c", label="visual (uni-video)")
        ax.plot(range(nL), nz(cr), "-o", ms=3, color="#d62728", label="audiovisual (cross-modal/ASD)")
        ax.set_title(f"{pair}  (n={ok})", fontsize=11)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if i >= 2: ax.set_xlabel("decoder layer")
        if i % 2 == 0: ax.set_ylabel("% of that sink type at this layer")
    fig.suptitle("Per-layer sink-type distribution by synthetic AV proxy pair "
                 "(audio / visual / audiovisual; each curve sums to 100%)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "av_pair_sink_decomp.png", dpi=300)
    print(f"wrote {OUT/'av_pair_sink_decomp.png'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--n_clips", type=int, default=100)
    main(p.parse_args())
