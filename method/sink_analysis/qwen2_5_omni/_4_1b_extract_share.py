"""
_4_1b_extract_share.py — Stage 4.1.b extraction (per dataset).

For each clip in the chosen dataset, teacher-force the previously-
generated caption appended to the prompt and run a single
`model.thinker` forward with hooks to compute, per (layer, head):

    num_audio = Σ_{q ∈ gen, k ∈ audio_sinks_L} A[L,h,q,k]
    num_video = Σ_{q ∈ gen, k ∈ video_sinks_L} A[L,h,q,k]
    den_audio = Σ_{q ∈ gen, k ∈ audio_pos}    A[L,h,q,k]
    den_video = Σ_{q ∈ gen, k ∈ video_pos}    A[L,h,q,k]

where  audio_sinks_L = (p_llm[L] & ~p_prop) & audio_pos_mask
       video_sinks_L = (p_llm[L] & ~p_prop) & video_pos_mask
       gen           = generated-text positions (after teacher-forced prompt)

p_llm[L] and p_prop are computed in-process (same conventions as
stage3_1: D_SINK={458,2570}, τ_sink=20, τ_prop=100). Pre-SA hidden
states drive p_llm; per-encoder L2 norms drive p_prop. Hooks reduce
in-place — no full attention matrix retained.

Output (`<output_dir>/share_per_head.csv`):
    clip, layer, head, n_gen, n_audio, n_video,
        n_audio_sink_L, n_video_sink_L,
        num_audio, num_video, den_audio, den_video

For AudioSet (audio-only): video columns are 0/empty.
For ActivityNet (video-only): audio columns are 0/empty.
For VGGSounder (av): both populated, use_audio_in_video=True.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))

from utils import build_conversation, load_omni, thinker_layers  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

# Sink constants — same as stage3_1
D_SINK = [458, 2570]
TAU_SINK = 20.0
TAU_PROP = 100.0

DATASETS = {
    "AudioSet": dict(
        modal_type="a", use_aiv=False,
        media_dir=_REPO / "data/AudioSet/audios",
        entries_json=_REPO / "results/qwen2_5_omni/AudioSet_describe/sampled_entities.json",
        task="AudioSet Captioning",
    ),
    "ActivityNet": dict(
        modal_type="v", use_aiv=False,
        media_dir=_REPO / "data/ActivityNet/videos",
        entries_json=_REPO / "results/qwen2_5_omni/ActivityNet_describe/sampled_entities.json",
        task="ActivityNet Captioning",
    ),
    "VGGSounder": dict(
        modal_type="av", use_aiv=True,
        media_dir=_REPO / "data/VGGSounder/videos",
        entries_json=_REPO / "results/qwen2_5_omni/VGGSounder_describe/sampled_entities.json",
        task="VGGSounder Captioning",
    ),
}


def _resolve_thinker_cfg(model):
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_token_index"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _thinker_rms_eps(model) -> float:
    cfg = getattr(model.thinker.config, "text_config", model.thinker.config)
    return float(getattr(cfg, "rms_norm_eps", 1e-6))


def _resolve_encoders(model):
    thinker = model.thinker
    audio_mod = visual_mod = None
    for attr in ("audio_tower", "audio_encoder"):
        if hasattr(thinker, attr):
            audio_mod = getattr(thinker, attr); break
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(thinker, attr):
            visual_mod = getattr(thinker, attr); break
    return audio_mod, visual_mod


def _extract_tokens(out):
    x = out
    if isinstance(x, (tuple, list)): x = x[0]
    if hasattr(x, "last_hidden_state"): x = x.last_hidden_state
    if x.dim() == 3: x = x[0]
    return x


def align_norms(enc_norms, n_llm):
    n_enc = len(enc_norms)
    if n_enc == n_llm: return enc_norms
    if n_enc > n_llm and n_enc % n_llm == 0:
        k = n_enc // n_llm
        return enc_norms.reshape(n_llm, k).mean(axis=1)
    if n_llm > n_enc and n_llm % n_enc == 0:
        k = n_llm // n_enc
        return np.repeat(enc_norms, k)
    return None


def _modal_positions(input_ids, thinker_cfg):
    ids = input_ids[0].cpu().numpy()
    a_id = int(getattr(thinker_cfg, "audio_token_index", 151646))
    v_id = int(getattr(thinker_cfg, "video_token_index", 151656))
    return (np.where(ids == a_id)[0].astype(np.int64),
            np.where(ids == v_id)[0].astype(np.int64))


def _apply_describe_suffix(prompt: str, task: str) -> str:
    SUFFIXES = {
        "AudioSet Captioning": (
            "\nRespond with ONLY a comma-separated list of labels from the list "
            "above that match the sounds you hear. No explanations, no other words."
        ),
        "ActivityNet Captioning": (
            "\nRespond with ONLY a comma-separated list of labels from the list "
            "above that match what you see. No explanations, no other words."
        ),
        "VGGSounder Captioning": (
            "\nRespond with ONLY a comma-separated list of labels from the list "
            "above that match what you see and hear. No explanations, no other words."
        ),
    }
    suffix = SUFFIXES.get(task)
    if suffix and suffix not in prompt:
        return prompt + suffix
    return prompt


def process_clip(entry, dataset_cfg, model, processor, thinker_cfg, layers,
                   audio_enc, visual_enc, eps_norm, d_sink_t):
    """Returns list of per-(layer, head) dicts or None on failure.

    Two forward passes per clip:
      Pass 1 (prompt only): pre-SA hooks for per-layer hidden state
        → p_llm; encoder hooks → p_prop. NO output_attentions.
      Pass 2 (prompt + teacher-forced caption): self_attn forward hooks
        capture per (layer, head) bin scalars over generated-text query
        positions. NO hidden-state hooks needed.

    Splitting avoids the cross-device hazard in mixing per-layer
    hidden-state computation with attention extraction in one forward.
    """
    media_path = dataset_cfg["media_dir"] / entry["video"]
    if not media_path.exists():
        return None, "missing_media"
    prompt = _apply_describe_suffix(entry["question"], entry["task"])
    caption = entry.get("generated_caption", "").strip()
    if not caption:
        return None, "empty_caption"
    modal_type = dataset_cfg["modal_type"]
    use_aiv = dataset_cfg["use_aiv"]

    conv = build_conversation(str(media_path), prompt, modal_type)
    try:
        audios, images, videos = process_mm_info(conv, use_audio_in_video=use_aiv)
    except Exception as e:
        return None, f"mm_info:{type(e).__name__}"

    prompt_text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False)
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]

    try:
        prompt_only = processor(
            text=prompt_text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=use_aiv,
        )
        inputs_full = processor(
            text=prompt_text + caption, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=use_aiv,
        )
    except Exception as e:
        return None, f"prep:{type(e).__name__}"

    gen_start = int(prompt_only["input_ids"].shape[1])
    S = int(inputs_full["input_ids"].shape[1])
    n_gen = S - gen_start
    if n_gen <= 0:
        return None, f"no_gen_tokens(S={S}, gen_start={gen_start})"

    prompt_only = prompt_only.to(model.device).to(model.dtype)
    inputs_full = inputs_full.to(model.device).to(model.dtype)
    audio_pos, video_pos = _modal_positions(inputs_full["input_ids"], thinker_cfg)
    n_audio = len(audio_pos); n_video = len(video_pos)

    n_layers = len(layers)

    # -------- Pass 1: prompt only — capture p_llm + p_prop. --------
    per_layer_h = [None] * n_layers
    a_enc_buf, v_enc_buf = [], []

    def make_layer_pre_hook(L_idx):
        def _h(_m, inp):
            hs = inp[0] if isinstance(inp, (tuple, list)) else inp
            if hs.shape[1] > 1:
                per_layer_h[L_idx] = hs[0].detach().cpu()  # CPU to dodge multi-device
        return _h

    def make_enc_hook(buf):
        def _h(_m, _i, out):
            tok = _extract_tokens(out)
            buf.append(tok.detach().norm(dim=-1).float().cpu().numpy())
        return _h

    handles = []
    if audio_enc is not None:
        handles.append(audio_enc.register_forward_hook(make_enc_hook(a_enc_buf)))
    if visual_enc is not None:
        handles.append(visual_enc.register_forward_hook(make_enc_hook(v_enc_buf)))
    for L in range(n_layers):
        handles.append(layers[L].register_forward_pre_hook(make_layer_pre_hook(L)))

    try:
        with torch.inference_mode():
            model.thinker(**prompt_only, use_audio_in_video=use_aiv,
                          output_attentions=False, return_dict=True,
                          use_cache=False)
    except Exception as e:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        return None, f"fwd1:{type(e).__name__}:{e}"
    finally:
        for h in handles: h.remove()

    if any(h is None for h in per_layer_h):
        torch.cuda.empty_cache()
        return None, "no_hidden_state"

    S_prompt = int(prompt_only["input_ids"].shape[1])
    audio_pos_p, video_pos_p = _modal_positions(prompt_only["input_ids"], thinker_cfg)

    # p_llm at prompt positions per layer (S_prompt,)
    p_llm_prompt = np.zeros((n_layers, S_prompt), dtype=bool)
    for L in range(n_layers):
        h_L = per_layer_h[L].float()
        rms = torch.sqrt(h_L.pow(2).mean(dim=-1, keepdim=True) + eps_norm)
        normed_abs = (h_L / rms).abs()
        p_llm_prompt[L] = (normed_abs[:, d_sink_t].amax(dim=-1) >= TAU_SINK).cpu().numpy()

    # p_prop on prompt positions
    p_prop_prompt = np.zeros(S_prompt, dtype=bool)
    if a_enc_buf and len(audio_pos_p) > 0:
        a_aligned = align_norms(np.concatenate(a_enc_buf), len(audio_pos_p))
        if a_aligned is not None:
            p_prop_prompt[audio_pos_p] = a_aligned > TAU_PROP
    if v_enc_buf and len(video_pos_p) > 0:
        v_aligned = align_norms(np.concatenate(v_enc_buf), len(video_pos_p))
        if v_aligned is not None:
            p_prop_prompt[video_pos_p] = v_aligned > TAU_PROP

    # Embed p_llm/p_prop/audio_mask/video_mask into the FULL sequence (S).
    # Modality positions only exist within the prompt segment (the
    # generated text is plain text tokens). Sink masks for positions
    # in the gen region are False — they're query positions, not keys
    # of interest.
    p_llm = np.zeros((n_layers, S), dtype=bool)
    p_llm[:, :S_prompt] = p_llm_prompt
    p_prop = np.zeros(S, dtype=bool)
    p_prop[:S_prompt] = p_prop_prompt

    audio_mask_cpu = torch.zeros(S, dtype=torch.bool)
    if n_audio > 0:
        audio_mask_cpu[audio_pos] = True
    video_mask_cpu = torch.zeros(S, dtype=torch.bool)
    if n_video > 0:
        video_mask_cpu[video_pos] = True

    # Free pass-1 buffers
    per_layer_h = None
    torch.cuda.empty_cache()

    # -------- Pass 2: full prompt+caption — attention hooks. --------
    records = []
    p_prop_t = torch.from_numpy(p_prop)
    p_llm_t = torch.from_numpy(p_llm)        # (n_layers, S) bool

    def make_attn_hook(L_idx):
        def _h(_m, _i, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return out
            aw = out[1]                              # (1, H, q, k)
            dev = aw.device
            a_gen = aw[0, :, gen_start:S, :].float()  # (H, n_gen, k)
            H = a_gen.shape[0]
            llm_emerged_L = (p_llm_t[L_idx].to(dev) & ~p_prop_t.to(dev))  # (S,)
            am = audio_mask_cpu.to(dev)
            vm = video_mask_cpu.to(dev)
            audio_sinks_L = llm_emerged_L & am
            video_sinks_L = llm_emerged_L & vm
            n_aud_sink_L = int(audio_sinks_L.sum().item())
            n_vid_sink_L = int(video_sinks_L.sum().item())
            a_qsum = a_gen.sum(dim=1)              # (H, k)
            num_audio = (a_qsum[:, audio_sinks_L].sum(dim=1).cpu().numpy()
                         if n_aud_sink_L > 0 else np.zeros(H, dtype=np.float32))
            num_video = (a_qsum[:, video_sinks_L].sum(dim=1).cpu().numpy()
                         if n_vid_sink_L > 0 else np.zeros(H, dtype=np.float32))
            den_audio = (a_qsum[:, am].sum(dim=1).cpu().numpy()
                         if n_audio > 0 else np.zeros(H, dtype=np.float32))
            den_video = (a_qsum[:, vm].sum(dim=1).cpu().numpy()
                         if n_video > 0 else np.zeros(H, dtype=np.float32))
            for h_idx in range(H):
                records.append(dict(
                    layer=L_idx, head=h_idx,
                    n_audio_sink_L=n_aud_sink_L,
                    n_video_sink_L=n_vid_sink_L,
                    num_audio=float(num_audio[h_idx]),
                    num_video=float(num_video[h_idx]),
                    den_audio=float(den_audio[h_idx]),
                    den_video=float(den_video[h_idx]),
                ))
            return (out[0], None) + tuple(out[2:])
        return _h

    handles2 = [layers[L].self_attn.register_forward_hook(make_attn_hook(L))
                 for L in range(n_layers)]
    try:
        with torch.inference_mode():
            model.thinker(**inputs_full, use_audio_in_video=use_aiv,
                          output_attentions=True, return_dict=True,
                          use_cache=False)
    except Exception as e:
        for h in handles2: h.remove()
        torch.cuda.empty_cache()
        return None, f"fwd2:{type(e).__name__}:{e}"
    finally:
        for h in handles2: h.remove()

    torch.cuda.empty_cache()
    if not records:
        return None, "no_records"
    base = dict(clip=entry["video"], n_gen=n_gen,
                n_audio=n_audio, n_video=n_video)
    rows = [{**base, **r} for r in records]
    return rows, None


def main(args):
    dsname = args.dataset
    if dsname not in DATASETS:
        raise SystemExit(f"--dataset must be one of {list(DATASETS)}")
    cfg = DATASETS[dsname]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sampled_entities from {cfg['entries_json']}")
    entries = json.load(open(cfg["entries_json"]))
    # Filter to entries with non-empty caption
    entries = [e for e in entries if e.get("generated_caption", "").strip()]
    print(f"  total entries: {len(entries)}")
    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(len(entries))[:args.n_clips]
    chosen = [entries[i] for i in pick]
    print(f"  selected n_clips = {len(chosen)} (seed={args.seed})")

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path, device_map=args.device_map)
    thinker_cfg = _resolve_thinker_cfg(model)
    layers = thinker_layers(model)
    n_layers = len(layers)
    eps_norm = _thinker_rms_eps(model)
    d_sink_t = torch.tensor(D_SINK, dtype=torch.long)
    audio_enc, visual_enc = _resolve_encoders(model)
    print(f"  n_layers={n_layers}, D_sink={D_SINK}, τ_sink={TAU_SINK}, "
          f"τ_prop={TAU_PROP}")

    rows = []
    failures: dict = {}
    for entry in tqdm(chosen, desc=f"clips({dsname})"):
        recs, err = process_clip(entry, cfg, model, processor, thinker_cfg,
                                   layers, audio_enc, visual_enc, eps_norm,
                                   d_sink_t)
        if recs is None:
            failures[err] = failures.get(err, 0) + 1
            tqdm.write(f"  [skip] {entry['video']}: {err}")
            continue
        rows.extend(recs)

    if failures:
        print(f"  failures: {failures}")
    if not rows:
        raise SystemExit("no records collected")

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"share_per_head_{dsname}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows, "
          f"{df['clip'].nunique()} unique clips)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--n_clips", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir",
                   default=str(_REPO /
                                 "results/qwen2_5_omni/sink_analysis/stage4_1b"))
    args = p.parse_args()
    main(args)
