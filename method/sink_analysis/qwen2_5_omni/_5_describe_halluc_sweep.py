"""_5_describe_halluc_sweep.py — does boosting sink attention reduce hallucinated tokens?

Re-generates the captions of an existing `sampled_entities.json` under the
`boost_inert_content` intervention at several γ, re-labels each caption with
eval.py's entity matcher, and reports the hallucination RATE.

Why rate and not count: the describe probes are closed-vocabulary label
listing (captions are 2-5 words, e.g. "Pop music, Music"). A shorter caption
lists fewer labels and therefore fewer hallucinated ones "for free", so raw
counts reward truncation. rate = hal / (hal + non_hal) is confound-free;
caption length + non-hal count are reported alongside as the truncation check.

γ is the attraction axis: the boost multiplies sink attention by (1+γ) and
row-renormalises, and renormalisation cancels in the sink/non-sink ratio, so
    attraction(γ) = (1 + γ) · attraction(0)     [exact on the prompt forward]

Pre-registered predictions (fixed before running):
  - VGGSounder (AV): rate DROPS, inverted-U with minimum at γ=3-5, caption
    length flat. Confirmed iff rate falls >=3 points with length within ±10%.
  - AudioSet (audio-only): rate RISES or is flat (consistent with MMAU -12.8).
  - Falsified if rate falls only alongside caption length (truncation), or
    falls monotonically with no optimum (degeneration, not mechanism).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 qwen_venv/bin/python \
    method/sink_analysis/qwen2_5_omni/_5_describe_halluc_sweep.py \
      --dataset VGGSounder_describe --modal_type av --gammas 0,1,3,5,8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))

import _5_explore as EX  # noqa: E402
import _5_intervene as IV  # noqa: E402
from utils import build_conversation, load_omni, trim_chat_artifacts  # noqa: E402

import eval as EVAL  # noqa: E402  (entity labelling: find_labels_in_text)

DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"
OUT_DIR = _REPO / "results/qwen2_5_omni/stage5_intervention/describe_sweep"

MODAL_TO_ROUTE = {"a": "AUDIO", "v": "VISUAL", "av": "AV"}
MEDIA_DIR = {
    "AudioSet_describe": "data/AudioSet/audios",
    "ActivityNet_describe": "data/ActivityNet/videos",
    "VGGSounder_describe": "data/VGGSounder/videos",
}


def label_caption(caption: str, entry: dict) -> dict:
    """Re-run eval.py's entity labelling against the SAME gt_entities."""
    gt = entry["gt_entities"]
    # candidate label vocabulary = every label eval.py considered for this entry
    vocab = sorted(set(entry.get("generated_entities", []))
                   | set(entry.get("hallucinated_entities", []))
                   | set(entry.get("non_hallucinated_entities", []))
                   | set(gt))
    found = EVAL.find_labels_in_text(caption, vocab)
    gt_lower = {g.lower() for g in gt}
    hal = [e for e in found if e.lower() not in gt_lower]
    non = [e for e in found if e.lower() in gt_lower]
    return dict(generated_entities=found, hallucinated_entities=hal,
                non_hallucinated_entities=non)


def is_degenerate(text: str) -> bool:
    toks = text.lower().split()
    if len(toks) < 8:
        return False
    return max(toks.count(t) for t in set(toks)) / len(toks) > 0.4


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(_REPO / f"results/qwen2_5_omni/{args.dataset}/sampled_entities.json"))
    if args.limit:
        entries = entries[: args.limit]
    media_dir = _REPO / MEDIA_DIR[args.dataset]
    route = MODAL_TO_ROUTE[args.modal_type]
    gammas = [float(g) for g in args.gammas.split(",")]

    model, processor = load_omni(args.model_path, attn_implementation="eager",
                                 device_map=args.device_map)
    n_patched = IV.patch_qwen_attention(model)
    print(f"[patch] {n_patched} attention modules patched", flush=True)

    head_sets = EX.load_head_sets(DEFAULT_HEADS)
    hs = head_sets["Audio head"] if args.head_set == "halluc" else head_sets["Inert"]
    l2h = EX.heads_by_layer(hs, "full")
    print(f"[heads] {args.mode} {len(hs)} {args.head_set} heads across {len(l2h)} layers", flush=True)

    # Build intervention levels. boost: gamma>0 boosts sink attn by (1+gamma).
    # suppress: reduce sink attraction by factor N (gamma=1/N); N=1 -> baseline
    # (no intervention); "mask" -> gamma=0 (sink attention fully removed).
    if args.mode == "suppress":
        levels = []
        for f in args.factors.split(","):
            f = f.strip()
            if f == "mask":
                levels.append(("mask", 0.0, True))
            elif float(f) == 1.0:
                levels.append(("1x_base", 1.0, False))
            else:
                levels.append((f"1/{f}", 1.0 / float(f), True))
    else:
        levels = [(str(g), float(g), float(g) > 0) for g in gammas]

    layers = EX.utils.thinker_layers(model) if hasattr(EX, "utils") else model.thinker.model.layers
    audio_enc, visual_enc = EX._resolve_encoders(model)
    eps_norm = EX._thinker_rms_eps(model)
    d_sink_t = torch.tensor(EX.D_SINK, dtype=torch.long)

    rows = []
    t0 = time.time()
    for label, gamma, intervene in levels:
        for ei, entry in enumerate(entries):
            media = media_dir / entry["video"]
            if not media.exists():
                continue
            conv = build_conversation(str(media), entry["question"], args.modal_type)
            use_aiv = args.modal_type == "av"
            try:
                masks, S = EX.compute_per_layer_sink_masks(
                    model, processor, conv, use_aiv, layers, audio_enc, visual_enc,
                    eps_norm, d_sink_t, route)
                if intervene:
                    IV.set_intervention(dict(layer_to_heads=l2h, layer_to_key_mask=masks,
                                             mode=args.mode, gamma=gamma))
                else:
                    IV.clear_intervention()
                out = EX.generate_with_intervention(model, processor, conv, use_aiv,
                                                    max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"  [skip] {label} {entry['video']}: {type(e).__name__}: {e}", flush=True)
                IV.clear_intervention()
                continue
            finally:
                IV.clear_intervention()
            cap = trim_chat_artifacts(out)
            lab = label_caption(cap, entry)
            rows.append(dict(level=label, gamma=gamma, video=entry["video"], caption=cap,
                             n_words=len(cap.split()), degenerate=int(is_degenerate(cap)),
                             n_hal=len(lab["hallucinated_entities"]),
                             n_non=len(lab["non_hallucinated_entities"])))
            if (ei + 1) % 25 == 0:
                el = (time.time() - t0) / 60
                print(f"  {label} {ei+1}/{len(entries)}  ({el:.1f} min)", flush=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.tag}.csv", index=False)
        print(f"[level {label} done] {(time.time()-t0)/60:.1f} min", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{args.tag}.csv", index=False)
    print(f"\n=== {args.dataset} ({args.modal_type}) {args.mode} {args.head_set} — n={len(entries)} clips")
    print(f"{'level':>9}{'rate':>8}{'hal/cap':>9}{'nonhal/cap':>12}{'len(w)':>8}{'degen%':>8}")
    for label, _, _ in levels:
        s = df[df.level == label]
        if not len(s):
            continue
        tot = s.n_hal.sum() + s.n_non.sum()
        rate = s.n_hal.sum() / tot if tot else float("nan")
        print(f"{label:>9}{rate:8.3f}{s.n_hal.mean():9.2f}{s.n_non.mean():12.2f}"
              f"{s.n_words.mean():8.1f}{s.degenerate.mean()*100:8.1f}")
    print(f"\nsaved: {OUT_DIR / (args.tag + '.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(MEDIA_DIR))
    p.add_argument("--modal_type", required=True, choices=["a", "v", "av"])
    p.add_argument("--gammas", default="0,1,3,5,8")
    p.add_argument("--head_set", default="inert", choices=["inert", "halluc"])
    p.add_argument("--mode", default="boost", choices=["boost", "suppress"])
    p.add_argument("--factors", default="1,2,4,6,9,mask",
                   help="suppress mode: reduce attraction by these factors; mask=full")
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device_map", default="balanced_low_0")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    a.tag = a.tag or f"{a.dataset}_sweep"
    main(a)
