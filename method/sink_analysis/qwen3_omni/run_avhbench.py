"""run_avhbench.py — Qwen3-Omni sink-boost, per-modality gamma schedule +
soft router, tuned on AVHBench DEV.

Faithful port of the Qwen2.5 method:
  - SINK TOKENS: D_SINK hidden-state rule. Qwen3 D_SINK=[1992]; the dim is
    uniformly high (mean~28), so tau is CALIBRATED HIGHER (~30-40) than
    Qwen2.5's 15. A token is a sink at layer L iff |RMSNorm(h_L)[1992]| >= tau.
  - PER-MODALITY SCHEDULE SHAPES: shape_a / shape_v = per-layer D_SINK sink
    density on the audio proxies (AudioSet+LibriSpeech) / visual proxies
    (ActivityNet+YouTubeVOS), normalized to mean 1 (forward schedule:
    gamma ~ sink density).
  - SOFT ROUTER: per clip, p_a / p_v = share of attention mass landing on the
    audio-span / visual-span key positions (model-derived, from the same
    measurement forward). No learned router needed.
  - gamma_schedule[L] = g_base * (shape_a[L]*p_a + shape_v[L]*p_v).
  Boosts the 1059 global-inert heads' attention to sink tokens. Sweeps
  (g_base, tau). Baseline always measured.

The measurement forward uses output_hidden_states (for D_SINK masks; (L,S,H) is
cheap) + the intervene.py attention hook (for the router; no output_attentions
so no S*S OOM on long AV prompts).
"""
import argparse, glob, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "method/qwen3_omni"))
import utils  # noqa: E402
sys.path.insert(0, str(_HERE))
import intervene as IV  # noqa: E402

MODEL = "/nobackup2/zyu362/hf_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
SPLIT_CSV = _REPO / "results/qwen2_5_omni/stage5_intervention/split.csv"
VIDEO_DIR = _REPO / "data/AVHBench/videos"
YES_NO_SUFFIX = " Answer with only 'Yes' or 'No'."
D_SINK = [1992]
EPS = 1e-6


def parse_yes_no(text):
    m = re.search(r"\b(yes|no)\b", text.lower())
    return {"yes": "Yes", "no": "No"}.get(m.group(1)) if m else "Unk"


def load_inert_heads(heads_csv):
    d = pd.read_csv(heads_csv)
    inert = d[d.category == "Global inert"][["layer", "head"]].values.tolist()
    l2h = {}
    for l, h in inert:
        l2h.setdefault(int(l), []).append(int(h))
    return l2h, len(inert)


def dsink_magnitude(hidden_states):
    """Per (layer, token) |RMSNorm(h)[D_SINK]| (max over sink dims). Returns
    list over layers of (S,) numpy. hidden_states: tuple len L+1 of (1,S,H)."""
    out = []
    for hs in hidden_states:
        h = hs[0].float()
        rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        v = (h / rms)[:, D_SINK].abs().amax(dim=-1)  # (S,)
        out.append(v.cpu().numpy())
    return out


def dsink_masks(mag, tau, S):
    """{decoder_layer_idx: bool (S,)} where |RMSNorm[1992]| >= tau. mag is
    indexed by hidden-state index (0=embedding); map decoder layer L -> mag[L+1]."""
    out = {}
    for L in range(len(mag) - 1):
        m = torch.zeros(S, dtype=torch.bool)
        v = mag[L + 1]
        k = min(len(v), S)
        m[:k] = torch.from_numpy(v[:k] >= tau)
        if m.any():
            out[L] = m
    return out


# Constrained-logit text router (verbatim from _5_phaseB_prep.py): classify the
# QUESTION text into Audio/Visual/AV, read soft probs from the label logits.
ROUTER_PROMPT_TPL_V2 = (
    "You are classifying questions by the modality they ask about. "
    "Audio if the question is about sounds, hearing, or what is audible. "
    "Visual if the question is about images, objects, or what is visible. "
    "AV if answering requires comparing or matching what is heard against "
    "what is seen (cross-modal consistency, joint description). "
    "Respond with exactly one word: Audio, Visual, or AV.\n\n"
    "Question: {q}\n\nClassification:"
)
ROUTER_LABELS = [" Audio", " Visual", " AV"]


def resolve_router_label_ids(processor):
    tok = processor.tokenizer
    conv = [{"role": "system", "content": [{"type": "text", "text": utils.OMNI_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text",
             "text": ROUTER_PROMPT_TPL_V2.format(q="Does a dog bark?")}]}]
    chat = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(chat, list):
        chat = chat[0]
    ids = []
    for lab in ROUTER_LABELS:
        full = tok(chat + lab, add_special_tokens=False).input_ids
        base = tok(chat, add_special_tokens=False).input_ids
        assert len(full) - len(base) == 1, f"{lab!r} not single-token (+{len(full)-len(base)})"
        ids.append(full[-1])
    return ids


def route(model, processor, question, label_ids):
    """3-way soft router (p_a, p_v, p_av) from constrained label logits on the
    text-only routing prompt for this question."""
    conv = [{"role": "system", "content": [{"type": "text", "text": utils.OMNI_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text",
             "text": ROUTER_PROMPT_TPL_V2.format(q=question)}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device).to(model.dtype)
    with torch.inference_mode():
        out = model.thinker(**inputs, return_dict=True, use_cache=False)
    sel = out.logits[0, -1, :].float()[label_ids]
    p = torch.softmax(sel, dim=-1).cpu().numpy()
    return float(p[0]), float(p[1]), float(p[2])


def measure_forward(model, processor, conv, modal, want_hidden=True):
    """One prompt-only thinker forward. Returns (inputs, mag_or_None, recv)."""
    inputs, _ = utils.prepare_inputs(processor, conv, modal, model.device, model.dtype)
    IV.start_measure()
    with torch.inference_mode():
        out = model.thinker(**inputs, return_dict=True, use_cache=False,
                            output_hidden_states=want_hidden)
    IV.stop_measure()
    recv = IV.get_measure()
    mag = dsink_magnitude(out.hidden_states) if want_hidden else None
    return inputs, mag, recv


def modality_shape(model, processor, media, modal, tau, n, L):
    """Per-layer D_SINK sink density over n proxy clips of one modality, turned
    into the REVERSE schedule shape: rev[L] = 1 - density[L]/max(density),
    normalized to mean 1 (peaks at LOW-density / early layers)."""
    dens = np.zeros(L)
    got = 0
    for m in media[:n]:
        conv = utils.build_conversation(str(m), "Describe what you perceive.", modal)
        try:
            _, mag, _ = measure_forward(model, processor, conv, modal)
        except Exception:
            continue
        for Ld in range(L):
            dens[Ld] += float((mag[Ld + 1] >= tau).mean())  # frac tokens = sink
        got += 1
        torch.cuda.empty_cache()
    dens = dens / max(got, 1)
    rev = 1.0 - dens / max(dens.max(), 1e-9)          # REVERSE: peaks where sparse
    shape = rev / max(rev.mean(), 1e-9)
    return shape, got, dens


def gen_answer(model, processor, inputs, use_aiv):
    with torch.inference_mode():
        out = model.generate(**inputs, use_audio_in_video=use_aiv, return_audio=False,
                             do_sample=False, thinker_max_new_tokens=8)
    seq = utils._extract_sequences(out)
    return processor.batch_decode(seq[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def main(a):
    g_bases = [float(x) for x in a.g_base.split(",")]
    taus = [float(x) for x in a.tau.split(",")]
    df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(5)
    rows = df[df.split == a.split] if a.split != "FULL" else df
    if a.limit:
        rows = rows.iloc[: a.limit]
    print(f"AVHBench {a.split}: {len(rows)} Q | g_base={g_bases} tau={taus} "
          f"(D_SINK={D_SINK})", flush=True)

    model, processor = utils.load_omni(MODEL)
    IV.patch(model); IV.clear_intervention()
    tcfg = utils.thinker_text_config(model)
    thinker_cfg = model.thinker.config  # has audio/vision start/end token ids
    L = tcfg.num_hidden_layers
    l2h, n_inert = load_inert_heads(a.heads_csv)
    print(f"boosting {n_inert} inert heads across {len(l2h)} layers", flush=True)
    router_label_ids = resolve_router_label_ids(processor)
    print(f"router label ids {dict(zip(ROUTER_LABELS, router_label_ids))}", flush=True)

    # ---- per-modality schedule shapes from proxy sink density ----
    audio_media = (sorted(glob.glob(str(_REPO / "data/AudioSet/audios/*.wav")))[: a.warmup]
                   + sorted(glob.glob(str(_REPO / "data/LibriSpeech/test-other/**/*.flac"),
                                      recursive=True))[: a.warmup])
    visual_media = (sorted(glob.glob(str(_REPO / "data/ActivityNet/videos/*.mp4")))[: a.warmup]
                    + sorted(glob.glob(str(_REPO / "data/YouTubeVOS/videos/*.mp4")))[: a.warmup])
    av_media = sorted(glob.glob(str(_REPO / "data/VGGSounder/videos/*.mp4")))[: 2 * a.warmup]
    shape_a, na, da = modality_shape(model, processor, audio_media, "a", a.tau_shape, 2 * a.warmup, L)
    shape_v, nv, dv = modality_shape(model, processor, visual_media, "v", a.tau_shape, 2 * a.warmup, L)
    shape_av, nav, dav = modality_shape(model, processor, av_media, "av", a.tau_shape, 2 * a.warmup, L)
    print(f"[shapes REVERSE] audio n={na} shape_a[0,24,47]={shape_a[[0,24,47]].round(2)}", flush=True)
    print(f"[shapes REVERSE] visual n={nv} shape_v[0,24,47]={shape_v[[0,24,47]].round(2)}", flush=True)
    print(f"[shapes REVERSE] av n={nav} shape_av[0,24,47]={shape_av[[0,24,47]].round(2)}", flush=True)

    configs = [] if a.skip_baseline else [("baseline", None, None)]
    for g in g_bases:
        for t in taus:
            configs.append((f"g{g}_tau{t}", g, t))
    correct = {c[0]: 0 for c in configs}
    total = {c[0]: 0 for c in configs}
    bytask = {c[0]: {} for c in configs}
    router_log = []

    t0 = time.time()
    for i, (_, r) in enumerate(rows.iterrows()):
        vp = VIDEO_DIR / f"{r.video_id}.mp4"
        if not vp.exists():
            continue
        conv = utils.build_conversation(str(vp), r.text + YES_NO_SUFFIX, "av")
        try:
            inputs, mag, recv = measure_forward(model, processor, conv, "av")
        except Exception as e:
            print(f"  measure err {r.video_id}: {e}", flush=True); continue
        use_aiv = True
        S = inputs["input_ids"].shape[1]
        p_a, p_v, p_av = route(model, processor, r.text, router_label_ids)
        router_log.append((r.task, p_a, p_v, p_av))

        for name, g, t in configs:
            if name == "baseline":
                IV.clear_intervention()
            else:
                masks = dsink_masks(mag, t, S)
                sched = (g * (shape_a * p_a + shape_v * p_v + shape_av * p_av)).tolist()
                if i == 0:
                    nsink = {Lx: int(m.sum()) for Lx, m in list(masks.items())[:4]}
                    print(f"  [diag {name}] p_a={p_a:.2f} p_v={p_v:.2f} p_av={p_av:.2f} "
                          f"masked_layers={len(masks)} first_sinks={nsink} "
                          f"sched[0,24,47]={[round(sched[j],2) for j in [0,24,47]]}", flush=True)
                IV.set_intervention(dict(layer_to_heads=l2h, layer_to_key_mask=masks,
                                         mode="boost", gamma=g, gamma_schedule=sched))
            try:
                out = gen_answer(model, processor, inputs, use_aiv)
            except Exception as e:
                print(f"  gen err {r.video_id} {name}: {e}", flush=True)
                IV.clear_intervention(); continue
            IV.clear_intervention()
            ok = int(parse_yes_no(out) == r.label)
            correct[name] += ok; total[name] += 1
            bt = bytask[name].setdefault(r.task, [0, 0]); bt[0] += ok; bt[1] += 1
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            trk = configs[0][0]  # baseline if present, else first config
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s "
                  f"{trk}={correct[trk]/max(total[trk],1):.3f}", flush=True)

    # router sanity by task
    rl = pd.DataFrame(router_log, columns=["task", "p_a", "p_v", "p_av"])
    print("\n[router] mean p_a/p_v/p_av by task:")
    for tk, grp in rl.groupby("task"):
        print(f"   {tk:38s}: p_a={grp.p_a.mean():.2f} p_v={grp.p_v.mean():.2f} p_av={grp.p_av.mean():.2f}")

    print("\n=== RESULTS (AVHBench %s, %d inert heads, D_SINK schedule+router) ===" %
          (a.split, n_inert))
    base = (correct["baseline"] / max(total["baseline"], 1)) if "baseline" in correct else None
    for name, *_ in configs:
        acc = correct[name] / max(total[name], 1)
        d = "" if (name == "baseline" or base is None) else f"  (dbase {100*(acc-base):+.2f})"
        tk = " | ".join(f"{t[:12]}:{c[0]/max(c[1],1):.2f}" for t, c in sorted(bytask[name].items()))
        print(f"  {name:16s}: {acc:.4f}  n={total[name]}{d}   [{tk}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="DEV")
    p.add_argument("--g_base", default="2,3,4")
    p.add_argument("--tau", default="30,34,38")
    p.add_argument("--tau_shape", type=float, default=34.0)
    p.add_argument("--warmup", type=int, default=12, help="proxy clips per dataset for shapes")
    p.add_argument("--heads_csv",
                   default=str(_REPO / "results/qwen3_omni/categorize_exp_4axis/heads.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_baseline", action="store_true",
                   help="Don't run the (config-independent) baseline config; "
                        "reuse a previously measured baseline for the delta.")
    main(p.parse_args())
