"""run_avhbench.py — Qwen3-Omni sink-boost tuning on AVHBench DEV.

Boosts a head set's attention to ATTENTION-SINK key positions (measured per
clip; Qwen3's hidden-state D_sink=1992 is uniform and NOT a sink-token
selector, so we use attention-received sinks instead), then sweeps
(g_base, tau) with a per-layer gamma schedule. Baseline (no intervention) is
always measured for reference.

Per clip: ONE measurement forward (accumulates attention-received per key via
the patched attention, no full attention matrices) -> per-layer sink masks
(recv > tau) -> for each config: set_intervention(boost, gamma=schedule*g_base)
-> greedy generate 8 tokens -> parse Yes/No -> score. Configs share the single
measurement, so the sweep is cheap.

Usage:
  qwen3_venv/bin/python method/sink_analysis/qwen3_omni/run_avhbench.py \
     --split DEV --g_base 1,2,3 --tau 0.01,0.02,0.05 --schedule flat
"""
import argparse, re, sys, time
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


def sink_masks_from_recv(recv, tau, S):
    """recv: {layer: (K,) mean attention received}. Return {layer: bool (S,)}."""
    out = {}
    for L, r in recv.items():
        m = torch.zeros(S, dtype=torch.bool)
        k = min(len(r), S)
        m[:k] = torch.from_numpy(r[:k] > tau)
        if m.any():
            out[L] = m
    return out


def gen_answer(model, processor, inputs, use_aiv):
    with torch.inference_mode():
        out = model.generate(**inputs, use_audio_in_video=use_aiv, return_audio=False,
                             do_sample=False, thinker_max_new_tokens=8)
    seq = utils._extract_sequences(out)
    gen = seq[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main(a):
    g_bases = [float(x) for x in a.g_base.split(",")]
    taus = [float(x) for x in a.tau.split(",")]
    df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(5)
    rows = df[df.split == a.split] if a.split != "FULL" else df
    if a.limit:
        rows = rows.iloc[: a.limit]
    print(f"AVHBench {a.split}: {len(rows)} questions | schedule={a.schedule} "
          f"g_base={g_bases} tau={taus}", flush=True)

    model, processor = utils.load_omni(MODEL)
    IV.patch(model)
    IV.clear_intervention()
    L = utils.thinker_text_config(model).num_hidden_layers
    l2h, n_inert = load_inert_heads(a.heads_csv)
    print(f"boosting {n_inert} inert heads across {len(l2h)} layers", flush=True)
    schedule = np.ones(L)  # flat; forward/reverse shapes plugged in if provided

    configs = [("baseline", None, None)]
    for g in g_bases:
        for t in taus:
            configs.append((f"g{g}_tau{t}", g, t))
    correct = {c[0]: 0 for c in configs}
    total = {c[0]: 0 for c in configs}
    bytask = {c[0]: {} for c in configs}

    t0 = time.time()
    for i, (_, r) in enumerate(rows.iterrows()):
        vp = VIDEO_DIR / f"{r.video_id}.mp4"
        if not vp.exists():
            continue
        prompt = r.text + YES_NO_SUFFIX
        conv = utils.build_conversation(str(vp), prompt, "av")
        try:
            inputs, use_aiv = utils.prepare_inputs(processor, conv, "av",
                                                   model.device, model.dtype)
        except Exception as e:
            print(f"  skip {r.video_id}: {e}", flush=True); continue
        S = inputs["input_ids"].shape[1]

        # measurement forward (attention-received per key). inputs already
        # encode the AV features; thinker.forward doesn't take use_audio_in_video.
        IV.start_measure()
        try:
            with torch.inference_mode():
                model.thinker(**inputs, return_dict=True, use_cache=False)
        except Exception as e:
            IV.stop_measure()
            print(f"  measure err {r.video_id}: {e}", flush=True); continue
        IV.stop_measure()
        recv = IV.get_measure()

        for name, g, t in configs:
            if name == "baseline":
                IV.clear_intervention()
            else:
                masks = sink_masks_from_recv(recv, t, S)
                if i == 0:  # one-time diagnostic: confirm sinks + boost fire
                    nsink = {L: int(m.sum()) for L, m in list(masks.items())[:4]}
                    IV.reset_debug()
                IV.set_intervention(dict(layer_to_heads=l2h, layer_to_key_mask=masks,
                                         mode="boost", gamma=g,
                                         gamma_schedule=(schedule * g).tolist()))
            try:
                out = gen_answer(model, processor, inputs, use_aiv)
            except Exception as e:
                print(f"  gen err {r.video_id} {name}: {e}", flush=True)
                IV.clear_intervention(); continue
            IV.clear_intervention()
            if i == 0 and name != "baseline":
                dbg = IV.get_debug()
                print(f"  [diag {name}] masked_layers={len(masks)} "
                      f"first_layer_sinks={nsink} boost_calls_modified={dbg['modified']}",
                      flush=True)
            pred = parse_yes_no(out)
            ok = int(pred == r.label)
            correct[name] += ok; total[name] += 1
            bt = bytask[name].setdefault(r.task, [0, 0]); bt[0] += ok; bt[1] += 1
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            b = correct["baseline"] / max(total["baseline"], 1)
            print(f"  [{i+1}/{len(rows)}] {el:.0f}s baseline={b:.3f}", flush=True)

    print("\n=== RESULTS (AVHBench %s, %d inert heads) ===" % (a.split, n_inert))
    base = correct["baseline"] / max(total["baseline"], 1)
    for name, _, _ in configs:
        acc = correct[name] / max(total[name], 1)
        d = "" if name == "baseline" else f"  (dbase {100*(acc-base):+.2f})"
        print(f"  {name:16s}: {acc:.4f}  n={total[name]}{d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="DEV")
    p.add_argument("--g_base", default="1,2,3")
    p.add_argument("--tau", default="0.01,0.02,0.05")
    p.add_argument("--schedule", default="flat", choices=["flat", "forward", "reverse"])
    p.add_argument("--heads_csv",
                   default=str(_REPO / "results/qwen3_omni/categorize_exp_4axis/heads.csv"))
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
