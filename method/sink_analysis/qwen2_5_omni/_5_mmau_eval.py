"""MMAU (test-mini, audio-only) eval harness for Qwen2.5-Omni.

MMAU = audio understanding/reasoning MCQ (1000 test-mini: sound/music/speech,
answer = lettered option text e.g. '(A) Man'). Official scoring = token
string_match (evaluation.py, copied verbatim below). Audio clips ~10s -> no
length-ceiling concern.

NOTE: MMAU is AUDIO-ONLY. The AV interventions largely degenerate here:
- ASD cross-modal sinks need video -> none -> no-op (== baseline).
- MAD/AVCD video contexts vanish. So the meaningful comparison is baseline vs
  ours (audio sink-boost, routed modality = AUDIO).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  qwen_venv/bin/python method/sink_analysis/qwen2_5_omni/_5_mmau_eval.py --method ours
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
sys.path.insert(0, str(_HERE))
from utils import load_omni, build_conversation, prepare_inputs  # noqa

ROWS = _REPO / "data/MMAU/rows.json"
OUT_DIR = _REPO / "results/qwen2_5_omni/mmau"

PROMPT_SUFFIX = ("\nChoose the single best option. Respond with the option's letter "
                 "and text exactly as written, e.g. '(A) ...'.")


# ---- official MMAU scoring (evaluation.py string_match, verbatim) ----------
def string_match(answer, prediction, choices):
    def tokenize(text):
        return set(re.findall(r"\b\w+\b", text.lower()))
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def prompt_for(row):
    return row["question"] + "\n" + "\n".join(row["choices"]) + PROMPT_SUFFIX


# ---- backends -------------------------------------------------------------
class Backend:
    needs_eager = False

    def load(self, args):
        impl = "eager" if self.needs_eager else "sdpa"
        self.model, self.processor = load_omni(args.model_path, attn_implementation=impl,
                                               device_map=args.device_map)
        self.args = args

    def conv(self, row):
        return build_conversation(row["audio"], prompt_for(row), modal_type="a")

    def _gen(self, inputs):
        with torch.inference_mode():
            ids = self.model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                                      do_sample=False, max_new_tokens=16)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


class BaselineBackend(Backend):
    def answer(self, row):
        inputs, _ = prepare_inputs(self.processor, self.conv(row), "a",
                                   self.model.device, self.model.dtype)
        return self._gen(inputs)


class OursBackend(Backend):
    """Audio sink-boost = ours with routed modality = AUDIO (no video on MMAU):
    boost_inert_content, 508 inert heads, per-clip audio sink masks (sink_mask=all),
    reverse gamma schedule = g_base * shape_a. Eager thinker."""
    needs_eager = True

    def load(self, args):
        super().load(args)
        import numpy as np
        import _5_explore as E
        import _5_intervene as IV
        self.E, self.IV = E, IV
        IV.patch_qwen_attention(self.model)
        E._RUNTIME_ARGS = type("A", (), dict(sink_mask=args.sink_mask, asd=False,
                                             random_mask=False, include_text_sinks=False))()
        self.layers = E.thinker_layers(self.model)
        self.eps = E._thinker_rms_eps(self.model)
        self.aenc, self.venc = E._resolve_encoders(self.model)
        self.d_sink = torch.tensor(E.D_SINK, dtype=torch.long)
        self.head_sets = E.load_head_sets(Path(args.heads_csv))
        self.sched_a = np.load(args.gamma_schedule_npz)["shape_a"]
        self.g_base = args.g_base
        print(f"[ours-audio] Inert={len(self.head_sets['Inert'])} g_base={self.g_base}", flush=True)

    def answer(self, row):
        conv = self.conv(row)
        l2k, S = self.E.compute_per_layer_sink_masks(
            self.model, self.processor, conv, False, self.layers,
            self.aenc, self.venc, self.eps, self.d_sink, "AUDIO")
        heads = self.E.heads_for_variant_routing(self.head_sets, "boost_inert_content", "AUDIO")
        l2h = self.E.heads_by_layer(heads, "full")
        cfg = dict(layer_to_heads=l2h, layer_to_key_mask=l2k, mode="boost", gamma=3.0)
        cfg["gamma_schedule"] = (self.g_base * self.sched_a).tolist()
        self.IV.set_intervention(cfg)
        try:
            inputs, _ = prepare_inputs(self.processor, conv, "a", self.model.device, self.model.dtype)
            txt = self._gen(inputs)
        finally:
            self.IV.clear_intervention()
        return txt


BACKENDS = {"baseline": BaselineBackend, "ours": OursBackend}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(BACKENDS))
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--device_map", default="balanced_low_0")
    ap.add_argument("--heads_csv", default=str(_REPO / "results/qwen2_5_omni/categorize_exp_2axis_common508/heads.csv"))
    ap.add_argument("--gamma_schedule_npz", default=str(_REPO / "results/qwen2_5_omni/sink_analysis/gamma_schedules/rev_sched_common508.npz"))
    ap.add_argument("--g_base", type=float, default=5.0)
    ap.add_argument("--sink_mask", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    rows = json.load(open(ROWS))
    if args.limit:
        rows = rows[:args.limit]
    print(f"MMAU test-mini: {len(rows)} questions; method={args.method}", flush=True)
    backend = BACKENDS[args.method](); backend.load(args)
    tag = args.tag or f"{args.method}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recs = []; fails = 0; t0 = time.time()
    for row in tqdm(rows, desc=tag):
        try:
            pred = backend.answer(row)
        except RuntimeError as e:
            fails += 1; pred = f"__ERR__{str(e)[:40]}"; torch.cuda.empty_cache()
        correct = int(string_match(row["answer"], pred, row["choices"]))
        recs.append({**{k: row[k] for k in ("id", "task", "category", "sub_category", "difficulty", "answer")},
                     "prediction": pred, "correct": correct})

    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / f"mmau_{tag}.csv", index=False)
    overall = df["correct"].mean()
    by_task = df.groupby("task")["correct"].mean().to_dict()
    by_diff = df.groupby("difficulty")["correct"].mean().to_dict()
    rating = dict(overall=round(float(overall), 4), n=len(df), fails=fails,
                  by_task={k: round(float(v), 4) for k, v in by_task.items()},
                  by_difficulty={k: round(float(v), 4) for k, v in by_diff.items()})
    json.dump(rating, open(OUT_DIR / f"mmau_{tag}_rating.json", "w"), indent=2)
    print(f"\n=== MMAU {tag} ===")
    print(f"overall: {overall:.4f}  n={len(df)} fails={fails}  {(time.time()-t0)/60:.1f} min")
    print("by task:", rating["by_task"])
    print("by difficulty:", rating["by_difficulty"])


if __name__ == "__main__":
    main()
