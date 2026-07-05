"""
stage3_3_make_examples.py — assemble top_tokens_examples.md from
per_clip_lens/*.npz dumps.

For L13 and L20: pull a handful of per-clip cases per cell (cross /
uni-audio / uni-video), show top-5 decoded lens tokens (pre-SA primary,
post-block in a side note), the clip's GT label, and which token (if
any) is on-topic. Also look for the ASD Fig.4 misinterpreted-object
analogue: a uni-audio sink that decodes to a clearly content-bearing
token that does NOT match the GT.
"""
import argparse
from pathlib import Path
import numpy as np
import re
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_3/per_clip_lens"
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_3/top_tokens_examples.md"

CELL_NAME = {0: "cross", 1: "uni_video", -1: "uni_audio"}

_FUNC = {"the","a","an","of","to","in","on","and","or","but","is","are","was","were","be","been","being","have","has","had","for","with","as","at","by","this","that","it","its","from","into","if","then","than","so","such","i","you","we","they","he","she","my","your","our","their","his","her","us","them","do","does","did","will","would","could","should","can","may","might","no","not","yes"}
_ALPHA = re.compile(r"[A-Za-z]{3,}")


def is_real_word(d: str) -> bool:
    s = d.strip().lower()
    return bool(_ALPHA.search(s)) and s not in _FUNC


def is_on_topic(d: str, words: set) -> bool:
    s = d.strip().lower()
    if len(s) < 3: return False
    return any(w in s or s in w for w in words)


def gather_examples(dump_dir: Path, tokenizer, layers=(13, 20)):
    per_layer_blocks = {L: {"cross": [], "uni_audio": [], "uni_video": [],
                              "misinterp": []} for L in layers}
    for f in sorted(dump_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        clip = str(z["clip"])
        words = set(z["label_words"].tolist()) if "label_words" in z.files else set()
        for L in layers:
            key = f"L{L}"
            if f"{key}_topk_pre" not in z.files:
                continue
            topk = z[f"{key}_topk_pre"]
            topkp = z[f"{key}_topk_pre_prob"]
            cell = z[f"{key}_cell"]
            in_prop = z[f"{key}_in_prop"]
            in_llm  = z[f"{key}_in_llm"]
            pos = z[f"{key}_positions"]
            for i in range(topk.shape[0]):
                if not in_prop[i]:                      # focus on P_prop
                    continue
                cname = CELL_NAME[int(cell[i])]
                # decode top-5 ids
                top5 = topk[i, :5]
                top5p = topkp[i, :5]
                decoded = [tokenizer.decode([int(t)], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
                            for t in top5]
                ontop = [is_on_topic(d, words) for d in decoded]
                real  = [is_real_word(d)       for d in decoded]
                example = dict(
                    clip=clip, pos=int(pos[i]), cell=cname,
                    label=sorted(words),
                    top5=list(zip([int(t) for t in top5], decoded,
                                   [float(p) for p in top5p],
                                   real, ontop)),
                )
                per_layer_blocks[L][cname].append(example)
                # Misinterpreted-object candidate: uni-audio with top-1 a
                # content noun that is NOT on-topic.
                if cname == "uni_audio" and real[0] and not ontop[0] and words:
                    per_layer_blocks[L]["misinterp"].append(example)
    return per_layer_blocks


def fmt_top5(ex):
    rows = []
    for tid, dec, p, real, ontop in ex["top5"]:
        tag = []
        if ontop: tag.append("✔ on-topic")
        elif real: tag.append("content")
        else:     tag.append("…")
        rows.append(f"      {tid:>7d}  {repr(dec):<18s}  p={p:.3f}  ({', '.join(tag)})")
    return "\n".join(rows)


def write_md(blocks, out_path: Path, top_n: int = 6):
    lines = []
    lines.append("# Stage 3.3 — top-K lens decode examples\n")
    lines.append(
        "Per-clip examples of pre-SA logit-lens output for P_prop sinks at the\n"
        "Stage 3.2 candidate layers L13 and L20. Each block shows up to 6\n"
        "representative clips per cell (cross-modal / uni-audio / uni-video),\n"
        "with the clip's ground-truth VGGSounder label words and the top-5\n"
        "lens-decoded tokens + probabilities. `✔ on-topic` means the decoded\n"
        "string contains (or is contained in) a GT label word.\n"
    )
    for L in sorted(blocks.keys()):
        lines.append(f"\n---\n\n## Layer L{L}\n")
        for cname in ("cross", "uni_audio", "uni_video"):
            exs = blocks[L][cname]
            # Sort: prefer examples whose top-5 contains an on-topic token,
            # then by top-1 prob desc, so the most interesting cases lead.
            exs.sort(key=lambda e: (
                -sum(t[4] for t in e["top5"]),         # # on-topic in top-5
                -e["top5"][0][2]                       # top-1 prob
            ))
            lines.append(f"\n### L{L}  cell = `{cname}`   ({len(exs)} P_prop sinks total)\n")
            if not exs:
                lines.append("  (none)\n")
                continue
            for ex in exs[:top_n]:
                lines.append(f"\n- **clip** `{ex['clip']}`  **GT label words**: "
                             f"{ex['label'] or '—'}  **pos**={ex['pos']}")
                lines.append(f"  top-5 lens (pre-SA):")
                lines.append(fmt_top5(ex))
        # Misinterpretation block
        mis = blocks[L]["misinterp"]
        mis.sort(key=lambda e: -e["top5"][0][2])
        lines.append(f"\n### L{L}  uni-audio sinks whose top-1 is a CONTENT word "
                     f"that is NOT in GT label  (ASD Fig.4 analogue)\n")
        if not mis:
            lines.append("  (none found at this layer)\n")
        else:
            for ex in mis[:top_n]:
                lines.append(f"\n- **clip** `{ex['clip']}`  **GT**: {ex['label']}  "
                             f"**pos**={ex['pos']}  -> top-1 decodes to "
                             f"`{repr(ex['top5'][0][1])}` (p={ex['top5'][0][2]:.3f})")
                lines.append(f"  full top-5:")
                lines.append(fmt_top5(ex))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", default=str(DEFAULT_DUMP))
    p.add_argument("--out_path", default=str(DEFAULT_OUT))
    p.add_argument("--top_n", type=int, default=6)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    blocks = gather_examples(Path(args.dump_dir), tok)
    write_md(blocks, Path(args.out_path), top_n=args.top_n)


if __name__ == "__main__":
    main()
