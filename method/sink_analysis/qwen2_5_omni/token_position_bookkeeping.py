"""
token_position_bookkeeping.py

Document the input_ids layout that Qwen2.5-Omni 7B sees across our three
datasets so downstream sink / attention analyses can sum over modality
spans unambiguously.

For each of 9 clips
    3 × AudioSet     (modal_type = "a")
    3 × ActivityNet  (modal_type = "v")
    3 × VGGSounder   (modal_type = "av")
we run the standard chat-template prompt, tokenise via the processor, run
greedy decoding for 50 tokens, and identify the absolute (start, end) index
range of each token category in the full (prompt + generated) sequence:

    system    — first <|im_start|> .. its matching <|im_end|>  (inclusive)
    audio     — inner span between <|audio_bos|> / <|audio_eos|>
    video     — inner span between <|vision_bos|> / <|vision_eos|>
    query     — user-turn text tokens after modal block(s),
                ending at the user-turn <|im_end|>
    generated — positions appended by model.generate(...) (greedy, 50 tokens)

Outputs:
    stdout              per-clip table + per-dataset summary
    token_layout.md     same content, plus prose conventions, in markdown

Sanity check:
    For each dataset, span-length min/max/mean across the 3 clips is reported.
    If lengths vary wildly within a dataset, the tokeniser is doing something
    non-deterministic and we should investigate.
"""

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# Reuse the existing Qwen2.5-Omni loader & conversation builder.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "method/qwen2_5_omni"))
from utils import build_conversation, load_omni, prepare_inputs  # noqa: E402


PROMPT = "Describe what you hear in detail."
SEED = 42
CATEGORIES = ("system", "audio", "video", "query", "generated")

DATASET_CONFIGS = [
    # (display name, modal_type, repo-relative dir, glob pattern)
    ("AudioSet",    "a",  "data/AudioSet/audios",   "*.wav"),
    ("ActivityNet", "v",  "data/ActivityNet/videos", "*.mp4"),
    ("VGGSounder",  "av", "data/VGGSounder/videos",  "*.mp4"),
]


# --------------------------------------------------------------------------
# Span detection helpers
# --------------------------------------------------------------------------

def _resolve_thinker_cfg(model):
    """Modal token IDs live on thinker.config OR thinker.config.text_config
    depending on transformers version."""
    cfg = model.thinker.config
    if not hasattr(cfg, "audio_start_token_id"):
        cfg = getattr(cfg, "text_config", cfg)
    return cfg


def _find_pair(ids: list, start_id: int, end_id: int):
    """First (start_pos, end_pos) where ids[start_pos] == start_id and
    ids[end_pos] == end_id with end_pos > start_pos. Returns (None, None) if
    either marker is missing."""
    try:
        s = ids.index(start_id)
        e = ids.index(end_id, s + 1)
        return s, e
    except ValueError:
        return None, None


def categorise(prompt_ids: list, gen_ids: list, tokenizer, thinker_cfg) -> dict:
    """Return {category: (start, end)} over the full (prompt+gen) sequence.
    end is exclusive (Python slice semantics)."""
    full = list(prompt_ids) + list(gen_ids)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    im_starts = [i for i, t in enumerate(full) if t == im_start_id]
    im_ends = [i for i, t in enumerate(full) if t == im_end_id]

    regions: dict[str, tuple[int, int]] = {}

    # System turn — first <|im_start|> through its matching <|im_end|>.
    if im_starts and im_ends:
        sys_s = im_starts[0]
        sys_e = next((e for e in im_ends if e > sys_s), None)
        if sys_e is not None:
            regions["system"] = (sys_s, sys_e + 1)

    # Audio and video inner spans.
    a_s, a_e = _find_pair(
        full, thinker_cfg.audio_start_token_id, thinker_cfg.audio_end_token_id
    )
    v_s, v_e = _find_pair(
        full, thinker_cfg.vision_start_token_id, thinker_cfg.vision_end_token_id
    )
    if a_s is not None:
        regions["audio"] = (a_s + 1, a_e)
    if v_s is not None:
        regions["video"] = (v_s + 1, v_e)

    # Query — user-turn text tokens. User-turn = (im_starts[1], im_ends[1]).
    if len(im_starts) >= 2 and len(im_ends) >= 2:
        user_s = im_starts[1]
        user_e = next((e for e in im_ends if e > user_s), None)
        if user_e is not None:
            # Query starts right after the last modal block in the user turn,
            # or right after the <|im_start|>user\n header if no modal block.
            modal_end = None
            if a_e is not None and user_s < a_e < user_e:
                modal_end = max(modal_end or 0, a_e + 1)
            if v_e is not None and user_s < v_e < user_e:
                modal_end = max(modal_end or 0, v_e + 1)
            if modal_end is None:
                # No modal block in this user turn — skip the
                # <|im_start|>user\n header (≈3 tokens) so we don't report it
                # as "query". Best-effort approximation.
                modal_end = user_s + 3
            regions["query"] = (modal_end, user_e)

    # Generated — everything past the prompt length.
    if gen_ids:
        prompt_len = len(prompt_ids)
        regions["generated"] = (prompt_len, prompt_len + len(gen_ids))

    return regions


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _fmt_span(regions: dict, name: str) -> str:
    if name not in regions:
        return "—"
    s, e = regions[name]
    return f"[{s},{e}) len={e - s}"


def main(args):
    random.seed(SEED)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-Omni ...")
    model, processor = load_omni(args.model_path)
    tokenizer = processor.tokenizer
    thinker_cfg = _resolve_thinker_cfg(model)

    all_rows: list[dict] = []

    for dataset_name, modal_type, rel_dir, pattern in DATASET_CONFIGS:
        d = _REPO / rel_dir
        if not d.exists():
            print(f"[{dataset_name}] dir {d} not found — skipping.")
            continue
        files = sorted(d.glob(pattern))
        if not files:
            print(f"[{dataset_name}] no files matching {pattern} in {d}")
            continue
        random.shuffle(files)
        clips = files[: args.n_per_dataset]
        for clip in tqdm(clips, desc=f"{dataset_name} ({modal_type})"):
            conv = build_conversation(str(clip), PROMPT, modal_type)
            try:
                inputs, use_aiv = prepare_inputs(
                    processor, conv, modal_type, model.device, model.dtype
                )
            except Exception as e:
                print(f"  skip {clip.name}: preprocess error: {e}")
                continue
            try:
                with torch.inference_mode():
                    text_ids = model.generate(
                        **inputs,
                        use_audio_in_video=use_aiv,
                        return_audio=False,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
            except Exception as e:
                print(f"  skip {clip.name}: generate error: {e}")
                continue

            prompt_len = inputs["input_ids"].shape[1]
            prompt_ids = inputs["input_ids"][0].tolist()
            gen_ids = text_ids[0, prompt_len:].tolist()
            regions = categorise(prompt_ids, gen_ids, tokenizer, thinker_cfg)
            all_rows.append({
                "dataset": dataset_name,
                "modal": modal_type,
                "clip": clip.name,
                "total_len": prompt_len + len(gen_ids),
                "regions": regions,
            })

    if not all_rows:
        raise SystemExit("No clips processed — bailing.")

    # ----- stdout table -----
    print()
    header = (
        f"{'Dataset':<12}{'Modal':<6}{'Clip':<32}{'Total':<8}"
        f"{'System':<18}{'Audio':<18}{'Video':<18}{'Query':<14}{'Generated':<16}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in all_rows:
        regs = r["regions"]
        print(
            f"{r['dataset']:<12}{r['modal']:<6}{r['clip'][:30]:<32}"
            f"{r['total_len']:<8}"
            f"{_fmt_span(regs, 'system'):<18}{_fmt_span(regs, 'audio'):<18}"
            f"{_fmt_span(regs, 'video'):<18}{_fmt_span(regs, 'query'):<14}"
            f"{_fmt_span(regs, 'generated'):<16}"
        )

    # ----- per-dataset sanity -----
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_ds[r["dataset"]].append(r)

    print("\n" + "=" * 70)
    print("Per-dataset span-length consistency")
    print("=" * 70)
    summary_blocks: list[str] = []
    for ds, rows in by_ds.items():
        block = [f"\n## {ds}\n", f"- clips probed: {len(rows)}\n"]
        for cat in CATEGORIES:
            lens = [
                r["regions"][cat][1] - r["regions"][cat][0]
                for r in rows if cat in r["regions"]
            ]
            if not lens:
                continue
            mn, mx = min(lens), max(lens)
            mean = statistics.mean(lens)
            ratio = (mx / mn) if mn > 0 else float("inf")
            warn = "  ⚠ wide spread" if mn > 0 and ratio > 3 else ""
            block.append(
                f"- {cat:<10}: n={len(lens):>2}  min={mn:<6} "
                f"max={mx:<6} mean={mean:.1f}{warn}\n"
            )
        # Layout pattern from clip 0 (categories sorted by start position).
        regs0 = rows[0]["regions"]
        layout = " → ".join(
            f"{name}[{s},{e})"
            for name, (s, e) in sorted(regs0.items(), key=lambda kv: kv[1][0])
        )
        block.append(f"- layout (clip 0): `{layout}`\n")
        summary_blocks.append("".join(block))
        print("".join(block))

    # ----- markdown summary -----
    with open(out_md, "w") as f:
        f.write("# Qwen2.5-Omni 7B token-position layout\n\n")
        f.write(
            "Generated by "
            "`method/sink_analysis/qwen2_5_omni/token_position_bookkeeping.py`.\n\n"
        )
        f.write(f"- Prompt: `{PROMPT}`\n")
        f.write(f"- Decoding: greedy, max_new_tokens={args.max_new_tokens}\n")
        f.write(
            f"- Datasets sampled: "
            f"AudioSet (modal=a), ActivityNet (modal=v), VGGSounder (modal=av). "
            f"`--n_per_dataset={args.n_per_dataset}` clips each.\n\n"
        )
        f.write("## Category definitions\n\n")
        f.write(
            "All indices are absolute positions in the full sequence "
            "(`prompt + generated`). `end` is exclusive (Python slice).\n\n"
        )
        f.write(
            "- **system** — first `<|im_start|>` through its matching "
            "`<|im_end|>` (inclusive).\n"
            "- **audio** — *inner* span between `<|audio_bos|>` "
            "(`audio_start_token_id`) and `<|audio_eos|>` "
            "(`audio_end_token_id`). The two markers themselves are not "
            "included.\n"
            "- **video** — *inner* span between `<|vision_bos|>` "
            "(`vision_start_token_id`) and `<|vision_eos|>` "
            "(`vision_end_token_id`).\n"
            "- **query** — user-turn text tokens **after** the last modal "
            "block in the user turn, ending at the user-turn `<|im_end|>`.\n"
            "- **generated** — positions appended by `model.generate(...)`. "
            "Starts at `prompt_len` and runs to the end of the sequence.\n\n"
        )
        f.write("## Per-clip layout\n\n")
        f.write(
            "| Dataset | Modal | Clip | Total | System | Audio | Video | "
            "Query | Generated |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            regs = r["regions"]
            f.write(
                f"| {r['dataset']} | {r['modal']} | `{r['clip']}` | "
                f"{r['total_len']} | {_fmt_span(regs, 'system')} | "
                f"{_fmt_span(regs, 'audio')} | {_fmt_span(regs, 'video')} | "
                f"{_fmt_span(regs, 'query')} | {_fmt_span(regs, 'generated')} |\n"
            )
        f.write("\n## Per-dataset summary\n")
        for blk in summary_blocks:
            f.write(blk)
        f.write(
            "\n## Conventions\n\n"
            "- For AudioSet (modal=a), only the audio block is present; "
            "the video span is absent.\n"
            "- For ActivityNet (modal=v), only the video block is present; "
            "the audio span is absent.\n"
            "- For VGGSounder (modal=av), the video block is added explicitly "
            "and the processor extracts an audio track from the same file "
            "via `use_audio_in_video=True`, producing both spans. Check the "
            "per-clip table above to see which comes first in the sequence.\n"
            "- Audio token count depends on clip duration (~25 tokens per "
            "second of audio); video token count depends on (frames × spatial "
            "patches) which is governed by `fps` and `max_pixels` set in "
            "`build_conversation`.\n"
        )

    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--n_per_dataset", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument(
        "--output_md",
        default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                    "stage0_2_token_bookkeeping/token_layout.md"),
    )
    args = p.parse_args()
    main(args)
