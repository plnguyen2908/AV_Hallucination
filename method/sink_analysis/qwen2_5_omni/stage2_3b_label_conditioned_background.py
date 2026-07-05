"""
stage2_3b_label_conditioned_background.py

Stage 2.3b — revised Stage 2.3 with per-VGGSounder-label SAM 3 masks.

Stage 2.3 used a fixed 12-prompt vocabulary as a class-agnostic foreground
proxy. The user's revision: for each clip, use that clip's actual VGGSounder
ground-truth labels (from data/VGGSounder/QA.json) as the SAM 3 prompts —
don't guess a vocabulary, use the dataset's own.

For each (clip, label) and each token-cell:
  - run SAM 3 with this label as the text prompt
  - average-pool the per-pixel foreground mask to the clip's H_eff × W_eff
    grid → per-cell foreground fraction; bg_fraction = 1 - fg
  - call cell "bg" iff bg_fraction > 0.5, else "non-bg"

Per (clip, label) metrics:
  p_sink_in_bg       = #(is_sink ∧ is_bg) / #(is_bg)
  p_nonsink_in_nonbg = #(¬is_sink ∧ ¬is_bg) / #(¬is_bg)
  sink_rate          = #(is_sink) / total_cells           (base rate)
  bg_rate            = #(is_bg) / total_cells

If the visual-sink prediction holds (sinks on bg, objects skipped):
  p_sink_in_bg > sink_rate          AND      p_nonsink_in_nonbg > (1 - sink_rate)

Aggregation:
  - per label: mean across clips that have this label
  - overall:   mean of per-label means (label-weighted, so each label counted
               equally regardless of how many clips it appears in)
  - bar plot: top-N labels by clip count, two bars per label (p_sink_in_bg,
    p_nonsink_in_nonbg), with the per-clip base rates shown as reference lines.

Outputs (--output_dir):
  stage2_3b_per_clip_label.csv     per (clip, label) — full long-form
  stage2_3b_per_label.csv          per label — mean / std / n_clips
  bg_label_bar.png                 top-N label bar plot
  stage2_3b_decision.txt           verdict + numbers
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
from stage2_3_video_background_sinks import (    # noqa: E402
    Sam3BackgroundExtractor, model_frame_indices, avg_pool_mask_to_grid,
    QWEN_FRAME_FACTOR, DEFAULT_DET_THR, MAX_DETS_PER_PROMPT,
    SAM3_BPE, SAM3_CKPT,
)


LAYER = 2
DEFAULT_NPZ = (_REPO / "results/qwen2_5_omni/sink_analysis/stage2_2_spatial/"
               "per_clip_sink_coords.npz")
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_QA = _REPO / "data/VGGSounder/QA.json"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_3b_label_conditioned"

MIN_BG_CELLS = 5            # per (clip, label): need ≥ this many bg cells to count
MIN_NONBG_CELLS = 5
MIN_CLIPS_PER_LABEL = 5     # bar plot: only labels with ≥ this many clips
BG_THRESHOLD = 0.5          # bg cell iff pooled bg_fraction > this


# ----------------------------------------------------------------------
# Per-label extension on top of Sam3BackgroundExtractor
# ----------------------------------------------------------------------

class Sam3PerLabelExtractor(Sam3BackgroundExtractor):
    """Adds a method that returns ONE bg mask per label (no unioning).
    Reuses the base loader so we don't duplicate model setup."""

    def __init__(self, **kw):
        # The parent constructor expects a `prompts` list — pass a placeholder;
        # we override prompts per call.
        kw.setdefault("prompts", ["object"])
        super().__init__(**kw)

    def per_label_background_masks(self, pil_image, labels):
        """For ONE image and a LIST of labels, batch all labels as queries
        in a single forward. Returns {label: bg_mask (H_orig, W_orig) bool}.
        Labels that produce no detection above threshold get an all-bg mask."""
        if not labels:
            return {}
        w_orig, h_orig = pil_image.size

        # Build one Datapoint carrying all labels as queries.
        dp = self._Datapoint(find_queries=[], images=[])
        dp.images = [self._SAMImage(data=pil_image, objects=[],
                                     size=[h_orig, w_orig])]
        qid_to_label = {}
        for lab in labels:
            dp.find_queries.append(self._FindQueryLoaded(
                query_text=lab,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=self._InferenceMetadata(
                    coco_image_id=self._global_counter,
                    original_image_id=self._global_counter,
                    original_category_id=1,
                    original_size=[w_orig, h_orig],
                    object_id=0,
                    frame_index=0)))
            qid_to_label[self._global_counter] = lab
            self._global_counter += 1

        dp = self.transform(dp)
        batch = self._collate([dp], dict_key="dummy")["dummy"]
        batch = self._to_device(batch, self._torch.device("cuda"),
                                non_blocking=True)
        with self._torch.inference_mode():
            output = self.model(batch)
        processed = self.postprocessor.process_results(output,
                                                        batch.find_metadatas)

        out = {}
        for qid, label in qid_to_label.items():
            fg = np.zeros((h_orig, w_orig), dtype=bool)
            result = processed.get(qid, None)
            if result is None:
                out[label] = ~fg
                continue
            scores = result.get("scores", None)
            masks = result.get("masks", None)
            if scores is None or masks is None or len(scores) == 0:
                out[label] = ~fg
                continue
            if hasattr(scores, "cpu"):
                scores = scores.float().cpu().numpy()
                masks = masks.float().cpu().numpy()
            else:
                scores = np.asarray(scores, dtype=np.float32)
                masks = np.asarray(masks, dtype=np.float32)
            keep = scores >= self.detection_threshold
            if keep.any():
                keep_idx = np.where(keep)[0]
                order = keep_idx[np.argsort(-scores[keep_idx])[: self.max_per_prompt]]
                if masks.ndim == 4:
                    masks = masks[:, 0]
                for mi in order:
                    m = (masks[mi] > 0.5).astype(bool)
                    if m.shape != (h_orig, w_orig):
                        import cv2
                        m = cv2.resize(m.astype(np.uint8),
                                        (w_orig, h_orig),
                                        interpolation=cv2.INTER_NEAREST
                                        ).astype(bool)
                    fg |= m
            out[label] = ~fg
        return out


# ----------------------------------------------------------------------
# Per-clip processing
# ----------------------------------------------------------------------

def process_clip(clip_path, T_eff, H_eff, W_eff, labels, sam):
    """Returns {label: bg_grid (T_eff, H_eff, W_eff)} for this clip."""
    from PIL import Image as PILImage
    idx, total_frames, vfps, nframes, vr = model_frame_indices(str(clip_path))
    if nframes // QWEN_FRAME_FACTOR != T_eff:
        raise RuntimeError(
            f"frame-sampling drift: T_eff={T_eff}, replica nframes/2="
            f"{nframes // QWEN_FRAME_FACTOR}  (total={total_frames}, "
            f"fps={vfps:.3f}, nframes={nframes})")

    per_label_bg = {lab: np.zeros((T_eff, H_eff, W_eff), dtype=np.float32)
                    for lab in labels}
    for f in range(T_eff):
        src_a, src_b = idx[2 * f], idx[2 * f + 1]
        frames = vr.get_batch([src_a, src_b]).asnumpy()
        # Sum across the 2 source frames; divide by 2 below.
        for fi in range(frames.shape[0]):
            pil = PILImage.fromarray(frames[fi].astype(np.uint8))
            label_to_bgmask = sam.per_label_background_masks(pil, labels)
            for lab, bg_mask in label_to_bgmask.items():
                per_label_bg[lab][f] += avg_pool_mask_to_grid(
                    bg_mask, H_eff, W_eff)
        for lab in labels:
            per_label_bg[lab][f] /= 2.0
    return per_label_bg


def compute_metrics(bg_grid, frames, rows, cols, T_eff, H_eff, W_eff):
    """For one (clip, label) bg_grid (T, H, W) in [0, 1], return the per
    (clip, label) metrics."""
    is_bg = bg_grid > BG_THRESHOLD
    is_sink = np.zeros_like(is_bg, dtype=bool)
    for f, r, c in zip(frames, rows, cols):
        if 0 <= f < T_eff and 0 <= r < H_eff and 0 <= c < W_eff:
            is_sink[f, r, c] = True
    n_bg = int(is_bg.sum())
    n_nonbg = int((~is_bg).sum())
    n_sink = int(is_sink.sum())
    n_cells = is_bg.size

    sink_rate = n_sink / max(n_cells, 1)
    bg_rate = n_bg / max(n_cells, 1)
    p_sink_in_bg = float((is_sink & is_bg).sum() / max(n_bg, 1)) if n_bg else float("nan")
    p_nonsink_in_nonbg = float(((~is_sink) & (~is_bg)).sum() / max(n_nonbg, 1)) if n_nonbg else float("nan")
    p_sink_in_nonbg = float((is_sink & ~is_bg).sum() / max(n_nonbg, 1)) if n_nonbg else float("nan")
    return dict(
        n_cells=n_cells, n_sink=n_sink, n_bg=n_bg, n_nonbg=n_nonbg,
        sink_rate=sink_rate, bg_rate=bg_rate,
        p_sink_in_bg=p_sink_in_bg,
        p_nonsink_in_nonbg=p_nonsink_in_nonbg,
        p_sink_in_nonbg=p_sink_in_nonbg)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def _pool_counts(per_clip_label_df):
    """Cell-pooled counts across valid (clip, label) records. Returns a dict
    of 5 scalars: sink_bg, sink_nonbg, nonsink_bg, nonsink_nonbg, n_records."""
    df = per_clip_label_df.copy()
    df = df[(df["n_bg"] >= MIN_BG_CELLS) & (df["n_nonbg"] >= MIN_NONBG_CELLS)]
    return dict(
        sink_bg       = float((df["p_sink_in_bg"]    * df["n_bg"]).sum()),
        sink_nonbg    = float((df["p_sink_in_nonbg"] * df["n_nonbg"]).sum()),
        nonsink_bg    = float((df["n_bg"]    * (1 - df["p_sink_in_bg"])).sum()),
        nonsink_nonbg = float((df["n_nonbg"] * (1 - df["p_sink_in_nonbg"])).sum()),
        n_bg          = float(df["n_bg"].sum()),
        n_nonbg       = float(df["n_nonbg"].sum()),
        n_records     = int(len(df)),
    )


def plot_aggregate_bar(per_clip_label_df, out_path):
    """Single aggregate plot, no per-label decomposition. Cell-pooled across
    ALL valid (clip, label) records.

        LEFT  — given SINK cell:     2 bars  P(bg|sink), P(non-bg|sink)
        RIGHT — given NON-SINK cell: 2 bars  P(bg|non-sink), P(non-bg|non-sink)

    Each subplot's two bars sum to 1. Visual-sink prediction:
        LEFT  — bg bar > non-bg bar
        RIGHT — non-bg bar > bg bar
    """
    import matplotlib.pyplot as plt
    c = _pool_counts(per_clip_label_df)
    if c["n_records"] == 0:
        print(f"  [warn] no valid records — skipping aggregate bar plot")
        return
    n_sink_pool    = c["sink_bg"]    + c["sink_nonbg"]
    n_nonsink_pool = c["nonsink_bg"] + c["nonsink_nonbg"]
    p_bg_s     = c["sink_bg"]     / max(n_sink_pool, 1)
    p_nbg_s    = c["sink_nonbg"]  / max(n_sink_pool, 1)
    p_bg_ns    = c["nonsink_bg"]    / max(n_nonsink_pool, 1)
    p_nbg_ns   = c["nonsink_nonbg"] / max(n_nonsink_pool, 1)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    xs = ["bg", "non-bg"]

    # LEFT: given SINK
    bars_l = ax_l.bar(xs, [p_bg_s, p_nbg_s],
                       color=["#d62728", "#fdbf6f"],
                       edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars_l, [p_bg_s, p_nbg_s]):
        ax_l.text(b.get_x() + b.get_width() / 2, v + 0.015,
                  f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax_l.set_title("given SINK cell  (prediction: bg > non-bg)", fontsize=12)
    ax_l.set_ylabel("proportion", fontsize=11)
    ax_l.set_ylim(0, 1.08); ax_l.grid(True, ls=":", alpha=0.4, axis="y")

    # RIGHT: given NON-SINK
    bars_r = ax_r.bar(xs, [p_bg_ns, p_nbg_ns],
                       color=["#9ecae1", "#2ca02c"],
                       edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars_r, [p_bg_ns, p_nbg_ns]):
        ax_r.text(b.get_x() + b.get_width() / 2, v + 0.015,
                  f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax_r.set_title("given NON-SINK cell  (prediction: non-bg > bg)", fontsize=12)
    ax_r.set_ylim(0, 1.08); ax_r.grid(True, ls=":", alpha=0.4, axis="y")

    fig.suptitle(
        f"Stage 2.3b — cell-pooled aggregate over all (clip, label) records  "
        f"(n_records = {c['n_records']}, "
        f"n_sink_cells = {int(n_sink_pool):,}, "
        f"n_nonsink_cells = {int(n_nonsink_pool):,})",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  P(bg|sink)        = {p_bg_s:.4f}")
    print(f"  P(non-bg|sink)    = {p_nbg_s:.4f}")
    print(f"  P(bg|non-sink)    = {p_bg_ns:.4f}")
    print(f"  P(non-bg|non-sink)= {p_nbg_ns:.4f}")


def plot_aggregate_bar_forward(per_clip_label_df, out_path):
    """Forward-conditioning aggregate plot, no per-label decomposition.
    Cell-pooled across ALL valid (clip, label) records.

        LEFT  — given BG cell:     2 bars  P(sink|bg), P(non-sink|bg)
        RIGHT — given NON-BG cell: 2 bars  P(sink|non-bg), P(non-sink|non-bg)

    Bars sum to 1 within each subplot. The sink bars are tiny (sink rate
    ≈ 2 % overall), so values are annotated; y-axis spans 0-1.05 for visual
    integrity. Visual-sink prediction:
        LEFT  — sink bar > sink bar in RIGHT
                (P(sink|bg) > P(sink|non-bg))
        RIGHT — non-sink bar > non-sink bar in LEFT
                (P(non-sink|non-bg) > P(non-sink|bg))
    """
    import matplotlib.pyplot as plt
    c = _pool_counts(per_clip_label_df)
    if c["n_records"] == 0:
        print(f"  [warn] no valid records — skipping forward aggregate plot")
        return
    n_bg_pool    = c["n_bg"]
    n_nonbg_pool = c["n_nonbg"]
    p_s_bg     = c["sink_bg"]     / max(n_bg_pool, 1)
    p_ns_bg    = c["nonsink_bg"]  / max(n_bg_pool, 1)
    p_s_nbg    = c["sink_nonbg"]    / max(n_nonbg_pool, 1)
    p_ns_nbg   = c["nonsink_nonbg"] / max(n_nonbg_pool, 1)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    xs = ["sink", "non-sink"]

    bars_l = ax_l.bar(xs, [p_s_bg, p_ns_bg],
                       color=["#d62728", "#9ecae1"],
                       edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars_l, [p_s_bg, p_ns_bg]):
        ax_l.text(b.get_x() + b.get_width() / 2, v + 0.015,
                  f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    ax_l.set_title("given BG cell  (prediction: sink bar > RIGHT's sink bar)",
                    fontsize=12)
    ax_l.set_ylabel("proportion", fontsize=11)
    ax_l.set_ylim(0, 1.08); ax_l.grid(True, ls=":", alpha=0.4, axis="y")

    bars_r = ax_r.bar(xs, [p_s_nbg, p_ns_nbg],
                       color=["#fdbf6f", "#2ca02c"],
                       edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars_r, [p_s_nbg, p_ns_nbg]):
        ax_r.text(b.get_x() + b.get_width() / 2, v + 0.015,
                  f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    ax_r.set_title("given NON-BG cell  (prediction: non-sink bar > LEFT's)",
                    fontsize=12)
    ax_r.set_ylim(0, 1.08); ax_r.grid(True, ls=":", alpha=0.4, axis="y")

    fig.suptitle(
        f"Stage 2.3b — cell-pooled aggregate, forward conditioning  "
        f"(n_records = {c['n_records']}, "
        f"n_bg_cells = {int(n_bg_pool):,}, "
        f"n_nonbg_cells = {int(n_nonbg_pool):,})",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  P(sink|bg)        = {p_s_bg:.4f}")
    print(f"  P(non-sink|bg)    = {p_ns_bg:.4f}")
    print(f"  P(sink|non-bg)    = {p_s_nbg:.4f}")
    print(f"  P(non-sink|non-bg)= {p_ns_nbg:.4f}")
    # Useful derived ratio for reporting:
    if p_s_nbg > 0:
        print(f"  ratio P(sink|bg)/P(sink|non-bg) = {p_s_bg / p_s_nbg:.2f}x")


def plot_per_label_bar_inverse(per_clip_label_df, per_label_df, out_path,
                                top_n=20):
    """Bayes-flipped view of the same data:
        LEFT  — given SINK:      P(bg | sink)    vs  P(non-bg | sink)
        RIGHT — given NON-SINK:  P(bg | non-sink) vs P(non-bg | non-sink)
    These pairs are complementary (sum to 1 per subplot). Visual-sink
    prediction:
        LEFT  — bg bar > non-bg bar     (sinks land on bg)
        RIGHT — non-bg bar > bg bar     (non-sinks land on objects)

    Aggregation uses the same per-(clip, label) records: for each record,
    derive P(bg|sink) = (#(sink ∧ bg)) / n_sink and likewise for the others,
    then mean per label across clips, then filter top_n by n_clips and sort
    by the lift  P(bg|sink) − P(bg|non-sink)  so the strongest labels rise
    to the top of the chart.
    """
    import matplotlib.pyplot as plt
    df = per_clip_label_df.copy()
    # Drop records that can't form one of the conditional probabilities.
    df = df[(df["n_bg"] >= MIN_BG_CELLS) & (df["n_nonbg"] >= MIN_NONBG_CELLS)]
    df = df[df["n_sink"] >= 5]
    if df.empty:
        print(f"  [warn] no records survive filters — skipping inverse bar plot")
        return
    # Derive the inverse-conditioning columns from what we already have.
    sink_bg     = df["p_sink_in_bg"]    * df["n_bg"]        # #(sink ∧ bg)
    sink_nonbg  = df["p_sink_in_nonbg"] * df["n_nonbg"]     # #(sink ∧ non-bg)
    nonsink_bg     = df["n_bg"]    - sink_bg                # #(non-sink ∧ bg)
    nonsink_nonbg  = df["n_nonbg"] - sink_nonbg             # #(non-sink ∧ non-bg)
    n_nonsink = df["n_cells"] - df["n_sink"]
    df = df.assign(
        p_bg_given_sink         = sink_bg     / df["n_sink"],
        p_nonbg_given_sink      = sink_nonbg  / df["n_sink"],
        p_bg_given_nonsink      = nonsink_bg   / n_nonsink.replace(0, np.nan),
        p_nonbg_given_nonsink   = nonsink_nonbg / n_nonsink.replace(0, np.nan),
    )
    # Per-label mean across clips.
    agg = df.groupby("label").agg(
        n_clips=("clip", "nunique"),
        p_bg_given_sink       =("p_bg_given_sink",       "mean"),
        p_nonbg_given_sink    =("p_nonbg_given_sink",    "mean"),
        p_bg_given_nonsink    =("p_bg_given_nonsink",    "mean"),
        p_nonbg_given_nonsink =("p_nonbg_given_nonsink", "mean"),
    ).reset_index()
    agg["lift_sink_to_bg"] = (agg["p_bg_given_sink"]
                               - agg["p_bg_given_nonsink"])
    agg = agg[agg["n_clips"] >= MIN_CLIPS_PER_LABEL]
    if agg.empty:
        print(f"  [warn] no labels with n_clips ≥ {MIN_CLIPS_PER_LABEL} after "
              f"n_sink ≥ 5 filter — skipping inverse bar plot")
        return
    agg = agg.sort_values("n_clips", ascending=False).head(top_n)
    agg = agg.sort_values("lift_sink_to_bg", ascending=False)

    labels = agg["label"].tolist()
    y = np.arange(len(labels))
    w = 0.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15,
                                       max(5, 0.4 * len(labels) + 1.5)),
                                       sharey=True)

    # --- LEFT — given SINK ---
    ax_l.barh(y - w / 2, agg["p_bg_given_sink"], height=w,
              color="#d62728", alpha=0.9, label="P(bg | sink)")
    ax_l.barh(y + w / 2, agg["p_nonbg_given_sink"], height=w,
              color="#fdbf6f", alpha=0.95,
              edgecolor="#cc6611", linewidth=0.4,
              label="P(non-bg | sink)")
    # Reference: mean clip bg_rate across these labels
    if "bg_rate" in per_clip_label_df.columns:
        bg_rate_sub = per_clip_label_df[
            per_clip_label_df["label"].isin(labels)]["bg_rate"].mean()
    else:
        bg_rate_sub = float("nan")
    if np.isfinite(bg_rate_sub):
        ax_l.axvline(bg_rate_sub, ls="--", color="#444", lw=1.0, alpha=0.7,
                     label=f"clip bg_rate base ≈ {bg_rate_sub:.3f}")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, agg["n_clips"])],
                          fontsize=9)
    ax_l.set_xlabel("proportion", fontsize=11)
    ax_l.set_xlim(0, 1.02)
    ax_l.set_title("given SINK cell  (visual-sink prediction: red > orange)",
                    fontsize=11)
    ax_l.legend(loc="lower right", fontsize=9)
    ax_l.grid(True, ls=":", alpha=0.4, axis="x")
    ax_l.invert_yaxis()

    # --- RIGHT — given NON-SINK ---
    ax_r.barh(y - w / 2, agg["p_bg_given_nonsink"], height=w,
              color="#9ecae1", alpha=0.9,
              edgecolor="#1f6491", linewidth=0.4,
              label="P(bg | non-sink)")
    ax_r.barh(y + w / 2, agg["p_nonbg_given_nonsink"], height=w,
              color="#2ca02c", alpha=0.9, label="P(non-bg | non-sink)")
    if np.isfinite(bg_rate_sub):
        ax_r.axvline(bg_rate_sub, ls="--", color="#444", lw=1.0, alpha=0.7,
                     label=f"clip bg_rate base ≈ {bg_rate_sub:.3f}")
    ax_r.set_xlabel("proportion", fontsize=11)
    ax_r.set_xlim(0, 1.02)
    ax_r.set_title("given NON-SINK cell  (prediction: green > blue)",
                    fontsize=11)
    ax_r.legend(loc="lower right", fontsize=9)
    ax_r.grid(True, ls=":", alpha=0.4, axis="x")

    fig.suptitle(f"Stage 2.3b — Bayes-flipped: cell-type distribution "
                 f"conditional on sink/non-sink, by VGGSounder label "
                 f"(top {len(agg)}, n_clips ≥ {MIN_CLIPS_PER_LABEL})",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_label_bar(per_label_df, out_path, top_n=20):
    """Two horizontal subplots (shared y-axis = labels):
        LEFT  — SINK:      P(sink|bg)        vs  P(sink|non-bg)
        RIGHT — NON-SINK:  P(non-sink|bg)    vs  P(non-sink|non-bg)
    Visual-sink prediction:
        LEFT  — bg bar > non-bg bar  (sinks prefer bg)
        RIGHT — non-bg bar > bg bar  (non-sinks prefer non-bg / objects)
    """
    import matplotlib.pyplot as plt
    df = per_label_df[per_label_df["n_clips"] >= MIN_CLIPS_PER_LABEL].copy()
    df = df.sort_values("n_clips", ascending=False).head(top_n)
    if df.empty:
        print(f"  [warn] no labels with n_clips ≥ {MIN_CLIPS_PER_LABEL} — skipping bar plot")
        return
    df = df.sort_values("p_sink_in_bg_mean", ascending=False)

    # Derive the two columns we don't already have:
    # P(non-sink|bg) = 1 - P(sink|bg);  P(sink|non-bg) is already in the CSV.
    df = df.assign(
        p_nonsink_in_bg_mean=1.0 - df["p_sink_in_bg_mean"],
    )

    labels = df["label"].tolist()
    y = np.arange(len(labels))
    w = 0.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, max(5, 0.4 * len(labels) + 1.5)),
                                       sharey=True)

    # --- LEFT subplot: SINK ---
    ax_l.barh(y - w / 2, df["p_sink_in_bg_mean"], height=w,
              color="#d62728", alpha=0.9, label="P(sink | bg cell)")
    ax_l.barh(y + w / 2, df["p_sink_in_nonbg_mean"], height=w,
              color="#fdbf6f", alpha=0.95,
              edgecolor="#cc6611", linewidth=0.4,
              label="P(sink | non-bg cell)")
    base_sink = float(df["sink_rate_mean"].mean())
    ax_l.axvline(base_sink, ls="--", color="#444", lw=1.0, alpha=0.7,
                 label=f"clip sink rate base ≈ {base_sink:.3f}")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, df["n_clips"])],
                          fontsize=9)
    ax_l.set_xlabel("proportion", fontsize=11)
    ax_l.set_title("SINK proportion  (visual-sink prediction: red > orange)",
                    fontsize=11)
    ax_l.legend(loc="lower right", fontsize=9)
    ax_l.grid(True, ls=":", alpha=0.4, axis="x")
    ax_l.invert_yaxis()
    # auto x-limit; data is small fractions (sink_rate < 0.1 typically), so
    # leave a little headroom.
    xmax_l = float(np.nanmax([df["p_sink_in_bg_mean"].max(),
                               df["p_sink_in_nonbg_mean"].max(), base_sink])) * 1.15
    ax_l.set_xlim(0, max(xmax_l, 0.05))

    # --- RIGHT subplot: NON-SINK ---
    ax_r.barh(y - w / 2, df["p_nonsink_in_bg_mean"], height=w,
              color="#9ecae1", alpha=0.9,
              edgecolor="#1f6491", linewidth=0.4,
              label="P(non-sink | bg cell)")
    ax_r.barh(y + w / 2, df["p_nonsink_in_nonbg_mean"], height=w,
              color="#2ca02c", alpha=0.9, label="P(non-sink | non-bg cell)")
    base_nonsink = 1.0 - base_sink
    ax_r.axvline(base_nonsink, ls="--", color="#444", lw=1.0, alpha=0.7,
                 label=f"clip non-sink rate base ≈ {base_nonsink:.3f}")
    ax_r.set_xlabel("proportion", fontsize=11)
    ax_r.set_title("NON-SINK proportion  (prediction: green > blue)",
                    fontsize=11)
    ax_r.legend(loc="lower left", fontsize=9)
    ax_r.grid(True, ls=":", alpha=0.4, axis="x")
    ax_r.set_xlim(0.85, 1.005)   # non-sink values are near 1; zoom in

    fig.suptitle(f"Stage 2.3b — sink/non-sink proportions by VGGSounder label "
                 f"(top {len(df)}, n_clips ≥ {MIN_CLIPS_PER_LABEL})",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 2.2 sinks ---
    z = np.load(args.npz, allow_pickle=True)
    clips = [str(c) for c in z["L2_clips"]]
    offs = z["L2_offsets"]
    rows_flat = z["L2_rows_flat"]
    cols_flat = z["L2_cols_flat"]
    frames_flat = z["L2_frames_flat"]
    H_arr = z["L2_H_eff"].astype(int)
    W_arr = z["L2_W_eff"].astype(int)
    T_arr = z["L2_T_eff"].astype(int)
    n_sink_arr = z["L2_n_sink"].astype(int)
    n = len(clips)
    assert offs[0] == 0 and offs[-1] == len(rows_flat)

    # --- VGGSounder labels per clip ---
    qa = json.load(open(args.qa))
    labels_by_vid = {q["video_id"]: list(q.get("label", [])) for q in qa}
    n_with_label = sum(1 for c in clips if labels_by_vid.get(c))
    n_total_pairs = sum(len(labels_by_vid.get(c, [])) for c in clips)
    print(f"loaded {n} clips, {n_with_label} with labels in VGGSounder QA, "
          f"{n_total_pairs} total (clip, label) pairs")

    if args.limit > 0:
        n = min(n, args.limit)
        clips = clips[:n]; H_arr = H_arr[:n]; W_arr = W_arr[:n]
        T_arr = T_arr[:n]; n_sink_arr = n_sink_arr[:n]
        print(f"  --limit {args.limit} → first {n} clips")

    # --- SAM 3 ---
    if not args.dry_run:
        sam = Sam3PerLabelExtractor(
            bpe_path=SAM3_BPE, checkpoint_path=SAM3_CKPT,
            detection_threshold=args.detection_threshold,
            max_per_prompt=args.max_per_prompt)
    else:
        sam = None

    # --- Per (clip, label) pass ---
    rows_per = []
    failures = {}
    t0 = time.time()
    video_dir = Path(args.video_dir)
    for i in tqdm(range(n), desc="clips"):
        clip = clips[i]
        labels = labels_by_vid.get(clip, [])
        if not labels:
            failures["no_labels"] = failures.get("no_labels", 0) + 1
            continue
        H_eff, W_eff, T_eff = int(H_arr[i]), int(W_arr[i]), int(T_arr[i])
        frames = frames_flat[offs[i]:offs[i + 1]]
        rs = rows_flat[offs[i]:offs[i + 1]]
        cs = cols_flat[offs[i]:offs[i + 1]]
        try:
            if args.dry_run:
                # synthetic: each label gets a fresh top=bg field
                per_label_bg = {}
                rng = np.random.default_rng(hash((clip, i)) & 0xffff)
                for lab in labels:
                    row_centers = (np.arange(H_eff) + 0.5)[:, None] / H_eff
                    base = np.broadcast_to(
                        0.8 - row_centers * 0.6, (H_eff, W_eff)).copy()
                    base = np.broadcast_to(base, (T_eff, H_eff, W_eff)).copy()
                    base += rng.normal(0, 0.05, size=base.shape)
                    per_label_bg[lab] = np.clip(base, 0, 1)
            else:
                cp = video_dir / clip
                if not cp.is_file():
                    raise FileNotFoundError(str(cp))
                per_label_bg = process_clip(cp, T_eff, H_eff, W_eff, labels, sam)
        except Exception as e:
            failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
            tqdm.write(f"  [skip] {clip}: {type(e).__name__}: {e}")
            continue
        for lab, bg_grid in per_label_bg.items():
            m = compute_metrics(bg_grid, frames, rs, cs, T_eff, H_eff, W_eff)
            m.update(clip=clip, label=lab, H=H_eff, W=W_eff, T=T_eff)
            rows_per.append(m)
    elapsed = time.time() - t0
    print(f"\nprocessed {n - sum(failures.values())} clips, "
          f"{len(rows_per)} (clip, label) records in {elapsed:.1f} s")
    if failures:
        print(f"  failures: {failures}")
    if not rows_per:
        raise SystemExit("no records produced — abort")

    df = pd.DataFrame(rows_per)
    df.to_csv(out_dir / "stage2_3b_per_clip_label.csv", index=False)
    print(f"wrote {out_dir / 'stage2_3b_per_clip_label.csv'}")

    # --- Per-label aggregation (mean across clips) ---
    df_valid = df[(df["n_bg"] >= MIN_BG_CELLS) & (df["n_nonbg"] >= MIN_NONBG_CELLS)]
    print(f"  per-label aggregation: {len(df_valid)}/{len(df)} (clip, label) "
          f"records pass n_bg ≥ {MIN_BG_CELLS} and n_nonbg ≥ {MIN_NONBG_CELLS}")
    per_label = df_valid.groupby("label").agg(
        n_clips=("clip", "nunique"),
        p_sink_in_bg_mean=("p_sink_in_bg", "mean"),
        p_sink_in_bg_std=("p_sink_in_bg", "std"),
        p_nonsink_in_nonbg_mean=("p_nonsink_in_nonbg", "mean"),
        p_nonsink_in_nonbg_std=("p_nonsink_in_nonbg", "std"),
        p_sink_in_nonbg_mean=("p_sink_in_nonbg", "mean"),
        sink_rate_mean=("sink_rate", "mean"),
        bg_rate_mean=("bg_rate", "mean"),
    ).reset_index()
    per_label["lift_p_sink"] = (per_label["p_sink_in_bg_mean"]
                                 - per_label["sink_rate_mean"])
    per_label = per_label.sort_values("n_clips", ascending=False)
    per_label.to_csv(out_dir / "stage2_3b_per_label.csv", index=False)
    print(f"wrote {out_dir / 'stage2_3b_per_label.csv'}")

    # --- Overall summary (label-weighted) ---
    sub = per_label[per_label["n_clips"] >= MIN_CLIPS_PER_LABEL]
    overall_p_sink_bg = float(sub["p_sink_in_bg_mean"].mean())
    overall_p_nonsink_nonbg = float(sub["p_nonsink_in_nonbg_mean"].mean())
    overall_sink_rate = float(sub["sink_rate_mean"].mean())
    print(f"\noverall (label-weighted, n_labels ≥ {MIN_CLIPS_PER_LABEL}, "
          f"n_labels = {len(sub)}):")
    print(f"  mean P(sink   | bg cell)    = {overall_p_sink_bg:.4f}")
    print(f"  mean P(non-sk | non-bg cell)= {overall_p_nonsink_nonbg:.4f}")
    print(f"  mean clip sink rate          = {overall_sink_rate:.4f}")
    print(f"  → P(sink|bg) > sink_rate ?   = {overall_p_sink_bg > overall_sink_rate}")
    print(f"  → P(nonsk|nonbg) > (1-rate)? = {overall_p_nonsink_nonbg > (1 - overall_sink_rate)}")

    # --- Bar plot ---
    plot_per_label_bar(per_label, out_dir / "bg_label_bar.png", top_n=args.top_n)

    # --- Verdict ---
    # For each label with n_clips ≥ MIN_CLIPS_PER_LABEL: did P(sink|bg) >
    # sink_rate? (sink concentration) Did P(nonsk|nonbg) > 1-sink_rate?
    # (object skipping)
    label_lifts = sub["lift_p_sink"]
    n_pos = int((label_lifts > 0).sum())
    n_lab = int(len(label_lifts))
    sign_p = float(scistats.binomtest(n_pos, n_lab, 0.5).pvalue) if n_lab > 0 else float("nan")
    # Wilcoxon on the lifts (one-sample, against 0)
    if n_lab >= 3:
        try:
            wil = scistats.wilcoxon(label_lifts, alternative="greater")
            wil_stat, wil_p = float(wil.statistic), float(wil.pvalue)
        except Exception:
            wil_stat, wil_p = float("nan"), float("nan")
    else:
        wil_stat, wil_p = float("nan"), float("nan")
    print(f"\nlabel-level sign test: {n_pos}/{n_lab} labels show "
          f"P(sink|bg) > sink_rate; binomial p = {sign_p:.2e}")
    print(f"label-level Wilcoxon (greater): stat = {wil_stat:.3f}, "
          f"p = {wil_p:.2e}")

    # Decision file
    with open(out_dir / "stage2_3b_decision.txt", "w") as f:
        f.write("Stage 2.3b — per-VGGSounder-label SAM 3 bg/non-bg, video L2 sinks  "
                f"(n_clips={n}, layer=L{LAYER}, "
                f"τ_detect={args.detection_threshold}, "
                f"bg_threshold={BG_THRESHOLD}, dry_run={args.dry_run})\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"(clip, label) records: {len(df)}; "
                f"with n_bg ≥ {MIN_BG_CELLS} and n_nonbg ≥ {MIN_NONBG_CELLS}: "
                f"{len(df_valid)}\n")
        f.write(f"labels with n_clips ≥ {MIN_CLIPS_PER_LABEL}: {len(sub)}\n\n")
        f.write("Overall (mean across labels, label-weighted):\n")
        f.write(f"  P(sink   | bg cell)        = {overall_p_sink_bg:.4f}\n")
        f.write(f"  P(non-sk | non-bg cell)    = {overall_p_nonsink_nonbg:.4f}\n")
        f.write(f"  mean clip sink rate         = {overall_sink_rate:.4f}\n")
        f.write(f"  mean clip non-sink rate     = {1 - overall_sink_rate:.4f}\n\n")
        f.write(f"label-level sign test on lift = P(sink|bg) - sink_rate:\n")
        f.write(f"  {n_pos}/{n_lab} labels positive; binomial p = {sign_p:.2e}\n")
        f.write(f"  Wilcoxon (alt='greater'): stat = {wil_stat:.3f}, "
                f"p = {wil_p:.2e}\n\n")
        f.write("Top 30 labels by n_clips (sorted by n_clips, then by lift):\n")
        show = per_label[per_label["n_clips"] >= MIN_CLIPS_PER_LABEL].copy()
        show = show.sort_values(["n_clips", "lift_p_sink"],
                                  ascending=[False, False]).head(30)
        f.write(show[["label", "n_clips", "p_sink_in_bg_mean",
                       "p_nonsink_in_nonbg_mean", "sink_rate_mean",
                       "lift_p_sink"]].to_string(index=False) + "\n\n")
        if overall_p_sink_bg > overall_sink_rate and overall_p_nonsink_nonbg > (1 - overall_sink_rate) and sign_p < 0.05:
            verdict = (f"SEMANTIC — across {len(sub)} labels with ≥ "
                       f"{MIN_CLIPS_PER_LABEL} clips each, sinks concentrate on "
                       f"SAM 3 label-bg (P(sink|bg) = {overall_p_sink_bg:.3f} "
                       f"vs sink rate {overall_sink_rate:.3f}) and objects "
                       f"avoid sinks (P(non-sink|non-bg) = "
                       f"{overall_p_nonsink_nonbg:.3f} vs {1 - overall_sink_rate:.3f}). "
                       f"Sign test {n_pos}/{n_lab} positive (p = {sign_p:.2e}). "
                       f"Per-VGGSounder-label result agrees with the fixed-prompt "
                       f"Stage 2.3 finding.")
        else:
            verdict = (f"INCONCLUSIVE / MIXED — P(sink|bg) = {overall_p_sink_bg:.3f} "
                       f"vs sink rate {overall_sink_rate:.3f}; P(non-sink|non-bg) "
                       f"= {overall_p_nonsink_nonbg:.3f} vs {1 - overall_sink_rate:.3f}; "
                       f"sign test {n_pos}/{n_lab} positive (p = {sign_p:.2e}). "
                       f"Per-label result does not cleanly support the bg story.")
        f.write(f"VERDICT: {verdict}\n")
        print(f"\n{'=' * 80}\nSTAGE 2.3b VERDICT (n_labels = {len(sub)})\n{'=' * 80}")
        print(verdict)
    print(f"\nwrote {out_dir / 'stage2_3b_decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--qa", default=str(DEFAULT_QA),
                   help="VGGSounder QA.json with per-clip 'label' lists.")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--detection_threshold", type=float, default=DEFAULT_DET_THR)
    p.add_argument("--max_per_prompt", type=int, default=MAX_DETS_PER_PROMPT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top_n", type=int, default=20,
                   help="Top-N labels to show in the bar plot.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(args)
