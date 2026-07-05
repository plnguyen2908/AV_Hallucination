"""
stage2_3b_replot.py

Regenerate Stage 2.3b's bar plot from the cached per-(clip, label) CSV,
without re-running SAM 3. Use after the full run finishes (or any time the
plot function changes).

  python stage2_3b_replot.py
  python stage2_3b_replot.py --output_dir /path/with/cached_csv
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from stage2_3b_label_conditioned_background import (  # noqa: E402
    plot_per_label_bar, plot_per_label_bar_inverse,
    plot_aggregate_bar, plot_aggregate_bar_forward,
    MIN_BG_CELLS, MIN_NONBG_CELLS, MIN_CLIPS_PER_LABEL,
)


def main(args):
    out_dir = Path(args.output_dir)
    csv = out_dir / "stage2_3b_per_clip_label.csv"
    if not csv.is_file():
        raise SystemExit(f"not found: {csv}")
    df = pd.read_csv(csv)
    print(f"loaded {len(df)} (clip, label) records from {csv}")

    valid = df[(df["n_bg"] >= MIN_BG_CELLS) & (df["n_nonbg"] >= MIN_NONBG_CELLS)]
    print(f"  {len(valid)}/{len(df)} pass n_bg ≥ {MIN_BG_CELLS} and "
          f"n_nonbg ≥ {MIN_NONBG_CELLS}")
    per_label = valid.groupby("label").agg(
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

    sub = per_label[per_label["n_clips"] >= MIN_CLIPS_PER_LABEL]
    print(f"\n{len(sub)} labels with n_clips ≥ {MIN_CLIPS_PER_LABEL}")
    print(f"  mean P(sink   | bg)        = {sub['p_sink_in_bg_mean'].mean():.4f}")
    print(f"  mean P(sink   | non-bg)    = {sub['p_sink_in_nonbg_mean'].mean():.4f}")
    print(f"  mean P(non-sk | bg)        = {1 - sub['p_sink_in_bg_mean'].mean():.4f}")
    print(f"  mean P(non-sk | non-bg)    = {sub['p_nonsink_in_nonbg_mean'].mean():.4f}")
    print(f"  mean clip sink_rate base    = {sub['sink_rate_mean'].mean():.4f}")

    plot_per_label_bar(per_label, out_dir / "bg_label_bar.png", top_n=args.top_n)
    # Bayes-flipped view: same data, inverse conditioning, per label
    plot_per_label_bar_inverse(df, per_label,
                                out_dir / "bg_label_bar_inverse.png",
                                top_n=args.top_n)
    # Single aggregate (cell-pooled across all valid (clip, label) records).
    # Two views — both useful for the paper:
    #   bg_aggregate_bar.png         — given SINK / given NON-SINK (inverse)
    #   bg_aggregate_bar_forward.png — given BG / given NON-BG     (forward)
    plot_aggregate_bar(df, out_dir / "bg_aggregate_bar.png")
    plot_aggregate_bar_forward(df, out_dir / "bg_aggregate_bar_forward.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default=str(_HERE.parent.parent.parent
                                / "results/qwen2_5_omni/sink_analysis"
                                  "/stage2_3b_label_conditioned"))
    p.add_argument("--top_n", type=int, default=20)
    args = p.parse_args()
    main(args)
