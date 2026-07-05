"""
stage3_2_overlap.py — Part 2 of Stage 3.2 (overlap analysis).

Consumes per-clip token dumps from stage3_2_token_dump.py and computes,
for each LLM decoder layer L pooled across the 50 VGGSounder clips:

  1. Whole-population overlap (P_prop vs P_llm sink sets):
        Jaccard      = |P_prop ∩ P_llm| / |P_prop ∪ P_llm|
        prop_in_llm  = |P_prop ∩ P_llm| / |P_prop|     (containment of small set)
        llm_in_prop  = |P_prop ∩ P_llm| / |P_llm|     (reverse — should be low)

  2. Cross-modal-restricted overlap (THE KEY NUMBER):
        n_prop_cross                = |P_prop ∩ cross_cell|
        n_prop_cross_in_llm_cross   = |P_prop ∩ cross_cell ∩ P_llm ∩ cross_cell|
        prop_cross_in_llm_cross_frac= ratio of those two

  3. Saturation flag: P_llm audio-span sink-fraction ≥ 0.5 (per-clip avg)
     is precomputed in the existing Stage 3.1 per-layer CSV and re-attached
     here so the verdict can lean on non-saturated layers.

POOLING. Sets are pooled per-clip (a clip's P_prop has ~18 elements;
P_llm at saturated layers is in the hundreds), and the SUMS of per-clip
intersection cardinalities and per-clip union/denominator cardinalities
are used to form the layer-level ratios. This is the correct pooled
estimator for Jaccard / containment (each clip contributes proportionally
to its set size). Per-clip ratios are NOT averaged.

Output: stage3_2/overlap_by_layer.csv
        layer, n_prop, n_llm, n_intersect, jaccard, prop_in_llm, llm_in_prop,
        n_prop_cross, n_prop_cross_in_llm_cross, prop_cross_in_llm_cross_frac,
        late_saturated_audio

Read-only on stage3_1_preSA outputs; writes only under stage3_2/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

DEFAULT_DUMP = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2/per_clip_tokens"
DEFAULT_OUT  = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_2"
DEFAULT_S3_1 = _REPO / "results/qwen2_5_omni/sink_analysis/stage3_1_preSA/ref_all/stage3_1_per_layer.csv"

CROSS_CELL = 0  # int8 value in mds_cell for cross-modal


def load_dumps(dump_dir: Path):
    paths = sorted(dump_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz under {dump_dir}")
    out = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        out.append(dict(
            clip=str(z["clip"]),
            p_llm=z["p_llm"],          # (n_layers, S) bool
            p_prop=z["p_prop"],        # (S,) bool
            mds_cell=z["mds_cell"],    # (n_layers, S) int8
            video_pos=z["video_pos"],
            audio_pos=z["audio_pos"],
            S=int(z["S"]),
        ))
    return out


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dumps = load_dumps(Path(args.dump_dir))
    n_layers = dumps[0]["p_llm"].shape[0]
    print(f"loaded {len(dumps)} per-clip dumps; n_layers={n_layers}")

    # Per-layer pooled tallies (sums across clips of set cardinalities).
    n_prop_per_L                  = np.zeros(n_layers, dtype=np.int64)
    n_llm_per_L                   = np.zeros(n_layers, dtype=np.int64)
    n_inter_per_L                 = np.zeros(n_layers, dtype=np.int64)
    n_union_per_L                 = np.zeros(n_layers, dtype=np.int64)
    n_prop_cross_per_L            = np.zeros(n_layers, dtype=np.int64)
    n_prop_cross_in_llm_per_L     = np.zeros(n_layers, dtype=np.int64)
    n_prop_cross_in_llmcross_per_L= np.zeros(n_layers, dtype=np.int64)
    n_llm_cross_per_L             = np.zeros(n_layers, dtype=np.int64)

    for d in dumps:
        p_prop = d["p_prop"]                 # (S,) bool
        for L in range(n_layers):
            p_llm_L = d["p_llm"][L]          # (S,) bool
            cell_L  = d["mds_cell"][L]       # (S,) int8
            cross_L = (cell_L == CROSS_CELL)

            inter = p_prop & p_llm_L
            union = p_prop | p_llm_L

            n_prop_per_L[L]                   += int(p_prop.sum())
            n_llm_per_L[L]                    += int(p_llm_L.sum())
            n_inter_per_L[L]                  += int(inter.sum())
            n_union_per_L[L]                  += int(union.sum())

            prop_cross   = p_prop & cross_L
            llm_cross    = p_llm_L & cross_L
            n_prop_cross_per_L[L]             += int(prop_cross.sum())
            n_llm_cross_per_L[L]              += int(llm_cross.sum())
            n_prop_cross_in_llm_per_L[L]      += int((prop_cross & p_llm_L).sum())
            n_prop_cross_in_llmcross_per_L[L] += int((prop_cross & llm_cross).sum())

    # Saturation tag from Stage 3.1 per-layer CSV (P_llm audio-span sink frac).
    s31 = pd.read_csv(args.s3_1_per_layer_csv)
    sat_lookup = dict(zip(s31["layer"].astype(int),
                          s31["mean_audio_llm_sink_fraction"].astype(float)))

    def _frac(num, denom):
        return float(num / denom) if denom > 0 else float("nan")

    rows = []
    for L in range(n_layers):
        n_prop  = int(n_prop_per_L[L])
        n_llm   = int(n_llm_per_L[L])
        n_int   = int(n_inter_per_L[L])
        n_union = int(n_union_per_L[L])
        n_pc    = int(n_prop_cross_per_L[L])
        n_pclc  = int(n_prop_cross_in_llmcross_per_L[L])
        n_lc    = int(n_llm_cross_per_L[L])
        sat = float(sat_lookup.get(L, float("nan")))
        rows.append(dict(
            layer=L,
            n_prop=n_prop, n_llm=n_llm,
            n_intersect=n_int, n_union=n_union,
            jaccard       = _frac(n_int, n_union),
            prop_in_llm   = _frac(n_int, n_prop),
            llm_in_prop   = _frac(n_int, n_llm),
            n_prop_cross  = n_pc,
            n_llm_cross   = n_lc,
            n_prop_cross_in_llm_cross    = n_pclc,
            prop_cross_in_llm_cross_frac = _frac(n_pclc, n_pc),
            mean_audio_llm_sink_fraction = sat,
            late_saturated_audio         = bool(L >= 20 and sat >= 0.5),
        ))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "overlap_by_layer.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(df)} rows)")

    # Pretty print
    print()
    cols_show = ["layer", "n_prop", "n_llm", "jaccard", "prop_in_llm",
                 "llm_in_prop", "n_prop_cross", "n_prop_cross_in_llm_cross",
                 "prop_cross_in_llm_cross_frac", "mean_audio_llm_sink_fraction",
                 "late_saturated_audio"]
    print(df[cols_show].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # Late-stack pooled summary (use both flavors).
    print("\nLate-stack summaries:")
    for label, mask in (
        ("L20-27 ALL",                 df["layer"].between(20, 27)),
        ("L20-27 NON-SATURATED audio", df["layer"].between(20, 27)
                                       & ~df["late_saturated_audio"]),
        ("L8-17  (mid stack)",         df["layer"].between(8, 17)),
        ("L20-22 (late, pre-sat)",     df["layer"].between(20, 22)),
    ):
        sub = df[mask]
        if sub.empty:
            print(f"  {label:30s} (no layers)")
            continue
        n_pc   = int(sub["n_prop_cross"].sum())
        n_pclc = int(sub["n_prop_cross_in_llm_cross"].sum())
        n_int  = int(sub["n_intersect"].sum())
        n_un   = int(sub["n_union"].sum())
        n_prop = int(sub["n_prop"].sum())
        n_llm  = int(sub["n_llm"].sum())
        print(f"  {label:30s}  layers={sub['layer'].tolist()}")
        print(f"    Jaccard        = {n_int/n_un:.3f}    "
              f"prop_in_llm = {n_int/n_prop:.3f}    llm_in_prop = {n_int/n_llm:.3f}")
        print(f"    prop_cross_in_llm_cross_frac = {n_pclc/n_pc:.3f}  "
              f"(n_prop_cross={n_pc}, hits={n_pclc})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", default=str(DEFAULT_DUMP))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--s3_1_per_layer_csv", default=str(DEFAULT_S3_1))
    args = p.parse_args()
    main(args)
