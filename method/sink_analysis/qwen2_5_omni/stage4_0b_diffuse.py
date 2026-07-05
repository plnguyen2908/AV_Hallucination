"""
stage4_0b_diffuse.py — pre-registered diffuse-audio vs concentrated-video
test on `heads.csv` per-head causal-influence scores.

Pure post-processing of:
  results/qwen2_5_omni/categorize_exp/heads.csv     (per-head score_A / score_V / score_AV)

No model, no GPU, no forward pass. Reads scores; computes D1 (sorted
shape), D2 (Gini + top-K shares), D3 (supra-threshold layer histograms);
applies the pre-registered verdict logic.

Verdict logic (DECIDED BEFORE LOOKING AT NUMBERS):
  DIFFUSE  iff D2 audio Gini < video Gini
            AND D2 audio top-5 share < video top-5 share
            AND D3 audio supra-threshold mean-layer > video AND audio layer-SD > video.
  REFUTED  iff audio is also concentrated (steep elbow, high Gini, high top-5)
            OR audio supra-threshold heads are early / clustered like video.
  MIXED    otherwise; report which discriminators agree.

Threshold robustness:
  D2 (Gini, top-K share) over ALL 784 heads is τ-independent → primary D2.
  D3 (supra-threshold layer mean/SD) is τ-dependent → reported at BOTH
    τ_90 and τ_95 (recomputed from the score columns; per
    `count_fix.md` heads.csv `category` is τ_90 only).

Outputs (`stage4_0b_diffuse/`):
  head_influence_distribution.csv
  decision.txt
  (figure produced by the sibling make_figures.py)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
DEFAULT_HEADS = _REPO / "results/qwen2_5_omni/categorize_exp/heads.csv"
DEFAULT_OUT   = _REPO / "results/qwen2_5_omni/sink_analysis/stage4_0b_diffuse"

TOPK_VALUES = (1, 3, 5, 10, 20, 40)
# Stage 1.2 sink peak layers (per CLAUDE.md): video L2, audio L21.
SINK_PEAK_LAYER = {"video": 2, "audio": 21}


def gini(x: np.ndarray) -> float:
    """Gini coefficient of nonneg array; 0=perfectly equal, 1=perfectly
    unequal (one head holds all mass). Standard discrete formula."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def topk_shares(x: np.ndarray, ks=TOPK_VALUES) -> dict:
    """Fraction of total mass held by the top-k entries."""
    x = np.asarray(x, dtype=float)
    s = np.sort(x)[::-1]
    total = s.sum()
    if total <= 0:
        return {f"top{k}_share": float("nan") for k in ks}
    return {f"top{k}_share": float(s[:k].sum() / total) for k in ks}


def supra_threshold_layer_stats(df: pd.DataFrame, score_col: str,
                                  tau_pool_val: float) -> dict:
    """Heads with |score_col| > tau_pool_val (the percentile-of-pooled
    |scores| threshold from heads.csv). Returns count, layer mean / SD /
    median / IQR for layer distribution of supra-threshold heads."""
    sup = df[df[score_col].abs() > tau_pool_val]
    if sup.empty:
        return dict(n=0, layer_mean=float("nan"), layer_sd=float("nan"),
                    layer_median=float("nan"), layer_iqr=float("nan"))
    return dict(
        n=int(len(sup)),
        layer_mean=float(sup["layer"].mean()),
        layer_sd=float(sup["layer"].std(ddof=1)) if len(sup) > 1 else float("nan"),
        layer_median=float(sup["layer"].median()),
        layer_iqr=float(np.percentile(sup["layer"], 75) -
                          np.percentile(sup["layer"], 25)),
    )


def pre_registered_verdict(metrics: dict, tau_label: str) -> str:
    """Apply the pre-registered rules to one τ's metrics; return one of
    {'DIFFUSE', 'REFUTED', 'MIXED'} with the failing/passing discriminators
    listed inline."""
    a = metrics["audio"]
    v = metrics["video"]

    d2_gini_ok  = a["gini"]         < v["gini"]
    d2_top5_ok  = a["top5_share"]   < v["top5_share"]
    d3_layer_mean_ok = a[f"{tau_label}_layer_mean"] > v[f"{tau_label}_layer_mean"]
    d3_layer_sd_ok   = a[f"{tau_label}_layer_sd"]   > v[f"{tau_label}_layer_sd"]
    checks = {
        "D2 audio Gini < video Gini":         d2_gini_ok,
        "D2 audio top-5 share < video top-5": d2_top5_ok,
        "D3 audio supra-thresh layer mean > video": d3_layer_mean_ok,
        "D3 audio supra-thresh layer SD > video":   d3_layer_sd_ok,
    }
    n_pass = sum(checks.values())
    if n_pass == 4:
        verdict = "DIFFUSE CONFIRMED"
    elif n_pass == 0:
        verdict = "REFUTED"
    else:
        verdict = "MIXED"
    return verdict, checks


def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.heads_csv)
    expected = {"layer", "head", "score_A", "score_V"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"heads.csv missing required cols: {missing}")
    print(f"heads.csv: {len(df)} rows, cols={df.columns.tolist()}")

    # τ values — recompute the percentile of |scores| pooled across A∪V∪AV,
    # matching av_fusion_categorize_exp.py.
    pool = np.abs(np.concatenate([df["score_A"].values, df["score_V"].values,
                                    df["score_AV"].values]))
    tau_pool = {p: float(np.percentile(pool, p)) for p in (90, 95)}
    print(f"τ values (pooled |scores| percentiles): τ_90={tau_pool[90]:.6g}  τ_95={tau_pool[95]:.6g}")

    abs_A = df["score_A"].abs().values
    abs_V = df["score_V"].abs().values

    # D2 — Gini and top-K shares (τ-independent, over ALL 784 heads).
    rows = []
    for mod, x in (("audio", abs_A), ("video", abs_V)):
        rec = dict(modality=mod, n_heads_total=int(x.size), gini=gini(x))
        rec.update(topk_shares(x))
        # Per-τ supra-threshold layer stats (D3)
        for tau_p, tau_val in tau_pool.items():
            score_col = "score_A" if mod == "audio" else "score_V"
            stats = supra_threshold_layer_stats(df, score_col, tau_val)
            rec[f"tau_{tau_p}_n_supra"]      = stats["n"]
            rec[f"tau_{tau_p}_layer_mean"]   = stats["layer_mean"]
            rec[f"tau_{tau_p}_layer_sd"]     = stats["layer_sd"]
            rec[f"tau_{tau_p}_layer_median"] = stats["layer_median"]
            rec[f"tau_{tau_p}_layer_iqr"]    = stats["layer_iqr"]
        rows.append(rec)
    metrics_df = pd.DataFrame(rows)
    out_csv = out_dir / "head_influence_distribution.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Per-τ verdict ----
    metrics_dict = {"audio": metrics_df.set_index("modality").loc["audio"].to_dict(),
                    "video": metrics_df.set_index("modality").loc["video"].to_dict()}
    decision_lines = []
    decision_lines.append("Stage 4.0b — Pre-registered verdict\n" + "=" * 60)
    decision_lines.append("\nPre-registered rules (DECIDED BEFORE LOOKING AT NUMBERS):")
    decision_lines.append("  DIFFUSE CONFIRMED  iff all four below hold:")
    decision_lines.append("    D2a: audio Gini < video Gini")
    decision_lines.append("    D2b: audio top-5 share < video top-5 share")
    decision_lines.append("    D3a: audio supra-threshold layer mean > video")
    decision_lines.append("    D3b: audio supra-threshold layer SD > video")
    decision_lines.append("  REFUTED  iff zero hold; MIXED otherwise.\n")

    for tau_p in (95, 90):
        tau_label = f"tau_{tau_p}"
        verdict, checks = pre_registered_verdict(metrics_dict, tau_label)
        decision_lines.append(f"\n--- τ_{tau_p} verdict ---")
        for name, ok in checks.items():
            decision_lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_pass = sum(checks.values())
        decision_lines.append(f"  → {verdict}  ({n_pass}/4 discriminators pass)")

    # Sink-peak triangulation
    decision_lines.append("\n--- D3 triangulation (Stage 1.2 sink peaks) ---")
    decision_lines.append("  Reference: video sink peak L2, audio sink peak L21.")
    for tau_p in (95, 90):
        a_mean = metrics_dict["audio"][f"tau_{tau_p}_layer_mean"]
        v_mean = metrics_dict["video"][f"tau_{tau_p}_layer_mean"]
        decision_lines.append(f"  τ_{tau_p}: audio supra-thresh layer-mean = {a_mean:.1f}  vs audio sink L21")
        decision_lines.append(f"       :  video supra-thresh layer-mean = {v_mean:.1f}  vs video sink L2")

    decision_lines.append("\nCaveats (state, do not fix):")
    decision_lines.append("  - heads.csv scores come from a PRE-3.1 (pre-SA-standardization) attribution.")
    decision_lines.append("    Re-attribution under the pre-SA convention could reshuffle scores/membership.")
    decision_lines.append("  - |score_A| large = 'this head's ablation moves audio-conditioned output' —")
    decision_lines.append("    may track general audio-processing importance, not only hallucination.")

    dec_path = out_dir / "decision.txt"
    dec_path.write_text("\n".join(decision_lines) + "\n")
    print(f"\nwrote {dec_path}\n")
    print("\n".join(decision_lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heads_csv", default=str(DEFAULT_HEADS))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args)
