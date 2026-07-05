"""
stage2_5b_log_transforms.py

Stage 2.5b — re-run the acoustic correlation analysis with log10(x + 1e-6)
applied to the positively-skewed features.

Stage 2.5 (raw features) returned WEAK (R² = 0.114, max |r| = 0.190). Several
features were heavy-tailed (silence_fraction cv = 3.4; rms_energy cv = 0.7),
so a linear pearson_r on the raw values may have understated a monotonic
relationship. This stage retests with log-compressed x; if any feature's
|r| crosses the 0.20 threshold after log, the framing changes.

Log10(x + 1e-6) is applied to:   rms_energy, peak_amplitude, zcr,
                                   spectral_centroid, silence_fraction.
Kept as-is (Shannon entropy is already log-based):
                                   spectral_entropy, temporal_entropy.

Additionally:  binary contrast on peak_amplitude ≥ 0.99 (clipped) vs
< 0.99 (unclipped), via Welch t + Mann-Whitney U.

Input:   results/.../stage2_5_acoustic_correlations/acoustic_features_per_clip.csv
Outputs (--output_dir):
    log_correlation_table.csv        raw vs log side-by-side, sorted by best |r|
    log_acoustic_correlations.png    7-panel scatters with transformed x-axes
    log_multivariate_coefficients.csv
    log_multivariate_R2.png
    clipping_comparison.png          peak≥0.99 vs <0.99 box + overlay hist
    stage2_5b_decision.txt           verdict text
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scistats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent


# --------------------------------------------------------------------------
# Constants (mirror Stage 2.5 thresholds where applicable)
# --------------------------------------------------------------------------

EPS = 1e-6                    # log floor: log10(0 + EPS) = -6
CAND_R_THR = 0.20             # |r| candidate threshold
CAND_P_THR = 0.01
ENERGY_R_VERDICT = 0.40
MULTIVAR_R2_LOW = 0.10
SPEC_R_VERDICT = 0.20

LOG_FEATS = ["rms_energy", "peak_amplitude", "zcr",
             "spectral_centroid", "silence_fraction"]
RAW_FEATS = ["spectral_entropy", "temporal_entropy"]  # Shannon — keep as-is
ALL_FEATS = LOG_FEATS + RAW_FEATS                      # 7 total

ENERGY_FEATS = ["rms_energy", "peak_amplitude", "silence_fraction"]
COMPLEXITY_FEATS = ["spectral_centroid", "spectral_entropy",
                    "temporal_entropy", "zcr"]
CLIP_THR = 0.99               # peak_amplitude clipping cutoff


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log_transform(x):
    return np.log10(np.asarray(x, dtype=np.float64) + EPS)


def correlations(x, y):
    r_p, p_p = scistats.pearsonr(x, y)
    r_s, p_s = scistats.spearmanr(x, y)
    return dict(pearson_r=float(r_p), pearson_p=float(p_p),
                spearman_r=float(r_s), spearman_p=float(p_s))


def build_side_by_side(df, target):
    """For each feature, compute raw correlations and (if a LOG_FEAT) log
    correlations. Flag the strict |r| threshold crossing — raw ≤ 0.20 AND
    log > 0.20 — separately from the p<0.01 candidate gate."""
    rows = []
    for f in ALL_FEATS:
        x_raw = df[f].values
        c_raw = correlations(x_raw, target)
        if f in LOG_FEATS:
            c_log = correlations(log_transform(x_raw), target)
        else:
            c_log = dict(pearson_r=np.nan, pearson_p=np.nan,
                         spearman_r=np.nan, spearman_p=np.nan)
        cand_raw = (abs(c_raw["pearson_r"]) > CAND_R_THR
                    and c_raw["pearson_p"] < CAND_P_THR)
        cand_log = (False if np.isnan(c_log["pearson_r"])
                    else (abs(c_log["pearson_r"]) > CAND_R_THR
                          and c_log["pearson_p"] < CAND_P_THR))
        # Strict threshold-crossing: raw was below, log is above (or vice versa).
        if np.isnan(c_log["pearson_r"]):
            crossed = False
        else:
            crossed = ((abs(c_log["pearson_r"]) > CAND_R_THR)
                       != (abs(c_raw["pearson_r"]) > CAND_R_THR))
        delta_abs_r = (abs(c_log["pearson_r"]) - abs(c_raw["pearson_r"])
                       if not np.isnan(c_log["pearson_r"]) else np.nan)
        rows.append(dict(
            feature=f, transform_applied=(f in LOG_FEATS),
            raw_pearson_r=c_raw["pearson_r"], raw_pearson_p=c_raw["pearson_p"],
            raw_spearman_r=c_raw["spearman_r"], raw_spearman_p=c_raw["spearman_p"],
            log_pearson_r=c_log["pearson_r"], log_pearson_p=c_log["pearson_p"],
            log_spearman_r=c_log["spearman_r"], log_spearman_p=c_log["spearman_p"],
            delta_abs_r=delta_abs_r,
            cand_raw=cand_raw, cand_log=cand_log,
            crossed_r_threshold=crossed,
        ))
    return pd.DataFrame(rows)


def multivariate_ols(X, y, feat_names):
    """Standardized OLS via numpy + scipy.t — same scheme as Stage 2.5."""
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1) + 1e-12
    Xs = (X - mu) / sd
    n, p = Xs.shape
    X_aug = np.column_stack([np.ones(n), Xs])
    XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
    beta = XtX_inv @ X_aug.T @ y
    y_pred = X_aug @ beta
    rss = float(((y - y_pred) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / max(tss, 1e-12)
    dof = max(n - p - 1, 1)
    sigma2 = rss / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    t_stat = beta / np.maximum(se, 1e-12)
    p_val = 2.0 * (1.0 - scistats.t.cdf(np.abs(t_stat), dof))
    t_crit = float(scistats.t.ppf(0.975, dof))
    coefs = []
    for i, f in enumerate(feat_names):
        b = float(beta[i + 1]); s = float(se[i + 1])
        coefs.append(dict(
            feature=f, std_coef=b, se=s,
            ci95_lo=b - t_crit * s, ci95_hi=b + t_crit * s,
            t=float(t_stat[i + 1]), p=float(p_val[i + 1]),
        ))
    return dict(r2=float(r2), n=int(n), dof=int(dof),
                intercept=float(beta[0]), coefs=coefs)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_scatters_log(df, target, table, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    sp = np.asarray(target, dtype=float)
    for i, feat in enumerate(ALL_FEATS):
        ax = axes[i]
        row = table[table.feature == feat].iloc[0]
        if feat in LOG_FEATS:
            x = log_transform(df[feat].values)
            r, p = row["log_pearson_r"], row["log_pearson_p"]
            rho = row["log_spearman_r"]
            xlabel = f"log10({feat} + {EPS:g})"
            cand = row["cand_log"]
            raw_annot = (f"\nraw r = {row['raw_pearson_r']:+.3f},  "
                         f"Δ|r| = {row['delta_abs_r']:+.3f}")
        else:
            x = df[feat].values
            r, p = row["raw_pearson_r"], row["raw_pearson_p"]
            rho = row["raw_spearman_r"]
            xlabel = feat
            cand = row["cand_raw"]
            raw_annot = "  (not transformed)"
        color = "#d62728" if cand else "#1f77b4"
        ax.scatter(x, sp, s=12, alpha=0.65, color=color, edgecolor="none")
        if np.ptp(x) > 0:
            m, b = np.polyfit(x, sp, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(xs, m * xs + b, color="black", lw=1.3, ls="--")
        star = " ★" if cand else ""
        title = (f"r = {r:+.3f} (p = {p:.1e}), ρ = {rho:+.3f}{star}"
                 + raw_annot)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("sink proportion @ L21", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
    axes[-1].axis("off")
    crossed = table.loc[table["crossed_r_threshold"], "feature"].tolist()
    cross_msg = ", ".join(crossed) if crossed else "(none)"
    axes[-1].text(0.0, 0.95,
                  f"n_clips = {len(sp)}\n"
                  f"sink_proportion: mean = {sp.mean():.3f}, std = "
                  f"{sp.std():.3f}\n     range = [{sp.min():.3f}, "
                  f"{sp.max():.3f}]\n\n"
                  f"★ candidate criterion (per panel):\n"
                  f"   |r| > {CAND_R_THR}  AND  p < {CAND_P_THR}\n\n"
                  f"crossed |r| > {CAND_R_THR} after log:\n   {cross_msg}",
                  transform=axes[-1].transAxes, fontsize=10, va="top",
                  family="monospace")
    fig.suptitle("Stage 2.5b — log-transformed acoustic features vs audio "
                 f"sink proportion @ L21  (n = {len(sp)})",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_multivariate(mv, out_path, title_suffix=""):
    feats = [c["feature"] for c in mv["coefs"]]
    coefs = np.array([c["std_coef"] for c in mv["coefs"]])
    los = np.array([c["std_coef"] - c["ci95_lo"] for c in mv["coefs"]])
    his = np.array([c["ci95_hi"] - c["std_coef"] for c in mv["coefs"]])
    colors = ["#d62728" if c["p"] < 0.01 else "#1f77b4" for c in mv["coefs"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(feats))
    ax.barh(y, coefs, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
    ax.errorbar(coefs, y, xerr=[los, his], fmt="none", color="black",
                capsize=3, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=10)
    ax.set_xlabel("standardized OLS coefficient (95% CI)", fontsize=11)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"Multivariate OLS{title_suffix}  —  R² = {mv['r2']:.3f}, "
                 f"n = {mv['n']}  (red = p < 0.01)", fontsize=12)
    ax.grid(True, ls=":", alpha=0.4, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def clipping_test(df, target, out_path):
    peak = df["peak_amplitude"].values
    sp = np.asarray(target, dtype=float)
    clipped = peak >= CLIP_THR
    n_c, n_u = int(clipped.sum()), int((~clipped).sum())
    sp_c, sp_u = sp[clipped], sp[~clipped]
    out = dict(
        clip_threshold=CLIP_THR,
        n_clipped=n_c, n_unclipped=n_u,
        mean_clipped=(float(sp_c.mean()) if n_c else float("nan")),
        std_clipped=(float(sp_c.std()) if n_c else float("nan")),
        mean_unclipped=(float(sp_u.mean()) if n_u else float("nan")),
        std_unclipped=(float(sp_u.std()) if n_u else float("nan")),
    )
    if n_c >= 2 and n_u >= 2:
        t, t_p = scistats.ttest_ind(sp_c, sp_u, equal_var=False)
        u, mw_p = scistats.mannwhitneyu(sp_c, sp_u, alternative="two-sided")
        out.update(t_stat=float(t), t_p=float(t_p),
                   mw_u=float(u), mw_p=float(mw_p),
                   delta_mean=out["mean_clipped"] - out["mean_unclipped"])
    else:
        out.update(t_stat=float("nan"), t_p=float("nan"),
                   mw_u=float("nan"), mw_p=float("nan"),
                   delta_mean=float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].boxplot([sp_u, sp_c], labels=[f"unclipped\n(n={n_u})",
                                          f"clipped\n(n={n_c})"],
                    widths=0.5)
    axes[0].set_ylabel("sink_proportion @ L21", fontsize=11)
    axes[0].set_title(f"clipped vs unclipped  (peak ≥ {CLIP_THR})",
                      fontsize=12)
    axes[0].grid(True, ls=":", alpha=0.4, axis="y")
    bins = np.linspace(sp.min(), sp.max(), 30)
    axes[1].hist([sp_u, sp_c], bins=bins,
                 label=[f"unclipped (n={n_u})", f"clipped (n={n_c})"],
                 alpha=0.75, color=["#1f77b4", "#d62728"])
    axes[1].set_xlabel("sink_proportion @ L21", fontsize=11)
    axes[1].set_ylabel("# clips", fontsize=11)
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].set_title("distribution overlay", fontsize=12)
    axes[1].grid(True, ls=":", alpha=0.4, axis="y")
    suptitle = (f"Stage 2.5b — peak-amplitude clipping test  "
                f"(Δμ = {out['delta_mean']:+.3f}, "
                f"Welch t p = {out['t_p']:.2e}, "
                f"Mann-Whitney p = {out['mw_p']:.2e})")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = Path(args.features_csv)
    if not feat_csv.is_file():
        raise SystemExit(f"Stage 2.5 features CSV not found at {feat_csv}")
    df = pd.read_csv(feat_csv)
    needed = ALL_FEATS + ["sink_proportion"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"features CSV missing columns: {missing}")
    df = df.dropna(subset=needed).reset_index(drop=True)
    target = df["sink_proportion"].values
    print(f"loaded {len(df)} clips from {feat_csv}")

    # ----- per-feature skew before/after log -----
    print("\nskew (Fisher) — raw vs log-transformed:")
    for f in ALL_FEATS:
        sk_raw = float(scistats.skew(df[f].values))
        if f in LOG_FEATS:
            sk_log = float(scistats.skew(log_transform(df[f].values)))
            tag = ""
            if abs(sk_raw) > 1 and abs(sk_log) < abs(sk_raw) / 2:
                tag = "   (log helped)"
            print(f"  {f:20s} raw {sk_raw:+.3f}  →  log {sk_log:+.3f}{tag}")
        else:
            print(f"  {f:20s} raw {sk_raw:+.3f}  (not transformed)")

    # ----- side-by-side correlations -----
    table = build_side_by_side(df, target)
    # Sort by max(|raw|, |log|) for "best correlation across both views".
    best_abs = np.where(
        table["log_pearson_r"].notna(),
        np.maximum(table["log_pearson_r"].abs(), table["raw_pearson_r"].abs()),
        table["raw_pearson_r"].abs(),
    )
    table_sorted = table.assign(_best_abs_r=best_abs) \
                        .sort_values("_best_abs_r", ascending=False) \
                        .drop(columns=["_best_abs_r"])
    table_sorted.to_csv(out_dir / "log_correlation_table.csv", index=False)
    print(f"\nwrote {out_dir / 'log_correlation_table.csv'}")

    print("\nside-by-side correlations (sorted by max(|raw_r|, |log_r|)):")
    hdr = (f"{'feature':22s} {'raw_r':>8s} {'raw_p':>10s} "
           f"{'log_r':>8s} {'log_p':>10s} {'Δ|r|':>8s} {'cross?':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in table_sorted.iterrows():
        log_r = (f"{r['log_pearson_r']:+8.3f}"
                 if not pd.isna(r["log_pearson_r"]) else "      —")
        log_p = (f"{r['log_pearson_p']:10.1e}"
                 if not pd.isna(r["log_pearson_p"]) else "         —")
        d = (f"{r['delta_abs_r']:+8.3f}"
             if not pd.isna(r["delta_abs_r"]) else "      —")
        cross = "★ yes" if r["crossed_r_threshold"] else "no"
        print(f"{r['feature']:22s} {r['raw_pearson_r']:+8.3f} "
              f"{r['raw_pearson_p']:10.1e} {log_r} {log_p} {d} {cross:>8s}")

    plot_scatters_log(df, target, table,
                      out_dir / "log_acoustic_correlations.png")

    # ----- mixed multivariate OLS: log on LOG_FEATS, raw on RAW_FEATS -----
    X = np.zeros((len(df), len(ALL_FEATS)), dtype=np.float64)
    feat_names = []
    for j, f in enumerate(ALL_FEATS):
        if f in LOG_FEATS:
            X[:, j] = log_transform(df[f].values)
            feat_names.append(f"log_{f}")
        else:
            X[:, j] = df[f].values
            feat_names.append(f)
    mv = multivariate_ols(X, target, feat_names)
    pd.DataFrame(mv["coefs"]).assign(
        r2=mv["r2"], n=mv["n"], dof=mv["dof"], intercept=mv["intercept"]
    ).to_csv(out_dir / "log_multivariate_coefficients.csv", index=False)
    print(f"\nwrote {out_dir / 'log_multivariate_coefficients.csv'}")
    print(f"multivariate OLS (mixed: log on {len(LOG_FEATS)}, raw on "
          f"{len(RAW_FEATS)})  R² = {mv['r2']:.4f}  "
          f"(n = {mv['n']}, dof = {mv['dof']})")
    for c in mv["coefs"]:
        flag = "★" if c["p"] < 0.01 else " "
        print(f"  {flag} {c['feature']:24s} β = {c['std_coef']:+.3f}  "
              f"[{c['ci95_lo']:+.3f}, {c['ci95_hi']:+.3f}]  p = {c['p']:.1e}")
    plot_multivariate(mv, out_dir / "log_multivariate_R2.png",
                      title_suffix=" (log on skewed features)")

    # ----- clipping comparison -----
    clip_summary = clipping_test(df, target, out_dir / "clipping_comparison.png")
    print(f"\npeak ≥ {CLIP_THR} clipping contrast:  "
          f"n_clipped = {clip_summary['n_clipped']}, "
          f"n_unclipped = {clip_summary['n_unclipped']}")
    if clip_summary["n_clipped"] >= 2 and clip_summary["n_unclipped"] >= 2:
        print(f"  mean sink_proportion:  clipped = "
              f"{clip_summary['mean_clipped']:.3f} "
              f"(std {clip_summary['std_clipped']:.3f}),  "
              f"unclipped = {clip_summary['mean_unclipped']:.3f} "
              f"(std {clip_summary['std_unclipped']:.3f})")
        print(f"  Δμ = {clip_summary['delta_mean']:+.3f}   "
              f"Welch t = {clip_summary['t_stat']:+.2f}, "
              f"p = {clip_summary['t_p']:.2e}   "
              f"Mann-Whitney p = {clip_summary['mw_p']:.2e}")
    else:
        print("  not enough samples in one group for hypothesis tests.")

    # ----- updated verdict -----
    # Best single correlation across raw + log.
    all_rs = []
    for _, r in table.iterrows():
        all_rs.append((r["feature"] + "[raw]",
                       float(r["raw_pearson_r"]), float(r["raw_pearson_p"])))
        if not pd.isna(r["log_pearson_r"]):
            all_rs.append((r["feature"] + "[log]",
                           float(r["log_pearson_r"]),
                           float(r["log_pearson_p"])))
    all_rs.sort(key=lambda t: abs(t[1]), reverse=True)
    best_feat, best_r, best_p = all_rs[0]
    abs_best = abs(best_r)

    def best_signed_r(feat):
        """Returns the version (raw or log) with the larger |r|, signed."""
        row = table[table.feature == feat].iloc[0]
        if pd.isna(row["log_pearson_r"]):
            return float(row["raw_pearson_r"])
        return (float(row["log_pearson_r"])
                if abs(row["log_pearson_r"]) > abs(row["raw_pearson_r"])
                else float(row["raw_pearson_r"]))

    rms_r = best_signed_r("rms_energy")
    peak_r = best_signed_r("peak_amplitude")
    sil_r = best_signed_r("silence_fraction")
    abs_energy_r = max(abs(rms_r), abs(peak_r), abs(sil_r))
    abs_spec_r = max(abs(best_signed_r(f)) for f in COMPLEXITY_FEATS)
    sign_consistent_prior = (rms_r < 0 and peak_r < 0 and sil_r > 0)

    if abs_energy_r > ENERGY_R_VERDICT and sign_consistent_prior:
        verdict = (f"CONFIRMED — silence/low-energy clips elicit more sinks. "
                   f"Energy |r| max = {abs_energy_r:.3f}  "
                   f"(rms r={rms_r:+.3f}, peak r={peak_r:+.3f}, "
                   f"silence r={sil_r:+.3f}).")
    elif mv["r2"] < MULTIVAR_R2_LOW:
        verdict = (f"NOT acoustic — multivariate R² = {mv['r2']:.3f} < "
                   f"{MULTIVAR_R2_LOW:.2f}, even with log transforms.")
    elif abs_spec_r > SPEC_R_VERDICT:
        verdict = (f"COMPLEXITY-driven — best spectral/complexity |r| = "
                   f"{abs_spec_r:.3f} (energy = {abs_energy_r:.3f}). "
                   f"Modulation is by acoustic complexity, not loudness.")
    elif abs_best > CAND_R_THR:
        verdict = (f"WEAK-but-present — strongest single correlation is "
                   f"{best_feat} r = {best_r:+.3f} (p = {best_p:.1e}), "
                   f"above the |r| > {CAND_R_THR} candidate threshold. "
                   f"Multivariate R² = {mv['r2']:.3f}.")
    else:
        verdict = (f"WEAK — even after log transform, no feature reaches "
                   f"|r| > {CAND_R_THR}. Best is {best_feat} r = "
                   f"{best_r:+.3f} (p = {best_p:.1e}). "
                   f"Multivariate R² = {mv['r2']:.3f}.")

    if not sign_consistent_prior:
        verdict += (f"  NOTE: rms / peak / silence signs (rms {rms_r:+.2f}, "
                    f"peak {peak_r:+.2f}, silence {sil_r:+.2f}) do NOT match "
                    f"the 'silence → more sinks' prior — direction is "
                    f"reversed or mixed.")

    crossed_list = table.loc[table["crossed_r_threshold"],
                              "feature"].tolist()

    print("\n" + "=" * 86)
    print(f"STAGE 2.5b VERDICT  (n = {len(df)})")
    print("=" * 86)
    print(verdict)
    print(f"\n  strongest correlation across raw+log:  "
          f"{best_feat}  r = {best_r:+.3f}  p = {best_p:.1e}")
    print(f"  features crossing |r| > {CAND_R_THR} after log:  "
          f"{crossed_list if crossed_list else 'none'}")
    print(f"  multivariate R² (raw, Stage 2.5)  = 0.114")
    print(f"  multivariate R² (mixed log/raw)   = {mv['r2']:.4f}  "
          f"(Δ = {mv['r2'] - 0.114:+.4f})")

    with open(out_dir / "stage2_5b_decision.txt", "w") as f:
        f.write("Stage 2.5b — log-transformed acoustic correlations on "
                "per-clip audio sink proportion @ L21\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"n_clips analyzed: {len(df)}\n")
        f.write(f"sink_proportion: mean = {target.mean():.3f}, "
                f"std = {target.std():.3f}\n\n")
        f.write("Side-by-side correlations (sorted by max(|raw_r|, |log_r|)):\n")
        f.write(table_sorted.to_string(index=False) + "\n\n")
        f.write(f"Multivariate OLS (mixed log/raw) R² = {mv['r2']:.4f}  "
                f"(n = {mv['n']}, dof = {mv['dof']})\n")
        for c in mv["coefs"]:
            f.write(f"  {c['feature']:24s} β = {c['std_coef']:+.3f}  "
                    f"[{c['ci95_lo']:+.3f}, {c['ci95_hi']:+.3f}]  "
                    f"p = {c['p']:.1e}\n")
        f.write(f"\nClipping test (peak ≥ {CLIP_THR}):\n")
        for k, v in clip_summary.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nCrossed |r| > {CAND_R_THR} after log: "
                f"{crossed_list if crossed_list else 'none'}\n")
        f.write(f"\nVERDICT: {verdict}\n")
    print(f"\nwrote {out_dir / 'stage2_5b_decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_5_acoustic_correlations/"
                               "acoustic_features_per_clip.csv"),
                   help="Stage 2.5 cached per-clip features CSV.")
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_5b_log_transforms"))
    args = p.parse_args()
    main(args)
