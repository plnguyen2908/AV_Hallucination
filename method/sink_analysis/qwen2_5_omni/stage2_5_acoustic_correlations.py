"""
stage2_5_acoustic_correlations.py

Stage 2.5 — What clip-level acoustic properties predict audio sink proportion?

Stage 2.4 found that at L21, audio sink proportion = 0.736 ± 0.081 across 300
AudioSet clips. Within-clip distribution is uniform (KS 0.062, IQR 0.494) — the
variance lives at the CLIP LEVEL, not the within-clip level. This stage asks
what clip-level acoustic properties correlate with that cross-clip variance.

Hypothesis: clips with LESS acoustic content (silence, simple, low-energy)
elicit MORE sinks (the LLM registers many "nothing here" positions). Clips
with MORE acoustic content (loud, complex, dense) elicit FEWER sinks
(positions carry real information rather than being register-able).

Inputs:
  - Stage 2.4's per_clip_temporal_stats.csv (provides sink_proportion @ L21).
  - The same 300 AudioSet wav clips, in the same seed=42 permutation order
    Stage 2.4 used. Alignment is verified via the n_audio sanity check
    (expected_n_audio = round(duration_sec * 25) vs Stage 2.4's n_audio).

Acoustic features (clip-level scalars; 25 ms windows, 10 ms hop):
  1. rms_energy         sqrt(mean(audio**2))            overall loudness
  2. peak_amplitude     max(|audio|)                    peak loudness
  3. zcr                zero-crossing rate              tonal vs noisy
  4. spectral_centroid  mean freq-weighted spectrum     bright vs dark
  5. spectral_entropy   mean per-frame entropy of |STFT| over freq bins
                                                        spread vs tonal
  6. temporal_entropy   entropy of the RMS envelope     bursty vs steady
  7. silence_fraction   frac of frames w/ RMS < 0.01 × peak RMS
                                                        empty-space share

Analysis:
  - Per-feature Pearson + Spearman correlation with sink_proportion.
  - Multivariate OLS (standardized features) with per-coefficient t / p / CI.
  - Optional: AudioSet top-label box plot if --labels_csv is provided.

Verdict (printed to stdout + stage2_5_decision.txt):
  - |r| > 0.40 on energy features (rms/peak/silence) with the expected
    signs (rms-,peak-,silence+) → CONFIRMED "fill the empty space" story.
  - Multivariate R² < 0.10                                    → NOT acoustic.
  - Spectral/complexity features dominate (|r| > 0.20) but not energy
                                                              → COMPLEXITY-driven.
  - Otherwise                                                 → WEAK.

Outputs (--output_dir):
  acoustic_features_per_clip.csv     clip basename + features + sink_proportion
  correlation_table.csv              sorted by |pearson_r|
  multivariate_coefficients.csv      standardized β + CI + p
  acoustic_correlations.png          7 scatter panels + stats panel
  multivariate_R2.png                horizontal bar chart of std β with 95% CI
  sink_proportion_by_label.png       optional, if --labels_csv given
  stage2_5_decision.txt              verdict text
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scistats
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Stage 0.2 token rate: AudioSet 10 s clips → 250 audio tokens (40 ms / token).
AUDIO_TOK_PER_SEC = 25.0
N_AUDIO_TOL = 2                # ±tokens tolerance for the alignment check
ALIGN_FAIL_FRAC = 0.05         # > this fraction of mismatches → abort

FRAME_MS = 25.0
HOP_MS = 10.0
SILENCE_REL_PEAK_RMS = 0.01    # frame is "silent" if RMS < 0.01 × peak frame RMS

FEATURE_KEYS = ["rms_energy", "peak_amplitude", "zcr",
                "spectral_centroid", "spectral_entropy",
                "temporal_entropy", "silence_fraction"]
ENERGY_FEATS = ["rms_energy", "peak_amplitude", "silence_fraction"]
COMPLEXITY_FEATS = ["spectral_centroid", "spectral_entropy",
                    "temporal_entropy", "zcr"]

CAND_R_THR = 0.20              # |r| above this AND p<0.01 → "candidate"
CAND_P_THR = 0.01
ENERGY_R_VERDICT = 0.40        # strong energy correlation → CONFIRMED
MULTIVAR_R2_LOW = 0.10         # R² below this → not acoustic
SPEC_R_VERDICT = 0.20          # spectral correlation above this → complexity-driven


# --------------------------------------------------------------------------
# Acoustic features
# --------------------------------------------------------------------------

def compute_features(audio: np.ndarray, sr: int) -> dict:
    """Compute the 7 clip-level scalar acoustic features.

    Sample-rate agnostic — frame/hop sizes derived from FRAME_MS / HOP_MS.
    Returns NaN-free floats; tiny eps used to guard log(0) / divide-by-zero.
    """
    import librosa  # imported lazily so module-import works without it

    eps = 1e-12
    n_fft = max(int(round(sr * FRAME_MS / 1000.0)), 256)
    hop = max(int(round(sr * HOP_MS / 1000.0)), 64)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    rms = float(np.sqrt(np.mean(audio ** 2) + eps))
    peak = float(np.max(np.abs(audio)) + eps)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        audio, frame_length=n_fft, hop_length=hop)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop)))

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) + eps
    # Per-frame distribution over frequency bins; mean entropy across frames.
    P = S / S.sum(axis=0, keepdims=True)
    spec_entropy = float(np.mean(-(P * np.log(P + eps)).sum(axis=0)))

    rms_env = librosa.feature.rms(y=audio, frame_length=n_fft,
                                  hop_length=hop)[0]
    # Temporal entropy on normalized RMS envelope.
    P_t = rms_env / (rms_env.sum() + eps)
    temp_entropy = float(-(P_t * np.log(P_t + eps)).sum())

    # Silence relative to the loudest frame in this clip (per-clip threshold,
    # so it doesn't conflate with absolute loudness — that's what RMS measures).
    silence_thr = SILENCE_REL_PEAK_RMS * float(rms_env.max() + eps)
    silence_frac = float((rms_env < silence_thr).mean())

    return dict(rms_energy=rms, peak_amplitude=peak, zcr=zcr,
                spectral_centroid=centroid, spectral_entropy=spec_entropy,
                temporal_entropy=temp_entropy, silence_fraction=silence_frac)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def correlation_table(features_df: pd.DataFrame,
                      target: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in FEATURE_KEYS:
        x = features_df[col].values
        r_p, p_p = scistats.pearsonr(x, target)
        r_s, p_s = scistats.spearmanr(x, target)
        rows.append(dict(
            feature=col,
            pearson_r=float(r_p), pearson_p=float(p_p),
            spearman_r=float(r_s), spearman_p=float(p_s),
            direction="+" if r_p > 0 else "-",
            candidate=bool(abs(r_p) > CAND_R_THR and p_p < CAND_P_THR),
        ))
    return pd.DataFrame(rows)


def multivariate_ols(features_df: pd.DataFrame,
                     target: np.ndarray) -> dict:
    """Standardized OLS. Returns R², per-coefficient β / SE / t / p / 95% CI.
    Done with numpy + scipy.t so we don't need statsmodels."""
    X_raw = features_df[FEATURE_KEYS].values.astype(np.float64)
    mu = X_raw.mean(axis=0); sd = X_raw.std(axis=0, ddof=1) + 1e-12
    X = (X_raw - mu) / sd
    y = np.asarray(target, dtype=np.float64)
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])              # n × (p+1)
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
    for i, f in enumerate(FEATURE_KEYS):
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

def plot_scatters(features_df: pd.DataFrame, target: np.ndarray,
                  corr_df: pd.DataFrame, out_path: Path) -> None:
    """7 scatter panels + a stats panel."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    sp = np.asarray(target, dtype=float)
    for i, feat in enumerate(FEATURE_KEYS):
        ax = axes[i]
        x = features_df[feat].values
        row = corr_df[corr_df.feature == feat].iloc[0]
        cand = bool(row["candidate"])
        color = "#d62728" if cand else "#1f77b4"
        ax.scatter(x, sp, s=12, alpha=0.65, color=color, edgecolor="none")
        # Linear regression line for visual reference (matches pearson_r).
        if np.ptp(x) > 0:
            m, b = np.polyfit(x, sp, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(xs, m * xs + b, color="black", lw=1.3, ls="--")
        star = " ★" if cand else ""
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("sink proportion @ L21", fontsize=10)
        ax.set_title(f"r={row['pearson_r']:+.3f} (p={row['pearson_p']:.1e}), "
                     f"ρ={row['spearman_r']:+.3f}{star}", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)

    axes[-1].axis("off")
    axes[-1].text(0.0, 0.95,
                  f"n_clips = {len(sp)}\n"
                  f"sink_proportion: mean={sp.mean():.3f}, std={sp.std():.3f}\n"
                  f"     range = [{sp.min():.3f}, {sp.max():.3f}]\n\n"
                  f"★ candidate criterion:\n"
                  f"   |pearson_r| > {CAND_R_THR}  AND  p < {CAND_P_THR}",
                  transform=axes[-1].transAxes, fontsize=10, va="top",
                  family="monospace")
    fig.suptitle("Stage 2.5 — acoustic features vs audio sink proportion @ L21 "
                 f"(n_clips={len(sp)})", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_multivariate(mv: dict, out_path: Path) -> None:
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
    ax.set_title(f"Multivariate OLS  —  R² = {mv['r2']:.3f}, n = {mv['n']}  "
                 f"(red = p < 0.01)", fontsize=12)
    ax.grid(True, ls=":", alpha=0.4, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_by_label(target: np.ndarray, labels_per_clip: list, out_path: Path,
                  top_k: int = 15):
    """Box plot of sink_proportion by clip's top label, top-k categories."""
    df = pd.DataFrame({"label": labels_per_clip,
                       "sink_proportion": np.asarray(target)})
    df = df.dropna(subset=["label"])
    if df.empty:
        print("  [warn] no labels matched any clip — skipping label plot.")
        return None
    counts = df["label"].value_counts()
    top = counts.head(top_k).index.tolist()
    df_top = df[df["label"].isin(top)]
    if df_top.empty:
        return None
    groups = [df_top[df_top.label == c]["sink_proportion"].values for c in top]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(groups, labels=[f"{c}\n(n={counts[c]})" for c in top],
               showfliers=False)
    ax.set_xlabel("AudioSet top label", fontsize=11)
    ax.set_ylabel("sink proportion @ L21", fontsize=11)
    ax.set_title(f"Sink proportion by AudioSet top label  "
                 f"(top {len(top)} of {len(counts)} labels)", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    # ANOVA across the top-k groups.
    nontriv = [g for g in groups if len(g) >= 2]
    if len(nontriv) < 2:
        return dict(n_categories=len(top), F=float("nan"), p=float("nan"))
    F, p = scistats.f_oneway(*nontriv)
    return dict(n_categories=len(top), F=float(F), p=float(p))


# --------------------------------------------------------------------------
# Optional AudioSet label parsing
# --------------------------------------------------------------------------

def parse_audioset_labels(seg_csv: Path, class_csv: Path = None) -> dict:
    """Parse AudioSet *_segments.csv (eval/balanced/unbalanced) → {YTID: label}.

    Standard AudioSet header: 3 comment lines, then `YTID,start,end,positive_labels`
    where positive_labels is quoted, comma-separated mids. If class_csv is given
    (`class_labels_indices.csv` with columns `index,mid,display_name`), mids are
    mapped to human-readable display names; otherwise the raw mid is returned."""
    seg = pd.read_csv(seg_csv, skiprows=3, header=None, sep=",",
                      quotechar='"', skipinitialspace=True,
                      names=["YTID", "start", "end", "labels"])
    seg["top_label_id"] = seg["labels"].apply(
        lambda s: str(s).split(",")[0].strip())
    if class_csv and Path(class_csv).is_file():
        cls = pd.read_csv(class_csv)
        if "mid" in cls.columns and "display_name" in cls.columns:
            id2name = dict(zip(cls["mid"], cls["display_name"]))
            seg["top_label"] = seg["top_label_id"].map(id2name).fillna(
                seg["top_label_id"])
        else:
            seg["top_label"] = seg["top_label_id"]
    else:
        seg["top_label"] = seg["top_label_id"]
    return dict(zip(seg["YTID"], seg["top_label"]))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Stage 2.4 per-clip stats
    # ------------------------------------------------------------------
    csv_path = Path(args.s24_csv)
    if not csv_path.is_file():
        raise SystemExit(f"Stage 2.4 CSV not found at {csv_path}")
    s24 = pd.read_csv(csv_path)
    if "sink_proportion" not in s24.columns:
        s24["sink_proportion"] = (s24["n_sink"]
                                  / s24["n_audio"].clip(lower=1))
    print(f"loaded {len(s24)} per-clip rows from {csv_path}")

    # ------------------------------------------------------------------
    # 2. Reconstruct Stage 2.4 clip ordering (same audio_dir + seed)
    # ------------------------------------------------------------------
    clip_dir = Path(args.audio_dir)
    clips_all = sorted(clip_dir.glob("*.wav"))
    if not clips_all:
        raise SystemExit(f"No .wav in {clip_dir}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[: args.n_clips]
    clips = [clips_all[i] for i in idx]
    print(f"reconstructed {len(clips)} clip paths  "
          f"(seed={args.seed}, audio_dir={clip_dir})")

    # If Stage 2.4 had failures, len(s24) < len(clips). Truncate to common
    # length and verify alignment by n_audio (next step) — any misalignment
    # surfaces immediately. If Stage 2.4 had 0 failures (the documented case),
    # this is just a clean 300-vs-300 pair.
    n_common = min(len(s24), len(clips))
    if len(s24) != len(clips):
        print(f"  [note] CSV rows ({len(s24)}) ≠ reconstructed clips "
              f"({len(clips)}) — truncating both to {n_common} and relying on "
              f"the n_audio sanity check below.")
    s24 = s24.iloc[:n_common].reset_index(drop=True)
    clips = clips[:n_common]

    # ------------------------------------------------------------------
    # 3. Compute acoustic features per clip
    # ------------------------------------------------------------------
    try:
        import librosa  # noqa: F401  -- early import so failures are loud
    except ImportError as e:
        raise SystemExit(f"librosa not installed in this env: {e}")

    feat_rows = []
    fail_count = 0
    for clip in tqdm(clips, desc="features"):
        try:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, sr = librosa.load(str(clip), sr=None, mono=True)
        except Exception as e:
            print(f"  [warn] failed to load {clip.name}: {e}")
            fail_count += 1
            feat_rows.append(dict(
                clip=clip.name, duration_sec=np.nan,
                expected_n_audio=np.nan, sample_rate=np.nan,
                **{k: np.nan for k in FEATURE_KEYS}))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = compute_features(wav, sr)
        dur = float(len(wav)) / float(sr)
        feat_rows.append(dict(
            clip=clip.name, duration_sec=dur,
            expected_n_audio=int(round(dur * AUDIO_TOK_PER_SEC)),
            sample_rate=int(sr), **feats))
    if fail_count:
        print(f"  [warn] {fail_count} clips failed to load.")
    feats_df = pd.DataFrame(feat_rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Alignment sanity check
    # ------------------------------------------------------------------
    matches = 0; mismatches = 0; first_mm = []
    for i in range(len(feats_df)):
        exp = feats_df.loc[i, "expected_n_audio"]
        got = s24.loc[i, "n_audio"]
        if pd.isna(exp):
            continue
        if abs(int(exp) - int(got)) <= N_AUDIO_TOL:
            matches += 1
        else:
            mismatches += 1
            if len(first_mm) < 5:
                first_mm.append((i, feats_df.loc[i, "clip"],
                                 int(exp), int(got)))
    total = matches + mismatches
    print(f"\nalignment: {matches}/{total} rows match within ±{N_AUDIO_TOL} "
          f"tokens of expected (duration × {AUDIO_TOK_PER_SEC} tok/s)")
    if mismatches:
        print(f"  first {len(first_mm)} mismatches "
              f"(idx, clip, expected, csv): {first_mm}")
    if total > 0 and mismatches / total > ALIGN_FAIL_FRAC:
        raise SystemExit(
            f"ABORT: {mismatches}/{total} alignment mismatches "
            f"(> {ALIGN_FAIL_FRAC*100:.0f}%) — Stage 2.4 clip ordering does "
            f"not match Stage 2.5's reconstruction. Re-run Stage 2.4 and add "
            f"clip_basename to the CSV, or pass --audio_dir / --seed that "
            f"reproduce the original run.")

    # ------------------------------------------------------------------
    # 5. Attach Stage 2.4 targets and drop any failed-feature rows
    # ------------------------------------------------------------------
    feats_df["sink_proportion"] = s24["sink_proportion"].values
    feats_df["n_audio_s24"] = s24["n_audio"].values
    feats_df["n_sink_s24"] = s24["n_sink"].values
    valid = (feats_df[FEATURE_KEYS].notna().all(axis=1)
             & feats_df["sink_proportion"].notna())
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  dropping {n_dropped} rows with NaN features/target.")
    fa = feats_df[valid].reset_index(drop=True)
    if len(fa) < 30:
        raise SystemExit(f"Only {len(fa)} usable clips — refusing to run "
                         "correlation analysis on fewer than 30.")

    # ------------------------------------------------------------------
    # 6. Per-feature variance check (a feature constant across clips can't
    #    explain anything; surface that loudly).
    # ------------------------------------------------------------------
    print(f"\nfeature distributions (n = {len(fa)}):")
    for f in FEATURE_KEYS:
        v = fa[f].values
        cv = v.std() / max(abs(v.mean()), 1e-12)
        print(f"  {f:20s} mean={v.mean():.4g}  std={v.std():.4g}  "
              f"std/mean={cv:.3f}")
        if v.std() <= 1e-9:
            print(f"     [warn] {f} is constant — it can't explain variance.")

    # Save per-clip features + targets.
    fa.to_csv(out_dir / "acoustic_features_per_clip.csv", index=False)
    print(f"\nwrote {out_dir / 'acoustic_features_per_clip.csv'}")

    # ------------------------------------------------------------------
    # 7. Single-feature correlations
    # ------------------------------------------------------------------
    corr_df = correlation_table(fa[FEATURE_KEYS], fa["sink_proportion"].values)
    corr_sorted = (corr_df.assign(abs_r=corr_df["pearson_r"].abs())
                          .sort_values("abs_r", ascending=False)
                          .drop(columns=["abs_r"]))
    corr_sorted.to_csv(out_dir / "correlation_table.csv", index=False)
    print(f"wrote {out_dir / 'correlation_table.csv'}\n")
    print("correlation table (sorted by |pearson_r|):")
    print(corr_sorted.to_string(index=False))

    plot_scatters(fa[FEATURE_KEYS], fa["sink_proportion"].values, corr_df,
                  out_dir / "acoustic_correlations.png")

    # ------------------------------------------------------------------
    # 8. Multivariate OLS
    # ------------------------------------------------------------------
    mv = multivariate_ols(fa[FEATURE_KEYS], fa["sink_proportion"].values)
    pd.DataFrame(mv["coefs"]).assign(
        r2=mv["r2"], n=mv["n"], dof=mv["dof"], intercept=mv["intercept"]
    ).to_csv(out_dir / "multivariate_coefficients.csv", index=False)
    print(f"\nwrote {out_dir / 'multivariate_coefficients.csv'}")
    print(f"multivariate OLS  R² = {mv['r2']:.4f}  "
          f"(n = {mv['n']}, dof = {mv['dof']})")
    for c in mv["coefs"]:
        flag = "★" if c["p"] < 0.01 else " "
        print(f"  {flag} {c['feature']:20s}  β = {c['std_coef']:+.3f}  "
              f"[{c['ci95_lo']:+.3f}, {c['ci95_hi']:+.3f}]  p = {c['p']:.1e}")
    plot_multivariate(mv, out_dir / "multivariate_R2.png")

    # ------------------------------------------------------------------
    # 9. Optional AudioSet labels
    # ------------------------------------------------------------------
    label_summary = None
    if args.labels_csv:
        try:
            label_map = parse_audioset_labels(
                Path(args.labels_csv),
                Path(args.class_labels_csv) if args.class_labels_csv else None)
            # AudioSet wav basenames typically include the YTID — match either
            # exact stem or stem-prefix on '_'.
            labels_per_clip = []
            hits = 0
            for c in clips:
                ytid = c.stem
                lab = label_map.get(ytid)
                if lab is None and "_" in ytid:
                    lab = label_map.get(ytid.split("_")[0])
                labels_per_clip.append(lab)
                if lab is not None:
                    hits += 1
            print(f"\nAudioSet labels matched {hits}/{len(clips)} clips.")
            # Subset to the valid-feature rows.
            valid_idx = feats_df.index[valid]
            labels_for_fa = [labels_per_clip[i] for i in valid_idx]
            label_summary = plot_by_label(
                fa["sink_proportion"].values, labels_for_fa,
                out_dir / "sink_proportion_by_label.png")
        except Exception as e:
            print(f"  [warn] label analysis failed: {e}")
            label_summary = None
    else:
        print("\n  --labels_csv not provided; skipping AudioSet label box-plot.")

    # ------------------------------------------------------------------
    # 10. Verdict
    # ------------------------------------------------------------------
    corr_by_feat = corr_df.set_index("feature")
    abs_energy_r = float(corr_by_feat.loc[ENERGY_FEATS, "pearson_r"].abs().max())
    abs_spec_r = float(corr_by_feat.loc[COMPLEXITY_FEATS, "pearson_r"].abs().max())
    rms_r = float(corr_by_feat.loc["rms_energy", "pearson_r"])
    peak_r = float(corr_by_feat.loc["peak_amplitude", "pearson_r"])
    sil_r = float(corr_by_feat.loc["silence_fraction", "pearson_r"])
    sign_consistent = (rms_r < 0 and peak_r < 0 and sil_r > 0)

    top3 = corr_sorted.head(3)[["feature", "pearson_r", "pearson_p"]] \
                      .to_dict(orient="records")

    if abs_energy_r > ENERGY_R_VERDICT and sign_consistent:
        verdict_label = "CONFIRMED — silence/low-energy clips elicit more sinks"
        verdict = (
            f"{verdict_label}. "
            f"Energy-feature |r| max = {abs_energy_r:.3f}  "
            f"(rms_energy r={rms_r:+.3f}, peak r={peak_r:+.3f}, "
            f"silence_fraction r={sil_r:+.3f}). "
            f"Audio sinks are a 'fill empty space with registers' mechanism: "
            f"content-dependent at the clip level (modulating intensity), "
            f"positionally indifferent within the clip (uniform).")
    elif mv["r2"] < MULTIVAR_R2_LOW:
        verdict_label = "NOT acoustic — cross-clip variance is not explained by acoustic features"
        verdict = (
            f"{verdict_label}. "
            f"Multivariate R² = {mv['r2']:.3f} < {MULTIVAR_R2_LOW:.2f}. "
            f"Audio sink proportion varies but not by acoustic content. "
            f"Likely driver: prompt/text interaction, label-class effects, "
            f"or LLM-side noise. Investigate non-acoustic covariates.")
    elif abs_spec_r > SPEC_R_VERDICT:
        verdict_label = "COMPLEXITY-driven (not loudness)"
        verdict = (
            f"{verdict_label}. "
            f"Best |r| among spectral/complexity features = {abs_spec_r:.3f} "
            f"vs energy features max = {abs_energy_r:.3f}. "
            f"The modulation is by acoustic complexity, not loudness — "
            f"interesting middle ground.")
    else:
        verdict_label = "WEAK — no single feature explains cross-clip variance"
        verdict = (
            f"{verdict_label}. "
            f"Multivariate R² = {mv['r2']:.3f}, energy |r| max = "
            f"{abs_energy_r:.3f}, spectral |r| max = {abs_spec_r:.3f}.")

    print("\n" + "=" * 86)
    print(f"STAGE 2.5 VERDICT  (n_clips = {len(fa)})")
    print("=" * 86)
    print(verdict)
    print(f"\n  top-3 features by |pearson_r|:")
    for t in top3:
        print(f"    {t['feature']:20s} r = {t['pearson_r']:+.3f}  "
              f"p = {t['pearson_p']:.1e}")
    if label_summary is not None:
        print(f"\n  label ANOVA across top-{label_summary['n_categories']} "
              f"AudioSet categories: F = {label_summary['F']:.3f}, "
              f"p = {label_summary['p']:.2e}")

    with open(out_dir / "stage2_5_decision.txt", "w") as f:
        f.write("Stage 2.5 — acoustic correlations on per-clip audio sink "
                "proportion @ L21\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"n_clips analyzed: {len(fa)}\n")
        f.write(f"sink_proportion: mean={fa['sink_proportion'].mean():.3f}, "
                f"std={fa['sink_proportion'].std():.3f}, "
                f"range=[{fa['sink_proportion'].min():.3f}, "
                f"{fa['sink_proportion'].max():.3f}]\n\n")
        f.write("Correlation table (sorted by |pearson_r|):\n")
        f.write(corr_sorted.to_string(index=False) + "\n\n")
        f.write(f"Multivariate OLS  R² = {mv['r2']:.4f}  "
                f"(n={mv['n']}, dof={mv['dof']})\n")
        for c in mv["coefs"]:
            f.write(f"  {c['feature']:20s} β={c['std_coef']:+.3f}  "
                    f"[{c['ci95_lo']:+.3f}, {c['ci95_hi']:+.3f}]  "
                    f"p={c['p']:.1e}\n")
        f.write(f"\nVERDICT: {verdict}\n")
        f.write(f"\nTop-3 features by |pearson_r|:\n")
        for t in top3:
            f.write(f"  {t['feature']:20s} r={t['pearson_r']:+.3f}  "
                    f"p={t['pearson_p']:.1e}\n")
        if label_summary is not None:
            f.write(f"\nLabel ANOVA across top-{label_summary['n_categories']} "
                    f"AudioSet categories: F={label_summary['F']:.3f}, "
                    f"p={label_summary['p']:.2e}\n")
    print(f"\nwrote {out_dir / 'stage2_5_decision.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default=str(_REPO / "data/AudioSet/audios"))
    p.add_argument("--s24_csv",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_4_temporal_sinks/per_clip_temporal_stats.csv"),
                   help="Stage 2.4 per-clip CSV; provides sink_proportion @ L21.")
    p.add_argument("--n_clips", type=int, default=300,
                   help="Must match Stage 2.4's n_clips for ordering alignment.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match Stage 2.4's seed for ordering alignment.")
    p.add_argument("--labels_csv", default=None,
                   help="Optional AudioSet *_segments.csv (eval / balanced_train / "
                        "unbalanced_train) for the per-label box plot.")
    p.add_argument("--class_labels_csv", default=None,
                   help="Optional AudioSet class_labels_indices.csv to map mids "
                        "→ display names (without this, mids are used directly).")
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_5_acoustic_correlations"))
    args = p.parse_args()
    main(args)
