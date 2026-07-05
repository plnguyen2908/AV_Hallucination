"""
stage2_5_supp_audioset_categorical.py

Stage 2.5 supplementary — categorical (AudioSet label) analysis on per-clip
audio sink proportion @ L21.

Stage 2.5/2.5b found that continuous acoustic features explain almost none of
the cross-clip variance (R² ≤ 0.114, no single |r| > 0.20 even with log
transforms). This stage tests the categorical alternative: does sink rate
depend on the audio category itself (Speech vs Music vs Animal …)?

Inputs:
  - Stage 2.4 CSV with sink_proportion @ L21 (per-clip, n=300 AudioSet).
  - data/AudioSet/QA.json — Yes/No questions of the form
    "Does the <X> sound appear in the audio?"  → if label="Yes", X is a
    positive fine-grained AudioSet label for that clip's video_id.
  - data/AudioSet/metadata/ontology.json — AudioSet ontology, used to roll
    fine-grained labels up to their 7 root categories.

Output dir: results/qwen2_5_omni/sink_analysis/stage2_5_supp_audioset/

STEP 1 — Label preprocessing
  A. Roll fine-grained labels up to AudioSet's 7 roots
     ("Human sounds", "Animal", "Music", "Natural sounds", "Sounds of things",
      "Source-ambiguous sounds", "Channel, environment and background").
     Each clip's primary root = root with most fine-grained labels for that
     clip; alphabetical tiebreak.
  B. Top-50 fine-grained labels across the 300 clips.

STEP 2 — Sink proportion by primary root.
STEP 3 — Sink proportion by top-50 fine-grained label (n ≥ 10).
STEP 4 — Multi-hot regression on top-50 labels.

Outputs:
  sink_proportion_by_root.png         box + jitter
  sink_proportion_by_label.png        horizontal bar, top-50
  label_regression_coefficients.png   top +/- coefficients
  audioset_categorical_stats.csv      per-clip: id, primary_root, labels[],
                                      sink_proportion
  label_root_summary.csv              per-root: n, mean, std, median, CI
  label_finegrained_summary.csv       per-label: n, mean, std, CI
  decision.txt                        verdict
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scistats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

# Same conventions as Stage 2.4/2.5.
SEED_DEFAULT = 42
N_CLIPS_DEFAULT = 300

# Verdict thresholds.
ANOVA_P_STRONG = 0.01
ANOVA_P_MOD = 0.05
SPREAD_STRONG = 0.05      # max - min of category means > this → meaningful
SPREAD_NULL = 0.02
N_PER_CAT_MIN = 10        # exclude categories with n<10 from ANOVA / bars

QA_QUESTION_RE = re.compile(
    r"^Does the\s+(.+?)\s+sound appear in the audio\??\s*$",
    flags=re.IGNORECASE)


# ----------------------------------------------------------------------
# Label loading
# ----------------------------------------------------------------------

def load_ontology(path: Path) -> tuple[dict, dict, list]:
    """Returns (by_id, name2id (lower-cased), roots)."""
    with open(path) as f:
        ont = json.load(f)
    by_id = {e["id"]: e for e in ont}
    name2id = {e["name"].lower(): e["id"] for e in ont}
    all_children = set()
    for e in ont:
        all_children.update(e.get("child_ids", []))
    roots = [e for e in ont if e["id"] not in all_children]
    return by_id, name2id, roots


def root_of(label_id: str, by_id: dict, root_ids: set) -> str | None:
    """Walk ontology parents up to find the root for a fine-grained label.
    by_id maps id → entry. root_ids is the set of true-root ids. Returns the
    root's name (or None if label_id is not in the ontology)."""
    if label_id in root_ids:
        return by_id[label_id]["name"]
    # Build parent index lazily on first call (cached on by_id).
    if "_parent_of" not in by_id:
        parent_of = {}
        for e in by_id.values():
            if isinstance(e, dict):
                for c in e.get("child_ids", []):
                    parent_of[c] = e["id"]
        by_id["_parent_of"] = parent_of
    parent_of = by_id["_parent_of"]
    cur = label_id; seen = set()
    while cur in parent_of and cur not in seen:
        seen.add(cur)
        cur = parent_of[cur]
        if cur in root_ids:
            return by_id[cur]["name"]
    return None


def load_audioset_labels_from_qa(qa_path: Path) -> dict[str, set[str]]:
    """Parse QA.json. Returns {video_id (full filename incl '.wav'):
    set of positive label names}. Auto-detects the QA format:
      - describe variant: q['label'] is a LIST of label names (multi-label
        ground truth directly attached). Use the list as-is — no text
        parsing needed.
      - Yes/No variant: q['label'] is a string 'Yes'/'No' and q['text']
        is "Does the X sound appear ...". Extract X from text only when
        the answer is Yes; pre-register the vid for negative-only clips
        so the (none) bucket is well-defined.
    """
    with open(qa_path) as f:
        qa = json.load(f)
    positive = defaultdict(set)
    fmt = None
    for q in qa:
        vid = q.get("video_id")
        labs = q.get("label", None)
        if isinstance(labs, list):
            # describe variant — list of fine-grained label names
            fmt = fmt or "describe"
            positive[vid].update(str(x).strip() for x in labs if str(x).strip())
        else:
            # Yes/No variant
            fmt = fmt or "yesno"
            text = q.get("text", "")
            lab = (labs or "").strip().lower() if isinstance(labs, str) else ""
            m = QA_QUESTION_RE.match(text)
            if not m:
                continue
            x = m.group(1).strip()
            if lab in ("yes", "true", "y"):
                positive[vid].add(x)
            elif lab in ("no", "false", "n"):
                positive.setdefault(vid, set())
    print(f"  QA format auto-detected: {fmt}")
    print(f"  parsed {len(qa)} QA entries → {len(positive)} distinct clips; "
          f"{sum(1 for v in positive.values() if v)} with ≥1 positive label, "
          f"{sum(1 for v in positive.values() if not v)} with empty set")
    return positive


def resolve_root_per_clip(labels_per_clip: dict[str, set[str]],
                          by_id: dict, name2id: dict,
                          root_ids: set) -> dict[str, dict]:
    """For each clip: resolve each label name → ontology id → root.
    Returns {vid: {labels: [...], roots_count: {root_name: n}, primary_root: str}}."""
    out = {}
    unresolved = Counter()
    for vid, labels in labels_per_clip.items():
        roots_count = Counter()
        for name in labels:
            lid = name2id.get(name.lower())
            if lid is None:
                unresolved[name] += 1
                continue
            r = root_of(lid, by_id, root_ids)
            if r:
                roots_count[r] += 1
        if roots_count:
            # Primary root: most labels for this clip, alphabetical tiebreak.
            max_n = max(roots_count.values())
            tied = sorted([k for k, v in roots_count.items() if v == max_n])
            primary = tied[0]
        else:
            primary = None
        out[vid] = dict(labels=sorted(labels),
                        roots_count=dict(roots_count),
                        primary_root=primary)
    if unresolved:
        print(f"  [warn] {sum(unresolved.values())} label name occurrences "
              f"({len(unresolved)} distinct names) couldn't be mapped to the "
              f"AudioSet ontology — top 10:")
        for n, c in unresolved.most_common(10):
            print(f"    {c:5d}  {n!r}")
    return out


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------

def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    h = sem * 1.96
    return (m - h, m + h)


def root_summary(per_clip_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r, g in per_clip_df.groupby("primary_root", dropna=False):
        sp = g["sink_proportion"].values
        lo, hi = ci95(sp)
        rows.append(dict(
            primary_root=("(none)" if r is None or pd.isna(r) else r),
            n=len(sp),
            mean=float(sp.mean()), std=float(sp.std()),
            median=float(np.median(sp)),
            ci95_lo=lo, ci95_hi=hi,
        ))
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def finegrained_summary(per_clip_df: pd.DataFrame, top_k: int = 50,
                        min_n: int = N_PER_CAT_MIN) -> pd.DataFrame:
    """Top-k by occurrence, filter to those with n>=min_n."""
    label_counts = Counter()
    for labs in per_clip_df["labels"]:
        for l in labs:
            label_counts[l] += 1
    top = [l for l, _ in label_counts.most_common(top_k)]
    rows = []
    for l in top:
        mask = per_clip_df["labels"].apply(lambda L: l in L)
        sp = per_clip_df.loc[mask, "sink_proportion"].values
        if len(sp) < min_n:
            continue
        lo, hi = ci95(sp)
        rows.append(dict(label=l, n=int(len(sp)),
                         mean=float(sp.mean()), std=float(sp.std()),
                         ci95_lo=lo, ci95_hi=hi))
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def multilabel_regression(per_clip_df: pd.DataFrame, top_labels: list,
                          y_col: str = "sink_proportion") -> dict:
    """Multi-hot OLS: y ~ Σ β_i · 1[label_i ∈ clip]. Standardized via
    centering of x_i (binary so std-scale is sqrt(p(1-p)))."""
    n = len(per_clip_df)
    p = len(top_labels)
    X = np.zeros((n, p), dtype=np.float64)
    for j, lab in enumerate(top_labels):
        X[:, j] = per_clip_df["labels"].apply(
            lambda L: 1.0 if lab in L else 0.0).values
    y = per_clip_df[y_col].values.astype(np.float64)
    # Drop near-constant columns to avoid singular X.
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    kept_labels = [l for l, k in zip(top_labels, keep) if k]
    X_aug = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
    beta = XtX_inv @ X_aug.T @ y
    y_pred = X_aug @ beta
    rss = float(((y - y_pred) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / max(tss, 1e-12)
    dof = max(n - X_aug.shape[1], 1)
    sigma2 = rss / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    t_stat = beta / np.maximum(se, 1e-12)
    p_val = 2 * (1 - scistats.t.cdf(np.abs(t_stat), dof))
    rows = []
    for i, lab in enumerate(kept_labels):
        rows.append(dict(label=lab,
                         coef=float(beta[i + 1]),
                         se=float(se[i + 1]),
                         t=float(t_stat[i + 1]),
                         p=float(p_val[i + 1])))
    return dict(r2=float(r2), n=int(n), p=int(X.shape[1]), dof=int(dof),
                intercept=float(beta[0]),
                coefs=pd.DataFrame(rows).sort_values("coef", ascending=False))


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_root_box(per_clip_df, root_sum, out_path):
    rs = root_sum[root_sum["n"] >= N_PER_CAT_MIN].sort_values("mean")
    cats = rs["primary_root"].tolist()
    groups = [per_clip_df.loc[per_clip_df["primary_root"] == c,
                              "sink_proportion"].values for c in cats]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bp = ax.boxplot(groups, tick_labels=[f"{c}\n(n={len(g)})"
                                         for c, g in zip(cats, groups)],
                    widths=0.55, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3"); patch.set_edgecolor("#1f77b4")
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        xs = i + 1 + (rng.random(len(g)) - 0.5) * 0.25
        ax.scatter(xs, g, s=8, alpha=0.45, color="#444444",
                   edgecolor="none")
    ax.set_ylabel("sink proportion @ L21", fontsize=11)
    ax.set_xlabel("AudioSet primary root category  "
                  f"(n_clip ≥ {N_PER_CAT_MIN})", fontsize=11)
    ax.set_title("Stage 2.5 supp — sink proportion by AudioSet primary root",
                 fontsize=12)
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_finegrained_bar(fine_sum, out_path):
    df = fine_sum.copy().sort_values("mean")
    fig, ax = plt.subplots(figsize=(11, max(6, 0.22 * len(df) + 1)))
    y = np.arange(len(df))
    err_lo = df["mean"] - df["ci95_lo"]
    err_hi = df["ci95_hi"] - df["mean"]
    bars = ax.barh(y, df["mean"], xerr=[err_lo, err_hi],
                   capsize=2, color="#88aacc", edgecolor="#1f4a72",
                   linewidth=0.5, error_kw=dict(lw=0.6))
    # Highlight 3 highest and 3 lowest.
    for i in list(range(min(3, len(df)))) + list(range(max(0, len(df) - 3),
                                                       len(df))):
        bars[i].set_color("#d62728" if i >= len(df) - 3 else "#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={n})"
                        for l, n in zip(df["label"], df["n"])], fontsize=8)
    ax.set_xlabel("mean sink proportion @ L21  (95% CI)", fontsize=11)
    ax.set_title(f"Stage 2.5 supp — sink proportion by top-{len(df)} "
                 f"AudioSet fine-grained label (n ≥ {N_PER_CAT_MIN})",
                 fontsize=12)
    ax.grid(True, ls=":", alpha=0.4, axis="x")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_regression_coefs(reg, out_path, k: int = 10):
    coefs = reg["coefs"]
    top_pos = coefs.head(k); top_neg = coefs.tail(k).iloc[::-1]
    show = pd.concat([top_pos, top_neg]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(show) + 1)))
    y = np.arange(len(show))
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in show["coef"]]
    ax.barh(y, show["coef"], color=colors, edgecolor="black",
            linewidth=0.4, alpha=0.85)
    ax.errorbar(show["coef"], y, xerr=1.96 * show["se"], fmt="none",
                color="black", capsize=3, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  (p={p:.1e})"
                        for lab, p in zip(show["label"], show["p"])],
                       fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("multi-hot OLS coefficient on sink_proportion @ L21  "
                  "(±1.96·SE)", fontsize=11)
    ax.set_title(f"Stage 2.5 supp — top {k} positive (red) and top {k} "
                 f"negative (green) label effects  (R² = {reg['r2']:.3f}, "
                 f"n = {reg['n']}, p = {reg['p']})", fontsize=11)
    ax.grid(True, ls=":", alpha=0.4, axis="x")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -- Stage 2.4 CSV --
    s24 = pd.read_csv(args.s24_csv)
    if "sink_proportion" not in s24.columns:
        s24["sink_proportion"] = s24["n_sink"] / s24["n_audio"].clip(lower=1)
    print(f"loaded {len(s24)} per-clip rows from {args.s24_csv}")

    # -- reconstruct clip ordering (same seed as Stage 2.4) --
    clip_dir = Path(args.audio_dir)
    clips_all = sorted(clip_dir.glob("*.wav"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(clips_all))[:args.n_clips]
    clips = [clips_all[i] for i in idx]
    n_common = min(len(s24), len(clips))
    s24 = s24.iloc[:n_common].reset_index(drop=True)
    clips = clips[:n_common]
    print(f"reconstructed {len(clips)} clip filenames "
          f"(seed={args.seed}, audio_dir={clip_dir})")

    # -- ontology + QA --
    print("\nloading AudioSet ontology …")
    by_id, name2id, roots = load_ontology(Path(args.ontology))
    root_ids = {r["id"] for r in roots}
    root_names = sorted(r["name"] for r in roots)
    print(f"  {len(by_id)} ontology entries, {len(roots)} roots: {root_names}")

    print("\nparsing QA.json for per-clip positive labels …")
    labels_by_vid = load_audioset_labels_from_qa(Path(args.qa_json))

    # Resolve clip → labels (by .wav filename; QA's video_id ends in '.wav').
    per_clip = []
    no_label = 0
    for i, clip in enumerate(clips):
        vid = clip.name  # e.g. '6Q5N1DfzGj0.wav'
        labs = labels_by_vid.get(vid, set())
        if not labs:
            no_label += 1
        per_clip.append(dict(clip=vid, labels=sorted(labs)))
    pcf = pd.DataFrame(per_clip)
    pcf["sink_proportion"] = s24["sink_proportion"].values
    pcf["n_audio"] = s24["n_audio"].values
    pcf["n_sink"] = s24["n_sink"].values

    print(f"\nclips with ≥1 positive AudioSet label: {len(pcf) - no_label}/{len(pcf)}")
    label_counts = Counter()
    for L in pcf["labels"]:
        label_counts.update(L)
    print(f"top-10 fine-grained labels (clip count):")
    for n, c in label_counts.most_common(10):
        print(f"   {c:5d}  {n}")

    # -- Resolve roots --
    print("\nresolving labels → root categories …")
    label_resolution = resolve_root_per_clip(
        {row["clip"]: set(row["labels"]) for _, row in pcf.iterrows()},
        by_id, name2id, root_ids)
    pcf["primary_root"] = pcf["clip"].map(
        lambda v: label_resolution.get(v, {}).get("primary_root"))

    print("\nclip count by primary root:")
    rc = pcf["primary_root"].fillna("(none)").value_counts()
    for k, v in rc.items():
        print(f"   {v:4d}  {k}")

    # -- per-clip CSV --
    pcf_out = pcf.copy()
    pcf_out["labels"] = pcf_out["labels"].apply(lambda L: "|".join(L))
    pcf_out.to_csv(out_dir / "audioset_categorical_stats.csv", index=False)
    print(f"\nwrote {out_dir / 'audioset_categorical_stats.csv'}")

    # ----- STEP 2 : primary root -----
    root_sum = root_summary(pcf)
    root_sum.to_csv(out_dir / "label_root_summary.csv", index=False)
    print("\nper-root summary:")
    print(root_sum.to_string(index=False))

    # ANOVA on categories with n>=10.  Note that root_summary uses the string
    # "(none)" for the NaN/unlabeled bucket — match it back to NaN here.
    big = root_sum[root_sum["n"] >= N_PER_CAT_MIN]["primary_root"].tolist()
    def _group_mask(r):
        return (pcf["primary_root"].isna() if r == "(none)"
                else pcf["primary_root"] == r)
    groups = [pcf.loc[_group_mask(r), "sink_proportion"].values for r in big]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        F, p_anova = scistats.f_oneway(*groups)
    else:
        F, p_anova = float("nan"), float("nan")
    spread = (max(np.mean(g) for g in groups) - min(np.mean(g) for g in groups)
              if groups else float("nan"))
    print(f"\none-way ANOVA across {len(big)} root cats (n≥{N_PER_CAT_MIN}): "
          f"F = {F:.3f}, p = {p_anova:.3e}; between-cat mean spread = {spread:.4f}")

    # Bonferroni-corrected pairwise Mann-Whitney.
    print("\npairwise Mann-Whitney U (Bonferroni-corrected, n≥10 cats):")
    pairs = []
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            u, p_mw = scistats.mannwhitneyu(groups[i], groups[j],
                                             alternative="two-sided")
            pairs.append(dict(a=big[i], b=big[j], u=float(u), p=float(p_mw)))
    m_pairs = max(len(pairs), 1)
    for d in pairs:
        d["p_bonf"] = min(1.0, d["p"] * m_pairs)
    pairs_df = pd.DataFrame(pairs).sort_values("p")
    pairs_df.to_csv(out_dir / "pairwise_mannwhitney_roots.csv", index=False)
    if not pairs_df.empty:
        print(pairs_df.to_string(index=False))

    plot_root_box(pcf, root_sum, out_dir / "sink_proportion_by_root.png")

    # ----- STEP 3 : top-50 fine-grained -----
    fine_sum = finegrained_summary(pcf, top_k=50, min_n=N_PER_CAT_MIN)
    fine_sum.to_csv(out_dir / "label_finegrained_summary.csv", index=False)
    print(f"\ntop-50 fine-grained (n≥{N_PER_CAT_MIN}) → {len(fine_sum)} pass filter.")
    if not fine_sum.empty:
        print("3 highest mean sink_proportion:")
        print(fine_sum.tail(3).to_string(index=False))
        print("3 lowest:")
        print(fine_sum.head(3).to_string(index=False))
        plot_finegrained_bar(fine_sum, out_dir / "sink_proportion_by_label.png")
    else:
        print("  (no fine-grained labels pass the n≥10 filter — skipping bar)")

    # ----- STEP 4 : multi-hot regression -----
    top_labels = [l for l, _ in label_counts.most_common(50)]
    reg = multilabel_regression(pcf, top_labels)
    reg["coefs"].assign(r2=reg["r2"], n=reg["n"]).to_csv(
        out_dir / "label_regression_coefficients.csv", index=False)
    print(f"\nmulti-hot OLS  R² = {reg['r2']:.4f}  "
          f"(n={reg['n']}, p={reg['p']}, dof={reg['dof']})")
    print("top 5 positive:")
    print(reg["coefs"].head(5).to_string(index=False))
    print("top 5 negative:")
    print(reg["coefs"].tail(5).iloc[::-1].to_string(index=False))
    plot_regression_coefs(reg, out_dir / "label_regression_coefficients.png",
                          k=10)

    # ----- Verdict -----
    if np.isfinite(p_anova) and p_anova < ANOVA_P_STRONG and spread > SPREAD_STRONG:
        verdict_label = "CATEGORICAL CONTENT MATTERS"
        verdict = (f"{verdict_label} — ANOVA p = {p_anova:.2e} < "
                   f"{ANOVA_P_STRONG} and between-category mean spread = "
                   f"{spread:.3f} > {SPREAD_STRONG}. Audio sink proportion "
                   f"depends on AudioSet category. Stage 2.5's 'acoustic-content-"
                   f"independent' framing should be updated: the cross-clip "
                   f"variance is at least partly explained by SEMANTIC class, "
                   f"not raw acoustic features.")
    elif np.isfinite(p_anova) and p_anova > ANOVA_P_MOD and spread <= SPREAD_NULL:
        verdict_label = "NO CATEGORICAL EFFECT"
        verdict = (f"{verdict_label} — ANOVA p = {p_anova:.2e} > {ANOVA_P_MOD} "
                   f"and between-category mean spread = {spread:.3f} ≤ "
                   f"{SPREAD_NULL}. Saturation interpretation locks in: the "
                   f"cross-clip variance is neither acoustic (Stage 2.5) nor "
                   f"categorical.")
    else:
        verdict_label = "WEAK CATEGORICAL EFFECT"
        verdict = (f"{verdict_label} — ANOVA p = {p_anova:.2e}, spread = "
                   f"{spread:.3f}. Some category-level structure but the "
                   f"effect is small. Report honestly; do not overclaim.")

    with open(out_dir / "decision.txt", "w") as f:
        f.write("Stage 2.5 supp — AudioSet categorical analysis on per-clip "
                "audio sink proportion @ L21\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"n_clips: {len(pcf)}, "
                f"n_with_label: {len(pcf) - no_label}, "
                f"sink_proportion mean = {pcf['sink_proportion'].mean():.3f}, "
                f"std = {pcf['sink_proportion'].std():.3f}\n\n")
        f.write("Per-primary-root summary:\n")
        f.write(root_sum.to_string(index=False) + "\n\n")
        f.write(f"One-way ANOVA across n≥{N_PER_CAT_MIN} primary roots: "
                f"F = {F:.3f}, p = {p_anova:.3e}, "
                f"between-cat mean spread = {spread:.4f}\n\n")
        f.write(f"Top fine-grained results (n_pass = {len(fine_sum)}):\n")
        if not fine_sum.empty:
            f.write("  3 highest mean sink_proportion:\n")
            f.write(fine_sum.tail(3).to_string(index=False) + "\n")
            f.write("  3 lowest mean sink_proportion:\n")
            f.write(fine_sum.head(3).to_string(index=False) + "\n\n")
        f.write(f"Multi-hot label OLS: R² = {reg['r2']:.4f}, "
                f"n = {reg['n']}, p = {reg['p']}\n\n")
        f.write(f"VERDICT: {verdict}\n")
    print(f"\nwrote {out_dir / 'decision.txt'}")
    print("\n" + "=" * 86)
    print(f"STAGE 2.5 SUPP VERDICT  (n_clips = {len(pcf)})")
    print("=" * 86)
    print(verdict)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default=str(_REPO / "data/AudioSet/audios"))
    p.add_argument("--qa_json", default=str(_REPO / "data/AudioSet/QA.json"))
    p.add_argument("--ontology",
                   default=str(_REPO / "data/AudioSet/metadata/ontology.json"))
    p.add_argument("--s24_csv",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_4_temporal_sinks/per_clip_temporal_stats.csv"))
    p.add_argument("--n_clips", type=int, default=N_CLIPS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output_dir",
                   default=str(_REPO / "results/qwen2_5_omni/sink_analysis/"
                               "stage2_5_supp_audioset"))
    args = p.parse_args()
    main(args)
