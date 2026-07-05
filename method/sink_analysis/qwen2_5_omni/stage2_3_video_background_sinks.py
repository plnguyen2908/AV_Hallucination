"""
stage2_3_video_background_sinks.py

Stage 2.3 — Do VIDEO attention sinks at L2 concentrate on BACKGROUND
(non-object) image regions, AFTER controlling for the strong positional
confound Stage 2.2 found (edge_frac = 1.0; ~58% of L2 sink mass in row 0)?

The scientific prediction from the visual-sink literature is that sinks land
on background patches. Stage 2.2 makes that claim ambiguous: frame tops are
disproportionately background (sky, blur, ceilings), so a raw "sinks overlap
SAM-background" number is confounded — it could be 100% explained by position.
This stage separates SEMANTIC preference for background from POSITIONAL
coincidence with the edge prior.

Inputs:
  - per_clip_sink_coords.npz from Stage 2.2 (CSR per-clip layout for L2).
  - VGGSounder .mp4 files (same dataset as Stage 2.1 / 2.2).
  - SAM 3 from /nobackup2/jaden/sam3btr/ + checkpoint /nobackup2/jaden/checkpoints/sam3.pt.
  - Text prompt: "object" (class-agnostic; foreground = union of returned masks).

Pipeline per clip (layer = L2):
  1. Replay the model's frame sampling: decord -> total_frames, fps; replicate
     qwen_omni_utils.v2_5.vision_process.smart_nframes(fps=1.0); compute
     idx = linspace(0, total_frames-1, nframes).round().long(). Token-frame
     f ∈ [0, T_eff) merges source frames idx[2f] and idx[2f+1] (temporal_patch_size=2).
  2. Run SAM 3 on each token-frame's source frames; union all returned
     instance masks → foreground; complement → background. Average-pool the
     pixel-level background mask to the clip's H_eff x W_eff grid via
     cv2.INTER_AREA, giving per-cell background fraction in [0, 1].
  3. obs_bg  = mean bg-fraction over SINK cells (one cell per (f, r, c) sink).
     base_bg = mean bg-fraction over ALL cells of the clip.
     raw_lift = obs_bg - base_bg.

Aggregate across clips:
  - raw_lift reported as the CLIP MEAN (not pooled) so big clips don't dominate.
  - PRIMARY control — pooled logistic regression over every cell of every clip
    (n_cells ~ 400K):
        is_bg ~ is_sink + row_norm + col_norm
    Report beta(is_sink), odds ratio, p. This is the real test: does sink-hood
    predict background AFTER position is partialled out?
    Clip fixed effects are an option (--clip-fe) but default off — 299 dummies
    on ~400K cells is slow; clip-demeaning is the lighter alternative when needed.
  - SECONDARY corroboration — interior-only: drop sinks within 2 cells of any
    edge, re-compute lift, run a Welch t-test on interior sink-cells vs interior
    non-sink-cells. EXPECT THIS TO BE UNDERPOWERED (most sinks are edge tokens).
    Treat as supporting evidence, NOT a co-equal gate.

Verdict gate:
  SEMANTIC           : raw_lift > 0 AND logit beta(is_sink) > 0 with p < 0.01
                       AND interior test agrees (positive lift). The visual-sink
                       prediction is supported.
  POSITIONAL ARTIFACT: raw_lift > 0 but logit shrinks to non-significant after
                       partialling out row/col. The "background" story IS the
                       edge prior — frame this as a substantive finding (it's a
                       border story, not a failure).
  NULL               : no raw lift at all.

--dry-run synthesizes a background field with a mild top=background bias and
runs the same pipeline without frames or SAM, to smoke-test the plumbing.

Outputs (--output_dir):
  stage2_3_decision.txt     verdict + numbers (matches other stage formats)
  stage2_3_per_clip.csv     per-clip: clip, n_sink, obs_bg, base_bg, lift, H, W, T
  sink_vs_nonsink_by_row.png  background rate by row band, sink vs non-sink
"""

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats
from tqdm import tqdm


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent

# ----------------------------------------------------------------------
# Constants — matched to Stage 2.2 / model config
# ----------------------------------------------------------------------

# Qwen2.5-Omni vision tokenization (spatial_merge_size=2, temporal_patch_size=2,
# patch_size=14). Frame-sampling defaults from build_conversation +
# qwen_omni_utils.v2_5.vision_process.
QWEN_VIDEO_FPS = 1.0           # build_conversation(...).VIDEO_FPS_DEFAULT
QWEN_FRAME_FACTOR = 2          # temporal_patch_size (smart_nframes rounds to multiple)
QWEN_FPS_MIN_FRAMES = 4
QWEN_FPS_MAX_FRAMES = 768

LAYER = 2                      # Stage 2.2 finding: L2 is the spatially-structured layer
# Multi-prompt SAM 3 union — generic text prompts like "object"/"anything"
# return uniformly low confidence in this SAM 3 checkpoint (top score 0.16-0.30
# instead of 0.93-0.98 for specific class names). Use a small VGGSounder-tuned
# vocabulary, then union all per-prompt masks above DETECTION_THRESHOLD as the
# foreground. All prompts batch into ONE datapoint per frame → 1 forward/frame.
DEFAULT_PROMPTS = (
    "person", "face", "hand",
    "animal", "dog", "cat", "bird",
    "vehicle", "musical instrument", "guitar", "drum",
    "screen",
)
DEFAULT_DET_THR = 0.30         # per-prompt detection score threshold
MAX_DETS_PER_PROMPT = 10       # cap per-prompt mask union (long-tail noise guard)
EDGE_CELLS = 2                 # interior-only filter: drop sinks within 2 cells of any edge

DEFAULT_NPZ = (_REPO / "results/qwen2_5_omni/sink_analysis/stage2_2_spatial/"
               "per_clip_sink_coords.npz")
DEFAULT_VIDEO_DIR = _REPO / "data/VGGSounder/videos"
DEFAULT_OUT = _REPO / "results/qwen2_5_omni/sink_analysis/stage2_3_background"

# SAM 3 (colleague's clone)
SAM3_ROOT = Path("/nobackup2/jaden/sam3btr")
SAM3_BPE = SAM3_ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
SAM3_CKPT = Path("/nobackup2/jaden/checkpoints/sam3.pt")


# ----------------------------------------------------------------------
# Qwen frame-sampling replica (vision_process.smart_nframes + linspace)
# ----------------------------------------------------------------------

def _round_by_factor(n, f):
    return round(n / f) * f


def _floor_by_factor(n, f):
    return (n // f) * f


def _ceil_by_factor(n, f):
    return math.ceil(n / f) * f


def smart_nframes_replica(total_frames: int, video_fps: float,
                          fps: float = QWEN_VIDEO_FPS,
                          frame_factor: int = QWEN_FRAME_FACTOR,
                          min_frames: int = QWEN_FPS_MIN_FRAMES,
                          max_frames: int = QWEN_FPS_MAX_FRAMES) -> int:
    """Replicates qwen_omni_utils.v2_5.vision_process.smart_nframes() for the
    fps-derived branch (no explicit `nframes` argument in build_conversation).

    nframes = total_frames / video_fps * fps, then clamped to
    [ceil_by_factor(min_frames), floor_by_factor(min(max_frames, total_frames))]
    and finally floor_by_factor'd to be a multiple of frame_factor.
    """
    nframes = total_frames / max(video_fps, 1e-9) * fps
    min_f = _ceil_by_factor(min_frames, frame_factor)
    max_f = _floor_by_factor(min(max_frames, total_frames), frame_factor)
    nframes = min(max(nframes, min_f), max_f)
    nframes = _floor_by_factor(nframes, frame_factor)
    if not (frame_factor <= nframes <= total_frames):
        raise ValueError(
            f"computed nframes={nframes} out of range "
            f"[{frame_factor}, {total_frames}]")
    return int(nframes)


def model_frame_indices(video_path: str):
    """Return the EXACT source-frame indices the Qwen2.5-Omni model would have
    seen for `video_path` under build_conversation defaults (fps=1, fully
    decoded, no time crop). Returns (idx_list, total_frames, video_fps)."""
    import decord
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    nframes = smart_nframes_replica(total_frames, video_fps)
    import torch
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    return idx, total_frames, video_fps, nframes, vr


# ----------------------------------------------------------------------
# SAM 3 wrapper (lazy import — only loaded when --dry-run is OFF)
# ----------------------------------------------------------------------

class Sam3BackgroundExtractor:
    """Loads SAM 3 once; for each PIL image, returns a binary background mask
    at the original image resolution.

    foreground = union of all instance masks returned for the "object" query.
    background = 1 - foreground.
    """

    def __init__(self, bpe_path=SAM3_BPE, checkpoint_path=SAM3_CKPT,
                 prompts=DEFAULT_PROMPTS,
                 detection_threshold=DEFAULT_DET_THR,
                 max_per_prompt=MAX_DETS_PER_PROMPT):
        sys.path.insert(0, str(SAM3_ROOT))
        import torch
        # bf16 throughout, matching the notebook setup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self._autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self._autocast.__enter__()

        from sam3 import build_sam3_image_model
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata)
        from sam3.train.data.collator import collate_fn_api as collate_fn
        from sam3.model.utils.misc import copy_data_to_device

        # Monkey-patch: data_misc dataclasses (BatchedInferenceMetadata,
        # BatchedDatapoint) gained new required fields after the
        # `collate_fn_api` helper was last updated. The newer `_per_instance`
        # collator handles them; this one doesn't. Wrap each __init__ to
        # supply empty-list defaults for missing kwargs, so the unupdated
        # path still constructs without touching the upstream files.
        from sam3.model import data_misc as _dm
        for _cls_name, _missing_kw in (
            ("BatchedInferenceMetadata", ("ann_id", "is_semantic")),
            ("BatchedDatapoint", ("find_is_semantic_batch",
                                   "find_noun_phrases_batch")),
        ):
            _cls = getattr(_dm, _cls_name, None)
            if _cls is None: continue
            _orig = _cls.__init__
            def _make_patched(orig, missing):
                def _patched(self, *a, **kw):
                    for k in missing:
                        kw.setdefault(k, [])
                    orig(self, *a, **kw)
                return _patched
            _cls.__init__ = _make_patched(_orig, _missing_kw)

        print(f"[SAM3] building model from {checkpoint_path} ...")
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"[SAM3] ready (prompts={list(prompts)}, "
              f"threshold={detection_threshold}, max_per_prompt={max_per_prompt})")

        self.prompts = list(prompts)
        self.max_per_prompt = int(max_per_prompt)
        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True,
                            consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # detection_threshold=-1 on the postprocessor → keep ALL detections;
        # we do per-prompt filtering ourselves so we can apply the threshold
        # per query-prompt without losing low-but-valid ones.
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=-1.0,
            to_cpu=True,
        )
        self.detection_threshold = float(detection_threshold)
        self._Datapoint = Datapoint
        self._SAMImage = SAMImage
        self._FindQueryLoaded = FindQueryLoaded
        self._InferenceMetadata = InferenceMetadata
        self._collate = collate_fn
        self._to_device = copy_data_to_device
        self._torch = torch
        self._global_counter = 1

    def _make_datapoint(self, pil_image):
        """Build one Datapoint carrying ALL prompts on this image, so a single
        forward processes all queries jointly. Returns (dp, list_of_qids in
        the same order as self.prompts)."""
        w, h = pil_image.size
        dp = self._Datapoint(find_queries=[], images=[])
        dp.images = [self._SAMImage(data=pil_image, objects=[], size=[h, w])]
        qids = []
        for prompt in self.prompts:
            dp.find_queries.append(self._FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=self._InferenceMetadata(
                    coco_image_id=self._global_counter,
                    original_image_id=self._global_counter,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )))
            qids.append(self._global_counter)
            self._global_counter += 1
        return dp, qids

    def background_mask(self, pil_image) -> np.ndarray:
        """Returns a binary numpy mask of shape (H_orig, W_orig); 1 = background.

        For each prompt, take detections with score >= detection_threshold
        (capped at max_per_prompt to avoid long-tail noise). Union all kept
        masks across all prompts → foreground; complement → background."""
        w_orig, h_orig = pil_image.size
        dp, qids = self._make_datapoint(pil_image)
        dp = self.transform(dp)
        batch = self._collate([dp], dict_key="dummy")["dummy"]
        batch = self._to_device(batch, self._torch.device("cuda"),
                                non_blocking=True)
        with self._torch.inference_mode():
            output = self.model(batch)
        processed = self.postprocessor.process_results(
            output, batch.find_metadatas)

        fg = np.zeros((h_orig, w_orig), dtype=bool)
        for qid in qids:
            result = processed.get(qid, None)
            if result is None: continue
            scores = result.get("scores", None)
            masks = result.get("masks", None)
            if scores is None or masks is None or len(scores) == 0:
                continue
            if hasattr(scores, "cpu"):
                # bf16 (from autocast) doesn't convert to numpy directly
                scores = scores.float().cpu().numpy()
                masks = masks.float().cpu().numpy()
            else:
                scores = np.asarray(scores, dtype=np.float32)
                masks = np.asarray(masks, dtype=np.float32)
            keep = scores >= self.detection_threshold
            if not keep.any():
                continue
            # Top-k among the surviving detections
            keep_idx = np.where(keep)[0]
            order = keep_idx[np.argsort(-scores[keep_idx])[: self.max_per_prompt]]
            if masks.ndim == 4:
                masks = masks[:, 0]
            for mi in order:
                m = (masks[mi] > 0.5).astype(bool)
                if m.shape != (h_orig, w_orig):
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (w_orig, h_orig),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                fg |= m
        return (~fg).astype(np.float32)


# ----------------------------------------------------------------------
# Per-clip background extraction
# ----------------------------------------------------------------------

def avg_pool_mask_to_grid(bg_pixel: np.ndarray, H_eff: int, W_eff: int) -> np.ndarray:
    """Average-pool a binary background mask (H_orig × W_orig, values in {0,1})
    down to a per-cell background fraction grid of shape (H_eff, W_eff)."""
    import cv2
    # cv2.resize wants (width, height) and INTER_AREA averages over source patches
    return cv2.resize(bg_pixel.astype(np.float32), (W_eff, H_eff),
                      interpolation=cv2.INTER_AREA)


def synth_bg_field(H_eff, W_eff, top_bias=0.4, seed=0):
    """Smoke-test synthetic background: row 0 ≈ 0.8 bg, row H-1 ≈ 0.2 bg,
    with mild per-cell noise. Returns (H_eff, W_eff) array in [0, 1]."""
    rng = np.random.default_rng(seed)
    row_idx = np.arange(H_eff)[:, None]
    base = 0.8 - (row_idx / max(H_eff - 1, 1)) * top_bias * 1.5  # 0.8 → ~0.2
    base = np.broadcast_to(base, (H_eff, W_eff)).copy()
    base += rng.normal(0, 0.05, size=(H_eff, W_eff))
    return np.clip(base, 0.0, 1.0)


def process_clip_real(clip_path, T_eff, H_eff, W_eff,
                      sinks_frcc, sam: Sam3BackgroundExtractor):
    """Returns bg_grid of shape (T_eff, H_eff, W_eff) in [0, 1]. Uses real SAM
    on the source frames the Qwen model saw.

    For each token-frame f, averages the SAM background masks of source frames
    idx[2f] and idx[2f+1] (the 2 frames merged by temporal_patch_size=2)."""
    from PIL import Image as PILImage
    idx, total_frames, vfps, nframes, vr = model_frame_indices(str(clip_path))
    if nframes // QWEN_FRAME_FACTOR != T_eff:
        raise RuntimeError(
            f"frame-sampling drift: T_eff from npz = {T_eff}, "
            f"replica nframes/2 = {nframes // QWEN_FRAME_FACTOR}  "
            f"(total_frames={total_frames}, fps={vfps:.3f}, nframes={nframes})")

    bg_grid = np.zeros((T_eff, H_eff, W_eff), dtype=np.float32)
    for f in range(T_eff):
        src_idx_a, src_idx_b = idx[2 * f], idx[2 * f + 1]
        # decord get_batch returns (n, H, W, C) numpy with values in [0, 255]
        frames = vr.get_batch([src_idx_a, src_idx_b]).asnumpy()
        masks = []
        for fi in range(frames.shape[0]):
            pil = PILImage.fromarray(frames[fi].astype(np.uint8))
            bg = sam.background_mask(pil)              # (H_orig, W_orig)
            masks.append(avg_pool_mask_to_grid(bg, H_eff, W_eff))
        bg_grid[f] = np.mean(masks, axis=0)
    return bg_grid


def process_clip_dry(T_eff, H_eff, W_eff, seed=0):
    """Synthetic version of process_clip_real — same per-token-frame field,
    no actual frame I/O or SAM."""
    rng = np.random.default_rng(seed)
    bg = np.zeros((T_eff, H_eff, W_eff), dtype=np.float32)
    for f in range(T_eff):
        bg[f] = synth_bg_field(H_eff, W_eff, seed=int(rng.integers(1 << 30)))
    return bg


# ----------------------------------------------------------------------
# Aggregation + statistics
# ----------------------------------------------------------------------

def per_clip_metrics(bg_grid, frames, rows, cols):
    """Per-clip raw obs_bg / base_bg / lift, plus the cell-level (is_sink,
    is_bg, row_norm, col_norm) records used downstream by the pooled logit."""
    T, H, W = bg_grid.shape
    # cell-level long-form arrays
    n_cells = T * H * W
    bg_long = bg_grid.reshape(-1)
    # one (is_sink) flag per cell
    is_sink_flat = np.zeros(n_cells, dtype=bool)
    for f, r, c in zip(frames, rows, cols):
        if 0 <= f < T and 0 <= r < H and 0 <= c < W:
            is_sink_flat[f * (H * W) + r * W + c] = True
    obs_bg = float(bg_long[is_sink_flat].mean()) if is_sink_flat.any() else float("nan")
    base_bg = float(bg_long.mean())
    lift = obs_bg - base_bg

    # row_norm and col_norm with cell centers (matches Stage 2.2 normalization)
    row_centers = (np.arange(H) + 0.5) / H
    col_centers = (np.arange(W) + 0.5) / W
    rr, cc = np.meshgrid(row_centers, col_centers, indexing="ij")
    rr_long = np.broadcast_to(rr, (T, H, W)).reshape(-1)
    cc_long = np.broadcast_to(cc, (T, H, W)).reshape(-1)
    return dict(obs_bg=obs_bg, base_bg=base_bg, lift=lift,
                is_sink=is_sink_flat, bg=bg_long.astype(np.float32),
                row_norm=rr_long.astype(np.float32),
                col_norm=cc_long.astype(np.float32))


def pooled_logit(is_sink, is_bg, row_norm, col_norm):
    """is_bg ~ is_sink + row_norm + col_norm. Returns beta, SE, z, p, OR for
    the is_sink coefficient. Uses statsmodels if available, else falls back to
    scipy point-and-shoot Newton-Raphson."""
    try:
        import statsmodels.api as sm
        X = np.column_stack([is_sink.astype(np.float64),
                              row_norm.astype(np.float64),
                              col_norm.astype(np.float64)])
        X = sm.add_constant(X)
        model = sm.GLM(is_bg.astype(np.float64), X,
                       family=sm.families.Binomial())
        res = model.fit(disp=False)
        # coefficients order: [const, is_sink, row_norm, col_norm]
        beta = float(res.params[1])
        se = float(res.bse[1])
        z = float(res.tvalues[1])
        p = float(res.pvalues[1])
        return dict(beta=beta, se=se, z=z, p=p, odds_ratio=float(np.exp(beta)),
                    n=int(len(is_bg)), backend="statsmodels.GLM(Binomial)")
    except ImportError:
        pass
    # Fallback: Newton-Raphson logistic regression (no covariate adjustment SE)
    from scipy.optimize import minimize
    X = np.column_stack([np.ones_like(is_sink, dtype=np.float64),
                          is_sink.astype(np.float64),
                          row_norm.astype(np.float64),
                          col_norm.astype(np.float64)])
    y = is_bg.astype(np.float64)
    def nll(b):
        z = X @ b
        # logistic loss
        return float(np.mean(np.log1p(np.exp(-y * z + (1 - y) * z)) + (1 - y) * z))
    res = minimize(nll, np.zeros(4), method="L-BFGS-B")
    beta = float(res.x[1])
    return dict(beta=beta, se=float("nan"), z=float("nan"), p=float("nan"),
                odds_ratio=float(np.exp(beta)), n=int(len(is_bg)),
                backend="scipy.L-BFGS-B (no SE/p)")


def interior_test(records, H_arr, W_arr):
    """Drop cells within EDGE_CELLS of any edge; report obs_bg vs nonsink_bg
    on the interior, plus a Welch t-test."""
    int_sink, int_nonsink = [], []
    for rec, H, W in zip(records, H_arr, W_arr):
        T = int(rec["bg"].size // (H * W))
        bg = rec["bg"].reshape(T, H, W)
        # interior mask per (r, c) — independent of f
        rrange = np.arange(H)
        crange = np.arange(W)
        rr, cc = np.meshgrid(rrange, crange, indexing="ij")
        interior_rc = ((rr >= EDGE_CELLS) & (rr < H - EDGE_CELLS) &
                        (cc >= EDGE_CELLS) & (cc < W - EDGE_CELLS))
        is_sink = rec["is_sink"].reshape(T, H, W)
        mask = np.broadcast_to(interior_rc, (T, H, W))
        bg_int = bg[mask]
        sk_int = is_sink[mask]
        if sk_int.any():
            int_sink.extend(bg_int[sk_int].tolist())
        if (~sk_int).any():
            int_nonsink.extend(bg_int[~sk_int].tolist())
    int_sink = np.asarray(int_sink); int_nonsink = np.asarray(int_nonsink)
    if int_sink.size < 5 or int_nonsink.size < 5:
        return dict(n_int_sink=int(int_sink.size),
                    n_int_nonsink=int(int_nonsink.size),
                    sink_mean=float("nan"), nonsink_mean=float("nan"),
                    lift=float("nan"), t=float("nan"), p=float("nan"))
    sm = float(int_sink.mean()); nm = float(int_nonsink.mean())
    t, p = scistats.ttest_ind(int_sink, int_nonsink, equal_var=False)
    return dict(n_int_sink=int(int_sink.size),
                n_int_nonsink=int(int_nonsink.size),
                sink_mean=sm, nonsink_mean=nm, lift=sm - nm,
                t=float(t), p=float(p))


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_bg_by_row(records, H_arr, out_path, n_bands=6):
    import matplotlib.pyplot as plt
    band_sink = [[] for _ in range(n_bands)]
    band_nonsink = [[] for _ in range(n_bands)]
    for rec, H in zip(records, H_arr):
        # row_norm is already per-cell in [0, 1], so band assignment is direct
        rn = rec["row_norm"]; bg = rec["bg"]; sk = rec["is_sink"]
        band_idx = np.minimum((rn * n_bands).astype(int), n_bands - 1)
        for b in range(n_bands):
            m = band_idx == b
            if not m.any(): continue
            band_sink[b].extend(bg[m & sk].tolist())
            band_nonsink[b].extend(bg[m & ~sk].tolist())
    means_sink = [np.mean(b) if b else float("nan") for b in band_sink]
    means_nonsink = [np.mean(b) if b else float("nan") for b in band_nonsink]
    counts_sink = [len(b) for b in band_sink]
    counts_nonsink = [len(b) for b in band_nonsink]
    band_centers = (np.arange(n_bands) + 0.5) / n_bands

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(band_centers, means_nonsink, marker="o", lw=2, color="#1f77b4",
            label=f"non-sink cells (n={sum(counts_nonsink):,})")
    ax.plot(band_centers, means_sink, marker="s", lw=2, color="#d62728",
            label=f"sink cells (n={sum(counts_sink):,})")
    for i, (c, m) in enumerate(zip(counts_sink, means_sink)):
        ax.annotate(f"n={c}", (band_centers[i], m), fontsize=8,
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", color="#d62728")
    ax.set_xlabel(f"row band (normalized; 0 = top of frame, {n_bands} bands)",
                  fontsize=11)
    ax.set_ylabel("mean background fraction", fontsize=11)
    ax.set_title(f"Stage 2.3 — background rate by row band, "
                 "sink vs non-sink cells (layer L2)", fontsize=12)
    ax.set_ylim(0, 1.02); ax.grid(True, ls=":", alpha=0.4); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Stage 2.2 npz (L2 sinks) ---
    print(f"loading sink coords from {args.npz}")
    z = np.load(args.npz, allow_pickle=True)
    clips_all = z["L2_clips"]
    offs = z["L2_offsets"]
    rows_flat = z["L2_rows_flat"]
    cols_flat = z["L2_cols_flat"]
    frames_flat = z["L2_frames_flat"]
    H_arr_all = z["L2_H_eff"].astype(int)
    W_arr_all = z["L2_W_eff"].astype(int)
    T_arr_all = z["L2_T_eff"].astype(int)
    n_sink_all = z["L2_n_sink"].astype(int)
    n_clips_total = len(clips_all)
    assert offs[0] == 0 and offs[-1] == len(rows_flat)
    assert np.array_equal(np.diff(offs), n_sink_all)
    print(f"  {n_clips_total} clips, {int(n_sink_all.sum())} total sinks at L2")

    # --- Optional subset for fast iteration ---
    if args.limit and args.limit > 0:
        n_clips_total = min(n_clips_total, args.limit)
        clips_all = clips_all[:n_clips_total]
        H_arr_all = H_arr_all[:n_clips_total]
        W_arr_all = W_arr_all[:n_clips_total]
        T_arr_all = T_arr_all[:n_clips_total]
        n_sink_all = n_sink_all[:n_clips_total]
        print(f"  --limit {args.limit} → processing first {n_clips_total} clips")

    # --- Build SAM 3 (skip in dry-run) ---
    sam = None
    prompts_used = list(DEFAULT_PROMPTS)
    if args.prompts:
        prompts_used = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not args.dry_run:
        sam = Sam3BackgroundExtractor(prompts=prompts_used,
                                       detection_threshold=args.detection_threshold,
                                       max_per_prompt=args.max_per_prompt)

    # --- Per-clip loop ---
    per_clip_rows = []
    records = []
    H_used, W_used = [], []
    failures = {}
    video_dir = Path(args.video_dir)
    t_start = time.time()
    for i in tqdm(range(n_clips_total), desc="clips"):
        clip = str(clips_all[i])
        H_eff, W_eff, T_eff = int(H_arr_all[i]), int(W_arr_all[i]), int(T_arr_all[i])
        n_sink = int(n_sink_all[i])
        frames = frames_flat[offs[i]:offs[i + 1]]
        rows = rows_flat[offs[i]:offs[i + 1]]
        cols = cols_flat[offs[i]:offs[i + 1]]
        try:
            if args.dry_run:
                bg_grid = process_clip_dry(T_eff, H_eff, W_eff, seed=i)
            else:
                clip_path = video_dir / clip
                if not clip_path.is_file():
                    raise FileNotFoundError(str(clip_path))
                bg_grid = process_clip_real(clip_path, T_eff, H_eff, W_eff,
                                             None, sam)
        except Exception as e:
            failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
            tqdm.write(f"  [skip] {clip}: {type(e).__name__}: {e}")
            continue
        rec = per_clip_metrics(bg_grid, frames, rows, cols)
        rec["clip"] = clip
        records.append(rec)
        H_used.append(H_eff); W_used.append(W_eff)
        per_clip_rows.append(dict(
            clip=clip, n_sink=n_sink,
            obs_bg=rec["obs_bg"], base_bg=rec["base_bg"], lift=rec["lift"],
            H=H_eff, W=W_eff, T=T_eff))
    elapsed = time.time() - t_start
    print(f"\nprocessed {len(records)}/{n_clips_total} clips in {elapsed:.1f} s "
          f"({elapsed/max(len(records),1):.2f} s/clip)")
    if failures:
        print(f"  failures: {failures}")
    if not records:
        raise SystemExit("no clips processed — cannot continue")

    # --- Write per-clip CSV ---
    pcdf = pd.DataFrame(per_clip_rows)
    pcdf.to_csv(out_dir / "stage2_3_per_clip.csv", index=False)
    print(f"wrote {out_dir / 'stage2_3_per_clip.csv'}")

    # --- CLIP-MEAN raw_lift (not pooled) ---
    raw_lifts = np.asarray([r["lift"] for r in records], dtype=float)
    raw_lifts = raw_lifts[np.isfinite(raw_lifts)]
    raw_mean_lift = float(raw_lifts.mean())
    raw_lift_se = float(raw_lifts.std(ddof=1) / max(np.sqrt(len(raw_lifts)), 1))
    raw_lift_t, raw_lift_p = scistats.ttest_1samp(raw_lifts, popmean=0.0)
    print(f"\nclip-mean raw_lift = {raw_mean_lift:+.4f}  "
          f"(SE {raw_lift_se:.4f}, t = {raw_lift_t:+.2f}, p = {raw_lift_p:.2e}, "
          f"n_clips = {len(raw_lifts)})")

    # --- PRIMARY: pooled logistic regression ---
    is_sink = np.concatenate([r["is_sink"] for r in records])
    bg = np.concatenate([r["bg"] for r in records])
    row_norm = np.concatenate([r["row_norm"] for r in records])
    col_norm = np.concatenate([r["col_norm"] for r in records])
    # Convert continuous bg fraction → binary "is_bg" via threshold 0.5 for logit
    is_bg = (bg >= 0.5).astype(np.float64)
    print(f"\npooled logit on n_cells = {len(is_bg):,}  "
          f"(bg threshold 0.5 → {is_bg.mean()*100:.1f}% bg cells; "
          f"{is_sink.sum():,} sink cells)")
    logit = pooled_logit(is_sink, is_bg, row_norm, col_norm)
    print(f"  beta(is_sink) = {logit['beta']:+.4f}  "
          f"(SE {logit['se']:.4f}, z = {logit['z']:+.2f}, p = {logit['p']:.2e}, "
          f"OR = {logit['odds_ratio']:.3f})  [{logit['backend']}]")

    # --- SECONDARY: interior-only Welch t-test ---
    interior = interior_test(records, H_used, W_used)
    print(f"\ninterior-only (drop cells within {EDGE_CELLS} of edge):  "
          f"n_sink_int = {interior['n_int_sink']}, "
          f"n_nonsink_int = {interior['n_int_nonsink']:,}")
    if np.isfinite(interior['lift']):
        print(f"  bg(sink_int) = {interior['sink_mean']:.4f}  vs  "
              f"bg(nonsink_int) = {interior['nonsink_mean']:.4f}  →  "
              f"lift = {interior['lift']:+.4f}  "
              f"(Welch t = {interior['t']:+.2f}, p = {interior['p']:.2e})")
    else:
        print("  underpowered: not enough sinks survive the interior filter "
              f"(expected — Stage 2.2 found ~all L2 sinks are edge tokens)")

    # --- Plot ---
    try:
        plot_bg_by_row(records, H_used, out_dir / "sink_vs_nonsink_by_row.png")
    except Exception as e:
        print(f"  [warn] plot failed: {type(e).__name__}: {e}")

    # --- VERDICT ---
    SEMANTIC = (raw_mean_lift > 0
                and logit["beta"] > 0
                and np.isfinite(logit["p"]) and logit["p"] < 0.01
                and (interior["lift"] > 0 if np.isfinite(interior["lift"]) else True))
    NULL = raw_mean_lift <= 0
    if NULL:
        verdict_tag = "NULL"
        verdict = (f"NULL — clip-mean raw_lift = {raw_mean_lift:+.4f} ≤ 0; "
                   f"sinks are not on background AT ALL. Visual-sink prediction "
                   f"not supported in this model.")
    elif SEMANTIC:
        verdict_tag = "SEMANTIC"
        interior_note = (f"interior lift = {interior['lift']:+.4f} (p = {interior['p']:.2e})"
                         if np.isfinite(interior['lift']) else "interior test underpowered")
        verdict = (f"SEMANTIC — raw clip-mean lift = {raw_mean_lift:+.4f} (p = {raw_lift_p:.2e}); "
                   f"pooled logit beta(is_sink) = {logit['beta']:+.4f} (OR = {logit['odds_ratio']:.2f}, "
                   f"p = {logit['p']:.2e}) AFTER partialling out row/col; {interior_note}. "
                   f"Visual-sink prediction supported; SAM 3 background story holds after "
                   f"controlling for the edge prior.")
    else:
        verdict_tag = "POSITIONAL ARTIFACT"
        verdict = (f"POSITIONAL ARTIFACT — raw clip-mean lift = {raw_mean_lift:+.4f} > 0, "
                   f"but pooled logit beta(is_sink) = {logit['beta']:+.4f} "
                   f"(p = {logit['p']:.2e}) is not significant after partialling out row/col. "
                   f"The 'sinks on background' pattern IS the edge prior — frame tops are "
                   f"disproportionately background and L2 sinks are disproportionately "
                   f"row-0. This is a SUBSTANTIVE finding (a border story, not a failure): "
                   f"L2 video sinks attach to spatial position, not semantic content.")

    # --- decision.txt ---
    decision_path = out_dir / "stage2_3_decision.txt"
    with open(decision_path, "w") as f:
        f.write("Stage 2.3 — video L2 sink ↔ SAM 3 background association  "
                f"(n_clips={len(records)}, layer=L{LAYER}, "
                f"τ_detect={args.detection_threshold}, "
                f"max_per_prompt={args.max_per_prompt}, "
                f"dry_run={args.dry_run})\n")
        f.write(f"prompts (multi-prompt union): {prompts_used}\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"Clip-mean raw_lift = {raw_mean_lift:+.4f} "
                f"(SE {raw_lift_se:.4f}, t = {raw_lift_t:+.2f}, "
                f"p = {raw_lift_p:.2e}, n_clips = {len(raw_lifts)})\n\n")
        f.write(f"PRIMARY pooled logistic regression  "
                f"[is_bg ~ is_sink + row_norm + col_norm,  n_cells = {len(is_bg):,}]\n")
        f.write(f"  beta(is_sink) = {logit['beta']:+.4f}  "
                f"(SE {logit['se']:.4f}, z = {logit['z']:+.2f}, "
                f"p = {logit['p']:.2e}, OR = {logit['odds_ratio']:.3f})\n")
        f.write(f"  backend: {logit['backend']}\n\n")
        f.write(f"SECONDARY interior-only (drop cells within {EDGE_CELLS} of edge)\n")
        f.write(f"  n_sink_int = {interior['n_int_sink']}, "
                f"n_nonsink_int = {interior['n_int_nonsink']:,}\n")
        if np.isfinite(interior['lift']):
            f.write(f"  bg(sink) = {interior['sink_mean']:.4f}, "
                    f"bg(nonsink) = {interior['nonsink_mean']:.4f}, "
                    f"lift = {interior['lift']:+.4f}  "
                    f"(Welch t = {interior['t']:+.2f}, p = {interior['p']:.2e})\n")
            f.write("  treat as corroboration, not a co-equal gate\n\n")
        else:
            f.write("  underpowered (expected — Stage 2.2 found ~all L2 sinks "
                    "are edge tokens)\n\n")
        f.write(f"VERDICT: [{verdict_tag}] {verdict}\n")
    print(f"\nwrote {decision_path}")
    print("\n" + "=" * 80)
    print(f"STAGE 2.3 VERDICT  (n_clips = {len(records)})")
    print("=" * 80)
    print(verdict)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ),
                   help="Stage 2.2 per_clip_sink_coords.npz with L2 sinks.")
    p.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR),
                   help="Directory of source .mp4 files (matches Stage 2.2).")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--prompts", default="",
                   help="Comma-separated SAM 3 text prompts. Default: "
                        f"{','.join(DEFAULT_PROMPTS)}. Foreground = union of "
                        "all detections >= detection_threshold across prompts.")
    p.add_argument("--detection_threshold", type=float, default=DEFAULT_DET_THR,
                   help=f"Per-prompt detection score threshold (default "
                        f"{DEFAULT_DET_THR}).")
    p.add_argument("--max_per_prompt", type=int, default=MAX_DETS_PER_PROMPT,
                   help=f"Cap on detections per prompt (default "
                        f"{MAX_DETS_PER_PROMPT}).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N clips (0 = all). Useful for smoke tests.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip SAM/frames; synthesize a background field with a "
                        "mild top=background bias. Smoke-tests the plumbing only.")
    args = p.parse_args()
    main(args)
