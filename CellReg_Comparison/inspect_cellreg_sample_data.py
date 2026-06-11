"""
inspect_cellreg_sample_data.py

Pull the CellReg sample spatial-footprint data out of a local clone of the
zivlab/CellReg repo, inspect it (shapes, dtypes, value ranges, sparsity,
per-cell footprint areas, weighted centroids), THEN run both registration
pipelines on it and tabulate how many cross-session matches each one reports.

  CellReg sample data:
    <repo>/SampleData/spatial_footprints_01.mat ... _05.mat   (5 sessions)
    Each file holds one 3-D array of weighted ROIs in N x M x K layout
    (N = neurons, M = y pixels, K = x pixels).

Pipeline (each phase is independently toggleable — see PHASE TOGGLES):
  1. Inspect each session → summary.json, per_cell_stats.csv, PNGs.
  2. Export the loaded footprints into the on-disk formats both pipelines need.
     Everything goes into ONE working folder (PIPELINE_DIR, default s2c_run/),
     the standard Stars2Cells layout where the sessions sit alongside the step
     results — so the GUI can load the sessions and see the step outputs:
       s2c_run/<session>.npy            {centroids_x, centroids_y,
                                          roi_ids, subject_id, session_id}
       s2c_run/spatial_mapping/<session>_footprints.npy  {footprints: NxHxW}
       s2c_run/step_1_results/ ... step_3_results/        (written by the steps)
     (subject_id/session_id are included so the Stars2Cells GUI loader, which
      requires them, can also open these files.)
  3. Run the CellReg API (CellReg_python.run_cellreg); cache it to
     cellreg_result.npy so the table can be rebuilt without rerunning CellReg.
  4. Run the Stars2Cells API (steps 1 → 3, headless), each step skippable.
  5. Count cross-session matches each method reports per session pair and write
     comparison_table.csv / .json highlighting the discrepancy.

RESUMING A PARTIAL RUN
----------------------
Every stage writes to disk, so you can re-run just the part that failed without
redoing the slow stages. Example: inspection + CellReg already succeeded but you
want to re-run Stars2Cells from Step 2.5 onward (e.g. after lowering
RANSAC_MIN_INLIER_RATIO) — set:

    RUN_INSPECTION = False
    RUN_EXPORT     = False
    RUN_CELLREG    = False      # reloads cellreg_result.npy for the table
    RUN_S2C        = True
    S2C_RUN_STEP_1   = False    # step_1_results/ already on disk
    S2C_RUN_STEP_1_5 = False
    S2C_RUN_STEP_2   = False    # the slow one — already done
    S2C_RUN_STEP_2_5 = True     # re-run with new params
    S2C_RUN_STEP_3   = True

There is NO ground truth for the CellReg sample data, so this does NOT compute
precision/recall/F1 — it only compares the *number of matches* each method
claims, which is the discrepancy we want to surface.

NOTE on alignment: the Python CellReg port uses identity alignment (it does not
register sessions into a common frame), so it links cells already within
MAXIMAL_DISTANCE_UM. Stars2Cells matches on rotation/translation-invariant quad
descriptors. A large part of any discrepancy reflects that difference.

Requires: numpy, scipy, h5py, matplotlib  (+ the Stars2Cells deps for step 4)
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252, which chokes on the emoji in the
# Stars2Cells logs and prints alarming (harmless) "--- Logging error ---"
# tracebacks. Force UTF-8 so those go away. Runs in spawned children too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Make the repo importable no matter where this script is launched from.
#   <repo_root>/CellReg_Comparison/CellReg_python.py   → run_cellreg
#   <repo_root>/steps, <repo_root>/utilities           → Stars2Cells pipeline
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../CellReg_Comparison
_REPO_ROOT = _THIS_DIR.parent                         # repo root
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ----------------------------------------------------------------------------
# Config -- edit these paths if your layout differs.
# ----------------------------------------------------------------------------
REPO_DIR = Path(r"C:\Users\ariAccount\Desktop\Stars2CellsPaper\CellReg")
OUTPUT_DIR = Path(r"C:\Users\ariAccount\Desktop\CellReg_SampleData_Inspection")

# Files are named spatial_footprints_0N.mat; search recursively so it works
# regardless of whether they sit in SampleData/ or somewhere nested.
GLOB_PATTERN = "spatial_footprints_*.mat"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE TOGGLES — set any to False to skip that phase and reuse on-disk results
# ═══════════════════════════════════════════════════════════════════════════
RUN_INSPECTION = True   # load .mat → stats, projections, summary.json/csv
RUN_EXPORT     = True   # write pipeline_input/<session>.npy (+ footprints)
RUN_CELLREG    = True    # CellReg API (result cached to cellreg_result.npy)
RUN_S2C        = True    # Stars2Cells API (sub-steps below)
BUILD_TABLE    = True    # comparison table from whatever results are available

# ── Stars2Cells sub-steps (only consulted when RUN_S2C) ──────────────────────
# Skipping a step assumes its output folder under s2c_run/ already exists.
S2C_RUN_STEP_1   = True   # quad generation        → step_1_results/
S2C_RUN_STEP_1_5 = True   # threshold calibration  → step_1_5_results/
S2C_RUN_STEP_2   = True   # descriptor matching    → step_2_results/   (slow)
S2C_RUN_STEP_2_5 = True   # RANSAC filtering       → step_2_5_results/
S2C_RUN_STEP_3   = True   # Hungarian + tracking   → step_3_results/

USE_FOOTPRINTS = True   # feed CellReg the spatial-correlation model too.
                        # Requires padding every session to a common FOV and
                        # writing dense NxHxW footprint files (can be large).
                        # Set False for CellReg centroid-only (no footprints).

# subject_id written into each exported session (so the GUI groups them as one
# animal). None → derive from the filename prefix (e.g. "spatial_footprints").
ANIMAL_ID = None

# ── CellReg parameters (see CellReg_python.run_cellreg) ──────────────────────
MICRONS_PER_PIXEL   = 1.0    # sample data has no scale metadata; treat px = µm
MAXIMAL_DISTANCE_UM = 12.0   # centroid search radius for CellReg
P_SAME_THRESHOLD    = 0.5

# ── Stars2Cells parameters (mirrors s2c_api.py defaults) ─────────────────────
SESSION_FILENAME_REGEX     = r'^([A-Za-z0-9_]+?)_(\d+)(.*?)\.npy$'
SESSION_PAIR_STRATEGY      = 'all_vs_all'   # compare every session pair (10 for 5 sessions)
KNN_K                      = 15
MAX_TRIANGLES_PER_DIAGONAL = 25
QUAD_KEEP_FRACTION         = 1.0
DIAGONAL_RNG_SEED          = 42
N_WORKERS                  = 8
PROCESSES                  = None

CALIB_SAMPLE_SIZE        = 1000
CALIB_TARGET_QUALITY     = 0.95
CALIB_THRESHOLD_MIN      = 0.0
CALIB_THRESHOLD_MAX      = 0.02   # tuned for the benchmark descriptor space;
                                  # widen if calibration looks degenerate on real data
CALIB_N_THRESHOLD_POINTS = 50
MAX_PAIRS_PER_ANIMAL     = 10

DISTANCE_METRIC       = 'cosine'
CONSISTENCY_THRESHOLD = 0.8

RANSAC_MAX_RESIDUAL     = 5.0
RANSAC_ITERATIONS       = 1000
RANSAC_MIN_INLIER_RATIO = 0.05   # ← lower this (e.g. 0.01) if Step 2.5 rejects
                                 #   every pair on real data ("inlier ratio < 5%")
RANSAC_ALLOW_SCALING    = False

HUNGARIAN_COST_MIN   = 0.0
HUNGARIAN_COST_MAX   = 2319.0
HUNGARIAN_COST_STEPS = 20
USE_QUAD_VOTING      = True

# Derived paths (used across phases)
RAW_DIR       = OUTPUT_DIR / "raw_mat_files"
FIG_DIR       = OUTPUT_DIR / "projections"

# Single working folder that holds BOTH the session .npy files AND every step's
# results + logs — the standard Stars2Cells layout (input_dir == output_dir, the
# way s2c_api.py and the benchmark folders are organized). To run in the GUI,
# point "Select Data Folder" here AND set output_dir to this same path in the
# pipeline-config dialog, so it loads the sessions and picks up the step results.
PIPELINE_DIR  = OUTPUT_DIR / "s2c_run"
INPUT_DIR     = PIPELINE_DIR            # sessions are written here
S2C_OUT_DIR   = PIPELINE_DIR            # step_*_results are written here too
FOOTPRINT_DIR = PIPELINE_DIR / "spatial_mapping"
# Kept OUTSIDE PIPELINE_DIR so the GUI's *.npy session loader never opens it.
CELLREG_CACHE = OUTPUT_DIR / "cellreg_result.npy"


# ----------------------------------------------------------------------------
# .mat loading -- handle both old-format (<=v7) and v7.3 (HDF5) files.
# ----------------------------------------------------------------------------
def load_footprints(path: Path) -> np.ndarray:
    """Return the spatial-footprint array as (N, M, K), float."""
    try:
        from scipy.io import loadmat

        mat = loadmat(path)
        arr = _pick_array(
            {k: v for k, v in mat.items() if not k.startswith("__")}
        )
        return np.asarray(arr, dtype=np.float64)
    except NotImplementedError:
        # scipy raises this for v7.3 (HDF5-backed) .mat files.
        return _load_v73(path)


def _load_v73(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        datasets = {
            k: np.array(v)
            for k, v in f.items()
            if isinstance(v, h5py.Dataset) and v.ndim >= 2
        }
    arr = _pick_array(datasets)
    # MATLAB is column-major; h5py returns the axes reversed, so an
    # (N, M, K) array comes back as (K, M, N). Restore the original order.
    return np.transpose(np.asarray(arr, dtype=np.float64), tuple(range(arr.ndim))[::-1])


def _pick_array(candidates: dict) -> np.ndarray:
    """Choose the spatial-footprint array from a dict of mat variables."""
    if not candidates:
        raise ValueError("No data variables found in .mat file.")
    # Prefer a named match; otherwise take the largest array.
    for name in ("spatial_footprints", "footprints", "sFootprints"):
        if name in candidates:
            return candidates[name]
    return max(candidates.values(), key=lambda a: np.asarray(a).size)


# ----------------------------------------------------------------------------
# Inspection
# ----------------------------------------------------------------------------
def inspect_session(arr: np.ndarray) -> dict:
    """Per-session summary. Assumes axis 0 = cells (CellReg N x M x K)."""
    n_cells, y, x = arr.shape
    flat_per_cell = arr.reshape(n_cells, -1)

    nonzero_per_cell = (flat_per_cell > 0).sum(axis=1)  # footprint area in px
    peak_per_cell = flat_per_cell.max(axis=1)

    # Intensity-weighted centroids (matches how CellReg derives centroids).
    centroids = _weighted_centroids(arr)

    overall_nonzero = float((arr > 0).mean())  # fraction of pixels that are >0

    return {
        "n_cells": int(n_cells),
        "fov_y": int(y),
        "fov_x": int(x),
        "dtype": str(arr.dtype),
        "value_min": float(arr.min()),
        "value_max": float(arr.max()),
        "value_mean_nonzero": float(arr[arr > 0].mean()) if arr.any() else 0.0,
        "global_nonzero_fraction": overall_nonzero,
        "footprint_area_px_mean": float(nonzero_per_cell.mean()),
        "footprint_area_px_median": float(np.median(nonzero_per_cell)),
        "footprint_area_px_min": int(nonzero_per_cell.min()),
        "footprint_area_px_max": int(nonzero_per_cell.max()),
        "peak_value_mean": float(peak_per_cell.mean()),
        "_centroids": centroids,            # (N, 2) y,x  -- not JSON-serialized
        "_area_per_cell": nonzero_per_cell,  # for CSV
        "_peak_per_cell": peak_per_cell,
    }


def _weighted_centroids(arr: np.ndarray) -> np.ndarray:
    n_cells, y, x = arr.shape
    yy, xx = np.mgrid[0:y, 0:x]
    out = np.full((n_cells, 2), np.nan)
    for i in range(n_cells):
        w = arr[i]
        total = w.sum()
        if total > 0:
            out[i, 0] = (w * yy).sum() / total
            out[i, 1] = (w * xx).sum() / total
    return out


def save_projection(arr: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = arr.max(axis=0)  # max across cells -> a "cell map" of the FOV
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(proj, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Export to the on-disk formats the two pipelines consume
# ----------------------------------------------------------------------------
def _split_stem(stem: str) -> tuple[str, str]:
    """Split 'spatial_footprints_01' → ('spatial_footprints', '01')."""
    m = re.match(r'^(.*?)_?(\d+)$', stem)
    if m and m.group(2):
        return (m.group(1) or "sample"), m.group(2)
    return stem, "1"


def _pad_to_common_fov(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Zero-pad every (N, H, W) stack to a shared (max H, max W).

    Padding is added at the bottom/right (origin stays top-left), so the
    weighted centroids computed on the originals remain valid. CellReg's
    spatial-correlation model ravels footprints from different sessions and
    multiplies them element-wise, which requires an identical (H, W).
    """
    max_y = max(a.shape[1] for a in arrays)
    max_x = max(a.shape[2] for a in arrays)
    out = []
    for a in arrays:
        n, y, x = a.shape
        if (y, x) == (max_y, max_x):
            out.append(a)
        else:
            padded = np.zeros((n, max_y, max_x), dtype=a.dtype)
            padded[:, :y, :x] = a
            out.append(padded)
    return out, (max_y, max_x)


def export_sessions(sessions: list[dict], input_dir: Path, footprint_dir: Path,
                    use_footprints: bool) -> list[Path]:
    """Write per-session centroid dicts (+ optional footprint files).

    sessions : list of {"stem": str, "arr": (N,H,W), "centroids": (N,2) y,x}

    The session dict carries subject_id/session_id so BOTH the headless
    pipeline AND the Stars2Cells GUI loader (which requires those fields) can
    read it. Returns the list of session .npy paths in session order.
    """
    input_dir.mkdir(parents=True, exist_ok=True)

    padded = None
    if use_footprints:
        footprint_dir.mkdir(parents=True, exist_ok=True)
        padded, fov = _pad_to_common_fov([s["arr"] for s in sessions])
        print(f"  Footprints padded to common FOV {fov[0]}x{fov[1]} for CellReg")

    session_paths = []
    for idx, s in enumerate(sessions):
        cents = s["centroids"]  # (N, 2) as (y, x)
        subject_id, session_id = _split_stem(s["stem"])
        if ANIMAL_ID is not None:
            subject_id = ANIMAL_ID
        sess_path = input_dir / f"{s['stem']}.npy"
        np.save(
            sess_path,
            {
                "centroids_x": cents[:, 1].astype(float),  # column
                "centroids_y": cents[:, 0].astype(float),  # row
                "roi_ids": np.arange(len(cents)),
                "subject_id": subject_id,    # required by the GUI loader
                "session_id": session_id,    # required by the GUI loader
            },
            allow_pickle=True,
        )
        session_paths.append(sess_path)

        if use_footprints:
            # Dense format only — the compact reader assumes a square FOV,
            # which the sample data is not. CellReg normalizes to sum=1 on load.
            np.save(
                footprint_dir / f"{s['stem']}_footprints.npy",
                {"footprints": padded[idx]},
                allow_pickle=True,
            )

    print(f"  Wrote {len(session_paths)} session files → {input_dir}")
    return session_paths


# ----------------------------------------------------------------------------
# Pipeline 1 — CellReg API (with on-disk caching)
# ----------------------------------------------------------------------------
def run_cellreg_api(session_paths: list[Path], footprint_dir: Path,
                    use_footprints: bool) -> dict:
    from CellReg_python import run_cellreg

    return run_cellreg(
        file_paths=[str(p) for p in session_paths],
        microns_per_pixel=MICRONS_PER_PIXEL,
        maximal_distance_um=MAXIMAL_DISTANCE_UM,
        p_same_threshold=P_SAME_THRESHOLD,
        footprint_dir=str(footprint_dir) if use_footprints else None,
    )


def save_cellreg_cache(result: dict, stems: list[str], path: Path) -> None:
    np.save(path, {"result": result, "stems": list(stems)}, allow_pickle=True)
    print(f"  Cached CellReg result → {path}")


def load_cellreg_cache(path: Path):
    """Return (result, stems) from a prior run, or (None, None)."""
    if not path.exists():
        return None, None
    blob = np.load(path, allow_pickle=True).item()
    return blob.get("result"), blob.get("stems")


# ----------------------------------------------------------------------------
# Pipeline 2 — Stars2Cells API (headless: steps 1 → 3, no Qt dialogs)
# ----------------------------------------------------------------------------
def run_s2c_api(input_dir: Path, output_dir: Path) -> Path:
    from utilities import PipelineConfig, setup_logging
    from steps.step_1_quad_generation import run_step_1_parallel
    from steps.step_1_5_threshold_generation import run_global_tuning_all_animals
    from steps.step_2_matching_generator import run_step_2_all_animals_parallel
    from steps.step_2_5_RANSAC import run_step_2_5_ransac
    from steps.step_3_neuron_matching import run_step_3_final_matching

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=True)

    def _warn_if_missing(subdir: str, needed_by: str):
        if not (output_dir / subdir).exists():
            print(f"  ⚠️  {needed_by} is enabled but {subdir}/ is missing — "
                  f"the step it depends on was skipped and never ran.")

    config = PipelineConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        verbose=True,
        skip_existing=True,
        animal_id=None,
        knn_k=KNN_K,
        max_triangles_per_diagonal=MAX_TRIANGLES_PER_DIAGONAL,
        quad_keep_fraction=QUAD_KEEP_FRACTION,
        diagonal_rng_seed=DIAGONAL_RNG_SEED,
        n_workers=N_WORKERS,
        session_group_regex=None,          # one animal, no condition subgroups
        session_pair_strategy=SESSION_PAIR_STRATEGY,
        distance_metric=DISTANCE_METRIC,
        consistency_threshold=CONSISTENCY_THRESHOLD,
    )

    if S2C_RUN_STEP_1:
        print("\n[S2C] Step 1 — quad generation")
        run_step_1_parallel(config, session_filename_regex=SESSION_FILENAME_REGEX)
    else:
        print("\n[S2C] Step 1 skipped")

    if S2C_RUN_STEP_1_5:
        print("\n[S2C] Step 1.5 — threshold calibration")
        _warn_if_missing("step_1_results", "Step 1.5")
        run_global_tuning_all_animals(
            input_dir=str(input_dir), output_dir=str(output_dir),
            sample_size=CALIB_SAMPLE_SIZE, target_quality=CALIB_TARGET_QUALITY,
            threshold_min=CALIB_THRESHOLD_MIN, threshold_max=CALIB_THRESHOLD_MAX,
            n_threshold_points=CALIB_N_THRESHOLD_POINTS, processes=PROCESSES,
            verbose=True, session_group_regex=None,
            session_pair_strategy=SESSION_PAIR_STRATEGY,
            max_pairs_per_animal=MAX_PAIRS_PER_ANIMAL,
        )
    else:
        print("\n[S2C] Step 1.5 skipped")

    if S2C_RUN_STEP_2:
        print("\n[S2C] Step 2 — descriptor matching")
        _warn_if_missing("step_1_5_results", "Step 2")
        run_step_2_all_animals_parallel(
            input_dir=str(input_dir), output_dir=str(output_dir),
            processes=PROCESSES, verbose=True, distance_metric=DISTANCE_METRIC,
            consistency_threshold=CONSISTENCY_THRESHOLD, session_group_regex=None,
            session_pair_strategy=SESSION_PAIR_STRATEGY,
        )
    else:
        print("\n[S2C] Step 2 skipped")

    if S2C_RUN_STEP_2_5:
        print("\n[S2C] Step 2.5 — RANSAC geometric filtering")
        _warn_if_missing("step_2_results", "Step 2.5")
        run_step_2_5_ransac(
            input_dir=str(input_dir), output_dir=str(output_dir),
            ransac_max_residual=RANSAC_MAX_RESIDUAL, ransac_iterations=RANSAC_ITERATIONS,
            ransac_min_inlier_ratio=RANSAC_MIN_INLIER_RATIO,
            ransac_allow_scaling=RANSAC_ALLOW_SCALING, processes=PROCESSES, verbose=True,
        )
    else:
        print("\n[S2C] Step 2.5 skipped")

    if S2C_RUN_STEP_3:
        print("\n[S2C] Step 3 — Hungarian matching + track consolidation")
        _warn_if_missing("step_2_5_results", "Step 3")
        run_step_3_final_matching(
            input_dir=str(input_dir), output_dir=str(output_dir),
            hungarian_cost_min=HUNGARIAN_COST_MIN, hungarian_cost_max=HUNGARIAN_COST_MAX,
            hungarian_cost_steps=HUNGARIAN_COST_STEPS, use_quad_voting=USE_QUAD_VOTING,
            processes=PROCESSES, verbose=True,
        )
    else:
        print("\n[S2C] Step 3 skipped")

    return output_dir / "step_3_results"


# ----------------------------------------------------------------------------
# Match counting
# ----------------------------------------------------------------------------
def cellreg_pair_counts(cmap: np.ndarray, stems: list[str]) -> dict:
    """Matches per session pair = clusters present in BOTH sessions.

    cmap is run_cellreg's cell_to_index_map: shape [n_clusters x n_sessions],
    with -1 marking an absent session and >=0 a 0-based cell index.
    """
    counts = {}
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            both = (cmap[:, i] >= 0) & (cmap[:, j] >= 0)
            counts[(stems[i], stems[j])] = int(both.sum())
    return counts


def _decode(field) -> str:
    """npz string fields come back as 0-d arrays / numpy str_ / bytes."""
    val = np.asarray(field)
    val = val.item() if val.ndim == 0 else val
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    return str(val)


def s2c_pair_counts(step3_dir: Path, stems: list[str]) -> dict:
    """Matches per session pair from Stars2Cells step_3_results/*_sweep.npz.

    Keyed by frozenset({ref, tgt}) so it's order-independent vs CellReg's pairs.
    """
    counts = {}
    if not step3_dir.exists():
        return counts
    stemset = set(stems)
    for f in sorted(step3_dir.glob("*_sweep.npz")):
        data = np.load(f, allow_pickle=False)
        ref = _decode(data["ref_session"])
        tgt = _decode(data["target_session"])
        if ref in stemset and tgt in stemset:
            counts[frozenset((ref, tgt))] = int(len(data["matched_ref_indices"]))
    return counts


def build_comparison_table(stems, cr_counts, s2c_counts, cr_result, out_dir):
    """Assemble + save + print the CellReg-vs-S2C match-count table.

    Enumerates every session pair from `stems`, so it works whether CellReg,
    Stars2Cells, or both produced results (missing side shows N/A).
    """
    def _cr(a, b):
        v = cr_counts.get((a, b))
        return v if v is not None else cr_counts.get((b, a))

    rows = []
    total_cr = 0
    total_s2c = 0
    have_cr = have_s2c = False
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = stems[i], stems[j]
            cr = _cr(a, b)
            s2c = s2c_counts.get(frozenset((a, b)))
            diff = (cr - s2c) if (cr is not None and s2c is not None) else None
            if cr is not None:
                total_cr += cr
                have_cr = True
            if s2c is not None:
                total_s2c += s2c
                have_s2c = True
            rows.append({
                "session_a": a,
                "session_b": b,
                "cellreg_matches": cr if cr is not None else "",
                "s2c_matches": s2c if s2c is not None else "",
                "difference_cr_minus_s2c": diff if diff is not None else "",
            })

    rows.append({
        "session_a": "TOTAL",
        "session_b": "",
        "cellreg_matches": total_cr if have_cr else "",
        "s2c_matches": total_s2c if have_s2c else "",
        "difference_cr_minus_s2c": (total_cr - total_s2c) if (have_cr and have_s2c) else "",
    })

    # ── CSV ──
    csv_path = out_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Per-method summary (descriptive — no ground truth) ──
    cmap = cr_result["cell_to_index_map"] if cr_result is not None else None
    summary = {
        "n_sessions": len(stems),
        "sessions": stems,
        "cellreg": None,
        "stars2cells": {
            "total_pairwise_matches": total_s2c if have_s2c else None,
            "n_pairs_with_results": sum(1 for r in rows[:-1] if r["s2c_matches"] != ""),
        },
        "pairs": rows[:-1],
        "totals": rows[-1],
    }
    if cmap is not None:
        per_cluster_sessions = (cmap >= 0).sum(axis=1)
        summary["cellreg"] = {
            "total_clusters": int(cmap.shape[0]),
            "clusters_in_2plus_sessions": int((per_cluster_sessions >= 2).sum()),
            "clusters_in_all_sessions": int((per_cluster_sessions == len(stems)).sum()),
            "total_pairwise_matches": total_cr,
            "mean_cell_score": round(float(np.nanmean(cr_result["cell_scores"])), 4),
            "model_mse": round(float(cr_result["model_mse"]), 4),
            "intersection_um": round(float(cr_result["intersection_um"]), 3),
            "best_model": cr_result.get("best_model", "centroid"),
        }

    json_path = out_dir / "comparison_table.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Pretty print ──
    def _short(stem):  # spatial_footprints_01 -> 01
        return stem.split("_")[-1]

    def _cell(v):
        return str(v) if v != "" else "N/A"

    print(f"\n{'═' * 58}")
    print("  CellReg vs Stars2Cells — cross-session match counts")
    print(f"{'═' * 58}")
    print(f"  {'Pair':<12}{'CellReg':>10}{'S2C':>10}{'Δ (CR−S2C)':>14}")
    print(f"  {'─' * 46}")
    for r in rows[:-1]:
        pair = f"{_short(r['session_a'])} × {_short(r['session_b'])}"
        print(f"  {pair:<12}{_cell(r['cellreg_matches']):>10}"
              f"{_cell(r['s2c_matches']):>10}{_cell(r['difference_cr_minus_s2c']):>14}")
    print(f"  {'─' * 46}")
    t = rows[-1]
    print(f"  {'TOTAL':<12}{_cell(t['cellreg_matches']):>10}"
          f"{_cell(t['s2c_matches']):>10}{_cell(t['difference_cr_minus_s2c']):>14}")
    print(f"{'═' * 58}")
    if have_cr and not have_s2c:
        print("  Note: Stars2Cells produced no matches (check Step 2.5 inlier "
              "ratio / Step 3 logs).")
    print(f"  Table → {csv_path}")
    print(f"          {json_path}")
    return summary


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _discover_stems(input_dir: Path) -> list[str]:
    if not input_dir.exists():
        return []
    return sorted(p.stem for p in input_dir.glob("*.npy"))


def main() -> None:
    print("Phases: "
          f"inspect={RUN_INSPECTION} export={RUN_EXPORT} "
          f"cellreg={RUN_CELLREG} s2c={RUN_S2C} table={BUILD_TABLE}")
    if RUN_S2C:
        print("S2C steps: "
              f"1={S2C_RUN_STEP_1} 1.5={S2C_RUN_STEP_1_5} 2={S2C_RUN_STEP_2} "
              f"2.5={S2C_RUN_STEP_2_5} 3={S2C_RUN_STEP_3}")

    for d in (OUTPUT_DIR, RAW_DIR, FIG_DIR, PIPELINE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load .mat data if we need to inspect or export ──
    sessions = None
    need_mat = RUN_INSPECTION or RUN_EXPORT
    if need_mat:
        if not REPO_DIR.exists():
            raise SystemExit(f"Repo not found: {REPO_DIR}")
        files = sorted(REPO_DIR.rglob(GLOB_PATTERN))
        if not files:
            raise SystemExit(
                f"No files matching {GLOB_PATTERN!r} under {REPO_DIR}.\n"
                "Check that the SampleData folder is present in the clone."
            )
        print(f"\nFound {len(files)} session file(s). Output -> {OUTPUT_DIR}\n")

        summary = {"source_repo": str(REPO_DIR), "sessions": {}}
        cell_rows = []
        sessions = []

        for path in files:
            session = path.stem
            arr = load_footprints(path)
            info = inspect_session(arr)
            centroids = info.pop("_centroids")
            areas = info.pop("_area_per_cell")
            peaks = info.pop("_peak_per_cell")
            sessions.append({"stem": session, "arr": arr, "centroids": centroids})

            if RUN_INSPECTION:
                shutil.copy2(path, RAW_DIR / path.name)  # the "pull" step
                print(
                    f"{session}: {info['n_cells']} cells | "
                    f"FOV {info['fov_y']}x{info['fov_x']} | dtype {info['dtype']} | "
                    f"range [{info['value_min']:.3g}, {info['value_max']:.3g}] | "
                    f"nonzero {info['global_nonzero_fraction']*100:.2f}% | "
                    f"mean area {info['footprint_area_px_mean']:.1f} px"
                )
                save_projection(arr, FIG_DIR / f"{session}_maxproj.png", session)
                for i in range(info["n_cells"]):
                    cell_rows.append({
                        "session": session,
                        "cell_index": i,
                        "centroid_y": round(float(centroids[i, 0]), 3),
                        "centroid_x": round(float(centroids[i, 1]), 3),
                        "footprint_area_px": int(areas[i]),
                        "peak_value": round(float(peaks[i]), 5),
                    })
                summary["sessions"][session] = info

        if RUN_INSPECTION:
            with open(OUTPUT_DIR / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(OUTPUT_DIR / "per_cell_stats.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                writer.writeheader()
                writer.writerows(cell_rows)
            total_cells = sum(s["n_cells"] for s in summary["sessions"].values())
            print(
                f"\nInspection done. {total_cells} cells across {len(files)} sessions.\n"
                f"  raw .mat copies -> {RAW_DIR}\n"
                f"  projections     -> {FIG_DIR}\n"
                f"  summary.json + per_cell_stats.csv -> {OUTPUT_DIR}"
            )
        else:
            print("Inspection skipped (loaded .mat data only, for export)")

    # ── Determine stems + session paths ──
    if sessions is not None:
        stems = [s["stem"] for s in sessions]
    else:
        stems = _discover_stems(INPUT_DIR)
        if not stems:
            raise SystemExit(
                f"No exported sessions in {INPUT_DIR}. Run with RUN_EXPORT=True "
                "at least once before skipping it."
            )
        print(f"\nUsing {len(stems)} existing session(s) from {INPUT_DIR}")
    session_paths = [INPUT_DIR / f"{s}.npy" for s in stems]

    # ── Export ──
    if RUN_EXPORT:
        print("\n── Exporting sessions for the registration pipelines ──")
        session_paths = export_sessions(sessions, INPUT_DIR, FOOTPRINT_DIR, USE_FOOTPRINTS)
    else:
        print(f"\nExport skipped — using existing sessions in {INPUT_DIR}")

    # ── Pipeline 1: CellReg ──
    cr_result = None
    cr_stems = None
    if RUN_CELLREG:
        print("\n── Running CellReg API ──")
        t0 = time.time()
        try:
            cr_result = run_cellreg_api(session_paths, FOOTPRINT_DIR, USE_FOOTPRINTS)
            cr_stems = stems
            save_cellreg_cache(cr_result, cr_stems, CELLREG_CACHE)
            print(f"  CellReg done in {time.time() - t0:.1f}s "
                  f"({cr_result['cell_to_index_map'].shape[0]} clusters)")
        except Exception as e:
            import traceback
            print(f"  CellReg FAILED: {e}\n{traceback.format_exc()}")
    elif BUILD_TABLE:
        cr_result, cr_stems = load_cellreg_cache(CELLREG_CACHE)
        if cr_result is not None:
            print(f"\nCellReg skipped — loaded cached result from {CELLREG_CACHE}")
        else:
            print(f"\nCellReg skipped — no cache at {CELLREG_CACHE} "
                  "(table will show CellReg N/A)")
    else:
        print("\nCellReg skipped")

    # ── Pipeline 2: Stars2Cells ──
    if RUN_S2C:
        print("\n── Running Stars2Cells API ──")
        t0 = time.time()
        try:
            run_s2c_api(INPUT_DIR, S2C_OUT_DIR)
            print(f"  Stars2Cells finished in {time.time() - t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"  Stars2Cells FAILED: {e}\n{traceback.format_exc()}")
    else:
        print("\nStars2Cells skipped")

    # ── Comparison table ──
    if BUILD_TABLE:
        step3_dir = S2C_OUT_DIR / "step_3_results"
        table_stems = cr_stems if cr_stems else stems
        cr_counts = (cellreg_pair_counts(cr_result["cell_to_index_map"], table_stems)
                     if cr_result is not None else {})
        s2c_counts = s2c_pair_counts(step3_dir, table_stems)
        if cr_counts or s2c_counts:
            build_comparison_table(table_stems, cr_counts, s2c_counts, cr_result, OUTPUT_DIR)
        else:
            print("\nNo CellReg or Stars2Cells results found — nothing to tabulate.")
    else:
        print("\nTable build skipped")

    print(
        f"\nTo open this in the Stars2Cells GUI:\n"
        f"  1. Launch stars2cells.py and click 'Select Data Folder'.\n"
        f"  2. Choose:  {PIPELINE_DIR}\n"
        f"     (it now holds the session .npy files AND step_*_results/).\n"
        f"  3. In the pipeline-config dialog, set output_dir to that SAME folder\n"
        f"     so the GUI reuses the existing step results instead of redoing them."
    )


if __name__ == "__main__":
    main()
