#!/usr/bin/env python3
"""
step_1_Q_Saturation_Add_On.py
─────────────────────────────────────────────────────────────────────────────
Quad descriptor saturation check — runs automatically after Step 1 completes.

Reads the self-contained .npz files written by process_single_session and
computes per-file saturation metrics without recomputing any quads.

Designed to be called at the end of run_step_1_parallel (or standalone).

Math
----
For each quad file:
  L_q        = longest pairwise distance among the 4 vertices        (pixels)
  σ_d        = c · σ_px / L_q                                        (descriptor units)
  nn_dist    = nearest-neighbour distance in R^4 descriptor space
  ratio      = median(nn_dist) / median(σ_d)
  saturated  ↔  ratio ≤ 1  (nn spacing ≤ descriptor blur radius)

Usage
-----
# Automatic — called from run_step_1_parallel (no extra args needed):
    from step_1_Q_Saturation_Add_On import run_saturation_check_after_step1
    run_saturation_check_after_step1(config)          # uses config paths

# Standalone CLI:
    python step_1_Q_Saturation_Add_On.py \\
        --step1_results_dir  /path/to/step_1_results \\
        --output_dir         /path/to/output_root \\
        --sigma_px 1.0 --c_factor 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

import logging
logger = logging.getLogger("neuron_mapping_parallel")

# ══════════════════════════════════════════════════════════════════════════
# Math kernels  (pure NumPy / C — no Python loops in hot paths)
# ══════════════════════════════════════════════════════════════════════════

def _longest_diagonal(centroids: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """L_q = longest pairwise distance among a quad's 4 vertices, for all Q quads."""
    pts  = centroids[quads]                                  # (Q, 4, 2)
    i, j = np.triu_indices(4, k=1)                          # 6 unique pairs
    d2   = np.sum((pts[:, i] - pts[:, j]) ** 2, axis=-1)    # (Q, 6)
    return np.sqrt(d2.max(axis=1))                           # (Q,)


def _sigma_d(L: np.ndarray, sigma_px: float, c: float) -> np.ndarray:
    """σ_d = c · σ / L  (vectorized, guarded against L=0)."""
    return (c * sigma_px) / np.maximum(L, 1e-12)


def _nn_distances(descs: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour distances in R^4 via cKDTree.
    workers=-1 releases the GIL → threaded C code, no Python loop.
    """
    tree = cKDTree(descs)
    dists, _ = tree.query(descs, k=2, workers=-1)
    return dists[:, 1]                                       # (Q,) — exclude self


def _pstats(x: np.ndarray) -> Dict[str, float]:
    p = np.percentile(x, [0, 10, 50, 90, 100])
    return dict(min=float(p[0]), p10=float(p[1]), median=float(p[2]),
                p90=float(p[3]), max=float(p[4]))


# ══════════════════════════════════════════════════════════════════════════
# Per-file worker
# ══════════════════════════════════════════════════════════════════════════

def _check_one_file(
    npz_path: Path,
    sigma_px: float,
    c_factor: float,
) -> Dict[str, Any]:
    """Load one step-1 .npz and return its saturation metrics."""
    d         = np.load(npz_path, allow_pickle=True)
    centroids = d["centroids"].astype(np.float64, copy=False)    # (N, 2)
    quads     = d["quad_idx"]                                     # (Q, 4)
    descs     = d["quad_desc"].astype(np.float64, copy=False)    # (Q, 4)

    if not np.issubdtype(quads.dtype, np.integer):
        quads = quads.astype(np.int32)

    L        = _longest_diagonal(centroids, quads)
    sigma_d  = _sigma_d(L, sigma_px, c_factor)
    nn_dists = _nn_distances(descs)

    med_nn = float(np.median(nn_dists))
    med_sd = float(np.median(sigma_d))
    ratio  = med_nn / (med_sd + 1e-12)
    is_sat = bool(med_nn <= med_sd)

    v_bbox = float(np.prod(
        np.maximum(descs.max(axis=0) - descs.min(axis=0), 0.0)
    ))

    # Parse metadata from filename: {animal_session}__{condition}__{seed}_centroids_quads
    parts = npz_path.stem.split("__")
    return {
        "filename":           npz_path.name,
        "animal_session":     parts[0] if len(parts) > 0 else "?",
        "condition":          parts[1] if len(parts) > 1 else "unknown",
        "seed":               parts[2].replace("_centroids_quads", "") if len(parts) > 2 else "unknown",
        "subject_id":         str(d["animal_id"]),
        "session_id":         str(d["session"]),
        "N":                  int(centroids.shape[0]),
        "Q":                  int(quads.shape[0]),
        "sigma_px":           sigma_px,
        "c_factor":           c_factor,
        "median_nn_distance": med_nn,
        "median_sigma_d":     med_sd,
        "saturation_ratio":   ratio,
        "is_saturated":       is_sat,
        "V_eff_bbox":         v_bbox,
        "L_stats_px":         _pstats(L),
        "nn_distance_stats":  _pstats(nn_dists),
        "sigma_d_stats":      _pstats(sigma_d),
    }


# ══════════════════════════════════════════════════════════════════════════
# Main runner  (called automatically after Step 1 or standalone)
# ══════════════════════════════════════════════════════════════════════════

def run_saturation_check(
    step1_results_dir: Path,
    output_dir: Path,
    sigma_px: float = 1.0,
    c_factor: float = 2.0,
    n_workers: Optional[int] = None,
    verbose: bool = True,
    only_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the saturation check over all .npz files in step1_results_dir.

    Parameters
    ----------
    step1_results_dir : Path
        Directory containing the .npz files written by Step 1.
    output_dir : Path
        Root output directory; results land in output_dir/saturation/.
    sigma_px : float
        Estimated localisation uncertainty in pixels.
    c_factor : float
        Scaling multiplier for σ_d = c·σ/L.
    n_workers : int, optional
        Thread-pool workers (default = CPU count).
    verbose : bool
        Print per-file progress.
    only_files : list of str, optional
        If given, restrict to these filenames (e.g. from the step-1 result list).
    """
    step1_results_dir = Path(step1_results_dir)
    output_dir        = Path(output_dir)

    all_npz = sorted(step1_results_dir.glob("*.npz"))

    # Optionally restrict to files just processed by step 1
    if only_files:
        only_set = set(only_files)
        all_npz  = [p for p in all_npz if p.name in only_set]

    if not all_npz:
        logger.info("[SAT] No .npz files found — skipping saturation check.")
        return []

    import os
    n_workers = n_workers or os.cpu_count()

    logger.info(f"\n[SAT] Quad Descriptor Saturation Check")
    logger.info("═" * 65)
    logger.info(f"  Step-1 dir  : {step1_results_dir}")
    logger.info(f"  Files       : {len(all_npz)}")
    logger.info(f"  σ_px={sigma_px}   c={c_factor}   workers={n_workers}")
    logger.info("═" * 65)

    results: List[Dict[str, Any]] = []
    errors:  List[str]            = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_check_one_file, p, sigma_px, c_factor): p.name
            for p in all_npz
        }
        done = 0
        for fut in as_completed(futures):
            fname = futures[fut]
            done += 1
            try:
                r = fut.result()
                results.append(r)
                if verbose:
                    tag = "SAT" if r["is_saturated"] else "   "
                    logger.info(
                        f"  [{tag}] [{done:>4d}/{len(all_npz)}]  "
                        f"{r['filename']:<48s}  "
                        f"ratio={r['saturation_ratio']:6.3f}  "
                        f"Q={r['Q']:>6d}"
                    )
            except Exception as exc:
                errors.append(f"{fname}: {exc}")
                logger.info(f"  [ERR] {fname}: {exc}")

    results.sort(key=lambda r: r["filename"])

    # ── Summary ──
    logger.info("\n" + "═" * 65)
    n_sat = sum(r["is_saturated"] for r in results)
    logger.info(f"  Processed : {len(results)}   Errors: {len(errors)}")
    logger.info(f"  Saturated : {n_sat}/{len(results)}  ({100*n_sat/max(len(results),1):.1f}%)")

    if results:
        ratios = np.array([r["saturation_ratio"] for r in results])
        p      = np.percentile(ratios, [0, 10, 50, 90, 100])
        logger.info(f"\n  ratio — min={p[0]:.4f}  p10={p[1]:.4f}  "
              f"median={p[2]:.4f}  p90={p[3]:.4f}  max={p[4]:.4f}")

        by_cond:  Dict[str, list] = defaultdict(list)
        sat_cond: Dict[str, int]  = defaultdict(int)
        for r in results:
            by_cond[r["condition"]].append(r["saturation_ratio"])
            sat_cond[r["condition"]] += int(r["is_saturated"])

        if len(by_cond) > 1 or list(by_cond.keys()) != ["unknown"]:
            logger.info("\n  Per-condition:")
            for cond in sorted(by_cond):
                v = np.array(by_cond[cond])
                logger.info(f"    {cond:<15s}  median={np.median(v):.4f}  "
                      f"sat={sat_cond[cond]}/{len(v)}")

    logger.info("═" * 65)

    # ── Save ──
    sat_dir = output_dir / "saturation"
    sat_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(results,  sat_dir / "saturation_results.csv")
    _save_json(results, sat_dir / "saturation_results.json")
    logger.info(f"\n[SAT] Results → {sat_dir}\n")

    return results


def run_saturation_check_after_step1(
    config,
    step1_summary: Optional[dict] = None,
    sigma_px: float = 1.0,
    c_factor: float = 2.0,
) -> List[Dict[str, Any]]:
    step1_results_dir = Path(config.output_dir) / "step_1_results"
    output_dir        = Path(config.output_dir)
    """
    Convenience wrapper — call this at the end of run_step_1_parallel.

    Automatically derives paths from config and optionally restricts
    to only the files produced in this run (via step1_summary).

    Parameters
    ----------
    config : PipelineConfig
        The same config passed to run_step_1_parallel.
    step1_summary : dict, optional
        The dict returned by run_step_1_parallel.  If provided, only the
        files processed in this run are checked (skips already-existing ones
        that were skipped by step 1).
    sigma_px, c_factor : float
        Saturation parameters — can also be added to PipelineConfig if desired.
    """
    # Skip if results already exist
    sat_json = output_dir / "saturation" / "saturation_results.json"
    if sat_json.exists():
        logger.info(f"[SAT] Saturation results already exist → {sat_json.name}, skipping.")
        return []

    # Optionally pull sigma/c from config if present
    sigma_px = getattr(config, "sat_sigma_px", sigma_px)
    c_factor = getattr(config, "sat_c_factor",  c_factor)

    # Build a set of filenames that were actually processed this run
    only_files = None
    if step1_summary is not None:
        processed = [
            r for r in step1_summary.get("results", [])
            if not r.get("skipped", False)
        ]
        if processed:
            only_files = [
                f"{r['session_name']}_centroids_quads.npz"
                for r in processed
            ]

    return run_saturation_check(
        step1_results_dir = step1_results_dir,
        output_dir        = output_dir,
        sigma_px          = sigma_px,
        c_factor          = c_factor,
        only_files        = only_files,
        verbose           = getattr(config, "verbose", True),
    )


# ══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════

_SCALAR = (int, float, str, bool, np.integer, np.floating)

def _save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    if not results:
        return
    keys = [k for k, v in results[0].items() if isinstance(v, _SCALAR)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    logger.info(f"  CSV  → {path.name}")

def _json_conv(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(type(o))

def _save_json(results: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_conv)
    logger.info(f"  JSON → {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# CLI  (standalone use)
# ══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Saturation check add-on for Stars2Cells Step 1 output.",
    )
    p.add_argument(
        "--step1_results_dir",
        required=True,
        help="Path to the step_1_results directory containing .npz files.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory; saturation/ subfolder will be created here.",
    )
    p.add_argument("--sigma_px", type=float, default=1.0)
    p.add_argument("--c_factor", type=float, default=2.0)
    p.add_argument("--workers",  type=int,   default=None)
    p.add_argument("--quiet",    action="store_true")
    args, _ = p.parse_known_args()
    return args


def main():
    args = _parse_args()
    run_saturation_check(
        step1_results_dir = Path(args.step1_results_dir),
        output_dir        = Path(args.output_dir),
        sigma_px          = args.sigma_px,
        c_factor          = args.c_factor,
        n_workers         = args.workers,
        verbose           = not args.quiet,
    )


if __name__ == "__main__":
    main()