"""
Step 2.5: Geometric Transform Estimation & Quad Filtering (RANSAC-based)

Vectorized RANSAC: batched SVD + chunked float32 residual evaluation.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from utilities import *

logger = logging.getLogger("neuron_mapping_ransac")


# ==============================================================================
# Vectorized Transform Estimation (batched SVD)
# ==============================================================================

def _batch_estimate_rigid_transforms(
    src_samples: np.ndarray,
    dst_samples: np.ndarray,
    allow_scaling: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate K rigid transforms simultaneously via batched SVD.

    Args:
        src_samples: (K, M, 2) source sample points
        dst_samples: (K, M, 2) destination sample points
        allow_scaling: allow uniform scaling (default False = rigid)

    Returns:
        A: (K, 2, 2) transformation matrices
        t: (K, 2)    translation vectors
        valid: (K,)  bool mask of numerically valid transforms
    """
    # Centers: (K, 2)
    src_centers = src_samples.mean(axis=1)
    dst_centers = dst_samples.mean(axis=1)

    # Centered samples: (K, M, 2)
    src_c = src_samples - src_centers[:, None, :]
    dst_c = dst_samples - dst_centers[:, None, :]

    # Cross-covariance: H[k] = src_c[k].T @ dst_c[k] → (K, 2, 2)
    H = np.einsum('kni,knj->kij', src_c, dst_c)

    # Batched SVD — works on (..., 2, 2) arrays
    U, S, Vt = np.linalg.svd(H)  # U:(K,2,2), S:(K,2), Vt:(K,2,2)

    # R = Vt.T @ U.T   → (K, 2, 2)
    R = np.einsum('kji,kli->kjl', Vt, U)

    # Fix reflections where det(R) < 0
    dets = np.linalg.det(R)
    reflect = dets < 0
    if np.any(reflect):
        Vt[reflect, -1, :] *= -1
        R[reflect] = np.einsum('kji,kli->kjl', Vt[reflect], U[reflect])

    # Scale
    if allow_scaling:
        numer = S.sum(axis=1)  # (K,)
        denom = (src_c ** 2).sum(axis=(1, 2))  # (K,)
        denom = np.maximum(denom, 1e-12)
        scale = numer / denom  # (K,)
        A = scale[:, None, None] * R
    else:
        A = R  # scale = 1.0

    # Translation: t = dst_center - A @ src_center
    t = dst_centers - np.einsum('kij,kj->ki', A, src_centers)

    # Validity check
    valid = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)

    return A, t, valid


def _batch_inlier_counts(
    src_f32: np.ndarray,
    dst_f32: np.ndarray,
    A_f32: np.ndarray,
    t_f32: np.ndarray,
    max_res_sq: float,
    iter_chunk: int,
) -> np.ndarray:
    """
    Compute inlier counts for K transforms against N points, in memory-safe chunks.

    All inputs should be float32 for speed.

    Args:
        src_f32: (N, 2)
        dst_f32: (N, 2)
        A_f32:   (K, 2, 2)
        t_f32:   (K, 2)
        max_res_sq: squared max residual threshold
        iter_chunk: how many transforms to evaluate simultaneously

    Returns:
        counts: (K,) int32 array of inlier counts per transform
    """
    K = A_f32.shape[0]
    counts = np.zeros(K, dtype=np.int32)

    for start in range(0, K, iter_chunk):
        end = min(start + iter_chunk, K)

        # predicted[c, n, :] = src[n, :] @ A[c, :, :].T + t[c, :]
        # shape: (C, N, 2)
        predicted = np.einsum('nj,cij->cni', src_f32, A_f32[start:end])
        predicted += t_f32[start:end, None, :]

        # Squared residuals: (C, N)
        diff = dst_f32[None, :, :] - predicted
        sq_dist = (diff * diff).sum(axis=-1)

        counts[start:end] = (sq_dist <= max_res_sq).sum(axis=1)

    return counts


# ==============================================================================
# Vectorized RANSAC
# ==============================================================================

def ransac_estimate_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    max_residual: float = 5.0,
    n_iterations: int = 1000,
    min_samples: int = 3,
    stop_inlier_ratio: float = 0.5,
    allow_scaling: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, np.ndarray]:
    """
    Fully vectorized RANSAC transform estimation.

    - All K sample transforms estimated in one batched SVD call.
    - Residuals evaluated in memory-safe float32 chunks.
    - No Python per-iteration loop.

    Returns:
        A: 2x2 transformation matrix (None if failed)
        t: 2x1 translation vector (None if failed)
        rotation_deg: rotation angle
        inlier_mask: boolean mask of inlier points (full point set)
    """
    n_points = len(src_points)

    logger.info(f"    [RANSAC] {n_points:,} point pairs, {n_iterations} iters, "
                f"mode={'SIMILARITY' if allow_scaling else 'RIGID'}")

    if n_points < min_samples:
        return None, None, 0.0, np.zeros(n_points, dtype=bool)

    # ── Generate ALL sample index sets at once ────────────────────────────────
    # For min_samples=4 from 160k points, collision probability per row is ~0.004%.
    # Rows with duplicate indices produce degenerate transforms → 0 inliers → ignored.
    all_samples = np.column_stack([
        np.random.randint(0, n_points, size=n_iterations)
        for _ in range(min_samples)
    ])  # (K, min_samples)

    # ── Batch estimate all K transforms ───────────────────────────────────────
    src_samples = src_points[all_samples]  # (K, M, 2)
    dst_samples = dst_points[all_samples]  # (K, M, 2)

    A_all, t_all, valid = _batch_estimate_rigid_transforms(
        src_samples, dst_samples, allow_scaling=allow_scaling
    )

    n_valid = int(valid.sum())
    if n_valid == 0:
        logger.info(f"    [RANSAC] All {n_iterations} transforms were degenerate")
        return None, None, 0.0, np.zeros(n_points, dtype=bool)

    # ── Evaluate residuals in float32 chunks ──────────────────────────────────
    src_f32 = src_points.astype(np.float32)
    dst_f32 = dst_points.astype(np.float32)
    A_f32 = A_all.astype(np.float32)
    t_f32 = t_all.astype(np.float32)
    max_res_sq = np.float32(max_residual ** 2)

    # Size chunks to use ~200 MB of temp memory
    bytes_per_iter = n_points * 20  # (C,N,2) predicted + diff + (C,N) sq_dist, float32
    iter_chunk = max(10, min(n_iterations, int(200e6 / max(bytes_per_iter, 1))))

    counts = _batch_inlier_counts(src_f32, dst_f32, A_f32, t_f32, max_res_sq, iter_chunk)

    # Mask out invalid transforms
    counts[~valid] = 0

    # ── Find best ─────────────────────────────────────────────────────────────
    best_idx = int(np.argmax(counts))
    best_count = int(counts[best_idx])

    if best_count == 0:
        logger.info(f"    [RANSAC] No inliers found in {n_iterations} iterations")
        return None, None, 0.0, np.zeros(n_points, dtype=bool)

    logger.info(f"    [RANSAC] Best: {best_count}/{n_points} inliers "
                f"({100 * best_count / n_points:.1f}%)")

    best_A = A_all[best_idx]  # (2, 2), float64
    best_t = t_all[best_idx]  # (2,),   float64

    # ── Refine on full point set (float64 for precision) ──────────────────────
    full_residuals = np.linalg.norm(
        dst_points - (best_A @ src_points.T).T - best_t, axis=1
    )
    full_inliers = full_residuals <= max_residual
    n_full = int(full_inliers.sum())

    if n_full >= min_samples:
        try:
            # Re-estimate from all inliers
            src_in = src_points[full_inliers]
            dst_in = dst_points[full_inliers]

            src_c = src_in - src_in.mean(axis=0)
            dst_c = dst_in - dst_in.mean(axis=0)
            H = src_c.T @ dst_c
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            if allow_scaling:
                scale = np.sum(S) / max(np.sum(src_c ** 2), 1e-12)
                A_ref = scale * R
            else:
                A_ref = R

            t_ref = dst_in.mean(axis=0) - A_ref @ src_in.mean(axis=0)
            rot_ref = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            # Recompute inliers with refined transform
            ref_residuals = np.linalg.norm(
                dst_points - (A_ref @ src_points.T).T - t_ref, axis=1
            )
            ref_inliers = ref_residuals <= max_residual
            n_refined = int(ref_inliers.sum())

            logger.info(f"    [RANSAC] Refined: {n_refined}/{n_points} inliers "
                        f"({100 * n_refined / n_points:.1f}%)")
            return A_ref, t_ref, rot_ref, ref_inliers

        except Exception as e:
            logger.info(f"    [RANSAC] Refinement failed: {e}")

    rot_deg = np.degrees(np.arctan2(best_A[1, 0], best_A[0, 0]))
    logger.info(f"    [RANSAC] Full set: {n_full}/{n_points} inliers "
                f"({100 * n_full / n_points:.1f}%)")
    return best_A, best_t, rot_deg, full_inliers


# ==============================================================================
# Quad-based Geometric Filtering
# ==============================================================================

def extract_quad_centroids(
    quad_indices: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Extract center positions for all quads: mean of 4 vertex positions."""
    return centroids[quad_indices].mean(axis=1)  # (N, 2)


def filter_quads_by_transform(
    ref_quad_indices: np.ndarray,
    tgt_quad_indices: np.ndarray,
    ref_centroids: np.ndarray,
    tgt_centroids: np.ndarray,
    max_residual: float = 5.0,
    ransac_iterations: int = 1000,
    min_inlier_ratio: float = 0.1,
    allow_scaling: bool = False,
    max_rotation_deg: Optional[float] = None,
    max_translation_px: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Use RANSAC to find dominant transformation and filter quad matches.

    Args:
        max_rotation_deg: If set, reject transforms with |rotation| exceeding
                          this many degrees. None = no limit.
        max_translation_px: If set, reject transforms with translation magnitude
                            exceeding this many pixels. None = no limit.

    Returns:
        inlier_mask: boolean mask of quads that fit the dominant transform
        transform_info: dict with transform parameters and statistics
    """
    n_quads = len(ref_quad_indices)

    logger.info(f"  [FILTER] Input: {n_quads:,} quad matches | "
                f"ref={len(ref_centroids)} tgt={len(tgt_centroids)} neurons | "
                f"{'SIMILARITY' if allow_scaling else 'RIGID'}")
    if max_rotation_deg is not None or max_translation_px is not None:
        rot_str = f"{max_rotation_deg:.1f}°" if max_rotation_deg is not None else "none"
        trans_str = f"{max_translation_px:.1f}px" if max_translation_px is not None else "none"
        logger.info(f"  [FILTER] Transform limits: max_rotation={rot_str}, max_translation={trans_str}")

    ref_centers = extract_quad_centroids(ref_quad_indices, ref_centroids)
    tgt_centers = extract_quad_centroids(tgt_quad_indices, tgt_centroids)

    A, t, rotation_deg, inlier_mask = ransac_estimate_transform(
        src_points=ref_centers,
        dst_points=tgt_centers,
        max_residual=max_residual,
        n_iterations=ransac_iterations,
        min_samples=4,
        stop_inlier_ratio=0.5,
        allow_scaling=allow_scaling,
    )

    n_inliers = int(np.sum(inlier_mask))
    inlier_ratio = n_inliers / n_quads if n_quads > 0 else 0.0  # raw-quad ratio (diagnostic only)

    # ── Neuron coverage: the meaningful acceptance metric ─────────────────────
    # The raw-quad inlier ratio is combinatorially deflated. Step 2 emits FAR more
    # candidate quad matches than there are true neuron correspondences (e.g. ~380k
    # quads for ~550 matchable neurons), so even a perfect rigid transform scores a
    # low %. What matters is how many DISTINCT neurons are recovered with a
    # geometrically consistent (inlier) quad — that is coverage, computed here.
    n_matchable = min(len(ref_centroids), len(tgt_centroids))
    if n_inliers > 0:
        n_covered = int(np.unique(ref_quad_indices[inlier_mask].ravel()).size)
    else:
        n_covered = 0
    neuron_coverage = min(n_covered / n_matchable, 1.0) if n_matchable > 0 else 0.0

    if A is not None and n_inliers > 0:
        residuals = np.linalg.norm(
            tgt_centers - (A @ ref_centers.T).T - t, axis=1
        )
        inlier_residuals = residuals[inlier_mask]

        scale = np.sqrt(max(np.linalg.det(A), 0.0))
        translation_mag = np.linalg.norm(t)

        logger.info(f"  [FILTER] Result: {n_inliers:,}/{n_quads:,} inlier quads "
                     f"({100*inlier_ratio:.1f}% of quads) | coverage {n_covered}/{n_matchable} "
                     f"neurons ({100*neuron_coverage:.1f}%) | "
                     f"rot={rotation_deg:.2f}° trans=({t[0]:.1f},{t[1]:.1f})px scale={scale:.4f} "
                     f"median_resid={np.median(inlier_residuals):.2f}px")

        # ── Enforce rotation / translation limits ─────────────────────────────
        rejected_reason = None
        if max_rotation_deg is not None and abs(rotation_deg) > max_rotation_deg:
            rejected_reason = (f"rotation {abs(rotation_deg):.2f}° exceeds "
                               f"max_rotation_deg={max_rotation_deg:.1f}°")
        elif max_translation_px is not None and translation_mag > max_translation_px:
            rejected_reason = (f"translation {translation_mag:.1f}px exceeds "
                               f"max_translation_px={max_translation_px:.1f}px")

        if rejected_reason is not None:
            logger.warning(f"  [FILTER] REJECTED — {rejected_reason}")
            transform_info = {
                'n_quads_total': int(n_quads),
                'n_inliers': 0,
                'inlier_ratio': 0.0,
                'rejected_reason': rejected_reason,
                'rotation_deg': float(rotation_deg),
                'translation_magnitude': float(translation_mag),
                'scale': float(scale),
                'transform_matrix': A.tolist(),
                'transform_translation': t.tolist(),
            }
            return np.zeros(n_quads, dtype=bool), transform_info

        # Quality warnings
        if abs(rotation_deg) > 20:
            logger.warning(f"  [FILTER] LARGE ROTATION ({abs(rotation_deg):.1f}° > 20°)")
        if translation_mag > 100:
            logger.warning(f"  [FILTER] LARGE TRANSLATION ({translation_mag:.1f}px > 100px)")
        if allow_scaling and (scale < 0.8 or scale > 1.2):
            logger.warning(f"  [FILTER] UNREALISTIC SCALE ({scale:.3f})")

        transform_info = {
            'n_quads_total': int(n_quads),
            'n_inliers': n_inliers,
            'inlier_ratio': float(inlier_ratio),
            'neuron_coverage': float(neuron_coverage),
            'n_neurons_covered': int(n_covered),
            'n_matchable_neurons': int(n_matchable),
            'translation_x': float(t[0]),
            'translation_y': float(t[1]),
            'translation_magnitude': float(translation_mag),
            'rotation_deg': float(rotation_deg),
            'scale': float(scale),
            'mean_residual': float(np.mean(inlier_residuals)),
            'median_residual': float(np.median(inlier_residuals)),
            'max_residual_threshold': float(max_residual),
            'transform_matrix': A.tolist(),
            'transform_translation': t.tolist(),
        }
    else:
        logger.warning(f"  [FILTER] RANSAC FAILED — no valid transform found")
        transform_info = {
            'n_quads_total': int(n_quads),
            'n_inliers': 0,
            'inlier_ratio': 0.0,
            'transform_matrix': None,
        }

    # ── Acceptance gate: neuron coverage, NOT raw-quad ratio ──────────────────
    # `min_inlier_ratio` is now the minimum fraction of MATCHABLE NEURONS that must
    # be recovered with a geometrically consistent quad (coverage). The old gate
    # divided inliers by the candidate-quad count, which is combinatorially inflated
    # and rejected good registrations (see note above). Setting rejected_reason here
    # also keeps the summary from mislabeling these as RANSAC failures / "noise".
    if neuron_coverage < min_inlier_ratio:
        reason = (f"neuron coverage {neuron_coverage:.1%} ({n_covered}/{n_matchable}) "
                  f"< min {min_inlier_ratio:.1%}")
        logger.warning(f"  [FILTER] Rejected: {reason}")
        transform_info['rejected_reason'] = transform_info.get('rejected_reason') or reason
        transform_info['n_inliers'] = 0
        transform_info['inlier_ratio'] = 0.0
        return np.zeros(n_quads, dtype=bool), transform_info

    logger.info(f"  [FILTER] Accepted (coverage {neuron_coverage:.1%} "
                f"≥ min {min_inlier_ratio:.1%}; {n_inliers:,} inlier quads)")
    return inlier_mask, transform_info


# ==============================================================================
# Session Pair Processing
# ==============================================================================

def process_session_pair(
    match_file: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """Process one session pair: load descriptor matches, apply RANSAC filtering."""
    try:
        data = np.load(match_file, allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load {match_file}: {e}")
        return None

    animal_id = decode_string_field(data['animal_id'])
    pair_name = decode_string_field(data['pair_name'])
    ref_session = decode_string_field(data['ref_session'])
    target_session = decode_string_field(data['target_session'])

    ref_centroids = data['ref_centroids']
    tgt_centroids = data['target_centroids']
    match_indices = data['match_indices']

    n_descriptor_matches = len(match_indices)

    logger.info(f"  [{ref_session} → {target_session}] "
                f"{len(ref_centroids)}→{len(tgt_centroids)} neurons, "
                f"{n_descriptor_matches:,} descriptor matches")

    if n_descriptor_matches == 0:
        logger.warning(f"  No descriptor matches for {pair_name}")
        return None

    ref_quad_indices = match_indices[:, :4].astype(int)
    tgt_quad_indices = match_indices[:, 4:].astype(int)

    # Read optional transform limits from config (None = no limit)
    max_rotation_deg = getattr(config, 'ransac_max_rotation_deg', None)
    max_translation_px = getattr(config, 'ransac_max_translation_px', None)

    inlier_mask, transform_info = filter_quads_by_transform(
        ref_quad_indices=ref_quad_indices,
        tgt_quad_indices=tgt_quad_indices,
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        max_residual=config.ransac_max_residual,
        ransac_iterations=config.ransac_iterations,
        min_inlier_ratio=config.ransac_min_inlier_ratio,
        allow_scaling=config.ransac_allow_scaling,
        max_rotation_deg=max_rotation_deg,
        max_translation_px=max_translation_px,
    )

    filtered_match_indices = match_indices[inlier_mask]
    n_filtered = len(filtered_match_indices)

    logger.info(f"  [{pair_name}] Filtered: {n_descriptor_matches:,} → {n_filtered:,} quads "
                f"({100 * n_filtered / n_descriptor_matches:.1f}%)")

    result = {
        'animal_id': animal_id,
        'pair_name': pair_name,
        'ref_session': ref_session,
        'target_session': target_session,
        'n_ref_neurons': int(len(ref_centroids)),
        'n_target_neurons': int(len(tgt_centroids)),
        'n_descriptor_matches': int(n_descriptor_matches),
        'n_geometric_inliers': int(n_filtered),
        'filtering_ratio': float(n_filtered / n_descriptor_matches) if n_descriptor_matches > 0 else 0.0,
        **transform_info,
    }

    output_file = output_dir / f"{pair_name}_filtered_matches.npz"
    np.savez_compressed(
        output_file,
        animal_id=animal_id,
        pair_name=pair_name,
        ref_session=ref_session,
        target_session=target_session,
        ref_centroids=ref_centroids.astype(np.float32),
        tgt_centroids=tgt_centroids.astype(np.float32),
        n_ref_neurons=len(ref_centroids),
        n_target_neurons=len(tgt_centroids),
        n_descriptor_matches=n_descriptor_matches,
        match_indices=filtered_match_indices.astype(np.int32),
        n_matches=n_filtered,
        transform_matrix=np.array(transform_info['transform_matrix']) if transform_info.get('transform_matrix') else np.eye(2),
        transform_translation=np.array(transform_info['transform_translation']) if transform_info.get('transform_translation') else np.zeros(2),
        rotation_deg=transform_info.get('rotation_deg', 0.0),
        scale=transform_info.get('scale', 1.0),
        max_residual_threshold=config.ransac_max_residual,
    )

    return result


# ==============================================================================
# Chunked Parallel Worker
# ==============================================================================

def _chunk_worker(args):
    """Process a chunk of match files in one worker — single setup_logging call."""
    match_file_strs, output_dir_str, ransac_params, log_dir_str, verbose, chunk_id = args

    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=verbose)

    child_logger = logging.getLogger("neuron_mapping_ransac")
    child_logger.info(f"[Worker {chunk_id}] Starting chunk of {len(match_file_strs)} pairs")

    output_dir = Path(output_dir_str)

    config = PipelineConfig(
        input_dir=".",
        output_dir=str(output_dir.parent),
        verbose=verbose,
    )
    config.ransac_max_residual = ransac_params['max_residual']
    config.ransac_iterations = ransac_params['iterations']
    config.ransac_min_inlier_ratio = ransac_params['min_inlier_ratio']
    config.ransac_allow_scaling = ransac_params['allow_scaling']
    config.ransac_max_rotation_deg = ransac_params.get('max_rotation_deg', None)
    config.ransac_max_translation_px = ransac_params.get('max_translation_px', None)

    results = []
    for mf_str in match_file_strs:
        result = process_session_pair(Path(mf_str), output_dir, config)
        if result:
            results.append(result)

    child_logger.info(f"[Worker {chunk_id}] Done — {len(results)}/{len(match_file_strs)} pairs")
    return results


# ==============================================================================
# Helper Functions for GUI Compatibility
# ==============================================================================

def discover_animals(step2_dir: Path, pattern: str = "*_matches_light.npz", verbose: bool = False) -> List[str]:
    """Discover unique animal IDs from Step 2 match files."""
    match_files = sorted(step2_dir.glob(pattern))
    animals = set()
    for f in match_files:
        try:
            data = np.load(f, allow_pickle=False)
            animal_id = decode_string_field(data['animal_id'])
            animals.add(animal_id)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not read animal ID from {f.name}: {e}")
    return sorted(animals)


def load_animal_data(animal_id: str, step2_dir: Path) -> List[Path]:
    """Load all match files for a specific animal."""
    return sorted(step2_dir.glob(f"{animal_id}_*_matches_light.npz"))


def run_step_2_5_all_animals(input_dir: str, output_dir: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Alias for backward compatibility."""
    return run_step_2_5_ransac(input_dir, output_dir, verbose=verbose)


def sweep_shape_threshold_for_animal(
    output_dir: str, animal_id: str,
    shape_thresholds=None, verbose: bool = True,
    max_pairs: Optional[int] = None, max_matches_per_pair: Optional[int] = None,
) -> Dict[str, Any]:
    """GUI-compatible wrapper for processing one animal with RANSAC."""
    output_path = Path(output_dir)
    step2_dir = output_path / "step_2_results"
    step2_5_dir = ensure_output_dir(output_dir, 2.5, verbose=False)

    config = PipelineConfig(input_dir=output_dir, output_dir=output_dir, verbose=verbose)

    match_files = sorted(step2_dir.glob(f"{animal_id}_*_matches_light.npz"))
    if not match_files:
        logger.warning(f"No match files found for animal {animal_id}")
        return {'animal_id': animal_id, 'n_pairs': 0, 'n_geometric_inliers': 0, 'error': 'No match files found'}

    logger.info(f"Processing {len(match_files)} session pairs for animal {animal_id}")

    pair_results = []
    total_desc, total_inliers = 0, 0
    for mf in match_files:
        result = process_session_pair(mf, step2_5_dir, config)
        if result:
            pair_results.append(result)
            total_desc += result.get('n_descriptor_matches', 0)
            total_inliers += result.get('n_geometric_inliers', 0)

    return {
        'animal_id': animal_id,
        'n_pairs': len(pair_results),
        'n_descriptor_matches': total_desc,
        'n_geometric_inliers': total_inliers,
        'filtering_ratio': total_inliers / total_desc if total_desc > 0 else 0.0,
        'optimal_threshold': total_inliers / total_desc if total_desc > 0 else 0.0,
    }


def save_all_animals_summary(output_dir: str, results: List[Dict[str, Any]]):
    """Save summary JSON file for all animals."""
    from utilities.step_info import get_step_output_dir
    output_path = get_step_output_dir(2.5, output_dir)
    save_json_summary(results, output_path / "all_animals_summary.json")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_step_2_5_ransac(
    input_dir: str,
    output_dir: str,
    ransac_max_residual: float = 5.0,
    ransac_iterations: int = 1000,
    ransac_min_inlier_ratio: float = 0.05,
    ransac_allow_scaling: bool = False,
    ransac_max_rotation_deg: Optional[float] = None,
    ransac_max_translation_px: Optional[float] = None,
    processes: Optional[int] = None,
    verbose: bool = True,
    skip_existing: bool = False,   # default off so direct callers still recompute;
                                   # the GUI forwards config.skip_existing explicitly
) -> List[Dict[str, Any]]:
    """
    Run Step 2.5: RANSAC-based geometric filtering of descriptor matches.
    Parallelized via chunked ProcessPoolExecutor with vectorized RANSAC.

    Args:
        ransac_max_rotation_deg: Reject transforms with |rotation| > this (degrees).
                                 None = no limit (default).
        ransac_max_translation_px: Reject transforms with translation magnitude > this (pixels).
                                   None = no limit (default).
        skip_existing: If True, skip pairs whose *_filtered_matches.npz already
                       exists and carry their prior summary entries forward.
    """
    log_dir = Path(output_dir) / "logs_step2_5_ransac"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=verbose)

    output_path = Path(output_dir)
    step2_dir = output_path / "step_2_results"

    if not step2_dir.exists():
        logger.error(f"Step 2 results not found: {step2_dir}")
        return []

    match_files = sorted(step2_dir.glob("*_matches_light.npz"))
    if not match_files:
        logger.error(f"No match files found in {step2_dir}")
        return []

    rot_limit_str = f"{ransac_max_rotation_deg:.1f}°" if ransac_max_rotation_deg is not None else "none"
    trans_limit_str = f"{ransac_max_translation_px:.1f}px" if ransac_max_translation_px is not None else "none"

    logger.info(f"{'#'*80}")
    logger.info(f"# STEP 2.5: RANSAC GEOMETRIC FILTERING (VECTORIZED)")
    logger.info(f"{'#'*80}")
    logger.info(f"Found {len(match_files)} session pairs to process")
    logger.info(f"RANSAC Config: max_residual={ransac_max_residual:.1f}px, "
                f"iterations={ransac_iterations}, min_inlier_ratio={ransac_min_inlier_ratio:.2%}, "
                f"transform={'SIMILARITY' if ransac_allow_scaling else 'RIGID (scale=1.0)'}")
    logger.info(f"Transform limits: max_rotation={rot_limit_str}, max_translation={trans_limit_str}")

    step2_5_dir = ensure_output_dir(output_dir, 2.5, verbose=False)

    # ── Honor skip_existing: skip pairs whose filtered output already exists ───
    # Step 3 reads the *_filtered_matches.npz files straight from disk, so the
    # skipped pairs' data is preserved. We also carry their prior summary entries
    # forward so all_pairs_summary.json stays complete after a partial re-run.
    prior_results: List[Dict[str, Any]] = []
    if skip_existing:
        pending, done_stems = [], []
        for mf in match_files:
            stem = mf.stem.replace('_matches_light', '')
            if (step2_5_dir / f"{stem}_filtered_matches.npz").exists():
                done_stems.append(stem)
            else:
                pending.append(mf)
        n_skipped = len(match_files) - len(pending)
        if n_skipped:
            logger.info(f"skip_existing: {n_skipped} pair(s) already filtered; "
                        f"{len(pending)} remaining")
            summary_path = step2_5_dir / "all_pairs_summary.json"
            if summary_path.exists():
                try:
                    import json as _json
                    with open(summary_path) as _f:
                        prior = _json.load(_f)
                    done_set = set(done_stems)
                    prior_results = [r for r in prior
                                     if isinstance(r, dict) and r.get('pair_name') in done_set]
                except Exception as _e:
                    logger.warning(f"skip_existing: could not reuse prior summary: {_e}")
        match_files = pending
        if not match_files:
            logger.info("skip_existing: all pairs already filtered — nothing to do")
            save_json_summary(prior_results, step2_5_dir / "all_pairs_summary.json")
            return prior_results

    config = PipelineConfig(input_dir=input_dir, output_dir=output_dir, verbose=verbose)
    config.ransac_max_residual = ransac_max_residual
    config.ransac_iterations = ransac_iterations
    config.ransac_min_inlier_ratio = ransac_min_inlier_ratio
    config.ransac_allow_scaling = ransac_allow_scaling
    config.ransac_max_rotation_deg = ransac_max_rotation_deg
    config.ransac_max_translation_px = ransac_max_translation_px

    # ── Diagnostic: time & profile ONE pair ───────────────────────────────────
    import psutil

    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e9
    avail_before = psutil.virtual_memory().available / 1e9

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"DIAGNOSTIC: Running 1 pair to measure baseline...")
    logger.info(f"  System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB total, "
                f"{avail_before:.1f} GB available")
    logger.info(f"  Parent process RSS: {ram_before:.2f} GB")

    t_diag_start = time.perf_counter()
    test_result = process_session_pair(match_files[0], step2_5_dir, config)
    t_diag_end = time.perf_counter()

    ram_after = proc.memory_info().rss / 1e9
    pair_baseline = t_diag_end - t_diag_start
    pair_ram_delta = max(0.05, ram_after - ram_before)

    logger.info(f"  Time: {pair_baseline:.3f}s")
    logger.info(f"  RAM delta: {pair_ram_delta * 1000:.0f} MB ({ram_before:.2f} → {ram_after:.2f} GB)")
    if test_result:
        logger.info(f"  {test_result['n_descriptor_matches']:,} descriptors → "
                    f"{test_result['n_geometric_inliers']:,} inliers "
                    f"({test_result['filtering_ratio']:.1%})")

    # Delete test output so it gets re-processed in the batch
    test_stem = match_files[0].stem.replace('_matches_light', '')
    test_output = step2_5_dir / f"{test_stem}_filtered_matches.npz"
    test_output.unlink(missing_ok=True)

    # ── Size the worker pool from measured data ──────────────────────────────
    n_cpus = multiprocessing.cpu_count()
    avail_gb = psutil.virtual_memory().available / 1e9

    if processes:
        n_workers = processes
        max_by_ram = max_by_cpu = processes
    else:
        per_worker_gb = pair_ram_delta * 3.0
        max_by_ram = max(1, int(avail_gb * 0.8 / per_worker_gb))
        max_by_cpu = max(1, min(n_cpus // 4, 16))
        n_workers = min(max_by_ram, max_by_cpu)

    n_workers = min(n_workers, len(match_files), 61)

    # ── Chunk the pairs ──────────────────────────────────────────────────────
    match_strs = [str(mf) for mf in match_files]
    chunk_size = max(1, len(match_strs) // n_workers)
    chunks = [match_strs[i:i + chunk_size] for i in range(0, len(match_strs), chunk_size)]
    n_workers = min(n_workers, len(chunks))

    est_total_s = pair_baseline * len(match_files) / n_workers
    est_ram_gb = n_workers * pair_ram_delta * 3.0

    logger.info(f"{'='*60}")
    logger.info(f"PLAN:")
    logger.info(f"  Workers: {n_workers} (CPUs={n_cpus}, max_by_ram={max_by_ram}, max_by_cpu={max_by_cpu})")
    logger.info(f"  Chunks: {len(chunks)} × ~{chunk_size} pairs/chunk")
    logger.info(f"  Est. RAM usage: ~{est_ram_gb:.1f} GB ({n_workers} × {pair_ram_delta*3*1000:.0f} MB/worker)")
    logger.info(f"  Est. time: ~{est_total_s:.0f}s ({pair_baseline:.2f}s/pair × {len(match_files)} / {n_workers})")
    logger.info(f"  Available RAM: {avail_gb:.1f} GB → headroom: {avail_gb - est_ram_gb:.1f} GB")
    logger.info(f"{'='*60}")
    logger.info(f"")

    ransac_params = {
        'max_residual': ransac_max_residual,
        'iterations': ransac_iterations,
        'min_inlier_ratio': ransac_min_inlier_ratio,
        'allow_scaling': ransac_allow_scaling,
        'max_rotation_deg': ransac_max_rotation_deg,
        'max_translation_px': ransac_max_translation_px,
    }

    worker_args = [
        (chunk, str(step2_5_dir), ransac_params, str(log_dir), verbose, i)
        for i, chunk in enumerate(chunks)
    ]

    # ── Run in parallel ──────────────────────────────────────────────────────
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_chunk_worker, args) for args in worker_args]

        for fut in as_completed(futures):
            try:
                chunk_results = fut.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Chunk processing error: {e}")

            elapsed = time.time() - t0
            logger.info(f"  Progress: {len(results)}/{len(match_files)} pairs "
                        f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    logger.info(f"All {len(match_files)} pairs processed in {elapsed:.1f}s "
                f"({elapsed / len(match_files):.2f}s/pair avg)")

    # ── Merge carried-forward (skipped) pairs back in before summarizing ──────
    if prior_results:
        logger.info(f"skip_existing: merging {len(prior_results)} previously-"
                    f"filtered pair(s) into the summary")
        results = prior_results + results

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_file = step2_5_dir / "all_pairs_summary.json"
    save_json_summary(results, summary_file)

    logger.info(f"{'#'*80}")
    logger.info(f"# SUMMARY")
    logger.info(f"{'#'*80}")
    logger.info(f"Processed {len(results)} session pairs")

    if results:
        total_desc = sum(r['n_descriptor_matches'] for r in results)
        total_filtered = sum(r['n_geometric_inliers'] for r in results)

        logger.info(f"Total descriptor matches: {total_desc:,}")
        logger.info(f"Total geometric inliers: {total_filtered:,}")
        logger.info(f"Overall quad filtering ratio: {100 * total_filtered / total_desc:.1f}% "
                    f"(expected low — quad candidates vastly outnumber true neuron pairs)")

        coverages = [r.get('neuron_coverage', 0.0) for r in results]
        avg_coverage = float(np.mean(coverages)) if coverages else 0.0
        logger.info(f"Avg neuron coverage: {100 * avg_coverage:.1f}% (the acceptance metric)")

        if avg_coverage < 0.05:
            logger.warning(f"Low neuron coverage ({100 * avg_coverage:.1f}%) — few neurons "
                          f"recovered a geometrically consistent match; check alignment "
                          f"and centroids, not just quad counts")

            rotations = [abs(r.get('rotation_deg', 0)) for r in results if r.get('rotation_deg') is not None]
            translations = [r.get('translation_magnitude', 0) for r in results if r.get('translation_magnitude') is not None]
            if rotations:
                logger.info(f"  Avg rotation: {np.mean(rotations):.1f}° (median: {np.median(rotations):.1f}°)")
            if translations:
                logger.info(f"  Avg translation: {np.mean(translations):.1f}px (median: {np.median(translations):.1f}px)")

            # Check if max_residual is very tight
            if ransac_max_residual <= 2.0:
                logger.info(f"  ⚠ max_residual={ransac_max_residual:.1f}px is very tight — "
                            f"RANSAC may discard valid matches with small alignment error. "
                            f"Try 3.0–5.0px if sessions have slight drift.")

            # Check if all pairs were rejected by limits vs genuinely no inliers
            n_rejected_by_limits = sum(1 for r in results if r.get('rejected_reason'))
            n_ransac_failed = sum(1 for r in results
                                 if not r.get('rejected_reason') and r.get('n_geometric_inliers', 0) == 0)
            if n_rejected_by_limits > 0 and n_rejected_by_limits == len(results):
                logger.info(f"  ⚠ ALL {n_rejected_by_limits} pairs were rejected by acceptance "
                            f"criteria (neuron coverage and/or transform limits), NOT by RANSAC "
                            f"finding zero inliers. See each pair's 'rejected_reason' below; "
                            f"lower min_inlier_ratio (coverage) or relax max_rotation_deg / "
                            f"max_translation_px as appropriate.")
            elif n_ransac_failed > 0:
                logger.info(f"  {n_ransac_failed} pairs had 0 RANSAC inliers (before limit checks) — "
                            f"descriptor matches are likely noise for these pairs.")

            logger.info(f"  Suggestions: increase descriptor_threshold in Step 2, "
                        f"check session alignment, verify centroids")
        else:
            logger.info(f"Data quality looks good (avg neuron coverage {100 * avg_coverage:.1f}%)")

        # Report rejected pairs
        rejected = [r for r in results if r.get('rejected_reason')]
        if rejected:
            logger.info(f"  {len(rejected)} pairs rejected by transform limits:")
            for r in rejected:
                logger.info(f"    {r['pair_name']}: {r['rejected_reason']}")

            # ── What-if diagnostic: suggest limits that would pass pairs ──────
            logger.info(f"")
            logger.info(f"  {'─'*50}")
            logger.info(f"  WHAT-IF: Suggested parameter relaxations")
            logger.info(f"  {'─'*50}")

            rej_rotations = sorted([abs(r.get('rotation_deg', 0)) for r in rejected
                                    if r.get('rotation_deg') is not None])
            rej_translations = sorted([r.get('translation_magnitude', 0) for r in rejected
                                       if r.get('translation_magnitude') is not None])

            if rej_rotations and ransac_max_rotation_deg is not None:
                logger.info(f"  Current max_rotation_deg = {ransac_max_rotation_deg:.1f}°")
                logger.info(f"  Rejected rotation range: {rej_rotations[0]:.1f}° – {rej_rotations[-1]:.1f}°")
                # Show how many pairs pass at various thresholds
                for candidate in sorted(set([15, 25, 45, 90, 180])):
                    if candidate <= ransac_max_rotation_deg:
                        continue
                    n_would_pass = sum(1 for rot in rej_rotations if rot <= candidate)
                    if n_would_pass > 0:
                        logger.info(f"    → max_rotation_deg={candidate:>3}° would recover "
                                    f"{n_would_pass}/{len(rejected)} rejected pairs")

            if rej_translations and ransac_max_translation_px is not None:
                logger.info(f"  Current max_translation_px = {ransac_max_translation_px:.1f}px")
                logger.info(f"  Rejected translation range: {rej_translations[0]:.1f}px – {rej_translations[-1]:.1f}px")
                for candidate in sorted(set([25, 50, 100, 200, 500])):
                    if candidate <= ransac_max_translation_px:
                        continue
                    n_would_pass = sum(1 for t_mag in rej_translations if t_mag <= candidate)
                    if n_would_pass > 0:
                        logger.info(f"    → max_translation_px={candidate:>3}px would recover "
                                    f"{n_would_pass}/{len(rejected)} rejected pairs")

            if ransac_max_translation_px is not None and ransac_max_translation_px == 0.0:
                logger.warning(f"  ⚠ max_translation_px=0.0 rejects ALL pairs — "
                               f"even perfectly aligned sessions have sub-pixel drift. "
                               f"Set to None (no limit) or a reasonable value like 50–100px.")

            logger.info(f"  {'─'*50}")
            logger.info(f"")

        # Group by animal
        animals = {}
        for r in results:
            aid = r['animal_id']
            if aid not in animals:
                animals[aid] = []
            animals[aid].append(r)

        for aid, animal_results in sorted(animals.items()):
            n_pairs = len(animal_results)
            avg_inliers = np.mean([r['n_geometric_inliers'] for r in animal_results])
            avg_rotation = np.mean([abs(r.get('rotation_deg', 0)) for r in animal_results if r.get('rotation_deg')])
            avg_translation = np.mean([r.get('translation_magnitude', 0) for r in animal_results if r.get('translation_magnitude')])
            logger.info(f"  {aid}: {n_pairs} pairs, avg {avg_inliers:.0f} inliers/pair, "
                        f"rot={avg_rotation:.1f}°, trans={avg_translation:.1f}px")

    return results


# Alias for GUI compatibility
def run_step_2_5_descriptor_sweep(
    input_dir: str, output_dir: str,
    ransac_allow_scaling: bool = False, verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Main entry point for Step 2.5 (backward compatible name)."""
    return run_step_2_5_ransac(input_dir, output_dir, ransac_allow_scaling=ransac_allow_scaling, verbose=verbose)
