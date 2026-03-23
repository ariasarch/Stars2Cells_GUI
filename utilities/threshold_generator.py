"""
threshold_generator.py

Centralized distance metrics, filtering, and threshold calibration.
All geometric constraints, similarity measures, quality metrics, and C value
computation in one place.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path


# ==============================================================================
# Quad Descriptor Distance (Astrometry.net style)
# ==============================================================================

def compute_quad_descriptor(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute normalized quad descriptor [xC, yC, xD, yD].
    
    Parameters
    ----------
    points : (4, 2) array
        Quad vertices in pixel space
        
    Returns
    -------
    descriptor : (4,) array or None
        Normalized descriptor, or None if degenerate
    """
    assert points.shape == (4, 2), f"Expected (4,2), got {points.shape}"
    
    # Find longest diagonal
    dists = []
    pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            d = np.linalg.norm(points[i] - points[j])
            dists.append(d)
            pairs.append((i, j))
    
    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]
    
    if max_dist < 1e-6:
        return None  # Degenerate quad
    
    A_idx, B_idx = pairs[max_idx]
    A = points[A_idx]
    B = points[B_idx]
    
    # Create coordinate frame
    AB = B - A
    ux = AB / max_dist
    uy = np.array([-ux[1], ux[0]])  # perpendicular
    
    # Get other two points
    other_idx = [i for i in range(4) if i not in (A_idx, B_idx)]
    C_idx, D_idx = other_idx[0], other_idx[1]
    C = points[C_idx]
    D = points[D_idx]
    
    # Project into AB frame
    AC = C - A
    AD = D - A
    
    xC = np.dot(AC, ux) / max_dist
    yC = np.dot(AC, uy) / max_dist
    xD = np.dot(AD, ux) / max_dist
    yD = np.dot(AD, uy) / max_dist
    
    # Canonical ordering: xC <= xD
    if xC > xD:
        xC, xD = xD, xC
        yC, yD = yD, yC
    
    # Mirror if needed: xC + xD <= 1
    if xC + xD > 1.0:
        # Swap A and B, recompute
        ux_flip = -ux
        uy_flip = -uy
        
        BC = C - B
        BD = D - B
        
        xC = np.dot(BC, ux_flip) / max_dist
        yC = np.dot(BC, uy_flip) / max_dist
        xD = np.dot(BD, ux_flip) / max_dist
        yD = np.dot(BD, uy_flip) / max_dist
        
        if xC > xD:
            xC, xD = xD, xC
            yC, yD = yD, yC
    
    # Check not collapsed onto diagonal
    if abs(yC) < 1e-4 and abs(yD) < 1e-4:
        return None
    
    return np.array([xC, yC, xD, yD], dtype=np.float32)

def compute_quad_descriptors_batch(
    points_batch: np.ndarray,
    min_pairwise_distance: float = 0.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Vectorized batch quad descriptor computation.
    
    Parameters
    ----------
    points_batch : (K, 4, 2) array
        K quads, each with 4 vertices
    min_pairwise_distance : float
        Minimum allowed distance between any pair of vertices
        
    Returns
    -------
    descriptors : (K', 4) float32 array or None
        Descriptors for valid quads
    valid_indices : (K',) int32 array or None
        Indices of valid quads in original batch
    """
    K = points_batch.shape[0]
    if K == 0:
        return None, None
    
    # Compute pairwise distances for each quad (K, 4, 4)
    diff = points_batch[:, :, None, :] - points_batch[:, None, :, :]
    dist_mats = np.linalg.norm(diff, axis=-1)  # (K, 4, 4)
    
    # Find longest diagonal per quad
    flat = dist_mats.reshape(K, -1)
    max_flat_idx = np.argmax(flat, axis=1)
    A_idx = max_flat_idx // 4
    B_idx = max_flat_idx % 4
    max_dists = flat[np.arange(K), max_flat_idx]
    
    # Filter on min_pairwise_distance + non-zero diagonal
    ut_i, ut_j = np.triu_indices(4, 1)
    pair_dists = dist_mats[:, ut_i, ut_j]  # (K, 6)
    min_pair_d = pair_dists.min(axis=1)    # (K,)
    
    keep = (max_dists > 1e-9) & (min_pair_d >= min_pairwise_distance)
    if not np.any(keep):
        return None, None
    
    A_idx = A_idx[keep]
    B_idx = B_idx[keep]
    max_dists_kept = max_dists[keep]
    pts_keep = points_batch[keep]
    valid_indices = np.where(keep)[0].astype(np.int32)
    
    # Build canonical frames
    batch = np.arange(valid_indices.shape[0])
    A = pts_keep[batch, A_idx]
    B = pts_keep[batch, B_idx]
    AB = B - A
    distAB = max_dists_kept[:, None]
    ux = AB / distAB
    uy = np.stack([-ux[:, 1], ux[:, 0]], axis=1)
    
    # Pick other two vertices
    all_idx = np.array([0, 1, 2, 3])
    C_idx = np.zeros(valid_indices.shape[0], dtype=np.int32)
    D_idx = np.zeros(valid_indices.shape[0], dtype=np.int32)
    for k in range(valid_indices.shape[0]):
        others = all_idx[(all_idx != A_idx[k]) & (all_idx != B_idx[k])]
        C_idx[k], D_idx[k] = others[0], others[1]
    
    C = pts_keep[batch, C_idx]
    D = pts_keep[batch, D_idx]
    AC = C - A
    AD = D - A
    
    xC = np.sum(AC * ux, axis=1) / max_dists_kept
    yC = np.sum(AC * uy, axis=1) / max_dists_kept
    xD = np.sum(AD * ux, axis=1) / max_dists_kept
    yD = np.sum(AD * uy, axis=1) / max_dists_kept
    
    # Canonical ordering: xC <= xD
    swap = xC > xD
    xC, xD = np.where(swap, xD, xC), np.where(swap, xC, xD)
    yC, yD = np.where(swap, yD, yC), np.where(swap, yC, yD)
    
    # Mirror if xC + xD > 1
    flip = (xC + xD) > 1
    if np.any(flip):
        A_flip = B[flip]
        B_flip = A[flip]
        AB_flip = B_flip - A_flip
        ux_flip = AB_flip / max_dists_kept[flip, None]
        uy_flip = np.stack([-ux_flip[:, 1], ux_flip[:, 0]], axis=1)
        
        AC_flip = C[flip] - A_flip
        AD_flip = D[flip] - A_flip
        
        xC_flip = np.sum(AC_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yC_flip = np.sum(AC_flip * uy_flip, axis=1) / max_dists_kept[flip]
        xD_flip = np.sum(AD_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yD_flip = np.sum(AD_flip * uy_flip, axis=1) / max_dists_kept[flip]
        
        swap_flip = xC_flip > xD_flip
        xC_flip_final = np.where(swap_flip, xD_flip, xC_flip)
        xD_flip_final = np.where(swap_flip, xC_flip, xD_flip)
        yC_flip_final = np.where(swap_flip, yD_flip, yC_flip)
        yD_flip_final = np.where(swap_flip, yC_flip, yD_flip)
        
        xC[flip] = xC_flip_final
        xD[flip] = xD_flip_final
        yC[flip] = yC_flip_final
        yD[flip] = yD_flip_final
    
    # Filter collapsed quads
    ok = ~((np.abs(yC) < 1e-4) & (np.abs(yD) < 1e-4))
    if not np.any(ok):
        return None, None
    
    valid_indices = valid_indices[ok]
    descriptors = np.stack(
        [xC[ok], yC[ok], xD[ok], yD[ok]],
        axis=1
    ).astype(np.float32)
    
    return descriptors, valid_indices

def quad_descriptor_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Euclidean distance between quad descriptors.
    
    Parameters
    ----------
    desc1, desc2 : (4,) arrays
        Quad descriptors
        
    Returns
    -------
    float
        Distance in descriptor space
    """
    return float(np.linalg.norm(desc1 - desc2))

# ==============================================================================
# Geometric Filters
# ==============================================================================

def compute_quad_centroid(points: np.ndarray) -> np.ndarray:
    """Compute centroid of 4 points."""
    return np.mean(points, axis=0)

def compute_quad_rotation(points: np.ndarray) -> float:
    """
    Compute rotation angle of quad (degrees).
    
    Uses direction from centroid to first vertex.
    
    Returns
    -------
    float
        Angle in degrees [0, 360)
    """
    centroid = compute_quad_centroid(points)
    v = points[0] - centroid
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    return float((angle_deg + 360) % 360)

def angle_difference_deg(a: float, b: float) -> float:
    """
    Smallest absolute difference between two angles in degrees.
    
    Returns
    -------
    float
        Difference in [0, 180] degrees
    """
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)

def check_centroid_distance(
    pts_ref: np.ndarray,
    pts_tgt: np.ndarray,
    max_distance: float
) -> bool:
    """
    Check if quad centroids are within max distance.
    
    Parameters
    ----------
    pts_ref, pts_tgt : (4, 2) arrays
        Quad vertices
    max_distance : float
        Maximum allowed centroid distance (pixels)
        
    Returns
    -------
    bool
        True if centroids are close enough
    """
    c_ref = compute_quad_centroid(pts_ref)
    c_tgt = compute_quad_centroid(pts_tgt)
    dist = np.linalg.norm(c_ref - c_tgt)
    return dist <= max_distance

def check_rotation_difference(
    pts_ref: np.ndarray,
    pts_tgt: np.ndarray,
    max_rotation_deg: float
) -> bool:
    """
    Check if quad rotations are within tolerance.
    
    Parameters
    ----------
    pts_ref, pts_tgt : (4, 2) arrays
        Quad vertices
    max_rotation_deg : float
        Maximum allowed rotation difference (degrees)
        
    Returns
    -------
    bool
        True if rotations are close enough
    """
    theta_ref = compute_quad_rotation(pts_ref)
    theta_tgt = compute_quad_rotation(pts_tgt)
    diff = angle_difference_deg(theta_ref, theta_tgt)
    return diff <= max_rotation_deg

# ==============================================================================
# Shape Distance (Multi-Component)
# ==============================================================================

def compute_quad_geometry(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute geometric properties of a quadrilateral.
    
    Parameters
    ----------
    points : (4, 2) array
        Quad vertex coordinates
        
    Returns
    -------
    sides : (4,) array
        Side lengths, sorted ascending
    diags : (2,) array
        Diagonal lengths, sorted ascending
    area : float
        Quad area via shoelace formula
    """
    assert points.shape == (4, 2), f"Expected (4,2) array, got {points.shape}"
    
    # Compute all pairwise distances
    diff = points[:, None, :] - points[None, :, :]  # (4, 4, 2)
    dist_mat = np.linalg.norm(diff, axis=-1)  # (4, 4)
    
    # Extract 6 unique distances from upper triangle
    i_idx, j_idx = np.triu_indices(4, 1)
    dists = dist_mat[i_idx, j_idx]
    dists_sorted = np.sort(dists)
    
    # 4 smallest = sides, 2 largest = diagonals
    sides = dists_sorted[:4]
    diags = dists_sorted[4:]
    
    # Area via shoelace formula (order vertices by angle around centroid)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    order = np.argsort(angles)
    poly = points[order]
    
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    return sides, diags, float(area)

def compute_shape_distance(
    pts_ref: np.ndarray,
    pts_tgt: np.ndarray,
    motion_correction_px: float = 10.0,
    rotation_tolerance_deg: float = 5.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute multi-component shape distance between two quads.
    
    Combines three components:
    1. Shape: Scale-invariant comparison of sides, diagonals, and area
    2. Position: Centroid displacement penalty
    3. Rotation: Angular difference penalty with tolerance
    
    Parameters
    ----------
    pts_ref : (4, 2) array
        Reference quad vertices
    pts_tgt : (4, 2) array
        Target quad vertices
    motion_correction_px : float
        Expected motion correction shift (normalizes position penalty)
    rotation_tolerance_deg : float
        Allowed rotation before penalty applies
    eps : float
        Small constant for numerical stability
        
    Returns
    -------
    float
        Combined shape distance (lower = more similar)
    """
    # Component 1: Scale-invariant shape descriptor
    sides_r, diags_r, area_r = compute_quad_geometry(pts_ref)
    sides_t, diags_t, area_t = compute_quad_geometry(pts_tgt)
    
    # Normalize by largest diagonal (scale invariance)
    max_r = max(diags_r.max(), eps)
    max_t = max(diags_t.max(), eps)
    
    sides_r_norm = sides_r / max_r
    diags_r_norm = diags_r / max_r
    area_r_norm = area_r / (max_r ** 2)
    
    sides_t_norm = sides_t / max_t
    diags_t_norm = diags_t / max_t
    area_t_norm = area_t / (max_t ** 2)
    
    # Concatenate into feature vector
    v_r = np.concatenate([sides_r_norm, diags_r_norm, [area_r_norm]])
    v_t = np.concatenate([sides_t_norm, diags_t_norm, [area_t_norm]])
    
    shape_term = np.linalg.norm(v_r - v_t)
    
    # Component 2: Position penalty (centroid displacement)
    c_ref = pts_ref.mean(axis=0)
    c_tgt = pts_tgt.mean(axis=0)
    centroid_dist = float(np.linalg.norm(c_ref - c_tgt))
    
    # Quadratic penalty normalized by expected motion
    mc = max(motion_correction_px, eps)
    position_term = (centroid_dist / mc) ** 2
    
    # Component 3: Rotation penalty with tolerance
    theta_ref = compute_quad_rotation(pts_ref)
    theta_tgt = compute_quad_rotation(pts_tgt)
    
    dtheta = angle_difference_deg(theta_ref, theta_tgt)
    
    if dtheta <= rotation_tolerance_deg:
        rotation_term = 0.0
    else:
        # Smooth penalty beyond tolerance
        rotation_term = ((dtheta - rotation_tolerance_deg) / 90.0) ** 2
    
    # Combine terms with equal weighting
    w_shape = 1.0
    w_pos = 1.0
    w_rot = 1.0
    
    return float(w_shape * shape_term + w_pos * position_term + w_rot * rotation_term)

# ==============================================================================
# Combined Filtering
# ==============================================================================

def filter_quad_match(
    pts_ref: np.ndarray,
    pts_tgt: np.ndarray,
    image_width: int = 640,
    image_height: int = 640,
    max_centroid_distance_pct: float = 5.0,
    max_rotation_deg: float = 5.0,
) -> bool:
    """
    Apply all geometric filters to a quad match.
    
    Parameters
    ----------
    pts_ref, pts_tgt : (4, 2) arrays
        Quad vertices
    image_width, image_height : int
        Image dimensions (pixels)
    max_centroid_distance_pct : float
        Max centroid distance as % of image diagonal
    max_rotation_deg : float
        Max rotation difference (degrees)
        
    Returns
    -------
    bool
        True if match passes all filters
    """
    # Compute max centroid distance
    image_diagonal = np.sqrt(image_width**2 + image_height**2)
    max_centroid_px = (max_centroid_distance_pct / 100.0) * image_diagonal
    
    # Check filters
    if not check_centroid_distance(pts_ref, pts_tgt, max_centroid_px):
        return False
    
    if not check_rotation_difference(pts_ref, pts_tgt, max_rotation_deg):
        return False
    
    return True

# ==============================================================================
# Consistency Filtering
# ==============================================================================

def filter_quad_matches_by_consistency(
    matches: List[Tuple],
    centroids_ref: np.ndarray,
    centroids_target: np.ndarray,
    consistency_threshold: float = 0.8,
) -> List[Tuple]:
    if not matches:
        return []

    # Extract all ref and target indices at once — (M, 4)
    ref_indices_all  = np.array([list(m[0]) for m in matches], dtype=np.int32)   # (M, 4)
    tgt_indices_all  = np.array([list(m[1]) for m in matches], dtype=np.int32)   # (M, 4)

    # Gather centroid coordinates — (M, 4, 2)
    ref_pts_all = centroids_ref[ref_indices_all]    # (M, 4, 2)
    tgt_pts_all = centroids_target[tgt_indices_all] # (M, 4, 2)

    # All 6 pairwise combinations for 4 points
    i_idx, j_idx = np.triu_indices(4, k=1)  # 6 pairs

    # Pairwise distances — (M, 6)
    ref_diffs = ref_pts_all[:, i_idx, :] - ref_pts_all[:, j_idx, :]  # (M, 6, 2)
    tgt_diffs = tgt_pts_all[:, i_idx, :] - tgt_pts_all[:, j_idx, :]  # (M, 6, 2)

    ref_dists = np.linalg.norm(ref_diffs, axis=-1)  # (M, 6)
    tgt_dists = np.linalg.norm(tgt_diffs, axis=-1)  # (M, 6)

    # Valid pairs: ref_dist > 1e-6, require at least 3 per match
    valid_mask = ref_dists > 1e-6  # (M, 6)
    n_valid    = valid_mask.sum(axis=1)  # (M,)

    # Scale ratios — safe divide (invalid pairs get 0)
    scale_ratios = np.where(valid_mask, tgt_dists / np.where(valid_mask, ref_dists, 1.0), 0.0)  # (M, 6)

    # Mean and std over valid pairs only
    valid_counts  = np.maximum(n_valid, 1).astype(np.float32)
    scale_sum     = (scale_ratios * valid_mask).sum(axis=1)   # (M,)
    scale_mean    = scale_sum / valid_counts                   # (M,)

    diff_sq = (scale_ratios - scale_mean[:, None]) ** 2       # (M, 6)
    scale_var = (diff_sq * valid_mask).sum(axis=1) / valid_counts
    scale_std = np.sqrt(scale_var)                             # (M,)

    # Consistency score — (M,)
    safe_mean = np.where(scale_mean > 1e-6, scale_mean, 1.0)
    scale_consistency = 1.0 - (scale_std / safe_mean)
    scale_consistency = np.where(scale_mean > 1e-6, scale_consistency, 0.0)

    # Apply both filters
    keep = (n_valid >= 3) & (scale_consistency >= consistency_threshold)

    return [m for m, k in zip(matches, keep) if k]

# ==============================================================================
# Quality Metrics
# ==============================================================================

def compute_match_quality(
    n_matches: int,
    n_filtered: int,
    reference_size: int
) -> float:
    """
    Compute quality metric from match statistics.
    
    Combines match rate and filter rate into a single quality score.
    Quality balances finding matches vs. filtering bad ones.
    
    Parameters
    ----------
    n_matches : int
        Number of initial matches
    n_filtered : int
        Number of matches after filtering
    reference_size : int
        Size of reference set (for computing match rate)
        
    Returns
    -------
    float
        Combined quality score (higher is better)
    """
    if reference_size <= 0 or n_matches == 0:
        return 0.0
    
    match_rate = n_matches / reference_size
    filter_rate = n_filtered / n_matches if n_matches > 0 else 0.0
    
    # Combined quality: filter_rate * (1 - exp(-2 * match_rate))
    # This rewards both high filter rate and sufficient matches
    quality = filter_rate * (1 - np.exp(-2 * match_rate))
    
    return float(quality)

# ==============================================================================
# Threshold Calibration (√N Scaling)
# ==============================================================================

def find_threshold_for_quality_target(
    thresholds: np.ndarray,
    qualities: np.ndarray,
    target_quality: float = 0.95
) -> float:
    """
    Find threshold that achieves target quality (as fraction of max).
    
    Parameters
    ----------
    thresholds : array
        Tested threshold values
    qualities : array
        Quality at each threshold
    target_quality : float
        Target quality as fraction of max (e.g., 0.95 = 95% of max)
        
    Returns
    -------
    float
        Threshold achieving target quality
    """
    max_q = np.max(qualities)
    target_q = target_quality * max_q
    
    # Find first threshold achieving target
    idx = np.where(qualities >= target_q)[0]
    if len(idx) == 0:
        # Fallback: threshold at max quality
        return float(thresholds[np.argmax(qualities)])
    
    # Return smallest threshold achieving target (conservative)
    return float(thresholds[idx[0]])

def compute_C_value_from_pairs(
    N_values: List[float],
    tau_values: List[float]
) -> Tuple[float, float, float]:
    """
    Compute C value from (N, τ) pairs using τ = C × √N.
    
    Uses regression through the origin for optimal C estimation.
    
    Parameters
    ----------
    N_values : list of float
        Neuron counts (N_avg for each session pair)
    tau_values : list of float
        Optimal thresholds for each pair
        
    Returns
    -------
    C : float
        Calibration constant (optimal least-squares fit)
    C_std : float
        Standard error of C estimate
    r_squared : float
        R² of fit (quality of τ = C × √N relationship)
    """
    if len(N_values) < 2:
        return 0.0, 0.0, 0.0
    
    N_arr = np.array(N_values)
    tau_arr = np.array(tau_values)
    sqrt_N = np.sqrt(N_arr)
    
    # Optimal C via least squares regression through origin: τ = C × √N
    # Minimizes sum((τ - C√N)²) → C = sum(τ√N) / sum(N)
    C = float(np.sum(tau_arr * sqrt_N) / np.sum(N_arr))
    
    # Compute R² of fit
    tau_pred = C * sqrt_N
    ss_res = np.sum((tau_arr - tau_pred) ** 2)
    ss_tot = np.sum((tau_arr - np.mean(tau_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Standard error of C
    # SE(C) = sqrt(sum((τ - C√N)²) / (n-1)) / sqrt(sum(N))
    n = len(N_values)
    if n > 1:
        residual_variance = ss_res / (n - 1)
        C_std = float(np.sqrt(residual_variance / np.sum(N_arr)))
    else:
        C_std = 0.0
    
    return C, C_std, float(r_squared)

def predict_threshold_from_C(C: float, N_avg: float) -> float:
    """
    Predict optimal threshold from C value and neuron count.
    
    Uses the relationship: τ_optimal = C × √N
    
    Parameters
    ----------
    C : float
        Calibration constant
    N_avg : float
        Average neuron count for session pair
        
    Returns
    -------
    float
        Predicted optimal threshold
    """
    return C * np.sqrt(N_avg)

# ==============================================================================
# FAISS Index Cache
# Builds one IVF index per session and reuses it across all pairs.
# This avoids rebuilding the index ~17x for each session that appears in pairs.
# ==============================================================================

class _FaissIndexCache:
    """
    Module-level cache mapping (session_key, metric) → trained FAISS index.
    
    A session_key is any hashable identifier — typically the session file path
    or a string like "{animal}_{session}". Indices are built on first use and
    reused for all subsequent pairs involving that session as the target.
    """
    def __init__(self):
        self._cache: Dict[Tuple, Any] = {}

    def get_or_build(
        self,
        key: str,
        descriptors: np.ndarray,
        metric: str,
        n_probe: int = 64,
        verbose: bool = False,
    ):
        """
        Return cached FAISS index for (key, metric), building if not present.

        Parameters
        ----------
        key : str
            Unique identifier for this session's descriptor set.
        descriptors : (N, D) float32 array
            Target descriptors to index.
        metric : str
            'euclidean' or 'cosine'.
        n_probe : int
            Number of IVF clusters to search. Higher = better recall, slower.
            64 gives ~99% recall for 4D descriptors at these scales.
        verbose : bool
            Print build timing info.
        """
        import faiss
        import time as _time

        cache_key = (key, metric)
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = _time.time()
        N, D = descriptors.shape

        if metric == 'cosine':
            vecs = (descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            vecs = np.ascontiguousarray(descriptors, dtype=np.float32)
            faiss_metric = faiss.METRIC_L2

        # IVF cluster count: sqrt(N) is the standard rule of thumb.
        # Clamp to [64, 65536] so tiny or huge sets are handled gracefully.
        n_clusters = int(np.clip(int(np.sqrt(N)), 64, 65536))

        # Need at least 39 * n_clusters training points for IVF to converge.
        # Fall back to flat index if the dataset is too small.
        if N < 39 * n_clusters:
            n_clusters = max(1, N // 39)

        if n_clusters < 2:
            # Dataset too small for IVF — use flat index
            if faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(D)
            else:
                index = faiss.IndexFlatL2(D)
            index.add(vecs)
        else:
            quantizer = faiss.IndexFlatL2(D) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(D)
            index = faiss.IndexIVFFlat(quantizer, D, n_clusters, faiss_metric)
            index.train(vecs)
            index.add(vecs)
            index.nprobe = min(n_probe, n_clusters)

        if verbose:
            elapsed = _time.time() - t0
            index_type = 'IVFFlat' if n_clusters >= 2 else 'Flat'
            print(
                f"    [FAISS cache] Built {index_type}({n_clusters} clusters) "
                f"for key='{key}' | N={N:,} D={D} metric={metric} "
                f"nprobe={getattr(index, 'nprobe', 'N/A')} | {elapsed:.2f}s",
                flush=True,
            )

        self._cache[cache_key] = index
        return index

    def clear(self):
        """Free all cached indices (call between animals or at pipeline end)."""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Module-level singleton — import and use directly
faiss_index_cache = _FaissIndexCache()


# ==============================================================================
# Quad Matching (GPU/CPU)
# ==============================================================================

def match_quads_efficient_arrays(
    ref_desc: np.ndarray,
    ref_idx: np.ndarray,
    target_desc: np.ndarray,
    target_idx: np.ndarray,
    centroids_ref: np.ndarray,
    centroids_target: np.ndarray,
    config,  # PipelineConfig
    similarity_threshold: float = 0.1,
    distance_metric: str = 'euclidean',
    top_k: int = 5,
    verbose: bool = False,
    target_session_key: Optional[str] = None,
) -> List[Tuple]:
    """
    Match quadrilaterals using GPU (if available) or CPU IVF-FAISS,
    with geometric filtering for centroid distance and rotation.
    
    Parameters
    ----------
    ref_desc : (N_ref, D) array
        Reference quad descriptors
    ref_idx : (N_ref, 4) array
        Reference quad vertex indices
    target_desc : (N_target, D) array
        Target quad descriptors
    target_idx : (N_target, 4) array
        Target quad vertex indices
    centroids_ref : (M_ref, 2) array
        Reference neuron centroids
    centroids_target : (M_target, 2) array
        Target neuron centroids
    config : PipelineConfig
        Configuration with filter parameters
    similarity_threshold : float
        Maximum descriptor distance for match
    distance_metric : str
        'euclidean', 'manhattan', or 'chebyshev'
    top_k : int
        Maximum matches per reference quad
    verbose : bool
        Show progress logs
    target_session_key : str, optional
        Unique key for the target session (e.g. "{animal}_{session}").
        When provided, the FAISS index for this session is cached and reused
        across all pairs where this session is the target.  When None, a fresh
        index is built every call (same behaviour as before).
    
    Returns
    -------
    list of tuples
        Filtered matches: (ref_indices, target_indices, ref_desc, target_desc, distance)
    """
    n_ref = ref_desc.shape[0]
    n_target = target_desc.shape[0]

    if n_ref == 0 or n_target == 0 or top_k < 1:
        return []

    if distance_metric not in ('euclidean', 'manhattan', 'chebyshev', 'cosine'):
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # FAISS only (GPU/torch disabled due to NumPy 2.x incompatibility)
    matches = _cpu_kdtree_match(
        ref_desc, ref_idx, target_desc, target_idx,
        similarity_threshold, distance_metric, top_k, verbose,
    )
    
    # Apply geometric filters
    filtered_matches = []
    for match in matches:
        ref_indices, target_indices, ref_desc_m, target_desc_m, dist = match
        
        ref_pts = centroids_ref[list(ref_indices)]
        tgt_pts = centroids_target[list(target_indices)]
        
        if filter_quad_match(
            ref_pts, tgt_pts,
            image_width=config.image_width,
            image_height=config.image_height,
            max_centroid_distance_pct=config.max_centroid_distance_pct,
            max_rotation_deg=config.max_rotation_deg
        ):
            filtered_matches.append(match)
    
    return filtered_matches

def _gpu_match(
    ref_desc: np.ndarray,
    ref_idx: np.ndarray,
    target_desc: np.ndarray,
    target_idx: np.ndarray,
    similarity_threshold: float,
    top_k: int,
    verbose: bool,
    distance_metric: str = 'euclidean' 
) -> Optional[List[Tuple]]:
    """GPU brute-force matching. Returns None if not applicable."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    n_ref = ref_desc.shape[0]
    n_target = target_desc.shape[0]

    MAX_GPU_PAIRS = 20_000_000
    if n_ref * n_target > MAX_GPU_PAIRS:
        return None

    device = torch.device('cuda')
    X = torch.from_numpy(ref_desc).to(device=device, dtype=torch.float32)
    Y = torch.from_numpy(target_desc).to(device=device, dtype=torch.float32)

    if distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)
        similarity = X_norm @ Y_norm.t()  # Cosine similarity
        d = 1.0 - similarity  # Cosine distance
    else:
        # Euclidean distance (existing code)
        X2 = (X ** 2).sum(dim=1, keepdim=True)
        Y2 = (Y ** 2).sum(dim=1)
        d2 = X2 + Y2 - 2.0 * (X @ Y.t())
        d = torch.sqrt(torch.clamp(d2, min=0.0))

    k = min(top_k, n_target)
    vals, idxs = torch.topk(d, k=k, dim=1, largest=False)

    vals_np = vals.cpu().numpy()
    idxs_np = idxs.cpu().numpy()

    matches = []
    for ref_row in range(n_ref):
        for dist_val, t_idx in zip(vals_np[ref_row], idxs_np[ref_row]):
            if dist_val <= similarity_threshold and t_idx < n_target:
                matches.append((
                    ref_idx[ref_row],
                    target_idx[int(t_idx)],
                    ref_desc[ref_row],
                    target_desc[int(t_idx)],
                    float(dist_val),
                ))

    return matches

def _cpu_kdtree_match(
    ref_desc: np.ndarray,
    ref_idx: np.ndarray,
    target_desc: np.ndarray,
    target_idx: np.ndarray,
    similarity_threshold: float,
    distance_metric: str,
    top_k: int,
    verbose: bool,
    target_session_key: Optional[str] = None,
) -> List[Tuple]:
    """
    CPU matching via FAISS IndexFlatL2/IP (exhaustive) — optimal for 4D descriptors.
    IVFFlat is slower than Flat for low-dimensional vectors due to clustering overhead.
    """
    import time as _time

    n_ref    = ref_desc.shape[0]
    n_target = target_desc.shape[0]
    k_eff    = min(top_k, n_target)
    matches  = []

    try:
        import faiss

        IVF_THRESHOLD = 10_000   # use IVFFlat above this size
        N_tgt_local = target_desc.shape[0]
        D_local     = target_desc.shape[1]

        if distance_metric == 'cosine':
            norm_ref = np.linalg.norm(ref_desc,    axis=1, keepdims=True) + 1e-8
            norm_tgt = np.linalg.norm(target_desc, axis=1, keepdims=True) + 1e-8
            ref_f = (ref_desc    / norm_ref).astype(np.float32)
            tgt_f = (target_desc / norm_tgt).astype(np.float32)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            def _to_dist(s): return 1.0 - s
        else:
            ref_f = np.ascontiguousarray(ref_desc,    dtype=np.float32)
            tgt_f = np.ascontiguousarray(target_desc, dtype=np.float32)
            faiss_metric = faiss.METRIC_L2
            def _to_dist(s): return np.sqrt(np.maximum(s, 0.0))

        tgt_f = np.ascontiguousarray(tgt_f)

        if N_tgt_local >= IVF_THRESHOLD:
            # IVFFlat: sqrt(N) clusters, nprobe=64 → ~99% recall for 4-D descriptors
            n_clusters = int(np.clip(int(np.sqrt(N_tgt_local)), 64, 65536))
            if N_tgt_local < 39 * n_clusters:          # too few points to train
                n_clusters = max(1, N_tgt_local // 39)

            if n_clusters >= 2:
                quantizer = (faiss.IndexFlatIP(D_local)
                             if faiss_metric == faiss.METRIC_INNER_PRODUCT
                             else faiss.IndexFlatL2(D_local))
                index = faiss.IndexIVFFlat(quantizer, D_local, n_clusters, faiss_metric)
                index.train(tgt_f)
                index.add(tgt_f)
                index.nprobe = min(64, n_clusters)
            else:
                # Fallback: flat (dataset too tiny for IVF despite threshold)
                index = (faiss.IndexFlatIP(D_local)
                         if faiss_metric == faiss.METRIC_INNER_PRODUCT
                         else faiss.IndexFlatL2(D_local))
                index.add(tgt_f)
        else:
            # Small dataset — flat index has negligible overhead
            index = (faiss.IndexFlatIP(D_local)
                     if faiss_metric == faiss.METRIC_INNER_PRODUCT
                     else faiss.IndexFlatL2(D_local))
            index.add(tgt_f)

        # ── Chunked search ─────────────────────────────────────────────────────
        D = ref_f.shape[1]
        CHUNK    = max(n_ref, int(50_000_000 / max(D, 1)))
        n_chunks = max(1, -(-n_ref // CHUNK))
        t_start  = _time.time()

        for chunk_i in range(n_chunks):
            lo = chunk_i * CHUNK
            hi = min(lo + CHUNK, n_ref)

            scores, idxs = index.search(ref_f[lo:hi], k_eff)
            dists = _to_dist(scores)

            valid_mask = (idxs >= 0) & (idxs < n_target) & (dists <= similarity_threshold)

            if np.any(valid_mask):
                row_offsets, k_offsets = np.where(valid_mask)
                ref_rows_global = lo + row_offsets
                tgt_rows        = idxs[row_offsets, k_offsets]
                dist_vals       = dists[row_offsets, k_offsets]

                for i in range(len(ref_rows_global)):
                    r = ref_rows_global[i]
                    t = tgt_rows[i]
                    matches.append((
                        ref_idx[r],
                        target_idx[t],
                        float(dist_vals[i]),
                    ))

            if verbose:
                elapsed = _time.time() - t_start
                rate    = hi / elapsed / 1e6 if elapsed > 0 else 0.0
                pct     = 100.0 * hi / n_ref
                eta     = (elapsed / hi) * (n_ref - hi) if hi > 0 else 0.0
                n_valid = int(np.sum(valid_mask))
                print(
                    f"    chunk {chunk_i+1}/{n_chunks}  "
                    f"{hi:>8,}/{n_ref:,}  ({pct:5.1f}%)  "
                    f"{rate:.2f}M q/s  "
                    f"chunk_matches={n_valid:,}  total={len(matches):,}  "
                    f"ETA={eta:.0f}s",
                    flush=True,
                )

    except ImportError:
        # ── Scipy KDTree fallback ──────────────────────────────────────────────
        from scipy.spatial import cKDTree
        if distance_metric == 'cosine':
            ref_norm = ref_desc    / (np.linalg.norm(ref_desc,    axis=1, keepdims=True) + 1e-8)
            tgt_norm = target_desc / (np.linalg.norm(target_desc, axis=1, keepdims=True) + 1e-8)
            tree = cKDTree(tgt_norm, leafsize=16)
            dist_matrix, indices = tree.query(ref_norm, k=k_eff, p=2)
        else:
            p_map = {'euclidean': 2, 'manhattan': 1, 'chebyshev': np.inf}
            tree = cKDTree(target_desc, leafsize=16)
            dist_matrix, indices = tree.query(
                ref_desc, k=k_eff,
                p=p_map[distance_metric],
                distance_upper_bound=similarity_threshold,
            )

        dist_matrix = np.atleast_2d(dist_matrix)
        indices     = np.atleast_2d(indices)

        valid_mask = (indices >= 0) & (indices < n_target) & (dist_matrix <= similarity_threshold)
        if np.any(valid_mask):
            row_offsets, k_offsets = np.where(valid_mask)
            tgt_rows  = indices[row_offsets, k_offsets]
            dist_vals = dist_matrix[row_offsets, k_offsets]
            for i in range(len(row_offsets)):
                r = row_offsets[i]
                t = tgt_rows[i]
                matches.append((
                    ref_idx[r],
                    target_idx[t],
                    float(dist_vals[i]),
                ))

    return matches

# ==============================================================================
# Compute Maximum Distances Based on Config
# ==============================================================================

def compute_max_centroid_distance(
    image_width: int = 640,
    image_height: int = 640,
    max_centroid_distance_pct: float = 5.0
) -> float:
    """
    Compute maximum centroid distance in pixels.
    
    Parameters
    ----------
    image_width, image_height : int
        Image dimensions
    max_centroid_distance_pct : float
        Percentage of image diagonal
        
    Returns
    -------
    float
        Max distance in pixels
    """
    diagonal = np.sqrt(image_width**2 + image_height**2)
    return (max_centroid_distance_pct / 100.0) * diagonal

def match_quads_descriptor_only(
    ref_desc: np.ndarray,
    ref_idx: np.ndarray,
    target_desc: np.ndarray,
    target_idx: np.ndarray,
    similarity_threshold: float = 0.1,
    distance_metric: str = 'euclidean',
    top_k: int = 5,
    verbose: bool = False,
) -> List[Tuple]:
    """
    Match quadrilaterals using only descriptor distance (no geometric filters).
    
    Parameters
    ----------
    ref_desc : (N_ref, D) array
        Reference quad descriptors
    ref_idx : (N_ref, 4) array
        Reference quad vertex indices
    target_desc : (N_target, D) array
        Target quad descriptors
    target_idx : (N_target, 4) array
        Target quad vertex indices
    similarity_threshold : float
        Maximum descriptor distance for match
    distance_metric : str
        'euclidean', 'manhattan', or 'chebyshev'
    top_k : int
        Maximum matches per reference quad
    verbose : bool
        Show progress logs
    target_session_key : str, optional
        Unique key for the target session for FAISS index caching.
        When provided, the index is built once and reused across pairs.
    
    Returns
    -------
    list of tuples
        Matches: (ref_indices, target_indices, ref_desc, target_desc, distance)
    """
    n_ref = ref_desc.shape[0]
    n_target = target_desc.shape[0]

    if n_ref == 0 or n_target == 0 or top_k < 1:
        return []

    if distance_metric not in ('euclidean', 'manhattan', 'chebyshev', 'cosine'):
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # FAISS only (GPU/torch disabled due to NumPy 2.x incompatibility)
    matches = _cpu_kdtree_match(
        ref_desc, ref_idx, target_desc, target_idx,
        similarity_threshold, distance_metric, top_k, verbose,
    )
    
    # No geometric filtering - return all matches
    return matches

# ==============================================================================
# Transform Consistency Filtering (RANSAC)
# ==============================================================================

def filter_matches_by_transform_consistency(
    ref_positions: np.ndarray,
    tgt_positions: np.ndarray,
    max_residual_px: float = 15.0,
    min_inliers: int = 4,
    transform_type: str = 'affine',
    ransac_max_trials: int = 1000,
) -> Tuple[np.ndarray, Optional[Any], np.ndarray]:
    """
    Estimate global transform from matched points, reject outliers via RANSAC.
    
    Matches are only valid if they're consistent with a coherent global motion
    (translation, rotation, scaling, shear). A 400px displacement is acceptable
    IF all other matches imply the same transformation.
    
    Parameters
    ----------
    ref_positions : (N, 2) array
        Matched reference neuron positions
    tgt_positions : (N, 2) array
        Matched target neuron positions
    max_residual_px : float
        Maximum allowed residual after transform (pixels)
    min_inliers : int
        Minimum inliers required for valid transform
    transform_type : str
        'affine' (6 DOF), 'similarity' (4 DOF), or 'euclidean' (3 DOF)
    ransac_max_trials : int
        Maximum RANSAC iterations
        
    Returns
    -------
    inlier_mask : (N,) bool array
        True for matches consistent with global transform
    transform : skimage transform object or None
        Estimated transform (None if failed)
    residuals : (N,) array
        Per-match residual distances in pixels
    """
    from skimage.transform import AffineTransform, SimilarityTransform, EuclideanTransform
    from skimage.measure import ransac
    
    N = len(ref_positions)
    
    # Edge cases
    if N < min_inliers:
        return np.ones(N, dtype=bool), None, np.zeros(N)
    
    # Select transform type
    transform_classes = {
        'affine': AffineTransform,
        'similarity': SimilarityTransform,
        'euclidean': EuclideanTransform,
    }
    
    if transform_type not in transform_classes:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    
    TransformClass = transform_classes[transform_type]
    
    # Minimum samples needed for each transform type
    min_samples_map = {'affine': 3, 'similarity': 2, 'euclidean': 2}
    min_samples = min_samples_map[transform_type]
    
    if N < min_samples:
        return np.ones(N, dtype=bool), None, np.zeros(N)
    
    try:
        # RANSAC: find transform that explains most matches
        transform, inliers = ransac(
            (ref_positions, tgt_positions),
            TransformClass,
            min_samples=min_samples,
            residual_threshold=max_residual_px,
            max_trials=ransac_max_trials,
        )
        
        # Compute residuals for ALL points (not just inliers)
        transformed = transform(ref_positions)
        residuals = np.linalg.norm(transformed - tgt_positions, axis=1)
        
        # Create mask from residuals (more reliable than RANSAC's inliers)
        inlier_mask = residuals <= max_residual_px
        
        # Require minimum inliers
        if np.sum(inlier_mask) < min_inliers:
            # Fall back to keeping all if there is no good transform
            return np.ones(N, dtype=bool), None, residuals
        
        return inlier_mask, transform, residuals
        
    except Exception as e:
        # If RANSAC fails, return all as inliers with zero residuals
        return np.ones(N, dtype=bool), None, np.zeros(N)

def compute_transform_quality_metrics(
    transform,
    inlier_mask: np.ndarray,
    residuals: np.ndarray,
) -> Dict[str, float]:
    """
    Compute quality metrics for estimated transform.
    
    Returns
    -------
    dict with keys:
        - n_inliers: number of inlier matches
        - inlier_fraction: fraction of matches that are inliers
        - mean_residual: mean residual for inliers (pixels)
        - max_residual: max residual for inliers (pixels)
        - translation_x, translation_y: estimated translation
        - rotation_deg: estimated rotation (degrees)
        - scale: estimated scale factor
    """
    metrics = {
        'n_inliers': int(np.sum(inlier_mask)),
        'inlier_fraction': float(np.mean(inlier_mask)),
        'mean_residual': float(np.mean(residuals[inlier_mask])) if np.any(inlier_mask) else 0.0,
        'max_residual': float(np.max(residuals[inlier_mask])) if np.any(inlier_mask) else 0.0,
    }
    
    if transform is not None:
        try:
            metrics['translation_x'] = float(transform.translation[0])
            metrics['translation_y'] = float(transform.translation[1])
            metrics['rotation_deg'] = float(np.degrees(transform.rotation))
            metrics['scale'] = float(transform.scale) if hasattr(transform, 'scale') else 1.0
        except:
            pass
    
    return metrics