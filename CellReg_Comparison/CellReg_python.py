#!/usr/bin/env python3
"""
cellreg_python.py

CellReg pipeline (Sheintuch et al. 2017) ported to Python.
Supports centroid-distance model AND spatial-correlation model.

When footprint_dir is provided, computes spatial correlations between
neighbor pairs, fits both models independently, then selects the best
model based on CellReg's cost metric (false_positive + false_negative + MSE).

Spatial correlation model faithfully ports compute_spatial_correlations_model.m:
  - Same-cell:      lognormal on (1 - correlation)
  - Different-cell:  Beta on (1 - correlation)
  - Fitted via EM (100 iterations)
  - Sigmoid smoothing on same-cell component
Footprints normalized to sum=1 (matching normalize_spatial_footprints.m).
Spatial correlation via corr2 equivalent (full-image Pearson).

Usage: edit FILE_PATHS, MICRONS_PER_PIXEL, and FOOTPRINT_DIR at the bottom.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from scipy.special import psi as digamma
from scipy.special import polygamma
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Load sessions
# ═══════════════════════════════════════════════════════════════════════════

def load_sessions(file_paths):
    """
    Load .npy session files.

    Parameters
    ----------
    file_paths : list of str or Path

    Returns
    -------
    centroid_locations : list of np.ndarray, each [N_cells × 2] (x, y)
    roi_ids_list       : list of np.ndarray, each [N_cells] (original IDs)
    """
    centroid_locations = []
    roi_ids_list = []

    for fp in file_paths:
        data = np.load(fp, allow_pickle=True).item()
        x = np.array(data['centroids_x'], dtype=float)
        y = np.array(data['centroids_y'], dtype=float)
        roi_ids = np.array(data['roi_ids'])
        centroids = np.stack([x, y], axis=1)  # [N × 2]
        centroid_locations.append(centroids)
        roi_ids_list.append(roi_ids)
        print(f"  Loaded {fp}  →  {len(x)} cells")

    return centroid_locations, roi_ids_list


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1b — Load spatial footprints (optional)
# ═══════════════════════════════════════════════════════════════════════════

def load_footprints(file_paths, footprint_dir):
    """
    Load spatial footprints from the spatial_mapping/ subfolder.
    Normalizes each footprint to sum=1 (matching normalize_spatial_footprints.m).

    Supports both compact (_compact.npy) and dense (_footprints.npy) formats.
    If compact, reconstructs dense on-the-fly.

    Parameters
    ----------
    file_paths     : list of str/Path — the session .npy file paths
    footprint_dir  : str/Path — path to spatial_mapping/ directory

    Returns
    -------
    footprints_list : list of np.ndarray, each [N_cells × H × W], sum-normalized
    """
    footprint_dir = Path(footprint_dir)
    footprints_list = []

    for fp in file_paths:
        session_stem = Path(fp).stem

        # Try compact first (much smaller files)
        compact_path = footprint_dir / f"{session_stem}_footprints_compact.npy"
        dense_path   = footprint_dir / f"{session_stem}_footprints.npy"

        if compact_path.exists():
            data = np.load(compact_path, allow_pickle=True).item()
            patches = data['patches']
            bboxes  = data['bboxes']
            fov     = data.get('fov_size', 512)
            n = len(patches)
            dense = np.zeros((n, fov, fov), dtype=np.float32)
            for i in range(n):
                y_lo, y_hi, x_lo, x_hi = bboxes[i]
                dense[i, y_lo:y_hi, x_lo:x_hi] = patches[i]
            # Normalize each footprint to sum=1 (CellReg normalize_spatial_footprints.m)
            sums = dense.sum(axis=(1, 2), keepdims=True)
            sums[sums == 0] = 1.0
            dense = dense / sums
            footprints_list.append(dense)
            print(f"  Loaded footprints (compact) for {session_stem}  →  {n} cells")

        elif dense_path.exists():
            data = np.load(dense_path, allow_pickle=True).item()
            dense = data['footprints']
            sums = dense.sum(axis=(1, 2), keepdims=True)
            sums[sums == 0] = 1.0
            dense = dense / sums
            footprints_list.append(dense)
            print(f"  Loaded footprints (dense) for {session_stem}  →  "
                  f"{dense.shape[0]} cells")
        else:
            raise FileNotFoundError(
                f"No footprint file found for {session_stem} in {footprint_dir}.\n"
                f"  Tried: {compact_path}\n"
                f"         {dense_path}")

    return footprints_list


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Alignment (skipped — no images, centroids already in same space)
# ═══════════════════════════════════════════════════════════════════════════

def identity_alignment(centroid_locations):
    """Pass-through: no image alignment needed for centroid-only data."""
    return centroid_locations


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3a — Compute neighbor distances across all session pairs
# ═══════════════════════════════════════════════════════════════════════════

def compute_neighbor_distances(centroid_locations, maximal_distance):
    """
    For every cell in every session, find all neighbors in every other
    session within maximal_distance pixels.

    Returns
    -------
    all_to_all_indexes    : list[list[list[np.ndarray]]]
                            [session_n][cell_k][session_m] → neighbor indices
    all_to_all_distances  : same shape → neighbor distances
    neighbors_distances   : flat np.ndarray of ALL neighbor distances (for model fitting)
    """
    n_sessions = len(centroid_locations)
    all_to_all_indexes   = [[None] * n_sessions for _ in range(n_sessions)]
    all_to_all_distances = [[None] * n_sessions for _ in range(n_sessions)]
    neighbors_distances_list = []

    for n in range(n_sessions):
        cents_n = centroid_locations[n]
        n_cells = len(cents_n)
        all_to_all_indexes[n]   = [None] * n_sessions
        all_to_all_distances[n] = [None] * n_sessions
        for m in range(n_sessions):
            all_to_all_indexes[n][m]   = [None] * n_cells
            all_to_all_distances[n][m] = [None] * n_cells

        for m in range(n_sessions):
            if m == n:
                for k in range(n_cells):
                    all_to_all_indexes[n][m][k]   = np.array([], dtype=int)
                    all_to_all_distances[n][m][k] = np.array([])
                continue

            cents_m = centroid_locations[m]
            for k in range(n_cells):
                diff = cents_m - cents_n[k]
                dists = np.sqrt((diff ** 2).sum(1))
                within = np.where(dists < maximal_distance)[0]
                all_to_all_indexes[n][m][k]   = within
                all_to_all_distances[n][m][k] = dists[within]
                neighbors_distances_list.extend(dists[within].tolist())

    neighbors_distances = np.array(neighbors_distances_list)
    print(f"  Neighbor pairs found: {len(neighbors_distances)}")
    return all_to_all_indexes, all_to_all_distances, neighbors_distances


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3a+ — Compute spatial correlations for all neighbor pairs
# ═══════════════════════════════════════════════════════════════════════════

def _spatial_correlation(fp1, fp2):
    """
    2D Pearson correlation over the full footprint images.
    Equivalent to MATLAB's corr2() used in compute_data_distribution.m.

    Returns float in [-1, 1], or 0.0 if either footprint is all-zero.
    """
    if fp1.sum() == 0 or fp2.sum() == 0:
        return 0.0
    v1 = fp1.ravel().astype(np.float64)
    v2 = fp2.ravel().astype(np.float64)
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    denom = np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((v1 * v2).sum() / denom)


def compute_neighbor_spatial_correlations(all_to_all_indexes, footprints_list):
    """
    For every neighbor pair already identified by centroid distance,
    compute the spatial correlation between their footprints.

    Returns
    -------
    all_to_all_spatial_corr : same shape as all_to_all_indexes
                              [sess_n][sess_m][cell_k] → np.ndarray of correlations
    neighbors_correlations  : flat np.ndarray of ALL correlations (for model fitting)
    """
    n_sessions = len(footprints_list)
    all_to_all_spatial_corr = [[None] * n_sessions for _ in range(n_sessions)]
    neighbors_correlations_list = []

    for n in range(n_sessions):
        all_to_all_spatial_corr[n] = [None] * n_sessions
        n_cells = footprints_list[n].shape[0]

        for m in range(n_sessions):
            all_to_all_spatial_corr[n][m] = [None] * n_cells
            if m == n:
                for k in range(n_cells):
                    all_to_all_spatial_corr[n][m][k] = np.array([])
                continue

            for k in range(n_cells):
                neighbor_idxs = all_to_all_indexes[n][m][k]
                if len(neighbor_idxs) == 0:
                    all_to_all_spatial_corr[n][m][k] = np.array([])
                    continue

                fp_k = footprints_list[n][k]  # [H × W]
                corrs = np.array([
                    _spatial_correlation(fp_k, footprints_list[m][j])
                    for j in neighbor_idxs
                ])
                all_to_all_spatial_corr[n][m][k] = corrs
                neighbors_correlations_list.extend(corrs.tolist())

    neighbors_correlations = np.array(neighbors_correlations_list)
    print(f"  Spatial correlations computed: {len(neighbors_correlations)}")
    print(f"  Mean={neighbors_correlations.mean():.3f}  "
          f"Median={np.median(neighbors_correlations):.3f}  "
          f"Max={neighbors_correlations.max():.3f}")
    return all_to_all_spatial_corr, neighbors_correlations


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3b — Fit centroid distance mixture model
# ═══════════════════════════════════════════════════════════════════════════

def _mixture_model(x, p, mu_ln, sigma_ln, a, c, b):
    """
    Weighted sum of:
      same_cells   : lognormal(x; mu_ln, sigma_ln)
      diff_cells   : b * x / (1 + exp(-a*(x - c)))   [sigmoid × linear ramp]
    """
    same = (1.0 / (x * sigma_ln * np.sqrt(2 * np.pi))) * \
           np.exp(-((np.log(x) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
    diff = b * x / (1.0 + np.exp(-a * (x - c)))
    return p * same + (1.0 - p) * diff


def fit_centroid_distance_model(neighbors_distances, microns_per_pixel,
                                 n_bins=50, maximal_distance=None):
    """
    Fit the lognormal + sigmoid-linear mixture model to the observed
    distribution of centroid distances.

    Returns
    -------
    bin_centers, hist_norm, p_same, same_model, diff_model,
    params, intersection_px, mse
    """
    if maximal_distance is None:
        maximal_distance = neighbors_distances.max()

    bin_edges   = np.linspace(0, maximal_distance, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width   = bin_centers[1] - bin_centers[0]

    counts, _ = np.histogram(neighbors_distances, bins=bin_edges)
    hist_norm = counts / counts.sum() / bin_width

    close_mask = microns_per_pixel * neighbors_distances < 9.0
    close_dists = neighbors_distances[close_mask]
    if len(close_dists) < 5:
        close_dists = neighbors_distances
    log_data = np.log(close_dists[close_dists > 0])
    mu0    = log_data.mean()
    sig0   = max(log_data.std(), 0.01)
    p0_mix = len(close_dists) / len(neighbors_distances)

    c0 = 6.0 / microns_per_pixel
    a0 = 1.0 * microns_per_pixel
    slope = (hist_norm[-1] - hist_norm[n_bins // 2]) / \
            (bin_centers[-1] - bin_centers[n_bins // 2] + 1e-9)
    b0 = max(slope / max(1.0 - p0_mix, 0.01), 0.001)

    p0     = [p0_mix, mu0, sig0, a0, c0, b0]
    bounds = ([0, -np.inf, 1e-6, 0, 0, 0],
              [1,  np.inf, np.inf, np.inf, np.inf, np.inf])

    x_fit = bin_centers.copy()
    x_fit[x_fit <= 0] = 1e-6

    try:
        popt, _ = curve_fit(
            _mixture_model, x_fit, hist_norm,
            p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError:
        print("  WARNING: curve_fit did not converge; using initial parameters")
        popt = p0

    p_fit, mu_fit, sig_fit, a_fit, c_fit, b_fit = popt

    x_eval = x_fit.copy()
    same_model_vals = (1.0 / (x_eval * sig_fit * np.sqrt(2 * np.pi))) * \
                      np.exp(-((np.log(x_eval) - mu_fit) ** 2) / (2 * sig_fit ** 2))
    diff_model_vals = b_fit * x_eval / (1.0 + np.exp(-a_fit * (x_eval - c_fit)))

    def _norm(v):
        s = v.sum() * bin_width
        return v / s if s > 0 else v

    same_norm = _norm(same_model_vals)
    diff_norm = _norm(diff_model_vals)

    denom = p_fit * same_norm + (1.0 - p_fit) * diff_norm
    p_same = np.where(denom > 0, p_fit * same_norm / denom, 0.0)
    p_same[0] = p_same[1]

    lo_px = 1.0  / microns_per_pixel
    hi_px = 10.0 / microns_per_pixel
    search = (bin_centers > lo_px) & (bin_centers < hi_px)
    if search.any():
        idx = np.argmin(np.abs(p_same[search] - 0.5))
        intersection_px = bin_centers[search][idx]
    else:
        intersection_px = c_fit

    weighted = p_fit * same_norm + (1.0 - p_fit) * diff_norm
    mse = 0.5 * np.mean(np.abs(hist_norm - weighted))

    params = dict(p=p_fit, mu_ln=mu_fit, sigma_ln=sig_fit,
                  a=a_fit, c=c_fit, b=b_fit)

    print(f"  Model fit  MSE={mse:.4f}  "
          f"P_same=0.5 at {intersection_px * microns_per_pixel:.2f} µm  "
          f"(mix fraction p={p_fit:.3f})")

    return bin_centers, hist_norm, p_same, same_norm, diff_norm, params, intersection_px, mse


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3b+ — Fit spatial correlation model (EM, ported from MATLAB)
# ═══════════════════════════════════════════════════════════════════════════

def _lognpdf(x, mu, sigma):
    """Lognormal PDF matching MATLAB lognpdf."""
    x = np.clip(x, 1e-10, None)
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


def _betapdf(x, p, q):
    """Beta PDF matching MATLAB betapdf."""
    from scipy.special import betaln
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    log_pdf = (p - 1) * np.log(x) + (q - 1) * np.log(1 - x) - betaln(p, q)
    return np.exp(log_pdf)


def _estimate_beta_mixture_params(assignments, data, maximal_distance=1.0):
    """
    Port of estimate_beta_mixture_params.m — Newton-Raphson for Beta params
    with weighted (soft) assignments.

    Parameters
    ----------
    assignments : np.ndarray — weights for the diff-cell component (1-gamma)
    data        : np.ndarray — values in (0, maximal_distance)
    maximal_distance : float

    Returns
    -------
    p, q : float — Beta shape parameters
    """
    data = data.copy()
    data[data >= maximal_distance] = 0.9999 * maximal_distance

    g1 = np.sum(assignments * np.log(data / maximal_distance)) / (np.sum(assignments) + 1e-12)
    g2 = np.sum(assignments * np.log((maximal_distance - data) / maximal_distance)) / (np.sum(assignments) + 1e-12)

    # Initial guess via moment-matching
    sample_mean = np.sum(assignments * data) / (np.sum(assignments) + 1e-12)
    sample_var = np.sum(assignments * (data - sample_mean) ** 2) / (np.sum(assignments) + 1e-12)

    xbar = sample_mean / maximal_distance
    xbar = np.clip(xbar, 0.01, 0.99)
    ssq = max(sample_var / maximal_distance ** 2, 1e-6)

    p = xbar * (xbar * (1 - xbar) / ssq - 1)
    q = (1 - xbar) * (xbar * (1 - xbar) / ssq - 1)
    p = max(p, 0.1)
    q = max(q, 0.1)

    # Newton-Raphson (100 iterations, matching MATLAB)
    for _ in range(100):
        grad = np.array([
            digamma(p) - digamma(p + q) - g1,
            digamma(q) - digamma(p + q) - g2,
        ])
        hess = np.array([
            [polygamma(1, p) - polygamma(1, p + q), -polygamma(1, p + q)],
            [-polygamma(1, p + q), polygamma(1, q) - polygamma(1, p + q)],
        ])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        p -= step[0]
        q -= step[1]
        p = max(p, 0.01)
        q = max(q, 0.01)

    return p, q


def fit_spatial_correlation_model(neighbors_correlations, n_bins=50):
    """
    Port of compute_spatial_correlations_model.m.

    Fits a mixture model via EM (100 iterations):
      same-cell:      lognormal on (1 - correlation)
      different-cell:  Beta on (1 - correlation)

    Applies sigmoid smoothing to the same-cell component and zeros
    the first 10% of bins (matching MATLAB).

    Returns
    -------
    bin_centers_sc     : np.ndarray [n_bins]
    hist_norm_sc       : np.ndarray [n_bins]
    p_same_sc          : np.ndarray [n_bins]
    same_model_sc      : np.ndarray [n_bins]
    diff_model_sc      : np.ndarray [n_bins]
    params_sc          : dict
    mse_sc             : float
    intersection_sc    : float
    """
    # Clean correlations (matching MATLAB)
    corrs = neighbors_correlations.copy()
    corrs = corrs[(corrs >= 0) & (corrs <= 1)]
    corrs[corrs == 0] = 1e-10
    corrs[corrs == 1] = 1 - 1e-10

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_centers[1] - bin_centers[0]

    # data = 1 - correlation (matching MATLAB line 31)
    data = 1.0 - corrs

    # ── Initial parameters for EM ──
    high_corr = corrs[corrs >= 0.7]
    low_corr = corrs[corrs < 0.7]

    if len(high_corr) < 3:
        high_corr = corrs
    if len(low_corr) < 3:
        low_corr = corrs

    # Lognormal init on (1-corr) for high-correlation (same-cell) pairs
    same_data = 1.0 - high_corr
    same_data = same_data[same_data > 0]
    mu = np.log(same_data).mean()
    sigma = max(np.log(same_data).std(), 0.01)

    # Beta init on (1-corr) for low-correlation (diff-cell) pairs
    diff_data = 1.0 - low_corr
    diff_data = np.clip(diff_data, 1e-6, 1 - 1e-6)
    xbar = diff_data.mean()
    xbar = np.clip(xbar, 0.01, 0.99)
    ssq = max(diff_data.var(), 1e-6)
    p_beta = max(xbar * (xbar * (1 - xbar) / ssq - 1), 0.5)
    q_beta = max((1 - xbar) * (xbar * (1 - xbar) / ssq - 1), 0.5)

    PIsame = 0.5

    # ── EM Algorithm (100 iterations, matching MATLAB) ──
    em_failed = False
    for _ in range(100):
        # E-step: compute responsibilities
        same_vals = _lognpdf(data, mu, sigma)
        diff_vals = _betapdf(data, p_beta, q_beta)

        denom = PIsame * same_vals + (1 - PIsame) * diff_vals
        denom = np.maximum(denom, 1e-300)
        assignments = PIsame * same_vals / denom  # gamma_k (prob of same-cell)

        # M-step: update parameters
        PIsame = np.sum(assignments) / len(assignments)
        PIsame = np.clip(PIsame, 0.001, 0.999)

        weighted_log = np.sum(assignments * np.log(np.maximum(data, 1e-10)))
        sum_assign = np.sum(assignments) + 1e-12
        mu = weighted_log / sum_assign
        sigma = np.sqrt(np.sum(assignments * (np.log(np.maximum(data, 1e-10)) - mu) ** 2) / sum_assign)
        sigma = max(sigma, 0.001)

        # Beta params via Newton-Raphson (matching MATLAB call)
        p_beta, q_beta = _estimate_beta_mixture_params(1.0 - assignments, data, 1.0)

        if np.isnan(p_beta) or np.isnan(q_beta):
            em_failed = True
            break

    # ── Evaluate models at bin centers ──
    # Models are evaluated at (1 - bin_centers), matching MATLAB lines 55-56
    x_eval = 1.0 - bin_centers
    x_eval = np.clip(x_eval, 1e-10, 1 - 1e-10)

    same_model = _lognpdf(x_eval, mu, sigma)
    diff_model = _betapdf(x_eval, p_beta, q_beta)

    # ── Normalize models (matching MATLAB lines 89-90) ──
    norm_factor = n_bins / (bin_width + (bin_centers[-1] - bin_centers[0]))
    same_model = same_model / (same_model.sum() + 1e-12) * norm_factor
    diff_model = diff_model / (diff_model.sum() + 1e-12) * norm_factor

    # ── Sigmoid smoothing on same-cell model (matching MATLAB lines 94-97) ──
    sigmoid = 1.0 / (1.0 + np.exp(-20 * (bin_centers - (bin_centers.min() + 0.5))))
    same_model = same_model * sigmoid
    same_model[:round(n_bins / 10)] = 0

    # ── Weighted sum ──
    weighted_sum = PIsame * same_model + (1 - PIsame) * diff_model

    # ── Histogram of data ──
    counts, _ = np.histogram(corrs, bins=bin_edges)
    hist_norm = counts / (counts.sum() + 1e-12) * norm_factor

    # ── MSE (matching MATLAB line 109) ──
    mse = np.sum(np.abs(hist_norm - weighted_sum) * (bin_width + (bin_centers[-1] - bin_centers[0])) / n_bins) / 2.0

    # ── P_same (matching MATLAB lines 113-117) ──
    denom = PIsame * same_model + (1 - PIsame) * diff_model
    p_same = np.where(denom > 1e-12, PIsame * same_model / denom, 0.0)

    # ── Smooth low-P_same bins (matching MATLAB lines 119-122) ──
    minimal_thresh = 0.001
    low_bins = np.where(same_model < minimal_thresh * same_model.max())[0]
    if len(low_bins) > 0:
        smooth_sig = 1.0 / (1.0 + np.exp(-0.05 * len(low_bins) *
                    (np.arange(len(low_bins)) - 0.8 * len(low_bins))))
        p_same[low_bins] *= smooth_sig

    # ── Intersection (P_same ≈ 0.5, matching MATLAB lines 124-134) ──
    active = np.where(same_model > minimal_thresh * same_model.max())[0]
    active = active[active < n_bins - 1]  # exclude last bin
    if len(active) > 0:
        weighted_same = PIsame * same_model[active]
        weighted_diff = (1 - PIsame) * diff_model[active]
        idx = np.argmin(np.abs(weighted_same - weighted_diff))
        intersection = round(bin_centers[active[idx]] * 100) / 100
    else:
        intersection = 0.5

    params = dict(PIsame=PIsame, mu=mu, sigma=sigma,
                  p_beta=p_beta, q_beta=q_beta)

    print(f"  Spatial model fit (EM)  MSE={mse:.4f}  "
          f"P_same=0.5 at corr={intersection:.3f}  "
          f"(mix fraction p={PIsame:.3f})")

    return bin_centers, hist_norm, p_same, same_model, diff_model, params, mse, intersection


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3c — Assign P_same to every neighbor pair
# ═══════════════════════════════════════════════════════════════════════════

def assign_p_same(all_to_all_distances, p_same_lookup, bin_centers):
    """
    For each neighbor pair look up its centroid-distance P_same.
    """
    n_sessions = len(all_to_all_distances)
    all_to_all_p_same = [[None] * n_sessions for _ in range(n_sessions)]

    for n in range(n_sessions):
        all_to_all_p_same[n] = [None] * n_sessions
        n_cells = len(all_to_all_distances[n][0])
        for m in range(n_sessions):
            all_to_all_p_same[n][m] = [None] * n_cells
            if m == n:
                for k in range(n_cells):
                    all_to_all_p_same[n][m][k] = np.array([])
                continue
            for k in range(n_cells):
                dists = all_to_all_distances[n][m][k]
                if len(dists) == 0:
                    all_to_all_p_same[n][m][k] = np.array([])
                else:
                    idxs = np.argmin(np.abs(bin_centers[:, None] - dists[None, :]), axis=0)
                    all_to_all_p_same[n][m][k] = p_same_lookup[idxs]

    return all_to_all_p_same


def assign_p_same_spatial(all_to_all_spatial_corr, p_same_sc_lookup, bin_centers_sc):
    """
    For each neighbor pair look up its spatial-correlation P_same.
    """
    n_sessions = len(all_to_all_spatial_corr)
    all_to_all_p_same_sc = [[None] * n_sessions for _ in range(n_sessions)]

    for n in range(n_sessions):
        all_to_all_p_same_sc[n] = [None] * n_sessions
        n_cells = len(all_to_all_spatial_corr[n][0])
        for m in range(n_sessions):
            all_to_all_p_same_sc[n][m] = [None] * n_cells
            if m == n:
                for k in range(n_cells):
                    all_to_all_p_same_sc[n][m][k] = np.array([])
                continue
            for k in range(n_cells):
                corrs = all_to_all_spatial_corr[n][m][k]
                if len(corrs) == 0:
                    all_to_all_p_same_sc[n][m][k] = np.array([])
                else:
                    corrs_clipped = np.clip(corrs, bin_centers_sc[0], bin_centers_sc[-1])
                    idxs = np.argmin(
                        np.abs(bin_centers_sc[:, None] - corrs_clipped[None, :]), axis=0)
                    all_to_all_p_same_sc[n][m][k] = p_same_sc_lookup[idxs]

    return all_to_all_p_same_sc


def choose_best_model(mse_cd, same_model_cd, diff_model_cd, p_same_cd,
                      mse_sc, same_model_sc, diff_model_sc, p_same_sc):
    """
    Port of choose_best_model.m — selects the best model based on
    false positive rate + false negative rate + MSE.

    CellReg does NOT combine models. It picks one.

    Returns
    -------
    best_model : str — 'centroid' or 'spatial'
    """
    # Spatial correlation model costs
    ind_05_sc = np.argmin(np.abs(0.5 - p_same_sc))
    fp_sc = diff_model_sc[ind_05_sc:].sum() / (diff_model_sc.sum() + 1e-12)
    fn_sc = same_model_sc[:ind_05_sc].sum() / (same_model_sc.sum() + 1e-12)

    # Centroid distance model costs
    ind_05_cd = np.argmin(np.abs(0.5 - p_same_cd))
    fp_cd = diff_model_cd[:ind_05_cd].sum() / (diff_model_cd.sum() + 1e-12)
    fn_cd = same_model_cd[ind_05_cd:].sum() / (same_model_cd.sum() + 1e-12)

    cost_cd = fp_cd + fn_cd + mse_cd
    cost_sc = fp_sc + fn_sc + mse_sc

    if mse_cd > 0.1:
        print("  WARNING: large discrepancy between centroid distances model and data")
    if mse_sc > 0.1:
        print("  WARNING: large discrepancy between spatial correlations model and data")

    if cost_cd <= cost_sc:
        print(f"  Model selection: centroid distance (cost={cost_cd:.4f} vs {cost_sc:.4f})")
        return 'centroid'
    else:
        print(f"  Model selection: spatial correlation (cost={cost_sc:.4f} vs {cost_cd:.4f})")
        return 'spatial'


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Initial greedy registration (centroid distance)
# ═══════════════════════════════════════════════════════════════════════════

def initial_registration(centroid_locations, maximal_distance, threshold_px):
    """
    Greedy nearest-neighbour registration seeded from session 0.

    NOTE: cell_to_index_map uses 1-based indexing internally (0 = empty).
    This matches MATLAB CellReg convention and avoids the ambiguity where
    cell index 0 would be indistinguishable from "no cell present".
    """
    n_sessions = len(centroid_locations)
    n_init     = len(centroid_locations[0])

    cell_to_index_map = np.zeros((n_init, n_sessions), dtype=int)
    cell_to_index_map[:, 0] = np.arange(n_init) + 1  # 1-based

    reg_centroids = centroid_locations[0].copy().tolist()
    dist_map      = np.zeros((n_init, n_sessions))
    extra_count   = 0

    for sess in range(1, n_sessions):
        new_cents = centroid_locations[sess]
        n_new     = len(new_cents)

        for k in range(n_new):
            cent_k = new_cents[k]
            reg_arr = np.array(reg_centroids)
            dists   = np.sqrt(((reg_arr - cent_k) ** 2).sum(1))
            candidates = np.where(dists < maximal_distance)[0]

            if len(candidates) == 0:
                reg_centroids.append(cent_k.tolist())
                new_row = np.zeros((1, n_sessions), dtype=int)
                new_row[0, sess] = k + 1  # 1-based
                cell_to_index_map = np.vstack([cell_to_index_map, new_row])
                dist_map = np.vstack([dist_map, np.zeros((1, n_sessions))])
                extra_count += 1
                continue

            best_local = np.argmin(dists[candidates])
            best_idx   = candidates[best_local]
            best_dist  = dists[best_idx]

            if best_dist > threshold_px:
                reg_centroids.append(cent_k.tolist())
                new_row = np.zeros((1, n_sessions), dtype=int)
                new_row[0, sess] = k + 1  # 1-based
                cell_to_index_map = np.vstack([cell_to_index_map, new_row])
                dist_map = np.vstack([dist_map, np.zeros((1, n_sessions))])
                extra_count += 1
            else:
                if cell_to_index_map[best_idx, sess] == 0:
                    cell_to_index_map[best_idx, sess] = k + 1  # 1-based
                    dist_map[best_idx, sess] = best_dist
                else:
                    if best_dist < dist_map[best_idx, sess]:
                        displaced = cell_to_index_map[best_idx, sess]
                        displaced_cent = new_cents[displaced - 1]  # back to 0-based for array access
                        cell_to_index_map[best_idx, sess] = k + 1  # 1-based
                        dist_map[best_idx, sess] = best_dist
                        reg_centroids.append(displaced_cent.tolist())
                        new_row = np.zeros((1, n_sessions), dtype=int)
                        new_row[0, sess] = displaced  # already 1-based
                        cell_to_index_map = np.vstack([cell_to_index_map, new_row])
                        dist_map = np.vstack([dist_map, np.zeros((1, n_sessions))])
                        extra_count += 1
                    else:
                        reg_centroids.append(cent_k.tolist())
                        new_row = np.zeros((1, n_sessions), dtype=int)
                        new_row[0, sess] = k + 1  # 1-based
                        cell_to_index_map = np.vstack([cell_to_index_map, new_row])
                        dist_map = np.vstack([dist_map, np.zeros((1, n_sessions))])
                        extra_count += 1

    n_total = cell_to_index_map.shape[0]
    print(f"  Initial registration: {n_total} candidate cells found")
    return cell_to_index_map


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Iterative clustering
# ═══════════════════════════════════════════════════════════════════════════

def cluster_cells(cell_to_index_map, all_to_all_p_same, all_to_all_indexes,
                  centroid_locations, maximal_distance, p_same_threshold,
                  max_iterations=10, num_changes_thresh=10):
    """
    Iteratively reassign cells to clusters to maximise P_same.
    Direct port of cluster_cells.m (Maximal similarity criterion).

    NOTE: all_to_all_p_same may be centroid-only OR combined (centroid+spatial).
    The clustering algorithm is agnostic — it just uses whatever P_same it gets.
    """
    n_sessions = len(centroid_locations)
    cluster_dist_thresh = 1.7 * maximal_distance
    cmap = cell_to_index_map.copy()

    def _cluster_centroids(cmap):
        n_clusters = cmap.shape[0]
        cents = np.zeros((2, n_clusters))
        for n in range(n_clusters):
            active = np.where(cmap[n] > 0)[0]
            if len(active) == 0:
                continue
            pts = np.array([centroid_locations[s][cmap[n, s] - 1] for s in active])  # 1-based → 0-based
            cents[:, n] = pts.mean(0)
        return cents

    changes_count = -1
    iteration = 0

    while (changes_count > num_changes_thresh or changes_count == -1) \
            and iteration < max_iterations:
        iteration += 1
        changes_count = 0
        separation_count = switch_count = move_count = delete_count = 0

        cluster_cents = _cluster_centroids(cmap)
        num_clusters  = cmap.shape[0]

        for sess in range(n_sessions):
            cluster_rows = np.where(cmap[:, sess] > 0)[0]
            for row in cluster_rows:
                this_cell = cmap[row, sess]          # 1-based
                this_cell_0 = this_cell - 1           # 0-based for array access
                this_cent = centroid_locations[sess][this_cell_0]

                dist_to_clusters = np.sqrt(
                    ((cluster_cents - this_cent[:, None]) ** 2).sum(0))
                candidates = np.where(dist_to_clusters < cluster_dist_thresh)[0]
                if len(candidates) == 0:
                    continue

                max_sim = -1.0
                best_cluster = row

                for cl in candidates:
                    cluster_sessions = np.where(cmap[cl] > 0)[0]
                    cluster_sessions = cluster_sessions[cluster_sessions != sess]
                    if len(cluster_sessions) == 0:
                        continue
                    for s2 in cluster_sessions:
                        in_cluster_cell = cmap[cl, s2]       # 1-based
                        neighbor_inds = all_to_all_indexes[sess][s2][this_cell_0]
                        match = np.where(neighbor_inds == (in_cluster_cell - 1))[0]  # compare 0-based
                        if len(match) == 0:
                            continue
                        ps_vec = all_to_all_p_same[sess][s2][this_cell_0]
                        if len(ps_vec) == 0:
                            continue
                        ps = float(ps_vec[match[0]])
                        if ps > max_sim:
                            max_sim = ps
                            best_cluster = cl

                if best_cluster == row:
                    continue

                if max_sim < p_same_threshold:
                    if cmap[row].sum() > cmap[row, sess]:
                        num_clusters += 1
                        new_row = np.zeros((1, n_sessions), dtype=int)
                        new_row[0, sess] = this_cell  # already 1-based
                        cmap = np.vstack([cmap, new_row])
                        cluster_cents = np.hstack(
                            [cluster_cents, this_cent[:, None]])
                        cmap[row, sess] = 0
                        changes_count += 1
                        separation_count += 1
                else:
                    target_sess_in_cluster = np.where(cmap[best_cluster] > 0)[0]
                    target_sess_in_cluster = target_sess_in_cluster[
                        target_sess_in_cluster != sess]

                    if sess in np.where(cmap[best_cluster] > 0)[0]:
                        occupant = cmap[best_cluster, sess]  # 1-based
                        occupant_0 = occupant - 1             # 0-based
                        occupant_sim = 0.0
                        for s2 in target_sess_in_cluster:
                            in_cl = cmap[best_cluster, s2]   # 1-based
                            ni    = all_to_all_indexes[sess][s2][occupant_0]
                            match = np.where(ni == (in_cl - 1))[0]  # compare 0-based
                            if len(match):
                                ps_v = all_to_all_p_same[sess][s2][occupant_0]
                                if len(ps_v):
                                    occupant_sim += float(ps_v[match[0]])
                        if max_sim > occupant_sim:
                            occupant_cent = centroid_locations[sess][occupant_0]
                            num_clusters += 1
                            new_row = np.zeros((1, n_sessions), dtype=int)
                            new_row[0, sess] = occupant  # already 1-based
                            cmap = np.vstack([cmap, new_row])
                            cluster_cents = np.hstack(
                                [cluster_cents, occupant_cent[:, None]])
                            cmap[best_cluster, sess] = this_cell  # 1-based
                            cmap[row, sess] = 0
                            changes_count += 1
                            switch_count  += 1
                    else:
                        cmap[row, sess] = 0
                        cmap[best_cluster, sess] = this_cell  # 1-based
                        changes_count += 1
                        move_count    += 1

        keep = np.where(cmap.sum(1) > 0)[0]
        cmap = cmap[keep]
        delete_count = len(keep) - cmap.shape[0]

        cluster_cents = _cluster_centroids(cmap)
        n_clusters = cmap.shape[0]
        merged = np.zeros(n_clusters, dtype=bool)

        for n in range(n_clusters):
            if merged[n]:
                continue
            this_sessions = np.where(cmap[n] > 0)[0]
            dist_to_cl = np.sqrt(
                ((cluster_cents - cluster_cents[:, n:n+1]) ** 2).sum(0))
            near = np.where(dist_to_cl < cluster_dist_thresh)[0]
            near = near[near != n]

            for cl in near:
                if merged[cl]:
                    continue
                cand_sessions = np.where(cmap[cl] > 0)[0]
                if len(np.intersect1d(this_sessions, cand_sessions)) > 0:
                    continue
                max_ps = 0.0
                for s1 in this_sessions:
                    for s2 in cand_sessions:
                        cell1 = cmap[n, s1]      # 1-based
                        cell2 = cmap[cl, s2]      # 1-based
                        ni    = all_to_all_indexes[s1][s2][cell1 - 1]  # 0-based access
                        match = np.where(ni == (cell2 - 1))[0]         # compare 0-based
                        if len(match):
                            ps_v = all_to_all_p_same[s1][s2][cell1 - 1]
                            if len(ps_v):
                                max_ps = max(max_ps, float(ps_v[match[0]]))
                if max_ps >= p_same_threshold:
                    cmap[n, cand_sessions] = cmap[cl, cand_sessions]
                    cmap[cl, :] = 0
                    merged[cl] = True
                    this_sessions = np.where(cmap[n] > 0)[0]
                    changes_count += 1

        keep = np.where(cmap.sum(1) > 0)[0]
        cmap = cmap[keep]

        print(f"  Iteration {iteration}: {changes_count} changes "
              f"(sep={separation_count} switch={switch_count} "
              f"move={move_count})")

    final_cents = _cluster_centroids(cmap)
    return cmap, final_cents


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5b — Cell scores
# ═══════════════════════════════════════════════════════════════════════════

def compute_scores(cell_to_index_map, all_to_all_indexes, all_to_all_p_same,
                   n_sessions):
    """
    Compute per-cluster quality scores.
    """
    n_clusters = cell_to_index_map.shape[0]
    scores_combined  = np.full(n_clusters, np.nan)
    scores_positive  = np.full(n_clusters, np.nan)
    scores_negative  = np.full(n_clusters, np.nan)
    scores_exclusive = np.full(n_clusters, np.nan)

    for n in range(n_clusters):
        cells = cell_to_index_map[n]
        good = good_pos = good_neg = good_excl = 0.0
        n_comp = n_comp_pos = n_comp_neg = 0

        for s1 in range(n_sessions):
            if cells[s1] == 0:
                continue
            cell1 = cells[s1]          # 1-based
            cell1_0 = cell1 - 1        # 0-based for array access
            for s2 in range(n_sessions):
                if s2 == s1:
                    continue
                n_comp += 1
                neighbors = all_to_all_indexes[s1][s2][cell1_0]
                ps_vec    = all_to_all_p_same[s1][s2][cell1_0]

                if cells[s2] == 0:
                    n_comp_neg += 1
                    if len(neighbors) == 0:
                        good += 1; good_neg += 1
                    elif len(ps_vec):
                        contrib = 1.0 - ps_vec.sum()
                        good += contrib; good_neg += contrib
                    else:
                        good += 1; good_neg += 1
                else:
                    n_comp_pos += 1
                    cell2 = cells[s2]       # 1-based
                    match = np.where(neighbors == (cell2 - 1))[0]  # compare 0-based
                    if len(match) and len(ps_vec):
                        tp = float(ps_vec[match[0]])
                        other_ps = np.delete(ps_vec, match[0])
                        good     += tp - other_ps.sum()
                        good_pos += tp
                        if len(other_ps) == 0:
                            good_excl += 1
                        else:
                            good_excl += 1.0 - other_ps.sum()
                    else:
                        good_pos  += 0

        if n_comp_pos > 0:
            scores_positive[n]  = good_pos  / n_comp_pos
            scores_exclusive[n] = good_excl / n_comp_pos
        if n_comp_neg > 0:
            scores_negative[n]  = good_neg  / n_comp_neg
        if n_comp > 0:
            scores_combined[n]  = good / n_comp

    return scores_positive, scores_negative, scores_exclusive, scores_combined


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_cellreg(file_paths, microns_per_pixel=1.0, maximal_distance_um=14.0,
                p_same_threshold=0.5, n_bins=50, footprint_dir=None):
    """
    Full CellReg pipeline for .npy session files.

    Parameters
    ----------
    file_paths          : list of str/Path
    microns_per_pixel   : float
    maximal_distance_um : float  (µm) — max centroid search radius
    p_same_threshold    : float  — P_same cutoff for registration
    n_bins              : int    — histogram bins for model fitting
    footprint_dir       : str/Path or None — if provided, load footprints,
                          fit spatial model, and let choose_best_model pick
    """
    maximal_distance_px = maximal_distance_um / microns_per_pixel
    use_spatial = footprint_dir is not None

    print("\n── Stage 1: Loading sessions ──")
    centroid_locations, roi_ids_list = load_sessions(file_paths)

    if use_spatial:
        print("\n── Stage 1b: Loading spatial footprints ──")
        footprints_list = load_footprints(file_paths, footprint_dir)

    print("\n── Stage 2: Alignment (identity pass-through) ──")
    centroid_locations = identity_alignment(centroid_locations)

    print("\n── Stage 3a: Computing neighbor distances ──")
    all_to_all_indexes, all_to_all_distances, neighbors_distances = \
        compute_neighbor_distances(centroid_locations, maximal_distance_px)

    if len(neighbors_distances) < 10:
        raise ValueError("Too few neighbor pairs — check maximal_distance or data scale.")

    print("\n── Stage 3b: Fitting centroid distance model ──")
    (bin_centers, hist_norm, p_same_lookup, same_model_cd, diff_model_cd,
     params, intersection_px, mse_cd) = fit_centroid_distance_model(
        neighbors_distances, microns_per_pixel, n_bins, maximal_distance_px)

    print("\n── Stage 3c: Assigning centroid-distance P_same ──")
    all_to_all_p_same_cd = assign_p_same(
        all_to_all_distances, p_same_lookup, bin_centers)

    # ── Spatial correlation path (optional) ──
    spatial_model_results = None
    best_model = 'centroid'
    all_to_all_p_same = all_to_all_p_same_cd

    if use_spatial:
        print("\n── Stage 3a+: Computing spatial correlations ──")
        all_to_all_spatial_corr, neighbors_correlations = \
            compute_neighbor_spatial_correlations(all_to_all_indexes, footprints_list)

        if len(neighbors_correlations) >= 10:
            print("\n── Stage 3b+: Fitting spatial correlation model (EM) ──")
            (bin_centers_sc, hist_norm_sc, p_same_sc_lookup,
             same_model_sc, diff_model_sc,
             params_sc, mse_sc, intersection_sc) = fit_spatial_correlation_model(
                neighbors_correlations, n_bins)

            print("\n── Stage 3c+: Assigning spatial P_same ──")
            all_to_all_p_same_sc = assign_p_same_spatial(
                all_to_all_spatial_corr, p_same_sc_lookup, bin_centers_sc)

            spatial_model_results = {
                'bin_centers_sc': bin_centers_sc,
                'p_same_sc_lookup': p_same_sc_lookup,
                'same_model_sc': same_model_sc,
                'diff_model_sc': diff_model_sc,
                'spatial_params': params_sc,
                'spatial_mse': mse_sc,
                'spatial_intersection': intersection_sc,
            }

            # ── Model selection (matching choose_best_model.m) ──
            print("\n── Stage 3d: Choosing best model ──")
            best_model = choose_best_model(
                mse_cd, same_model_cd, diff_model_cd, p_same_lookup,
                mse_sc, same_model_sc, diff_model_sc, p_same_sc_lookup)

            if best_model == 'spatial':
                all_to_all_p_same = all_to_all_p_same_sc
            else:
                all_to_all_p_same = all_to_all_p_same_cd
        else:
            print("  WARNING: Too few spatial correlations; using centroid-only")

    print("\n── Stage 4: Initial greedy registration ──")
    cell_to_index_map = initial_registration(
        centroid_locations, maximal_distance_px, intersection_px)

    print("\n── Stage 5: Iterative clustering ──")
    optimal_cell_to_index_map, cluster_centroids = cluster_cells(
        cell_to_index_map, all_to_all_p_same, all_to_all_indexes,
        centroid_locations, maximal_distance_px, p_same_threshold)

    print("\n── Stage 5b: Computing cell scores ──")
    n_sessions = len(centroid_locations)
    scores_pos, scores_neg, scores_excl, scores = compute_scores(
        optimal_cell_to_index_map, all_to_all_indexes,
        all_to_all_p_same, n_sessions)

    # Convert cell_to_index_map back to 0-based indexing for output
    # (internally we used 1-based to avoid the cell-0 ambiguity)
    # Use -1 as the "empty" sentinel so cell index 0 is valid
    output_cmap = optimal_cell_to_index_map.copy()
    output_cmap[output_cmap == 0] = -1       # empty → -1
    output_cmap[output_cmap > 0] -= 1        # 1-based → 0-based
    optimal_cell_to_index_map = output_cmap

    n_cells   = optimal_cell_to_index_map.shape[0]
    n_sess_counts = (optimal_cell_to_index_map >= 0).sum(1)
    registered_all = (n_sess_counts == n_sessions).sum()

    if use_spatial and spatial_model_results:
        mode_str = f"best model = {best_model} (centroid + spatial evaluated)"
    else:
        mode_str = "centroid-only"

    print(f"\n══════════════════════════════════════")
    print(f"  Mode                 : {mode_str}")
    print(f"  Total registered cells : {n_cells}")
    print(f"  Present in all sessions: {registered_all}")
    print(f"  Mean cell score        : {np.nanmean(scores):.3f}")
    print(f"  Mean true positive     : {np.nanmean(scores_pos):.3f}")
    print(f"  Mean true negative     : {np.nanmean(scores_neg):.3f}")
    print(f"  Model MSE (centroid)   : {mse_cd:.4f}")
    if spatial_model_results:
        print(f"  Model MSE (spatial)    : {spatial_model_results['spatial_mse']:.4f}")
    print(f"  P_same=0.5 threshold   : {intersection_px * microns_per_pixel:.2f} µm")
    print(f"══════════════════════════════════════\n")

    result = {
        'cell_to_index_map':    optimal_cell_to_index_map,
        'cluster_centroids':    cluster_centroids,
        'cell_scores':          scores,
        'cell_scores_positive': scores_pos,
        'cell_scores_negative': scores_neg,
        'cell_scores_exclusive': scores_excl,
        'p_same_lookup':        p_same_lookup,
        'bin_centers':          bin_centers,
        'model_params':         params,
        'model_mse':            mse_cd,
        'intersection_um':      intersection_px * microns_per_pixel,
        'mode':                 mode_str,
        'best_model':           best_model,
    }
    if spatial_model_results:
        result.update(spatial_model_results)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# ▶  Configure and run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║                     PATH CONFIGURATION                           ║
    # ╠═══════════════════════════════════════════════════════════════════╣

    # DATA_DIR   = Path(r"C:\Users\ariAccount\Desktop\Stars2Cells_Benchmark_100n")
    # OUTPUT_DIR = Path(r"C:\Users\ariAccount\Desktop\CellReg_100_test")
    # GT_PATH    = Path(r"C:\Users\ariAccount\Desktop\Ground_Truth_Sample") / "100_ground_truth.npy"
    # S2C_DIR    = DATA_DIR / "step_3_results"

    # # ── Spatial footprints (set to None to run centroid-only) ──
    # FOOTPRINT_DIR = DATA_DIR / "spatial_mapping"
    # # FOOTPRINT_DIR = None   # ← uncomment to disable spatial correlations

    # FILE_PATHS = [
    #     DATA_DIR / "101_1__A_combined__seed0.npy",
    #     DATA_DIR / "101_2__A_combined__seed0.npy",
    #     DATA_DIR / "101_3__A_combined__seed0.npy",
    #     DATA_DIR / "101_4__A_combined__seed0.npy",
    #     DATA_DIR / "101_5__A_combined__seed0.npy",
    # ]

    # ╠═══════════════════════════════════════════════════════════════════╣
    # ║                    PARAMETER CONFIGURATION                       ║
    # ╠═══════════════════════════════════════════════════════════════════╣

    # MICRONS_PER_PIXEL   = 1.0
    # MAXIMAL_DISTANCE_UM = 20.0
    # P_SAME_THRESHOLD    = 0.5

    # ╚═══════════════════════════════════════════════════════════════════╝

    DATA_DIR   = Path(r"C:\Users\ariAccount\Desktop\cellreg_test_2sess")
    OUTPUT_DIR = Path(r"C:\Users\ariAccount\Desktop\cellreg_test_2sess\output")
    GT_PATH    = DATA_DIR / "ground_truth.npy"
    S2C_DIR    = DATA_DIR / "step_3_results"

    FOOTPRINT_DIR = DATA_DIR / "spatial_mapping"

    FILE_PATHS = [
        DATA_DIR / "test_sess1.npy",
        DATA_DIR / "test_sess2.npy",
    ]

    MICRONS_PER_PIXEL   = 1.0
    MAXIMAL_DISTANCE_UM = 30.0
    P_SAME_THRESHOLD    = 0.5

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Run pipeline ──
    results = run_cellreg(
        file_paths          = FILE_PATHS,
        microns_per_pixel   = MICRONS_PER_PIXEL,
        maximal_distance_um = MAXIMAL_DISTANCE_UM,
        p_same_threshold    = P_SAME_THRESHOLD,
        footprint_dir       = FOOTPRINT_DIR,
    )

    np.save(OUTPUT_DIR / "cellreg_results.npy", results, allow_pickle=True)
    print(f"Results saved → {OUTPUT_DIR / 'cellreg_results.npy'}")

    cmap = results['cell_to_index_map']
    print("cell_to_index_map shape:", cmap.shape, "  (rows=cells, cols=sessions)")

    # ═══════════════════════════════════════════════════════════════════════
    # Helper: pairwise scoring against ground truth
    # ═══════════════════════════════════════════════════════════════════════

    def score_pair_simple(matched_ref, matched_tgt, ref_base_ids, tgt_base_ids):
        ref_id_set = set(ref_base_ids)
        tgt_id_set = set(tgt_base_ids)
        shared_ids = ref_id_set & tgt_id_set

        matched_ref = np.asarray(matched_ref, dtype=int)
        matched_tgt = np.asarray(matched_tgt, dtype=int)

        tp = fp = 0
        correctly_matched = set()
        for ri, ti in zip(matched_ref, matched_tgt):
            if ri < len(ref_base_ids) and ti < len(tgt_base_ids):
                if ref_base_ids[ri] == tgt_base_ids[ti]:
                    tp += 1
                    correctly_matched.add(ref_base_ids[ri])
                else:
                    fp += 1
            else:
                fp += 1

        fn = len(shared_ids - correctly_matched)

        matched_ref_set = set(matched_ref.tolist())
        matched_tgt_set = set(matched_tgt.tolist())
        nonshared_ref_caught = sum(
            1 for i, bid in enumerate(ref_base_ids)
            if bid not in shared_ids and i in matched_ref_set
        )
        nonshared_tgt_caught = sum(
            1 for i, bid in enumerate(tgt_base_ids)
            if bid not in shared_ids and i in matched_tgt_set
        )
        n_nonshared = (
            sum(1 for bid in ref_base_ids if bid not in shared_ids) +
            sum(1 for bid in tgt_base_ids if bid not in shared_ids)
        )
        tn = n_nonshared - nonshared_ref_caught - nonshared_tgt_caught

        prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else float('nan')
        return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                    n_shared=len(shared_ids),
                    precision=prec, recall=recall, f1=f1)

    def cmap_to_pairs(cmap, si, sj):
        rows = np.where((cmap[:, si] >= 0) & (cmap[:, sj] >= 0))[0]
        return cmap[rows, si], cmap[rows, sj]

    def plot_confusion(ax, scores, title):
        mat     = np.array([[scores['tp'], scores['fp']],
                            [scores['fn'], scores['tn']]], dtype=float)
        mat_pct = mat / mat.sum() * 100 if mat.sum() > 0 else mat
        colors  = np.array([
            [[0.2, 0.75, 0.3, 0.8],  [0.85, 0.2, 0.2, 0.7]],
            [[0.85, 0.2, 0.2, 0.7],  [0.2, 0.75, 0.3, 0.8]],
        ])
        ax.imshow(colors, aspect='auto', interpolation='none')
        labels = [['True Positive', 'False Positive'],
                  ['False Negative', 'True Negative']]
        for r in range(2):
            for c in range(2):
                ax.text(c, r,
                        f"{labels[r][c]}\n{int(mat[r,c])}  ({mat_pct[r,c]:.1f}%)",
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted +', 'Predicted -'], fontsize=10)
        ax.set_yticklabels(['Actual +', 'Actual -'], fontsize=10)
        p = scores['precision']; r = scores['recall']; f = scores['f1']
        pstr = f"{p*100:.1f}%" if not np.isnan(p) else "N/A"
        rstr = f"{r*100:.1f}%" if not np.isnan(r) else "N/A"
        fstr = f"{f*100:.1f}%" if not np.isnan(f) else "N/A"
        ax.set_title(f"{title}\nP={pstr}  R={rstr}  F1={fstr}",
                     fontsize=11, pad=8)

    # ═══════════════════════════════════════════════════════════════════════
    # Evaluate
    # ═══════════════════════════════════════════════════════════════════════

    if not GT_PATH.exists():
        print(f"\nGround truth not found at {GT_PATH} -- skipping evaluation.")
    else:
        print(f"\n-- Ground truth evaluation (pairwise) --")
        gt = np.load(GT_PATH, allow_pickle=True).item()

        gt_base_ids = []
        for fp in FILE_PATHS:
            key = Path(fp).name
            if key not in gt:
                raise KeyError(f"GT key not found: {key}")
            gt_base_ids.append(gt[key]['ground_truth_base_ids'])

        n_sess     = len(FILE_PATHS)
        sess_names = [Path(fp).name.replace(".npy", "") for fp in FILE_PATHS]
        pairs = [(i, j) for i in range(n_sess) for j in range(n_sess) if i < j]

        cr_pair_scores = {}
        for si, sj in pairs:
            matched_ref, matched_tgt = cmap_to_pairs(cmap, si, sj)
            s = score_pair_simple(matched_ref, matched_tgt,
                                  gt_base_ids[si], gt_base_ids[sj])
            cr_pair_scores[(si, sj)] = s

        def _agg(pair_scores_dict):
            tp = sum(s['tp'] for s in pair_scores_dict.values())
            fp = sum(s['fp'] for s in pair_scores_dict.values())
            fn = sum(s['fn'] for s in pair_scores_dict.values())
            tn = sum(s['tn'] for s in pair_scores_dict.values())
            prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else float('nan')
            return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                        precision=prec, recall=recall, f1=f1)

        cellreg_scores = _agg(cr_pair_scores)
        print(f"  CellReg ({results['mode']})  -- "
              f"TP={cellreg_scores['tp']}  FP={cellreg_scores['fp']}  "
              f"FN={cellreg_scores['fn']}  TN={cellreg_scores['tn']}")
        print(f"             P={cellreg_scores['precision']*100:.1f}%  "
              f"R={cellreg_scores['recall']*100:.1f}%  "
              f"F1={cellreg_scores['f1']*100:.1f}%")

        print(f"\n  {'Pair':<12} {'Shared':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'TN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"  {'-'*65}")
        for (si, sj), s in cr_pair_scores.items():
            pstr = f"{s['precision']*100:.1f}%" if not np.isnan(s['precision']) else "N/A"
            rstr = f"{s['recall']*100:.1f}%"    if not np.isnan(s['recall'])    else "N/A"
            fstr = f"{s['f1']*100:.1f}%"        if not np.isnan(s['f1'])        else "N/A"
            print(f"  sess{si+1} x sess{sj+1}  "
                  f"{s['n_shared']:>6} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                  f"{s['tn']:>4} {pstr:>7} {rstr:>7} {fstr:>7}")

        # ── Score S2C pairwise ──
        s2c_scores     = None
        s2c_pair_scores = {}

        if S2C_DIR.exists():
            all_sweeps = sorted(S2C_DIR.glob("*_sweep.npz"))
            relevant_sweeps = [
                f for f in all_sweeps
                if any(s in f.stem for s in sess_names)
            ]
            print(f"\n-- Stars2Cells comparison --")
            print(f"  Using {len(relevant_sweeps)} / {len(all_sweeps)} sweep files")

            sweep_index = {}
            for sweep_file in relevant_sweeps:
                data = np.load(sweep_file, allow_pickle=False)
                ref_s = str(data['ref_session'])
                tgt_s = str(data['target_session'])
                if ref_s in sess_names and tgt_s in sess_names:
                    sweep_index[(ref_s, tgt_s)] = data

            for si, sj in pairs:
                ref_name = sess_names[si]
                tgt_name = sess_names[sj]
                data = sweep_index.get((ref_name, tgt_name)) or \
                       sweep_index.get((tgt_name, ref_name))
                if data is None:
                    print(f"  WARN: no sweep file for sess{si+1} x sess{sj+1}")
                    continue
                matched_ref = data['matched_ref_indices']
                matched_tgt = data['matched_tgt_indices']
                if (tgt_name, ref_name) in sweep_index and \
                   (ref_name, tgt_name) not in sweep_index:
                    s = score_pair_simple(matched_tgt, matched_ref,
                                          gt_base_ids[si], gt_base_ids[sj])
                else:
                    s = score_pair_simple(matched_ref, matched_tgt,
                                          gt_base_ids[si], gt_base_ids[sj])
                s2c_pair_scores[(si, sj)] = s

            if s2c_pair_scores:
                s2c_scores = _agg(s2c_pair_scores)
                print(f"  S2C      -- TP={s2c_scores['tp']}  FP={s2c_scores['fp']}  "
                      f"FN={s2c_scores['fn']}  TN={s2c_scores['tn']}")
                print(f"             P={s2c_scores['precision']*100:.1f}%  "
                      f"R={s2c_scores['recall']*100:.1f}%  "
                      f"F1={s2c_scores['f1']*100:.1f}%")

                print(f"\n  {'Pair':<12} {'Shared':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
                      f"{'TN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
                print(f"  {'-'*65}")
                for (si, sj), s in s2c_pair_scores.items():
                    pstr = f"{s['precision']*100:.1f}%" if not np.isnan(s['precision']) else "N/A"
                    rstr = f"{s['recall']*100:.1f}%"    if not np.isnan(s['recall'])    else "N/A"
                    fstr = f"{s['f1']*100:.1f}%"        if not np.isnan(s['f1'])        else "N/A"
                    print(f"  sess{si+1} x sess{sj+1}  "
                          f"{s['n_shared']:>6} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                          f"{s['tn']:>4} {pstr:>7} {rstr:>7} {fstr:>7}")
        else:
            print(f"\nS2C results dir not found at {S2C_DIR} -- skipping.")

        # ── Plot confusion matrices ──
        n_plots = 2 if s2c_scores else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        plot_confusion(axes[0], cellreg_scores,
                       f"CellReg ({results['mode']})\nvs Ground Truth")
        if s2c_scores:
            plot_confusion(axes[1], s2c_scores, "Stars2Cells vs Ground Truth")

        fig.tight_layout()
        fig_path = OUTPUT_DIR / "confusion_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n  Confusion matrix saved -> {fig_path}")

        # ── JSON summary ──
        def _fmt(d):
            if d is None:
                return None
            return {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in d.items()}

        def _fmt_pairs(pair_dict):
            return {
                f"sess{si+1}_x_sess{sj+1}": _fmt(s)
                for (si, sj), s in pair_dict.items()
            }

        summary = {
            "config": {
                "data_dir":            str(DATA_DIR),
                "output_dir":          str(OUTPUT_DIR),
                "gt_path":             str(GT_PATH),
                "s2c_dir":             str(S2C_DIR),
                "footprint_dir":       str(FOOTPRINT_DIR) if FOOTPRINT_DIR else None,
                "file_paths":          [str(fp) for fp in FILE_PATHS],
                "microns_per_pixel":   MICRONS_PER_PIXEL,
                "maximal_distance_um": MAXIMAL_DISTANCE_UM,
                "p_same_threshold":    P_SAME_THRESHOLD,
                "mode":                results['mode'],
            },
            "cellreg": {
                "n_clusters":      int(cmap.shape[0]),
                "n_sessions":      int(cmap.shape[1]),
                "mean_cell_score": round(float(np.nanmean(results['cell_scores'])), 4),
                "model_mse":       round(float(results['model_mse']), 4),
                "intersection_um": round(float(results['intersection_um']), 3),
                "aggregate":       _fmt(cellreg_scores),
                "per_pair":        _fmt_pairs(cr_pair_scores),
            },
            "stars2cells": {
                "aggregate": _fmt(s2c_scores),
                "per_pair":  _fmt_pairs(s2c_pair_scores),
            },
        }

        json_path = OUTPUT_DIR / "evaluation_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON summary saved  -> {json_path}")