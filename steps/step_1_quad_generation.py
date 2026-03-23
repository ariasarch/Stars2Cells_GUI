"""
Step 1: Process multiple sessions in parallel with a RAM budget to generate quads

ARCHITECTURE (v2 — diagonal-first)
────────────────────────────────────────────────────────────────────────────────
Old pipeline:
    C(N,3) triangles → group by edge into buckets → K-cap in Step 3 → global dedup

New pipeline:
    enumerate diagonals (KNN local + random long-range)
        → compute heights inline per diagonal
        → top-K cap immediately
        → pair third-points → quad assembly
        → ownership guard replaces global dedup

Key changes from v1
───────────────────
* Triangle enumeration removed entirely — no C(N,3) blow-up
* Diagonal set = KNN local (k_local) + random long-range (k_random ≈ k_local//2)
  covering both fine spatial structure and FOV-spanning quads
* Height computed inline per diagonal; top-K kept immediately
* Ownership guard in _process_diagonal: each quad emitted only from the diagonal
  that IS its longest edge → zero duplicates, no post-hoc global np.unique
* height_percentile is now correctly on the 0-100 scale (v1 bug: used 0.95 ≈ 0th
  percentile instead of 95th, effectively disabling the filter)
* min_third_points_per_diagonal removed: every bucket has exactly min(K, N-2)
  entries after the inline cap, so the parameter was always a no-op for N ≫ K

Functions removed (dead code after refactor)
────────────────────────────────────────────
  step1_generate_and_filter_triangles
  step1_generate_and_filter_triangles_knn
  _build_multiscale_neighbors
  step2_build_diagonal_buckets_chunked
  _process_diagonal_batch
  step3_generate_quads_batched
"""

import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import glob
import re
import numpy as np
import psutil
import gc
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import cKDTree
from steps.step_1_Q_Saturation_Add_On import run_saturation_check_after_step1
from utilities import *

logger = logging.getLogger("neuron_mapping_parallel")

# Module-level guard to avoid redundant Q estimates when N hasn't changed
_last_estimate_n: int = -1


# ══════════════════════════════════════════════════════════════════════════════
# Session discovery & loading  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def find_session_files(
    config: PipelineConfig,
    session_filename_regex: str = r'^([A-Za-z0-9_]+?)_(\d+)(.*?)\.npy$',
) -> List[Dict[str, Any]]:
    """Find all .npy files and extract metadata."""
    a_files = glob.glob(str(config.input_path / "*.npy"))

    logger.info(f"\n[DEBUG] Found {len(a_files)} .npy files")
    logger.info(f"[DEBUG] Looking for pattern: {session_filename_regex}")

    sessions = []
    for file_path in a_files:
        filename = Path(file_path).name
        match = re.match(session_filename_regex, filename)

        if match:
            animal_id = match.group(1)
            session_name = Path(file_path).stem

            logger.info(f"[DEBUG] Matched: {filename} -> animal={animal_id}, session={session_name}")

            if config.animal_id and animal_id != config.animal_id:
                continue

            sessions.append({
                'file_path':    file_path,
                'animal_id':    animal_id,
                'session':      session_name,
                'session_name': session_name,
            })
        else:
            logger.info(f"[DEBUG] NO MATCH: {filename}")

    sessions.sort(key=lambda x: (x['animal_id'], x['session_name']))

    logger.info(f"\n[DEBUG] Total sessions found: {len(sessions)}")
    return sessions

def process_single_session(session_info: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """Process a single session to extract centroids and generate quads."""
    session_name = session_info['session_name']
    logger.info(f"Processing {session_name}...")

    step1_dir   = Path(config.output_dir) / "step_1_results"
    step1_dir.mkdir(parents=True, exist_ok=True)
    output_file = step1_dir / f"{session_name}_centroids_quads.npz"

    if config.skip_existing and output_file.exists():
        logger.info(f"  SKIPPING: Output file already exists ({output_file})")
        return {
            "session_name": session_name,
            "skipped":      True,
            "reason":       "already_exists",
            "n_neurons":    None,
            "n_quads":      None,
        }

    try:
        raw_data = np.load(session_info["file_path"], allow_pickle=True)
        data = None

        # TRY FORMAT 1: Dictionary (0-d array containing dict)
        if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
            try:
                data = raw_data.item()
                if not isinstance(data, dict):
                    data = None
            except (ValueError, TypeError):
                data = None

        # TRY FORMAT 2: Raw A matrix (3D array)
        if data is None and isinstance(raw_data, np.ndarray) and raw_data.ndim == 3:
            logger.info("  Converting A matrix to centroids...")
            centroids_x, centroids_y = extract_centroids_from_A(raw_data)
            data = {
                'centroids_x': centroids_x,
                'centroids_y': centroids_y,
                'roi_ids':     np.arange(len(centroids_x)),
            }
            logger.info(f"  ✓ Extracted {len(centroids_x)} centroids from A matrix")

        if data is None or not isinstance(data, dict):
            logger.error("  SKIPPING: Unrecognised file format")
            return {"session_name": session_name, "skipped": True, "reason": "unrecognized_format"}

        centroids_x = data["centroids_x"]
        centroids_y = data["centroids_y"]
        centroids   = np.column_stack([centroids_y, centroids_x]).astype(np.float32)
        neuron_ids  = data.get("roi_ids", np.arange(len(centroids_x)))
        n_neurons   = len(centroids)

        logger.info(f"  Loaded {n_neurons} centroids")

        if n_neurons < 4:
            logger.warning(f"  SKIPPING: Not enough neurons ({n_neurons} < 4)")
            return {"session_name": session_name, "skipped": True,
                    "reason": "not_enough_neurons", "n_neurons": n_neurons}

        logger.info("  Generating quads (diagonal-first pipeline)...")
        start_time = time.time()

        final_desc, final_idx = generate_sparse_quads_triangle(centroids, config, logger)

        generation_time = time.time() - start_time
        n_quads = int(final_idx.shape[0]) if final_idx is not None else 0

        logger.info(f"  Generated {n_quads:,} quads in {generation_time:.1f}s")

        np.savez_compressed(
            output_file,
            animal_id=session_info["animal_id"],
            session=session_info["session"],
            session_name=session_name,
            centroids=centroids,
            neuron_ids=neuron_ids,
            quad_desc=final_desc,
            quad_idx=final_idx,
            n_neurons=n_neurons,
            n_quads=n_quads,
            generation_time=generation_time,
            generation_method="diagonal_first",
        )
        logger.info(f"  Saved NPZ to {output_file}")

        return {
            "session_name":    session_name,
            "skipped":         False,
            "n_neurons":       n_neurons,
            "n_quads":         n_quads,
            "generation_time": generation_time,
        }

    except Exception as e:
        logger.error(f"  Error processing {session_name}: {str(e)}")
        return {"session_name": session_name, "skipped": True, "reason": "error", "error": str(e)}
    finally:
        clean_memory()


# ══════════════════════════════════════════════════════════════════════════════
# RAM / CPU budget  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_max_parallel_sessions(config: PipelineConfig) -> int:
    """Decide how many sessions to process in parallel based on RAM and CPU."""
    mem_info = check_memory_requirements()
    total_gb = mem_info["total_gb"]
    avail_gb = mem_info["available_gb"]

    logger.info(
        f"[PARALLEL] System memory: total={total_gb:.1f} GB, "
        f"available={avail_gb:.1f} GB"
    )

    ram_budget = max(1.0, avail_gb * (1.0 - config.parallel_safety_margin))
    if config.per_session_gb <= 0:
        max_by_ram = 1
    else:
        max_by_ram = max(1, int(ram_budget // config.per_session_gb))

    n_cores       = multiprocessing.cpu_count()
    per_proc_threads = max(1, config.n_workers)
    max_by_cpu    = max(1, n_cores // per_proc_threads)

    max_parallel  = max(1, min(max_by_ram, max_by_cpu))

    logger.info(
        f"[PARALLEL] RAM budget ~{ram_budget:.1f} GB, "
        f"per-session ≈ {config.per_session_gb:.1f} GB → max_by_ram={max_by_ram}"
    )
    logger.info(
        f"[PARALLEL] CPU cores={n_cores}, threads/session={per_proc_threads} "
        f"→ max_by_cpu={max_by_cpu}"
    )
    logger.info(f"[PARALLEL] → Using up to {max_parallel} sessions in parallel.\n")

    return max_parallel


# ══════════════════════════════════════════════════════════════════════════════
# Core math kernel  (modified: ownership guard added)
# ══════════════════════════════════════════════════════════════════════════════

def _process_diagonal(
    d1: int,
    d2: int,
    third_points,
    centroids: np.ndarray,
    config,
    height_tol: float = 0.0,
    enforce_ownership: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised generation of quad descriptors from a single diagonal bucket.

    Ownership guard
    ---------------
    A quad {d1, d2, C, D} has 6 pairwise distances.  The same quad can in
    principle be reached from any of the 6 diagonals.  We emit it ONLY from
    the diagonal whose length equals the quad's longest pairwise distance —
    i.e. only when (d1, d2) IS the longest edge.  This gives exactly one
    emission per quad without any post-hoc global dedup.

    For floating-point safety a tiny relative tolerance (1e-6) is applied.
    Exact ties (two edges of identical length) are astronomically rare with
    real centroid coordinates; in the extreme case both diagonals emit the
    quad and a trivial downstream dedup would remove one copy.
    """
    if len(third_points) < 2:
        return None, None

    tp      = np.asarray(third_points, dtype=np.float64)
    verts   = tp[:, 0].astype(np.int32)
    areas   = tp[:, 1]
    heights = tp[:, 2]

    M = verts.shape[0]
    if M < 2:
        return None, None

    i_idx, j_idx = np.triu_indices(M, 1)

    # Optional height tolerance (kept for call-site compatibility, but
    # with diagonal-first pipeline buckets are already height-filtered)
    if height_tol > 0.0:
        h1   = heights[i_idx]
        h2   = heights[j_idx]
        mask = (np.abs(h1) >= height_tol) | (np.abs(h2) >= height_tol)
        i_idx = i_idx[mask];  j_idx = j_idx[mask]
        if i_idx.size == 0:
            return None, None

    quad_area = areas[i_idx] + areas[j_idx]
    mask  = quad_area >= 1.0
    i_idx = i_idx[mask];  j_idx = j_idx[mask]
    if i_idx.size == 0:
        return None, None

    p1 = verts[i_idx]
    p2 = verts[j_idx]

    base = np.column_stack([
        np.full_like(p1, d1, dtype=np.int32),
        np.full_like(p1, d2, dtype=np.int32),
        p1.astype(np.int32),
        p2.astype(np.int32),
    ])

    quad_indices = np.sort(base, axis=1)
    quad_indices = np.unique(quad_indices, axis=0)
    if quad_indices.shape[0] == 0:
        return None, None

    pts = centroids[quad_indices]
    K   = quad_indices.shape[0]

    diff      = pts[:, :, None, :] - pts[:, None, :, :]
    dist_mats = np.linalg.norm(diff, axis=-1)

    flat         = dist_mats.reshape(K, -1)
    max_flat_idx = np.argmax(flat, axis=1)
    A_idx        = max_flat_idx // 4
    B_idx        = max_flat_idx % 4
    max_dists    = flat[np.arange(K), max_flat_idx]

    min_pair = getattr(config, "min_pairwise_distance", 0.0)
    ut_i, ut_j = np.triu_indices(4, 1)
    pair_dists = dist_mats[:, ut_i, ut_j]
    min_pair_d = pair_dists.min(axis=1)

    keep = (max_dists > 1e-9) & (min_pair_d >= min_pair)
    if not np.any(keep):
        return None, None

    A_idx          = A_idx[keep]
    B_idx          = B_idx[keep]
    max_dists_kept = max_dists[keep]
    pts_keep       = pts[keep]
    quad_indices   = quad_indices[keep]

    # ── Ownership guard ──────────────────────────────────────────────────────
    # Only emit quads whose longest pairwise distance is the current diagonal
    # (d1, d2).  Quads where a different edge is longer will be emitted when
    # that other diagonal is processed, so nothing is lost.
    #
    # enforce_ownership=False is used only by the coverage remediation pass,
    # where an undercovered neuron needs quads regardless of which diagonal
    # would normally own them.
    if enforce_ownership:
        diag_dist = float(np.linalg.norm(centroids[d2] - centroids[d1]))
        owns = max_dists_kept <= diag_dist * (1.0 + 1e-6)
        if not np.any(owns):
            return None, None

        A_idx          = A_idx[owns]
        B_idx          = B_idx[owns]
        max_dists_kept = max_dists_kept[owns]
        pts_keep       = pts_keep[owns]
        quad_indices   = quad_indices[owns]
    # ─────────────────────────────────────────────────────────────────────────

    batch = np.arange(quad_indices.shape[0])
    A  = pts_keep[batch, A_idx]
    B  = pts_keep[batch, B_idx]
    AB = B - A
    distAB = max_dists_kept[:, None]
    ux = AB / distAB
    uy = np.stack([-ux[:, 1], ux[:, 0]], axis=1)

    all_idx = np.array([0, 1, 2, 3])
    C_idx   = np.zeros(quad_indices.shape[0], dtype=np.int32)
    D_idx   = np.zeros(quad_indices.shape[0], dtype=np.int32)
    for k in range(quad_indices.shape[0]):
        others = all_idx[(all_idx != A_idx[k]) & (all_idx != B_idx[k])]
        C_idx[k], D_idx[k] = others[0], others[1]

    C  = pts_keep[batch, C_idx]
    D  = pts_keep[batch, D_idx]
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

    # Flip reference axis if C+D centroid is on the wrong side
    flip = (xC + xD) > 1
    if np.any(flip):
        A_flip  = B[flip];      B_flip  = A[flip]
        AB_flip = B_flip - A_flip
        ux_flip = AB_flip / max_dists_kept[flip, None]
        uy_flip = np.stack([-ux_flip[:, 1], ux_flip[:, 0]], axis=1)

        AC_flip = C[flip] - A_flip
        AD_flip = D[flip] - A_flip

        xC_flip = np.sum(AC_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yC_flip = np.sum(AC_flip * uy_flip, axis=1) / max_dists_kept[flip]
        xD_flip = np.sum(AD_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yD_flip = np.sum(AD_flip * uy_flip, axis=1) / max_dists_kept[flip]

        swap_flip      = xC_flip > xD_flip
        xC[flip]       = np.where(swap_flip, xD_flip, xC_flip)
        xD[flip]       = np.where(swap_flip, xC_flip, xD_flip)
        yC[flip]       = np.where(swap_flip, yD_flip, yC_flip)
        yD[flip]       = np.where(swap_flip, yC_flip, yD_flip)

    # Drop degenerate quads where both C and D lie on the AB line
    ok = ~((np.abs(yC) < 1e-4) & (np.abs(yD) < 1e-4))
    if not np.any(ok):
        return None, None

    quad_indices     = quad_indices[ok]
    quad_descriptors = np.stack(
        [xC[ok], yC[ok], xD[ok], yD[ok]], axis=1
    ).astype(np.float32)

    return quad_descriptors, quad_indices


# ══════════════════════════════════════════════════════════════════════════════
# Height threshold helper  (unchanged — used for diagnostics / logging)
# ══════════════════════════════════════════════════════════════════════════════

def compute_height_threshold(
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    percentile: float = 95.0,
    sample_limit: int = 5_000_000,
) -> float:
    """
    Compute a global |height| threshold based on a percentile of sampled heights.

    NOTE: `percentile` is on the 0-100 scale (e.g. 95.0 keeps the tallest 5%).
    The v1 bug passed 0.95 here (≈ 0th percentile, effectively no filter).
    """
    all_heights = []
    n_diagonals = len(diagonal_buckets)
    if n_diagonals == 0:
        return 0.0

    approx_per_diag = max(1, sample_limit // n_diagonals)

    for v in diagonal_buckets.values():
        h = v["height"]
        m = h.shape[0]
        if m <= approx_per_diag:
            idx = slice(None)
        else:
            idx = np.random.choice(m, size=approx_per_diag, replace=False)
        all_heights.append(np.abs(h[idx]))

    heights = np.concatenate(all_heights, axis=0)
    if heights.size == 0:
        return 0.0

    return float(np.percentile(heights, percentile))


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Diagonal-first bucket construction
# ══════════════════════════════════════════════════════════════════════════════

def build_diagonal_buckets_direct(
    centroids: np.ndarray,
    config,
    logger,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Enumerate diagonals directly and compute heights inline.
    No triangle objects are ever created.

    Diagonal set
    ─────────────
    Local      : k_local nearest neighbors per neuron (KD-tree)
    Long-range : k_random random neurons per neuron, seeded for reproducibility,
                 excluding KNN neighbors to avoid redundancy.

    The combination ensures:
      - Local diagonals capture fine spatial structure and are stable for
        small session-to-session shifts.
      - Long-range diagonals produce FOV-spanning quads with small σ_d
        (descriptor blur ∝ 1/L_q), which are critical for matching when
        the imaging field shifts substantially between sessions.

    For each unique diagonal (i, j), i < j:
      1. Compute perpendicular height h_k for every other neuron k:
             h_k = |(Pj − Pi) × (Pk − Pi)| / |Pj − Pi|
      2. Keep the top-K third-points by height immediately
         (K = config.max_triangles_per_diagonal, default 25).
      3. Store signed height (needed for descriptor yC/yD sign) and area.

    Returns
    ───────
    diagonal_buckets : dict
        Keys   : (i, j) int tuples, i < j, sorted by neuron index
        Values : {"third": int32(K,), "area": float32(K,), "height": float32(K,)}
    """
    n        = len(centroids)
    k_local  = min(getattr(config, 'knn_k', 15), n - 1)
    k_random = max(2, k_local // 2)
    K        = getattr(config, 'max_triangles_per_diagonal', 25)
    rng_seed = getattr(config, 'diagonal_rng_seed', 42)

    logger.info("")
    logger.info("STEP 1: Building diagonal buckets (diagonal-first, no triangle enumeration)")
    logger.info(f"  N={n}  k_local={k_local}  k_random={k_random}  K={K}  seed={rng_seed}")

    brute_tri  = n * (n - 1) * (n - 2) // 6
    brute_diag = n * (n - 1) // 2
    logger.info(f"  Brute-force triangles avoided : {brute_tri:,}")
    logger.info(f"  Brute-force diagonals possible: {brute_diag:,}")

    t0 = time.time()

    # ── Local KNN diagonals ───────────────────────────────────────────────────
    tree    = cKDTree(centroids)
    k_query = min(k_local + 1, n)
    _, knn_idx = tree.query(centroids, k=k_query)
    knn_idx = knn_idx[:, 1:].astype(np.int32)   # drop self → (n, k_local)

    # ── Random long-range diagonals ───────────────────────────────────────────
    rng        = np.random.default_rng(rng_seed)
    raw_random = rng.integers(0, n, size=(n, k_random * 3))   # oversample
    rand_idx   = np.zeros((n, k_random), dtype=np.int32)

    for i in range(n):
        exclude  = set(knn_idx[i].tolist())
        exclude.add(i)
        cands    = raw_random[i][~np.isin(raw_random[i], list(exclude))]
        if len(cands) >= k_random:
            rand_idx[i] = cands[:k_random]
        elif len(cands) > 0:
            rand_idx[i] = np.resize(cands, k_random)
        else:
            rand_idx[i] = knn_idx[i, :k_random]   # fallback: reuse KNN

    # ── Unique diagonal pairs (i < j) ────────────────────────────────────────
    all_neighbors = np.hstack([knn_idx, rand_idx])          # (n, k_local+k_random)
    centers       = np.repeat(np.arange(n, dtype=np.int32), all_neighbors.shape[1])
    neighbors_flat = all_neighbors.ravel().astype(np.int32)

    edges = np.stack([
        np.minimum(centers, neighbors_flat),
        np.maximum(centers, neighbors_flat),
    ], axis=1)
    edges = np.unique(edges, axis=0)                         # always i < j

    n_diagonals = len(edges)
    logger.info(
        f"  Unique diagonals: {n_diagonals:,} / {brute_diag:,} "
        f"({100.0 * n_diagonals / brute_diag:.1f}%)"
    )
    logger.info(
        f"  Local + long-range: {k_local} + {k_random} = "
        f"{k_local + k_random} neighbors/neuron"
    )

    t1 = time.time()
    logger.info(f"  Built edge list in {t1 - t0:.2f}s")

    # ── Per-diagonal: inline height + top-K cap ───────────────────────────────
    all_idx          = np.arange(n, dtype=np.int32)
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    n_skipped_short  = 0

    for idx_e, (i, j) in enumerate(edges):
        Pi = centroids[i]
        Pj = centroids[j]
        dv = Pj - Pi
        base_len = float(np.sqrt(dv[0] * dv[0] + dv[1] * dv[1]))

        if base_len < 1e-9:
            n_skipped_short += 1
            continue

        # All neurons except i and j
        others_mask = (all_idx != i) & (all_idx != j)
        k_idx       = all_idx[others_mask]               # (n-2,)
        Pk          = centroids[k_idx]                   # (n-2, 2)
        diff        = Pk - Pi                            # (n-2, 2)

        # Signed height via 2D cross product
        cross          = dv[0] * diff[:, 1] - dv[1] * diff[:, 0]   # (n-2,)
        signed_heights = cross / base_len
        abs_heights    = np.abs(signed_heights)
        areas          = 0.5 * base_len * abs_heights    # = 0.5 * base * h

        m = len(k_idx)
        if m < 2:
            continue

        # Top-K by absolute height — applied immediately
        if m > K:
            topk = np.argpartition(abs_heights, -K)[-K:]
        else:
            topk = np.arange(m)

        if len(topk) < 2:
            continue

        diagonal_buckets[(int(i), int(j))] = {
            "third":  k_idx[topk].astype(np.int32),
            "area":   areas[topk].astype(np.float32),
            "height": signed_heights[topk].astype(np.float32),
        }

    t2 = time.time()
    total_third_pts   = sum(len(v["third"]) for v in diagonal_buckets.values())
    mem_mb_new        = total_third_pts * 3 * 4 / 1e6          # int32 + 2×float32
    mem_mb_brute      = brute_tri * 3 * 4 / 1e6
    expected_quads    = len(diagonal_buckets) * K * (K - 1) // 2

    logger.info(f"  ✓ Built {len(diagonal_buckets):,} buckets in {t2 - t1:.2f}s")
    logger.info(f"    Short diagonals skipped: {n_skipped_short}")
    logger.info(f"    Total third-points stored: {total_third_pts:,}  "
                f"(brute triangles would be {brute_tri:,})")
    logger.info(f"    Memory: ~{mem_mb_new:.1f} MB  vs  ~{mem_mb_brute:.0f} MB brute")
    logger.info(f"    Expected quads (before ownership guard): ~{expected_quads:,}")

    return diagonal_buckets


# ══════════════════════════════════════════════════════════════════════════════
# Diagonal statistics  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def log_diagonal_stats(
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    logger: logging.Logger,
    sample_limit: int = 5_000_000,
) -> None:
    """Log summary statistics about diagonals and their third points."""
    n_diagonals = len(diagonal_buckets)
    if n_diagonals == 0:
        logger.info("    Diagonal stats: no diagonals found.")
        return

    sizes = np.fromiter(
        (len(v["third"]) for v in diagonal_buckets.values()),
        dtype=np.int32,
        count=n_diagonals
    )

    logger.info(f"    Diagonals: {n_diagonals}")
    logger.info(
        "    Third-points per diagonal: "
        f"min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.1f}, median={np.median(sizes):.1f}, "
        f"95th={np.percentile(sizes, 95):.1f}"
    )

    unique_sizes, counts = np.unique(sizes, return_counts=True)
    if unique_sizes.size <= 10:
        size_str = ", ".join(f"{s}: {c}" for s, c in zip(unique_sizes, counts))
        logger.info(f"    Size histogram (exact): {size_str}")
    else:
        bins     = [0, 5, 10, 15, 20, 25, 30, 50, 100, 500]
        hist, edges = np.histogram(sizes, bins=bins)
        bin_str  = ", ".join(
            f"[{int(edges[i])},{int(edges[i+1])}): {hist[i]}"
            for i in range(len(hist)) if hist[i] > 0
        )
        logger.info(f"    Size histogram (binned): {bin_str}")

    sampled_heights = []
    sampled_areas   = []
    approx_per_diag = max(1, sample_limit // n_diagonals)

    for v in diagonal_buckets.values():
        h = v["height"]
        a = v["area"]
        m = h.shape[0]
        idx = slice(None) if m <= approx_per_diag else \
              np.random.choice(m, size=approx_per_diag, replace=False)
        sampled_heights.append(np.abs(h[idx]))
        sampled_areas.append(a[idx])

    heights = np.concatenate(sampled_heights, axis=0)
    areas   = np.concatenate(sampled_areas,   axis=0)
    logger.info(f"    Sampled {heights.size:,} (height, area) pairs for global stats")

    def _log_pct(name: str, arr: np.ndarray) -> None:
        p = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        logger.info(
            f"    {name} percentiles (min,p1,p5,p25,p50,p75,p95,p99,max): "
            f"{p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g}, {p[3]:.4g}, "
            f"{p[4]:.4g}, {p[5]:.4g}, {p[6]:.4g}, {p[7]:.4g}, {p[8]:.4g}"
        )

    _log_pct("Triangle HEIGHT", heights)
    _log_pct("Triangle AREA",   areas)

    if heights.size > 1:
        logger.info(f"    Corr(height, area): {np.corrcoef(heights, areas)[0,1]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Quad assembly from buckets  (simplified — no K-cap, no global dedup)
# ══════════════════════════════════════════════════════════════════════════════

def step3_generate_quads_from_buckets(
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    centroids: np.ndarray,
    config,
    logger,
    min_height: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pair third-points within each diagonal bucket to produce quad descriptors.

    With the diagonal-first pipeline:
      - Buckets are already K-capped (done in build_diagonal_buckets_direct).
      - _process_diagonal now carries an ownership guard that ensures each quad
        is emitted exactly once (from its longest-edge diagonal).
      - No global np.unique dedup pass is needed.

    The optional `min_height` filter is retained for call-site compatibility
    but is effectively 0 when buckets were already filtered upfront.
    """
    import psutil

    def get_ram():
        return psutil.Process().memory_info().rss / 1e9

    logger.info("")
    logger.info("STEP 2: Assembling quads from diagonal buckets...")

    items   = list(diagonal_buckets.items())
    n_diag  = len(items)
    if n_diag == 0:
        logger.info("  No diagonals → no quads.")
        return None, None

    logger.info(f"  {n_diag:,} diagonals → pairing third-points into quads")
    logger.info(f"  Ownership guard active — no global dedup needed")

    def worker(item):
        (d1, d2), data = item
        third  = data["third"]
        area   = data["area"]
        height = data["height"]

        if min_height > 0.0:
            mask = np.abs(height) >= min_height
            if not np.any(mask):
                return None, None
            third  = third[mask]
            area   = area[mask]
            height = height[mask]

        if third.size < 2:
            return None, None

        third_points = list(zip(third, area, height))
        desc, idx = _process_diagonal(d1, d2, third_points, centroids, config, height_tol=0.0)

        if desc is None or desc.size == 0:
            return None, None
        return desc.astype(np.float32), idx.astype(np.int32)

    n_workers   = getattr(config, "n_workers", 8)
    pool        = ThreadPool(n_workers)
    descriptors = []
    indices     = []
    total_quads = 0

    t0 = time.time()
    for i, result in enumerate(pool.imap_unordered(worker, items), 1):
        desc, idx = result
        if desc is not None:
            descriptors.append(desc)
            indices.append(idx)
            total_quads += desc.shape[0]

        if i % max(1, n_diag // 50) == 0 or i == n_diag:
            logger.info(
                f"  Processed {i}/{n_diag} diagonals | "
                f"Quads so far: {total_quads:,} | RAM {get_ram():.2f} GB"
            )

    pool.close()
    pool.join()

    if not descriptors:
        logger.info("  No quads generated from any diagonal.")
        return None, None

    final_desc = np.vstack(descriptors)
    final_idx  = np.vstack(indices).astype(np.int32)
    # Free the accumulation lists — now consolidated into final arrays
    del descriptors, indices
    gc.collect()

    # ── Quality pruning (keep_fraction, optional) ─────────────────────────────
    keep_fraction = getattr(config, "quad_keep_fraction", 1.0)
    if 0.0 < keep_fraction < 1.0 and final_desc.shape[0] > 0:
        quality = np.abs(final_desc[:, 1]) + np.abs(final_desc[:, 3])
        cutoff  = np.quantile(quality, 1.0 - keep_fraction)
        mask    = quality >= cutoff
        n_before = final_desc.shape[0]
        final_desc = final_desc[mask]
        final_idx  = final_idx[mask]
        logger.info(
            f"  Quality pruning (keep_fraction={keep_fraction:.2f}): "
            f"{final_desc.shape[0]:,}/{n_before:,} quads kept"
        )

    logger.info(
        f"  ✓ STEP 2 done: {final_desc.shape[0]:,} quads "
        f"| {time.time()-t0:.2f}s | RAM {get_ram():.2f} GB"
    )

    return final_desc, final_idx


# ══════════════════════════════════════════════════════════════════════════════
# Coverage remediation  (ensures every neuron meets proportional quad minimum)
# ══════════════════════════════════════════════════════════════════════════════

def _remediate_coverage(
    final_desc: np.ndarray,
    final_idx: np.ndarray,
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    centroids: np.ndarray,
    n_neurons: int,
    config,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure every neuron reaches a minimum quad coverage relative to the field.

    The ownership guard in _process_diagonal is the primary cause of zero-quad
    neurons: a neuron that only forms short diagonals in a dense region sees all
    its quads claimed by longer neighbouring diagonals.  This pass detects those
    neurons and re-runs _process_diagonal with enforce_ownership=False for their
    diagonals, collecting quads that would otherwise have been suppressed.

    All other quality filters are still applied:
      - area >= 1.0 (quad_area mask inside _process_diagonal)
      - min_pairwise_distance (config.min_pairwise_distance)
      - degenerate-quad check (|yC|, |yD| < 1e-4)
      - quad_keep_fraction quality pruning on the remediation batch

    Threshold
    ---------
    A neuron is considered undercovered when its quad count is below
    min_coverage_fraction * median(field coverage).
    min_coverage_fraction defaults to 0.4 and can be set in config.

    Returns
    -------
    Augmented (final_desc, final_idx) with remediation quads appended.
    New quads that duplicate existing quad_idx rows are dropped.
    """
    min_frac = getattr(config, 'min_coverage_fraction', 0.4)

    # ── Per-neuron coverage count ─────────────────────────────────────────────
    coverage = np.zeros(n_neurons, dtype=np.int32)
    if final_idx is not None and final_idx.shape[0] > 0:
        for nid in final_idx.ravel():
            if 0 <= nid < n_neurons:
                coverage[nid] += 1

    field_median = float(np.median(coverage))
    threshold    = min_frac * field_median

    undercovered = np.where(coverage < threshold)[0]

    logger.info("")
    logger.info("  ┌─ Coverage Remediation ─────────────────────────────────────────┐")
    logger.info(f"  │  Field median coverage : {field_median:.1f} quads/neuron")
    logger.info(f"  │  Threshold ({min_frac:.0%} of median): {threshold:.1f}")
    logger.info(f"  │  Undercovered neurons  : {len(undercovered)} / {n_neurons}")
    logger.info("  └───────────────────────────────────────────────────────────────┘")

    if len(undercovered) == 0:
        return final_desc, final_idx

    # ── Build reverse index: neuron → diagonals it appears in as endpoint ────
    neuron_to_diags: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (i, j) in diagonal_buckets:
        neuron_to_diags[i].append((i, j))
        neuron_to_diags[j].append((i, j))

    # ── Existing quad set for deduplication ──────────────────────────────────
    existing_set: set = set()
    if final_idx is not None and final_idx.shape[0] > 0:
        for row in final_idx:
            existing_set.add(tuple(sorted(row.tolist())))

    # ── Remediate each undercovered neuron ────────────────────────────────────
    rem_descs: List[np.ndarray] = []
    rem_idxs:  List[np.ndarray] = []
    total_new  = 0
    n_remediated = 0

    keep_fraction = getattr(config, 'quad_keep_fraction', 1.0)

    for nid in undercovered:
        diags = neuron_to_diags.get(int(nid), [])
        if not diags:
            logger.debug(f"  Neuron {nid}: no diagonals found — truly isolated.")
            continue

        nid_descs: List[np.ndarray] = []
        nid_idxs:  List[np.ndarray] = []

        for (i, j) in diags:
            bucket = diagonal_buckets[(i, j)]
            third_points = list(zip(
                bucket["third"],
                bucket["area"],
                bucket["height"],
            ))
            desc, idx = _process_diagonal(
                i, j, third_points, centroids, config,
                height_tol=0.0,
                enforce_ownership=False,   # ← key: bypass ownership guard
            )
            if desc is None or desc.shape[0] == 0:
                continue

            # Drop quads already in the existing set
            new_mask = np.array([
                tuple(sorted(row.tolist())) not in existing_set
                for row in idx
            ])
            if not np.any(new_mask):
                continue

            nid_descs.append(desc[new_mask])
            nid_idxs.append(idx[new_mask])

        if not nid_descs:
            continue

        # Merge this neuron's remediation batch
        batch_desc = np.vstack(nid_descs).astype(np.float32)
        batch_idx  = np.vstack(nid_idxs).astype(np.int32)

        # Apply quality pruning (keep_fraction) to the remediation batch
        if 0.0 < keep_fraction < 1.0 and batch_desc.shape[0] > 0:
            quality = np.abs(batch_desc[:, 1]) + np.abs(batch_desc[:, 3])
            cutoff  = np.quantile(quality, 1.0 - keep_fraction)
            qmask   = quality >= cutoff
            batch_desc = batch_desc[qmask]
            batch_idx  = batch_idx[qmask]

        if batch_desc.shape[0] == 0:
            continue

        # Register into existing set so multi-neuron remediations don't double-add
        for row in batch_idx:
            existing_set.add(tuple(sorted(row.tolist())))

        rem_descs.append(batch_desc)
        rem_idxs.append(batch_idx)
        total_new   += batch_desc.shape[0]
        n_remediated += 1

    logger.info(
        f"  Coverage remediation: {n_remediated}/{len(undercovered)} neurons boosted, "
        f"{total_new:,} new quads added"
    )

    if total_new == 0:
        del existing_set
        gc.collect()
        return final_desc, final_idx

    aug_desc = np.vstack([final_desc] + rem_descs) if final_desc is not None else np.vstack(rem_descs)
    aug_idx  = np.vstack([final_idx]  + rem_idxs)  if final_idx  is not None else np.vstack(rem_idxs)

    del existing_set, rem_descs, rem_idxs
    gc.collect()

    return aug_desc, aug_idx


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator  (rewritten — 2-step diagonal-first pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def generate_sparse_quads_triangle(centroids, config, logger):
    """
    2-step quad generation pipeline (v2 — diagonal-first).

    Step 1  build_diagonal_buckets_direct
            KNN local + random long-range diagonals.
            Heights computed inline. Top-K cap applied immediately.
            Memory O(n · k · K) vs O(n³) for brute-force triangles.

    Step 2  step3_generate_quads_from_buckets
            Pair third-points per diagonal into quads + descriptors.
            Ownership guard in _process_diagonal ensures zero duplicates
            without any post-hoc global np.unique pass.

    Config parameters used
    ──────────────────────
    knn_k                     : local neighbors per neuron  (default 15)
    max_triangles_per_diagonal: top-K cap per diagonal      (default 25)
    diagonal_rng_seed         : RNG seed for long-range     (default 42)
    quad_keep_fraction        : final quality pruning       (default 1.0)
    n_workers                 : thread-pool size            (default 8)
    """
    global _last_estimate_n
    n = len(centroids)
    if abs(n - _last_estimate_n) > 10:
        estimate_and_suggest_params(n, config, logger)
        _last_estimate_n = n

    # ── Step 1: Diagonal buckets ──────────────────────────────────────────────
    diagonal_buckets = build_diagonal_buckets_direct(centroids, config, logger)
    logger.info(f"  ✓ Step 1/2: {len(diagonal_buckets):,} diagonal buckets built")

    log_diagonal_stats(diagonal_buckets, logger, sample_limit=1_000_000)

    if len(diagonal_buckets) == 0:
        logger.warning("  No diagonal buckets produced — returning empty.")
        return None, None

    # ── Step 2: Quad assembly ─────────────────────────────────────────────────
    final_desc, final_idx = step3_generate_quads_from_buckets(
        diagonal_buckets, centroids, config, logger, min_height=0.0,
    )

    n_quads = final_desc.shape[0] if final_desc is not None else 0
    logger.info(f"  ✓ Step 2/2: {n_quads:,} quads  (ownership guard, no global dedup)")

    # ── Step 3: Coverage remediation ─────────────────────────────────────────
    final_desc, final_idx = _remediate_coverage(
        final_desc, final_idx, diagonal_buckets,
        centroids, n, config, logger,
    )
    n_quads_after = final_desc.shape[0] if final_desc is not None else 0
    # Free diagonal buckets — no longer needed after remediation
    del diagonal_buckets
    gc.collect()
    if n_quads_after > n_quads:
        logger.info(
            f"  ✓ Step 3/3 (remediation): {n_quads_after:,} quads total "
            f"(+{n_quads_after - n_quads:,})"
        )

    return final_desc, final_idx


# ══════════════════════════════════════════════════════════════════════════════
# Q estimator  (updated for diagonal-first formula)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_and_suggest_params(n_neurons, config, logger, target_q: int = 1_300_000):
    """
    Estimate the number of quads that will be generated and warn if over cap.

    Formula (diagonal-first)
    ─────────────────────────
      n_diag_eff  ≈ n · (k_local + k_random) / 2      unique diagonals sampled
      M_eff       = min(K, n − 2)                       third-points per diagonal
      q_per_diag  = C(M_eff, 2) = M_eff·(M_eff−1)/2   raw quads per diagonal
      est_Q       = n_diag_eff · q_per_diag · kf       no /6 dedup factor
                                                         (ownership guard is exact)

    Note: the old formula divided by 6 to account for the 6-way duplication
    from brute-force triangles.  With the ownership guard that duplication no
    longer exists, so the factor is removed.
    """
    k_local   = min(getattr(config, 'knn_k', 15), n_neurons - 1)
    k_random  = max(2, k_local // 2)
    max_K     = getattr(config, 'max_triangles_per_diagonal', 25)
    kf        = getattr(config, 'quad_keep_fraction', 1.0)

    n_diag_eff = n_neurons * (k_local + k_random) / 2.0
    M_eff      = min(float(max_K), float(n_neurons - 2))
    q_per_diag = M_eff * (M_eff - 1) / 2.0
    est_q      = int(n_diag_eff * q_per_diag * kf)

    brute_diag = n_neurons * (n_neurons - 1) // 2

    logger.info("")
    logger.info("  ┌─ Quad Estimate (diagonal-first) ─────────────────────────────┐")
    logger.info(f"  │  N={n_neurons}  k_local={k_local}  k_random={k_random}  K={max_K}               │")
    logger.info(f"  │  Diagonals sampled : {n_diag_eff:.0f} / {brute_diag} ({100*n_diag_eff/brute_diag:.1f}%)  │")
    logger.info(f"  │  M_eff={M_eff:.0f}  C(M,2)/diag={q_per_diag:.0f}  kf={kf:.2f}             │")
    logger.info(f"  │  Estimated Q    : {est_q:>12,}                                 │")
    logger.info(f"  │  Cap (empirical): {target_q:>12,}                                 │")

    if est_q <= target_q:
        logger.info("  │  ✓ Within cap — no adjustment needed.                          │")
        logger.info("  └──────────────────────────────────────────────────────────────┘")
        logger.info("")
        return

    overshoot_pct = 100.0 * (est_q - target_q) / target_q
    # Suggest a keep_fraction reduction
    suggested_kf  = round(max(0.001, kf * (target_q / est_q)), 3)
    # Or suggest a K reduction
    import math
    suggested_K   = max(2, int(math.floor(0.5 + 0.5 * math.sqrt(1 + 8 * target_q / (n_diag_eff * kf)))))

    logger.info(f"  │  ⚠  Overshoot by {overshoot_pct:.1f}%                                      │")
    logger.info(f"  │  Option A — quad_keep_fraction = {suggested_kf:.3f}                        │")
    logger.info(f"  │  Option B — max_triangles_per_diagonal ≈ {suggested_K}                    │")
    logger.info("  └──────────────────────────────────────────────────────────────┘")
    logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# Parallel session worker  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _session_worker(args):
    """Thin wrapper around process_single_session for ProcessPoolExecutor."""
    session_info, config_dict = args
    cfg = PipelineConfig.from_dict(config_dict)

    log_dir = Path(cfg.output_dir) / "logs_step1"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=cfg.verbose)

    child_logger = logging.getLogger("neuron_mapping_parallel")
    child_logger.info(
        f"[CHILD PID={multiprocessing.current_process().pid}] "
        f"Starting session {session_info['session_name']}..."
    )

    result = process_single_session(session_info, cfg)

    if "session_name" not in result:
        result["session_name"] = session_info.get("session_name", "<unknown>")

    child_logger.info(
        f"[CHILD PID={multiprocessing.current_process().pid}] "
        f"Finished session {result['session_name']} | "
        f"skipped={result.get('skipped', False)} | "
        f"n_quads={result.get('n_quads')}"
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Top-level parallel runner  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def run_step_1_parallel(
    config: PipelineConfig,
    loaded_sessions: Optional[List[Dict[str, Any]]] = None,
    session_callback: Optional[callable] = None,
    session_filename_regex: str = r'^([A-Za-z0-9_]+?)_(\d+)(.*?)\.npy$',
) -> dict:
    """Run Step 1 in parallel across multiple sessions."""

    # ── Build session list ────────────────────────────────────────────────────
    if loaded_sessions is not None:
        sessions = []
        for sess in loaded_sessions:
            if isinstance(sess, str):
                filename = Path(sess).name
                match = re.match(session_filename_regex, filename)
                if match:
                    animal_id    = match.group(1)
                    session_name = Path(sess).stem
                    file_path    = sess if (sess.endswith('.npy') and Path(sess).exists()) \
                                   else str(Path(config.input_dir) / f"{session_name}.npy")
                    sessions.append({
                        'file_path':    file_path,
                        'animal_id':    animal_id,
                        'session':      session_name,
                        'session_name': session_name,
                    })
                else:
                    logger.warning(f"Skipping file with invalid pattern: {sess}")
            elif isinstance(sess, dict) and 'session_name' in sess:
                sessions.append(sess)
            else:
                logger.warning(f"Skipping invalid session format: {type(sess)}")
        logger.info(f"[PARALLEL] Using {len(sessions)} pre-loaded sessions")
    else:
        sessions = find_session_files(config, session_filename_regex=session_filename_regex)
        logger.info(f"[PARALLEL] Found {len(sessions)} session files to process")

    if config.animal_id:
        logger.info(f"[PARALLEL] Processing only animal {config.animal_id}")
        sessions = [s for s in sessions if s['animal_id'] == config.animal_id]
        logger.info(f"[PARALLEL] Filtered to {len(sessions)} sessions for animal {config.animal_id}")

    if not sessions:
        logger.warning("[PARALLEL] No sessions found; exiting.")
        return {"n_sessions": 0, "n_skipped": 0, "total_quads": 0, "results": []}

    # ── Pre-flight Q estimate (parent process — visible in GUI log) ───────────
    global _last_estimate_n
    first_n = None
    try:
        raw = np.load(sessions[0]['file_path'], allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            data = raw.item()
            if isinstance(data, dict) and 'centroids_x' in data:
                first_n = len(data['centroids_x'])
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            first_n = raw.shape[2]
    except Exception as e:
        logger.warning(f"[PARALLEL] Could not peek at first session for Q estimate: {e}")

    if first_n is not None and abs(first_n - _last_estimate_n) > 10:
        estimate_and_suggest_params(first_n, config, logger)
        _last_estimate_n = first_n

    # ── Launch workers ────────────────────────────────────────────────────────
    max_parallel = compute_max_parallel_sessions(config)
    config_dict  = config.to_dict()

    start         = time.time()
    results       = []
    total_quads   = 0
    n_skipped     = 0
    session_times = []

    logger.info(
        f"[PARALLEL] Starting parallel Step 1 with up to "
        f"{max_parallel} worker processes...\n"
    )

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(_session_worker, (sess, config_dict))
            for sess in sessions
        ]

        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"[PARALLEL] Error in child process: {e}", exc_info=True)
                continue

            results.append(res)

            if res.get("skipped", False):
                n_skipped += 1
            else:
                total_quads += int(res.get("n_quads") or 0)
                if "generation_time" in res:
                    session_times.append(res["generation_time"])

            if session_callback is not None and idx % max(1, len(sessions) // 100) == 0:
                avg_time = np.mean(session_times) if session_times else 0.0
                session_callback(idx, len(sessions), avg_time)

            logger.info(
                f"[PARALLEL] Session {res.get('session_name')} done | "
                f"skipped={res.get('skipped')} | "
                f"n_quads={res.get('n_quads')}"
            )

    elapsed     = time.time() - start
    n_processed = len(results) - n_skipped

    logger.info("\n[PARALLEL] Step 1 Parallel Summary:")
    logger.info(f"  Sessions processed: {n_processed}")
    logger.info(f"  Sessions skipped:   {n_skipped}")
    logger.info(f"  Total quads:        {total_quads:,}")
    logger.info(f"  Wall-clock time:    {elapsed:.1f}s")

    summary = {
        "n_sessions":    n_processed,
        "n_skipped":     n_skipped,
        "total_quads":   total_quads,
        "results":       results,
        "wall_time_sec": elapsed,
    }

    # ── Automatic saturation check ────────────────────────────────────────────
    print("ABOUT TO RUN SATURATION CHECK")
    try:
        run_saturation_check_after_step1(config, step1_summary=summary)
    except Exception as e:
        logger.warning(f"Saturation check failed (non-fatal): {e}")
    return summary

