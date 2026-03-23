"""
Step 3: Hungarian Cost Sweep + Consolidated Neuron Tracking

V3: RANSAC-informed matching with vote normalization, asymmetric dummy
    padding, post-filter, and second-pass recovery.

Changes from baseline
─────────────────────
Only TWO functions were modified:

  build_cost_matrix_from_filtered_quads  →  now uses normalized votes,
      RANSAC-transformed distances, and returns both the cost matrix and
      the distance matrix for downstream post-filtering.

  process_session_pair_sweep  →  replaces the cost-threshold sweep with
      padded Hungarian (asymmetric dummies) + transformed-distance
      post-filter + second-pass recovery on unmatched neurons.
      No sweep parameters needed — the dummy costs self-calibrate.

Everything below the "UNCHANGED" marker is identical to the original.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from utilities import *
logger = logging.getLogger("neuron_mapping_consolidated")


# ── Internal configuration ────────────────────────────────────────────────────
# When True, dummy costs are scaled per-neuron by proximity (closer neurons
# get higher dummy costs, making them harder to leave unmatched).
# When False, every neuron gets the same uniform dummy cost (global_median_cost).
_USE_ASYMMETRIC_DUMMY_COSTS = False

# When True, neuron pairs with zero quad votes are completely blocked (cost=inf)
# and can never be matched.  When False, zero-vote pairs within the distance
# cutoff get a distance-only fallback cost (penalized relative to voted pairs,
# but still reachable by Hungarian).
_BLOCK_ZERO_VOTE_PAIRS = False


# ==============================================================================
# CHANGED: Cost Matrix Construction (RANSAC-informed, normalized votes)
# ==============================================================================

def build_cost_matrix_from_filtered_quads(
    ref_centroids: np.ndarray,
    tgt_centroids: np.ndarray,
    match_indices: np.ndarray,
    transform_matrix: np.ndarray,
    transform_translation: np.ndarray,
    ransac_max_residual: float = 5.0,
    use_quad_voting: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build neuron-to-neuron cost matrix combining normalized quad votes
    with RANSAC-transformed spatial distance.

    Changes from baseline:
      - Votes normalized by sqrt(ref_degree × tgt_degree) so undercovered
        neurons aren't systematically outbid by high-degree ones.
      - RANSAC transform projects ref centroids into tgt space; pairwise
        distance is added to the cost with auto-derived weight.
      - Pairs with zero votes OR transformed distance > 3× ransac_max_residual
        are blocked (set to inf).

    Returns:
        cost_matrix:  (N_ref, N_tgt) combined costs for Hungarian
        dist_matrix:  (N_ref, N_tgt) transformed distances for post-filtering
    """
    n_ref = len(ref_centroids)
    n_tgt = len(tgt_centroids)

    # ── Transformed spatial distances ─────────────────────────────────────
    A = np.array(transform_matrix, dtype=np.float64)
    t = np.array(transform_translation, dtype=np.float64)
    ref_transformed = (A @ ref_centroids.T).T + t
    diff = ref_transformed[:, None, :] - tgt_centroids[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    if not use_quad_voting:
        # Fallback: pure distance cost (unchanged from baseline)
        cost_matrix = np.full((n_ref, n_tgt), 1e6, dtype=np.float32)
        for quad_match in match_indices:
            ref_neurons = quad_match[:4].astype(int)
            tgt_neurons = quad_match[4:].astype(int)
            for i_ref in ref_neurons:
                for i_tgt in tgt_neurons:
                    d = float(dist_matrix[i_ref, i_tgt])
                    cost_matrix[i_ref, i_tgt] = min(cost_matrix[i_ref, i_tgt], d)
        return cost_matrix, dist_matrix

    # ── Normalized quad voting ────────────────────────────────────────────
    vote_matrix = np.zeros((n_ref, n_tgt), dtype=np.float64)
    ref_degree = np.zeros(n_ref, dtype=np.float64)
    tgt_degree = np.zeros(n_tgt, dtype=np.float64)

    for qm in match_indices:
        ref_ns = qm[:4].astype(int)
        tgt_ns = qm[4:].astype(int)
        for ri in ref_ns:
            if 0 <= ri < n_ref:
                ref_degree[ri] += 1
        for ti in tgt_ns:
            if 0 <= ti < n_tgt:
                tgt_degree[ti] += 1
        for ri in ref_ns:
            for ti in tgt_ns:
                if 0 <= ri < n_ref and 0 <= ti < n_tgt:
                    vote_matrix[ri, ti] += 1

    ref_degree = np.maximum(ref_degree, 1.0)
    tgt_degree = np.maximum(tgt_degree, 1.0)
    norm_factor = np.sqrt(ref_degree[:, None] * tgt_degree[None, :])
    vote_norm = vote_matrix / norm_factor

    max_vnorm = vote_norm.max() if vote_norm.max() > 0 else 1.0
    vote_cost = (max_vnorm - vote_norm).astype(np.float64)

    # Block: too far apart (always enforced)
    dist_cutoff = 3.0 * ransac_max_residual
    vote_cost[dist_matrix > dist_cutoff] = np.inf

    if _BLOCK_ZERO_VOTE_PAIRS:
        # Hard block: zero-vote pairs cannot be matched
        vote_cost[vote_matrix == 0] = np.inf

    # ── Combine vote cost + distance with auto-weight ─────────────────────
    voted_mask = np.isfinite(vote_cost) & (vote_matrix > 0)
    if voted_mask.sum() == 0:
        return vote_cost.astype(np.float32), dist_matrix

    median_vote_cost = np.median(vote_cost[voted_mask])
    median_dist = max(np.median(dist_matrix[voted_mask]), 1e-9)
    dist_weight = median_vote_cost / median_dist

    combined_cost = np.where(voted_mask,
                             vote_cost + dist_weight * dist_matrix,
                             np.inf)

    if not _BLOCK_ZERO_VOTE_PAIRS:
        # Fallback: zero-vote pairs within distance cutoff get a
        # distance-only cost with a penalty so voted pairs are always
        # preferred, but these remain reachable by Hungarian.
        zero_vote_nearby = (vote_matrix == 0) & (dist_matrix <= dist_cutoff)
        if zero_vote_nearby.any():
            # Penalty = max finite voted cost, so any voted pair wins over
            # a distance-only fallback at the same distance.
            finite_voted = combined_cost[voted_mask]
            vote_penalty = float(np.max(finite_voted)) if len(finite_voted) > 0 else max_vnorm
            combined_cost[zero_vote_nearby] = vote_penalty + dist_weight * dist_matrix[zero_vote_nearby]
            n_fallback = int(zero_vote_nearby.sum())
            logger.info(f"    Zero-vote fallback: {n_fallback} pairs given distance-only cost")

    logger.info(f"    Built cost matrix: {n_ref} × {n_tgt} "
                f"(dist_weight={dist_weight:.2f}, cutoff={dist_cutoff:.1f}px)")

    return combined_cost, dist_matrix


# ==============================================================================
# CHANGED: Session Pair Processing (padded Hungarian + post-filter + 2nd pass)
# ==============================================================================

def process_session_pair_sweep(
    filter_file: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Process one session pair with RANSAC-informed matching.

    Changes from baseline:
      - Cost matrix uses normalized votes + transformed distance
        (via updated build_cost_matrix_from_filtered_quads).
      - Hungarian runs on a dummy-padded matrix so neurons can go
        unmatched instead of being forced into bad pairings.
      - Dummy costs are asymmetric (when _USE_ASYMMETRIC_DUMMY_COSTS is
        True): based on each neuron's min transformed distance (isolated
        neurons go unmatched cheaply).  When False, all dummies are
        uniform at global_median_cost.
      - Post-filter cuts any match where transformed distance exceeds
        ransac_max_residual (reuses existing Step 2.5 parameter).
      - Second pass re-runs a smaller Hungarian on unmatched neurons
        with a relaxed distance cutoff (2× ransac_max_residual) and
        relaxed dummy cost (75th percentile) to recover FN_avoidable.
      - The cost-threshold sweep is no longer needed — dummy padding
        self-calibrates. The hungarian_cost_values parameter is accepted
        for interface compatibility but not used for threshold selection.

    Returns:
        Same dict structure as baseline for full pipeline compatibility.
    """
    try:
        data = np.load(filter_file, allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load {filter_file}: {e}")
        return None

    # ── Metadata ──────────────────────────────────────────────────────────
    animal_id = decode_string_field(data.get('animal_id', ''))
    if not animal_id:
        animal_id = filter_file.stem.split('_')[0]

    pair_name = decode_string_field(data.get('pair_name', ''))
    if not pair_name:
        pair_name = filter_file.stem.replace('_filtered_matches', '')

    ref_session = decode_string_field(data.get('ref_session', ''))
    target_session = decode_string_field(data.get('target_session', ''))

    ref_centroids = data['ref_centroids']
    tgt_centroids = data['tgt_centroids']
    match_indices = data['match_indices']
    transform_matrix = data['transform_matrix']
    transform_translation = data['transform_translation']

    n_ref = len(ref_centroids)
    n_tgt = len(tgt_centroids)
    n_inlier_quads = len(match_indices)

    logger.info(f"  [{ref_session} → {target_session}] "
                f"{n_ref}→{n_tgt} neurons, {n_inlier_quads:,} inlier quads")

    if n_inlier_quads == 0:
        logger.warning(f"  No inlier matches for {pair_name}")
        return None

    # ── Retrieve RANSAC residual threshold from config or default ─────────
    ransac_max_residual = float(data.get('max_residual_threshold', 5.0))

    # ── Build RANSAC-informed cost matrix ─────────────────────────────────
    combined_cost, dist_matrix = build_cost_matrix_from_filtered_quads(
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        match_indices=match_indices,
        transform_matrix=transform_matrix,
        transform_translation=transform_translation,
        ransac_max_residual=ransac_max_residual,
        use_quad_voting=use_quad_voting,
    )

    dist_cutoff = 3.0 * ransac_max_residual

    # ── Dummy costs ───────────────────────────────────────────────────────
    finite_costs = combined_cost[np.isfinite(combined_cost)]
    if len(finite_costs) == 0:
        logger.warning(f"  No finite costs for {pair_name} — skipping")
        return None

    global_median_cost = float(np.median(finite_costs))

    if _USE_ASYMMETRIC_DUMMY_COSTS:
        # Per-neuron dummy costs scaled by proximity to nearest candidate
        ref_min_dist = dist_matrix.min(axis=1)
        tgt_min_dist = dist_matrix.min(axis=0)

        proximity_ref = np.clip(1.0 - ref_min_dist / dist_cutoff, 0.1, 1.0)
        ref_dummy_costs = global_median_cost * proximity_ref

        proximity_tgt = np.clip(1.0 - tgt_min_dist / dist_cutoff, 0.1, 1.0)
        tgt_dummy_costs = global_median_cost * proximity_tgt
    else:
        # Uniform dummy costs — every neuron pays the same price to go unmatched
        ref_dummy_costs = np.full(n_ref, global_median_cost)
        tgt_dummy_costs = np.full(n_tgt, global_median_cost)

    # ── Padded cost matrix ────────────────────────────────────────────────
    padded = np.full((n_ref + n_tgt, n_tgt + n_ref),
                     global_median_cost * 2, dtype=np.float64)

    padded[:n_ref, :n_tgt] = np.where(np.isfinite(combined_cost),
                                       combined_cost,
                                       global_median_cost * 10)

    for i in range(n_ref):
        padded[i, n_tgt:] = ref_dummy_costs[i]
    for j in range(n_tgt):
        padded[n_ref:, j] = tgt_dummy_costs[j]
    padded[n_ref:, n_tgt:] = 0.0

    # ── Pass 1: Hungarian on padded matrix ────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(padded)

    real_mask = (row_ind < n_ref) & (col_ind < n_tgt)
    pass1_ref = row_ind[real_mask]
    pass1_tgt = col_ind[real_mask]

    # ── Post-filter: cut matches with bad transformed distance ────────────
    pass1_dists = np.array([dist_matrix[r, c] for r, c in zip(pass1_ref, pass1_tgt)])
    keep = pass1_dists <= ransac_max_residual
    filtered_ref = pass1_ref[keep]
    filtered_tgt = pass1_tgt[keep]
    filtered_costs = np.array([combined_cost[r, c]
                               for r, c in zip(filtered_ref, filtered_tgt)])

    n_pass1 = len(pass1_ref)
    n_filtered = len(filtered_ref)
    logger.info(f"    Pass 1: {n_pass1} matches → {n_filtered} after post-filter")

    # ── Pass 2: recover unmatched neurons ─────────────────────────────────
    matched_ref_set = set(filtered_ref.tolist())
    matched_tgt_set = set(filtered_tgt.tolist())
    unmatched_ref = [i for i in range(n_ref) if i not in matched_ref_set]
    unmatched_tgt = [j for j in range(n_tgt) if j not in matched_tgt_set]

    n_recovered = 0
    if len(unmatched_ref) > 0 and len(unmatched_tgt) > 0:
        ur = np.array(unmatched_ref, dtype=np.int32)
        ut = np.array(unmatched_tgt, dtype=np.int32)
        n_ur, n_ut = len(ur), len(ut)

        sub_cost = combined_cost[np.ix_(ur, ut)].copy()
        sub_dist = dist_matrix[np.ix_(ur, ut)]
        relaxed_cutoff = 2.0 * ransac_max_residual
        sub_cost[sub_dist > relaxed_cutoff] = np.inf

        sub_finite = sub_cost[np.isfinite(sub_cost)]
        if len(sub_finite) > 0:
            sub_dummy = float(np.percentile(sub_finite, 75))

            sub_padded = np.full((n_ur + n_ut, n_ut + n_ur),
                                 sub_dummy * 2, dtype=np.float64)
            sub_padded[:n_ur, :n_ut] = np.where(np.isfinite(sub_cost),
                                                  sub_cost, sub_dummy * 10)
            sub_padded[:n_ur, n_ut:] = sub_dummy
            sub_padded[n_ur:, :n_ut] = sub_dummy
            sub_padded[n_ur:, n_ut:] = 0.0

            sr, sc = linear_sum_assignment(sub_padded)
            real2 = (sr < n_ur) & (sc < n_ut)
            p2_ref_local = sr[real2]
            p2_tgt_local = sc[real2]

            p2_dists = np.array([dist_matrix[ur[r], ut[c]]
                                 for r, c in zip(p2_ref_local, p2_tgt_local)])
            keep2 = p2_dists <= relaxed_cutoff
            p2_ref = ur[p2_ref_local[keep2]]
            p2_tgt = ut[p2_tgt_local[keep2]]
            p2_costs = np.array([combined_cost[r, c]
                                 for r, c in zip(p2_ref, p2_tgt)])

            if len(p2_ref) > 0:
                filtered_ref = np.concatenate([filtered_ref, p2_ref])
                filtered_tgt = np.concatenate([filtered_tgt, p2_tgt])
                filtered_costs = np.concatenate([filtered_costs, p2_costs])
                n_recovered = len(p2_ref)

    n_final = len(filtered_ref)
    logger.info(f"    Pass 2: +{n_recovered} recovered → {n_final} total matches")

    # ── Save (same format as baseline for pipeline compatibility) ─────────
    matched_ref_indices = filtered_ref.astype(np.int32)
    matched_tgt_indices = filtered_tgt.astype(np.int32)
    matched_costs_arr = filtered_costs.astype(np.float32)

    sweep_file = output_dir / f"{pair_name}_sweep.npz"
    np.savez_compressed(
        sweep_file,
        animal_id=animal_id,
        pair_name=pair_name,
        ref_session=ref_session,
        target_session=target_session,
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        n_ref_neurons=n_ref,
        n_target_neurons=n_tgt,
        n_inlier_quads=n_inlier_quads,
        # Sweep fields (kept for compatibility; single-value since no sweep)
        cost_thresholds=hungarian_cost_values,
        match_counts=np.array([n_final] * len(hungarian_cost_values), dtype=np.int32),
        match_rates=np.array([n_final / n_ref] * len(hungarian_cost_values), dtype=np.float32),
        optimal_threshold=0.0,
        optimal_matches=n_final,
        optimal_rate=n_final / n_ref if n_ref > 0 else 0.0,
        # Actual match data
        matched_ref_indices=matched_ref_indices,
        matched_tgt_indices=matched_tgt_indices,
        matched_costs=matched_costs_arr,
    )

    return {
        'animal_id': animal_id,
        'pair_name': pair_name,
        'ref_session': ref_session,
        'target_session': target_session,
        'n_ref_neurons': n_ref,
        'n_target_neurons': n_tgt,
        'n_inlier_quads': n_inlier_quads,
        'optimal_threshold': 0.0,
        'optimal_matches': int(n_final),
        'optimal_rate': float(n_final / n_ref) if n_ref > 0 else 0.0,
        'matched_ref_indices': matched_ref_indices,
        'matched_tgt_indices': matched_tgt_indices,
        'matched_costs': matched_costs_arr,
    }


# ==============================================================================
# ██  EVERYTHING BELOW IS UNCHANGED FROM BASELINE  ██
# ==============================================================================


# ==============================================================================
# Neuron Track Consolidation
# ==============================================================================

def consolidate_neuron_tracks(
    animal_id: str,
    pair_results: List[Dict[str, Any]],
    step2_5_dir: Path,
) -> Dict[str, Any]:
    """
    Consolidate pairwise matches into global neuron tracks across all sessions.

    This builds a mapping: global_neuron_id -> {session_idx: local_neuron_idx}

    Args:
        animal_id: Animal identifier
        pair_results: List of pairwise matching results
        step2_5_dir: Directory with original centroid data

    Returns:
        Dictionary with consolidated tracking data
    """
    print(f"\n{'='*100}")
    print(f"[CONSOLIDATE] Building global neuron tracks for {animal_id}")
    print(f"{'='*100}")

    # Extract all unique sessions
    sessions_set = set()
    for result in pair_results:
        sessions_set.add(result['ref_session'])
        sessions_set.add(result['target_session'])

    sessions = sorted(sessions_set)
    n_sessions = len(sessions)
    session_to_idx = {s: i for i, s in enumerate(sessions)}

    print(f"[CONSOLIDATE] Sessions: {n_sessions} → {sessions}")

    # Load centroids for each session
    session_centroids = {}
    session_n_neurons = {}

    for session in sessions:
        found = False
        for result in pair_results:
            if result['ref_session'] == session:
                sweep_file = step2_5_dir.parent / "step_3_results" / f"{result['pair_name']}_sweep.npz"
                if sweep_file.exists():
                    data = np.load(sweep_file, allow_pickle=False)
                    session_centroids[session] = data['ref_centroids']
                    session_n_neurons[session] = len(data['ref_centroids'])
                    found = True
                    break
            elif result['target_session'] == session:
                sweep_file = step2_5_dir.parent / "step_3_results" / f"{result['pair_name']}_sweep.npz"
                if sweep_file.exists():
                    data = np.load(sweep_file, allow_pickle=False)
                    session_centroids[session] = data['tgt_centroids']
                    session_n_neurons[session] = len(data['tgt_centroids'])
                    found = True
                    break

        if not found:
            logger.warning(f"Could not find centroids for session {session}")

    print(f"[CONSOLIDATE] Loaded centroids for {len(session_centroids)} sessions")

    # Build pairwise match graph
    pairwise_matches = {}

    for result in pair_results:
        ref_session = result['ref_session']
        tgt_session = result['target_session']

        match_map = {}
        for ref_idx, tgt_idx in zip(result['matched_ref_indices'], result['matched_tgt_indices']):
            match_map[int(ref_idx)] = int(tgt_idx)

        pairwise_matches[(ref_session, tgt_session)] = match_map

        print(f"[CONSOLIDATE]   {ref_session} → {tgt_session}: {len(match_map)} matches")

    # Build global tracks using transitive closure
    print(f"\n[CONSOLIDATE] Building global neuron tracks...")

    first_session = sessions[0]
    n_first = session_n_neurons.get(first_session, 0)

    neuron_tracks = {}
    next_global_id = 0

    for local_idx in range(n_first):
        neuron_tracks[next_global_id] = {0: local_idx}
        next_global_id += 1

    print(f"[CONSOLIDATE] Initialized {n_first} tracks from {first_session}")

    for session_idx in range(1, n_sessions):
        current_session = sessions[session_idx]
        n_current = session_n_neurons.get(current_session, 0)

        print(f"\n[CONSOLIDATE] Processing session {session_idx}: {current_session} ({n_current} neurons)")

        assigned_locals = set()

        for prev_session_idx in range(session_idx):
            prev_session = sessions[prev_session_idx]

            matches_forward = pairwise_matches.get((prev_session, current_session), {})
            matches_backward = pairwise_matches.get((current_session, prev_session), {})

            if matches_backward:
                matches = {v: k for k, v in matches_backward.items()}
            else:
                matches = matches_forward

            if not matches:
                continue

            print(f"[CONSOLIDATE]   Matching to {prev_session}: {len(matches)} pairs")

            for prev_local, curr_local in matches.items():
                if curr_local in assigned_locals:
                    continue

                found_global_id = None
                for global_id, track in neuron_tracks.items():
                    if prev_session_idx in track and track[prev_session_idx] == prev_local:
                        found_global_id = global_id
                        break

                if found_global_id is not None:
                    neuron_tracks[found_global_id][session_idx] = curr_local
                    assigned_locals.add(curr_local)

        n_new = 0
        for local_idx in range(n_current):
            if local_idx not in assigned_locals:
                neuron_tracks[next_global_id] = {session_idx: local_idx}
                next_global_id += 1
                n_new += 1

        print(f"[CONSOLIDATE]   Extended {len(assigned_locals)} existing tracks")
        print(f"[CONSOLIDATE]   Created {n_new} new tracks")

    track_lengths = np.array([len(track) for track in neuron_tracks.values()], dtype=np.int32)

    n_total_tracks = len(neuron_tracks)
    avg_track_length = np.mean(track_lengths)
    max_track_length = np.max(track_lengths)

    full_length_tracks = np.sum(track_lengths == n_sessions)
    partial_tracks = n_total_tracks - full_length_tracks

    print(f"\n[CONSOLIDATE] ===== Consolidation Results =====")
    print(f"[CONSOLIDATE] Total global tracks: {n_total_tracks}")
    print(f"[CONSOLIDATE] Full-length tracks (all {n_sessions} sessions): {full_length_tracks}")
    print(f"[CONSOLIDATE] Partial tracks: {partial_tracks}")
    print(f"[CONSOLIDATE] Average track length: {avg_track_length:.1f} sessions")
    print(f"[CONSOLIDATE] Maximum track length: {max_track_length} sessions")
    print(f"{'='*100}\n")

    return {
        'animal_id': animal_id,
        'sessions': sessions,
        'n_sessions': n_sessions,
        'neuron_tracks': neuron_tracks,
        'track_lengths': track_lengths,
        'session_centroids': session_centroids,
        'n_total_tracks': n_total_tracks,
        'full_length_tracks': full_length_tracks,
        'avg_track_length': float(avg_track_length),
        'max_track_length': int(max_track_length),
    }


# ==============================================================================
# Animal Processing (Sweep + Consolidation)
# ==============================================================================

def process_animal_complete(
    animal_id: str,
    step2_5_dir: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Process one animal: sweep for optimal thresholds, match at optimal, consolidate tracks.
    """
    print(f"\n{'#'*100}")
    print(f"# ANIMAL: {animal_id}")
    print(f"{'#'*100}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing animal: {animal_id}")
    logger.info(f"{'='*80}")

    pattern = f"{animal_id}_*_filtered_matches.npz"
    filter_files = sorted(step2_5_dir.glob(pattern))

    if not filter_files:
        logger.warning(f"No filtered match files found for {animal_id}")
        print(f"✗ No filtered match files found for {animal_id}")
        print(f"{'#'*100}\n")
        return {
            'animal_id': animal_id,
            'n_pairs': 0,
            'error': 'No filtered match files found'
        }

    print(f"Found {len(filter_files)} session pairs")
    print(f"Testing {len(hungarian_cost_values)} cost thresholds")
    logger.info(f"Found {len(filter_files)} session pairs for {animal_id}")

    print(f"\n{'─'*100}")
    print(f"PHASE 1: THRESHOLD SWEEP")
    print(f"{'─'*100}")

    sweep_results = []

    for filter_file in filter_files:
        result = process_session_pair_sweep(
            filter_file, output_dir, use_quad_voting, hungarian_cost_values
        )
        if result:
            sweep_results.append(result)

    if not sweep_results:
        print(f"✗ No valid sweep results for {animal_id}")
        print(f"{'#'*100}\n")
        return {
            'animal_id': animal_id,
            'n_pairs': 0,
            'error': 'No valid sweep results'
        }

    print(f"\n{'─'*100}")
    print(f"PHASE 2: TRACK CONSOLIDATION")
    print(f"{'─'*100}")

    tracking_data = consolidate_neuron_tracks(animal_id, sweep_results, step2_5_dir)

    print(f"\n{'─'*100}")
    print(f"PHASE 3: SAVE CONSOLIDATED TRACKING")
    print(f"{'─'*100}")

    tracking_file = output_dir / f"{animal_id}_consolidated_tracking.npz"

    np.savez_compressed(
        tracking_file,
        animal_id=animal_id,
        sessions=np.array(tracking_data['sessions'], dtype=object),
        n_sessions=tracking_data['n_sessions'],
        neuron_tracks=tracking_data['neuron_tracks'],
        track_lengths=tracking_data['track_lengths'],
        n_total_tracks=tracking_data['n_total_tracks'],
        full_length_tracks=tracking_data['full_length_tracks'],
        avg_track_length=tracking_data['avg_track_length'],
        max_track_length=tracking_data['max_track_length'],
    )

    print(f"[SAVE] ✓ Saved consolidated tracking: {tracking_file.name}")
    print(f"[SAVE]   - {tracking_data['n_total_tracks']} global tracks")
    print(f"[SAVE]   - {tracking_data['full_length_tracks']} full-length tracks")
    print(f"[SAVE]   - Average track length: {tracking_data['avg_track_length']:.1f} sessions")

    avg_optimal_threshold = np.mean([r['optimal_threshold'] for r in sweep_results])
    avg_optimal_rate = np.mean([r['optimal_rate'] for r in sweep_results])
    total_optimal_matches = sum(r['optimal_matches'] for r in sweep_results)

    json_safe_pairs = []
    for r in sweep_results:
        json_safe_pairs.append({
            'pair_name': r['pair_name'],
            'ref_session': r['ref_session'],
            'target_session': r['target_session'],
            'n_ref_neurons': r['n_ref_neurons'],
            'n_target_neurons': r['n_target_neurons'],
            'n_inlier_quads': r['n_inlier_quads'],
            'optimal_threshold': r['optimal_threshold'],
            'optimal_matches': r['optimal_matches'],
            'optimal_rate': r['optimal_rate'],
        })

    print(f"\n{'#'*100}")
    print(f"# ANIMAL {animal_id} COMPLETE")
    print(f"# Session pairs: {len(sweep_results)}")
    print(f"# Avg optimal threshold: {avg_optimal_threshold:.2f}")
    print(f"# Avg optimal match rate: {avg_optimal_rate*100:.1f}%")
    print(f"# Total tracks: {tracking_data['n_total_tracks']}")
    print(f"# Full-length tracks: {tracking_data['full_length_tracks']}")
    print(f"{'#'*100}\n")

    return {
        'animal_id': animal_id,
        'n_pairs': len(sweep_results),
        'pair_results': json_safe_pairs,
        'avg_optimal_threshold': float(avg_optimal_threshold),
        'avg_optimal_rate': float(avg_optimal_rate),
        'total_optimal_matches': int(total_optimal_matches),
        'n_total_tracks': int(tracking_data['n_total_tracks']),
        'full_length_tracks': int(tracking_data['full_length_tracks']),
        'avg_track_length': float(tracking_data['avg_track_length']),
        'max_track_length': int(tracking_data['max_track_length']),
    }

def worker_process_animal_complete(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel animal processing."""
    animal_id, step2_5_dir_str, output_dir_str, use_quad_voting, hungarian_cost_values = args

    step2_5_dir = Path(step2_5_dir_str)
    output_dir = Path(output_dir_str)

    log_dir = output_dir / "logs_step3"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=True)

    log_worker_start("Step 3 Complete", animal_id, {})

    result = process_animal_complete(animal_id, step2_5_dir, output_dir, use_quad_voting, hungarian_cost_values)

    log_worker_finish("Step 3 Complete", animal_id, result)

    return result

# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def run_step_3_final_matching(
    input_dir: str,
    output_dir: str,
    hungarian_cost_min: float = 0.0,
    hungarian_cost_max: float = 2319.0,
    hungarian_cost_steps: int = 20,
    use_quad_voting: bool = True,
    processes: Optional[int] = None,
    verbose: bool = True,
    # Legacy parameters (ignored but accepted for compatibility)
    hungarian_max_cost: Optional[float] = None,
    target_match_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run Step 3: Hungarian sweep + consolidated neuron tracking.

    This is the complete Step 3 that:
    1. Sweeps over hungarian_max_cost values to find optimal thresholds
    2. Performs neuron matching at optimal thresholds
    3. Consolidates matches into global neuron tracks across sessions
    4. Saves both sweep results and consolidated tracking files

    Args:
        input_dir: Input directory (not used, kept for compatibility)
        output_dir: Output directory containing Step 2.5 results
        hungarian_cost_min: Minimum cost threshold to test
        hungarian_cost_max: Maximum cost threshold to test
        hungarian_cost_steps: Number of threshold values to test
        use_quad_voting: If True, use quad voting for cost matrix
        processes: Number of parallel workers (None = use all CPUs)
        verbose: Enable verbose logging

    Returns:
        List of results dictionaries, one per animal
    """
    output_path = Path(output_dir)

    possible_dirs = [
        output_path / 'step_2_5_results',
        output_path / 'step_2_5',
    ]

    step2_5_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            step2_5_dir = dir_path
            break

    if step2_5_dir is None:
        logger.error(f"Step 2.5 results not found. Checked:")
        for d in possible_dirs:
            logger.error(f"  - {d}")
        logger.error("Please run Step 2.5 RANSAC filtering first!")
        return []

    hungarian_cost_values = np.linspace(hungarian_cost_min, hungarian_cost_max, hungarian_cost_steps)

    logger.info(f"Loading Step 2.5 results from: {step2_5_dir}")
    logger.info(f"Hungarian cost sweep: {hungarian_cost_min} to {hungarian_cost_max} ({hungarian_cost_steps} steps)")

    filtered_files = sorted(step2_5_dir.glob("*_filtered_matches.npz"))

    if not filtered_files:
        logger.error(f"No *_filtered_matches.npz files found in {step2_5_dir}")
        return []

    print(f"\n{'#'*100}")
    print(f"# STEP 3: HUNGARIAN SWEEP + CONSOLIDATED TRACKING")
    print(f"{'#'*100}")
    print(f"Loading from: {step2_5_dir}")
    print(f"Found {len(filtered_files)} filtered match files")
    print(f"Cost range: {hungarian_cost_min} to {hungarian_cost_max}")
    print(f"Steps: {hungarian_cost_steps}")
    print(f"Testing thresholds: {hungarian_cost_values}")

    animals = set()
    for f in filtered_files:
        animal_id = f.stem.split('_')[0]
        animals.add(animal_id)

    animals = sorted(animals)

    if not animals:
        logger.error("No animals found in Step 2.5 results")
        return []

    print(f"Animals to process: {len(animals)} → {animals}")
    print(f"Method: {'Quad Voting' if use_quad_voting else 'Minimum Distance'}")
    print(f"{'#'*100}\n")

    logger.info(f"Found {len(animals)} animals to process: {animals}")

    step3_dir = ensure_output_dir(output_dir, 3, verbose=False)

    args_list = [
        (animal_id, str(step2_5_dir), str(step3_dir), use_quad_voting, hungarian_cost_values)
        for animal_id in animals
    ]

    results = run_parallel_animals(
        worker_process_animal_complete,
        args_list,
        max_workers=processes,
        verbose=verbose,
    )

    summary_file = step3_dir / "step3_summary.json"
    save_json_summary({
        'cost_min': float(hungarian_cost_min),
        'cost_max': float(hungarian_cost_max),
        'cost_steps': int(hungarian_cost_steps),
        'cost_values': hungarian_cost_values.tolist(),
        'use_quad_voting': use_quad_voting,
        'n_animals': len(results),
        'animals': results,
    }, summary_file)

    print(f"\n{'#'*100}")
    print(f"# STEP 3 COMPLETE")
    print(f"{'#'*100}")
    print(f"Animals processed: {len(results)}")
    print(f"Results saved to: {step3_dir}")

    if results:
        print(f"\n{'Per-Animal Summary:':^100}")
        print(f"{'Animal ID':<15} {'Pairs':<8} {'Opt Thresh':<12} {'Match Rate':<12} {'Tracks':<10} {'Full Tracks':<12}")
        print(f"{'-'*100}")
        for r in results:
            if r['n_pairs'] > 0:
                print(f"{r['animal_id']:<15} {r['n_pairs']:<8} "
                      f"{r['avg_optimal_threshold']:>11.2f} {r['avg_optimal_rate']*100:>11.1f}% "
                      f"{r['n_total_tracks']:>9} {r['full_length_tracks']:>11}")

    print(f"{'#'*100}\n")

    logger.info(f"Step 3 complete: {len(results)} animals")

    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY - Match Rates by Animal:")
        print(f"{'='*80}")
        for r in results:
            if r.get('n_pairs', 0) > 0:
                match_rate = r.get('avg_optimal_rate', 0) * 100
                print(f"  {r['animal_id']}: {match_rate:.1f}% average match rate ({r['n_pairs']} pairs)")
        print(f"{'='*80}\n")

    return results