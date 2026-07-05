"""
Step 3: Hungarian Cost Sweep + Consolidated Neuron Tracking

V3.1: RANSAC-informed matching with vote normalization, asymmetric dummy
      padding, post-filter, second-pass recovery, per-match confidence
      scoring, and per-track confidence propagation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from utilities import *
logger = logging.getLogger("neuron_mapping_consolidated")


# ==============================================================================
# Cost Matrix Construction (RANSAC-informed, normalized votes)
# ==============================================================================

def build_cost_matrix_from_filtered_quads(
    ref_centroids: np.ndarray,
    tgt_centroids: np.ndarray,
    match_indices: np.ndarray,
    transform_matrix: np.ndarray,
    transform_translation: np.ndarray,
    ransac_max_residual: float = 5.0,
    use_quad_voting: bool = True,
    block_zero_vote_pairs: bool = False,
    dist_cutoff_multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build neuron-to-neuron cost matrix combining normalized quad votes
    with RANSAC-transformed spatial distance.

    Returns:
        cost_matrix, dist_matrix, vote_matrix, vote_norm
    """
    n_ref = len(ref_centroids)
    n_tgt = len(tgt_centroids)

    A = np.array(transform_matrix, dtype=np.float64)
    t = np.array(transform_translation, dtype=np.float64)
    ref_transformed = (A @ ref_centroids.T).T + t
    diff = ref_transformed[:, None, :] - tgt_centroids[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    if not use_quad_voting:
        cost_matrix = np.full((n_ref, n_tgt), 1e6, dtype=np.float32)
        vote_matrix = np.zeros((n_ref, n_tgt), dtype=np.float64)
        for quad_match in match_indices:
            ref_neurons = quad_match[:4].astype(int)
            tgt_neurons = quad_match[4:].astype(int)
            for i_ref in ref_neurons:
                for i_tgt in tgt_neurons:
                    d = float(dist_matrix[i_ref, i_tgt])
                    cost_matrix[i_ref, i_tgt] = min(cost_matrix[i_ref, i_tgt], d)
                    vote_matrix[i_ref, i_tgt] += 1
        return cost_matrix, dist_matrix, vote_matrix, vote_matrix.copy()

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

    dist_cutoff = dist_cutoff_multiplier * ransac_max_residual
    vote_cost[dist_matrix > dist_cutoff] = np.inf

    if block_zero_vote_pairs:
        vote_cost[vote_matrix == 0] = np.inf

    voted_mask = np.isfinite(vote_cost) & (vote_matrix > 0)
    if voted_mask.sum() == 0:
        return vote_cost.astype(np.float32), dist_matrix, vote_matrix, vote_norm

    # Distance term expressed in units of the RANSAC residual. Using a FIXED
    # residual-normalized scale (rather than dist_weight = median_vote_cost /
    # median_dist) is critical: for clean data the strongly-voted true pairs all
    # share the maximum vote_norm, so their vote_cost -> 0 and median_vote_cost
    # -> 0, which zeroed dist_weight and dropped distance from the cost entirely.
    # That produced an all-equal (all-zero) cost matrix and an arbitrary
    # Hungarian assignment. Keeping distance always-on lets geometry break ties
    # between equally-voted candidates.
    dist_term = dist_matrix / max(ransac_max_residual, 1e-9)

    combined_cost = np.where(voted_mask,
                             vote_cost + dist_term,
                             np.inf)

    if not block_zero_vote_pairs:
        zero_vote_nearby = (vote_matrix == 0) & (dist_matrix <= dist_cutoff)
        if zero_vote_nearby.any():
            finite_voted = combined_cost[voted_mask]
            vote_penalty = float(np.max(finite_voted)) if len(finite_voted) > 0 else max_vnorm
            combined_cost[zero_vote_nearby] = vote_penalty + dist_term[zero_vote_nearby]
            n_fallback = int(zero_vote_nearby.sum())
            logger.info(f"    Zero-vote fallback: {n_fallback} pairs given distance-only cost")

    logger.info(f"    Built cost matrix: {n_ref} x {n_tgt} "
                f"(residual-normalized distance, cutoff={dist_cutoff:.1f}px)")

    return combined_cost, dist_matrix, vote_matrix, vote_norm


# ==============================================================================
# Per-match confidence scoring
# ==============================================================================

def compute_match_confidence(
    matched_ref: np.ndarray,
    matched_tgt: np.ndarray,
    combined_cost: np.ndarray,
    dist_matrix: np.ndarray,
    vote_matrix: np.ndarray,
    vote_norm: np.ndarray,
    dist_cutoff: float,
    match_pass: np.ndarray,
) -> np.ndarray:
    """
    Compute a [0, 1] confidence score for each matched neuron pair.

    Four signals, equally weighted:
      vote_score    degree-normalized votes / field max
      dist_score    1 - transformed_dist / dist_cutoff
      margin_score  (2nd_best_cost - best_cost) / 2nd_best_cost
      pass_score    1.0 for pass-1, 0.5 for pass-2
    """
    M = len(matched_ref)
    if M == 0:
        return np.array([], dtype=np.float32)

    max_vnorm = vote_norm.max() if vote_norm.max() > 0 else 1.0

    vote_scores = np.array([
        vote_norm[r, t] / max_vnorm for r, t in zip(matched_ref, matched_tgt)
    ])

    dist_scores = np.array([
        np.clip(1.0 - dist_matrix[r, t] / dist_cutoff, 0.0, 1.0)
        for r, t in zip(matched_ref, matched_tgt)
    ])

    margin_scores = np.zeros(M, dtype=np.float64)
    for k, (r, t) in enumerate(zip(matched_ref, matched_tgt)):
        row = combined_cost[r, :]
        finite = row[np.isfinite(row)]
        if len(finite) < 2:
            margin_scores[k] = 0.0
            continue
        s = np.sort(finite)
        margin_scores[k] = np.clip((s[1] - s[0]) / s[1], 0.0, 1.0) if s[1] > 0 else 0.0

    pass_scores = np.where(match_pass == 1, 1.0, 0.5)

    confidence = (vote_scores + dist_scores + margin_scores + pass_scores) / 4.0

    logger.info(f"    Confidence: mean={np.mean(confidence):.3f} "
                f"med={np.median(confidence):.3f} "
                f"min={np.min(confidence):.3f} max={np.max(confidence):.3f}")
    logger.info(f"      components: vote={np.mean(vote_scores):.3f} "
                f"dist={np.mean(dist_scores):.3f} "
                f"margin={np.mean(margin_scores):.3f} "
                f"pass={np.mean(pass_scores):.3f}")

    return confidence.astype(np.float32)


# ==============================================================================
# Session Pair Processing (padded Hungarian + post-filter + 2nd pass)
# ==============================================================================

def process_session_pair_sweep(
    filter_file: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
    use_asymmetric_dummy_costs: bool = False,
    block_zero_vote_pairs: bool = False,
    dist_cutoff_multiplier: float = 3.0,
    postfilter_residual_multiplier: float = 1.0,
    pass2_cutoff_multiplier: float = 2.0,
    pass2_dummy_percentile: float = 75.0,
) -> Optional[Dict[str, Any]]:
    try:
        data = np.load(filter_file, allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load {filter_file}: {e}")
        return None

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

    logger.info(f"  [{ref_session} -> {target_session}] "
                f"{n_ref}->{n_tgt} neurons, {n_inlier_quads:,} inlier quads")

    if n_inlier_quads == 0:
        logger.warning(f"  No inlier matches for {pair_name}")
        return None

    ransac_max_residual = float(data.get('max_residual_threshold', 5.0))

    combined_cost, dist_matrix, vote_matrix, vote_norm = \
        build_cost_matrix_from_filtered_quads(
            ref_centroids=ref_centroids,
            tgt_centroids=tgt_centroids,
            match_indices=match_indices,
            transform_matrix=transform_matrix,
            transform_translation=transform_translation,
            ransac_max_residual=ransac_max_residual,
            use_quad_voting=use_quad_voting,
            block_zero_vote_pairs=block_zero_vote_pairs,
            dist_cutoff_multiplier=dist_cutoff_multiplier,
        )

    dist_cutoff = dist_cutoff_multiplier * ransac_max_residual

    finite_costs = combined_cost[np.isfinite(combined_cost)]
    if len(finite_costs) == 0:
        logger.warning(f"  No finite costs for {pair_name} — skipping")
        return None

    global_median_cost = float(np.median(finite_costs))
    # Guard against a degenerate all-low-cost field (e.g. many equally-voted
    # true pairs): a zero median would make every dummy / unmatched entry 0 and
    # turn the padded matrix into all-zeros, giving an arbitrary assignment.
    if global_median_cost <= 1e-6:
        pos = finite_costs[finite_costs > 1e-6]
        global_median_cost = float(np.median(pos)) if pos.size else 1.0

    if use_asymmetric_dummy_costs:
        ref_min_dist = dist_matrix.min(axis=1)
        tgt_min_dist = dist_matrix.min(axis=0)
        proximity_ref = np.clip(1.0 - ref_min_dist / dist_cutoff, 0.1, 1.0)
        ref_dummy_costs = global_median_cost * proximity_ref
        proximity_tgt = np.clip(1.0 - tgt_min_dist / dist_cutoff, 0.1, 1.0)
        tgt_dummy_costs = global_median_cost * proximity_tgt
    else:
        ref_dummy_costs = np.full(n_ref, global_median_cost)
        tgt_dummy_costs = np.full(n_tgt, global_median_cost)

    padded = np.full((n_ref + n_tgt, n_tgt + n_ref),
                     global_median_cost * 2, dtype=np.float64)
    padded[:n_ref, :n_tgt] = np.where(np.isfinite(combined_cost),
                                       combined_cost, global_median_cost * 10)
    for i in range(n_ref):
        padded[i, n_tgt:] = ref_dummy_costs[i]
    for j in range(n_tgt):
        padded[n_ref:, j] = tgt_dummy_costs[j]
    padded[n_ref:, n_tgt:] = 0.0

    # Pass 1
    row_ind, col_ind = linear_sum_assignment(padded)
    real_mask = (row_ind < n_ref) & (col_ind < n_tgt)
    pass1_ref = row_ind[real_mask]
    pass1_tgt = col_ind[real_mask]

    postfilter_cutoff = postfilter_residual_multiplier * ransac_max_residual
    pass1_dists = np.array([dist_matrix[r, c] for r, c in zip(pass1_ref, pass1_tgt)])
    keep = pass1_dists <= postfilter_cutoff
    filtered_ref = pass1_ref[keep]
    filtered_tgt = pass1_tgt[keep]
    filtered_costs = np.array([combined_cost[r, c] for r, c in zip(filtered_ref, filtered_tgt)])
    filtered_pass = np.ones(len(filtered_ref), dtype=np.int32)

    n_pass1 = len(pass1_ref)
    n_filtered = len(filtered_ref)
    logger.info(f"    Pass 1: {n_pass1} matches -> {n_filtered} after post-filter")

    # Pass 2
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
        relaxed_cutoff = pass2_cutoff_multiplier * ransac_max_residual
        sub_cost[sub_dist > relaxed_cutoff] = np.inf

        sub_finite = sub_cost[np.isfinite(sub_cost)]
        if len(sub_finite) > 0:
            sub_dummy = float(np.percentile(sub_finite, pass2_dummy_percentile))

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
            p2_costs = np.array([combined_cost[r, c] for r, c in zip(p2_ref, p2_tgt)])

            if len(p2_ref) > 0:
                filtered_ref = np.concatenate([filtered_ref, p2_ref])
                filtered_tgt = np.concatenate([filtered_tgt, p2_tgt])
                filtered_costs = np.concatenate([filtered_costs, p2_costs])
                filtered_pass = np.concatenate([
                    filtered_pass, np.full(len(p2_ref), 2, dtype=np.int32)
                ])
                n_recovered = len(p2_ref)

    n_final = len(filtered_ref)
    logger.info(f"    Pass 2: +{n_recovered} recovered -> {n_final} total matches")

    # Confidence
    confidence = compute_match_confidence(
        matched_ref=filtered_ref, matched_tgt=filtered_tgt,
        combined_cost=combined_cost, dist_matrix=dist_matrix,
        vote_matrix=vote_matrix, vote_norm=vote_norm,
        dist_cutoff=dist_cutoff, match_pass=filtered_pass,
    )

    # Save
    matched_ref_indices = filtered_ref.astype(np.int32)
    matched_tgt_indices = filtered_tgt.astype(np.int32)
    matched_costs_arr = filtered_costs.astype(np.float32)

    sweep_file = output_dir / f"{pair_name}_sweep.npz"
    np.savez_compressed(
        sweep_file,
        animal_id=animal_id, pair_name=pair_name,
        ref_session=ref_session, target_session=target_session,
        ref_centroids=ref_centroids, tgt_centroids=tgt_centroids,
        n_ref_neurons=n_ref, n_target_neurons=n_tgt,
        n_inlier_quads=n_inlier_quads,
        cost_thresholds=hungarian_cost_values,
        match_counts=np.array([n_final] * len(hungarian_cost_values), dtype=np.int32),
        match_rates=np.array([n_final / n_ref] * len(hungarian_cost_values), dtype=np.float32),
        optimal_threshold=0.0, optimal_matches=n_final,
        optimal_rate=n_final / n_ref if n_ref > 0 else 0.0,
        matched_ref_indices=matched_ref_indices,
        matched_tgt_indices=matched_tgt_indices,
        matched_costs=matched_costs_arr,
        match_confidence=confidence,
        match_pass=filtered_pass,
    )

    return {
        'animal_id': animal_id, 'pair_name': pair_name,
        'ref_session': ref_session, 'target_session': target_session,
        'n_ref_neurons': n_ref, 'n_target_neurons': n_tgt,
        'n_inlier_quads': n_inlier_quads,
        'optimal_threshold': 0.0,
        'optimal_matches': int(n_final),
        'optimal_rate': float(n_final / n_ref) if n_ref > 0 else 0.0,
        'matched_ref_indices': matched_ref_indices,
        'matched_tgt_indices': matched_tgt_indices,
        'matched_costs': matched_costs_arr,
        'match_confidence': confidence,
        'match_pass': filtered_pass,
    }


# ==============================================================================
# Neuron Track Consolidation (with per-track confidence)
# ==============================================================================

def consolidate_neuron_tracks(
    animal_id: str,
    pair_results: List[Dict[str, Any]],
    step2_5_dir: Path,
) -> Dict[str, Any]:
    """
    Consolidate pairwise matches into global neuron tracks.
    Per-track confidence = min confidence across all links in the chain.
    """
    print(f"\n{'='*100}")
    print(f"[CONSOLIDATE] Building global neuron tracks for {animal_id}")
    print(f"{'='*100}")

    sessions_set = set()
    for result in pair_results:
        sessions_set.add(result['ref_session'])
        sessions_set.add(result['target_session'])
    sessions = sorted(sessions_set)
    n_sessions = len(sessions)

    print(f"[CONSOLIDATE] Sessions: {n_sessions} -> {sessions}")

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

    # Build pairwise match graph with per-match confidence
    pairwise_matches = {}
    pairwise_confidence = {}

    for result in pair_results:
        ref_s, tgt_s = result['ref_session'], result['target_session']
        match_map = {}
        conf_map = {}
        confidence = result.get('match_confidence')

        for idx, (ri, ti) in enumerate(
            zip(result['matched_ref_indices'], result['matched_tgt_indices'])
        ):
            ri, ti = int(ri), int(ti)
            match_map[ri] = ti
            if confidence is not None and idx < len(confidence):
                conf_map[(ri, ti)] = float(confidence[idx])
            else:
                conf_map[(ri, ti)] = 0.5

        pairwise_matches[(ref_s, tgt_s)] = match_map
        pairwise_confidence[(ref_s, tgt_s)] = conf_map
        print(f"[CONSOLIDATE]   {ref_s} -> {tgt_s}: {len(match_map)} matches")

    # Build global tracks
    print(f"\n[CONSOLIDATE] Building global neuron tracks...")
    first_session = sessions[0]
    n_first = session_n_neurons.get(first_session, 0)

    neuron_tracks = {}
    track_min_conf = {}
    track_link_confs = {}
    next_global_id = 0

    for local_idx in range(n_first):
        neuron_tracks[next_global_id] = {0: local_idx}
        track_min_conf[next_global_id] = 1.0
        track_link_confs[next_global_id] = []
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
            conf_forward = pairwise_confidence.get((prev_session, current_session), {})
            matches_backward = pairwise_matches.get((current_session, prev_session), {})
            conf_backward = pairwise_confidence.get((current_session, prev_session), {})

            if matches_backward:
                matches = {v: k for k, v in matches_backward.items()}
                conf = {(ti, ri): c for (ri, ti), c in conf_backward.items()}
            else:
                matches = matches_forward
                conf = conf_forward

            if not matches:
                continue

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

                    link_conf = conf.get((prev_local, curr_local), 0.5)
                    track_link_confs[found_global_id].append(link_conf)
                    track_min_conf[found_global_id] = min(
                        track_min_conf[found_global_id], link_conf
                    )

        n_new = 0
        for local_idx in range(n_current):
            if local_idx not in assigned_locals:
                neuron_tracks[next_global_id] = {session_idx: local_idx}
                track_min_conf[next_global_id] = 1.0
                track_link_confs[next_global_id] = []
                next_global_id += 1
                n_new += 1

        print(f"[CONSOLIDATE]   Extended {len(assigned_locals)} existing tracks, {n_new} new")

    track_lengths = np.array([len(t) for t in neuron_tracks.values()], dtype=np.int32)

    track_ids = sorted(neuron_tracks.keys())
    track_confidence = np.array(
        [track_min_conf[gid] for gid in track_ids], dtype=np.float32
    )
    track_mean_confidence = np.array([
        float(np.mean(track_link_confs[gid])) if track_link_confs[gid] else 1.0
        for gid in track_ids
    ], dtype=np.float32)

    n_total_tracks = len(neuron_tracks)
    avg_track_length = float(np.mean(track_lengths))
    max_track_length = int(np.max(track_lengths))
    full_length_tracks = int(np.sum(track_lengths == n_sessions))

    full_mask = track_lengths == n_sessions
    if np.any(full_mask):
        full_conf = track_confidence[full_mask]
        avg_full_conf = float(np.mean(full_conf))
        high_conf_full = int(np.sum(full_conf >= 0.5))
    else:
        avg_full_conf = 0.0
        high_conf_full = 0

    print(f"\n[CONSOLIDATE] ===== Results =====")
    print(f"[CONSOLIDATE] Total tracks: {n_total_tracks}")
    print(f"[CONSOLIDATE] Full-length: {full_length_tracks} (all {n_sessions} sessions)")
    print(f"[CONSOLIDATE] Avg track length: {avg_track_length:.1f}")
    print(f"[CONSOLIDATE] Full-length avg confidence: {avg_full_conf:.3f}")
    print(f"[CONSOLIDATE] Full-length high-conf (>=0.5): {high_conf_full}/{full_length_tracks}")
    print(f"{'='*100}\n")

    return {
        'animal_id': animal_id,
        'sessions': sessions,
        'n_sessions': n_sessions,
        'neuron_tracks': neuron_tracks,
        'track_lengths': track_lengths,
        'track_confidence': track_confidence,
        'track_mean_confidence': track_mean_confidence,
        'session_centroids': session_centroids,
        'n_total_tracks': n_total_tracks,
        'full_length_tracks': full_length_tracks,
        'avg_track_length': avg_track_length,
        'max_track_length': max_track_length,
    }


# ==============================================================================
# Animal Processing
# ==============================================================================

def _load_sweep_result(sweep_file: Path) -> Optional[Dict[str, Any]]:
    """Reconstruct a process_session_pair_sweep result dict from a saved
    *_sweep.npz, so skip_existing can reuse a pair without recomputing it while
    still feeding track consolidation. Mirrors the live return dict's keys."""
    try:
        d = np.load(sweep_file, allow_pickle=False)
    except Exception:
        return None
    return {
        'animal_id': decode_string_field(d.get('animal_id', '')),
        'pair_name': decode_string_field(d.get('pair_name', '')),
        'ref_session': decode_string_field(d.get('ref_session', '')),
        'target_session': decode_string_field(d.get('target_session', '')),
        'n_ref_neurons': int(d['n_ref_neurons']),
        'n_target_neurons': int(d['n_target_neurons']),
        'n_inlier_quads': int(d['n_inlier_quads']),
        'optimal_threshold': float(d['optimal_threshold']),
        'optimal_matches': int(d['optimal_matches']),
        'optimal_rate': float(d['optimal_rate']),
        'matched_ref_indices': d['matched_ref_indices'],
        'matched_tgt_indices': d['matched_tgt_indices'],
        'matched_costs': d['matched_costs'],
        'match_confidence': d['match_confidence'],
        'match_pass': d['match_pass'],
    }


def process_animal_complete(
    animal_id: str,
    step2_5_dir: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
    use_asymmetric_dummy_costs: bool = False,
    block_zero_vote_pairs: bool = False,
    dist_cutoff_multiplier: float = 3.0,
    postfilter_residual_multiplier: float = 1.0,
    pass2_cutoff_multiplier: float = 2.0,
    pass2_dummy_percentile: float = 75.0,
    skip_existing: bool = True,
) -> Dict[str, Any]:
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
        return {'animal_id': animal_id, 'n_pairs': 0, 'error': 'No filtered match files found'}

    print(f"Found {len(filter_files)} session pairs")

    sweep_results = []
    n_reused = 0
    for filter_file in filter_files:
        pair_name = filter_file.stem.replace('_filtered_matches', '')
        sweep_file = output_dir / f"{pair_name}_sweep.npz"

        # Honor skip_existing: reuse a saved sweep instead of recomputing it.
        # Reload (don't drop) so track consolidation still sees every pair.
        if skip_existing and sweep_file.exists():
            reused = _load_sweep_result(sweep_file)
            if reused is not None:
                n_reused += 1
                sweep_results.append(reused)
                continue

        result = process_session_pair_sweep(
            filter_file, output_dir, use_quad_voting, hungarian_cost_values,
            use_asymmetric_dummy_costs=use_asymmetric_dummy_costs,
            block_zero_vote_pairs=block_zero_vote_pairs,
            dist_cutoff_multiplier=dist_cutoff_multiplier,
            postfilter_residual_multiplier=postfilter_residual_multiplier,
            pass2_cutoff_multiplier=pass2_cutoff_multiplier,
            pass2_dummy_percentile=pass2_dummy_percentile,
        )
        if result:
            sweep_results.append(result)

    if n_reused:
        logger.info(f"  skip_existing: reused {n_reused} existing sweep(s) for {animal_id}")

    if not sweep_results:
        return {'animal_id': animal_id, 'n_pairs': 0, 'error': 'No valid results'}

    tracking_data = consolidate_neuron_tracks(animal_id, sweep_results, step2_5_dir)

    tracking_file = output_dir / f"{animal_id}_consolidated_tracking.npz"
    np.savez_compressed(
        tracking_file,
        animal_id=animal_id,
        sessions=np.array(tracking_data['sessions'], dtype=object),
        n_sessions=tracking_data['n_sessions'],
        neuron_tracks=tracking_data['neuron_tracks'],
        track_lengths=tracking_data['track_lengths'],
        track_confidence=tracking_data['track_confidence'],
        track_mean_confidence=tracking_data['track_mean_confidence'],
        n_total_tracks=tracking_data['n_total_tracks'],
        full_length_tracks=tracking_data['full_length_tracks'],
        avg_track_length=tracking_data['avg_track_length'],
        max_track_length=tracking_data['max_track_length'],
    )

    avg_rate = float(np.mean([r['optimal_rate'] for r in sweep_results]))
    total_matches = sum(r['optimal_matches'] for r in sweep_results)

    json_safe_pairs = [{
        'pair_name': r['pair_name'], 'ref_session': r['ref_session'],
        'target_session': r['target_session'],
        'n_ref_neurons': r['n_ref_neurons'], 'n_target_neurons': r['n_target_neurons'],
        'n_inlier_quads': r['n_inlier_quads'],
        'optimal_threshold': r['optimal_threshold'],
        'optimal_matches': r['optimal_matches'], 'optimal_rate': r['optimal_rate'],
    } for r in sweep_results]

    print(f"\n# {animal_id} COMPLETE: {avg_rate*100:.1f}% rate, "
          f"{tracking_data['full_length_tracks']}/{tracking_data['n_total_tracks']} full tracks\n")

    return {
        'animal_id': animal_id,
        'n_pairs': len(sweep_results),
        'pair_results': json_safe_pairs,
        'avg_optimal_threshold': 0.0,
        'avg_optimal_rate': avg_rate,
        'total_optimal_matches': int(total_matches),
        'n_total_tracks': int(tracking_data['n_total_tracks']),
        'full_length_tracks': int(tracking_data['full_length_tracks']),
        'avg_track_length': tracking_data['avg_track_length'],
        'max_track_length': tracking_data['max_track_length'],
    }


def worker_process_animal_complete(args: Tuple) -> Dict[str, Any]:
    (animal_id, step2_5_dir_str, output_dir_str, use_quad_voting,
     hungarian_cost_values, use_asymmetric_dummy_costs, block_zero_vote_pairs,
     dist_cutoff_multiplier, postfilter_residual_multiplier,
     pass2_cutoff_multiplier, pass2_dummy_percentile, skip_existing) = args

    step2_5_dir = Path(step2_5_dir_str)
    output_dir = Path(output_dir_str)

    log_dir = output_dir / "logs_step3"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=True)

    log_worker_start("Step 3 Complete", animal_id, {})
    result = process_animal_complete(
        animal_id, step2_5_dir, output_dir, use_quad_voting, hungarian_cost_values,
        use_asymmetric_dummy_costs=use_asymmetric_dummy_costs,
        block_zero_vote_pairs=block_zero_vote_pairs,
        dist_cutoff_multiplier=dist_cutoff_multiplier,
        postfilter_residual_multiplier=postfilter_residual_multiplier,
        pass2_cutoff_multiplier=pass2_cutoff_multiplier,
        pass2_dummy_percentile=pass2_dummy_percentile,
        skip_existing=skip_existing,
    )
    log_worker_finish("Step 3 Complete", animal_id, result)
    return result


# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def run_step_3_final_matching(
    input_dir: str,
    output_dir: str,
    use_quad_voting: bool = True,
    use_asymmetric_dummy_costs: bool = False,
    block_zero_vote_pairs: bool = False,
    dist_cutoff_multiplier: float = 3.0,
    postfilter_residual_multiplier: float = 1.0,
    pass2_cutoff_multiplier: float = 2.0,
    pass2_dummy_percentile: float = 75.0,
    processes: Optional[int] = None,
    verbose: bool = True,
    skip_existing: bool = False,   # default off so direct callers still recompute;
                                   # the GUI forwards config.skip_existing explicitly
    # Legacy (accepted, ignored)
    hungarian_cost_min: float = 0.0,
    hungarian_cost_max: float = 2319.0,
    hungarian_cost_steps: int = 20,
    hungarian_max_cost: Optional[float] = None,
    target_match_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    output_path = Path(output_dir)

    step2_5_dir = None
    for d in [output_path / 'step_2_5_results', output_path / 'step_2_5']:
        if d.exists():
            step2_5_dir = d
            break

    if step2_5_dir is None:
        logger.error("Step 2.5 results not found!")
        return []

    hungarian_cost_values = np.linspace(hungarian_cost_min, hungarian_cost_max, hungarian_cost_steps)

    filtered_files = sorted(step2_5_dir.glob("*_filtered_matches.npz"))
    if not filtered_files:
        logger.error(f"No filtered match files in {step2_5_dir}")
        return []

    animals = sorted({f.stem.split('_')[0] for f in filtered_files})
    if not animals:
        logger.error("No animals found")
        return []

    logger.info(f"Step 3 V3.1: {len(animals)} animals, {len(filtered_files)} pairs")

    step3_dir = ensure_output_dir(output_dir, 3, verbose=False)

    args_list = [
        (aid, str(step2_5_dir), str(step3_dir), use_quad_voting,
         hungarian_cost_values, use_asymmetric_dummy_costs, block_zero_vote_pairs,
         dist_cutoff_multiplier, postfilter_residual_multiplier,
         pass2_cutoff_multiplier, pass2_dummy_percentile, skip_existing)
        for aid in animals
    ]

    results = run_parallel_animals(
        worker_process_animal_complete, args_list,
        max_workers=processes, verbose=verbose,
    )

    summary_file = step3_dir / "step3_summary.json"
    save_json_summary({
        'use_quad_voting': use_quad_voting,
        'use_asymmetric_dummy_costs': use_asymmetric_dummy_costs,
        'block_zero_vote_pairs': block_zero_vote_pairs,
        'dist_cutoff_multiplier': dist_cutoff_multiplier,
        'postfilter_residual_multiplier': postfilter_residual_multiplier,
        'pass2_cutoff_multiplier': pass2_cutoff_multiplier,
        'pass2_dummy_percentile': pass2_dummy_percentile,
        'n_animals': len(results),
        'animals': results,
    }, summary_file)

    logger.info(f"Step 3 complete: {len(results)} animals")
    for r in results:
        if r.get('n_pairs', 0) > 0:
            logger.info(f"  {r['animal_id']}: {r['avg_optimal_rate']*100:.1f}% rate, "
                        f"{r['full_length_tracks']}/{r['n_total_tracks']} full tracks")

    return results