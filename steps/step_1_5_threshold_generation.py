"""
Step 1.5: Global similarity threshold calibration with √N scaling.
"""

import gc
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import multiprocessing
import queue as queue_module

from utilities import *
logger = logging.getLogger("neuron_mapping")

# The first animal ID to start will become the "verbose" worker.
# All others only send pair_done and done messages (no per-threshold spam).
_VERBOSE_WORKER_ID = None

# ==============================================================================
# Per-Animal Threshold Calibration
# ==============================================================================

def _convert_matches_to_array(matches: List) -> np.ndarray:
    if not matches:
        return np.array([])
    result = []
    for match in matches:
        ref_idx  = match[0]
        tgt_idx  = match[1]
        distance = match[4] if len(match) > 4 else 0.0
        row = list(ref_idx) + list(tgt_idx) + [distance]
        result.append(row)
    return np.array(result, dtype=np.float32)


# ==============================================================================
# Memory helper — works on Windows + Linux, no hard dependency on psutil
# ==============================================================================

import sys
import os

def _mem_gb():
    """Current process RSS in GB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            return -1.0


def _system_mem_report():
    """System-wide memory snapshot string."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (f"System: {vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f} GB "
                f"({vm.percent}% used)  available={vm.available / (1024**3):.1f} GB")
    except ImportError:
        return "System: psutil not available"


# ==============================================================================
# Patched compute_optimal_thresholds_per_pair
# ==============================================================================

def compute_optimal_thresholds_per_pair(
    sessions: List[Dict],
    config: PipelineConfig,
    sample_size: int = 150,
    test_thresholds: Optional[np.ndarray] = None,
    target_quality: float = 0.95,
    verbose: bool = True,
    probe_first_pair: bool = False,
    pair_callback=None,
) -> Dict:
    """
    Compute optimal threshold for each session pair using ONLY descriptor distance.
    """
    empty_result = {
        'N_values': [], 'tau_values': [],
        'test_thresholds': np.array([]), 'per_pair_qualities': np.array([]),
        'pair_names': [], 'example_matches': None,
        'example_ref_centroids': None, 'example_tgt_centroids': None,
        'n_matches_per_threshold': np.array([]),
        'n_filtered_per_threshold': np.array([]),
        'reference_sizes': [],
    }

    if len(sessions) < 2:
        return empty_result

    if test_thresholds is None:
        test_thresholds = np.linspace(0.0, 1.0, 50)

    N_values, tau_values = [], []
    all_qualities, pair_names = [], []
    all_n_matches, all_n_filtered, reference_sizes = [], [], []
    example_matches = example_ref_centroids = example_tgt_centroids = None
    best_match_count = 0

    n_pairs_total = len(sessions) * (len(sessions) - 1) // 2
    pair_idx = 0

    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            pair_idx += 1
            sess_i, sess_j = sessions[i], sessions[j]

            if sess_i['n_neurons'] >= sess_j['n_neurons']:
                ref_data, tgt_data = sess_i, sess_j
            else:
                ref_data, tgt_data = sess_j, sess_i

            N_avg = (ref_data['n_neurons'] + tgt_data['n_neurons']) / 2
            pair_name = f"{ref_data['session_name']}->{tgt_data['session_name']}"

            is_first = (pair_idx == 1)

            if is_first and probe_first_pair:
                logger.info(f"\n  📍 First-pair probe: {pair_name}")
                logger.info(f"     ref={ref_data['n_quads']:,} quads  "
                            f"tgt={tgt_data['n_quads']:,} quads  "
                            f"N_avg={N_avg:.0f}")
                print(f"     [DBG] Worker PID={os.getpid()}  "
                      f"RSS={_mem_gb():.2f} GB  {_system_mem_report()}",
                      file=sys.stderr, flush=True)

            pair_sample = _build_single_pair_sample(ref_data, tgt_data, sample_size)
            if pair_sample is None:
                continue

            pair_names.append(pair_name)
            min_size = min(pair_sample['ref_desc'].shape[0],
                           pair_sample['tgt_desc'].shape[0])
            reference_sizes.append(min_size)

            # ── Debug: sample memory footprint ────────────────────────────
            if is_first and probe_first_pair:
                ref_mb = pair_sample['ref_desc'].nbytes / (1024**2)
                tgt_mb = pair_sample['tgt_desc'].nbytes / (1024**2)
                cent_mb = (pair_sample['ref_centroids'].nbytes +
                           pair_sample['tgt_centroids'].nbytes) / (1024**2)
                print(f"     [DBG] Sample arrays: ref_desc={ref_mb:.1f}MB  "
                      f"tgt_desc={tgt_mb:.1f}MB  centroids={cent_mb:.1f}MB  "
                      f"RSS={_mem_gb():.2f} GB",
                      file=sys.stderr, flush=True)

            qualities, n_matches_for_pair, n_filtered_for_pair = [], [], []
            best_filtered_for_pair = []

            t_pair_start = time.time()

            # ── Plateau detection config ──────────────────────────────
            PLATEAU_WINDOW = 10        # stop after this many unchanged
            plateau_count  = 0
            last_quality   = None
            QUALITY_EPS    = 1e-6      # floating-point tolerance

            for thr_idx, thr in enumerate(test_thresholds):
                t_thr = time.time()

                # ── Periodic GC to prevent stack accumulation ─────────
                if thr_idx > 0 and thr_idx % 20 == 0:
                    gc.collect()

                # ── Debug: pre-call marker (every 10th + first 5) ─────────
                if is_first and probe_first_pair and (thr_idx < 5 or thr_idx % 10 == 0):
                    print(f"     [DBG] thr[{thr_idx+1:02d}] pre-match  thr={thr:.6f}  "
                          f"RSS={_mem_gb():.2f} GB",
                          file=sys.stderr, flush=True)

                try:
                    matches = match_quads_descriptor_only(
                        pair_sample['ref_desc'], pair_sample['ref_idx'],
                        pair_sample['tgt_desc'], pair_sample['tgt_idx'],
                        similarity_threshold=thr,
                        distance_metric=config.distance_metric,
                        top_k=1, verbose=False,
                    )
                except Exception as match_err:
                    print(f"     [DBG] thr[{thr_idx+1:02d}] MATCH ERROR: "
                          f"{type(match_err).__name__}: {match_err}",
                          file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    matches = []

                n_raw = len(matches)

                # ── Debug: post-match marker (every 10th + first 5) ───────
                if is_first and probe_first_pair and (thr_idx < 5 or thr_idx % 10 == 0):
                    print(f"     [DBG] thr[{thr_idx+1:02d}] post-match "
                          f"n_raw={n_raw}  RSS={_mem_gb():.2f} GB",
                          file=sys.stderr, flush=True)

                if matches:
                    try:
                        filtered = filter_quad_matches_by_consistency(
                            matches,
                            pair_sample['ref_centroids'],
                            pair_sample['tgt_centroids'],
                            consistency_threshold=config.consistency_threshold,
                        )
                    except Exception as filt_err:
                        print(f"     [DBG] thr[{thr_idx+1:02d}] FILTER ERROR: "
                              f"{type(filt_err).__name__}: {filt_err}",
                              file=sys.stderr, flush=True)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        sys.stderr.flush()
                        filtered = []
                    n_filt = len(filtered)
                    quality = compute_match_quality(n_raw, n_filt, min_size)
                    if len(filtered) > len(best_filtered_for_pair):
                        best_filtered_for_pair = filtered
                else:
                    n_filt, quality = 0, 0.0

                qualities.append(quality)
                n_matches_for_pair.append(n_raw)
                n_filtered_for_pair.append(n_filt)

                if is_first and probe_first_pair:
                    elapsed_thr = time.time() - t_thr
                    logger.info(
                        f"     thr[{thr_idx+1:02d}/{len(test_thresholds)}]"
                        f"  thr={thr:.4f}"
                        f"  raw={n_raw:4d}  filt={n_filt:4d}"
                        f"  quality={quality:.4f}"
                        f"  ({elapsed_thr:.2f}s)"
                    )

                # ── Plateau early exit ────────────────────────────────
                if last_quality is not None and abs(quality - last_quality) < QUALITY_EPS:
                    plateau_count += 1
                else:
                    plateau_count = 0
                last_quality = quality

                if plateau_count >= PLATEAU_WINDOW and thr_idx >= 15:
                    # Fill remaining thresholds with last known values
                    remaining = len(test_thresholds) - thr_idx - 1
                    if remaining > 0:
                        qualities.extend([quality] * remaining)
                        n_matches_for_pair.extend([n_raw] * remaining)
                        n_filtered_for_pair.extend([n_filt] * remaining)
                    if is_first and probe_first_pair:
                        logger.info(
                            f"     ✂ Plateau detected at thr[{thr_idx+1}] "
                            f"(quality={quality:.4f} unchanged for "
                            f"{PLATEAU_WINDOW} steps). Skipping {remaining} "
                            f"remaining thresholds."
                        )
                    break
                # ── End plateau check ─────────────────────────────────

            # ── Debug: pair completed ─────────────────────────────────────
            if is_first and probe_first_pair:
                print(f"     [DBG] First pair threshold loop COMPLETED "
                      f"({len(test_thresholds)} thresholds)  "
                      f"RSS={_mem_gb():.2f} GB  {_system_mem_report()}",
                      file=sys.stderr, flush=True)

            pair_time = time.time() - t_pair_start

            if is_first and probe_first_pair:
                tau_probe = find_threshold_for_quality_target(
                    test_thresholds, np.array(qualities), target_quality
                )
                est_total = pair_time * n_pairs_total
                logger.info(f"\n     ✓ First pair done in {pair_time:.1f}s")
                logger.info(f"       optimal tau = {tau_probe:.4f}")
                logger.info(f"       Est. total for {n_pairs_total} pairs: "
                            f"~{est_total:.0f}s ({est_total/60:.1f} min)")

            all_qualities.append(qualities)
            all_n_matches.append(n_matches_for_pair)
            all_n_filtered.append(n_filtered_for_pair)

            tau = find_threshold_for_quality_target(
                test_thresholds, np.array(qualities), target_quality
            )
            N_values.append(N_avg)
            tau_values.append(tau)

            if len(best_filtered_for_pair) > best_match_count:
                best_match_count = len(best_filtered_for_pair)
                example_matches = _convert_matches_to_array(best_filtered_for_pair[:50])
                example_ref_centroids = pair_sample['ref_centroids']
                example_tgt_centroids = pair_sample['tgt_centroids']

            if verbose:
                opt_idx = np.argmin(np.abs(test_thresholds - tau))
                n_at_opt = n_filtered_for_pair[opt_idx]
                rate_at_opt = n_at_opt / min_size if min_size > 0 else 0
                logger.info(
                    f"  [{pair_idx}/{n_pairs_total}] {pair_name}: "
                    f"N={N_avg:.0f}  tau={tau:.4f}  "
                    f"matches={n_at_opt}/{min_size} ({rate_at_opt:.1%})  "
                    f"({pair_time:.1f}s)"
                )

            # ── Debug: periodic memory report every 10 pairs ──────────────
            if pair_idx % 10 == 0:
                print(f"     [DBG] Pair {pair_idx}/{n_pairs_total}  "
                      f"RSS={_mem_gb():.2f} GB  "
                      f"best_matches={best_match_count}  "
                      f"{_system_mem_report()}",
                      file=sys.stderr, flush=True)

            if pair_callback is not None:
                pair_callback(pair_idx, n_pairs_total, pair_time, pair_name)

    return {
        'N_values': N_values,
        'tau_values': tau_values,
        'test_thresholds': test_thresholds,
        'per_pair_qualities': np.array(all_qualities) if all_qualities else np.array([]),
        'pair_names': pair_names,
        'example_matches': example_matches,
        'example_ref_centroids': example_ref_centroids,
        'example_tgt_centroids': example_tgt_centroids,
        'n_matches_per_threshold': np.array(all_n_matches) if all_n_matches else np.array([]),
        'n_filtered_per_threshold': np.array(all_n_filtered) if all_n_filtered else np.array([]),
        'reference_sizes': reference_sizes,
    }


def _build_single_pair_sample(
    ref_data: Dict,
    tgt_data: Dict,
    sample_size: int
) -> Optional[Dict]:
    n_ref_quads = ref_data['n_quads']
    n_tgt_quads = tgt_data['n_quads']
    if n_ref_quads == 0 or n_tgt_quads == 0:
        return None
    if 'quad_desc' not in ref_data:
        ref_data = _load_session_arrays(ref_data)
    if 'quad_desc' not in tgt_data:
        tgt_data = _load_session_arrays(tgt_data)
    ref_sel = np.random.choice(n_ref_quads, min(sample_size, n_ref_quads), replace=False)
    tgt_sel = np.random.choice(n_tgt_quads, min(sample_size, n_tgt_quads), replace=False)
    return {
        'ref_desc':      ref_data['quad_desc'][ref_sel],
        'ref_idx':       ref_data['quad_idx'][ref_sel],
        'tgt_desc':      tgt_data['quad_desc'][tgt_sel],
        'tgt_idx':       tgt_data['quad_idx'][tgt_sel],
        'ref_centroids': ref_data['centroids'],
        'tgt_centroids': tgt_data['centroids'],
    }


def _scan_sessions_fast(output_dir: str, animal_id: str) -> List[Dict]:
    """
    Scan Step 1 NPZ files for one animal and return lightweight session dicts.
    Only reads array shapes — no heavy data loading.
    Returns ONLY serialization-safe plain Python types (no numpy arrays).
    """
    step1_dir = Path(output_dir) / "step_1_results"
    pattern   = f"{animal_id}_*_centroids_quads.npz"
    npz_files = sorted(step1_dir.glob(pattern))

    sessions = []
    for f in npz_files:
        session_name = f.stem.replace("_centroids_quads", "")
        try:
            with np.load(f, allow_pickle=False) as data:
                n_neurons = int(data["centroids"].shape[0])
                qd = data.get("quad_desc", data.get("descriptors", None))
                n_quads = int(qd.shape[0]) if qd is not None else 0
        except Exception as e:
            logger.warning(f"  Could not read {f.name}: {e}")
            continue
        sessions.append({
            "session_name": session_name,       # str
            "animal_id":    animal_id,          # str
            "npz_path":     str(f),             # str
            "n_neurons":    n_neurons,          # plain int
            "n_quads":      n_quads,            # plain int
            # Heavy arrays intentionally absent — loaded lazily in worker
        })
    return sessions

def _bin_animals_by_neuron_count(
    all_sessions_by_animal: Dict[str, List[Dict]],
    bin_edges: Optional[List[float]] = None,
) -> Dict[str, List[str]]:
    """
    Groups animals by average neuron count into tiers.
    Returns {representative_animal_id: [all animal_ids in that bin]}.
    
    Default bin_edges cover 100 / 250 / 500 / 1000-neuron benchmarks:
      [0, 175)  → 100-neuron tier
      [175, 375) → 250-neuron tier
      [375, 750) → 500-neuron tier
      [750, ∞)  → 1000-neuron tier
    """
    if bin_edges is None:
        bin_edges = [0, 175, 375, 750, float('inf')]

    avg_neurons = {
        aid: float(np.mean([s['n_neurons'] for s in sess])) if sess else 0.0
        for aid, sess in all_sessions_by_animal.items()
    }

    tier_buckets: Dict[int, List[str]] = {}
    for aid, avg_n in avg_neurons.items():
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= avg_n < bin_edges[i + 1]:
                tier_buckets.setdefault(i, []).append(aid)
                break

    # Representative = animal with the most sessions (most data)
    rep_map: Dict[str, List[str]] = {}
    for bucket_aids in tier_buckets.values():
        rep = max(bucket_aids,
                  key=lambda a: len(all_sessions_by_animal.get(a, [])))
        rep_map[rep] = bucket_aids

    return rep_map

def _load_session_arrays(session: Dict) -> Dict:
    """
    Load quad_desc, quad_idx, centroids for one session on demand.
    Returns a new dict merging the lightweight metadata with the arrays.
    """
    data = np.load(session["npz_path"], allow_pickle=True)
    centroids = data["centroids"].astype(np.float32)
    quad_desc = data.get("quad_desc", data.get("descriptors")).astype(np.float32)
    quad_idx  = data["quad_idx"]
    return {**session, "centroids": centroids, "quad_desc": quad_desc, "quad_idx": quad_idx}


def auto_tune_threshold_with_scaling(
    config: PipelineConfig,
    sample_size: int = 150,
    target_quality: float = 0.95,
    test_thresholds: Optional[np.ndarray] = None,
    probe_first_pair: bool = True,
    pre_scanned_sessions: Optional[List[Dict]] = None,
    pair_callback=None,
    max_pairs_per_animal: int = None,
) -> Dict[str, float]:

    logger.info("")
    logger.info(f"DESCRIPTOR THRESHOLD CALIBRATION (sqrt(N)) FOR ANIMAL {config.animal_id}")
    logger.info("=" * 80)
    logger.info("Using descriptor distance ONLY (no geometric filters)")
    logger.info(f"  session_group_regex = {repr(config.session_group_regex)}")
    logger.info(f"  session_pair_strategy = {repr(config.session_pair_strategy)}")

    if pre_scanned_sessions is not None:
        sessions = pre_scanned_sessions
        logger.info(f"  Using {len(sessions)} pre-scanned sessions")
    else:
        t_load = time.time()
        logger.info(f"  Scanning sessions for animal {config.animal_id}...")
        sessions = _scan_sessions_fast(config.output_dir, config.animal_id)
        logger.info(f"  Found {len(sessions)} sessions in {time.time()-t_load:.2f}s")

    if not sessions:
        logger.warning(f"No data for animal {config.animal_id}")
        return {'animal_id': config.animal_id, 'C': 0.0, 'C_std': 0.0,
                'r_squared': 0.0, 'n_pairs': 0}

    logger.info(f"  quads: min={min(s['n_quads'] for s in sessions):,}  "
                f"max={max(s['n_quads'] for s in sessions):,}  "
                f"mean={int(np.mean([s['n_quads'] for s in sessions])):,}")

    if test_thresholds is None:
        test_thresholds = np.linspace(0.0, 1.0, 50)

    from collections import defaultdict
    import random
    session_groups = defaultdict(list)
    for s in sessions:
        session_groups[config._parse_group(s['session_name'])].append(s)

    logger.info(f"  {len(session_groups)} session groups")
    logger.info(f"  Sample group keys: {list(session_groups.keys())[:5]}")

    # ── Subsample sessions to cap total pairs ─────────────────────────────────
    if max_pairs_per_animal is not None and max_pairs_per_animal > 0:
        total_pairs_est = sum(
            len(grp) * (len(grp) - 1) // 2
            for grp in session_groups.values() if len(grp) >= 2
        )
        if total_pairs_est > max_pairs_per_animal:
            rng = random.Random(42)
            n_groups_with_pairs = sum(1 for grp in session_groups.values() if len(grp) >= 2)
            for gk in list(session_groups.keys()):
                grp = session_groups[gk]
                if len(grp) < 2:
                    continue
                target_per_group = max(2, max_pairs_per_animal // max(1, n_groups_with_pairs))
                # solve n*(n-1)/2 >= target_per_group for n
                n_keep = int((1 + (1 + 8 * target_per_group) ** 0.5) / 2)
                n_keep = max(2, min(n_keep, len(grp)))
                session_groups[gk] = rng.sample(grp, n_keep)
            new_total = sum(
                len(grp) * (len(grp) - 1) // 2
                for grp in session_groups.values() if len(grp) >= 2
            )
            logger.info(f"  Subsampled: {total_pairs_est} -> ~{new_total} pairs "
                        f"(max_pairs_per_animal={max_pairs_per_animal})")

    N_values, tau_values = [], []
    all_pair_results = []
    first_group = True

    for gk, grp in session_groups.items():
        if len(grp) < 2:
            continue
        n_pairs_in_group = len(grp) * (len(grp) - 1) // 2
        logger.info(f"  Group '{gk}': {len(grp)} sessions -> {n_pairs_in_group} pairs")

        pr = compute_optimal_thresholds_per_pair(
            grp, config, sample_size,
            test_thresholds=test_thresholds,
            target_quality=target_quality,
            probe_first_pair=(first_group and probe_first_pair),
            pair_callback=pair_callback,
        )
        first_group = False
        N_values.extend(pr['N_values'])
        tau_values.extend(pr['tau_values'])
        all_pair_results.append(pr)

    pair_results = all_pair_results[0] if all_pair_results else \
        compute_optimal_thresholds_per_pair([], config, sample_size)
    pair_results['N_values']   = N_values
    pair_results['tau_values'] = tau_values

    for key in ['per_pair_qualities', 'n_matches_per_threshold', 'n_filtered_per_threshold']:
        arrays = [pr[key] for pr in all_pair_results if len(pr[key]) > 0]
        pair_results[key] = np.concatenate(arrays, axis=0) if arrays else np.array([])
    pair_results['pair_names']      = [n for pr in all_pair_results for n in pr['pair_names']]
    pair_results['reference_sizes'] = [s for pr in all_pair_results for s in pr['reference_sizes']]

    if len(N_values) < 2:
        logger.warning(f"Insufficient pairs for animal {config.animal_id}")
        return {'animal_id': config.animal_id, 'C': 0.0, 'C_std': 0.0,
                'r_squared': 0.0, 'n_pairs': len(N_values)}

    C, C_std, r_squared = compute_C_value_from_pairs(N_values, tau_values)

    per_pair_qualities       = pair_results['per_pair_qualities']
    n_filtered_per_threshold = pair_results['n_filtered_per_threshold']
    n_matches_per_threshold  = pair_results['n_matches_per_threshold']

    mean_quality    = np.mean(per_pair_qualities, axis=0)       if len(per_pair_qualities) > 0       else np.array([])
    mean_n_matches  = np.mean(n_matches_per_threshold, axis=0)  if len(n_matches_per_threshold) > 0  else np.array([])
    mean_n_filtered = np.mean(n_filtered_per_threshold, axis=0) if len(n_filtered_per_threshold) > 0 else np.array([])

    if len(mean_quality) > 0:
        optimal_idx       = np.argmax(mean_quality)
        optimal_threshold = test_thresholds[optimal_idx]
    else:
        optimal_threshold, optimal_idx = 0.0, 0

    avg_ref_size = np.mean(pair_results['reference_sizes']) if pair_results['reference_sizes'] else 1
    match_rate   = mean_n_filtered[optimal_idx] / avg_ref_size if avg_ref_size > 0 else 0

    logger.info(f">>> C = {C:.4f} +/- {C_std:.4f}")
    logger.info(f"    R^2 = {r_squared:.3f}")
    logger.info(f"    From {len(N_values)} session pairs")
    logger.info(f"    Formula: tau = {C:.4f} * sqrt(N)")
    logger.info(f"    optimal_threshold = {optimal_threshold:.4f}")
    logger.info(f"    match_rate @ optimal = {match_rate:.1%}")
    logger.info(f"  Animal {config.animal_id}: C={C:.4f}  R^2={r_squared:.3f}  "
                f"optimal_threshold={optimal_threshold:.4f}  match_rate={match_rate:.1%}")

    result = {
        'animal_id': config.animal_id,
        'C': C, 'C_std': C_std, 'r_squared': r_squared,
        'n_pairs': len(N_values),
        'N_values':    [float(n) for n in N_values],
        'tau_values':  [float(t) for t in tau_values],
        'test_thresholds':        test_thresholds,
        'per_pair_qualities':     per_pair_qualities,
        'mean_quality':           mean_quality,
        'pair_names':             pair_results['pair_names'],
        'optimal_threshold':      optimal_threshold,
        'example_matches':        pair_results['example_matches'],
        'example_ref_centroids':  pair_results['example_ref_centroids'],
        'example_tgt_centroids':  pair_results['example_tgt_centroids'],
        'n_matches_per_threshold':  n_matches_per_threshold,
        'n_filtered_per_threshold': n_filtered_per_threshold,
        'mean_n_matches':           mean_n_matches,
        'mean_n_filtered':          mean_n_filtered,
        'reference_sizes':          pair_results['reference_sizes'],
    }

    _save_calibration_results(result, config)
    return result


def _save_calibration_results(result: Dict, config: PipelineConfig):
    animal_id = result['animal_id']
    logger.info(f"Saving results for animal {animal_id}...")
    calib_dir = ensure_output_dir(config.output_dir, 1.5, verbose=False)

    npz_path = calib_dir / f"{animal_id}_threshold_calibration.npz"
    save_dict = {
        'animal_id': animal_id,
        'C': result['C'], 'C_std': result['C_std'],
        'r_squared': result['r_squared'], 'n_pairs': result['n_pairs'],
        'N_values': np.array(result['N_values']),
        'tau_values': np.array(result['tau_values']),
    }
    for k in ['test_thresholds', 'per_pair_qualities', 'mean_quality']:
        if result.get(k) is not None:
            save_dict[k] = np.array(result[k])
    if result.get('pair_names') is not None:
        save_dict['pair_names'] = np.array(result['pair_names'], dtype=object)
    if 'optimal_threshold' in result:
        save_dict['optimal_threshold'] = result['optimal_threshold']
    for k in ['n_matches_per_threshold', 'n_filtered_per_threshold',
              'mean_n_matches', 'mean_n_filtered']:
        if result.get(k) is not None and len(result[k]) > 0:
            save_dict[k] = np.array(result[k])
    if result.get('reference_sizes'):
        save_dict['reference_sizes'] = np.array(result['reference_sizes'])
    if result.get('example_matches') is not None:
        save_dict['example_matches'] = np.array(result['example_matches'])
    if result.get('example_ref_centroids') is not None:
        save_dict['example_ref_centroids'] = np.array(result['example_ref_centroids'])
    if result.get('example_tgt_centroids') is not None:
        save_dict['example_tgt_centroids'] = np.array(result['example_tgt_centroids'])

    np.savez(npz_path, **save_dict)
    logger.info(f"Saved calibration data: {npz_path}")

    json_result = {
        'animal_id': result['animal_id'],
        'C': result['C'], 'C_std': result['C_std'],
        'r_squared': result['r_squared'], 'n_pairs': result['n_pairs'],
        'N_values': result['N_values'], 'tau_values': result['tau_values'],
        'optimal_threshold': result.get('optimal_threshold', 0.0),
        'pair_names': result.get('pair_names', []),
        'n_example_matches': len(result['example_matches']) if result.get('example_matches') is not None else 0,
        'avg_reference_size': float(np.mean(result['reference_sizes'])) if result.get('reference_sizes') else 0,
    }
    json_path = calib_dir / f"{animal_id}_threshold_calibration.json"
    save_json_summary(json_result, json_path, verbose=False)

    if len(result['N_values']) >= 2:
        _plot_calibration(result, calib_dir)


def _plot_calibration(result: Dict, output_dir: Path):
    N_arr   = np.array(result['N_values'])
    tau_arr = np.array(result['tau_values'])
    C       = result['C']
    sort_idx = np.argsort(N_arr)
    N_s, tau_s = N_arr[sort_idx], tau_arr[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.sqrt(N_s), tau_s, s=100, alpha=0.6, label='Measured τ')
    sqrt_range = np.linspace(np.sqrt(N_s).min(), np.sqrt(N_s).max(), 100)
    ax.plot(sqrt_range, C * sqrt_range, 'r--', linewidth=2,
            label=f'τ = {C:.3f} × √N (R²={result["r_squared"]:.3f})')
    ax.set_xlabel('√N', fontsize=12)
    ax.set_ylabel('Optimal Threshold τ', fontsize=12)
    ax.set_title(f'Threshold Calibration – Animal {result["animal_id"]}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plot_path = output_dir / f"{result['animal_id']}_threshold_calibration.png"
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved calibration plot: {plot_path}")


# ==============================================================================
# Multi-Animal Parallel Processing
# ==============================================================================

def _worker_tune_single_animal(args, log_queue=None, is_verbose=False):
    """
    Worker that runs calibration for one animal and sends a single 'done'
    message when complete. No pair-level messages -- eliminates concurrent
    pipe writes that deadlock Windows SimpleQueue.
    """
    animal_id, base_cfg_dict, sample_size, target_quality, \
        threshold_min, threshold_max, n_threshold_points, \
        pre_scanned_sessions, *_extra = args
    max_pairs_per_animal = _extra[0] if _extra else 50

    import traceback
    _n_sess_str = len(pre_scanned_sessions) if pre_scanned_sessions is not None else "None"
    print(f"[WORKER {animal_id} PID={os.getpid()}] STARTUP  "
          f"RSS={_mem_gb():.2f} GB  {_system_mem_report()}  "
          f"n_sessions={_n_sess_str}",
          file=sys.stderr, flush=True)

    class QueueHandler(logging.Handler):
        def __init__(self, q, aid, verbose):
            super().__init__()
            self.q       = q
            self.aid     = aid
            self.verbose = verbose
        def emit(self, record):
            if not self.verbose:
                return
            try:
                msg = self.format(record)
                if self.q is not None:
                    self.q.put(('log', self.aid, msg))
            except Exception:
                pass

    if log_queue is None:
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = QueueHandler(log_queue, animal_id, is_verbose)
    handler.setFormatter(logging.Formatter('%(message)s'))

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    nm_logger = logging.getLogger("neuron_mapping")
    nm_logger.handlers = []
    nm_logger.addHandler(handler)
    nm_logger.setLevel(logging.INFO)
    nm_logger.propagate = False

    _log_dir = Path(base_cfg_dict['output_dir']) / "logs_step1_5"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _step_log = _log_dir / f"{animal_id}_{os.getpid()}.log"
    _fh = logging.FileHandler(_step_log)
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root.addHandler(_fh)
    nm_logger.addHandler(_fh)

    def _reattach_if_needed():
        if handler not in nm_logger.handlers:
            nm_logger.handlers = []
            nm_logger.addHandler(handler)
            nm_logger.addHandler(_fh)
            nm_logger.propagate = False

    result = {
        'animal_id': animal_id, 'C': 0.0, 'C_std': 0.0,
        'r_squared': 0.0, 'n_pairs': 0,
    }

    try:
        cfg_dict = dict(base_cfg_dict)
        cfg_dict['animal_id'] = animal_id
        config = PipelineConfig.from_dict(cfg_dict)
        _reattach_if_needed()

        log_worker_start("threshold calibration", animal_id)
        _reattach_if_needed()

        print(f"[WORKER {animal_id}] Config loaded  RSS={_mem_gb():.2f} GB",
              file=sys.stderr, flush=True)

        test_thresholds = np.linspace(threshold_min, threshold_max, n_threshold_points)

        _last_heartbeat = [0.0]

        def _pair_heartbeat(pair_idx, n_pairs_total, pair_time, pair_name):
            import time as _t
            now = _t.time()
            if log_queue is not None and (now - _last_heartbeat[0]) > 10.0:
                _last_heartbeat[0] = now
                try:
                    log_queue.put(
                        ('heartbeat', animal_id,
                         {'pair_idx': pair_idx, 'n_pairs': n_pairs_total}),
                        timeout=2.0
                    )
                except Exception:
                    pass

        result = auto_tune_threshold_with_scaling(
            config,
            sample_size=sample_size,
            target_quality=target_quality,
            test_thresholds=test_thresholds,
            probe_first_pair=is_verbose,
            pre_scanned_sessions=pre_scanned_sessions,
            pair_callback=_pair_heartbeat,
            max_pairs_per_animal=max_pairs_per_animal,
        )

        print(f"[WORKER {animal_id}] Calibration done  "
              f"RSS={_mem_gb():.2f} GB  {_system_mem_report()}  "
              f"n_pairs={result.get('n_pairs', 0)}",
              file=sys.stderr, flush=True)

        log_worker_finish("threshold calibration", animal_id, result)

    except Exception as e:
        nm_logger.error(f"[WORKER {animal_id}] ERROR: {e}\n{traceback.format_exc()}")
        print(f"[WORKER {animal_id}] EXCEPTION  RSS={_mem_gb():.2f} GB  "
              f"{_system_mem_report()}\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)

    finally:
        print(f"[WORKER {animal_id}] FINALLY block  RSS={_mem_gb():.2f} GB",
              file=sys.stderr, flush=True)
        if log_queue is not None:
            try:
                _HEAVY_KEYS = {
                    'per_pair_qualities',
                    'n_matches_per_threshold',
                    'n_filtered_per_threshold',
                    'mean_quality',
                    'mean_n_matches',
                    'mean_n_filtered',
                    'test_thresholds',
                    'example_matches',
                    'example_ref_centroids',
                    'example_tgt_centroids',
                    'reference_sizes',
                    'N_values',
                    'tau_values',
                    'pair_names',
                }
                slim_result = {k: v for k, v in result.items()
                               if k not in _HEAVY_KEYS}
                log_queue.put(('done', animal_id, slim_result), timeout=30.0)
                print(f"[WORKER {animal_id}] sent done to queue",
                      file=sys.stderr, flush=True)
            except Exception as qe:
                print(f"[WORKER {animal_id}] slim_result put() FAILED: "
                      f"{type(qe).__name__}: {qe}",
                      file=sys.stderr, flush=True)
                try:
                    minimal = {
                        'animal_id': animal_id,
                        'C':         float(result.get('C', 0.0)),
                        'C_std':     float(result.get('C_std', 0.0)),
                        'r_squared': float(result.get('r_squared', 0.0)),
                        'n_pairs':   int(result.get('n_pairs', 0)),
                    }
                    log_queue.put(('done', animal_id, minimal), timeout=10.0)
                except Exception as qe2:
                    print(f"[WORKER {animal_id}] CRITICAL: minimal put() also failed: "
                          f"{type(qe2).__name__}: {qe2}",
                          file=sys.stderr, flush=True)

    return result

def run_global_tuning_all_animals(
    input_dir: str,
    output_dir: str,
    sample_size: int = 1000,
    target_quality: float = 0.95,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    n_threshold_points: int = 50,
    processes: Optional[int] = None,
    verbose: bool = True,
    session_callback: Optional[callable] = None,
    session_group_regex: Optional[str] = None,
    session_pair_strategy: str = 'consecutive',
    max_pairs_per_animal: int = 10,
    **kwargs,
) -> List[Dict]:
    import time as _time
    print(f"[CANARY] run_global_tuning_all_animals v2 called", file=sys.stderr, flush=True)

    # Skip if calibration results already exist
    step1_5_dir = Path(output_dir) / "step_1_5_results"
    summary_json = step1_5_dir / "all_animals_summary.json"
    if summary_json.exists():
        logger.info(f"[STEP 1.5] Calibration results already exist → {summary_json.name}, skipping.")
        logger.info(f"[STEP 1.5] Delete {summary_json} to force recalibration.")
        import json
        with open(summary_json) as f:
            existing = json.load(f)
        return existing if isinstance(existing, list) else []

    base_config = PipelineConfig(
        input_dir=input_dir, output_dir=output_dir,
        verbose=verbose, animal_id=None, skip_existing=True,
    )
    base_cfg_dict = base_config.to_dict()
    _main_log_dir = Path(output_dir) / "logs_step1_5"
    _main_log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(_main_log_dir, verbose=verbose)
    base_cfg_dict['calib_threshold_min']      = threshold_min
    base_cfg_dict['calib_threshold_max']      = threshold_max
    base_cfg_dict['calib_threshold_n_points'] = n_threshold_points
    if session_group_regex is not None:
        base_cfg_dict['session_group_regex'] = session_group_regex
    base_cfg_dict['session_pair_strategy'] = session_pair_strategy

    step1_dir = Path(output_dir) / "step_1_results"
    npz_files = sorted(step1_dir.glob("*_centroids_quads.npz"))
    if not npz_files:
        raise RuntimeError(f"No *_centroids_quads.npz files found in {step1_dir}")

    animal_ids = sorted({f.name.split("_")[0] for f in npz_files})
    logger.info(f"Discovered {len(animal_ids)} animals: {animal_ids}")
    logger.info(f"Threshold sweep: {threshold_min:.3f} to {threshold_max:.3f} ({n_threshold_points} points)")
    logger.info(f"sample_size={sample_size}  max_pairs_per_animal={max_pairs_per_animal}")

    logger.info(f"Scanning sessions in main process...")
    all_sessions_by_animal = {}
    for aid in animal_ids:
        sessions = _scan_sessions_fast(output_dir, aid)
        clean_sessions = [
            {
                'session_name': str(s['session_name']),
                'animal_id':    str(s['animal_id']),
                'npz_path':     str(s['npz_path']),
                'n_neurons':    int(s['n_neurons']),
                'n_quads':      int(s['n_quads']),
            }
            for s in sessions
        ]
        all_sessions_by_animal[aid] = clean_sessions
        n_pairs_est = len(clean_sessions) * (len(clean_sessions) - 1) // 2
        logger.info(f"  Animal {aid}: {len(clean_sessions)} sessions, ~{n_pairs_est} pairs "
                    f"(will cap at {max_pairs_per_animal})")

    use_bins = kwargs.get('use_neuron_count_bins', False)
    bin_edges = kwargs.get('neuron_count_bin_edges', None)

    if use_bins and len(animal_ids) > 1:
        rep_map = _bin_animals_by_neuron_count(all_sessions_by_animal, bin_edges)
        rep_ids = list(rep_map.keys())
        logger.info(f"Neuron-count binning: {len(animal_ids)} animals → "
                    f"{len(rep_ids)} representative(s): {rep_ids}")
        for rep, members in rep_map.items():
            avg_n = np.mean([s['n_neurons']
                             for s in all_sessions_by_animal[rep]])
            logger.info(f"  Rep {rep} (avg N={avg_n:.0f}) covers: {members}")
        calibrate_ids = rep_ids
    else:
        rep_map = {aid: [aid] for aid in animal_ids}
        calibrate_ids = animal_ids

    args_list = [
        (aid, base_cfg_dict, sample_size, target_quality,
         threshold_min, threshold_max, n_threshold_points,
         all_sessions_by_animal[aid], max_pairs_per_animal)
        for aid in calibrate_ids          # ← only reps, not all animals
    ]

    max_workers = compute_max_workers(cpu_fraction=0.25) if processes is None else processes
    logger.info(f"Using {max_workers} parallel worker(s)")

    ctx = multiprocessing.get_context('spawn')
    log_queue = ctx.Queue()

    def _n_sessions(a):
        return len(all_sessions_by_animal.get(a[0], []))
    args_sorted = sorted(args_list, key=_n_sessions)

    processes_list = []
    submit_times   = {}
    active_count   = 0

    for idx, a in enumerate(args_sorted):
        aid        = a[0]
        is_verbose = (idx == 0)

        while active_count >= max_workers:
            _time.sleep(0.5)
            active_count = sum(1 for _, p in processes_list if p.is_alive())

        p = ctx.Process(
            target=_worker_tune_single_animal,
            args=(a,),
            kwargs={'log_queue': log_queue, 'is_verbose': is_verbose},
        )
        p.start()
        processes_list.append((aid, p))
        submit_times[aid] = _time.time()
        active_count += 1
        label = " (verbose)" if is_verbose else ""
        logger.info(f"  Launched worker for animal {aid} (pid={p.pid}){label}")

    print(f"[MAIN] All {len(args_sorted)} workers launched. n_total={len(args_list)}  n_workers={len(processes_list)}", file=sys.stderr, flush=True)
    print(f"[MAIN] animal_ids={animal_ids}", file=sys.stderr, flush=True)
    print(f"[MAIN] Duplicate animal_ids? {len(animal_ids) != len(set(animal_ids))}", file=sys.stderr, flush=True)

    results_map  = {}
    animal_times = []
    n_done       = 0
    n_total      = len(args_list)
    n_workers    = len(args_list)

    print(f"[MAIN] Entering collection loop. Waiting for {n_total} done messages...", file=sys.stderr, flush=True)

    while n_done < n_total:
        print(f"[MAIN] Top of loop: n_done={n_done}/{n_total}  results_map keys={sorted(results_map.keys())}", file=sys.stderr, flush=True)
        try:
            print(f"[MAIN] Calling log_queue.get(timeout=5.0)...", file=sys.stderr, flush=True)
            msg_type, aid, payload = log_queue.get(timeout=5.0)
            print(f"[MAIN] Got message: type={msg_type}  aid={aid}", file=sys.stderr, flush=True)

            if msg_type == 'log':
                logger.info(payload)

            elif msg_type == 'heartbeat':
                pair_idx = payload.get('pair_idx', 0)
                n_pairs  = payload.get('n_pairs', 1)
                frac     = pair_idx / max(n_pairs, 1)
                logger.info(
                    f"[MAIN] Animal {aid} in progress: "
                    f"pair {pair_idx}/{n_pairs} ({frac:.0%})"
                )
                if session_callback is not None:
                    try:
                        session_callback(n_done, n_total, 0.0)
                    except Exception:
                        pass

            elif msg_type == 'done':
                print(f"[MAIN] Processing DONE for aid={aid}  already_in_map={aid in results_map}", file=sys.stderr, flush=True)
                elapsed = _time.time() - submit_times[aid]
                animal_times.append(elapsed)
                results_map[aid] = payload
                n_done += 1

                avg_animal    = float(np.mean(animal_times))
                rem_animals   = n_total - n_done
                est_total_rem = avg_animal * min(rem_animals, n_workers)
                amins, asecs  = divmod(int(est_total_rem), 60)
                atime_str     = f"{amins}m {asecs}s" if amins > 0 else f"{asecs}s"

                logger.info(
                    f"[MAIN] Animal {aid} done in {elapsed:.1f}s  "
                    f"({n_done}/{n_total} animals)  "
                    f"est remaining: ~{atime_str}"
                )
                print(f"[MAIN] n_done now={n_done}/{n_total}  results_map size={len(results_map)}", file=sys.stderr, flush=True)

                if session_callback is not None:
                    try:
                        session_callback(n_done, n_total, elapsed)
                    except Exception as cb_err:
                        logger.warning(f"session_callback error: {cb_err}")

            else:
                print(f"[MAIN] WARNING: unknown msg_type={msg_type}  aid={aid}", file=sys.stderr, flush=True)

        except queue_module.Empty:
            pending = [(aid, p) for aid, p in processes_list
                       if aid not in results_map]
            alive   = [(aid, p) for aid, p in pending if p.is_alive()]
            dead    = [(aid, p) for aid, p in pending if not p.is_alive()]

            elapsed_strs = []
            for aid, p in alive:
                secs = int(_time.time() - submit_times.get(aid, _time.time()))
                mins, s = divmod(secs, 60)
                elapsed_strs.append(f"{aid}({mins}m{s:02d}s)")

            print(
                f"[MAIN] TIMEOUT (Empty): {n_done}/{n_total} done  "
                f"| alive({len(alive)}): {' '.join(elapsed_strs) or 'none'}  "
                f"| dead_without_result({len(dead)}): {[a for a,_ in dead]}",
                file=sys.stderr, flush=True
            )

            for aid, p in dead:
                print(f"[MAIN] Dead worker {aid}: exitcode={p.exitcode}", file=sys.stderr, flush=True)
                logger.error(
                    f"[MAIN] Worker {aid} died unexpectedly "
                    f"(exitcode={p.exitcode})"
                )
                results_map[aid] = {
                    'animal_id': aid, 'C': 0.0, 'C_std': 0.0,
                    'r_squared': 0.0, 'n_pairs': 0,
                }
                n_done += 1
                if session_callback is not None:
                    try:
                        session_callback(n_done, n_total, 0.0)
                    except Exception:
                        pass

        except BaseException as e:
            print(f"[MAIN] BaseException in queue read: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            logger.error(
                f"[MAIN] Queue read failed: {type(e).__name__}: {e}  "
                f"({n_done}/{n_total} done). Switching to poll mode."
            )
            for aid, p in processes_list:
                if aid not in results_map:
                    print(f"[MAIN] poll-mode joining {aid}...", file=sys.stderr, flush=True)
                    p.join(timeout=300.0)
                    print(f"[MAIN] poll-mode joined {aid}  exitcode={p.exitcode}", file=sys.stderr, flush=True)
                    results_map[aid] = {
                        'animal_id': aid, 'C': 0.0, 'C_std': 0.0,
                        'r_squared': 0.0, 'n_pairs': 0,
                    }
                    n_done += 1
                    if session_callback is not None:
                        try:
                            session_callback(n_done, n_total, 0.0)
                        except Exception:
                            pass
            break

    print(f"[MAIN] Exited collection loop: n_done={n_done}  n_total={n_total}  results_map size={len(results_map)}", file=sys.stderr, flush=True)

    if session_callback is not None:
        try:
            session_callback(n_total, n_total, 0.0)
        except Exception:
            pass

    print(f"[MAIN] Starting p.join() for all workers...", file=sys.stderr, flush=True)
    for aid, p in processes_list:
        print(f"[MAIN] Joining {aid} (alive={p.is_alive()})...", file=sys.stderr, flush=True)
        try:
            p.join(timeout=10.0)
        except Exception as e:
            logger.warning(f"[MAIN] Error joining worker {aid}: {e}")
        print(f"[MAIN] Joined {aid}  exitcode={p.exitcode}  still_alive={p.is_alive()}", file=sys.stderr, flush=True)
        if p.is_alive():
            logger.warning(f"[MAIN] Worker {aid} still alive after join, terminating")
            try:
                p.terminate()
                p.join(timeout=5.0)
            except Exception:
                pass

    print(f"[MAIN] All workers joined. Building results list...", file=sys.stderr, flush=True)

    # Build a lookup: rep_aid -> its result
    rep_results = {
        aid: results_map.get(aid, {'animal_id': aid, 'C': 0.0,
                                    'C_std': 0.0, 'r_squared': 0.0,
                                    'n_pairs': 0})
        for aid in calibrate_ids
    }

    # Build full results list in original animal_ids order
    member_to_rep = {
        member: rep
        for rep, members in rep_map.items()
        for member in members
    }
    results = []
    for aid in animal_ids:
        rep = member_to_rep[aid]
        r = dict(rep_results[rep])   # shallow copy
        r['animal_id'] = aid         # stamp correct animal_id
        results.append(r)
        if aid != rep:
            logger.info(f"  Animal {aid}: reusing calibration from rep {rep} "
                        f"(C={r['C']:.4f}, tau=C*sqrt(N))")

    step1_5_dir  = ensure_output_dir(output_dir, 1.5, verbose=False)
    summary_path = step1_5_dir / "all_animals_summary.json"
    logger.info(f"output_dir={output_dir}, step1_5_dir={step1_5_dir}")

    json_safe_results = []
    for r in results:
        json_safe_results.append({
            'animal_id':         r.get('animal_id'),
            'C':                 float(r.get('C', 0.0)),
            'C_std':             float(r.get('C_std', 0.0)),
            'r_squared':         float(r.get('r_squared', 0.0)),
            'n_pairs':           int(r.get('n_pairs', 0)),
            'optimal_threshold': float(r.get('optimal_threshold', 0.0)),
            'N_values':          [float(x) for x in r.get('N_values', [])],
            'tau_values':        [float(x) for x in r.get('tau_values', [])],
            'pair_names':        list(r.get('pair_names', [])) if r.get('pair_names') is not None else [],
        })

    print(f"[MAIN] Saving summary JSON...", file=sys.stderr, flush=True)
    save_json_summary(json_safe_results, summary_path)
    print(f"[MAIN] run_global_tuning_all_animals RETURNING {len(results)} results", file=sys.stderr, flush=True)
    return results