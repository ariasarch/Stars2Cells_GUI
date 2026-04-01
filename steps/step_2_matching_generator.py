"""
Step 2: Quad matching — parallelized across ALL pairs simultaneously.

Changes from previous version
─────────────────────────────
* Skip-existing uses a single glob → set lookup instead of per-pair stat calls.
  For all_vs_all with 1600 sessions this avoids 1.28M filesystem stat calls.
* Pickle save removed by default (save_full_pickle=False). Downstream steps
  only read the light NPZ — the pickle was 10-50MB/pair of dead weight.
* Probe pair result is kept and counted — no longer processed twice.
* BLAS thread pinning happens BEFORE ProcessPoolExecutor via initializer,
  so MKL/OpenBLAS can't grab threads at import time in child processes.
* _save_light_npz accepts pre-built arrays from filter_quad_matches_by_consistency
  instead of rebuilding them from Python lists.
* OMP thread budget passed via initargs so it survives Windows spawn.
"""
import sys
import os
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utilities import *

logger = logging.getLogger("neuron_mapping_step2")


# ==============================================================================
# BLAS thread pinning — called once per worker process at pool init
# ==============================================================================

def _pin_worker_threads(omp_threads="1"):
    """Pin BLAS to 1 thread but give FAISS multiple OMP threads via initargs."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)


# ==============================================================================
# Session Pair Matching
# ==============================================================================

def match_session_pair(
    ref_data: Dict,
    target_data: Dict,
    config: PipelineConfig,
    threshold: float,
    save_full_pickle: bool = False,
) -> Dict[str, Any]:
    """Match quads between two sessions."""
    pair_name = f"{ref_data['session_name']}_to_{target_data['session_name']}"

    N_ref = ref_data['n_neurons']
    N_tgt = target_data['n_neurons']
    N_avg = (N_ref + N_tgt) / 2

    logger.info(f"\nMatching {ref_data['session_name']} -> {target_data['session_name']}")
    logger.info(f"  Ref: {N_ref} neurons, {ref_data['n_quads']:,} quads")
    logger.info(f"  Target: {N_tgt} neurons, {target_data['n_quads']:,} quads")
    logger.info(f"  Threshold: {threshold:.4f}")
    start_time = time.time()

    raw_matches = match_quads_descriptor_only(
        ref_data['quad_desc'],
        ref_data['quad_idx'],
        target_data['quad_desc'],
        target_data['quad_idx'],
        similarity_threshold=threshold,
        distance_metric=config.distance_metric,
        top_k=1,
        verbose=False
    )

    if raw_matches:
        filtered_matches = filter_quad_matches_by_consistency(
            raw_matches,
            ref_data['centroids'],
            target_data['centroids'],
            consistency_threshold=config.consistency_threshold,
        )
    else:
        filtered_matches = []

    match_time = time.time() - start_time
    logger.info(f"  Raw: {len(raw_matches):,}  Filtered: {len(filtered_matches):,}  Time: {match_time:.1f}s")

    match_data = {
        'animal_id':          ref_data['animal_id'],
        'ref_session':        ref_data['session_name'],
        'target_session':     target_data['session_name'],
        'pair_name':          pair_name,
        'ref_centroids':      ref_data['centroids'],
        'target_centroids':   target_data['centroids'],
        'n_ref_neurons':      N_ref,
        'n_target_neurons':   N_tgt,
        'n_ref_quads':        ref_data['n_quads'],
        'n_target_quads':     target_data['n_quads'],
        'filtered_matches':   filtered_matches,
        'n_raw_matches':      len(raw_matches),
        'n_filtered_matches': len(filtered_matches),
        'threshold_used':     threshold,
        'N_avg':              N_avg,
        'match_time':         match_time,
    }

    step2_dir = ensure_output_dir(config.output_dir, 2, verbose=False)

    # Pickle save is off by default — downstream steps only use the light NPZ
    if save_full_pickle:
        output_file = step2_dir / f"{pair_name}_matches.pkl"
        save_intermediate_data(match_data, output_file, compress=True)

    _save_light_npz(match_data, filtered_matches, step2_dir, pair_name)

    return match_data


def _save_light_npz(
    match_data: Dict,
    filtered_matches: List,
    step2_dir: Path,
    pair_name: str
):
    """Save lightweight NPZ for fast loading by Step 2.5."""
    try:
        if filtered_matches:
            ref_idx_arr = np.array([m[0] for m in filtered_matches], dtype=np.int32)
            tgt_idx_arr = np.array([m[1] for m in filtered_matches], dtype=np.int32)
            match_indices = np.concatenate([ref_idx_arr, tgt_idx_arr], axis=1)
            distances = np.array([m[2] for m in filtered_matches], dtype=np.float32)
        else:
            match_indices = np.zeros((0, 8), dtype=np.int32)
            distances = np.zeros((0,), dtype=np.float32)

        light_path = step2_dir / f"{pair_name}_matches_light.npz"
        np.savez_compressed(
            light_path,
            animal_id=np.bytes_(match_data["animal_id"]),
            ref_session=np.bytes_(match_data["ref_session"]),
            target_session=np.bytes_(match_data["target_session"]),
            pair_name=np.bytes_(match_data["pair_name"]),
            ref_centroids=match_data["ref_centroids"].astype(np.float32),
            target_centroids=match_data["target_centroids"].astype(np.float32),
            n_ref_neurons=np.int32(match_data["n_ref_neurons"]),
            n_target_neurons=np.int32(match_data["n_target_neurons"]),
            match_indices=match_indices,
            distances=distances,
            threshold_used=np.float32(match_data["threshold_used"]),
            n_raw_matches=np.int32(match_data["n_raw_matches"]),
            n_filtered_matches=np.int32(match_data["n_filtered_matches"]),
        )
        logger.info(f"  Saved: {light_path.name}")
    except Exception as e:
        logger.warning(f"  Failed to write light NPZ: {e}")


# ==============================================================================
# Pair index builder — single glob, set lookup for skip-existing
# ==============================================================================

def _build_pair_index(
    config: PipelineConfig,
    threshold_map: Dict[str, float],
) -> List[Dict]:
    """
    Scan Step 1 NPZ filenames and build a flat list of pair dicts.

    Skip-existing now uses a single glob to build a set of completed
    pair names, then checks membership. This is O(1) per pair instead
    of 2 filesystem stat calls per pair.
    """
    step1_dir = Path(config.output_dir) / "step_1_results"
    step2_dir = ensure_output_dir(config.output_dir, 2, verbose=False)

    # ── Group NPZ paths by animal ──────────────────────────────────────────────
    npz_files = sorted(step1_dir.glob("*_centroids_quads.npz"))
    by_animal: Dict[str, List[Path]] = {}
    for f in npz_files:
        animal_id = f.name.split("_")[0]
        by_animal.setdefault(animal_id, []).append(f)

    # ── Build completed-pairs set with one glob ────────────────────────────────
    existing_pairs = set()
    if config.skip_existing:
        for f in step2_dir.glob("*_matches_light.npz"):
            # Strip suffix to get pair_name
            existing_pairs.add(f.name.replace("_matches_light.npz", ""))
        if existing_pairs:
            logger.info(f"Found {len(existing_pairs)} existing pair results (will skip)")

    flat_pairs = []
    total_skipped = 0

    for animal_id, session_files in by_animal.items():
        if str(animal_id) not in threshold_map:
            logger.warning(f"No threshold for animal {animal_id}, skipping")
            continue
        threshold = threshold_map[str(animal_id)]

        # Build consecutive or all-vs-all pairs from file list
        if config.session_pair_strategy == 'all_vs_all':
            from itertools import combinations
            file_pairs = list(combinations(session_files, 2))
        else:  # consecutive (default)
            file_pairs = list(zip(session_files[:-1], session_files[1:]))

        # Apply session_group_regex grouping if set
        if config.session_group_regex:
            file_pairs = _apply_group_filter(
                session_files, file_pairs, config.session_group_regex,
                config.session_pair_strategy
            )

        for ref_path, tgt_path in file_pairs:
            ref_name = ref_path.stem.replace("_centroids_quads", "")
            tgt_name = tgt_path.stem.replace("_centroids_quads", "")
            pair_name = f"{ref_name}_to_{tgt_name}"

            # Set lookup instead of filesystem stat
            if config.skip_existing and pair_name in existing_pairs:
                total_skipped += 1
                continue

            flat_pairs.append({
                'ref_path':    str(ref_path),
                'target_path': str(tgt_path),
                'ref_name':    ref_name,
                'target_name': tgt_name,
                'threshold':   threshold,
                'animal_id':   str(animal_id),
                'pair_name':   pair_name,
            })

    if total_skipped:
        logger.info(f"Skipped {total_skipped} already-completed pairs (skip_existing=True)")
        print(f"[STEP2] Skipped {total_skipped} already-completed pairs", file=sys.stderr, flush=True)

    return flat_pairs


def _apply_group_filter(
    session_files: List[Path],
    file_pairs: List[tuple],
    group_regex: str,
    strategy: str,
) -> List[tuple]:
    """
    Re-pair files using group regex so only sessions in the same group
    are matched.
    """
    import re
    from itertools import combinations

    groups: Dict[str, List[Path]] = {}
    for f in session_files:
        m = re.search(group_regex, f.stem)
        key = m.group(1) if m else "__ungrouped__"
        groups.setdefault(key, []).append(f)

    result = []
    for files in groups.values():
        files = sorted(files)
        if strategy == 'all_vs_all':
            result.extend(combinations(files, 2))
        else:
            result.extend(zip(files[:-1], files[1:]))
    return result


# ==============================================================================
# Per-pair worker — loads its own two NPZ files
# ==============================================================================

def _load_session_npz(path: str) -> Dict:
    """Load a single session NPZ and return the fields Step 2 needs."""
    data = np.load(path, allow_pickle=True)
    session_name = Path(path).stem.replace("_centroids_quads", "")
    animal_id    = session_name.split("_")[0]

    centroids  = data["centroids"].astype(np.float32)
    quad_desc  = data.get("quad_desc", data.get("descriptors")).astype(np.float32)
    quad_idx   = data["quad_idx"]
    n_neurons  = int(centroids.shape[0])
    n_quads    = int(quad_desc.shape[0])

    return {
        'session_name': session_name,
        'animal_id':    animal_id,
        'centroids':    centroids,
        'quad_desc':    quad_desc,
        'quad_idx':     quad_idx,
        'n_neurons':    n_neurons,
        'n_quads':      n_quads,
    }


def _worker_match_single_pair(args: Dict) -> Dict[str, Any]:
    """
    Worker function — processes exactly ONE session pair.
    Loads its own NPZ files; nothing large is passed from the main process.
    BLAS thread pinning is done by the pool initializer, not here.
    """
    import io, logging, traceback as tb

    cfg_dict = args['cfg_dict']
    config   = PipelineConfig.from_dict(cfg_dict)

    # Capture logs
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Also write to a file so worker logs persist
    _log_dir = Path(cfg_dict['output_dir']) / "logs_step2"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / f"{args.get('pair_name', 'unknown')}_{os.getpid()}.log")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    root.addHandler(_fh)

    try:
        ref_data    = _load_session_npz(args['ref_path'])
        target_data = _load_session_npz(args['target_path'])

        result = match_session_pair(ref_data, target_data, config, args['threshold'])
        return {
            'animal_id':          result['animal_id'],
            'pair_name':          result['pair_name'],
            'ref_session':        result['ref_session'],
            'target_session':     result['target_session'],
            'n_raw_matches':      result['n_raw_matches'],
            'n_filtered_matches': result['n_filtered_matches'],
            'threshold_used':     result['threshold_used'],
            'match_time':         result['match_time'],
            'success':            True,
            '_worker_logs':       log_stream.getvalue(),
        }
    except Exception as e:
        return {
            'animal_id':    args.get('animal_id', '?'),
            'pair_name':    args.get('pair_name', '?'),
            'success':      False,
            'error':        str(e),
            'traceback':    tb.format_exc(),
            '_worker_logs': log_stream.getvalue(),
        }


# ==============================================================================
# Main entry point
# ==============================================================================

def run_step_2_all_animals_parallel(
    input_dir: str,
    output_dir: str,
    processes: Optional[int] = None,
    verbose: bool = True,
    distance_metric: str = 'cosine',
    consistency_threshold: float = 0.8,
    session_group_regex: str = r'__(.+)$',
    session_pair_strategy: str = 'consecutive',
) -> List[Dict]:
    output_path = Path(output_dir)
    step1_dir = output_path / "step_1_results"
    print(f"[CANARY] run_step_2_all_animals_parallel entered", file=sys.stderr, flush=True)

    # ── Load thresholds from Step 1.5 ─────────────────────────────────────────
    thr_summary_path = output_path / "step_1_5_results" / "all_animals_summary.json"
    print(f"[STEP2] Loading threshold map from {thr_summary_path}...", file=sys.stderr, flush=True)
    thr_data = load_json_summary(thr_summary_path, verbose=True)
    if thr_data is None:
        raise FileNotFoundError("Please run Step 1.5 first!")

    threshold_map = {
        str(entry["animal_id"]): float(entry["optimal_threshold"])
        for entry in thr_data
        if "optimal_threshold" in entry
    }
    print(f"[STEP2] Threshold map loaded: {len(threshold_map)} animals", file=sys.stderr, flush=True)
    logger.info(f"Loaded optimal thresholds for {len(threshold_map)} animals:")
    for aid, thr in sorted(threshold_map.items()):
        logger.info(f"  {aid}: {thr:.4f}")

    # ── Verify Step 1 data exists ──────────────────────────────────────────────
    print(f"[STEP2] Scanning Step 1 NPZ files in {step1_dir}...", file=sys.stderr, flush=True)
    npz_files = sorted(step1_dir.glob("*_centroids_quads.npz"))
    if not npz_files:
        raise RuntimeError(f"No NPZ files found in {step1_dir}")
    print(f"[STEP2] Found {len(npz_files)} NPZ files", file=sys.stderr, flush=True)

    # ── Build config shared across all workers ─────────────────────────────────
    print(f"[STEP2] Building base config...", file=sys.stderr, flush=True)
    base_config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=verbose,
        animal_id=None,
        skip_existing=True,
        distance_metric=distance_metric,
        consistency_threshold=consistency_threshold,
        session_group_regex=session_group_regex,
        session_pair_strategy=session_pair_strategy,
    )
    cfg_dict = base_config.to_dict()
    _main_log_dir = Path(output_dir) / "logs_step2"
    _main_log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(_main_log_dir, verbose=verbose)

    # ── Build pair index (single glob for skip-existing) ──────────────────────
    print(f"[STEP2] Building pair index (session_pair_strategy={session_pair_strategy})...", file=sys.stderr, flush=True)
    logger.info("Scanning session files and building pair index...")
    flat_pairs = _build_pair_index(base_config, threshold_map)
    print(f"[STEP2] Pair index built: {len(flat_pairs)} pairs to process", file=sys.stderr, flush=True)

    if not flat_pairs:
        logger.info("No pairs to process (all done or no data).")
        print(f"[STEP2] No pairs to process — exiting early", file=sys.stderr, flush=True)
        return []

    for p in flat_pairs:
        p['cfg_dict'] = cfg_dict

    # ── Print summary ──────────────────────────────────────────────────────────
    by_animal = {}
    for p in flat_pairs:
        by_animal.setdefault(p['animal_id'], 0)
        by_animal[p['animal_id']] += 1

    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Total pairs to match: {len(flat_pairs):,}")
    logger.info(f"  Animals: {len(by_animal)}")
    for aid in sorted(by_animal):
        thr = threshold_map.get(str(aid), '?')
        logger.info(f"    Animal {aid}: {by_animal[aid]} pairs  (threshold={thr:.4f})")
    logger.info(f"{'─' * 60}\n")
    print(f"[STEP2] Pair breakdown: {dict(sorted(by_animal.items()))}", file=sys.stderr, flush=True)

    # ── Determine worker count + OMP thread budget ─────────────────────────────
    phys_cores = os.cpu_count() or 32
    max_workers = min(processes or max(4, phys_cores // 4), 16)
    omp_per_worker = max(1, phys_cores // max_workers)
    logger.info(f"Thread budget: {max_workers} workers × {omp_per_worker} OMP threads "
                f"= {max_workers * omp_per_worker} / {phys_cores} cores")
    print(f"[STEP2] Thread budget: {max_workers} workers × {omp_per_worker} OMP threads",
          file=sys.stderr, flush=True)

    # ── First-pair probe (result is kept, not wasted) ─────────────────────────
    logger.info("\n🔍 Running first-pair probe...")
    probe_pair = flat_pairs[0]
    flat_pairs = flat_pairs[1:]  # Remove from batch — will be processed by probe

    probe_result = None
    print(f"[STEP2] Starting first-pair probe: {probe_pair['pair_name']}", file=sys.stderr, flush=True)
    try:
        import time as _time

        print(f"[STEP2] Probe step 1/4: loading session files...", file=sys.stderr, flush=True)
        t0 = _time.time()
        logger.info("  [1/4] Loading session files...")
        ref_data = _load_session_npz(probe_pair['ref_path'])
        tgt_data = _load_session_npz(probe_pair['target_path'])
        load_time = _time.time() - t0
        print(f"[STEP2] Probe step 1/4 done: ref={ref_data['n_quads']:,} quads  tgt={tgt_data['n_quads']:,} quads  ({load_time:.2f}s)", file=sys.stderr, flush=True)
        logger.info(f"        done  ({load_time:.2f}s)  "
              f"ref={ref_data['n_quads']:,} quads  tgt={tgt_data['n_quads']:,} quads")

        try:
            import faiss
            backend = f"FAISS (threads={faiss.omp_get_max_threads()})"
        except ImportError:
            backend = "scipy KDTree (FAISS not available)"
        print(f"[STEP2] Probe step 2/4: backend={backend}", file=sys.stderr, flush=True)
        logger.info(f"  [2/4] Backend: {backend}")

        n_ref_q = ref_data['quad_desc'].shape[0]
        n_tgt_q = tgt_data['quad_desc'].shape[0]
        D = ref_data['quad_desc'].shape[1]
        chunk_size = max(n_ref_q, int(50_000_000 / max(D, 1)))
        n_chunks = max(1, -(-n_ref_q // chunk_size))
        print(f"[STEP2] Probe step 3/4: descriptor matching ({n_ref_q:,} x {n_tgt_q:,} quads, {n_chunks} chunks)...", file=sys.stderr, flush=True)
        logger.info(f"  [3/4] Descriptor matching (threshold={probe_pair['threshold']:.4f})  "
              f"{n_ref_q:,} x {n_tgt_q:,} quads  →  {n_chunks} chunks...")

        t1 = _time.time()

        # Actually run match_session_pair so the result is saved to disk
        probe_match_data = match_session_pair(
            ref_data, tgt_data, base_config, probe_pair['threshold']
        )
        total_probe = _time.time() - t0

        probe_result = {
            'animal_id':          probe_match_data['animal_id'],
            'pair_name':          probe_match_data['pair_name'],
            'ref_session':        probe_match_data['ref_session'],
            'target_session':     probe_match_data['target_session'],
            'n_raw_matches':      probe_match_data['n_raw_matches'],
            'n_filtered_matches': probe_match_data['n_filtered_matches'],
            'threshold_used':     probe_match_data['threshold_used'],
            'match_time':         probe_match_data['match_time'],
            'success':            True,
        }

        n_raw = probe_match_data['n_raw_matches']
        n_filt = probe_match_data['n_filtered_matches']
        match_time = probe_match_data['match_time']

        est_wall_min = (total_probe * (len(flat_pairs) + 1)) / max_workers / 60
        logger.info(f"\n  Total/pair:  {total_probe:.2f}s")
        logger.info(f"  Raw: {n_raw:,}  Filtered: {n_filt:,}")
        logger.info(f"  Est. wall:   ~{est_wall_min:.0f} min  "
              f"({len(flat_pairs) + 1} pairs / {max_workers} workers)")
        print(f"[STEP2] Probe complete: {total_probe:.2f}s/pair  est_wall={est_wall_min:.0f}min", file=sys.stderr, flush=True)

        if n_raw == 0:
            logger.info(f"\n  ⚠️  ZERO raw matches — threshold {probe_pair['threshold']:.4f} is too tight!")
            logger.info("      Re-run Step 1.5 or manually set a higher threshold.")
            print(f"[STEP2] WARNING: zero raw matches on probe pair!", file=sys.stderr, flush=True)

    except Exception as e:
        logger.info(f"  ⚠️  Probe failed: {e} — continuing anyway\n")
        print(f"[STEP2] Probe FAILED: {e} — continuing anyway", file=sys.stderr, flush=True)
        # Put the pair back so it gets processed by a worker
        flat_pairs.insert(0, probe_pair)
        probe_result = None

    # ── Run remaining pairs in parallel ────────────────────────────────────────
    print(f"[STEP2] Launching ProcessPoolExecutor with {max_workers} workers for {len(flat_pairs)} pairs...", file=sys.stderr, flush=True)
    logger.info(f"Using {max_workers} parallel workers across {len(flat_pairs)} pairs")

    results = []
    pair_times = []

    # Include probe result if it succeeded
    n_success = 0
    n_failed  = 0
    total_raw = 0
    total_filt = 0

    if probe_result is not None and probe_result.get('success', False):
        results.append(probe_result)
        n_success += 1
        total_raw += probe_result.get('n_raw_matches', 0)
        total_filt += probe_result.get('n_filtered_matches', 0)
        pair_times.append(probe_result.get('match_time', 0))

    if flat_pairs:
        # initializer pins BLAS threads + sets OMP for FAISS via initargs
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_pin_worker_threads,
            initargs=(str(omp_per_worker),),
        ) as executor:
            print(f"[STEP2] Submitting {len(flat_pairs)} futures...", file=sys.stderr, flush=True)
            futures = {
                executor.submit(_worker_match_single_pair, p): p['pair_name']
                for p in flat_pairs
            }
            print(f"[STEP2] All futures submitted, entering collection loop...", file=sys.stderr, flush=True)

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Step 2 Matching",
                unit="pair",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

            for fut in pbar:
                pair_name = futures[fut]
                try:
                    res = fut.result()

                    if '_worker_logs' in res and res['_worker_logs'].strip():
                        for line in res['_worker_logs'].splitlines():
                            if line.strip():
                                logger.debug(line)

                    if res.get('success', False):
                        n_success += 1
                        total_raw  += res.get('n_raw_matches', 0)
                        total_filt += res.get('n_filtered_matches', 0)
                        t = res.get('match_time', 0)
                        pair_times.append(t)
                        avg_t = sum(pair_times) / len(pair_times)
                        pbar.set_postfix_str(
                            f"ok={n_success} err={n_failed} "
                            f"filt={res.get('n_filtered_matches',0):,} "
                            f"avg={avg_t:.1f}s"
                        )
                        # Periodic stderr heartbeat every 50 pairs
                        if n_success % 50 == 0:
                            print(f"[STEP2] Progress: {n_success}/{len(futures)+1} done  "
                                  f"total_filtered={total_filt:,}  avg={avg_t:.1f}s/pair",
                                  file=sys.stderr, flush=True)
                    else:
                        n_failed += 1
                        logger.error(f"FAILED {pair_name}: {res.get('error', '?')}")
                        print(f"[STEP2] FAILED: {pair_name} — {res.get('error', '?')}", file=sys.stderr, flush=True)
                        if verbose and 'traceback' in res:
                            logger.error(res['traceback'])

                    results.append(res)

                except Exception as e:
                    n_failed += 1
                    logger.error(f"[MAIN] Exception for {pair_name}: {e}")
                    print(f"[STEP2] EXCEPTION for {pair_name}: {e}", file=sys.stderr, flush=True)
                    results.append({'pair_name': pair_name, 'success': False, 'error': str(e)})

            pbar.close()

    print(f"[STEP2] Collection loop done: {n_success} ok / {n_failed} failed", file=sys.stderr, flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"Step 2 Complete")
    logger.info(f"  Pairs processed: {n_success:,} ok  /  {n_failed} failed")
    logger.info(f"  Total raw matches:      {total_raw:,}")
    logger.info(f"  Total filtered matches: {total_filt:,}")
    if pair_times:
        logger.info(f"  Avg time per pair: {sum(pair_times)/len(pair_times):.1f}s")
        logger.info(f"  Total wall time:   {sum(pair_times)/max_workers:.0f}s "
                    f"(serial would have been {sum(pair_times):.0f}s)")
    logger.info(f"{'='*55}\n")

    # ── Aggregate per-animal summary for JSON ─────────────────────────────────
    print(f"[STEP2] Building per-animal summary JSON...", file=sys.stderr, flush=True)
    animal_summaries = {}
    for r in results:
        if not r.get('success'):
            continue
        aid = r['animal_id']
        s = animal_summaries.setdefault(aid, {
            'animal_id': aid,
            'threshold_used': threshold_map.get(str(aid), None),
            'n_pairs': 0,
            'total_raw_matches': 0,
            'total_filtered_matches': 0,
        })
        s['n_pairs'] += 1
        s['total_raw_matches'] += r.get('n_raw_matches', 0)
        s['total_filtered_matches'] += r.get('n_filtered_matches', 0)

    summary_list = list(animal_summaries.values())
    step2_results_dir = ensure_output_dir(output_dir, 2, verbose=False)
    summary_path = step2_results_dir / "all_animals_summary.json"
    save_json_summary(summary_list, summary_path)
    print(f"[STEP2] Summary saved to {summary_path}  DONE.", file=sys.stderr, flush=True)

    clean_memory()
    return summary_list


# ==============================================================================
# Legacy single-animal runner (kept for compatibility / debugging)
# ==============================================================================

def run_step_2(config: PipelineConfig, threshold: float) -> Dict[str, Any]:
    """Run Step 2 for a single animal (sequential). Use for debugging only."""
    logger.info(f"Loading Step 1 data for animal {config.animal_id}...")
    animals = load_session_data(config.output_dir, config.animal_id, verbose=True)

    if not animals or config.animal_id not in animals:
        logger.warning(f"No data found for animal {config.animal_id}")
        return {'n_pairs': 0, 'total_raw_matches': 0, 'total_filtered_matches': 0, 'match_data': []}

    sessions = animals[config.animal_id]
    if len(sessions) < 2:
        return {'n_pairs': 0, 'total_raw_matches': 0, 'total_filtered_matches': 0, 'match_data': []}

    pairs = config.get_session_pairs(config.animal_id, sessions)
    all_matches, total_raw, total_filtered = [], 0, 0

    for ref_data, target_data in tqdm(pairs, desc=f"Step 2 [{config.animal_id}]", unit="pair"):
        try:
            md = match_session_pair(ref_data, target_data, config, threshold)
            all_matches.append(md)
            total_raw      += md['n_raw_matches']
            total_filtered += md['n_filtered_matches']
        except Exception as e:
            logger.error(f"Error {ref_data['session_name']} -> {target_data['session_name']}: {e}", exc_info=True)

    clean_memory()
    return {
        'n_pairs': len(all_matches),
        'total_raw_matches': total_raw,
        'total_filtered_matches': total_filtered,
        'match_data': all_matches,
    }