#!/usr/bin/env python3
"""
Stars2Cells Pipeline — Standalone Runner with Progress Dialogs

Hard-code your paths below and run from the S2C project root:
    python run_s2c_pipeline.py

Uses Qt progress dialogs identical to the GUI — each step pops up a
progress window with ETA, then closes when that step finishes.
"""

import sys
import os
import time
import logging
import threading
import queue
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIG — edit these
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_DIR  = r"C:\Users\ariAccount\Desktop\Stars2CellsPaper\Stars2Cells_Benchmark_1000n_Tier_C"
OUTPUT_DIR = r"C:\Users\ariAccount\Desktop\Stars2CellsPaper\Stars2Cells_Benchmark_1000n_Tier_C"

# Which steps to run (set False to skip)
RUN_STEP_1   = True
RUN_STEP_1_5 = True
RUN_STEP_2   = True
RUN_STEP_2_5 = True
RUN_STEP_3   = True

# Optional: restrict to a single animal (None = process all)
ANIMAL_ID = None  # e.g. "M123"

# ─── Step 1 (Quad Generation) ────────────────────────────────────────────────
SESSION_FILENAME_REGEX      = r'^([A-Za-z0-9_]+?)_(\d+)(.*?)\.npy$'
KNN_K                       = 15
MAX_TRIANGLES_PER_DIAGONAL  = 25
QUAD_KEEP_FRACTION          = 1.0
DIAGONAL_RNG_SEED           = 42
N_WORKERS                   = 8
SKIP_EXISTING               = True

# ─── Step 1.5 (Threshold Calibration) ────────────────────────────────────────
CALIB_SAMPLE_SIZE        = 1000
CALIB_TARGET_QUALITY     = 0.95
CALIB_THRESHOLD_MIN      = 0.0
CALIB_THRESHOLD_MAX      = 0.02      
CALIB_N_THRESHOLD_POINTS = 50
MAX_PAIRS_PER_ANIMAL     = 10
SESSION_GROUP_REGEX      = r'__(.+)$'
SESSION_PAIR_STRATEGY    = 'consecutive'

# ─── Step 2 (Descriptor Matching) ────────────────────────────────────────────
DISTANCE_METRIC          = 'cosine'
CONSISTENCY_THRESHOLD    = 0.8

# ─── Step 2.5 (RANSAC Geometric Filtering) ───────────────────────────────────
RANSAC_MAX_RESIDUAL      = 5.0
RANSAC_ITERATIONS        = 1000
RANSAC_MIN_INLIER_RATIO  = 0.05
RANSAC_ALLOW_SCALING     = False

# ─── Step 3 (Hungarian Matching + Track Consolidation) ────────────────────────
HUNGARIAN_COST_MIN       = 0.0
HUNGARIAN_COST_MAX       = 2319.0
HUNGARIAN_COST_STEPS     = 20
USE_QUAD_VOTING          = True

# ─── Parallelism ─────────────────────────────────────────────────────────────
PROCESSES = None


# ═══════════════════════════════════════════════════════════════════════════════
# Qt bootstrap — must happen before any other Qt import
# ═══════════════════════════════════════════════════════════════════════════════

from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import Qt

_qapp = QApplication.instance() or QApplication(sys.argv)


# ═══════════════════════════════════════════════════════════════════════════════
# Progress-bar helper
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_eta(t0: float, current: int, total: int) -> str:
    elapsed = time.time() - t0
    if current <= 0:
        return "calculating…"
    avg = elapsed / current
    remaining = avg * (total - current)
    m, s = divmod(int(remaining), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h}h {m}m"
    if m:   return f"{m}m {s}s"
    return f"{s}s"


def _fmt_elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


class StepProgressContext:
    """
    Context manager that pops up a QProgressDialog, runs a blocking function
    in a daemon thread, and pumps Qt events + a callback / file-monitor
    queue until done.

    For steps with a native ``session_callback`` (Steps 1 & 1.5):
        with StepProgressContext("Step 1", n) as ctx:
            result = ctx.run_blocking(func, ..., session_callback=ctx.callback)

    For steps without one (Steps 2, 2.5, 3) — pass monitor_dir/pattern
    and progress is inferred from new output files:
        with StepProgressContext("Step 2", n,
                                 monitor_dir=path, monitor_pattern="*.npz") as ctx:
            result = ctx.run_blocking(func, ...)
    """

    def __init__(self, label: str, total: int, *,
                 monitor_dir: Path = None, monitor_pattern: str = None):
        self.label = label
        self.total = max(total, 1)
        self.monitor_dir = monitor_dir
        self.monitor_pattern = monitor_pattern
        self._q: queue.Queue = queue.Queue()
        self._result = None
        self._error = None
        self._done = threading.Event()
        self._t0 = None

        self._dlg = QProgressDialog(f"Starting {label}…", None, 0, self.total)
        self._dlg.setWindowTitle(label)
        self._dlg.setWindowModality(Qt.ApplicationModal)
        self._dlg.setMinimumDuration(0)
        self._dlg.setMinimumWidth(420)
        self._dlg.setCancelButton(None)
        self._dlg.setValue(0)

    def callback(self, current, total, _time_arg):
        """Thread/process-safe — drops (current, total) into a queue."""
        self._q.put((current, total))

    def __enter__(self):
        self._t0 = time.time()
        self._dlg.show()
        QApplication.processEvents()
        return self

    def __exit__(self, *exc):
        self._dlg.close()
        return False

    def run_blocking(self, fn, *args, **kwargs):
        """Spawn fn in a thread, pump Qt events until it returns."""
        def _worker():
            try:
                self._result = fn(*args, **kwargs)
            except Exception as e:
                self._error = e
            finally:
                self._done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Snapshot existing files for monitor mode
        initial_mtimes = {}
        if self.monitor_dir and self.monitor_pattern:
            for f in self.monitor_dir.glob(self.monitor_pattern):
                try:
                    initial_mtimes[f.name] = f.stat().st_mtime
                except OSError:
                    pass

        last_monitor = 0.0

        while not self._done.is_set():
            # ── Drain callback queue ──────────────────────────────────────
            try:
                while True:
                    current, total = self._q.get_nowait()
                    if total > 0 and total != self.total:
                        self.total = total
                        self._dlg.setMaximum(total)
                    self._update_label(current)
            except queue.Empty:
                pass

            # ── File-monitor fallback (poll every 1 s) ────────────────────
            now = time.time()
            if (self.monitor_dir and self.monitor_pattern
                    and now - last_monitor > 1.0):
                last_monitor = now
                new_count = 0
                for f in self.monitor_dir.glob(self.monitor_pattern):
                    try:
                        if (f.name not in initial_mtimes
                                or f.stat().st_mtime > initial_mtimes[f.name]):
                            new_count += 1
                    except OSError:
                        pass
                if new_count > 0:
                    self._update_label(new_count, suffix="files written")

            QApplication.processEvents()
            self._done.wait(timeout=0.1)

        # Final drain + close
        try:
            while True:
                current, _ = self._q.get_nowait()
                self._update_label(current)
        except queue.Empty:
            pass

        self._dlg.setValue(self.total)
        QApplication.processEvents()
        t.join(timeout=30)

        if self._error is not None:
            raise self._error
        return self._result

    def _update_label(self, current: int, suffix: str = ""):
        current = min(current, self.total)
        self._dlg.setValue(current)
        parts = [
            self.label,
            f"{current}/{self.total}" + (f"  {suffix}" if suffix else ""),
            f"ETA: {_fmt_eta(self._t0, current, self.total)}    "
            f"Elapsed: {_fmt_elapsed(self._t0)}",
        ]
        self._dlg.setLabelText("\n".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_root_logging():
    fmt = logging.Formatter("[%(asctime)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def banner(step_label: str):
    print(f"\n{'#' * 80}")
    print(f"#  {step_label}")
    print(f"{'#' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup_root_logging()
    logger = logging.getLogger("s2c_runner")

    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        logger.error(f"INPUT_DIR does not exist: {INPUT_DIR}")
        sys.exit(1)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    from utilities import PipelineConfig, setup_logging

    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=True)

    config = PipelineConfig(
        input_dir=str(input_path),
        output_dir=str(output_path),
        verbose=True,
        skip_existing=SKIP_EXISTING,
        animal_id=ANIMAL_ID,
        knn_k=KNN_K,
        max_triangles_per_diagonal=MAX_TRIANGLES_PER_DIAGONAL,
        quad_keep_fraction=QUAD_KEEP_FRACTION,
        diagonal_rng_seed=DIAGONAL_RNG_SEED,
        n_workers=N_WORKERS,
        session_group_regex=SESSION_GROUP_REGEX,
        session_pair_strategy=SESSION_PAIR_STRATEGY,
        distance_metric=DISTANCE_METRIC,
        consistency_threshold=CONSISTENCY_THRESHOLD,
    )

    pipeline_t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Quad Generation
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_STEP_1:
        banner("STEP 1: Quad Generation (diagonal-first)")
        from steps.step_1_quad_generation import run_step_1_parallel

        n_sessions = len(list(input_path.glob("*.npy")))
        logger.info(f"Found {n_sessions} .npy session files")

        with StepProgressContext("Step 1 — Quad Generation", n_sessions) as ctx:
            step1_result = ctx.run_blocking(
                run_step_1_parallel,
                config,
                session_filename_regex=SESSION_FILENAME_REGEX,
                session_callback=ctx.callback,
            )

        logger.info(
            f"Step 1 done — {step1_result['n_sessions']} sessions, "
            f"{step1_result['total_quads']:,} quads, "
            f"wall {_fmt_elapsed(pipeline_t0)}"
        )
    else:
        logger.info("Step 1 skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1.5 — Threshold Calibration
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_STEP_1_5:
        banner("STEP 1.5: Threshold Calibration (sqrt-N scaling)")
        from steps.step_1_5_threshold_generation import run_global_tuning_all_animals

        step1_dir = output_path / "step_1_results"
        animal_ids = sorted({
            f.name.split("_")[0]
            for f in step1_dir.glob("*_centroids_quads.npz")
        }) if step1_dir.exists() else []
        logger.info(f"Calibrating {len(animal_ids)} animal(s): {animal_ids}")
        logger.info(f"Threshold sweep: {CALIB_THRESHOLD_MIN} → {CALIB_THRESHOLD_MAX}")

        with StepProgressContext("Step 1.5 — Threshold Calibration",
                                 max(len(animal_ids), 1)) as ctx:
            step1_5_results = ctx.run_blocking(
                run_global_tuning_all_animals,
                input_dir=str(input_path),
                output_dir=str(output_path),
                sample_size=CALIB_SAMPLE_SIZE,
                target_quality=CALIB_TARGET_QUALITY,
                threshold_min=CALIB_THRESHOLD_MIN,
                threshold_max=CALIB_THRESHOLD_MAX,
                n_threshold_points=CALIB_N_THRESHOLD_POINTS,
                processes=PROCESSES,
                verbose=True,
                session_group_regex=SESSION_GROUP_REGEX,
                session_pair_strategy=SESSION_PAIR_STRATEGY,
                max_pairs_per_animal=MAX_PAIRS_PER_ANIMAL,
                session_callback=ctx.callback,
            )

        for r in (step1_5_results or []):
            logger.info(
                f"  {r.get('animal_id')}: C={r.get('C', 0):.4f}  "
                f"R²={r.get('r_squared', 0):.3f}  "
                f"optimal_thr={r.get('optimal_threshold', 0):.4f}"
            )
    else:
        logger.info("Step 1.5 skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Descriptor Matching
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_STEP_2:
        banner("STEP 2: Descriptor Matching (FAISS / KDTree)")
        from steps.step_2_matching_generator import run_step_2_all_animals_parallel

        step2_out = output_path / "step_2_results"
        step2_out.mkdir(parents=True, exist_ok=True)

        n_step1 = len(list(step1_dir.glob("*_centroids_quads.npz"))) \
            if step1_dir.exists() else 0
        n_already_done = len(list(step2_out.glob("*_matches_light.npz")))
        est_pairs = max(n_step1 - 1 - n_already_done, 1)
        if n_already_done:
            logger.info(f"Step 2: {n_already_done} pairs already complete, "
                        f"~{est_pairs} remaining")

        step2_out = output_path / "step_2_results"
        step2_out.mkdir(parents=True, exist_ok=True)

        with StepProgressContext(
            "Step 2 — Descriptor Matching", est_pairs,
            monitor_dir=step2_out,
            monitor_pattern="*_matches_light.npz",
        ) as ctx:
            step2_results = ctx.run_blocking(
                run_step_2_all_animals_parallel,
                input_dir=str(input_path),
                output_dir=str(output_path),
                processes=PROCESSES,
                verbose=True,
                distance_metric=DISTANCE_METRIC,
                consistency_threshold=CONSISTENCY_THRESHOLD,
                session_group_regex=SESSION_GROUP_REGEX,
                session_pair_strategy=SESSION_PAIR_STRATEGY,
            )

        logger.info(f"Step 2 done — {len(step2_results)} animal summaries")
    else:
        logger.info("Step 2 skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2.5 — RANSAC Geometric Filtering
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_STEP_2_5:
        banner("STEP 2.5: RANSAC Geometric Filtering")
        from steps.step_2_5_RANSAC import run_step_2_5_ransac

        step2_dir = output_path / "step_2_results"
        n_match_files = len(list(step2_dir.glob("*_matches_light.npz"))) \
            if step2_dir.exists() else 0

        step2_5_out = output_path / "step_2_5_results"
        step2_5_out.mkdir(parents=True, exist_ok=True)

        with StepProgressContext(
            "Step 2.5 — RANSAC Filtering", max(n_match_files, 1),
            monitor_dir=step2_5_out,
            monitor_pattern="*_filtered_matches.npz",
        ) as ctx:
            step2_5_results = ctx.run_blocking(
                run_step_2_5_ransac,
                input_dir=str(input_path),
                output_dir=str(output_path),
                ransac_max_residual=RANSAC_MAX_RESIDUAL,
                ransac_iterations=RANSAC_ITERATIONS,
                ransac_min_inlier_ratio=RANSAC_MIN_INLIER_RATIO,
                ransac_allow_scaling=RANSAC_ALLOW_SCALING,
                processes=PROCESSES,
                verbose=True,
            )

        if step2_5_results:
            total_desc = sum(r['n_descriptor_matches'] for r in step2_5_results)
            total_inlier = sum(r['n_geometric_inliers'] for r in step2_5_results)
            logger.info(
                f"  Descriptors: {total_desc:,} → Inliers: {total_inlier:,} "
                f"({100 * total_inlier / max(total_desc, 1):.1f}%)"
            )
    else:
        logger.info("Step 2.5 skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Hungarian Matching + Track Consolidation
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_STEP_3:
        banner("STEP 3: Hungarian Matching + Track Consolidation")
        from steps.step_3_neuron_matching import run_step_3_final_matching

        step2_5_dir = output_path / "step_2_5_results"
        n_filtered = len(list(step2_5_dir.glob("*_filtered_matches.npz"))) \
            if step2_5_dir.exists() else 0

        step3_out = output_path / "step_3_results"
        step3_out.mkdir(parents=True, exist_ok=True)

        with StepProgressContext(
            "Step 3 — Hungarian + Tracking", max(n_filtered, 1),
            monitor_dir=step3_out,
            monitor_pattern="*_sweep.npz",
        ) as ctx:
            step3_results = ctx.run_blocking(
                run_step_3_final_matching,
                input_dir=str(input_path),
                output_dir=str(output_path),
                hungarian_cost_min=HUNGARIAN_COST_MIN,
                hungarian_cost_max=HUNGARIAN_COST_MAX,
                hungarian_cost_steps=HUNGARIAN_COST_STEPS,
                use_quad_voting=USE_QUAD_VOTING,
                processes=PROCESSES,
                verbose=True,
            )

        for r in (step3_results or []):
            if r.get('n_pairs', 0) > 0:
                logger.info(
                    f"  {r['animal_id']}: {r['n_pairs']} pairs  "
                    f"match_rate={r.get('avg_optimal_rate', 0) * 100:.1f}%  "
                    f"tracks={r.get('n_total_tracks', 0)}  "
                    f"full_length={r.get('full_length_tracks', 0)}"
                )
    else:
        logger.info("Step 3 skipped")

    # ══════════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════════
    banner("PIPELINE COMPLETE")
    logger.info(f"Total wall time: {_fmt_elapsed(pipeline_t0)}")
    logger.info(f"Results in: {output_path}")


if __name__ == "__main__":
    main()