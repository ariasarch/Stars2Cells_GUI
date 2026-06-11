#!/usr/bin/env python3
"""
cellreg_batch_all_perturbed.py

Same as cellreg_batch_all.py but runs CellReg against PERTURBED spatial
footprints (from generate_spatial_footprints_perturbed.py).

All outputs go to separate directories so nothing overwrites existing data:
  - CellReg results → CellReg_{n}_batch_perturbed_{preset}/cellreg_outputs/
  - JSON summary    → CellReg_{n}_batch_perturbed_{preset}/all_results_{n}n.json

Change PERTURBATION_PRESET below to match whichever preset you generated.
"""

import os
import sys
import json
import time
import traceback
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found — install with: pip install tqdm")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from CellReg_python import run_cellreg


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                                 ║
# ╠═══════════════════════════════════════════════════════════════════════╣

BASE_DIR = Path(r"C:\Users\ariAccount\Desktop\Stars2CellsPaper")
TIER = "C"  
# NEURON_COUNTS = [100, 250, 500, 1000]
NEURON_COUNTS = [1000]

SEEDS = [0, 1, 2, 3, 4]

_ALL_CONDITIONS = [
    "A_rot", "A_trans", "A_combined",
    "B_dropout", "B_drift", "B_rot", "B_combined",
    "C_walk",
]
_A_ONLY = ["A_rot", "A_trans", "A_combined"]

# CONDITIONS_BY_NC = {
#     100:  _ALL_CONDITIONS,
#     250:  _ALL_CONDITIONS,
#     500:  _ALL_CONDITIONS,
#     1000: _A_ONLY,
# }

CONDITIONS_BY_NC = {
    1000: ["B_dropout", "B_drift", "B_rot", "B_combined"] if TIER == "B" else ["C_walk"],
}

N_SESSIONS = 5

MICRONS_PER_PIXEL   = 1.0
MAXIMAL_DISTANCE_UM = 20.0
P_SAME_THRESHOLD    = 0.5

MAX_WORKERS = 50

ABORT_AFTER_CONSECUTIVE_ERRORS = 10

# ╠═══════════════════════════════════════════════════════════════════════╣
# ║  THIS IS THE ONLY NEW THING — pick which perturbation preset to use   ║
# ║                                                                       ║
# ║  Must match the folder name from generate_spatial_footprints_perturbed║
# ║    'gentle'   → spatial_mapping_perturbed_gentle/                     ║
# ║    'moderate'  → spatial_mapping_perturbed_moderate/                  ║
# ║    'harsh'     → spatial_mapping_perturbed_harsh/                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

PERTURBATION_PRESET = "best_case"   # ← change this to match your footprints

# FOOTPRINT_SUBFOLDER = f"spatial_mapping_perturbed_{PERTURBATION_PRESET}"
FOOTPRINT_SUBFOLDER = f"spatial_mapping"

# ═══════════════════════════════════════════════════════════════════════════
# Suppress stdout
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


# ═══════════════════════════════════════════════════════════════════════════
# Scoring helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def score_pair_simple(matched_ref, matched_tgt, ref_base_ids, tgt_base_ids):
    ref_base_ids = np.asarray(ref_base_ids)
    tgt_base_ids = np.asarray(tgt_base_ids)
    matched_ref  = np.asarray(matched_ref, dtype=int)
    matched_tgt  = np.asarray(matched_tgt, dtype=int)

    shared_ids = set(ref_base_ids) & set(tgt_base_ids)
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
    nonshared_caught = sum(
        1 for i, bid in enumerate(ref_base_ids) if bid not in shared_ids and i in matched_ref_set
    ) + sum(
        1 for i, bid in enumerate(tgt_base_ids) if bid not in shared_ids and i in matched_tgt_set
    )
    n_nonshared = sum(1 for bid in ref_base_ids if bid not in shared_ids) + \
                  sum(1 for bid in tgt_base_ids if bid not in shared_ids)
    tn = n_nonshared - nonshared_caught

    prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                n_shared=len(shared_ids),
                precision=prec, recall=recall, f1=f1)


def cmap_to_pairs(cmap, si, sj):
    rows = np.where((cmap[:, si] > 0) & (cmap[:, sj] > 0))[0]
    return cmap[rows, si], cmap[rows, sj]


def aggregate_pairs(pair_scores_dict):
    tp = sum(s['tp'] for s in pair_scores_dict.values())
    fp = sum(s['fp'] for s in pair_scores_dict.values())
    fn = sum(s['fn'] for s in pair_scores_dict.values())
    tn = sum(s['tn'] for s in pair_scores_dict.values())
    prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=recall, f1=f1)


def fmt_scores(d):
    if d is None:
        return None
    return {k: (round(v, 4) if isinstance(v, float) and not (v != v) else
                (None if isinstance(v, float) and (v != v) else v))
            for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Worker function
# ═══════════════════════════════════════════════════════════════════════════

def run_combo(animal, condition, seed,
              data_dir, gt_path, s2c_dir, cr_save_dir,
              footprint_subfolder,
              microns_per_pixel, maximal_distance_um, p_same_threshold,
              n_sessions):
    combo_key = f"{animal}_{condition}_seed{seed}"
    cr_save_path = Path(cr_save_dir) / f"{combo_key}_cellreg.npy"
    t0 = time.time()

    try:
        file_paths = [
            Path(data_dir) / f"{animal}_{sess}__{condition}__seed{seed}.npy"
            for sess in range(1, n_sessions + 1)
        ]
        for fp in file_paths:
            if not fp.exists():
                return combo_key, {"error": f"Missing file: {fp}"}, time.time() - t0, False

        # ── Run CellReg OR load cached results ──
        was_cached = cr_save_path.exists()
        if was_cached:
            results = np.load(cr_save_path, allow_pickle=True).item()
        else:
            with suppress_stdout():
                results = run_cellreg(
                    file_paths          = file_paths,
                    microns_per_pixel   = microns_per_pixel,
                    maximal_distance_um = maximal_distance_um,
                    p_same_threshold    = p_same_threshold,
                    # ── THIS IS THE KEY CHANGE ──
                    footprint_dir       = str(Path(data_dir) / footprint_subfolder),
                )
            np.save(cr_save_path, results, allow_pickle=True)

        cmap = results['cell_to_index_map']

        # ── Ground-truth scoring ──
        gt = np.load(gt_path, allow_pickle=True).item()
        gt_base_ids = []
        for fp in file_paths:
            key = fp.name
            if key not in gt:
                return combo_key, {"error": f"GT key missing: {key}"}, time.time() - t0, was_cached
            gt_base_ids.append(gt[key]['ground_truth_base_ids'])

        sess_names = [fp.name.replace(".npy", "") for fp in file_paths]
        pairs      = [(i, j) for i in range(n_sessions)
                              for j in range(n_sessions) if i < j]

        # ── CellReg pairwise scores ──
        cr_pair_scores = {}
        for si, sj in pairs:
            matched_ref, matched_tgt = cmap_to_pairs(cmap, si, sj)
            cr_pair_scores[f"sess{si+1}_x_sess{sj+1}"] = fmt_scores(
                score_pair_simple(matched_ref, matched_tgt,
                                  gt_base_ids[si], gt_base_ids[sj])
            )
        cellreg_agg = fmt_scores(aggregate_pairs(cr_pair_scores))

        # ── Stars2Cells pairwise scores ──
        s2c_pair_scores = {}
        s2c_agg         = None
        s2c_dir_path    = Path(s2c_dir)

        if s2c_dir_path.exists():
            all_sweeps = sorted(s2c_dir_path.glob("*_sweep.npz"))
            relevant   = [f for f in all_sweeps
                          if any(s in f.stem for s in sess_names)]
            sweep_index = {}
            for sf in relevant:
                data  = np.load(sf, allow_pickle=False)
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
                    continue
                mref = data['matched_ref_indices']
                mtgt = data['matched_tgt_indices']
                if (tgt_name, ref_name) in sweep_index and \
                   (ref_name, tgt_name) not in sweep_index:
                    mref, mtgt = mtgt, mref
                s2c_pair_scores[f"sess{si+1}_x_sess{sj+1}"] = fmt_scores(
                    score_pair_simple(mref, mtgt,
                                      gt_base_ids[si], gt_base_ids[sj])
                )
            if s2c_pair_scores:
                s2c_agg = fmt_scores(aggregate_pairs(s2c_pair_scores))

        return combo_key, {
            "animal":    animal,
            "condition": condition,
            "seed":      seed,
            "footprint_source": footprint_subfolder,
            "cellreg": {
                "n_clusters":      int(cmap.shape[0]),
                "mean_cell_score": round(float(np.nanmean(results['cell_scores'])), 4),
                "model_mse":       round(float(results['model_mse']), 4),
                "intersection_um": round(float(results['intersection_um']), 3),
                "aggregate":       cellreg_agg,
                "per_pair":        cr_pair_scores,
            },
            "stars2cells": {
                "aggregate": s2c_agg,
                "per_pair":  s2c_pair_scores,
            },
        }, time.time() - t0, was_cached

    except Exception as e:
        return combo_key, {
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }, time.time() - t0, False


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(all_results):
    def _collect(results, pipeline):
        by_cond = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
        overall  = {"precision": [], "recall": [], "f1": []}
        for combo_key, entry in results.items():
            if "error" in entry:
                continue
            agg = entry.get(pipeline, {}).get("aggregate")
            if agg is None:
                continue
            cond = entry["condition"]
            for metric in ["precision", "recall", "f1"]:
                v = agg.get(metric)
                if v is not None and v == v:
                    by_cond[cond][metric].append(v)
                    overall[metric].append(v)
        summary = {}
        for cond, m in by_cond.items():
            summary[cond] = {
                k: round(float(np.mean(v)), 4) if v else None
                for k, v in m.items()
            }
        summary["_overall"] = {
            k: round(float(np.mean(v)), 4) if v else None
            for k, v in overall.items()
        }
        return summary

    return {
        "cellreg":     _collect(all_results, "cellreg"),
        "stars2cells": _collect(all_results, "stars2cells"),
    }


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h{int(m)}m{int(s)}s"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    grand_t0 = time.time()

    print(f"\n  Perturbation preset: {PERTURBATION_PRESET}")
    print(f"  Footprint subfolder: {FOOTPRINT_SUBFOLDER}")

    for nc_idx, n_neurons in enumerate(NEURON_COUNTS):

        conditions = CONDITIONS_BY_NC[n_neurons]

        data_dir    = BASE_DIR / f"Stars2Cells_Benchmark_{n_neurons}n_Tier_{TIER}"
        # ── OUTPUT DIR includes the preset name so it never collides ──
        output_dir  = BASE_DIR / f"CellReg_{n_neurons}n_Tier_{TIER}_perturbed_{PERTURBATION_PRESET}"
        gt_path     = data_dir / "ground_truth.npy"
        s2c_dir     = data_dir / "step_3_results"
        cr_save_dir = output_dir / "cellreg_outputs"
        animals     = [n_neurons + i for i in range(1, 9)]

        # ── Verify the perturbed footprint folder exists ──
        fp_dir = data_dir / FOOTPRINT_SUBFOLDER
        if not fp_dir.exists():
            print(f"\n  ✗ Footprint dir not found: {fp_dir}")
            print(f"    Run generate_spatial_footprints_perturbed.py first.")
            print(f"    Skipping {n_neurons}n\n")
            continue

        output_dir.mkdir(exist_ok=True, parents=True)
        cr_save_dir.mkdir(exist_ok=True, parents=True)

        all_combos = list(product(animals, conditions, SEEDS))

        n_cached = sum(
            1 for a, c, s in all_combos
            if (cr_save_dir / f"{a}_{c}_seed{s}_cellreg.npy").exists()
        )
        n_to_run = len(all_combos) - n_cached

        print(f"\n{'━'*60}")
        print(f"  {n_neurons}n [{PERTURBATION_PRESET}]  │  {len(all_combos)} combos  │  "
              f"{n_cached} cached  │  {n_to_run} to run")
        print(f"  GT: {gt_path}")
        print(f"  Footprints: {fp_dir}")
        print(f"  Conditions: {', '.join(conditions)}")
        print(f"{'━'*60}")

        if not gt_path.exists():
            print(f"  ✗ Ground truth not found — skipping {n_neurons}n\n")
            continue

        fixed = dict(
            data_dir            = str(data_dir),
            gt_path             = str(gt_path),
            s2c_dir             = str(s2c_dir),
            cr_save_dir         = str(cr_save_dir),
            footprint_subfolder = FOOTPRINT_SUBFOLDER,
            microns_per_pixel   = MICRONS_PER_PIXEL,
            maximal_distance_um = MAXIMAL_DISTANCE_UM,
            p_same_threshold    = P_SAME_THRESHOLD,
            n_sessions          = N_SESSIONS,
        )

        all_results = {}
        timings     = []
        nc_t0       = time.time()
        aborted     = False
        first_error = None

        n_workers = min(MAX_WORKERS, len(all_combos))
        n_ok = n_err = n_cache_hits = 0

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_combo, a, c, s, **fixed): (a, c, s)
                for a, c, s in all_combos
            }

            pbar = tqdm(
                total=len(all_combos),
                desc=f"{n_neurons}n [{PERTURBATION_PRESET}]",
                unit="combo",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                ncols=110,
            )

            for future in as_completed(futures):
                combo_key, result, elapsed, was_cached = future.result()
                all_results[combo_key] = result
                had_error = "error" in result

                if had_error:
                    n_err += 1
                    if first_error is None:
                        first_error = (combo_key, result.get("error", "???"))
                else:
                    n_ok += 1
                if was_cached:
                    n_cache_hits += 1

                timings.append((combo_key, elapsed, was_cached, had_error))

                n_done = n_ok + n_err
                if (n_ok == 0
                        and n_err >= ABORT_AFTER_CONSECUTIVE_ERRORS
                        and n_done >= ABORT_AFTER_CONSECUTIVE_ERRORS):
                    pbar.close()
                    aborted = True

                    print(f"\n  ✗ ABORTING {n_neurons}n — first "
                          f"{ABORT_AFTER_CONSECUTIVE_ERRORS} results were "
                          f"ALL errors (0 ok, {n_err} err)")
                    print(f"\n  First error ({first_error[0]}):")
                    print(f"    {first_error[1]}")
                    print(f"  Waiting for in-flight workers to finish...")
                    break

                fresh_times = [t for _, t, c, _ in timings if not c]
                avg_fresh = np.mean(fresh_times) if fresh_times else 0

                pbar.set_postfix_str(
                    f"ok={n_ok} err={n_err} cache={n_cache_hits} "
                    f"last={fmt_time(elapsed)} avg={fmt_time(avg_fresh)}"
                )
                pbar.update(1)

            if not aborted:
                pbar.close()

        if aborted:
            print(f"\n  Skipping to next neuron count...\n")
            continue

        nc_elapsed = time.time() - nc_t0

        summary = compute_summary(all_results)

        fresh_times = [t for _, t, c, _ in timings if not c]
        cache_times = [t for _, t, c, _ in timings if c]

        print(f"\n  {n_neurons}n [{PERTURBATION_PRESET}] complete in {fmt_time(nc_elapsed)}")
        print(f"  {n_ok} ok  │  {n_err} errors  │  {n_cache_hits} from cache")
        if fresh_times:
            print(f"  Fresh combos: avg={fmt_time(np.mean(fresh_times))}  "
                  f"min={fmt_time(min(fresh_times))}  "
                  f"max={fmt_time(max(fresh_times))}")
        if cache_times:
            print(f"  Cached combos: avg={fmt_time(np.mean(cache_times))}")

        if first_error and n_ok > 0:
            print(f"\n  First error ({first_error[0]}):")
            print(f"    {first_error[1]}")

        cr_summ  = summary.get("cellreg", {})
        s2c_summ = summary.get("stars2cells", {})
        cr_overall  = cr_summ.get("_overall", {})
        s2c_overall = s2c_summ.get("_overall", {})

        def _f1str(d):
            v = d.get("f1") if d else None
            return f"{v*100:.1f}%" if v is not None else "  N/A"

        print(f"\n  {'Condition':<15} {'CellReg F1':>11} {'S2C F1':>11}")
        print(f"  {'─'*39}")
        for cond in conditions:
            print(f"  {cond:<15} {_f1str(cr_summ.get(cond)):>11} "
                  f"{_f1str(s2c_summ.get(cond)):>11}")
        print(f"  {'─'*39}")
        print(f"  {'OVERALL':<15} {_f1str(cr_overall):>11} "
              f"{_f1str(s2c_overall):>11}")

        final = {
            "_meta": {
                "n_neurons":            n_neurons,
                "perturbation_preset":  PERTURBATION_PRESET,
                "footprint_subfolder":  FOOTPRINT_SUBFOLDER,
                "n_combos":             len(all_combos),
                "n_ok":                 n_ok,
                "n_errors":             n_err,
                "n_cached":             n_cache_hits,
                "wall_time_sec":        round(nc_elapsed, 1),
                "animals":              animals,
                "seeds":                SEEDS,
                "conditions":           conditions,
                "n_sessions":           N_SESSIONS,
                "microns_per_pixel":    MICRONS_PER_PIXEL,
                "maximal_distance_um":  MAXIMAL_DISTANCE_UM,
                "p_same_threshold":     P_SAME_THRESHOLD,
                "data_dir":             str(data_dir),
                "gt_path":              str(gt_path),
                "s2c_dir":              str(s2c_dir),
                "cr_save_dir":          str(cr_save_dir),
                "footprint_dir":        str(fp_dir),
            },
            "_summary": summary,
            "results":  all_results,
        }

        json_path = output_dir / f"all_results_{n_neurons}n_perturbed_{PERTURBATION_PRESET}.json"
        with open(json_path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n  Saved → {json_path}")

    grand_elapsed = time.time() - grand_t0
    print(f"\n{'━'*60}")
    print(f"  ALL DONE [{PERTURBATION_PRESET}] — total wall time: {fmt_time(grand_elapsed)}")
    print(f"{'━'*60}")