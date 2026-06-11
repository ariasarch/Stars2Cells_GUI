#!/usr/bin/env python3
"""
generate_spatial_footprints_perturbed.py

Variant of generate_spatial_footprints.py that introduces per-session shape
perturbations to each neuron's footprint. This creates a HARDER scenario for
CellReg's spatial correlation matching — footprints now vary slightly across
sessions, as they would in real miniscope data due to:

  - Focus drift (axial shift → apparent size change)
  - Motion-correction residuals (sub-pixel warping artifacts)
  - CNMF re-estimation across sessions (different noise → different shapes)
  - Photobleaching / expression changes (intensity drift)

Perturbation levels (presets):
  'gentle'    — barely noticeable: ~0.1 px sigma, ~2° orientation, ~2% intensity
  'moderate'  — realistic session-to-session: ~0.5 px sigma, ~5°, ~5%
  'harsh'     — bad day at the scope: ~1.5 px sigma, ~15°, ~15%

Saves to spatial_mapping_perturbed/ (or custom name) alongside the original
spatial_mapping/ so you can compare CellReg performance head-to-head.

Usage:
    python generate_spatial_footprints_perturbed.py
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FootprintConfig:
    """Configuration for spatial footprint generation."""
    fov_size: int = 600
    microns_per_pixel: float = 1.0

    radius_by_n: dict = None

    eccentricity_range: tuple = (0.6, 1.0)
    orientation_jitter: bool = True
    intensity_range: tuple = (0.6, 1.0)
    radius_cv: float = 0.15

    footprint_crop_radius_factor: float = 2.5
    normalize_footprints: bool = True
    storage_mode: str = 'both'

    def __post_init__(self):
        if self.radius_by_n is None:
            self.radius_by_n = {
                100:  12.0,
                250:  10.0,
                500:   8.0,
                1000:  6.0,
            }

    def get_mean_radius(self, n_neurons: int) -> float:
        tiers = sorted(self.radius_by_n.keys())
        if n_neurons <= tiers[0]:
            return self.radius_by_n[tiers[0]]
        if n_neurons >= tiers[-1]:
            return self.radius_by_n[tiers[-1]]
        for i in range(len(tiers) - 1):
            if tiers[i] <= n_neurons <= tiers[i + 1]:
                frac = (n_neurons - tiers[i]) / (tiers[i + 1] - tiers[i])
                r_lo = self.radius_by_n[tiers[i]]
                r_hi = self.radius_by_n[tiers[i + 1]]
                return r_lo + frac * (r_hi - r_lo)
        return self.radius_by_n[tiers[-1]]


@dataclass
class PerturbationConfig:
    """
    Per-session shape perturbation settings.

    Each session gets independent random jitter on top of the animal's
    base footprint params. The jitter is drawn per-neuron per-session,
    so different neurons in the same session get different perturbations
    (as would happen with real CNMF re-estimation).

    All values are in ABSOLUTE units (pixels for sigma, radians for
    orientation, fraction for intensity) and represent the STANDARD
    DEVIATION of a zero-mean Gaussian perturbation.
    """
    # Sigma perturbation (pixels) — applied independently to major & minor axes
    sigma_jitter_std: float = 0.1       # gentle default: ~0.1 px

    # Orientation perturbation (radians)
    orientation_jitter_std: float = 0.035  # ~2 degrees

    # Peak intensity perturbation (fraction of original)
    intensity_jitter_std: float = 0.02   # ~2%

    # Output folder name suffix
    output_folder_name: str = "spatial_mapping_perturbed"

    # Human-readable label for logging
    preset_label: str = "gentle"


# ── Presets ──

PERTURBATION_PRESETS = {
    'gentle': PerturbationConfig(
        sigma_jitter_std=0.1,           # ~0.1 px — barely visible
        orientation_jitter_std=0.035,   # ~2°
        intensity_jitter_std=0.02,      # ~2%
        output_folder_name="spatial_mapping_perturbed_gentle",
        preset_label="gentle",
    ),
    'moderate': PerturbationConfig(
        sigma_jitter_std=0.5,           # ~0.5 px — realistic
        orientation_jitter_std=0.087,   # ~5°
        intensity_jitter_std=0.05,      # ~5%
        output_folder_name="spatial_mapping_perturbed_moderate",
        preset_label="moderate",
    ),
    'harsh': PerturbationConfig(
        sigma_jitter_std=1.5,           # ~1.5 px — bad day
        orientation_jitter_std=0.26,    # ~15°
        intensity_jitter_std=0.15,      # ~15%
        output_folder_name="spatial_mapping_perturbed_harsh",
        preset_label="harsh",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Core: generate base footprint templates (identical to original script)
# ═══════════════════════════════════════════════════════════════════════════

def generate_base_footprint_params(n_neurons, animal_seed, config: FootprintConfig):
    """
    Generate per-neuron footprint shape parameters for one animal.
    Same as the non-perturbed version — base shapes are FIXED.
    """
    rng = np.random.RandomState(animal_seed + 99999)

    mean_radius = config.get_mean_radius(n_neurons)

    sigma_major = rng.normal(mean_radius, mean_radius * config.radius_cv, n_neurons)
    sigma_major = np.clip(sigma_major, mean_radius * 0.5, mean_radius * 1.8)

    ecc = rng.uniform(*config.eccentricity_range, n_neurons)
    sigma_minor = sigma_major * ecc

    if config.orientation_jitter:
        orientation = rng.uniform(0, np.pi, n_neurons)
    else:
        orientation = np.zeros(n_neurons)

    peak_intensity = rng.uniform(*config.intensity_range, n_neurons)

    return {
        'sigma_major': sigma_major.astype(np.float32),
        'sigma_minor': sigma_minor.astype(np.float32),
        'orientation': orientation.astype(np.float32),
        'peak_intensity': peak_intensity.astype(np.float32),
        'mean_radius': mean_radius,
        'n_neurons': n_neurons,
    }


def perturb_params_for_session(base_params, session_seed, perturb_cfg: PerturbationConfig):
    """
    Apply per-session, per-neuron shape perturbations to base footprint params.

    Each neuron gets independent jitter drawn from N(0, std) for each
    shape parameter. The perturbed values are clipped to stay physical.

    Parameters
    ----------
    base_params   : dict from generate_base_footprint_params
    session_seed  : int, unique per session for reproducibility
    perturb_cfg   : PerturbationConfig

    Returns
    -------
    perturbed : dict with same keys as base_params, values jittered
    """
    rng = np.random.RandomState(session_seed)
    n = base_params['n_neurons']

    # Sigma perturbation — independent for major & minor
    d_major = rng.normal(0, perturb_cfg.sigma_jitter_std, n)
    d_minor = rng.normal(0, perturb_cfg.sigma_jitter_std, n)
    sigma_major = np.clip(base_params['sigma_major'] + d_major, 1.0, None).astype(np.float32)
    sigma_minor = np.clip(base_params['sigma_minor'] + d_minor, 1.0, None).astype(np.float32)
    # Ensure minor <= major
    sigma_minor = np.minimum(sigma_minor, sigma_major)

    # Orientation perturbation
    d_ori = rng.normal(0, perturb_cfg.orientation_jitter_std, n)
    orientation = (base_params['orientation'] + d_ori).astype(np.float32)
    # Wrap to [0, π)
    orientation = orientation % np.float32(np.pi)

    # Intensity perturbation (multiplicative)
    d_int = rng.normal(0, perturb_cfg.intensity_jitter_std, n)
    peak_intensity = np.clip(
        base_params['peak_intensity'] * (1.0 + d_int), 0.1, 1.5
    ).astype(np.float32)

    return {
        'sigma_major': sigma_major,
        'sigma_minor': sigma_minor,
        'orientation': orientation,
        'peak_intensity': peak_intensity,
        'mean_radius': base_params['mean_radius'],
        'n_neurons': n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Rendering (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

def render_footprint_compact(cx, cy, sigma_major, sigma_minor, orientation, peak,
                             fov_size, crop_factor=2.5):
    max_sigma = max(sigma_major, sigma_minor)
    crop_r = int(np.ceil(crop_factor * max_sigma))

    y_lo = max(0, int(cy) - crop_r)
    y_hi = min(fov_size, int(cy) + crop_r + 1)
    x_lo = max(0, int(cx) - crop_r)
    x_hi = min(fov_size, int(cx) + crop_r + 1)

    if y_hi <= y_lo or x_hi <= x_lo:
        return np.zeros((1, 1), dtype=np.float32), (0, 1, 0, 1)

    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dx = xx - cx
    dy = yy - cy
    cos_t = np.cos(orientation)
    sin_t = np.sin(orientation)
    dx_rot = dx * cos_t + dy * sin_t
    dy_rot = -dx * sin_t + dy * cos_t
    exponent = -0.5 * ((dx_rot / sigma_major) ** 2 + (dy_rot / sigma_minor) ** 2)
    patch = (peak * np.exp(exponent)).astype(np.float32)

    return patch, (y_lo, y_hi, x_lo, x_hi)


def reconstruct_dense_from_compact(compact_data, fov_size=None):
    if fov_size is None:
        fov_size = compact_data.get('fov_size', 600)
    patches = compact_data['patches']
    bboxes = compact_data['bboxes']
    n = len(patches)
    footprints = np.zeros((n, fov_size, fov_size), dtype=np.float32)
    for i in range(n):
        y_lo, y_hi, x_lo, x_hi = bboxes[i]
        footprints[i, y_lo:y_hi, x_lo:x_hi] = patches[i]
    return footprints


# ═══════════════════════════════════════════════════════════════════════════
# Build footprints for one session (now with perturbation)
# ═══════════════════════════════════════════════════════════════════════════

def build_session_footprints(session_data, perturbed_params, config: FootprintConfig):
    """
    Build footprints for one session using PERTURBED shape params.
    """
    cx = np.array(session_data['centroids_x'], dtype=float)
    cy = np.array(session_data['centroids_y'], dtype=float)
    roi_ids = np.array(session_data['roi_ids'], dtype=int)
    n_cells = len(cx)
    fov = config.fov_size
    want_dense = config.storage_mode in ('dense', 'both')
    want_compact = config.storage_mode in ('compact', 'both')

    dense_arr = np.zeros((n_cells, fov, fov), dtype=np.float32) if want_dense else None
    patches = [] if want_compact else None
    bboxes = [] if want_compact else None

    for i in range(n_cells):
        base_id = roi_ids[i]
        idx = base_id % perturbed_params['n_neurons']

        sm = perturbed_params['sigma_major'][idx]
        sn = perturbed_params['sigma_minor'][idx]
        ori = perturbed_params['orientation'][idx]
        pk = perturbed_params['peak_intensity'][idx]

        patch, bbox = render_footprint_compact(
            cx[i], cy[i], sm, sn, ori, pk, fov,
            config.footprint_crop_radius_factor)

        if config.normalize_footprints:
            norm = np.sqrt((patch ** 2).sum())
            if norm > 0:
                patch = patch / norm

        if want_compact:
            patches.append(patch)
            bboxes.append(bbox)

        if want_dense:
            y_lo, y_hi, x_lo, x_hi = bbox
            dense_arr[i, y_lo:y_hi, x_lo:x_hi] = patch

    result = {}
    if want_dense:
        result['dense'] = dense_arr
    if want_compact:
        result['compact'] = {'patches': patches, 'bboxes': bboxes}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main: iterate over benchmark directories with perturbation
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_footprints_perturbed(
    benchmark_dirs: list,
    n_neurons_list: list,
    n_animals: int = 8,
    config: Optional[FootprintConfig] = None,
    perturb_cfg: Optional[PerturbationConfig] = None,
    verbose: bool = True,
):
    """
    Generate perturbed spatial footprints for each benchmark directory.

    Same structure as the original but footprint shapes jitter per session.
    """
    if config is None:
        config = FootprintConfig()
    if perturb_cfg is None:
        perturb_cfg = PERTURBATION_PRESETS['gentle']

    for bench_dir, n_neurons in zip(benchmark_dirs, n_neurons_list):
        bench_dir = Path(bench_dir)
        if not bench_dir.exists():
            print(f"SKIP (not found): {bench_dir}")
            continue

        out_dir = bench_dir / perturb_cfg.output_folder_name

        # ── Skip if already generated ──
        existing = list(out_dir.glob("*_footprints_compact.npy")) if out_dir.exists() else []
        session_files = sorted(bench_dir.glob("*__seed*.npy"))
        if len(existing) >= len(session_files) and len(existing) > 0:
            if verbose:
                print(f"\n  SKIP {bench_dir.name}: {perturb_cfg.output_folder_name}/ already has "
                      f"{len(existing)} files (>= {len(session_files)} sessions)")
            continue

        out_dir.mkdir(exist_ok=True)

        if verbose:
            mean_r = config.get_mean_radius(n_neurons)
            print(f"\n{'═' * 60}")
            print(f"  {bench_dir.name}  [{perturb_cfg.preset_label.upper()} perturbation]")
            print(f"  n_neurons={n_neurons}  mean_radius={mean_r:.1f} px")
            print(f"  σ_jitter={perturb_cfg.sigma_jitter_std:.2f} px  "
                  f"θ_jitter={np.degrees(perturb_cfg.orientation_jitter_std):.1f}°  "
                  f"I_jitter={perturb_cfg.intensity_jitter_std:.0%}")
            print(f"{'═' * 60}")

        # ── Generate base footprint params per animal (SAME seeds as original) ──
        base_params_by_animal = {}
        for animal_idx in range(n_animals):
            animal_id = str(n_neurons + animal_idx + 1)
            animal_seed = animal_idx * 10_000
            params = generate_base_footprint_params(n_neurons, animal_seed, config)
            base_params_by_animal[animal_id] = params

            np.save(out_dir / f"{animal_id}_base_footprint_params.npy",
                    params, allow_pickle=True)

        # ── Process every session file ──
        if not session_files:
            session_files = sorted(bench_dir.glob("*__seed*.npy"))
        file_count = 0
        skip_count = 0

        for sf_idx, sf in enumerate(session_files):
            parts = sf.stem.split('_')
            animal_id = parts[0]

            if animal_id not in base_params_by_animal:
                continue

            out_base = sf.stem + "_footprints"
            compact_exists = (out_dir / (out_base + "_compact.npy")).exists()
            dense_exists   = (out_dir / (out_base + ".npy")).exists()
            want_compact = config.storage_mode in ('compact', 'both')
            want_dense   = config.storage_mode in ('dense', 'both')
            if (want_compact and compact_exists) or (want_dense and dense_exists):
                skip_count += 1
                continue

            session_data = np.load(sf, allow_pickle=True).item()
            base_params = base_params_by_animal[animal_id]

            # ── THE KEY DIFFERENCE: perturb shapes for this session ──
            # Seed is unique per session file so perturbations are reproducible
            # but different across sessions (which is the whole point).
            session_perturb_seed = hash(sf.stem) % (2**31)
            perturbed_params = perturb_params_for_session(
                base_params, session_perturb_seed, perturb_cfg
            )

            result = build_session_footprints(session_data, perturbed_params, config)

            if 'compact' in result:
                compact_data = {
                    'patches': result['compact']['patches'],
                    'bboxes': result['compact']['bboxes'],
                    'centroids_x': session_data['centroids_x'],
                    'centroids_y': session_data['centroids_y'],
                    'roi_ids': session_data['roi_ids'],
                    'fov_size': config.fov_size,
                    'mean_radius_px': config.get_mean_radius(n_neurons),
                    'perturbation_preset': perturb_cfg.preset_label,
                    'sigma_jitter_std': perturb_cfg.sigma_jitter_std,
                    'orientation_jitter_std': perturb_cfg.orientation_jitter_std,
                    'intensity_jitter_std': perturb_cfg.intensity_jitter_std,
                }
                np.save(out_dir / (out_base + "_compact.npy"),
                        compact_data, allow_pickle=True)

            if 'dense' in result:
                dense_data = {
                    'footprints': result['dense'],
                    'centroids_x': session_data['centroids_x'],
                    'centroids_y': session_data['centroids_y'],
                    'roi_ids': session_data['roi_ids'],
                    'fov_size': config.fov_size,
                    'mean_radius_px': config.get_mean_radius(n_neurons),
                    'perturbation_preset': perturb_cfg.preset_label,
                }
                np.save(out_dir / (out_base + ".npy"),
                        dense_data, allow_pickle=True)

            file_count += 1

        if verbose:
            print(f"  Generated {file_count} footprint files"
                  f"{f', skipped {skip_count} existing' if skip_count else ''}"
                  f" → {out_dir}")

    print(f"\nDone.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # BASE = Path(r"C:\Users\ariAccount\Desktop")

    # NEURON_COUNTS = [100, 250, 500, 1000]
    # BENCHMARK_DIRS = [BASE / f"Stars2Cells_Benchmark_{n}n" for n in NEURON_COUNTS]

    BASE = Path(r"C:\Users\ariAccount\Desktop\Stars2CellsPaper")
    NEURON_COUNTS = [1000, 1000]
    BENCHMARK_DIRS = [
        BASE / "Stars2Cells_Benchmark_1000n_Tier_B",
        BASE / "Stars2Cells_Benchmark_1000n_Tier_C",
    ]

    N_ANIMALS = 8

    config = FootprintConfig(
        fov_size=600,
        microns_per_pixel=1.0,
        radius_by_n={
            100:  12.0,
            250:  10.0,
            500:   8.0,
            1000:  6.0,
        },
        eccentricity_range=(0.55, 1.0),
        intensity_range=(0.5, 1.0),
        radius_cv=0.20,
        normalize_footprints=True,
        storage_mode='compact',
    )

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  Pick your perturbation level here.                              ║
    # ║                                                                   ║
    # ║  'gentle'  — 0.1 px sigma, ~2° ori, ~2% intensity               ║
    # ║              "We barely touched it and CellReg still struggled"   ║
    # ║                                                                   ║
    # ║  'moderate' — 0.5 px sigma, ~5° ori, ~5% intensity              ║
    # ║              "Realistic session-to-session variation"             ║
    # ║                                                                   ║
    # ║  'harsh'   — 1.5 px sigma, ~15° ori, ~15% intensity             ║
    # ║              "Bad day at the scope"                               ║
    # ║                                                                   ║
    # ║  Or define your own PerturbationConfig() for custom values.      ║
    # ╚═══════════════════════════════════════════════════════════════════╝

    PRESET = 'moderate'  # ← change this to 'moderate' or 'harsh'

    perturb_cfg = PERTURBATION_PRESETS[PRESET]

    generate_all_footprints_perturbed(
        benchmark_dirs=BENCHMARK_DIRS,
        n_neurons_list=NEURON_COUNTS,
        n_animals=N_ANIMALS,
        config=config,
        perturb_cfg=perturb_cfg,
        verbose=True,
    )