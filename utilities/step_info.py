"""
Step Information - Single Source of Truth
==========================================
All step-specific metadata, configurations, and patterns.
Modify this file to add new steps or change step behavior.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
from pathlib import Path


# ==============================================================================
# PARAMETER SCHEMAS - SINGLE SOURCE OF TRUTH FOR CONFIG
# ==============================================================================

PARAMETER_SCHEMAS = {
    # ========== GENERAL ==========
    'n_workers': {'default': 4, 'type': int, 'min': 1, 'max': 32, 'widget': 'spinbox', 'description': 'Number of parallel workers'},
    'verbose': {'default': True, 'type': bool, 'widget': 'checkbox', 'description': 'Enable verbose logging'},
    'skip_existing': {'default': True, 'type': bool, 'widget': 'checkbox', 'description': 'Skip if output exists'},

    # ========== STEP 1: QUAD GENERATION ==========
    # Filename parsing
    'session_filename_regex': {
        'default': r'^([A-Za-z0-9_]+?)_(\d+)__.*\.npy$',
        'type': str,
        'widget': 'lineedit',
        'description': 'Regex to parse animal_id and session_number from .npy filenames. '
                       'Must have exactly 2 capture groups: (animal_id, session_number).'
    },
    # Diagonal sampling
    'knn_k': {
        'default': 15,
        'type': int, 'min': 5, 'max': 100, 'widget': 'spinbox',
        'description': 'Local KNN neighbors per neuron for diagonal generation. '
                       'Long-range connections = knn_k // 2 (auto). Increase for denser coverage.'
    },
    'diagonal_rng_seed': {
        'default': 42,
        'type': int, 'min': 0, 'max': 999999, 'widget': 'spinbox',
        'description': 'RNG seed for long-range diagonal sampling. '
                       'Change only if you suspect seed-specific coverage gaps.'
    },
    # K-cap (applied inline during diagonal construction)
    'max_triangles_per_diagonal': {
        'default': 25,
        'type': int, 'min': 2, 'max': 500, 'widget': 'spinbox',
        'description': 'Max third-points kept per diagonal (K-cap). '
                       'Top-K by height, applied upfront. '
                       'Quads per diagonal = C(K,2). Reduce to cut quad count.'
    },
    # Quality filters
    'min_pairwise_distance': {
        'default': 0.0,
        'type': float, 'min': 0.0, 'max': 100.0, 'widget': 'doublespinbox',
        'description': 'Min pairwise pixel distance between any two quad vertices. '
                       '0 = disabled. Use to discard quads with nearly-coincident neurons.'
    },
    'quad_keep_fraction': {
        'default': 1.0,
        'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 2,
        'description': 'Final quality pruning: keep this fraction of quads ranked by |yC|+|yD|. '
                       '1.0 = keep all. Reduce if quad count exceeds saturation target.'
    },
    'min_coverage_fraction': {
        'default': 0.4,
        'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 2,
        'description': 'Coverage remediation threshold: neurons with fewer than this fraction '
                       'of the field median quad count are boosted by relaxing the ownership guard. '
                       '0.4 = any neuron below 40% of median gets remediated. 0 = disabled.'
    },
    # height_percentile retained for diagnostics / log_diagonal_stats display
    'height_percentile': {
        'default': 95.0,
        'type': float, 'min': 0.0, 'max': 100.0, 'widget': 'doublespinbox',
        'description': 'Percentile (0-100 scale) used in diagnostic height statistics. '
                       'Does NOT gate the pipeline — the K-cap (max_triangles_per_diagonal) '
                       'controls third-point selection by height directly.'
    },
    # RAM / parallel budget
    'parallel_safety_margin': {
        'default': 0.1,
        'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 2,
        'description': 'Fraction of available RAM to reserve as safety margin'
    },
    'per_session_gb': {
        'default': 2.0,
        'type': float, 'min': 0.1, 'max': 200.0, 'widget': 'doublespinbox',
        'description': 'Estimated peak RAM (GB) per session. '
                       'Diagonal-first pipeline is O(N·k·K) not O(N³), '
                       'so this is much lower than the old triangle-based estimate.'
    },

    # ========== STEP 1.5: THRESHOLD CALIBRATION ==========
    'sample_size': {'default': 10000, 'type': int, 'min': 10, 'max': 1_000_000, 'widget': 'spinbox', 'description': 'Quads to sample per session'},
    'target_quality': {'default': 0.95, 'type': float, 'min': 0.5, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 7, 'description': 'Target quality'},
    'threshold_min': {'default': 0.0, 'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 7, 'description': 'Min threshold to test'},
    'threshold_max': {'default': 1.0, 'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 7, 'description': 'Max threshold to test'},
    'n_threshold_points': {'default': 50, 'type': int, 'min': 5, 'max': 200, 'widget': 'spinbox', 'description': 'Number of test points'},

    'session_group_regex': {
        'default': r'__(.+)$',
        'type': str,
        'widget': 'lineedit',
        'description': 'Regex to extract session group key from session name'
    },
    'session_pair_strategy': {
        'default': 'consecutive',
        'type': str,
        'options': ['consecutive', 'all_vs_all'],
        'widget': 'combobox',
        'description': 'How to generate session pairs within a group'
    },

    # ========== STEP 2: QUAD MATCHING ==========
    'threshold': {'default': None, 'type': float, 'min': 0.01, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 4, 'nullable': True, 'description': 'Similarity threshold (None=calibrated)'},
    'distance_metric': {'default': 'cosine', 'type': str, 'options': ['cosine', 'euclidean'], 'widget': 'combobox', 'description': 'Distance metric'},
    'consistency_threshold': {'default': 0.8, 'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 2, 'description': 'Geometric consistency threshold'},

    # ========== STEP 2.5: RANSAC ==========
    'ransac_max_residual': {'default': 5.0, 'type': float, 'min': 0.1, 'max': 100.0, 'widget': 'doublespinbox', 'description': 'Max RANSAC residual (px)'},
    'ransac_iterations': {'default': 1000, 'type': int, 'min': 100, 'max': 10000, 'widget': 'spinbox', 'description': 'RANSAC iterations'},
    'ransac_min_inlier_ratio': {'default': 0.05, 'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 5, 'description': 'Min inlier ratio'},
    'ransac_max_rotation_deg': {
        'default': None, 'type': float, 'min': 0.0, 'max': 180.0,
        'widget': 'doublespinbox', 'decimals': 1, 'nullable': True,
        'description': 'Max allowed rotation (degrees). Reject transforms exceeding this. None = no limit.'
    },
    'ransac_max_translation_px': {
        'default': None, 'type': float, 'min': 0.0, 'max': 1000.0,
        'widget': 'doublespinbox', 'decimals': 1, 'nullable': True,
        'description': 'Max allowed translation (pixels). Reject transforms exceeding this. None = no limit.'
    },

    # ========== STEP 3: NEURON MATCHING (V3) ==========
    'use_quad_voting': {
        'default': True,
        'type': bool,
        'widget': 'checkbox',
        'description': 'Use normalized quad voting for cost matrix (vs. pure distance fallback)'
    },
    'use_asymmetric_dummy_costs': {
        'default': False,
        'type': bool,
        'widget': 'checkbox',
        'description': 'Scale dummy costs per-neuron by proximity '
                       '(closer neurons are harder to leave unmatched). '
                       'False = uniform dummy cost for all neurons.'
    },
    'block_zero_vote_pairs': {
        'default': False,
        'type': bool,
        'widget': 'checkbox',
        'description': 'Hard-block neuron pairs with zero quad votes (cost=inf). '
                       'False = zero-vote pairs within distance cutoff get a '
                       'distance-only fallback cost.'
    },
    'dist_cutoff_multiplier': {
        'default': 3.0,
        'type': float,
        'min': 1.0,
        'max': 20.0,
        'widget': 'doublespinbox',
        'decimals': 1,
        'description': 'Distance cutoff = this × ransac_max_residual. '
                       'Pairs farther apart are blocked (cost=inf).'
    },
    'postfilter_residual_multiplier': {
        'default': 1.0,
        'type': float,
        'min': 0.5,
        'max': 10.0,
        'widget': 'doublespinbox',
        'decimals': 1,
        'description': 'Post-filter cutoff = this × ransac_max_residual. '
                       'Pass-1 matches exceeding this transformed distance are removed.'
    },
    'pass2_cutoff_multiplier': {
        'default': 2.0,
        'type': float,
        'min': 1.0,
        'max': 10.0,
        'widget': 'doublespinbox',
        'decimals': 1,
        'description': 'Second-pass recovery cutoff = this × ransac_max_residual. '
                       'Relaxed relative to pass-1 to recover missed neurons.'
    },
    'pass2_dummy_percentile': {
        'default': 75.0,
        'type': float,
        'min': 10.0,
        'max': 99.0,
        'widget': 'doublespinbox',
        'decimals': 0,
        'description': 'Percentile of finite costs used as dummy cost in '
                       'second-pass recovery. Higher = more permissive matching.'
    },

    # ========== LEGACY (kept for compatibility, ignored by V3) ==========
    'target_match_rate': {'default': None, 'type': float, 'min': 0.0, 'max': 1.0, 'widget': 'doublespinbox', 'decimals': 2, 'nullable': True, 'description': 'Target match rate (legacy, ignored)'},
    'hungarian_max_cost': {'default': None, 'type': float, 'min': 0.0, 'max': 10000.0, 'widget': 'doublespinbox', 'nullable': True, 'description': 'Max Hungarian cost (legacy, ignored)'},
    'hungarian_cost_min': {'default': 0.0, 'type': float, 'min': 0.0, 'max': 99.0, 'widget': 'doublespinbox', 'description': 'Min cost for sweep (legacy, ignored)'},
    'hungarian_cost_max': {'default': 100.0, 'type': float, 'min': 0.0, 'max': 100.0, 'widget': 'doublespinbox', 'description': 'Max cost for sweep (legacy, ignored)'},
    'hungarian_cost_steps': {'default': 20, 'type': int, 'min': 2, 'max': 1000, 'widget': 'spinbox', 'description': 'Number of cost steps (legacy, ignored)'},
    'run_sweep': {'default': False, 'type': bool, 'widget': 'checkbox', 'description': 'Run Hungarian cost sweep (legacy, ignored)'},

    # ========== IMAGE ==========
    'image_width': {'default': 640, 'type': int, 'min': 1, 'max': 10000, 'widget': 'spinbox', 'description': 'Image width (px)'},
    'image_height': {'default': 640, 'type': int, 'min': 1, 'max': 10000, 'widget': 'spinbox', 'description': 'Image height (px)'},
    'max_centroid_distance_pct': {'default': 5.0, 'type': float, 'min': 0.0, 'max': 100.0, 'widget': 'doublespinbox', 'description': 'Max centroid distance (%)'},
}

# Helper functions for parameter schemas
def get_parameter_schema(param_name: str) -> Optional[Dict]:
    """Get schema for a parameter"""
    return PARAMETER_SCHEMAS.get(param_name)

def get_parameter_default(param_name: str) -> Any:
    """Get default value for a parameter"""
    schema = PARAMETER_SCHEMAS.get(param_name, {})
    return schema.get('default')

def get_step_parameters(step: float) -> list:
    """Get list of parameter names used by a step"""
    info = STEP_METADATA.get(step, {})
    run_info = info.get('run_kwargs', {})
    return run_info.get('extra_params', [])

# ==============================================================================
# STEP METADATA - SINGLE SOURCE OF TRUTH
# ==============================================================================

STEP_METADATA = {
    1: {
        # ===== DIRECTORY & FILE INFO =====
        'name': 'Quad Generation',
        'output_subdir': 'step_1_results',
        'file_pattern': '*_centroids_quads.npz',
        'logger_name': 'neuron_mapping_parallel',

        # ===== PIPELINE EXECUTION =====
        'run_module': 'step_1_quad_generation',
        'run_function': 'run_step_1_parallel',
        'run_kwargs': {
            'needs_config': True,
            'needs_sessions': True,
            'needs_callback': True,  # session_callback
            'extra_params': [
                'session_filename_regex',
                'knn_k',
                'diagonal_rng_seed',
                'max_triangles_per_diagonal',
                'quad_keep_fraction',
                'min_pairwise_distance',
                'min_coverage_fraction',
                'session_group_regex',
                'session_pair_strategy'
            ]
        },
        'progress_signal': 'session_progress',
        'count_unit': 'sessions',

        # ===== STATISTICS LOADING =====
        'stats': {
            'extract_fields': ['n_neurons', 'n_quads', 'generation_time'],
            'accumulate_fields': {
                'total_quads': 'n_quads',
                'total_neurons': 'n_neurons'
            },
            'per_item_storage': 'sessions',
            'item_name_field': 'session_name',
            'aggregate_func': None,
            'temp_accumulators': None,
            'has_tracking': False,
        },

        # ===== UI DISPLAY =====
        'label': 'Step 1: Quad Generation',
        'icon': '🔷',
        'description': 'Generate quad descriptors from neuron centroids (diagonal-first pipeline)',
        'prerequisites': [],
        'enables': [1.5],
    },

    1.5: {
        # ===== DIRECTORY & FILE INFO =====
        'name': 'Threshold Calibration',
        'output_subdir': 'step_1_5_results',
        'file_pattern': '*_threshold_calibration.npz',
        'logger_name': 'neuron_mapping',

        # ===== PIPELINE EXECUTION =====
        'run_module': 'step_1_5_threshold_generation',
        'run_function': 'run_global_tuning_all_animals',
        'run_kwargs': {
            'needs_config': True,
            'needs_sessions': False,
            'needs_callback': True,
            'extra_params': [
                'sample_size',
                'target_quality',
                'threshold_min',
                'threshold_max',
                'n_threshold_points',
                'session_group_regex',
                'session_pair_strategy'
            ]
        },
        'progress_signal': 'animal_progress',
        'count_unit': 'animals',

        # ===== STATISTICS LOADING =====
        'stats': {
            'extract_fields': ['C', 'C_std', 'r_squared', 'n_pairs', 'optimal_threshold'],
            'accumulate_fields': {},
            'per_item_storage': None,
            'item_name_field': None,
            'aggregate_func': 'aggregate_step1_5',
            'temp_accumulators': None,
            'has_tracking': False,
        },

        # ===== UI DISPLAY =====
        'label': 'Step 1.5: Threshold Calibration',
        'icon': '🎯',
        'description': 'Calibrate C thresholds using √N scaling',
        'prerequisites': [1],
        'enables': [2],
    },

    2: {
        # ===== DIRECTORY & FILE INFO =====
        'name': 'Quad Matching',
        'output_subdir': 'step_2_results',
        'file_pattern': '*_matches_light.npz',
        'logger_name': 'neuron_mapping_step2',

        # ===== PIPELINE EXECUTION =====
        'run_module': 'step_2_matching_generator',
        'run_function': 'run_step_2_all_animals_parallel',
        'run_kwargs': {
            'needs_config': True,
            'needs_sessions': False,
            'needs_callback': False,
            'extra_params': [
                'distance_metric',
                'consistency_threshold',
                'session_group_regex',
                'session_pair_strategy'
            ]
        },
        'progress_signal': 'animal_progress',
        'count_unit': 'pairs',

        # ===== STATISTICS LOADING =====
        'stats': {
            'extract_fields': ['n_filtered_matches', 'n_ref_neurons', 'n_target_neurons', 'threshold_used'],
            'accumulate_fields': {
                'total_matches': 'n_filtered_matches'
            },
            'per_item_storage': 'pairs',
            'item_name_field': 'pair_name',
            'aggregate_func': 'aggregate_step2',
            'temp_accumulators': ['thresholds'],
            'has_tracking': False,
        },

        # ===== UI DISPLAY =====
        'label': 'Step 2: Quad Matching',
        'icon': '🔗',
        'description': 'Match quads across sessions using descriptor similarity',
        'prerequisites': [1, 1.5],
        'enables': [2.5],
    },

    2.5: {
        # ===== DIRECTORY & FILE INFO =====
        'name': 'RANSAC Filtering',
        'output_subdir': 'step_2_5_results',
        'file_pattern': '*_filtered_matches.npz',
        'logger_name': 'neuron_mapping_ransac',

        # ===== PIPELINE EXECUTION =====
        'run_module': 'step_2_5_RANSAC',
        'run_function': 'run_step_2_5_ransac',
        'run_kwargs': {
            'needs_config': True,
            'needs_sessions': False,
            'needs_callback': False,
            'extra_params': [
                'ransac_max_residual',
                'ransac_iterations',
                'ransac_min_inlier_ratio',
                'ransac_max_rotation_deg',
                'ransac_max_translation_px',
            ]
        },
        'progress_signal': 'animal_progress',
        'count_unit': 'pairs',

        # ===== STATISTICS LOADING =====
        'stats': {
            'extract_fields': [
                'n_descriptor_matches', 'n_matches', 'filtering_ratio',
                'translation_magnitude', 'rotation_deg', 'scale', 'mean_residual'
            ],
            'accumulate_fields': {
                'total_descriptor_matches': 'n_descriptor_matches',
                'total_geometric_inliers': 'n_matches'
            },
            'per_item_storage': 'pairs',
            'item_name_field': 'pair_name',
            'aggregate_func': 'aggregate_step2_5',
            'temp_accumulators': ['filtering_ratios', 'translations', 'rotations'],
            'has_tracking': False,
        },

        # ===== UI DISPLAY =====
        'label': 'Step 2.5: RANSAC Filtering',
        'icon': '🎲',
        'description': 'Geometric filtering using RANSAC consensus',
        'prerequisites': [2],
        'enables': [3],
    },

    3: {
        # ===== DIRECTORY & FILE INFO =====
        'name': 'Neuron Matching',
        'output_subdir': 'step_3_results',
        'file_pattern': '*_sweep.npz',
        'logger_name': 'neuron_mapping_consolidated',

        # ===== PIPELINE EXECUTION =====
        'run_module': 'step_3_neuron_matching',
        'run_function': 'run_step_3_final_matching',
        'run_kwargs': {
            'needs_config': True,
            'needs_sessions': False,
            'needs_callback': False,
            'extra_params': [
                'use_quad_voting',
                'use_asymmetric_dummy_costs',
                'block_zero_vote_pairs',
                'dist_cutoff_multiplier',
                'postfilter_residual_multiplier',
                'pass2_cutoff_multiplier',
                'pass2_dummy_percentile',
            ]
        },
        'progress_signal': 'animal_progress',
        'count_unit': 'animals',

        # ===== STATISTICS LOADING =====
        'stats': {
            'extract_fields': ['n_ref_neurons', 'n_target_neurons', 'optimal_threshold', 'optimal_matches', 'match_rate'],
            'accumulate_fields': {
                'total_neuron_matches': 'optimal_matches'
            },
            'per_item_storage': 'pairs',
            'item_name_field': 'pair_name',
            'aggregate_func': 'aggregate_step3',
            'temp_accumulators': ['match_rates'],
            'has_tracking': False,
        },

        # ===== UI DISPLAY =====
        'label': 'Step 3: Neuron Matching',
        'icon': '📊',
        'description': 'RANSAC-informed Hungarian matching with dummy padding and second-pass recovery',
        'prerequisites': [2.5],
        'enables': [],
    },
}

# ==============================================================================
# HELPER FUNCTIONS - STEP QUERIES
# ==============================================================================

def get_step_info(step: float) -> Dict[str, Any]:
    """Get all metadata for a step"""
    return STEP_METADATA.get(step, {})

def get_step_output_dir(step: float, base_dir: str) -> Path:
    """Get output directory for a step"""
    subdir = STEP_METADATA[step]['output_subdir']
    return Path(base_dir) / subdir

def get_step_file_pattern(step: float) -> str:
    """Get file pattern for a step"""
    return STEP_METADATA[step]['file_pattern']

def get_step_logger_name(step: float) -> str:
    """Get logger name for a step"""
    return STEP_METADATA[step]['logger_name']

def get_step_prerequisites(step: float) -> list:
    """Get prerequisite steps"""
    return STEP_METADATA[step]['prerequisites']

def get_step_enables(step: float) -> list:
    """Get which steps this enables"""
    return STEP_METADATA[step]['enables']

def get_all_step_numbers() -> list:
    """Get all step numbers"""
    return sorted(STEP_METADATA.keys())

def validate_step(step: float) -> bool:
    """Check if step exists"""
    return step in STEP_METADATA

def get_step_label(step: float) -> str:
    """Get UI label for step"""
    info = STEP_METADATA.get(step, {})
    return info.get('label', f'Step {step}')

def get_step_description(step: float) -> str:
    """Get step description"""
    info = STEP_METADATA.get(step, {})
    return info.get('description', '')

# ==============================================================================
# DIRECTORY NAME ALIASES - For backwards compatibility
# ==============================================================================

STEP_DIR_ALIASES = {
    1: ['step_1_results', 'intermediate'],
    1.5: ['step_1_5_results'],
    2: ['step_2_results', 'intermediate', 'step2_results'],
    2.5: ['step_2_5_results', 'step2_5_results'],
    3: ['step_3_results', 'step_3_sweep', 'step3_results', 'final_results', 'step_3_final'],
}

def get_step_dir_aliases(step: float) -> list:
    """Get all possible directory names for a step"""
    return STEP_DIR_ALIASES.get(step, [])

def parse_step_from_dirname(dirname: str) -> Optional[float]:
    """Try to parse step number from directory name"""
    for step, aliases in STEP_DIR_ALIASES.items():
        if dirname in aliases:
            return step
    return None

# ==============================================================================
# STEP ORDERING - For pipeline execution
# ==============================================================================

STEP_ORDER = [1, 1.5, 2, 2.5, 3]

def get_next_step(current_step: float) -> Optional[float]:
    """Get the next step in the pipeline"""
    try:
        idx = STEP_ORDER.index(current_step)
        if idx < len(STEP_ORDER) - 1:
            return STEP_ORDER[idx + 1]
    except ValueError:
        pass
    return None

def get_previous_step(current_step: float) -> Optional[float]:
    """Get the previous step in the pipeline"""
    try:
        idx = STEP_ORDER.index(current_step)
        if idx > 0:
            return STEP_ORDER[idx - 1]
    except ValueError:
        pass
    return None

# ==============================================================================
# CONFIGURATION EXTRACTION - For building run arguments
# ==============================================================================

def build_run_kwargs(step: float, config) -> Dict[str, Any]:
    """
    Build kwargs dictionary for running a step.

    Args:
        step: Step number
        config: PipelineConfig object

    Returns:
        Dictionary of kwargs to pass to run function
    """
    info = STEP_METADATA[step]
    run_info = info['run_kwargs']

    kwargs = {
        'input_dir': str(config.input_dir),
        'output_dir': str(config.output_dir),
        'processes': None,
        'verbose': True,
    }

    if 'extra_params' in run_info:
        for param in run_info['extra_params']:
            if hasattr(config, param):
                kwargs[param] = getattr(config, param)

    return kwargs

# ==============================================================================
# ANIMAL COUNTING FUNCTIONS - For progress tracking
# ==============================================================================

def count_step1_animals(output_dir: str) -> int:
    """Count animals from Step 1 results"""
    step1_dir = Path(output_dir) / STEP_METADATA[1]['output_subdir']
    if not step1_dir.exists():
        return 0
    animals = set()
    pattern = STEP_METADATA[1]['file_pattern']
    for npz_file in step1_dir.glob(pattern):
        animal_id = npz_file.stem.split('_')[0]
        animals.add(animal_id)
    return len(animals)

def count_step2_animals(output_dir: str) -> int:
    """Count animals from Step 2 results"""
    from steps.step_2_5_RANSAC import discover_animals
    step2_dir = Path(output_dir) / STEP_METADATA[2]['output_subdir']
    if not step2_dir.exists():
        return 0
    pattern = STEP_METADATA[2]['file_pattern']
    animals = discover_animals(step2_dir, pattern=pattern, verbose=False)
    return len(animals)

def count_step2_5_animals(output_dir: str) -> int:
    """Count animals from Step 2.5 results"""
    step2_5_dir = Path(output_dir) / STEP_METADATA[2.5]['output_subdir']
    if not step2_5_dir.exists():
        return 0
    animals = []
    pattern = STEP_METADATA[2.5]['file_pattern']
    for npz_file in sorted(step2_5_dir.glob(pattern)):
        animal_id = npz_file.stem.replace("_filtered_matches", "")
        animals.append(animal_id)
    return len(set(animals))

def count_items_for_step(step: float, output_dir: str, sessions=None) -> int:
    """
    Count items to process for a step (sessions or animals).

    Args:
        step: Step number
        output_dir: Output directory
        sessions: Loaded sessions (for Step 1 only)

    Returns:
        Number of items to process
    """
    if step == 1:
        return len(sessions) if sessions else 0
    elif step == 1.5:
        return count_step1_animals(output_dir)
    elif step == 2:
            # Count session pairs, not animals
            step1_dir = Path(output_dir) / STEP_METADATA[1]['output_subdir']
            if not step1_dir.exists():
                return 0
            n_sessions = len(list(step1_dir.glob(STEP_METADATA[1]['file_pattern'])))
            # Consecutive pairing ≈ sessions grouped by animal, ~(sessions_per_animal - 1) per animal
            # For a rough estimate, use total sessions - number of animals
            n_animals = count_step1_animals(output_dir)
            return max(0, n_sessions - n_animals) if n_animals > 0 else max(0, n_sessions - 1)
    elif step == 2.5:
        step2_dir = Path(output_dir) / STEP_METADATA[2]['output_subdir']
        if not step2_dir.exists():
            return 0
        return len(list(step2_dir.glob(STEP_METADATA[2]['file_pattern'])))
    elif step == 3:
        return count_step2_5_animals(output_dir)
    elif step == 3.1:
        return count_step2_5_animals(output_dir)
    return 0

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

def check_prerequisites(step: float, output_dir: str, verbose: bool = False) -> tuple:
    """
    Check if all prerequisite steps have been run.

    Args:
        step: Step to check prerequisites for
        output_dir: Base output directory
        verbose: Print debug info

    Returns:
        (all_met, missing_steps): True if all met, list of missing step numbers
    """
    prereqs = get_step_prerequisites(step)
    missing = []

    for prereq_step in prereqs:
        prereq_dir = get_step_output_dir(prereq_step, output_dir)
        pattern = get_step_file_pattern(prereq_step)

        if not prereq_dir.exists():
            missing.append(prereq_step)
            if verbose:
                print(f"✗ Missing prerequisite: {prereq_dir}")
            continue

        files = list(prereq_dir.glob(pattern))
        if not files:
            missing.append(prereq_step)
            if verbose:
                print(f"✗ No files found for prerequisite: {prereq_dir}/{pattern}")

    if verbose and not missing:
        print(f"✓ All prerequisites met for Step {step}")

    return len(missing) == 0, missing