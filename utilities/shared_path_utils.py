"""
Path and Directory Utilities 
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from scipy.ndimage import center_of_mass

from .step_info import *
logger = logging.getLogger("neuron_mapping")

# ============================================================================
# DIRECTORY DETECTION - Fully step-agnostic
# ============================================================================

def auto_detect_output_dir(
    required_steps: Optional[List[float]] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Auto-detect output directory by searching for step results.
    
    Uses step_info.py to determine which directories to look for.
    
    Searches in:
    - Current directory
    - Parent directory
    - ./Results
    - ../Results
    
    Parameters
    ----------
    required_steps : list of float, optional
        Step numbers that must exist (e.g., [1, 1.5])
        If None, just looks for any Results-like directory
    verbose : bool
        Verbose logging
        
    Returns
    -------
    str or None
        Path to detected directory, or None if not found
    """
    current = Path.cwd()
    candidates = [
        current,
        current.parent,
        current / "Results",
        current.parent / "Results",
    ]
    
    if verbose:
        logger.info("Auto-detecting output directory...")
    
    for candidate in candidates:
        if not candidate.exists():
            continue
        
        # If no requirements, just check if it looks like a results dir
        if required_steps is None:
            # Check for any step results directory using step_info
            for step in get_all_step_numbers():
                for alias in get_step_dir_aliases(step):
                    if (candidate / alias).exists():
                        if verbose:
                            logger.info(f"✓ Detected: {candidate}")
                        return str(candidate)
            continue
        
        # Check specific required steps exist
        all_exist = True
        for step in required_steps:
            step_dir = get_step_output_dir(step, str(candidate))
            if not step_dir.exists():
                all_exist = False
                break
        
        if all_exist:
            if verbose:
                logger.info(f"✓ Detected: {candidate}")
            return str(candidate)
    
    return None

def validate_prerequisites(
    output_dir: str,
    required_steps: List[float],
    verbose: bool = True
) -> Tuple[bool, List[float]]:
    """
    Validate that prerequisite steps have been run.
    
    Uses step_info.py for all step metadata.
    
    Parameters
    ----------
    output_dir : str
        Base output directory
    required_steps : list of float
        Required step numbers (e.g., [1, 1.5])
    verbose : bool
        Verbose logging
        
    Returns
    -------
    valid : bool
        True if all prerequisites exist
    missing : list of float
        List of missing step numbers
    """
    # Use centralized check from step_info
    return check_prerequisites(output_dir, required_steps, verbose)

def ensure_output_dir(
    base_dir: str,
    step: float,
    verbose: bool = True
) -> Path:
    """
    Ensure output directory exists for a step.
    
    Uses step_info.py to get correct subdirectory name.
    
    Parameters
    ----------
    base_dir : str
        Base output directory
    step : float
        Step number
    verbose : bool
        Verbose logging
        
    Returns
    -------
    Path
        Path to step output directory
    """
    output_dir = get_step_output_dir(step, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logger.info(f"Output directory: {output_dir}")
    
    return output_dir

def print_detection_error(
    required_steps: List[float],
    search_locations: Optional[List[Path]] = None
) -> None:
    """
    Print helpful error message when directory detection fails.
    
    Uses step_info.py for step labels and metadata.
    
    Parameters
    ----------
    required_steps : list of float
        Steps that were required
    search_locations : list of Path, optional
        Locations that were searched (uses defaults if None)
    """
    print("\n❌ ERROR: Could not find required step results")
    
    # Get labels from step_info
    step_labels = [get_step_label(s) for s in required_steps]
    print(f"\nRequired: {', '.join(step_labels)}")
    
    if search_locations is None:
        current = Path.cwd()
        search_locations = [
            current,
            current.parent,
            current / "Results",
            current.parent / "Results",
        ]
    
    print("\nSearched in:")
    for loc in search_locations:
        status = "✓" if loc.exists() else "✗"
        subdirs_status = []
        
        for step in required_steps:
            step_dir = get_step_output_dir(step, str(loc))
            if step_dir.exists():
                subdirs_status.append(f"{step_dir.name}: ✓")
            else:
                subdirs_status.append(f"{step_dir.name}: ✗")
        
        print(f"  {status} {loc}")
        if loc.exists():
            for s in subdirs_status:
                print(f"      {s}")
    
    print("\nPlease either:")
    print("  1. Run this script from your results directory, OR")
    print("  2. Specify --output-dir explicitly")
    print("\nExample:")
    print("  python script.py --output-dir /path/to/Results")

def setup_input_output_dirs(
    args_output_dir: Optional[str],
    args_input_dir: Optional[str],
    required_steps: List[float],
    auto_create_input: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Setup and validate input/output directories.
    
    Uses step_info.py for all step validation.
    
    Parameters
    ----------
    args_output_dir : str or None
        Output directory from command line args
    args_input_dir : str or None
        Input directory from command line args
    required_steps : list of float
        Required prerequisite step numbers
    auto_create_input : bool
        If True, auto-create input dir path even if it doesn't exist
        
    Returns
    -------
    output_dir : str or None
        Validated output directory (None if validation failed)
    input_dir : str or None
        Input directory (may not exist if auto_create_input=True)
    """
    # Handle output directory
    if args_output_dir:
        output_dir = args_output_dir
        print(f"Using specified output directory: {output_dir}")
    else:
        # Auto-detect using step_info
        output_dir = auto_detect_output_dir(
            required_steps=required_steps,
            verbose=True
        )
        
        if output_dir is None:
            print_detection_error(required_steps)
            return None, None
    
    # Validate prerequisites using step_info
    valid, missing = validate_prerequisites(
        output_dir, required_steps, verbose=True
    )
    
    if not valid:
        print(f"\n❌ ERROR: Missing prerequisite steps")
        print("\nPlease run these steps first:")
        for step in missing:
            print(f"  - {get_step_label(step)}")
        return None, None
    
    # Handle input directory
    if args_input_dir:
        input_dir = args_input_dir
    else:
        input_dir = str(Path(output_dir) / "input")
        if not Path(input_dir).exists() and not auto_create_input:
            print(f"⚠️  Warning: Input directory not found: {input_dir}")
    
    return output_dir, input_dir

# ============================================================================
# SESSION LOADING - Step-independent but uses standard formats
# ============================================================================

def load_sessions_from_folder(folder_path: Path) -> Dict[str, Any]:
    """
    Load neuron sessions from .npy files in a folder.
    
    HANDLES TWO FORMATS:
    -------------------
    Format 1 - Dictionary (standard):
        {
            'centroids_x': array,
            'centroids_y': array,
            'roi_ids': array,
            'subject_id': str/int,
            'session_id': str/int
        }
    
    Format 2 - Raw A matrix (auto-converts):
        A_matrix with shape (n_neurons, height, width)
        Filename must match: {animal_id}_{session_id}_A_final.npy
        Will automatically extract centroids
    
    Args:
        folder_path: Path to folder containing .npy session files
        
    Returns:
        Dictionary containing:
        {
            'sessions': Dict[str, Dict],
            'file_paths': Dict[str, str],
            'session_list': List[str],
            'metadata': List[Dict]
        }
    """
    folder_path = Path(folder_path)
    
    sessions = {}
    file_paths = {}
    session_list = []
    metadata = []
    
    # Find .npy files
    npy_files = sorted(folder_path.glob("*.npy"))
    
    if not npy_files:
        return {
            'sessions': sessions,
            'file_paths': file_paths,
            'session_list': session_list,
            'metadata': metadata,
            'error': f'No .npy files found in {folder_path}'
        }
    
    print(f"\n{'='*60}")
    print(f"Loading sessions from: {folder_path}")
    print(f"{'='*60}")
    
    for npy_file in npy_files:
        try:
            # Load the file
            raw_data = np.load(npy_file, allow_pickle=True)
            data = None
            
            # TRY FORMAT 1: Dictionary
            if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
                try:
                    data = raw_data.item()
                    if not isinstance(data, dict):
                        data = None
                except (ValueError, TypeError):
                    data = None
            
            # TRY FORMAT 2: Raw A matrix
            if data is None and isinstance(raw_data, np.ndarray):
                if raw_data.ndim == 3:
                    print(f"🔄 Converting A matrix: {npy_file.name}")
                    
                    centroids_x, centroids_y = extract_centroids_from_A(raw_data)
                    
                    if len(centroids_x) < 4:
                        print(f"⚠️  Skipped {npy_file.name}: Too few neurons ({len(centroids_x)} < 4)")
                        continue
                    
                    # Parse filename for metadata
                    filename = npy_file.stem
                    
                    # Remove suffixes
                    for suffix in ['_A_final', '_A', '_final']:
                        if filename.endswith(suffix):
                            filename = filename[:-len(suffix)]
                            break
                    
                    # Split into animal_id and session_id
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        animal_id = parts[0]
                        session_id = parts[1]
                    else:
                        print(f"⚠️  Skipped {npy_file.name}: Can't parse IDs")
                        continue
                    
                    # Create dictionary format
                    data = {
                        'centroids_x': centroids_x,
                        'centroids_y': centroids_y,
                        'roi_ids': np.arange(len(centroids_x)),
                        'subject_id': animal_id,
                        'session_id': session_id,
                        'source_format': 'A_matrix',
                        'original_shape': raw_data.shape
                    }
                    
                    print(f"  ✓ Extracted {len(centroids_x)} centroids")
            
            # Skip if still no valid data
            if data is None:
                print(f"⚠️  Skipped {npy_file.name}: Unrecognized format")
                continue
            
            # Validate dictionary structure
            if not isinstance(data, dict):
                print(f"⚠️  Skipped {npy_file.name}: Not a dictionary")
                continue
            
            required = {'centroids_x', 'centroids_y', 'subject_id', 'session_id'}
            if not required.issubset(data.keys()):
                missing = required - set(data.keys())
                print(f"⚠️  Skipped {npy_file.name}: Missing fields: {missing}")
                continue
            
            # Use base filename without suffix for session_key
            session_key = npy_file.stem
            for suffix in ['_A_final', '_A', '_final', '_C_final', '_C_interpolated']:
                if session_key.endswith(suffix):
                    session_key = session_key[:-len(suffix)]
                    break
            
            sessions[session_key] = data
            session_list.append(session_key)
            file_paths[session_key] = str(npy_file)
            
            # Store metadata
            num_rois = len(data.get('centroids_x', []))
            subject = data.get('subject_id', 'Unknown')
            session_id = data.get('session_id', 'Unknown')
            source_fmt = data.get('source_format', 'dictionary')
            
            metadata.append({
                'session_key': session_key,
                'subject_id': subject,
                'session_id': session_id,
                'num_rois': num_rois,
                'file_path': str(npy_file),
                'source_format': source_fmt
            })
            
            print(f"✓ {npy_file.name} ({num_rois} ROIs)")
            
        except Exception as e:
            print(f"❌ Failed to load {npy_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"{'='*60}")
    print(f"Loaded {len(sessions)} sessions")
    print(f"{'='*60}\n")
    
    return {
        'sessions': sessions,
        'file_paths': file_paths,
        'session_list': session_list,
        'metadata': metadata
    }

def extract_centroids_from_A(A_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centroids from spatial footprint matrix.
    
    Parameters
    ----------
    A_matrix : np.ndarray
        Shape (n_neurons, height, width)
        
    Returns
    -------
    centroids_x : np.ndarray
        X coordinates (columns)
    centroids_y : np.ndarray
        Y coordinates (rows)
    """
    n_neurons = A_matrix.shape[0]
    centroids_x = []
    centroids_y = []
    
    for i in range(n_neurons):
        footprint = A_matrix[i]
        
        try:
            # center_of_mass returns (row, col) which is (y, x)
            y, x = center_of_mass(footprint)
            
            # Check for NaN
            if np.isnan(x) or np.isnan(y):
                # Use mean of nonzero pixels
                nonzero_coords = np.argwhere(footprint > 0)
                if len(nonzero_coords) > 0:
                    y = nonzero_coords[:, 0].mean()
                    x = nonzero_coords[:, 1].mean()
                else:
                    continue
            
            centroids_x.append(float(x))
            centroids_y.append(float(y))
            
        except Exception:
            continue
    
    return np.array(centroids_x), np.array(centroids_y)

# ============================================================================
# CONFIG UPDATE - Step-agnostic using step_info
# ============================================================================

def update_config_from_step_path(config, step: float, folder: Path):
    """
    Update pipeline config paths based on loaded step results.
    
    COMPLETELY STEP-AGNOSTIC - uses step_info.py for all metadata!
    
    Args:
        config: PipelineConfig object
        step: Step number
        folder: Path where step results were found
        
    Returns:
        Updated config object
    """
    folder = Path(folder)
    
    # Get all possible directory names for this step from step_info
    valid_names = get_step_dir_aliases(step)
    
    if folder.name in valid_names:
        # We're IN the results folder → parent is base output dir
        config.output_dir = str(folder.parent)
        config.input_dir = str(folder.parent / "input")
    elif any((folder / name).exists() for name in valid_names):
        # Folder CONTAINS a results directory → this IS base output dir
        config.output_dir = str(folder)
        config.input_dir = str(folder / "input")
    else:
        # Assume folder is base output dir
        config.output_dir = str(folder)
        config.input_dir = str(folder / "input")
    
    # Rebuild path objects
    config.input_path = Path(config.input_dir)
    config.output_path = Path(config.output_dir)
    
    # Update intermediate_path (for backward compatibility)
    config.intermediate_path = config.output_path / "step_2_results"
    
    return config

def detect_pipeline_results(folder: Path) -> Dict[str, Any]:
    """
    Auto-detect which pipeline steps have been completed.
    
    COMPLETELY STEP-AGNOSTIC - uses step_info.py for all metadata!
    
    Args:
        folder: Starting folder to search from
        
    Returns:
        Dictionary with detection results:
        {
            'project_root': Path or None,
            'step_1': {'found': bool, 'path': Path, 'files': list},
            'step_1_5': {'found': bool, 'path': Path, 'files': list},
            ...
        }
    """
    folder = Path(folder)
    
    # Search locations
    search_locations = [
        folder,
        folder.parent,
        folder / "Results",
        folder.parent / "Results",
    ]
    
    results = {'project_root': None}
    
    # Initialize entries for all steps from step_info
    for step in get_all_step_numbers():
        step_key = f'step_{str(step).replace(".", "_")}'
        results[step_key] = {
            'found': False,
            'path': None,
            'files': [],
            'label': get_step_label(step),
        }
    
    # Search for project root
    for location in search_locations:
        if not location.exists():
            continue
        
        # Check if any step directories exist here (using step_info)
        found_any = False
        for step in get_all_step_numbers():
            for alias in get_step_dir_aliases(step):
                if (location / alias).exists():
                    found_any = True
                    break
            if found_any:
                break
        
        if found_any:
            results['project_root'] = location
            break
    
    # If no project root found, return early
    if results['project_root'] is None:
        return results
    
    # Check each step using step_info metadata
    for step in get_all_step_numbers():
        step_key = f'step_{str(step).replace(".", "_")}'
        pattern = get_step_file_pattern(step)
        
        # Check all possible directory names for this step
        for alias in get_step_dir_aliases(step):
            step_dir = results['project_root'] / alias
            if not step_dir.exists():
                continue
            
            # Find files matching pattern
            files = list(step_dir.glob(pattern))
            if files:
                results[step_key] = {
                    'found': True,
                    'path': step_dir,
                    'files': files,
                    'label': get_step_label(step),
                }
                break
    
    return results

def extract_animal_session_from_filename(
    filename: str,
    suffix: str = ''
) -> Optional[Tuple[str, str]]:
    """
    Extract animal_id and session_id from filename.
    
    Expected format: {animal_id}_{session_id}{suffix}
    Example: "408021_758519303_centroids_quads" -> ("408021", "758519303")
    
    Parameters
    ----------
    filename : str
        Filename (without extension)
    suffix : str
        Expected suffix to remove (e.g., '_centroids_quads')
        
    Returns
    -------
    (animal_id, session_id) or None
        Extracted IDs or None if parsing fails
    """
    # Remove suffix if present
    if suffix and filename.endswith(suffix):
        filename = filename[:-len(suffix)]
    
    # Split by underscore
    parts = filename.split('_')
    
    if len(parts) >= 2:
        animal_id = parts[0]
        session_id = parts[1]
        return (animal_id, session_id)
    
    return None

# ============================================================================
# PARAMETER VALIDATION - New! Uses PARAMETER_SCHEMAS
# ============================================================================

def validate_config_parameters(config) -> Tuple[bool, List[str]]:
    """
    Validate all config parameters against PARAMETER_SCHEMAS.
    
    Uses step_info.py PARAMETER_SCHEMAS for validation.
    
    Args:
        config: PipelineConfig object
        
    Returns:
        (valid, errors): True if all valid, list of error messages
    """
    errors = []
    
    for param_name, schema in PARAMETER_SCHEMAS.items():
        if not hasattr(config, param_name):
            continue
        
        value = getattr(config, param_name)
        
        # Check nullable
        if value is None:
            if not schema.get('nullable', False):
                errors.append(f"{param_name}: Cannot be None")
            continue
        
        # Check type
        expected_type = schema['type']
        if not isinstance(value, expected_type):
            errors.append(
                f"{param_name}: Expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            continue
        
        # Check numeric ranges
        if expected_type in (int, float):
            if 'min' in schema and value < schema['min']:
                errors.append(
                    f"{param_name}: {value} < minimum {schema['min']}"
                )
            if 'max' in schema and value > schema['max']:
                errors.append(
                    f"{param_name}: {value} > maximum {schema['max']}"
                )
        
        # Check string options
        if expected_type == str and 'options' in schema:
            if value not in schema['options']:
                errors.append(
                    f"{param_name}: '{value}' not in {schema['options']}"
                )
    
    return len(errors) == 0, errors

def get_config_summary(config, step: Optional[float] = None) -> str:
    """
    Generate a human-readable config summary.
    
    Uses step_info.py to organize parameters by step.
    
    Args:
        config: PipelineConfig object
        step: Optional step number to show only relevant parameters
        
    Returns:
        Formatted string summary
    """
    lines = ["Configuration Summary", "=" * 60]
    
    lines.append(f"Input:  {config.input_dir}")
    lines.append(f"Output: {config.output_dir}")
    if hasattr(config, 'animal_id') and config.animal_id:
        lines.append(f"Animal: {config.animal_id}")
    lines.append("")
    
    if step is not None:
        # Show only parameters for this step
        from .step_info import get_step_parameters
        lines.append(f"Parameters for {get_step_label(step)}:")
        lines.append("-" * 60)
        
        for param_name in get_step_parameters(step):
            if hasattr(config, param_name):
                value = getattr(config, param_name)
                schema = get_parameter_schema(param_name)
                desc = schema.get('description', param_name) if schema else param_name
                lines.append(f"  {desc}: {value}")
    else:
        # Show all parameters grouped by step
        from .step_info import STEP_METADATA
        
        for step_num in get_all_step_numbers():
            step_info = STEP_METADATA[step_num]
            params = step_info.get('parameters', [])
            
            if params:
                lines.append(f"{step_info['label']}:")
                lines.append("-" * 60)
                
                for param_name in params:
                    if hasattr(config, param_name):
                        value = getattr(config, param_name)
                        schema = get_parameter_schema(param_name)
                        desc = schema.get('description', param_name) if schema else param_name
                        lines.append(f"  {desc}: {value}")
                
                lines.append("")
    
    return "\n".join(lines)
