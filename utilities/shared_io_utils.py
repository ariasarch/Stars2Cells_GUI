"""
I/O utilities for loading and saving pipeline data.

Common functions for NPZ file handling, session data loading,
and JSON summaries.
"""

import gc
import json
import glob
import pickle
import psutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger("neuron_mapping")

def decode_string_field(arr: Any) -> str:
    """
    Safely decode string field from NPZ (handles bytes or str).
    
    Parameters
    ----------
    arr : any
        Field from NPZ file (may be numpy array, bytes, or str)
        
    Returns
    -------
    str
        Decoded string
    """
    try:
        val = arr.item() if hasattr(arr, "item") else arr
    except Exception:
        val = arr
    
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)

def safe_load_npz(file_path: Path, allow_pickle: bool = True) -> Optional[Dict]:
    """
    Load NPZ file with error handling.
    
    Parameters
    ----------
    file_path : Path
        Path to NPZ file
    allow_pickle : bool
        Whether to allow pickle (default: True)
        
    Returns
    -------
    dict or None
        Loaded data or None if failed
    """
    try:
        return np.load(file_path, allow_pickle=allow_pickle)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None

def load_session_data(
    output_dir: str,
    animal_id: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Load all processed session data grouped by animal.

    Reads NPZ files from Step 1: {session_name}_centroids_quads.npz

    Parameters
    ----------
    output_dir : str
        Base output directory
    animal_id : str, optional
        If provided, only load this animal
    verbose : bool
        Verbose logging

    Returns
    -------
    dict
        Mapping of animal_id -> list of session dicts, each containing:
        - animal_id, session, session_name
        - centroids, quad_desc, quad_idx
        - n_neurons, n_quads
    """
    step1_dir = Path(output_dir) / "step_1_results"
    pattern = str(step1_dir / "*_centroids_quads.npz")
    session_files = glob.glob(pattern)

    animals = defaultdict(list)

    if not session_files:
        logger.warning(f"No Step 1 NPZ files found with pattern: {pattern}")
        return animals

    total_quads = 0
    total_neurons = 0

    if verbose:
        logger.info(f"Looking for Step 1 NPZ files: {pattern}")
        logger.info(f"Found {len(session_files)} NPZ files")

    for file_path in session_files:
        npz = safe_load_npz(file_path, allow_pickle=True)
        if npz is None:
            continue

        try:
            aid = str(npz["animal_id"])
            
            # Respect animal filter
            if animal_id and aid != animal_id:
                continue

            session = str(npz["session"])
            session_name = str(npz["session_name"])
            centroids = npz["centroids"]
            quad_desc = npz["quad_desc"]
            quad_idx = npz["quad_idx"]
            n_neurons = int(npz["n_neurons"])
            n_quads = int(npz["n_quads"])

            if verbose:
                logger.info(
                    f"  Loaded {session_name} | animal={aid} "
                    f"| neurons={n_neurons} | quads={n_quads:,}"
                )

            animals[aid].append({
                "animal_id": aid,
                "session": session,
                "session_name": session_name,
                "centroids": centroids,
                "quad_desc": quad_desc,
                "quad_idx": quad_idx,
                "n_neurons": n_neurons,
                "n_quads": n_quads,
            })

            total_neurons += n_neurons
            total_quads += n_quads

        except Exception as e:
            logger.warning(f"Failed to extract data from {file_path}: {e}")

    # Sort sessions by session index per animal
    for aid in animals:
        animals[aid] = sorted(animals[aid], key=lambda x: x["session"])

    if verbose:
        logger.info(
            f"Loaded data for {len(animals)} animals from {len(session_files)} files "
            f"(total_neurons={total_neurons:,}, total_quads={total_quads:,})"
        )

        for aid, sessions in animals.items():
            n_sess = len(sessions)
            n_neur = sum(s["n_neurons"] for s in sessions)
            n_quad = sum(s["n_quads"] for s in sessions)
            logger.info(
                f"  Animal {aid}: {n_sess} sessions "
                f"| neurons={n_neur:,} | quads={n_quad:,}"
            )

    return animals

def discover_animals(
    directory: Path,
    pattern: str = "*_matches_light.npz",
    verbose: bool = True
) -> List[str]:
    """
    Discover unique animal IDs from NPZ files.
    
    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern for files (default: *_matches_light.npz)
    verbose : bool
        Verbose logging
        
    Returns
    -------
    list of str
        Sorted list of unique animal IDs
    """
    files = sorted(directory.glob(pattern))
    animal_ids = set()
    
    for fp in files:
        npz = safe_load_npz(fp, allow_pickle=False)
        if npz is None:
            continue
        
        try:
            animal_id = decode_string_field(npz["animal_id"])
            animal_ids.add(animal_id)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not extract animal_id from {fp.name}: {e}")
    
    result = sorted(animal_ids)
    
    if verbose:
        logger.info(f"Discovered {len(result)} animals in {directory}")
    
    return result

def save_json_summary(
    data: Any,
    output_file: Path,
    verbose: bool = True
) -> None:
    """
    Save data as JSON with consistent formatting.
    
    Parameters
    ----------
    data : any
        Data to save (must be JSON-serializable)
    output_file : Path
        Output file path
    verbose : bool
        Log the save operation
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    if verbose:
        logger.info(f"Saved JSON summary: {output_file}")

def load_json_summary(
    input_file: Path,
    verbose: bool = True
) -> Optional[Any]:
    """
    Load JSON summary with error handling.
    
    Parameters
    ----------
    input_file : Path
        Input file path
    verbose : bool
        Log the load operation
        
    Returns
    -------
    data or None
        Loaded data or None if failed
    """
    if not input_file.exists():
        if verbose:
            logger.error(f"JSON file not found: {input_file}")
        return None
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if verbose:
            logger.info(f"Loaded JSON summary: {input_file}")
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to load {input_file}: {e}")
        return None

def save_intermediate_data(data: Any, filepath: Path, compress: bool = True):
    """Save intermediate data with optional compression."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        import gzip
        with gzip.open(str(filepath) + '.gz', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_intermediate_data(filepath: Path, compressed: bool = True):
    """Load intermediate data."""
    if compressed and filepath.with_suffix('.pkl.gz').exists():
        import gzip
        with gzip.open(str(filepath) + '.gz', 'rb') as f:
            return pickle.load(f)
    elif filepath.exists():
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No data file found at {filepath}")
    
def save_debug_info(data: Dict[str, Any], filename: str, config):
    """Save debug information if debug mode is enabled."""
    if config.debug:
        debug_path = config.debug_dir / filename
        with open(debug_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

def check_memory_requirements() -> Dict[str, float]:
    """Check system memory availability."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }

def estimate_quad_memory(n_neurons: int) -> Dict[str, Any]:
    """Estimate memory requirements for quad generation."""
    # Number of combinations C(n, 4)
    if n_neurons < 4:
        n_quads = 0
    else:
        n_quads = n_neurons * (n_neurons-1) * (n_neurons-2) * (n_neurons-3) // 24
    
    # Each quad: 4 floats (descriptor) + 4 ints (indices) + overhead
    bytes_per_quad = 4 * 8 + 4 * 4 + 32  # ~80 bytes with overhead
    
    # Memory estimates
    quads_memory_gb = (n_quads * bytes_per_quad) / (1024**3)
    
    # Add overhead for processing
    total_memory_gb = quads_memory_gb * 1.5
    
    # Get current memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    return {
        'n_neurons': n_neurons,
        'n_quads': n_quads,
        'bytes_per_quad': bytes_per_quad,
        'quads_memory_gb': quads_memory_gb,
        'total_memory_gb': total_memory_gb,
        'available_gb': available_gb,
        'feasible': total_memory_gb < available_gb * 0.8
    }

def clean_memory():
    """Force garbage collection and clear memory."""
    gc.collect()
