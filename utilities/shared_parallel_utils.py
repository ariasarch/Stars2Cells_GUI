"""
Parallel processing utilities for multi-animal pipeline execution.

Provides standardized patterns for ProcessPoolExecutor usage,
worker function handling, and result collection.
"""

import logging
import multiprocessing
from typing import List, Dict, Any, Callable, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("neuron_mapping")

def compute_max_workers(
    cpu_fraction: float = 0.25,
    min_workers: int = 1,
    max_workers: Optional[int] = None
) -> int:
    """
    Compute number of parallel workers based on CPU count.
    
    Parameters
    ----------
    cpu_fraction : float
        Fraction of CPUs to use (default: 0.25 = 25%)
    min_workers : int
        Minimum workers to return (default: 1)
    max_workers : int, optional
        Maximum workers to allow (overrides calculation if provided)
        
    Returns
    -------
    int
        Number of workers to use
    """
    if max_workers is not None and max_workers > 0:
        return max(min_workers, max_workers)
    
    n_cpus = multiprocessing.cpu_count()
    calculated = max(min_workers, int(n_cpus * cpu_fraction))
    
    return calculated

def run_parallel_animals(
    worker_func: Callable,
    args_list: List[Tuple],
    max_workers: Optional[int] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run parallel processing for multiple animals.
    
    Standard pattern for executing worker functions across animals
    with progress logging and error handling.
    
    Parameters
    ----------
    worker_func : callable
        Worker function that takes args tuple and returns dict
    args_list : list of tuple
        List of argument tuples for each worker
    max_workers : int, optional
        Number of parallel workers (auto-computed if None)
    verbose : bool
        Verbose logging
        
    Returns
    -------
    list of dict
        Results from all workers
    """
    if max_workers is None or max_workers <= 0:
        max_workers = compute_max_workers(cpu_fraction=0.25)
    
    if verbose:
        logger.info(f"Running {len(args_list)} tasks with {max_workers} workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(worker_func, args): args
            for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        total = len(futures)
        
        for fut in as_completed(futures):
            args = futures[fut]
            
            try:
                result = fut.result()
                results.append(result)
                completed += 1
                
                if verbose:
                    # Try to extract animal_id from result for logging
                    animal_id = result.get('animal_id', 'unknown')
                    logger.info(
                        f"[MAIN] Completed {completed}/{total}: {animal_id}"
                    )
                    
            except Exception as e:
                # Try to extract identifying info from args
                try:
                    identifier = args[0] if args else "unknown"
                except Exception:
                    identifier = "unknown"
                
                logger.error(
                    f"[MAIN] Error in worker {identifier}: {e}",
                    exc_info=True
                )
    
    if verbose:
        logger.info(f"Parallel execution complete: {len(results)}/{total} succeeded")
    
    return results

def log_worker_start(
    worker_name: str,
    animal_id: str,
    extra_info: Optional[Dict] = None
) -> None:
    """
    Log standardized worker start message.
    
    Parameters
    ----------
    worker_name : str
        Name of the worker/step
    animal_id : str
        Animal being processed
    extra_info : dict, optional
        Additional info to log
    """
    pid = multiprocessing.current_process().pid
    msg = f"[PID={pid}] Starting {worker_name} for animal {animal_id}"
    
    if extra_info:
        info_str = ", ".join(f"{k}={v}" for k, v in extra_info.items())
        msg += f" ({info_str})"
    
    logger.info(msg)

def log_worker_finish(
    worker_name: str,
    animal_id: str,
    result: Dict[str, Any]
) -> None:
    """
    Log standardized worker completion message.
    
    Parameters
    ----------
    worker_name : str
        Name of the worker/step
    animal_id : str
        Animal that was processed
    result : dict
        Result dictionary with metrics
    """
    pid = multiprocessing.current_process().pid
    msg = f"[PID={pid}] Finished {worker_name} for animal {animal_id}"
    
    # Extract common metrics
    metrics = []
    for key in ['n_pairs', 'total_matches', 'n_total_matches', 'n_sessions']:
        if key in result:
            metrics.append(f"{key}={result[key]}")
    
    if metrics:
        msg += f" | {', '.join(metrics)}"
    
    logger.info(msg)