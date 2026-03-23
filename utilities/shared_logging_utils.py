"""
Logging utilities for consistent output formatting.

Provides standardized banners, progress tracking, and timing utilities.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger("neuron_mapping")

def print_step_banner(
    step_name: str,
    step_number: Optional[str] = None,
    width: int = 80
) -> None:
    """
    Print standardized step banner.
    
    Parameters
    ----------
    step_name : str
        Name of the step (e.g., "Parallel Quad Generation")
    step_number : str, optional
        Step number (e.g., "1", "1.5", "2")
    width : int
        Banner width in characters
    """
    separator = "=" * width
    
    if step_number:
        title = f"STEP {step_number}: {step_name}"
    else:
        title = step_name
    
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)

def print_config_summary(config_dict: Dict[str, Any]) -> None:
    """
    Print configuration summary in consistent format.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary to display
    """
    import json
    
    logger.info("")
    logger.info("Configuration:")
    for line in json.dumps(config_dict, indent=2).split('\n'):
        logger.info(f"  {line}")
    logger.info("")

def log_progress(
    current: int,
    total: int,
    message: str = "",
    frequency: int = 10
) -> None:
    """
    Log progress at regular intervals.
    
    Parameters
    ----------
    current : int
        Current iteration (1-indexed)
    total : int
        Total iterations
    message : str
        Additional message to include
    frequency : int
        Log every N percent (default: 10)
    """
    if current == 1 or total <= frequency:
        # Always log first item or if total is small
        pct = 100 * current / total
        logger.info(f"  Progress: {current}/{total} ({pct:.1f}%) {message}")
    else:
        # Log at regular intervals
        interval = max(1, total // frequency)
        if current % interval == 0:
            pct = 100 * current / total
            logger.info(f"  Progress: {current}/{total} ({pct:.1f}%) {message}")

@contextmanager
def time_operation(
    operation_name: str,
    log_start: bool = True,
    log_end: bool = True
):
    """
    Context manager for timing operations.
    
    Parameters
    ----------
    operation_name : str
        Name of operation being timed
    log_start : bool
        Log start message
    log_end : bool
        Log completion with elapsed time
        
    Yields
    ------
    dict
        Dictionary with 'elapsed' key (updated on exit)
        
    Examples
    --------
    >>> with time_operation("Loading data") as timer:
    ...     # do work
    ...     pass
    >>> print(f"Took {timer['elapsed']:.1f}s")
    """
    if log_start:
        logger.info(f"Starting: {operation_name}")
    
    timer = {'start': time.time(), 'elapsed': 0.0}
    
    try:
        yield timer
    finally:
        timer['elapsed'] = time.time() - timer['start']
        if log_end:
            logger.info(
                f"Completed: {operation_name} ({timer['elapsed']:.1f}s)"
            )

def print_summary_stats(
    title: str,
    stats: Dict[str, Any],
    width: int = 80
) -> None:
    """
    Print summary statistics in consistent format.
    
    Parameters
    ----------
    title : str
        Summary title
    stats : dict
        Statistics to display (key: value pairs)
    width : int
        Banner width
    """
    separator = "=" * width
    
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)
    
    for key, value in stats.items():
        # Format large numbers with commas
        if isinstance(value, int) and value > 999:
            logger.info(f"{key}: {value:,}")
        elif isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("")

def print_completion_message(
    step_name: str,
    elapsed_time: float,
    summary_stats: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print standardized completion message.
    
    Parameters
    ----------
    step_name : str
        Name of completed step
    elapsed_time : float
        Total elapsed time in seconds
    summary_stats : dict, optional
        Key statistics to display
    """
    logger.info(f"\n✓ {step_name} complete!")
    logger.info(f"  Time: {elapsed_time:.1f}s")
    
    if summary_stats:
        for key, value in summary_stats.items():
            if isinstance(value, int) and value > 999:
                logger.info(f"  {key}: {value:,}")
            elif isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    from pathlib import Path
    from datetime import datetime
    import logging
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    logger = logging.getLogger('neuron_mapping')
    logger.setLevel(log_level)
    
    # Only add handlers if none exist — if the GUI's GUIHandler is already
    # attached, skip entirely to avoid duplicate output
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    logger.info(f"Log file: {log_file}")
    
    return logger

