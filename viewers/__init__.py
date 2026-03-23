"""
Stars2Cells Viewer Components
"""

from .step_1_viewer import Step1Viewer
from .step_1_5_viewer import Step1_5Viewer
from .step_2_viewer import Step2Viewer
from .step_2_5_viewer import Step2_5Viewer
from .step_3_viewer import Step3Viewer

__all__ = [
    'Step1Viewer',
    'Step1_5Viewer',
    'Step2Viewer',
    'Step2_5Viewer',
    'Step3Viewer',
]

__version__ = '1.0.0'
__author__ = 'Neumaier Lab'

# Viewer metadata for programmatic access
VIEWER_METADATA = {
    1: {
        'class': Step1Viewer,
        'name': 'Quad Generation Viewer',
        'description': 'Visualize and analyze quad generation results',
        'icon': '📍',
    },
    1.5: {
        'class': Step1_5Viewer,
        'name': 'Descriptor Calibration Viewer',
        'description': 'Inspect √N scaling calibration and C value optimization',
        'icon': '📊',
    },
    2: {
        'class': Step2Viewer,
        'name': 'Matching Inspector',
        'description': 'Examine quad matching results between session pairs',
        'icon': '🔗',
    },
    2.5: {
        'class': Step2_5Viewer,
        'name': 'Shape Sweep Inspector',
        'description': 'Explore threshold optimization and match rate curves',
        'icon': '📈',
    },
    3: {
        'class': Step3Viewer,
        'name': 'Final Matching Viewer',
        'description': 'Review pruned matches and tracking statistics',
        'icon': '✅',
    },
}

def get_viewer_for_step(step):
    """
    Get the appropriate viewer class for a given pipeline step.
    
    Parameters
    ----------
    step : int or float
        Pipeline step number (1, 1.5, 2, 2.5, or 3)
    
    Returns
    -------
    class
        The viewer class for the specified step
    
    Raises
    ------
    ValueError
        If step is not recognized
    
    Examples
    --------
    >>> ViewerClass = get_viewer_for_step(1.5)
    >>> viewer = ViewerClass(config)
    >>> viewer.show()
    """
    if step not in VIEWER_METADATA:
        valid_steps = ', '.join(map(str, sorted(VIEWER_METADATA.keys())))
        raise ValueError(f"Invalid step {step}. Valid steps: {valid_steps}")
    
    return VIEWER_METADATA[step]['class']

def get_viewer_info(step):
    """
    Get metadata about a viewer for a given step.
    
    Parameters
    ----------
    step : int or float
        Pipeline step number
    
    Returns
    -------
    dict
        Dictionary with 'class', 'name', 'description', and 'icon' keys
    """
    if step not in VIEWER_METADATA:
        return None
    return VIEWER_METADATA[step].copy()

def list_available_viewers():
    """
    List all available viewers with their metadata.
    
    Returns
    -------
    dict
        Dictionary mapping step numbers to viewer metadata
    """
    return VIEWER_METADATA.copy()