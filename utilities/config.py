"""
All parameter metadata comes from step_info.py
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from .step_info import PARAMETER_SCHEMAS, get_parameter_default

class PipelineConfig:
    """
    Minimal pipeline configuration that defers to step_info.py for all metadata.
    
    This class only stores:
    - Directory paths
    - Animal ID
    - Parameter values (using names from PARAMETER_SCHEMAS)
    
    All defaults, types, constraints come from step_info.py
    """
    
    def __init__(
        self,
        input_dir: str = ".",
        output_dir: str = "./Results",
        animal_id: Optional[str] = None,
        **kwargs  # Accept any parameter defined in PARAMETER_SCHEMAS
    ):
        """
        Initialize config with defaults from step_info.py.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input neuron data
        output_dir : str
            Directory for outputs
        animal_id : str, optional
            Specific animal ID to process
        **kwargs : any
            Override any parameter from PARAMETER_SCHEMAS
        """
        # Directories
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.intermediate_path = self.output_path / "intermediate"
        self.plots_dir = self.output_path / "plots"
        
        # Animal ID
        self.animal_id = animal_id
        
        # Initialize all parameters from PARAMETER_SCHEMAS with defaults
        for param_name, schema in PARAMETER_SCHEMAS.items():
            default_value = schema['default']
            # Override with provided kwargs if present
            value = kwargs.get(param_name, default_value)
            setattr(self, param_name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        result = {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'animal_id': self.animal_id,
        }
        # Add all parameters from PARAMETER_SCHEMAS
        for param_name in PARAMETER_SCHEMAS.keys():
            if hasattr(self, param_name):
                result[param_name] = getattr(self, param_name)
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dict"""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save to JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        if self.verbose:
            print(f"Config saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Load from JSON"""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
            
    def _parse_group(self, name: str) -> str:
        import re
        if not self.session_group_regex:
            return ''
        match = re.search(self.session_group_regex, name)
        return match.group(1) if match else ''

    def get_session_pairs(self, animal_id: str, sessions: list) -> list:
        if len(sessions) < 2:
            return []

        def _get_name(s):
            if isinstance(s, dict):
                for key in ('session_name', 'session', 'name', 'session_id'):
                    if key in s:
                        return s[key]
            return str(s)

        from collections import defaultdict
        groups = defaultdict(list)
        for s in sessions:
            name = _get_name(s)
            group_key = self._parse_group(name)
            groups[group_key].append(s)

        pairs = []
        for group_key in sorted(groups):
            group = groups[group_key]
            if len(group) < 2:
                continue

            sorted_group = sorted(group, key=lambda s: _get_name(s))

            if self.session_pair_strategy == 'all_vs_all':
                for i in range(len(sorted_group)):
                    for j in range(i + 1, len(sorted_group)):
                        pairs.append((sorted_group[i], sorted_group[j]))
            else:  # consecutive (default)
                for i in range(len(sorted_group) - 1):
                    pairs.append((sorted_group[i], sorted_group[i + 1]))

        return pairs

    def __repr__(self) -> str:
        """String representation"""
        lines = ["PipelineConfig("]
        lines.append(f"  input_dir='{self.input_dir}',")
        lines.append(f"  output_dir='{self.output_dir}',")
        lines.append(f"  animal_id={self.animal_id},")
        
        # Group parameters by step
        step_groups = {
            'General': ['n_workers', 'verbose', 'skip_existing'],
            'Step 1': [p for p in PARAMETER_SCHEMAS.keys() if p.startswith(('min_tri', 'height', 'min_third', 'parallel', 'per_session', 'triangle', 'max_triangle', 'diagonal', 'quad', 'min_pairwise'))],
            'Step 1.5': [p for p in PARAMETER_SCHEMAS.keys() if p.startswith('calib')],
            'Step 2': ['threshold', 'distance_metric', 'consistency_threshold'],
            'Step 2.5': [p for p in PARAMETER_SCHEMAS.keys() if 'ransac' in p or p == 'transform_residual_threshold' or p == 'min_inlier_ratio'],
            'Step 3': ['target_match_rate', 'motion_correction_px', 'max_rotation_deg', 'use_quad_voting'],
            'Image': ['image_width', 'image_height', 'max_centroid_distance_pct'],
        }
        
        for group_name, params in step_groups.items():
            lines.append(f"  # {group_name}")
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    if isinstance(value, str):
                        lines.append(f"  {param}='{value}',")
                    else:
                        lines.append(f"  {param}={value},")
        
        lines.append(")")
        return "\n".join(lines)

# Utility functions
def create_default_config(input_dir: str, output_dir: str, animal_id: Optional[str] = None) -> PipelineConfig:
    """Create config with defaults from step_info.py"""
    return PipelineConfig(input_dir=input_dir, output_dir=output_dir, animal_id=animal_id)

def load_or_create_config(filepath: str, input_dir: str, output_dir: str) -> PipelineConfig:
    """Load existing or create new config"""
    filepath = Path(filepath)
    if filepath.exists():
        return PipelineConfig.load(str(filepath))
    else:
        config = create_default_config(input_dir, output_dir)
        config.save(str(filepath))
        return config