"""
Shared Step Statistics 
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from .step_info import *

def load_step_statistics(step: float, output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Unified statistics loader for ANY pipeline step - uses step_info.py metadata"""
    step_info = get_step_info(step)
    if not step_info:
        return {'n_animals': 0, 'n_pairs': 0, 'animals': {}, 'error': f'Unknown step: {step}'}
    
    stats_config = step_info['stats']
    step_dir = get_step_output_dir(step, output_dir)
    
    if not step_dir.exists():
        if verbose: print(f"Step {step} results directory not found: {step_dir}")
        return {'n_animals': 0, 'n_pairs': 0, 'animals': {}, 'error': f'Directory not found: {step_dir}'}
    
    file_pattern = get_step_file_pattern(step)
    npz_files = sorted(step_dir.glob(file_pattern))
    
    if not npz_files:
        if verbose: print(f"No files found matching {file_pattern} in {step_dir}")
        return {'n_animals': 0, 'n_pairs': 0, 'animals': {}, 'error': f'No {file_pattern} files found'}
    
    animals = defaultdict(lambda: _create_animal_dict(stats_config))
    accumulate_fields = stats_config['accumulate_fields']
    global_accumulators = {field: 0 for field in accumulate_fields.keys()}
    n_pairs = 0
    all_extracted_data = []
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            animal_id = _decode_field(data.get('animal_id', npz_path.stem.split('_')[0]))
            extracted = _extract_data(data, npz_path, stats_config)
            all_extracted_data.append(extracted)
            
            item_name_field = stats_config['item_name_field']
            item_name = _decode_field(data.get(item_name_field, npz_path.stem)) if item_name_field else None
            
            per_item_storage = stats_config['per_item_storage']
            if per_item_storage:
                animals[animal_id][per_item_storage][item_name] = extracted
                animals[animal_id]['n_pairs'] += 1
            else:
                animals[animal_id].update(extracted)
            
            for global_field, data_field in accumulate_fields.items():
                value = extracted.get(data_field, 0)
                animals[animal_id][global_field] = animals[animal_id].get(global_field, 0) + value
                global_accumulators[global_field] += value
            
            _handle_temp_accumulators(animals[animal_id], extracted, stats_config)
            n_pairs += 1
            
            if verbose and item_name:
                print(f"  {item_name}: {extracted}")
                
        except Exception as e:
            if verbose: print(f"Error loading {npz_path.name}: {e}")
            continue
    
    if stats_config.get('has_tracking'):
        _load_tracking_data(step_dir, animals, verbose)
    
    aggregate_stats = {}
    aggregate_func_name = stats_config.get('aggregate_func')
    if aggregate_func_name:
        aggregate_func = _get_aggregate_function(aggregate_func_name)
        if aggregate_func:
            aggregate_stats = aggregate_func(animals, all_extracted_data)
    
    result = {
        'n_animals': len(animals),
        'n_pairs': n_pairs,
        'animals': dict(animals),
        **global_accumulators,
        **aggregate_stats
    }
    
    if n_pairs > 0:
        if 'total_matches' in global_accumulators:
            result['mean_matches_per_pair'] = global_accumulators['total_matches'] / n_pairs
        if 'total_neuron_matches' in global_accumulators:
            result['mean_matches_per_pair'] = global_accumulators['total_neuron_matches'] / n_pairs
    
    return result

def _create_animal_dict(stats_config: dict) -> dict:
    """Create initial animal data structure based on config"""
    animal_dict = {'n_pairs': 0}
    for field in stats_config['accumulate_fields'].keys():
        animal_dict[field] = 0
    if stats_config['per_item_storage']:
        animal_dict[stats_config['per_item_storage']] = {}
    return animal_dict

def _decode_field(field: Any) -> str:
    """Decode NPZ string fields"""
    if hasattr(field, 'item'): field = field.item()
    if isinstance(field, bytes): field = field.decode('utf-8')
    return str(field)

def _extract_data(data: dict, npz_path: Path, stats_config: dict) -> dict:
    """Extract step-specific data from NPZ file"""
    extracted = {}
    for field in stats_config['extract_fields']:
        value = data.get(field, None)
        if hasattr(value, 'item'): value = value.item()
        if isinstance(value, bytes): value = value.decode('utf-8')
        if isinstance(value, (np.integer, np.floating)): value = value.item()
        elif value is None: value = 0
        extracted[field] = value
    return extracted

def _handle_temp_accumulators(animal_dict: dict, extracted: dict, stats_config: dict):
    """Handle temporary accumulators used for aggregation"""
    temp_accumulators = stats_config.get('temp_accumulators')
    if not temp_accumulators: return
    
    field_map = {
        'thresholds': 'threshold_used',
        'filtering_ratios': 'filtering_ratio',
        'translations': 'translation_magnitude',
        'rotations': 'rotation_deg',
        'match_rates': 'optimal_rate',
    }
    
    for temp_field in temp_accumulators:
        if temp_field not in animal_dict:
            animal_dict[temp_field] = []
        source_field = field_map.get(temp_field)
        if source_field and source_field in extracted:
            animal_dict[temp_field].append(extracted[source_field])

def _load_tracking_data(step_dir: Path, animals: dict, verbose: bool):
    """Load Step 3 tracking data"""
    for animal_id in animals.keys():
        tracking_file = step_dir / f"{animal_id}_consolidated_tracking.npz"
        if not tracking_file.exists(): continue
        
        try:
            tracking_data = np.load(tracking_file, allow_pickle=True)
            neuron_tracks = tracking_data['neuron_tracks'].item()
            track_lengths = tracking_data['track_lengths']
            n_sessions = int(tracking_data['n_sessions'])
            
            animals[animal_id]['n_tracked_neurons'] = len(neuron_tracks)
            animals[animal_id]['n_sessions'] = n_sessions
            animals[animal_id]['mean_track_length'] = float(np.mean(track_lengths)) if len(track_lengths) > 0 else 0.0
            animals[animal_id]['max_track_length'] = int(np.max(track_lengths)) if len(track_lengths) > 0 else 0
            
            if verbose:
                print(f"  Loaded tracking for {animal_id}: {len(neuron_tracks)} neurons across {n_sessions} sessions")
        except Exception as e:
            if verbose: print(f"  Error loading tracking for {animal_id}: {e}")

def _get_aggregate_function(func_name: str):
    """Get aggregate function by name"""
    return {
        'aggregate_step1_5': aggregate_step1_5,
        'aggregate_step2': aggregate_step2,
        'aggregate_step2_5': aggregate_step2_5,
        'aggregate_step3': aggregate_step3,
    }.get(func_name)

def aggregate_step1_5(animals: dict, all_data: list) -> dict:
    """Aggregate Step 1.5 calibration data"""
    all_C = [a['C'] for a in animals.values() if 'C' in a]
    all_r2 = [a['r_squared'] for a in animals.values() if 'r_squared' in a]
    return {
        'mean_C': float(np.mean(all_C)) if all_C else 0.0,
        'std_C': float(np.std(all_C)) if all_C else 0.0,
        'mean_r_squared': float(np.mean(all_r2)) if all_r2 else 0.0,
    }

def aggregate_step2(animals: dict, all_data: list) -> dict:
    """Aggregate Step 2 matching data"""
    all_thresholds = []
    for a in animals.values():
        if 'thresholds' in a: all_thresholds.extend(a['thresholds'])
    return {
        'mean_threshold': float(np.mean(all_thresholds)) if all_thresholds else 0.0,
        'std_threshold': float(np.std(all_thresholds)) if all_thresholds else 0.0,
    }

def aggregate_step2_5(animals: dict, all_data: list) -> dict:
    """Aggregate Step 2.5 RANSAC filtering data"""
    all_filtering_ratios, all_translations, all_rotations = [], [], []
    for a in animals.values():
        if 'filtering_ratios' in a: all_filtering_ratios.extend(a['filtering_ratios'])
        if 'translations' in a: all_translations.extend(a['translations'])
        if 'rotations' in a: all_rotations.extend(a['rotations'])
    return {
        'mean_filtering_ratio': float(np.mean(all_filtering_ratios)) if all_filtering_ratios else 0.0,
        'mean_translation': float(np.mean(all_translations)) if all_translations else 0.0,
        'mean_rotation': float(np.mean(all_rotations)) if all_rotations else 0.0,
    }

def aggregate_step3(animals: dict, all_data: list) -> dict:
    """Aggregate Step 3 final matching data"""
    all_match_rates = []
    for a in animals.values():
        if 'match_rates' in a: all_match_rates.extend(a['match_rates'])
    return {
        'mean_match_rate': float(np.mean(all_match_rates)) if all_match_rates else 0.0,
        'std_match_rate': float(np.std(all_match_rates)) if all_match_rates else 0.0,
    }

# Convenience wrappers
def load_step1_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    stats = load_step_statistics(1, output_dir, verbose)
    stats['n_sessions'] = stats.pop('n_pairs', 0)
    for animal_data in stats.get('animals', {}).values():
        animal_data['n_sessions'] = animal_data.pop('n_pairs', 0)
    return stats

def load_step1_5_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    return load_step_statistics(1.5, output_dir, verbose)

def load_step2_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    return load_step_statistics(2, output_dir, verbose)

def load_step2_5_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    return load_step_statistics(2.5, output_dir, verbose)

def load_step3_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    return load_step_statistics(3, output_dir, verbose)

def load_all_pipeline_statistics(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Load statistics from all pipeline steps - completely step-agnostic"""
    all_stats = {}
    
    for step in get_all_step_numbers():
        if verbose:
            step_info = get_step_info(step)
            print(f"\nLoading {step_info['label']}...")
        
        step_key = f'step_{str(step).replace(".", "_")}'
        all_stats[step_key] = load_step1_statistics(output_dir, verbose) if step == 1 else load_step_statistics(step, output_dir, verbose)
    
    steps_completed = []
    all_animals = set()
    
    for step in get_all_step_numbers():
        key = f'step_{str(step).replace(".", "_")}'
        if all_stats[key]['n_animals'] > 0:
            steps_completed.append(key)
            if 'animals' in all_stats[key]:
                all_animals.update(all_stats[key]['animals'].keys())
    
    all_stats['summary'] = {
        'total_animals': len(all_animals),
        'pipeline_complete': len(steps_completed) == len(get_all_step_numbers()),
        'steps_completed': steps_completed
    }
    
    return all_stats
