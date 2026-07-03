"""
Shared Viewer Utilities 
"""

import json
import zipfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from PyQt5.QtWidgets import (QGroupBox, QFormLayout, QLabel, QHBoxLayout,
                             QVBoxLayout, QPushButton, QComboBox, QWidget,
                             QMessageBox, QTabWidget, QSpinBox, QCheckBox,
                             QSlider, QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon


from .step_info import get_step_info, get_step_output_dir, get_step_file_pattern, get_all_step_numbers
from .shared_step_stats import (load_step_statistics, load_step1_statistics, load_step1_5_statistics,
                                    load_step2_statistics, load_step2_5_statistics, load_step3_statistics,
                                    load_all_pipeline_statistics)

# ==============================================================================
# Directory and File Utilities
# ==============================================================================

def get_step_results_dir(output_dir: str, step: float) -> Path:
    """Get results directory for a step - uses step_info.py"""
    return get_step_output_dir(step, output_dir)

def scan_results_directory(directory: Path, pattern: str, verbose: bool = True) -> List[Path]:
    """Scan directory for result files"""
    if not directory.exists():
        if verbose: print(f"Directory not found: {directory}")
        return []
    
    files = sorted(directory.glob(pattern))
    if not files: files = sorted(directory.glob(f"*/{pattern}"))
    if verbose: print(f"Found {len(files)} files matching {pattern} in {directory}")
    return files

def load_npz_safely(filepath: Path, verbose: bool = True) -> Optional[np.lib.npyio.NpzFile]:
    """Load NPZ file with error handling"""
    try:
        data = np.load(filepath, allow_pickle=True)
        if verbose: print(f"✓ Loaded {filepath.name}")
        return data
    except Exception as e:
        if verbose: print(f"✗ Error loading {filepath.name}: {e}")
        return None

def load_json_safely(filepath: Path, verbose: bool = True) -> Optional[Any]:
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if verbose: print(f"✓ Loaded {filepath.name}")
        return data
    except Exception as e:
        if verbose: print(f"✗ Error loading {filepath.name}: {e}")
        return None

def decode_npz_string(value: Any) -> str:
    """Decode string value from NPZ (handles bytes, arrays, etc)"""
    if hasattr(value, 'item'): value = value.item()
    if isinstance(value, bytes): return value.decode('utf-8')
    return str(value)

# ==============================================================================
# UI Component Builders
# ==============================================================================

def create_stat_group(title: str, labels: List[Tuple[str, str]]) -> Tuple[QGroupBox, Dict[str, QLabel]]:
    """Create a statistics group box with label/value pairs"""
    group = QGroupBox(title)
    layout = QFormLayout()
    label_dict = {}
    for field_name, display_label in labels:
        value_label = QLabel('--')
        layout.addRow(f'{display_label}:', value_label)
        label_dict[field_name] = value_label
    group.setLayout(layout)
    return group, label_dict

def create_matplotlib_tab(title: str, icon: str = '📊') -> Tuple[QWidget, Figure, FigureCanvasQTAgg]:
    """Create a tab with matplotlib figure"""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    figure = Figure(figsize=(10, 6))
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(canvas)
    return tab, figure, canvas

def create_action_buttons(buttons: Dict[str, Callable], add_stretch: bool = True) -> QWidget:
    """Create action button bar"""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    if add_stretch: layout.addStretch()
    for text, callback in buttons.items():
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)
    return widget

def create_top_controls(title: str, combos: List[Tuple[str, Callable]], refresh_callback: Callable) -> QGroupBox:
    """Create standard top control panel"""
    group = QGroupBox(title)
    layout = QHBoxLayout()
    combo_widgets = []
    for label, callback in combos:
        layout.addWidget(QLabel(f'{label}:'))
        combo = QComboBox()
        combo.currentTextChanged.connect(callback)
        layout.addWidget(combo)
        combo_widgets.append(combo)
    layout.addStretch()
    refresh_btn = QPushButton('🔄 Refresh')
    refresh_btn.clicked.connect(refresh_callback)
    layout.addWidget(refresh_btn)
    group.setLayout(layout)
    group.combos = combo_widgets
    return group

# ==============================================================================
# Statistics Tab Creation - Step-Agnostic
# ==============================================================================

def create_stats_tab(step: float, output_dir: str, parent_widget: Optional[QWidget] = None) -> Tuple[QWidget, QTextEdit, Callable]:
    """Create a statistics display tab for any step - uses step_info.py"""
    step_info = get_step_info(step)
    if not step_info: raise ValueError(f"Unknown step: {step}")
    
    step_loaders = {1: load_step1_statistics, 1.5: load_step1_5_statistics, 2: load_step2_statistics,
                    2.5: load_step2_5_statistics, 3: load_step3_statistics}
    loader_func = step_loaders.get(step, lambda od, v: load_step_statistics(step, od, v))
    
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setStyleSheet('''
        QTextEdit {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;
            border: 1px solid #3c3c3c; padding: 10px; line-height: 1.4;
        }
    ''')
    layout.addWidget(text_edit)
    
    def refresh_stats():
        try:
            stats = loader_func(output_dir, verbose=False)
            text_edit.setPlainText(_format_step_statistics(step, stats))
        except Exception as e:
            import traceback
            text_edit.setPlainText(f"Error loading statistics:\n{e}\n\n{traceback.format_exc()}")
    
    btn_layout = QHBoxLayout()
    btn_layout.addStretch()
    refresh_btn = QPushButton('🔄 Refresh Statistics')
    refresh_btn.clicked.connect(refresh_stats)
    refresh_btn.setStyleSheet('''
        QPushButton { background-color: #0e639c; color: white; padding: 8px 16px;
                      border-radius: 4px; font-weight: bold; }
        QPushButton:hover { background-color: #1177bb; }
    ''')
    btn_layout.addWidget(refresh_btn)
    layout.addLayout(btn_layout)
    
    refresh_stats()
    return tab, text_edit, refresh_stats

def _fmt_stat_value(value: Any) -> str:
    """Format a single stat value for text display.

    Ints get thousands separators; small floats (e.g. calibration C ≈ 0.002)
    keep 4 decimals so they don't collapse to '0.00'; larger floats use 2.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return f"{v:.4f}" if v != 0 and abs(v) < 0.1 else f"{v:.2f}"
    return str(value)

def _format_step_statistics(step: float, stats: dict) -> str:
    """Format statistics for display - step-agnostic formatter"""
    step_info = get_step_info(step)
    lines = ["=" * 60, f"{step_info['icon']} {step_info['label'].upper()}", "=" * 60, "",
             f"Total Animals: {stats.get('n_animals', 0)}"]
    
    if 'n_sessions' in stats: lines.append(f"Total Sessions: {stats.get('n_sessions', 0)}")
    if 'n_pairs' in stats:
        # `stats['n_pairs']` counts result files. For Step 2/2.5/3 that equals
        # the session-pair count (one file per pair). For Step 1.5 there is one
        # file PER ANIMAL, so the file count is really the animal count -- prefer
        # the summed per-animal pair counts, which equal the file count for the
        # other steps and the true session-pair total for Step 1.5.
        summed_pairs = sum(a.get('n_pairs', 0) for a in (stats.get('animals') or {}).values())
        lines.append(f"Total Pairs: {summed_pairs or stats.get('n_pairs', 0)}")
    lines.append("")
    
    stats_config = step_info['stats']
    for field_name in stats_config['accumulate_fields'].keys():
        if field_name in stats:
            value = stats[field_name]
            display_name = field_name.replace('_', ' ').title()
            lines.append(f"{display_name}: {value:,}" if isinstance(value, int) else f"{display_name}: {value:.2f}")
    
    if 'mean_C' in stats: lines.append(f"\nMean C: {stats['mean_C']:.4f} ± {stats.get('std_C', 0):.4f}")
    if 'mean_r_squared' in stats: lines.append(f"Mean R²: {stats['mean_r_squared']:.3f}")
    if 'mean_threshold' in stats: lines.append(f"\nMean Threshold: {stats['mean_threshold']:.4f}")
    if 'mean_filtering_ratio' in stats: lines.append(f"\nMean Filtering Ratio: {stats['mean_filtering_ratio']:.2%}")
    if 'mean_match_rate' in stats: lines.append(f"\nMean Match Rate: {stats['mean_match_rate']:.2%}")
    
    if stats.get('animals'):
        lines.extend(["\n" + "-" * 60, "PER-ANIMAL BREAKDOWN", "-" * 60])
        # Show accumulated totals when a step defines them; otherwise fall back
        # to the directly-extracted per-animal fields. Step 1.5 has no
        # accumulate_fields -- it stores C / R² / n_pairs / optimal_threshold
        # straight onto the animal dict via extract_fields -- so without this
        # fallback the breakdown renders every animal as a blank heading.
        per_animal_fields = (list(stats_config['accumulate_fields'].keys())
                             or list(stats_config.get('extract_fields', [])))
        for animal_id, animal_data in stats['animals'].items():
            lines.append(f"\n{animal_id}:")
            for field_name in per_animal_fields:
                if field_name in animal_data:
                    display_name = field_name.replace('_', ' ').title()
                    lines.append(f"  {display_name}: {_fmt_stat_value(animal_data[field_name])}")
            _append_pair_breakdown(lines, animal_data)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

def _append_pair_breakdown(lines: list, animal_data: dict) -> None:
    """Append a per-pair match-% breakdown (Step 1.5) under an animal heading."""
    breakdown = animal_data.get('pair_breakdown')
    if not breakdown:
        return
    kind = 'raw' if breakdown[0].get('used_raw') else 'filtered'
    opt = animal_data.get('optimal_threshold')
    thr_note = f" @ τ={opt:.4f}" if isinstance(opt, (int, float)) else ""
    lines.append(f"  Per-pair match rate ({kind} matches / sampled quads{thr_note}):")
    for i, pb in enumerate(breakdown, 1):
        ref = int(round(pb.get('ref_size', 0)))
        n_matched = int(round(pb.get('n_matched', 0)))
        n_note = f", N={int(round(pb['N']))}" if pb.get('N') else ""
        lines.append(f"    [{i}] {pb['pair_name']}: {pb['match_pct'] * 100:.1f}%  "
                     f"({n_matched:,}/{ref:,}{n_note})")
    mean_pct = 100.0 * sum(pb['match_pct'] for pb in breakdown) / len(breakdown)
    lines.append(f"    mean: {mean_pct:.1f}%")

def create_pipeline_overview_tab(output_dir: str) -> Tuple[QWidget, QTextEdit, Callable]:
    """Create a tab showing overview statistics from ALL pipeline steps - step-agnostic"""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    header = QLabel('📊 Pipeline Overview')
    header.setStyleSheet('QLabel { font-size: 16px; font-weight: bold; padding: 10px; color: #00AAFF; }')
    layout.addWidget(header)
    
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setStyleSheet('''
        QTextEdit { font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;
                    border: 1px solid #3c3c3c; padding: 10px; }
    ''')
    layout.addWidget(text_edit)
    
    def refresh_overview():
        try:
            all_stats = load_all_pipeline_statistics(output_dir, verbose=False)
            lines = ["=" * 70, "PIPELINE OVERVIEW", "=" * 70, ""]
            summary = all_stats['summary']
            lines.extend([f"Total Animals: {summary['total_animals']}",
                         f"Steps Completed: {', '.join(summary['steps_completed'])}",
                         f"Pipeline Complete: {'✓ Yes' if summary['pipeline_complete'] else '✗ No'}", ""])
            
            for step in get_all_step_numbers():
                step_info = get_step_info(step)
                step_key = f'step_{str(step).replace(".", "_")}'
                if step_key not in all_stats: continue
                
                step_stats = all_stats[step_key]
                lines.extend(["-" * 70, f"{step_info['icon']} {step_info['label'].upper()}", "-" * 70,
                             f"Animals: {step_stats.get('n_animals', 0)}"])
                
                if 'n_sessions' in step_stats: lines.append(f"Sessions: {step_stats['n_sessions']}")
                if 'n_pairs' in step_stats: lines.append(f"Pairs: {step_stats['n_pairs']}")
                
                for field_name in step_info['stats']['accumulate_fields'].keys():
                    if field_name in step_stats:
                        value = step_stats[field_name]
                        display_name = field_name.replace('_', ' ').title()
                        lines.append(f"{display_name}: {value:,}" if isinstance(value, int) else f"{display_name}: {value:.2f}")
                
                if 'mean_C' in step_stats: lines.append(f"Mean C: {step_stats['mean_C']:.4f}")
                if 'mean_threshold' in step_stats: lines.append(f"Mean Threshold: {step_stats['mean_threshold']:.4f}")
                if 'mean_filtering_ratio' in step_stats: lines.append(f"Mean Filtering Ratio: {step_stats['mean_filtering_ratio']:.2%}")
                if 'mean_match_rate' in step_stats: lines.append(f"Mean Match Rate: {step_stats['mean_match_rate']:.2%}")
                lines.append("")
            
            lines.append("=" * 70)
            text_edit.setPlainText("\n".join(lines))
        except Exception as e:
            import traceback
            text_edit.setPlainText(f"Error loading pipeline overview:\n{e}\n\n{traceback.format_exc()}")
    
    btn_layout = QHBoxLayout()
    btn_layout.addStretch()
    refresh_btn = QPushButton('🔄 Refresh Overview')
    refresh_btn.clicked.connect(refresh_overview)
    refresh_btn.setStyleSheet('''
        QPushButton { background-color: #0e639c; color: white; padding: 8px 16px;
                      border-radius: 4px; font-weight: bold; }
        QPushButton:hover { background-color: #1177bb; }
    ''')
    btn_layout.addWidget(refresh_btn)
    layout.addLayout(btn_layout)
    
    refresh_overview()
    return tab, text_edit, refresh_overview

# ==============================================================================
# Match Visualization Controls
# ==============================================================================

def create_match_viz_controls(n_examples_changed: Callable, seed_changed: Callable, refresh_callback: Callable,
                              show_labels_changed: Optional[Callable] = None, initial_n_examples: int = 5,
                              initial_seed: int = 42) -> Tuple[QGroupBox, QSpinBox, QSpinBox, Optional[QCheckBox]]:
    """Create match visualization control panel"""
    group = QGroupBox('Match Visualization Controls')
    layout = QHBoxLayout()
    
    layout.addWidget(QLabel('Examples:'))
    n_spin = QSpinBox()
    n_spin.setRange(1, 50)
    n_spin.setValue(initial_n_examples)
    n_spin.valueChanged.connect(n_examples_changed)
    layout.addWidget(n_spin)
    
    layout.addWidget(QLabel('Seed:'))
    seed_spin = QSpinBox()
    seed_spin.setRange(0, 9999)
    seed_spin.setValue(initial_seed)
    seed_spin.valueChanged.connect(seed_changed)
    layout.addWidget(seed_spin)
    
    labels_check = None
    if show_labels_changed is not None:
        labels_check = QCheckBox('Show Labels')
        labels_check.setChecked(True)
        labels_check.stateChanged.connect(show_labels_changed)
        layout.addWidget(labels_check)
    
    layout.addStretch()
    
    refresh_btn = QPushButton('🔄 Refresh Viz')
    refresh_btn.clicked.connect(refresh_callback)
    refresh_btn.setStyleSheet('''
        QPushButton { background-color: #6600CC; color: white; padding: 6px 12px;
                      border-radius: 4px; font-weight: bold; }
        QPushButton:hover { background-color: #7700DD; }
    ''')
    layout.addWidget(refresh_btn)
    
    group.setLayout(layout)
    return group, n_spin, seed_spin, labels_check

# ==============================================================================
# Slider/Spinbox Synchronization
# ==============================================================================

def create_slider_spinbox_pair(label: str, value_range: Tuple[float, float], initial_value: float,
                               decimals: int = 4, step: float = 0.01, slider_resolution: int = 1000,
                               style_color: str = '#00AAFF') -> Tuple[QHBoxLayout, QSlider, QDoubleSpinBox]:
    """Create synchronized slider and spinbox pair"""
    layout = QHBoxLayout()
    layout.addWidget(QLabel(f'{label}:'))
    
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, slider_resolution)
    slider.setValue(slider_resolution // 2)
    layout.addWidget(slider, stretch=2)
    
    spinbox = QDoubleSpinBox()
    spinbox.setRange(value_range[0], value_range[1])
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setValue(initial_value)
    spinbox.setMinimumWidth(100)
    spinbox.setStyleSheet(f'''
        QDoubleSpinBox {{ background-color: #2A2A2A; color: white; border: 2px solid {style_color};
                          border-radius: 4px; padding: 4px; font-size: 12px; font-weight: bold; }}
        QDoubleSpinBox:focus {{ border: 2px solid {style_color}; }}
    ''')
    layout.addWidget(spinbox)
    
    return layout, slider, spinbox

def sync_slider_to_spinbox(slider: QSlider, spinbox: QDoubleSpinBox, value_range: Tuple[float, float],
                           slider_resolution: int = 1000) -> None:
    """Update spinbox when slider changes"""
    slider_val = slider.value()
    value = value_range[0] + (slider_val / slider_resolution) * (value_range[1] - value_range[0])
    spinbox.blockSignals(True)
    spinbox.setValue(value)
    spinbox.blockSignals(False)

def sync_spinbox_to_slider(spinbox: QDoubleSpinBox, slider: QSlider, value_range: Tuple[float, float],
                           slider_resolution: int = 1000) -> None:
    """Update slider when spinbox changes"""
    value = spinbox.value()
    slider_val = int(((value - value_range[0]) / (value_range[1] - value_range[0])) * slider_resolution)
    slider.blockSignals(True)
    slider.setValue(slider_val)
    slider.blockSignals(False)

# ==============================================================================
# Override Control Buttons
# ==============================================================================

def create_override_buttons(refresh_callback: Callable, set_optimal_callback: Callable) -> QHBoxLayout:
    """Create "Refresh Plots" and "Set as Optimal" buttons"""
    layout = QHBoxLayout()
    
    refresh_btn = QPushButton('🔄 Refresh Plots')
    refresh_btn.clicked.connect(refresh_callback)
    refresh_btn.setStyleSheet('''
        QPushButton { background-color: #0088DD; color: white; padding: 6px 12px;
                      border-radius: 4px; font-weight: bold; }
        QPushButton:hover { background-color: #00AAFF; }
    ''')
    layout.addWidget(refresh_btn)
    
    set_optimal_btn = QPushButton('⭐ Set as Optimal')
    set_optimal_btn.clicked.connect(set_optimal_callback)
    set_optimal_btn.setStyleSheet('''
        QPushButton { background-color: #00AA00; color: white; padding: 6px 12px;
                      border-radius: 4px; font-weight: bold; }
        QPushButton:hover { background-color: #00CC00; }
    ''')
    layout.addWidget(set_optimal_btn)
    
    return layout

# ==============================================================================
# NPZ File Operations
# ==============================================================================

def save_value_to_npz(npz_path: Path, key: str, new_value: Any, dtype: type = np.float32,
                     verbose: bool = True) -> bool:
    """Save a single value to NPZ file (update or add)"""
    if not npz_path.exists():
        if verbose: print(f"Warning: NPZ file not found: {npz_path}")
        return False
    
    try:
        old_data = np.load(npz_path, allow_pickle=True)
        new_data = {k: (dtype(new_value) if k == key else old_data[k]) for k in old_data.files}
        if key not in old_data.files: new_data[key] = dtype(new_value)
        np.savez_compressed(npz_path, **new_data)
        if verbose: print(f"✓ Saved {key}={new_value} to {npz_path.name}")
        return True
    except Exception as e:
        if verbose: print(f"Error saving to NPZ: {e}")
        return False

# ==============================================================================
# Match Data Loading - Step-Agnostic
# ==============================================================================

def load_match_data_from_step(output_dir: str, step: float, animal_id: str,
                              verbose: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """Load match data from a step for visualization - uses step_info.py"""
    step_dir = get_step_results_dir(output_dir, step)
    pattern = get_step_file_pattern(step)
    pair_pattern = f'{animal_id}_*_to_*{pattern[1:]}'
    step_files = scan_results_directory(step_dir, pair_pattern, verbose=verbose)
    
    if not step_files:
        if verbose: print(f"[MATCH LOAD] No match files found for animal {animal_id} in step {step}")
        return None
    
    if verbose: print(f"[MATCH LOAD] Found {len(step_files)} files for {animal_id}")
    
    best_file, max_matches, best_data = None, 0, None
    
    for match_file in step_files:
        try:
            match_data = load_npz_safely(match_file, verbose=False)
            if match_data is None: continue
            
            n_matches = len(match_data.get('match_indices', []))
            if verbose:
                pair_name = decode_npz_string(match_data.get('pair_name', match_file.stem))
                print(f"[MATCH LOAD]   {pair_name}: {n_matches} matches")
            
            if n_matches > max_matches:
                max_matches = n_matches
                best_file = match_file
                best_data = {
                    'ref_centroids': match_data['ref_centroids'],
                    'target_centroids': match_data['target_centroids'],
                    'match_indices': match_data['match_indices'],
                }
        except (KeyError, ValueError, zipfile.BadZipFile, EOFError) as e:
            if verbose: print(f"[MATCH LOAD] Skipping corrupted file {match_file.name}: {e}")
            continue
    
    if best_data is None or max_matches == 0:
        if verbose: print(f"[MATCH LOAD] No files with matches found for animal {animal_id}")
        return None
    
    if verbose: print(f"[MATCH LOAD] ✓ Selected {best_file.name} ({max_matches} matches)")
    return best_data

def load_match_data_from_step2(output_dir: str, animal_id: str, verbose: bool = False):
    """Backward compatibility wrapper for Step 2 match loading"""
    return load_match_data_from_step(output_dir, 2, animal_id, verbose)

def convert_match_indices_to_list(match_indices: np.ndarray) -> List[Tuple]:
    """Convert match indices array to match list format for visualization"""
    match_list = []
    for row in match_indices:
        ref_idx = tuple(row[:4].astype(int))
        tgt_idx = tuple(row[4:8].astype(int))
        distance = float(row[8]) if len(row) > 8 else 0.0
        match_list.append((ref_idx, tgt_idx, None, None, distance))
    return match_list

# ==============================================================================
# Match Visualization Plotting
# ==============================================================================

def plot_quad_matches(figure: Figure, canvas: FigureCanvasQTAgg, ref_centroids: np.ndarray,
                     tgt_centroids: np.ndarray, match_data: List[Tuple], n_examples: int = 5,
                     seed: int = 42, show_labels: bool = True, title_prefix: str = "Quad Matches",
                     image_size: int = 640) -> None:
    """Plot example quad matches showing reference and target quads OVERLAID"""
    figure.clear()
    
    if not match_data:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No matches to visualize', ha='center', va='center', fontsize=14)
        canvas.draw()
        return
    
    rng = np.random.default_rng(seed=seed)
    n_matches = len(match_data)
    n_show = min(n_examples, n_matches)
    
    if n_matches > n_show:
        indices = rng.choice(n_matches, size=n_show, replace=False)
        selected_matches = [match_data[i] for i in sorted(indices)]
    else:
        selected_matches = match_data[:n_show]
    
    n_cols = min(5, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    
    for i, match in enumerate(selected_matches):
        ref_idx, tgt_idx, ref_desc, tgt_desc, dist = match
        ax = figure.add_subplot(n_rows, n_cols, i + 1)
        plot_overlaid_quad(ax, ref_centroids, tgt_centroids, ref_idx, tgt_idx,
                          title=f'Match #{i+1} (d={dist:.3f})', show_labels=show_labels,
                          image_size=image_size)
    
    figure.suptitle(f'{title_prefix} – Showing {n_show} of {n_matches} matches',
                    fontsize=14, fontweight='bold')
    figure.tight_layout()
    canvas.draw()

def plot_overlaid_quad(ax, ref_centroids: np.ndarray, tgt_centroids: np.ndarray,
                       ref_quad_indices: np.ndarray, tgt_quad_indices: np.ndarray,
                       title: str, show_labels: bool = True, image_size: int = 640) -> None:
    """Plot reference and target quads overlaid on the same axes"""
    if isinstance(ref_quad_indices, tuple): ref_quad_indices = np.array(ref_quad_indices, dtype=int)
    if isinstance(tgt_quad_indices, tuple): tgt_quad_indices = np.array(tgt_quad_indices, dtype=int)
    
    ref_quad_pts = ref_centroids[ref_quad_indices]
    tgt_quad_pts = tgt_centroids[tgt_quad_indices]
    
    ax.scatter(ref_centroids[:, 0], ref_centroids[:, 1], c='lightgray', s=10, alpha=0.2, zorder=1, label='All neurons')
    ax.scatter(ref_quad_pts[:, 0], ref_quad_pts[:, 1], c='blue', s=150, alpha=0.7, zorder=3,
              edgecolors='darkblue', linewidths=2, marker='o', label='Reference')
    ax.scatter(tgt_quad_pts[:, 0], tgt_quad_pts[:, 1], c='red', s=150, alpha=0.7, zorder=4,
              edgecolors='darkred', linewidths=2, marker='s', label='Target')
    
    ref_poly = Polygon(ref_quad_pts, fill=False, edgecolor='blue', linewidth=2.5, alpha=0.8, zorder=2, linestyle='-')
    ax.add_patch(ref_poly)
    tgt_poly = Polygon(tgt_quad_pts, fill=False, edgecolor='red', linewidth=2.5, alpha=0.8, zorder=2, linestyle='--')
    ax.add_patch(tgt_poly)
    
    if show_labels:
        for j, idx in enumerate(ref_quad_indices):
            ax.text(ref_quad_pts[j, 0], ref_quad_pts[j, 1], f'R{idx}', fontsize=7, color='white',
                   ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle='circle', facecolor='blue', edgecolor='darkblue', alpha=0.9), zorder=5)
        for j, idx in enumerate(tgt_quad_indices):
            ax.text(tgt_quad_pts[j, 0], tgt_quad_pts[j, 1], f'T{idx}', fontsize=7, color='white',
                   ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle='square', facecolor='red', edgecolor='darkred', alpha=0.9), zorder=5)
    
    ref_centroid = ref_quad_pts.mean(axis=0)
    tgt_centroid = tgt_quad_pts.mean(axis=0)
    ax.plot(ref_centroid[0], ref_centroid[1], 'bx', markersize=12, markeredgewidth=3, zorder=6)
    ax.plot(tgt_centroid[0], tgt_centroid[1], 'rx', markersize=12, markeredgewidth=3, zorder=6)
    ax.plot([ref_centroid[0], tgt_centroid[0]], [ref_centroid[1], tgt_centroid[1]],
           'k--', linewidth=1.5, alpha=0.5, zorder=1)
    
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

# ==============================================================================
# Error Dialogs - Step-Agnostic
# ==============================================================================

def show_no_results_error(parent: QWidget, directory: Path, step: float) -> None:
    """Show error dialog for missing results directory"""
    step_info = get_step_info(step)
    QMessageBox.warning(parent, 'No Results',
                       f'{step_info["label"]} results directory not found:\n{directory}\n\n'
                       f'Please run {step_info["label"]} first.')

def show_no_files_error(parent: QWidget, directory: Path, pattern: str, step: float) -> None:
    """Show error dialog for no files found"""
    step_info = get_step_info(step)
    QMessageBox.warning(parent, 'No Results',
                       f'No files ({pattern}) found in:\n{directory}\n\n'
                       f'Please run {step_info["label"]} first.')

# ==============================================================================
# Data Processing Utilities
# ==============================================================================

def update_stat_labels(label_dict: Dict[str, QLabel], values: Dict[str, Any]) -> None:
    """Update statistics labels with new values"""
    for field, label in label_dict.items():
        if field in values:
            value = values[field]
            if isinstance(value, float):
                text = f"{value:.2e}" if abs(value) < 0.01 or abs(value) > 1000 else f"{value:.2f}"
            elif isinstance(value, int):
                text = f"{value:,}" if value > 999 else str(value)
            else:
                text = str(value)
            label.setText(text)
        else:
            label.setText('--')

# ==============================================================================
# Common Plotting Functions
# ==============================================================================

def plot_histogram(figure: Figure, canvas: FigureCanvasQTAgg, data: np.ndarray, bins: int,
                  title: str, xlabel: str, color: str = 'steelblue',
                  show_mean: bool = True, show_median: bool = True) -> None:
    """Plot histogram with optional mean/median lines"""
    figure.clear()
    ax = figure.add_subplot(111)
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        canvas.draw()
        return
    
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color=color)
    
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    if show_median:
        median_val = np.median(data)
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if show_mean or show_median: ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    canvas.draw()

def plot_bar_chart(figure: Figure, canvas: FigureCanvasQTAgg, labels: List[str], values: List[float],
                   title: str, xlabel: str, ylabel: str, color: str = 'purple',
                   rotate_labels: bool = True) -> None:
    """Plot bar chart with labels"""
    figure.clear()
    ax = figure.add_subplot(111)
    
    if len(labels) == 0 or len(values) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        canvas.draw()
        return
    
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, alpha=0.7, color=color)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45 if rotate_labels else 0, ha='right' if rotate_labels else 'center')
    ax.grid(True, alpha=0.3, axis='y')
    canvas.draw()

def plot_line_with_marker(figure: Figure, canvas: FigureCanvasQTAgg, x_data: np.ndarray,
                          y_data: np.ndarray, title: str, xlabel: str, ylabel: str,
                          marker_x: Optional[float] = None, marker_label: Optional[str] = None,
                          color: str = 'blue') -> None:
    """Plot line with optional vertical marker"""
    figure.clear()
    ax = figure.add_subplot(111)
    
    if len(x_data) == 0 or len(y_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        canvas.draw()
        return
    
    ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=4, color=color)
    
    if marker_x is not None:
        label = marker_label or f'Selected: {marker_x:.4f}'
        ax.axvline(marker_x, color='r', linestyle='--', linewidth=2, label=label)
        ax.legend()
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    canvas.draw()

def create_pyqtgraph_tab(title: str) -> Tuple[QWidget, pg.PlotWidget]:
    """Create a tab with pyqtgraph plot widget"""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    plot = pg.PlotWidget()
    plot.setBackground('w')
    layout.addWidget(plot)
    return tab, plot

def setup_pyqtgraph_plot(plot: pg.PlotWidget, title: str, xlabel: str, ylabel: str) -> None:
    """Configure pyqtgraph plot with labels"""
    plot.setTitle(title, size='13pt')
    plot.setLabel('bottom', xlabel, **{'font-size': '12pt'})
    plot.setLabel('left', ylabel, **{'font-size': '12pt'})
    plot.showGrid(x=True, y=True, alpha=0.3)

def plot_histogram_with_stats(
    ax,
    data: np.ndarray,
    bins: int = 30,
    title: str = '',
    xlabel: str = '',
    ylabel: str = 'Frequency',
    show_mean: bool = True,
    show_median: bool = True
) -> None:
    """
    Plot histogram with optional statistics - takes axis directly.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    data : array
        Data to histogram
    bins : int or array
        Histogram bins
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    show_mean : bool
        Show mean line
    show_median : bool
        Show median line
    """
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
    
    if show_median:
        median_val = np.median(data)
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    if show_mean or show_median:
        ax.legend()
    
    ax.grid(True, alpha=0.3)