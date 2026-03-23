"""
Step 1.5 Results Viewer: Descriptor Threshold Calibration Inspector
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, QFormLayout,
                             QMessageBox, QSplitter, QTabWidget, QWidget, QLabel)
from PyQt5.QtCore import Qt

from utilities import *

class Step1_5Viewer(QDialog):
    """Viewer for Step 1.5: Descriptor Threshold Calibration Results"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.calibration_data = {}
        
        # Override tracking
        self.current_threshold_idx = None
        self.optimal_threshold_override = {}
        
        # Match visualization settings
        self.viz_n_examples = 5
        self.viz_seed = 42
        self.viz_show_labels = True
        self.current_match_data = None
        
        self.setWindowTitle('Step 1.5: Descriptor Threshold Calibration Results')
        self.resize(1400, 900)
        
        self.init_ui()
        self.load_results()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Top controls
        self.controls = create_top_controls(
            'Animal Selection',
            combos=[('Animal', self.on_animal_changed)],
            refresh_callback=self.load_results
        )
        self.animal_combo = self.controls.combos[0]
        layout.addWidget(self.controls)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        
        # Bottom buttons
        buttons = create_action_buttons({
            '💾 Export Configuration': self.export_configuration,
            '✖ Close': self.accept
        })
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def create_left_panel(self):
        """Create left panel with stats and controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary statistics
        stat_labels = [
            ('n_pairs', 'Session Pairs'),
            ('n_thresholds', 'Thresholds Tested'),
            ('optimal_threshold', 'Optimal Threshold'),
            ('peak_quality', 'Peak Quality'),
            ('quad_match_rate', 'Quad Match Rate'),
            ('sampled_quads', 'Sampled Quads'),
            ('avg_neurons', 'Avg Neurons (N)'),
            ('raw_matches', 'Raw Matches (at opt)'),
            ('filtered_matches', 'Filtered Matches'),
            ('C_value', 'C Value (sqrt-N scaling)'),
            ('r_squared', 'R² (Fit Quality)'),
        ]
        self.stats_group, self.stat_labels = create_stat_group('Calibration Summary', stat_labels)
        layout.addWidget(self.stats_group)
        
        # Threshold explorer
        explorer_group = self.create_threshold_explorer()
        layout.addWidget(explorer_group)
        
        # Recommendations
        rec_group = QGroupBox('Recommendations')
        rec_layout = QVBoxLayout()
        self.recommendation_label = QLabel('Load data to see recommendations')
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet('padding: 10px; background-color: #E8F4F8;')
        rec_layout.addWidget(self.recommendation_label)
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)
        
        layout.addStretch()
        return widget
    
    def create_threshold_explorer(self):
        """Create threshold explorer group"""
        group = QGroupBox('Descriptor Threshold Explorer')
        layout = QVBoxLayout()
        
        # Slider + Spinbox
        slider_layout, self.threshold_slider, self.threshold_spinbox = create_slider_spinbox_pair(
            'Threshold', (0.0, 1.0), 0.5, decimals=4, step=0.01
        )
        self.threshold_slider.valueChanged.connect(self.on_threshold_slider_changed)
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_spinbox_changed)
        layout.addLayout(slider_layout)
        
        # Override buttons
        override_layout = create_override_buttons(
            refresh_callback=self.refresh_plots_with_current_threshold,
            set_optimal_callback=self.set_current_as_optimal
        )
        layout.addLayout(override_layout)
        
        group.setLayout(layout)
        return group
        
    def create_right_panel(self):
        """Create right panel with plots"""
        tabs = QTabWidget()
        
        # Tab 1: Quality vs Threshold
        quality_tab, self.quality_figure, self.quality_canvas = create_matplotlib_tab('Quality vs Threshold')
        tabs.addTab(quality_tab, '📊 Quality vs Threshold')
        
        # Tab 2: Match Rate vs Threshold
        match_rate_tab, self.match_rate_figure, self.match_rate_canvas = create_matplotlib_tab('Match Rate vs Threshold')
        tabs.addTab(match_rate_tab, '📈 Match Rate')
        
        # Tab 3: Per-pair breakdown
        pairs_tab, self.pairs_figure, self.pairs_canvas = create_matplotlib_tab('Per-Pair Analysis')
        tabs.addTab(pairs_tab, '🔗 Per-Pair Quality')
        
        # Tab 4: Match visualization
        match_tab = self.create_match_viz_tab()
        tabs.addTab(match_tab, '🔍 Example Matches')
        
        # Tab 5: Spatial participation
        participation_tab, self.participation_figure, self.participation_canvas = create_matplotlib_tab('Spatial Participation')
        tabs.addTab(participation_tab, '🎯 Spatial Participation')
        
        # Tab 6: Statistics
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(1.5, self.config.output_dir)
        tabs.addTab(stats_tab, '📋 Statistics')
        
        return tabs
    
    def create_match_viz_tab(self):
        """Create match visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls, self.viz_n_spin, self.viz_seed_spin, self.viz_labels_check = create_match_viz_controls(
            n_examples_changed=lambda v: setattr(self, 'viz_n_examples', v),
            seed_changed=lambda v: setattr(self, 'viz_seed', v),
            refresh_callback=self.refresh_match_viz,
            show_labels_changed=lambda s: setattr(self, 'viz_show_labels', s == Qt.Checked),
            initial_n_examples=self.viz_n_examples,
            initial_seed=self.viz_seed
        )
        layout.addWidget(controls)
        
        # Figure
        _, self.match_viz_figure, self.match_viz_canvas = create_matplotlib_tab('Match Viz')
        layout.addWidget(self.match_viz_canvas)
        
        return tab
        
    def load_results(self):
        """Load Step 1.5 calibration results"""
        step1_5_dir = get_step_results_dir(self.config.output_dir, 1.5)
        
        if not step1_5_dir.exists():
            show_no_results_error(self, step1_5_dir, 'Step 1.5')
            return
        
        # Clear previous data
        self.calibration_data.clear()
        self.animal_combo.clear()
        self.optimal_threshold_override.clear()
        
        # Load calibration files
        calib_files = scan_results_directory(step1_5_dir, '*_threshold_calibration.npz', verbose=True)
        
        if not calib_files:
            show_no_files_error(self, step1_5_dir, '*_threshold_calibration.npz', 'Step 1.5')
            return
        
        # Load each file
        for calib_file in calib_files:
            data = load_npz_safely(calib_file, verbose=True)
            if data is None:
                continue
            
            animal_id = str(data['animal_id'])
            
            self.calibration_data[animal_id] = {
                'file': calib_file,
                'C': float(data['C']),
                'C_std': float(data['C_std']),
                'r_squared': float(data['r_squared']),
                'n_pairs': int(data['n_pairs']),
                'N_values': data['N_values'],
                'tau_values': data['tau_values'],
                # Quality curve data (may not exist in old files)
                'test_thresholds': data.get('test_thresholds', None),
                'per_pair_qualities': data.get('per_pair_qualities', None),
                'mean_quality': data.get('mean_quality', None),
                'pair_names': data.get('pair_names', None),
                'optimal_threshold': float(data['optimal_threshold']) if 'optimal_threshold' in data else None,
                # Match count data (may not exist in old files)
                'n_matches_per_threshold': data.get('n_matches_per_threshold', None),
                'n_filtered_per_threshold': data.get('n_filtered_per_threshold', None),
                'mean_n_matches': data.get('mean_n_matches', None),
                'mean_n_filtered': data.get('mean_n_filtered', None),
                'reference_sizes': data.get('reference_sizes', None),
                # Match visualization data (may not exist in old files)
                'example_matches': data.get('example_matches', None),
                'example_ref_centroids': data.get('example_ref_centroids', None),
                'example_tgt_centroids': data.get('example_tgt_centroids', None),
            }
        
        if not self.calibration_data:
            QMessageBox.warning(self, 'Load Failed', 'Could not load any calibration files')
            return
        
        # Populate combo
        animals = sorted(self.calibration_data.keys())
        self.animal_combo.addItems(animals)
        
        if animals:
            self.animal_combo.setCurrentIndex(0)

        self.load_pipeline_stats()        
    
    def on_animal_changed(self, animal_id):
        """Handle animal selection change"""
        if not animal_id or animal_id not in self.calibration_data:
            return
        
        self.current_animal = animal_id
        self.display_animal(animal_id)
        
    def display_animal(self, animal_id):
        """Display calibration results for selected animal"""
        data = self.calibration_data[animal_id]
        
        # Check the new quality curve data
        has_quality_curves = data['test_thresholds'] is not None and data['mean_quality'] is not None
        
        # Check match count data
        has_match_counts = data['mean_n_matches'] is not None or data['n_matches_per_threshold'] is not None
        
        if has_quality_curves:
            thresholds = data['test_thresholds']
            mean_quality = data['mean_quality']
            
            # Get optimal threshold
            if animal_id in self.optimal_threshold_override:
                optimal_threshold = self.optimal_threshold_override[animal_id]
                optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            elif data['optimal_threshold'] is not None:
                optimal_threshold = data['optimal_threshold']
                optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            else:
                optimal_idx = np.argmax(mean_quality)
                optimal_threshold = thresholds[optimal_idx]
            
            peak_quality = mean_quality[optimal_idx]
            n_thresholds = len(thresholds)
            
            # Configure slider
            self.threshold_slider.setRange(0, len(thresholds) - 1)
            self.threshold_slider.setValue(optimal_idx)
            self.threshold_spinbox.setRange(float(thresholds.min()), float(thresholds.max()))
            self.threshold_spinbox.setValue(optimal_threshold)
            self.current_threshold_idx = optimal_idx
        else:
            # Old format - no quality curves
            optimal_threshold = np.mean(data['tau_values']) if len(data['tau_values']) > 0 else 0.0
            peak_quality = 0.0
            n_thresholds = 0
            self.current_threshold_idx = None
        
        # Compute peak match rate if data available
        peak_match_rate = self._compute_peak_match_rate(data, optimal_idx if has_quality_curves else None)
        
        # Compute match counts at optimal threshold
        avg_neurons = np.mean(data['N_values']) if len(data['N_values']) > 0 else 0
        avg_ref_size = np.mean(data['reference_sizes']) if data['reference_sizes'] is not None and len(data['reference_sizes']) > 0 else 0
        raw_matches_at_opt = 0
        filtered_matches_at_opt = 0

        if has_quality_curves and optimal_idx is not None:
            if data['mean_n_matches'] is not None and optimal_idx < len(data['mean_n_matches']):
                raw_matches_at_opt = data['mean_n_matches'][optimal_idx]
            if data['mean_n_filtered'] is not None and optimal_idx < len(data['mean_n_filtered']):
                filtered_matches_at_opt = data['mean_n_filtered'][optimal_idx]

        # Update statistics
        stats = {
            'n_pairs': data['n_pairs'],
            'n_thresholds': n_thresholds,
            'peak_quality': f'{peak_quality:.1%}' if has_quality_curves else 'N/A',
            'peak_match_rate': f'{peak_match_rate:.1%}' if peak_match_rate is not None else 'N/A',
            'avg_neurons': f'{avg_neurons:.0f}',
            'sampled_quads': f'{avg_ref_size:.0f}', 
            'raw_matches': f'{raw_matches_at_opt:.0f}' if raw_matches_at_opt > 0 else 'N/A',
            'filtered_matches': f'{filtered_matches_at_opt:.0f}' if filtered_matches_at_opt > 0 else 'N/A',
            'C_value': f"{data['C']:.4f} +/- {data['C_std']:.4f}",
            'r_squared': f"{data['r_squared']:.3f}",
            'quad_match_rate': f'{peak_match_rate:.1%}' if peak_match_rate is not None else 'N/A',
        }
        
        # Show if threshold is overridden
        if animal_id in self.optimal_threshold_override:
            self.stat_labels['optimal_threshold'].setText(f'{optimal_threshold:.4f} (OVERRIDE)')
            self.stat_labels['optimal_threshold'].setStyleSheet('color: #00AA00; font-weight: bold;')
        else:
            stats['optimal_threshold'] = f'{optimal_threshold:.4f}'
            self.stat_labels['optimal_threshold'].setStyleSheet('')
        
        update_stat_labels(self.stat_labels, stats)
        
        # Update recommendation
        self.update_recommendation(data, optimal_threshold, peak_quality, peak_match_rate, has_quality_curves, has_match_counts)
        
        # Update plots
        if has_quality_curves:
            self.plot_quality_vs_threshold(data, optimal_idx)
            self.plot_per_pair_quality(data, optimal_idx)
        else:
            self.plot_no_quality_data()
        
        # Plot match rate
        self.plot_match_rate_vs_threshold(data, optimal_idx if has_quality_curves else None)
        
        # Load match data for visualization
        self._load_match_data(data)
        self.refresh_match_viz()
        self.plot_spatial_participation()
    
    def _compute_peak_match_rate(self, data, optimal_idx):
        """Compute match rate at optimal threshold.
        
        Uses reference_sizes (sampled quads) as denominator, NOT N_values (neurons).
        Match rate = filtered_quad_matches / sampled_quads
        """
        if optimal_idx is None:
            return None
        
        avg_ref_size = np.mean(data['reference_sizes']) if data['reference_sizes'] is not None and len(data['reference_sizes']) > 0 else None
        
        if avg_ref_size is None or avg_ref_size <= 0:
            return None
        
        # Try mean_n_filtered first (preferred)
        if data['mean_n_filtered'] is not None and optimal_idx < len(data['mean_n_filtered']):
            mean_filtered = data['mean_n_filtered'][optimal_idx]
            return mean_filtered / avg_ref_size
        
        # Fallback to n_filtered_per_threshold
        if data['n_filtered_per_threshold'] is not None:
            n_filtered = data['n_filtered_per_threshold']
            if n_filtered.ndim == 2 and optimal_idx < n_filtered.shape[1]:
                mean_filtered = np.mean(n_filtered[:, optimal_idx])
                return mean_filtered / avg_ref_size
        
        # Last fallback: raw matches
        if data['mean_n_matches'] is not None and optimal_idx < len(data['mean_n_matches']):
            mean_matches = data['mean_n_matches'][optimal_idx]
            return mean_matches / avg_ref_size
        
        return None
        
    def _load_match_data(self, data):
        """Load match data from calibration results or Step 2"""
        # First try to load from Step 1.5 saved matches
        if (data.get('example_matches') is not None and 
            data.get('example_ref_centroids') is not None and
            data.get('example_tgt_centroids') is not None):
            
            matches = data['example_matches']
            
            if len(matches) > 0:
                match_indices = matches[:, :8].astype(np.int32)
                
                self.current_match_data = {
                    'match_indices': match_indices,
                    'ref_centroids': data['example_ref_centroids'],
                    'target_centroids': data['example_tgt_centroids'],
                }
                print(f"[MATCH VIZ] Loaded {len(matches)} example matches from Step 1.5")
                return
        
        # Fall back to Step 2 results if available
        from utilities import load_match_data_from_step2
        
        try:
            step2_data = load_match_data_from_step2(
                self.config.output_dir, self.current_animal, verbose=True
            )
            if step2_data is not None:
                self.current_match_data = step2_data
                print(f"[MATCH VIZ] Loaded matches from Step 2 results")
                return
        except Exception as e:
            print(f"[MATCH VIZ] Could not load Step 2 data: {e}")
        
        self.current_match_data = None
        print(f"[MATCH VIZ] No match data available for {self.current_animal}")
    
    def refresh_match_viz(self):
        """Refresh match visualization"""
        if self.current_match_data is None:
            self.match_viz_figure.clear()
            ax = self.match_viz_figure.add_subplot(111)
            
            msg = 'No match data available\n\n'
            msg += 'Match visualization requires either:\n'
            msg += '  Re-run Step 1.5 with updated code to save example matches\n'
            msg += '  Or run Step 2 first to generate match data'
            
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.match_viz_canvas.draw()
            return
        
        match_list = convert_match_indices_to_list(self.current_match_data['match_indices'])
        
        if len(match_list) == 0:
            self.match_viz_figure.clear()
            ax = self.match_viz_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No matches found at current threshold',
                   ha='center', va='center', fontsize=12)
            self.match_viz_canvas.draw()
            return
        
        print(f"[MATCH VIZ] Plotting {len(match_list)} matches...")
        
        plot_quad_matches(
            self.match_viz_figure,
            self.match_viz_canvas,
            self.current_match_data['ref_centroids'],
            self.current_match_data['target_centroids'],
            match_list,
            n_examples=self.viz_n_examples,
            seed=self.viz_seed,
            show_labels=self.viz_show_labels,
            title_prefix="Descriptor-Based Quad Matches",
            image_size=self.config.image_width
        )
    
    def plot_spatial_participation(self):
        """Plot spatial distribution of neurons involved in quad matches"""
        self.participation_figure.clear()
        
        if self.current_match_data is None:
            ax = self.participation_figure.add_subplot(111)
            msg = 'No match data available for spatial participation analysis\n\n'
            msg += 'Re-run Step 1.5 or run Step 2 to generate match data'
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, transform=ax.transAxes)
            ax.axis('off')
            self.participation_canvas.draw()
            return
        
        match_indices = self.current_match_data['match_indices']
        ref_centroids = self.current_match_data['ref_centroids']
        tgt_centroids = self.current_match_data['target_centroids']
        
        if len(match_indices) == 0:
            ax = self.participation_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No matches available', ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            self.participation_canvas.draw()
            return
        
        x1 = ref_centroids[:, 1]
        y1 = ref_centroids[:, 0]
        x2 = tgt_centroids[:, 1]
        y2 = tgt_centroids[:, 0]
        
        ref_counts = np.zeros(len(x1), dtype=int)
        tgt_counts = np.zeros(len(x2), dtype=int)
        
        for match in match_indices:
            ref_quad = match[:4]
            tgt_quad = match[4:]
            ref_counts[ref_quad] += 1
            tgt_counts[tgt_quad] += 1
        
        fig = self.participation_figure
        
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_participation_heatmap(ax1, x1, y1, ref_counts, 'Reference Session')
        
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_participation_heatmap(ax2, x2, y2, tgt_counts, 'Target Session')
        
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_radial_distribution(ax3, x1, y1, ref_counts, 'Reference Session')
        
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_radial_distribution(ax4, x2, y2, tgt_counts, 'Target Session')
        
        fig.tight_layout()
        self.participation_canvas.draw()
    
    def _plot_participation_heatmap(self, ax, x, y, counts, title):
        """Plot 2D heatmap of neuron participation"""
        bins = 30
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=counts)
        H_total, _, _ = np.histogram2d(x, y, bins=bins)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            participation_rate = H / H_total
            participation_rate = np.nan_to_num(participation_rate)
        
        im = ax.imshow(participation_rate.T, origin='lower', cmap='hot', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      aspect='auto', interpolation='bilinear')
        
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.set_title(f'{title}\nAvg Match Participation per Neuron', fontsize=11, fontweight='bold')
        
        cbar = self.participation_figure.colorbar(im, ax=ax)
        cbar.set_label('Avg Matches/Neuron', fontsize=9)
    
    def _plot_radial_distribution(self, ax, x, y, counts, title):
        """Plot participation vs distance from center"""
        cx = np.mean(x)
        cy = np.mean(y)
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        n_bins = 20
        max_dist = np.max(distances)
        bins = np.linspace(0, max_dist, n_bins + 1)
        bin_indices = np.digitize(distances, bins)
        
        avg_participation = []
        bin_centers = []
        
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_participation.append(np.mean(counts[mask]))
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        if bin_centers:
            ax.plot(bin_centers, avg_participation, 'o-', linewidth=2, markersize=6, color='steelblue')
            ax.fill_between(bin_centers, 0, avg_participation, alpha=0.3, color='steelblue')
            ax.set_xlabel('Distance from Center (pixels)', fontsize=10)
            ax.set_ylabel('Avg Matches per Neuron', fontsize=10)
            ax.set_title(f'{title}\nCenter vs Periphery Participation', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_dist)
            ax.set_ylim(0, None)
    
    def on_threshold_slider_changed(self, idx):
        """Handle threshold slider change"""
        if self.current_animal is None:
            return
        
        data = self.calibration_data[self.current_animal]
        if data['test_thresholds'] is None:
            return
        
        thresholds = data['test_thresholds']
        if idx >= len(thresholds):
            return
        
        threshold = thresholds[idx]
        self.current_threshold_idx = idx
        
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(threshold)
        self.threshold_spinbox.blockSignals(False)
    
    def on_threshold_spinbox_changed(self, value):
        """Handle threshold spinbox change"""
        if self.current_animal is None:
            return
        
        data = self.calibration_data[self.current_animal]
        if data['test_thresholds'] is None:
            return
        
        thresholds = data['test_thresholds']
        idx = np.argmin(np.abs(thresholds - value))
        
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(idx)
        self.threshold_slider.blockSignals(False)
        
        self.current_threshold_idx = idx
    
    def refresh_plots_with_current_threshold(self):
        """Refresh all plots with current threshold"""
        if not self.current_animal or self.current_threshold_idx is None:
            return
        
        data = self.calibration_data[self.current_animal]
        
        if data['test_thresholds'] is not None:
            self.plot_quality_vs_threshold(data, self.current_threshold_idx)
            self.plot_per_pair_quality(data, self.current_threshold_idx)
            self.plot_match_rate_vs_threshold(data, self.current_threshold_idx)
        
        current_threshold = data['test_thresholds'][self.current_threshold_idx]
        QMessageBox.information(
            self, 'Plots Refreshed',
            f'All plots updated with threshold:\n{current_threshold:.4f}'
        )
    
    def set_current_as_optimal(self):
        """Set current threshold as optimal"""
        if not self.current_animal or self.current_threshold_idx is None:
            return
        
        data = self.calibration_data[self.current_animal]
        if data['test_thresholds'] is None:
            return
        
        new_optimal = data['test_thresholds'][self.current_threshold_idx]
        
        reply = QMessageBox.question(
            self, 'Set Optimal Threshold',
            f'Set threshold {new_optimal:.4f} as optimal for {self.current_animal}?\n\n'
            f'This will save to the NPZ file.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.optimal_threshold_override[self.current_animal] = new_optimal
            
            # Save to NPZ
            step1_5_dir = get_step_results_dir(self.config.output_dir, 'step_1_5')
            npz_path = step1_5_dir / f'{self.current_animal}_threshold_calibration.npz'
            
            success = save_value_to_npz(npz_path, 'optimal_threshold', new_optimal, dtype=np.float64)
            
            if success:
                self.display_animal(self.current_animal)
                QMessageBox.information(
                    self, 'Threshold Updated',
                    f'Optimal threshold updated to {new_optimal:.4f}'
                )
    
    def update_recommendation(self, data, optimal_threshold, peak_quality, peak_match_rate, has_quality_curves, has_match_counts):
        """Update recommendation text"""
        C = data['C']
        r_squared = data['r_squared']
        
        text = f'<b>Optimal Descriptor Threshold: {optimal_threshold:.4f}</b><br><br>'
        
        if has_quality_curves:
            text += f'At this threshold, peak quality is <b>{peak_quality:.1%}</b>.<br>'
            if peak_match_rate is not None:
                text += f'Match rate: <b>{peak_match_rate:.1%}</b> of neurons matched.<br><br>'
            else:
                text += '<br>'
        else:
            text += '<b>Old Format:</b> No quality curves available.<br>'
            text += 'Re-run Step 1.5 to get quality sweep data.<br><br>'
        
        if not has_match_counts:
            text += '<i>Tip: Re-run Step 1.5 with updated code to see match rate curves.</i><br><br>'
        
        text += f'<b>sqrt-N Scaling:</b> C = {C:.4f} (R2 = {r_squared:.3f})<br>'
        text += f'Formula: tau = {C:.4f} x sqrt(N)<br><br>'
        
        text += '<b>Example Predictions:</b><br>'
        text += f'  N = 100 neurons: tau = {C * np.sqrt(100):.4f}<br>'
        text += f'  N = 300 neurons: tau = {C * np.sqrt(300):.4f}<br>'
        text += f'  N = 500 neurons: tau = {C * np.sqrt(500):.4f}<br><br>'
        
        text += '<b>Guidelines:</b><br>'
        text += '  Lower threshold = stricter matching<br>'
        text += '  Higher threshold = more permissive<br>'
        text += '  Look for the peak in the quality curve'
        
        self.recommendation_label.setText(text)
    
    def plot_quality_vs_threshold(self, data, optimal_idx):
        """Plot quality vs threshold"""
        thresholds = data['test_thresholds']
        mean_quality = data['mean_quality']
        quality_pct = mean_quality * 100
        
        plot_line_with_marker(
            self.quality_figure,
            self.quality_canvas,
            thresholds,
            quality_pct,
            'Descriptor Match Quality vs Threshold',
            'Descriptor Threshold',
            'Match Quality (%)',
            marker_x=thresholds[optimal_idx],
            marker_label=f'Selected: {thresholds[optimal_idx]:.4f}'
        )
    
    def plot_match_rate_vs_threshold(self, data, optimal_idx):
        """Plot match rate (n_matches / sampled_quads) vs threshold"""
        self.match_rate_figure.clear()
        ax = self.match_rate_figure.add_subplot(111)
        
        thresholds = data['test_thresholds']
        
        if data['reference_sizes'] is None or len(data['reference_sizes']) == 0:
            ax.text(0.5, 0.5, 
                'No reference_sizes data available\n\n'
                'Re-run Step 1.5 to generate this data.',
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes)
            self.match_rate_canvas.draw()
            return
        
        avg_ref_size = np.mean(data['reference_sizes'])
        if avg_ref_size <= 0:
            return
        
        has_data = False
        
        if data['mean_n_filtered'] is not None:
            mean_filtered = np.array(data['mean_n_filtered'])
            if len(mean_filtered) == len(thresholds):
                match_rate_pct = (mean_filtered / avg_ref_size) * 100
                ax.plot(thresholds, match_rate_pct, 'b-', linewidth=2, 
                    label='Filtered Matches / Sampled Quads')
                has_data = True
        
        if data['mean_n_matches'] is not None:
            mean_matches = np.array(data['mean_n_matches'])
            if len(mean_matches) == len(thresholds):
                raw_rate_pct = (mean_matches / avg_ref_size) * 100
                color = 'orange' if has_data else 'b'
                linestyle = '--' if has_data else '-'
                ax.plot(thresholds, raw_rate_pct, color=color, linestyle=linestyle,
                    linewidth=2, label='Raw Matches / Sampled Quads')
                has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No match count data available',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
            self.match_rate_canvas.draw()
            return
        
        if optimal_idx is not None and thresholds is not None:
            ax.axvline(thresholds[optimal_idx], color='r', linestyle='--',
                    linewidth=2, label=f'Selected: {thresholds[optimal_idx]:.4f}')
        
        ax.set_xlabel('Descriptor Threshold', fontsize=12)
        ax.set_ylabel('Match Rate (%)', fontsize=12)
        ax.set_title(f'Match Rate vs Threshold\n(matches / {avg_ref_size:.0f} sampled quads)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(bottom=0)
        
        self.match_rate_figure.tight_layout()
        self.match_rate_canvas.draw()

    def plot_per_pair_quality(self, data, optimal_idx):
        """Plot quality for each session pair"""
        self.pairs_figure.clear()
        ax = self.pairs_figure.add_subplot(111)
        
        thresholds = data['test_thresholds']
        per_pair_qualities = data['per_pair_qualities']
        pair_names = data['pair_names']
        
        if per_pair_qualities is None or len(per_pair_qualities) == 0:
            ax.text(0.5, 0.5, 'No per-pair data available', 
                   ha='center', va='center', fontsize=12)
            self.pairs_canvas.draw()
            return
        
        n_pairs = per_pair_qualities.shape[0]
        max_pairs_to_plot = min(10, n_pairs)
        
        for i in range(max_pairs_to_plot):
            pair_name = decode_npz_string(pair_names[i]) if pair_names is not None and i < len(pair_names) else f'Pair {i+1}'
            
            values = per_pair_qualities[i, :] * 100
            ax.plot(thresholds, values, 'o-',
                   label=pair_name, markersize=3, linewidth=1, alpha=0.7)
        
        ax.axvline(thresholds[optimal_idx], color='r', linestyle='--',
                  linewidth=2, label=f'Selected: {thresholds[optimal_idx]:.4f}')
        
        ax.set_xlabel('Descriptor Threshold', fontsize=12)
        ax.set_ylabel('Match Quality (%)', fontsize=12)
        ax.set_title(f'Per-Pair Quality (showing {max_pairs_to_plot}/{n_pairs})',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        self.pairs_figure.tight_layout()
        self.pairs_canvas.draw()
    
    def plot_no_quality_data(self):
        """Show placeholder when no quality curves available"""
        for fig, canvas in [(self.quality_figure, self.quality_canvas),
                            (self.pairs_figure, self.pairs_canvas),
                            (self.match_rate_figure, self.match_rate_canvas)]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 
                   'No quality curve data available\n\n'
                   'This file was created with an older version.\n'
                   'Re-run Step 1.5 to generate quality curves.',
                   ha='center', va='center', fontsize=12)
            canvas.draw()
    
    def export_configuration(self):
        """Export optimal configuration"""
        if not self.current_animal:
            return
        
        data = self.calibration_data[self.current_animal]
        optimal_threshold = self.optimal_threshold_override.get(
            self.current_animal,
            data.get('optimal_threshold', np.mean(data['tau_values']))
        )
        
        QMessageBox.information(
            self, 'Configuration Export',
            f'Calibration for {self.current_animal}:\n\n'
            f'Optimal Threshold: {optimal_threshold:.4f}\n'
            f'C Value: {data["C"]:.4f} +/- {data["C_std"]:.4f}\n'
            f'R2: {data["r_squared"]:.3f}\n\n'
            f'Formula: tau = {data["C"]:.4f} x sqrt(N)\n\n'
            f'This is saved in step_1_5_results/\n'
            f'Step 2 will use these calibration values.'
        )