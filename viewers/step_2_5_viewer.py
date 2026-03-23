"""
Step 2.5 Results Viewer: Geometric Filtering Inspector
"""
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
                             QMessageBox, QSplitter, QTabWidget, QWidget, QTableWidget,
                             QTableWidgetItem)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from utilities import *

def load_npz_safely(filepath, verbose=False):
    """Safely load NPZ file"""
    try:
        return np.load(filepath, allow_pickle=False)
    except Exception as e:
        if verbose:
            print(f"Error loading {filepath}: {e}")
        return None


class Step2_5Viewer(QDialog):
    """Viewer for Step 2.5: RANSAC Geometric Filtering Results"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_pair = None
        self.filtering_data = {}
        
        self.setWindowTitle('Step 2.5: Geometric Filtering Results (RANSAC)')
        self.resize(1600, 1000)
        
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
            '📊 Export Summary': self.export_summary,
            '✖ Close': self.accept
        })
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def create_left_panel(self):
        """Create left panel with stats and session details"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary statistics
        stat_labels = [
            ('n_pairs', 'Session Pairs'),
            ('total_quads', 'Total Quads'),
            ('total_inliers', 'Geometric Inliers'),
            ('avg_inlier_ratio', 'Avg Inlier Ratio'),
            ('avg_translation', 'Avg Translation'),
            ('avg_rotation', 'Avg Rotation'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Filtering Summary', stat_labels
        )
        layout.addWidget(self.stats_group)
        
        # Session pair details table
        table_group = QGroupBox('Session Pair Details')
        table_layout = QVBoxLayout()
        
        self.pairs_table = QTableWidget()
        self.pairs_table.setColumnCount(6)
        self.pairs_table.setHorizontalHeaderLabels([
            'Pair', 'Quads', 'Inliers', 'Inlier %', 'Translation', 'Rotation'
        ])
        self.pairs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pairs_table.setSelectionMode(QTableWidget.SingleSelection)
        self.pairs_table.itemSelectionChanged.connect(self.on_pair_selected)
        table_layout.addWidget(self.pairs_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # RANSAC info
        info_group = QGroupBox('About RANSAC Filtering')
        info_layout = QVBoxLayout()
        info_label = QLabel(
            '<b>How it works:</b><br>'
            '1. Find dominant spatial transformation<br>'
            '2. Identify quads that fit transformation<br>'
            '3. Keep only geometrically consistent quads<br><br>'
            '<b>Parameters:</b><br>'
            f'• Max residual: {getattr(self.config, "transform_residual_threshold", 5.0):.1f}px<br>'
            f'• Min inlier ratio: {getattr(self.config, "min_inlier_ratio", 0.05):.1%}<br>'
            f'• RANSAC iterations: {getattr(self.config, "ransac_iterations", 1000)}<br><br>'
            '<i>No threshold sweeping needed -<br>'
            'geometry determines what passes!</i>'
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet('padding: 10px; background-color: #E8F4F8;')
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        return widget
        
    def create_right_panel(self):
        """Create right panel with plots"""
        tabs = QTabWidget()
        
        # Tab 1: Side-by-side neuron fields
        neuron_tab, self.neuron_figure, self.neuron_canvas = create_matplotlib_tab(
            'Neuron Fields'
        )
        tabs.addTab(neuron_tab, '🔵 Neuron Fields')
        
        # Tab 2: Inliers vs Outliers
        match_tab, self.match_figure, self.match_canvas = create_matplotlib_tab(
            'Match Quality'
        )
        tabs.addTab(match_tab, '✅ Inliers vs Outliers')
        
        # Tab 3: Transform alignment
        align_tab, self.align_figure, self.align_canvas = create_matplotlib_tab(
            'Transform Alignment'
        )
        tabs.addTab(align_tab, '🎯 Transform Alignment')
        
        # Tab 4: Residual heatmap
        residual_spatial_tab, self.residual_spatial_figure, self.residual_spatial_canvas = create_matplotlib_tab(
            'Residual Heatmap'
        )
        tabs.addTab(residual_spatial_tab, '🌡️ Residual Heatmap')
        
        # Tab 5: Grid deformation
        grid_tab, self.grid_figure, self.grid_canvas = create_matplotlib_tab(
            'Grid Deformation'
        )
        tabs.addTab(grid_tab, '📐 Grid Deformation')
        
        # Tab 6: Statistics
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(2.5, self.config.output_dir)

        tabs.addTab(stats_tab, '📋 Statistics')
        
        return tabs
        
    def load_results(self):
        """Load Step 2.5 RANSAC filtering results"""
        # Try both possible directory names
        output_path = Path(self.config.output_dir)
        possible_dirs = [
            output_path / 'step_2_5_results',
            output_path / 'step_2_5',
        ]
        
        step2_5_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                step2_5_dir = dir_path
                break
        
        print(f"\n{'='*60}")
        if step2_5_dir:
            print(f"✓ Found Step 2.5 results: {step2_5_dir}")
        else:
            print(f"✗ Step 2.5 directory not found. Checked:")
            for dir_path in possible_dirs:
                print(f"    - {dir_path} (exists: {dir_path.exists()})")
        
        if step2_5_dir is None:
            QMessageBox.critical(
                self,
                'Step 2.5 Not Found',
                f'Step 2.5 results directory not found.\n\n'
                f'Checked locations:\n' +
                '\n'.join(f'  • {d}' for d in possible_dirs) +
                f'\n\nPlease run Step 2.5 first.'
            )
            return
        
        # Clear previous data
        self.filtering_data.clear()
        self.animal_combo.clear()
        
        # Look for filtered_matches.npz files
        filter_files = sorted(step2_5_dir.glob("*_filtered_matches.npz"))
        
        print(f"Found {len(filter_files)} filtered match files in {step2_5_dir.name}/")
        
        if not filter_files:
            QMessageBox.warning(
                self,
                'No Files Found',
                f'No *_filtered_matches.npz files found in:\n{step2_5_dir}\n\n'
                f'Please run Step 2.5 first.'
            )
            return
        
        # Group by animal
        animals_data = {}
        
        for filter_file in filter_files:
            data = load_npz_safely(filter_file, verbose=False)
            if data is None:
                print(f"  ✗ Failed to load: {filter_file.name}")
                continue
            
            animal_id = decode_npz_string(data.get('animal_id', ''))
            if not animal_id:
                animal_id = filter_file.stem.split('_')[0]
            
            pair_name = decode_npz_string(data.get('pair_name', ''))
            if not pair_name:
                pair_name = filter_file.stem.replace('_filtered_matches', '')
            
            if animal_id not in animals_data:
                animals_data[animal_id] = []
            
            # Extract filtering statistics
            pair_info = {
                'pair_name': pair_name,
                'file': filter_file,
                'n_descriptor_matches': int(data.get('n_descriptor_matches', 0)),
                'n_geometric_inliers': int(data.get('n_matches', len(data.get('match_indices', [])))),
                'filtering_ratio': float(data.get('filtering_ratio', 0.0)),
                'translation_x': float(data.get('translation_x', 0.0)),
                'translation_y': float(data.get('translation_y', 0.0)),
                'translation_magnitude': float(np.linalg.norm([
                    data.get('translation_x', 0.0),
                    data.get('translation_y', 0.0)
                ])),
                'rotation_deg': float(data.get('rotation_deg', 0.0)),
                'scale': float(data.get('scale', 1.0)),
                'mean_residual': float(data.get('mean_residual', 0.0)),
                'median_residual': float(data.get('median_residual', 0.0)),
                # Store data for visualization
                'ref_centroids': data.get('ref_centroids'),
                'tgt_centroids': data.get('tgt_centroids'),
                'match_indices': data.get('match_indices'),
                'transform_matrix': data.get('transform_matrix'),
                'transform_translation': data.get('transform_translation'),
            }
            
            animals_data[animal_id].append(pair_info)
            
            print(f"  ✓ {pair_name}: {pair_info['n_geometric_inliers']} inliers "
                  f"({pair_info['filtering_ratio']:.1%})")
        
        # Store processed data
        for animal_id, pairs in animals_data.items():
            self.filtering_data[animal_id] = {
                'pairs': pairs,
                'n_pairs': len(pairs),
                'total_quads': sum(p['n_descriptor_matches'] for p in pairs),
                'total_inliers': sum(p['n_geometric_inliers'] for p in pairs),
            }
        
        print(f"{'='*60}")
        print(f"Loaded {len(animals_data)} animals")
        print(f"{'='*60}\n")
        
        # Populate combo
        animals = sorted(animals_data.keys())
        self.animal_combo.addItems(animals)
        
        if animals:
            self.animal_combo.setCurrentIndex(0)
        
        self.load_pipeline_stats()
        
    def on_animal_changed(self, animal_id):
        """Handle animal selection"""
        if not animal_id or animal_id not in self.filtering_data:
            return
        
        self.current_animal = animal_id
        self.display_animal(animal_id)
    
    def display_animal(self, animal_id):
        """Display filtering results for selected animal"""
        if animal_id not in self.filtering_data:
            return
        
        data = self.filtering_data[animal_id]
        pairs = data['pairs']
        
        if len(pairs) == 0:
            QMessageBox.warning(
                self, 'No Data',
                f'No filtering data found for {animal_id}'
            )
            return
        
        # Calculate aggregate statistics
        n_pairs = data['n_pairs']
        total_quads = data['total_quads']
        total_inliers = data['total_inliers']
        
        inlier_ratios = [p['filtering_ratio'] for p in pairs]
        avg_inlier_ratio = np.mean(inlier_ratios)
        
        translations = [p['translation_magnitude'] for p in pairs]
        avg_translation = np.mean(translations)
        
        rotations = [abs(p['rotation_deg']) for p in pairs]
        avg_rotation = np.mean(rotations)
        
        # Update statistics
        stats = {
            'n_pairs': n_pairs,
            'total_quads': f"{total_quads:,}",
            'total_inliers': f"{total_inliers:,}",
            'avg_inlier_ratio': f"{avg_inlier_ratio:.1%}",
            'avg_translation': f"{avg_translation:.1f}px",
            'avg_rotation': f"{avg_rotation:.2f}°",
        }
        
        update_stat_labels(self.stat_labels, stats)
        
        # Update table
        self._update_pairs_table(pairs)
        
        # Select first pair for visualization
        if len(pairs) > 0:
            self.pairs_table.selectRow(0)
    
    def _update_pairs_table(self, pairs):
        """Update session pairs table"""
        self.pairs_table.setRowCount(len(pairs))
        
        for i, pair in enumerate(pairs):
            self.pairs_table.setItem(i, 0, QTableWidgetItem(pair['pair_name']))
            self.pairs_table.setItem(i, 1, QTableWidgetItem(
                f"{pair['n_descriptor_matches']:,}"
            ))
            self.pairs_table.setItem(i, 2, QTableWidgetItem(
                f"{pair['n_geometric_inliers']:,}"
            ))
            self.pairs_table.setItem(i, 3, QTableWidgetItem(
                f"{pair['filtering_ratio']:.1%}"
            ))
            self.pairs_table.setItem(i, 4, QTableWidgetItem(
                f"{pair['translation_magnitude']:.1f}px"
            ))
            self.pairs_table.setItem(i, 5, QTableWidgetItem(
                f"{pair['rotation_deg']:.2f}°"
            ))
        
        self.pairs_table.resizeColumnsToContents()
    
    def on_pair_selected(self):
        """Handle pair selection in table"""
        selected_rows = self.pairs_table.selectedItems()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        
        if not self.current_animal:
            return
        
        pairs = self.filtering_data[self.current_animal]['pairs']
        if row >= len(pairs):
            return
        
        pair = pairs[row]
        self.current_pair = pair
        
        # Update all geometric visualization plots
        self.plot_neuron_fields(pair)
        self.plot_inliers_vs_outliers(pair)
        self.plot_transform_alignment(pair)
        self.plot_residual_heatmap(pair)
        self.plot_grid_deformation(pair)
    
    # =========================================================================
    # GEOMETRIC VISUALIZATION PLOTS
    # =========================================================================
    
    def plot_neuron_fields(self, pair):
        """Plot 1: Side-by-side neuron fields showing the raw alignment problem"""
        self.neuron_figure.clear()
        fig = self.neuron_figure
        
        ref_centroids = pair.get('ref_centroids')
        tgt_centroids = pair.get('tgt_centroids')
        
        if ref_centroids is None or tgt_centroids is None:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Position data not available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.neuron_canvas.draw()
            return
        
        # Left: Reference session
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(ref_centroids[:, 0], ref_centroids[:, 1],
                   s=20, c='steelblue', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('X Position (pixels)', fontsize=10)
        ax1.set_ylabel('Y Position (pixels)', fontsize=10)
        ax1.set_title(f'Reference Session\n{len(ref_centroids)} neurons',
                     fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right: Target session
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(tgt_centroids[:, 0], tgt_centroids[:, 1],
                   s=20, c='coral', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('X Position (pixels)', fontsize=10)
        ax2.set_ylabel('Y Position (pixels)', fontsize=10)
        ax2.set_title(f'Target Session\n{len(tgt_centroids)} neurons',
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        fig.tight_layout()
        self.neuron_canvas.draw()
    
    def plot_inliers_vs_outliers(self, pair):
        """Plot matched pairs showing geometric inliers"""
        self.match_figure.clear()
        fig = self.match_figure
        
        ref_centroids = pair.get('ref_centroids')
        tgt_centroids = pair.get('tgt_centroids')
        match_indices = pair.get('match_indices')
        
        if ref_centroids is None or tgt_centroids is None or match_indices is None:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Match data not available',
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.match_canvas.draw()
            return
        
        if len(match_indices) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No inlier matches for this pair',
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.match_canvas.draw()
            return
        
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot all neurons in gray
        ax.scatter(ref_centroids[:, 0], ref_centroids[:, 1],
                s=15, c='lightgray', alpha=0.3, label='Unmatched (ref)')
        ax.scatter(tgt_centroids[:, 0], tgt_centroids[:, 1],
                s=15, c='lightgray', alpha=0.3, label='Unmatched (tgt)')
        
        # Compute residuals for all inlier matches
        A = pair.get('transform_matrix')
        t = pair.get('transform_translation')
        
        if A is not None and t is not None:
            residuals = []
            for match in match_indices:
                ref_quad = match[:4].astype(int)
                tgt_quad = match[4:].astype(int)
                
                ref_center = ref_centroids[ref_quad].mean(axis=0)
                tgt_center = tgt_centroids[tgt_quad].mean(axis=0)
                
                # Compute residual
                predicted = (A @ ref_center) + t
                residual = np.linalg.norm(tgt_center - predicted)
                residuals.append(residual)
            
            residuals = np.array(residuals)
            threshold = getattr(self.config, 'transform_residual_threshold', 5.0)
            
            # Plot matches with color by residual quality
            for i, match in enumerate(match_indices[:200]):  # Limit for visibility
                ref_quad = match[:4].astype(int)
                tgt_quad = match[4:].astype(int)
                
                ref_center = ref_centroids[ref_quad].mean(axis=0)
                tgt_center = tgt_centroids[tgt_quad].mean(axis=0)
                
                # Color by residual quality (all are inliers, but show quality variation)
                color = 'darkgreen' if residuals[i] <= threshold * 0.5 else 'green'
                alpha = 0.8 if residuals[i] <= threshold * 0.5 else 0.5
                
                # Draw arrow from ref to target
                dx = tgt_center[0] - ref_center[0]
                dy = tgt_center[1] - ref_center[1]
                
                ax.arrow(ref_center[0], ref_center[1], dx, dy,
                        head_width=3, head_length=3, fc=color, ec=color,
                        alpha=alpha, linewidth=1.5)
        
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)
        ax.set_title(f'Geometric Inliers (Post-RANSAC)\n{len(match_indices)} matches passed geometric filtering',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        fig.tight_layout()
        self.match_canvas.draw()

    def plot_transform_alignment(self, pair):
        """Plot 3: Apply transform to reference and show overlap with target"""
        self.align_figure.clear()
        fig = self.align_figure
        
        ref_centroids = pair.get('ref_centroids')
        tgt_centroids = pair.get('tgt_centroids')
        A = pair.get('transform_matrix')
        t = pair.get('transform_translation')
        
        if ref_centroids is None or tgt_centroids is None or A is None or t is None:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Transform data not available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.align_canvas.draw()
            return
        
        # Apply transform to reference neurons
        ref_transformed = (A @ ref_centroids.T).T + t
        
        # Plot before and after
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(ref_centroids[:, 0], ref_centroids[:, 1],
                   s=30, c='steelblue', alpha=0.6, label='Reference', marker='o')
        ax1.scatter(tgt_centroids[:, 0], tgt_centroids[:, 1],
                   s=30, c='coral', alpha=0.6, label='Target', marker='s')
        ax1.set_xlabel('X Position (pixels)', fontsize=10)
        ax1.set_ylabel('Y Position (pixels)', fontsize=10)
        ax1.set_title('BEFORE Transform\n(Misaligned)', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(ref_transformed[:, 0], ref_transformed[:, 1],
                   s=30, c='steelblue', alpha=0.6, label='Ref (transformed)', marker='o')
        ax2.scatter(tgt_centroids[:, 0], tgt_centroids[:, 1],
                   s=30, c='coral', alpha=0.6, label='Target', marker='s')
        ax2.set_xlabel('X Position (pixels)', fontsize=10)
        ax2.set_ylabel('Y Position (pixels)', fontsize=10)
        ax2.set_title('AFTER Transform\n(Aligned)', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        transform_info = (f'Translation: ({pair["translation_x"]:.1f}, {pair["translation_y"]:.1f})px  '
                         f'Rotation: {pair["rotation_deg"]:.2f}°  '
                         f'Scale: {pair["scale"]:.3f}')
        fig.suptitle(f'Transform Effect\n{transform_info}',
                    fontsize=12, fontweight='bold')
        
        fig.tight_layout()
        self.align_canvas.draw()
    
    def plot_residual_heatmap(self, pair):
        """Plot 4: Spatial heatmap of residual errors"""
        self.residual_spatial_figure.clear()
        fig = self.residual_spatial_figure
        
        ref_centroids = pair.get('ref_centroids')
        tgt_centroids = pair.get('tgt_centroids')
        match_indices = pair.get('match_indices')
        A = pair.get('transform_matrix')
        t = pair.get('transform_translation')
        
        if (ref_centroids is None or tgt_centroids is None or 
            match_indices is None or A is None or t is None):
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Residual data not available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.residual_spatial_canvas.draw()
            return
        
        if len(match_indices) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No matches to visualize',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.residual_spatial_canvas.draw()
            return
        
        # Compute residuals for all matches
        quad_centers = []
        residuals = []
        
        for match in match_indices:
            ref_quad = match[:4].astype(int)
            tgt_quad = match[4:].astype(int)
            
            ref_center = ref_centroids[ref_quad].mean(axis=0)
            tgt_center = tgt_centroids[tgt_quad].mean(axis=0)
            
            # Compute residual
            predicted = (A @ ref_center) + t
            residual = np.linalg.norm(tgt_center - predicted)
            
            quad_centers.append(ref_center)
            residuals.append(residual)
        
        quad_centers = np.array(quad_centers)
        residuals = np.array(residuals)
        
        # Plot as scatter with color map
        ax = fig.add_subplot(1, 1, 1)
        
        scatter = ax.scatter(quad_centers[:, 0], quad_centers[:, 1],
                            c=residuals, s=80, cmap='RdYlGn_r',
                            vmin=0, vmax=residuals.max(),
                            edgecolors='black', linewidth=0.5, alpha=0.8)
        
        threshold = getattr(self.config, 'transform_residual_threshold', 5.0)
        
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)
        ax.set_title(f'Spatial Distribution of Residual Errors\n'
                    f'Mean: {np.mean(residuals):.2f}px, Median: {np.median(residuals):.2f}px',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Residual Error (pixels)', fontsize=10)
        cbar.ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        
        # Add text annotation
        n_good = np.sum(residuals <= threshold)
        n_total = len(residuals)
        ax.text(0.02, 0.98, 
               f'{n_good}/{n_total} below threshold ({threshold:.1f}px)\n'
               f'Red = high error, Green = low error',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        self.residual_spatial_canvas.draw()
    
    def plot_grid_deformation(self, pair):
        """Plot 5: Grid deformation showing transform effect"""
        self.grid_figure.clear()
        fig = self.grid_figure
        
        A = pair.get('transform_matrix')
        t = pair.get('transform_translation')
        ref_centroids = pair.get('ref_centroids')
        
        if A is None or t is None or ref_centroids is None:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Transform data not available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.grid_canvas.draw()
            return
        
        # Create a grid covering the neuron field
        x_min, y_min = ref_centroids.min(axis=0) - 50
        x_max, y_max = ref_centroids.max(axis=0) + 50
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 15),
            np.linspace(y_min, y_max, 15)
        )
        
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Apply transform to grid
        grid_transformed = (A @ grid_points.T).T + t
        
        grid_x_new = grid_transformed[:, 0].reshape(grid_x.shape)
        grid_y_new = grid_transformed[:, 1].reshape(grid_y.shape)
        
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot original grid in blue
        for i in range(grid_x.shape[0]):
            ax.plot(grid_x[i, :], grid_y[i, :], 'b-', alpha=0.5, linewidth=1)
        for j in range(grid_x.shape[1]):
            ax.plot(grid_x[:, j], grid_y[:, j], 'b-', alpha=0.5, linewidth=1)
        
        # Plot transformed grid in red
        for i in range(grid_x_new.shape[0]):
            ax.plot(grid_x_new[i, :], grid_y_new[i, :], 'r-', alpha=0.7, linewidth=1.5)
        for j in range(grid_x_new.shape[1]):
            ax.plot(grid_x_new[:, j], grid_y_new[:, j], 'r-', alpha=0.7, linewidth=1.5)
        
        # Overlay neurons for reference
        ax.scatter(ref_centroids[:, 0], ref_centroids[:, 1],
                  s=10, c='gray', alpha=0.3, label='Neurons')
        
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)
        ax.set_title(f'Grid Deformation Visualization\n'
                    f'Blue = Original, Red = Transformed',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        
        # Add transform info
        transform_text = (
            f'Translation: ({pair["translation_x"]:.1f}, {pair["translation_y"]:.1f})px\n'
            f'Rotation: {pair["rotation_deg"]:.2f}°\n'
            f'Scale: {pair["scale"]:.3f}'
        )
        ax.text(0.02, 0.98, transform_text,
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        self.grid_canvas.draw()
    
    def export_summary(self):
        """Export filtering summary"""
        if not self.current_animal:
            return
        
        data = self.filtering_data[self.current_animal]
        pairs = data['pairs']
        
        summary = f'RANSAC Geometric Filtering Summary: {self.current_animal}\n\n'
        summary += f'Session Pairs: {data["n_pairs"]}\n'
        summary += f'Total Descriptor Matches: {data["total_quads"]:,}\n'
        summary += f'Geometric Inliers: {data["total_inliers"]:,}\n'
        summary += f'Overall Filter Rate: {data["total_inliers"]/data["total_quads"]:.1%}\n\n'
        
        summary += 'RANSAC Parameters:\n'
        summary += f'  Max Residual: {getattr(self.config, "transform_residual_threshold", 5.0):.1f}px\n'
        summary += f'  Min Inlier Ratio: {getattr(self.config, "min_inlier_ratio", 0.05):.1%}\n'
        summary += f'  RANSAC Iterations: {getattr(self.config, "ransac_iterations", 1000)}\n\n'
        
        summary += 'Per-Pair Details:\n'
        summary += '-' * 80 + '\n'
        
        for pair in pairs:
            summary += f"{pair['pair_name']}:\n"
            summary += f"  Quads: {pair['n_descriptor_matches']:,} → {pair['n_geometric_inliers']:,} ({pair['filtering_ratio']:.1%})\n"
            summary += f"  Transform: T=({pair['translation_x']:.1f}, {pair['translation_y']:.1f})px, R={pair['rotation_deg']:.2f}°, S={pair['scale']:.3f}\n"
            summary += f"  Fit Quality: mean={pair['mean_residual']:.2f}px, median={pair['median_residual']:.2f}px\n\n"
        
        QMessageBox.information(
            self, 'Export Summary',
            summary + '\n(Full export functionality coming soon)'
        )