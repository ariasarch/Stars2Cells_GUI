"""
Step 2 Results Viewer: Matching Inspector (REFACTORED)
Visualize and analyze quad matching results between session pairs

CHANGES FROM ORIGINAL:
- Uses shared viewer utilities for UI creation
- Simplified file loading with error handling
- Reduced code duplication
- Spatial plot draws actual quad polygons, not just centroid lines
- Displacement tab removed
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
                             QMessageBox, QSplitter, QTabWidget, QWidget)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

from utilities import *


class Step2Viewer(QDialog):
    """Viewer for Step 2: Matching Results"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_pair = None
        self.match_data = {}
        self.animal_pairs = {}

        self.setWindowTitle('Step 2: Quad Matching Results')
        self.resize(1400, 900)

        self.init_ui()
        self.load_results()

    def init_ui(self):
        layout = QVBoxLayout()

        # Top controls
        self.controls = create_top_controls(
            'Session Pair Selection',
            combos=[
                ('Animal', self.on_animal_changed),
                ('Pair', self.on_pair_changed),
            ],
            refresh_callback=self.load_results,
        )
        self.animal_combo, self.pair_combo = self.controls.combos
        layout.addWidget(self.controls)

        # Sample-match controls
        match_ctl = QHBoxLayout()
        match_ctl.addWidget(QLabel('Quad matches to draw:'))
        self.n_matches_spin = QSpinBox()
        self.n_matches_spin.setRange(0, 50000)
        self.n_matches_spin.setValue(200)
        self.n_matches_spin.setSingleStep(50)
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(400)
        self._redraw_timer.timeout.connect(self._redraw_spatial)
        self.n_matches_spin.valueChanged.connect(lambda: self._redraw_timer.start())
        match_ctl.addWidget(self.n_matches_spin)
        self.reshuffle_btn = QPushButton('Reshuffle')
        self.reshuffle_btn.clicked.connect(self._redraw_spatial)
        match_ctl.addWidget(self.reshuffle_btn)
        match_ctl.addStretch()
        layout.addLayout(match_ctl)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Bottom buttons
        buttons = create_action_buttons({
            '💾 Export Summary': self.export_summary,
            '✖ Close': self.accept,
        })
        layout.addWidget(buttons)

        self.setLayout(layout)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        stat_labels = [
            ('n_raw_matches', 'Raw Matches'),
            ('n_filtered_matches', 'Filtered Matches'),
            ('filter_rate', 'Filter Pass Rate'),
            ('threshold_used', 'Threshold Used'),
            ('avg_distance', 'Avg Descriptor Distance'),
            ('median_distance', 'Median Distance'),
            ('position_error', 'Avg Position Error'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Match Statistics', stat_labels,
        )
        layout.addWidget(self.stats_group)

        # Match table
        table_group = QGroupBox('Top Matches')
        table_layout = QVBoxLayout()
        self.match_table = QTableWidget()
        self.match_table.setColumnCount(5)
        self.match_table.setHorizontalHeaderLabels([
            'Ref Quad', 'Target Quad', 'Distance', 'dx', 'dy',
        ])
        table_layout.addWidget(self.match_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        return widget

    # ── Right panel ───────────────────────────────────────────────────────────

    def _create_right_panel(self):
        tabs = QTabWidget()

        # Tab 1: Spatial matches (quad polygons)
        spatial_tab, self.spatial_plot = create_pyqtgraph_tab('Spatial Matches')
        setup_pyqtgraph_plot(
            self.spatial_plot, 'Matched Quads',
            'X Coordinate', 'Y Coordinate',
        )
        self.spatial_plot.setAspectLocked(True)
        tabs.addTab(spatial_tab, '📍 Spatial Matches')

        # Tab 2: Score distribution
        score_tab, self.score_figure, self.score_canvas = create_matplotlib_tab(
            'Score Distribution',
        )
        tabs.addTab(score_tab, '📊 Score Distribution')

        # Tab 3: Spatial participation
        part_tab, self.participation_figure, self.participation_canvas = (
            create_matplotlib_tab('Spatial Participation')
        )
        tabs.addTab(part_tab, '🎯 Spatial Participation')

        # Tab 4: Parameters
        tabs.addTab(self._create_config_tab(), '⚙️ Parameters')

        # Tab 5: Statistics
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(
            2, self.config.output_dir,
        )
        tabs.addTab(stats_tab, '📋 Statistics')

        return tabs

    def _create_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel('Adjust threshold and re-run matching for current pair')
        info.setStyleSheet('font-weight: bold; padding: 10px;')
        layout.addWidget(info)

        form = QFormLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 1.0)
        self.threshold_spin.setValue(
            self.config.threshold if self.config.threshold else 0.01,
        )
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.001)
        form.addRow('Descriptor Threshold:', self.threshold_spin)
        layout.addLayout(form)

        info_text = QLabel(
            'The threshold is the maximum descriptor distance for a match.\n'
            'This value comes from Step 1.5 calibration.\n\n'
            'Lower threshold = stricter matching, fewer matches.\n'
            'Higher threshold = looser matching, more matches.\n\n'
            'Matches also pass through a consistency filter\n'
            'that checks geometric scale consistency.',
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet('color: gray; font-size: 10px; padding: 10px;')
        layout.addWidget(info_text)

        buttons = create_action_buttons({
            '↺ Reset to Step 1.5 Value': self.reset_params,
            '▶️ Re-run Current Pair': self.rerun_current_pair,
            '▶️ Re-run All Pairs': self.rerun_all_pairs,
        }, add_stretch=False)
        layout.addWidget(buttons)
        layout.addStretch()

        return widget

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_results(self):
        step2_dir = get_step_results_dir(self.config.output_dir, 2)

        print(f"\n{'='*60}")
        print(f"Loading Step 2 results from: {step2_dir}")

        if not step2_dir.exists():
            show_no_results_error(self, step2_dir, 'Step 2')
            return

        match_files = scan_results_directory(
            step2_dir, '*_matches_light.npz', verbose=True,
        )

        if not match_files:
            show_no_files_error(
                self, step2_dir, '*_matches_light.npz', 'Step 2',
            )
            return

        print(f"Found {len(match_files)} match files")

        self.match_data.clear()
        self.animal_pairs.clear()
        self.animal_combo.clear()
        self.pair_combo.clear()

        loaded_count = 0
        for match_file in match_files:
            data = load_npz_safely(match_file, verbose=True)
            if data is None:
                continue

            stem = match_file.stem.replace('_matches_light', '')
            parts = stem.split('_to_')

            if len(parts) == 2:
                left_parts = parts[0].split('_')
                if len(left_parts) >= 2:
                    animal_id = left_parts[0]
                    sess1 = '_'.join(left_parts[1:])
                    sess2 = parts[1]

                    self.animal_pairs.setdefault(animal_id, []).append(
                        f"{sess1}\u2192{sess2}"
                    )

                    pair_key = f"{sess1}\u2192{sess2}"
                    full_key = f"{animal_id}_{pair_key}"
                    self.match_data[full_key] = {
                        'file': match_file,
                        'data': data,
                        'animal': animal_id,
                        'sess1': sess1,
                        'sess2': sess2,
                    }
                    loaded_count += 1
                    print(f"  loaded {full_key}")

        print(f"{'='*60}")
        print(f"Successfully loaded {loaded_count} session pairs")
        print(f"{'='*60}\n")

        if loaded_count == 0:
            QMessageBox.warning(
                self, 'Load Failed',
                f'Could not load any match files from:\n{step2_dir}',
            )
            return

        animals = sorted(self.animal_pairs.keys())
        self.animal_combo.addItems(animals)
        if animals:
            self.animal_combo.setCurrentIndex(0)

        self.load_pipeline_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # Selection
    # ──────────────────────────────────────────────────────────────────────────

    def on_animal_changed(self, animal_id):
        if not animal_id or animal_id not in self.animal_pairs:
            return
        self.current_animal = animal_id
        self.pair_combo.clear()
        pairs = sorted(self.animal_pairs[animal_id])
        self.pair_combo.addItems(pairs)
        if pairs:
            self.pair_combo.setCurrentIndex(0)

    def on_pair_changed(self, pair_key):
        if not pair_key or not self.current_animal:
            return
        self.current_pair = pair_key
        full_key = f"{self.current_animal}_{pair_key}"
        if full_key in self.match_data:
            self.display_pair(full_key)

    # ──────────────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────────────

    def display_pair(self, full_key):
        match_info = self.match_data[full_key]
        data = match_info['data']

        if 'match_indices' not in data:
            QMessageBox.warning(
                self, 'No Matches',
                f'Could not find match_indices in {full_key}\n\n'
                f'Available fields: {", ".join(data.keys())}',
            )
            return

        match_indices = data['match_indices']
        n_filtered = len(match_indices)
        n_raw = int(data.get('n_raw_matches', n_filtered))
        threshold_used = float(data.get('threshold_used', 0))

        ref_centroids = data['ref_centroids']
        target_centroids = data['target_centroids']

        if 'distances' in data and len(data['distances']) > 0:
            distances = data['distances']
        elif 'ref_descriptors' in data and 'tgt_descriptors' in data:
            ref_d = data['ref_descriptors']
            tgt_d = data['tgt_descriptors']
            if len(ref_d) > 0 and len(tgt_d) > 0:
                distances = np.linalg.norm(ref_d - tgt_d, axis=1)
            else:
                distances = np.zeros(n_filtered)
        else:
            distances = np.zeros(n_filtered)

        # centroids are (y, x)
        sess1_y = ref_centroids[:, 0]
        sess1_x = ref_centroids[:, 1]
        sess2_y = target_centroids[:, 0]
        sess2_x = target_centroids[:, 1]

        # Position errors
        pos_errors = []
        for i in range(min(1000, n_filtered)):
            m = match_indices[i]
            dx = np.mean(sess2_x[m[4:]]) - np.mean(sess1_x[m[:4]])
            dy = np.mean(sess2_y[m[4:]]) - np.mean(sess1_y[m[:4]])
            pos_errors.append(np.sqrt(dx * dx + dy * dy))

        filter_rate = n_filtered / n_raw * 100 if n_raw > 0 else 0

        stats = {
            'n_raw_matches': f"{n_raw:,}",
            'n_filtered_matches': f"{n_filtered:,}",
            'filter_rate': f"{filter_rate:.1f}%",
            'threshold_used': f"{threshold_used:.4f}",
        }
        if len(distances) > 0:
            stats['avg_distance'] = f"{np.mean(distances):.4f}"
            stats['median_distance'] = f"{np.median(distances):.4f}"
        if pos_errors:
            stats['position_error'] = f"{np.mean(pos_errors):.2f} px"

        update_stat_labels(self.stat_labels, stats)

        # Cache for redraw
        self._cached_display = {
            'x1': sess1_x, 'y1': sess1_y,
            'x2': sess2_x, 'y2': sess2_y,
            'match_indices': match_indices,
            'distances': distances,
        }

        self._update_match_table(
            match_indices, distances, sess1_x, sess1_y, sess2_x, sess2_y,
        )
        self.plot_spatial_matches(
            sess1_x, sess1_y, sess2_x, sess2_y, match_indices,
        )
        self.plot_score_distribution(distances)
        self.plot_spatial_participation(
            sess1_x, sess1_y, sess2_x, sess2_y, match_indices,
        )

    def _redraw_spatial(self):
        if hasattr(self, '_cached_display'):
            d = self._cached_display
            self.plot_spatial_matches(
                d['x1'], d['y1'], d['x2'], d['y2'], d['match_indices'],
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Match table
    # ──────────────────────────────────────────────────────────────────────────

    def _update_match_table(self, match_indices, distances, x1, y1, x2, y2):
        n_display = min(100, len(match_indices))
        self.match_table.setRowCount(n_display)

        for i in range(n_display):
            m = match_indices[i]
            dist = distances[i] if i < len(distances) else 0
            ref_q, tgt_q = m[:4], m[4:]

            dx = np.mean(x2[tgt_q]) - np.mean(x1[ref_q])
            dy = np.mean(y2[tgt_q]) - np.mean(y1[ref_q])

            self.match_table.setItem(
                i, 0, QTableWidgetItem(f"[{','.join(map(str, ref_q))}]"),
            )
            self.match_table.setItem(
                i, 1, QTableWidgetItem(f"[{','.join(map(str, tgt_q))}]"),
            )
            self.match_table.setItem(i, 2, QTableWidgetItem(f"{dist:.4f}"))
            self.match_table.setItem(i, 3, QTableWidgetItem(f"{dx:.2f}"))
            self.match_table.setItem(i, 4, QTableWidgetItem(f"{dy:.2f}"))

    # ──────────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────────

    def plot_spatial_matches(self, x1, y1, x2, y2, match_indices):
        """Draw actual quad polygons for a random sample of matches.

        For each sampled match:
          - Reference quad drawn as a closed cyan polygon
          - Target quad drawn as a closed coral polygon
          - Thin grey lines connect corresponding vertices (0->0, 1->1, …)
        All other neurons shown as dim background dots.
        """
        self.spatial_plot.clear()

        if len(x1) == 0 or len(x2) == 0:
            return

        n_matches = len(match_indices)

        # Background: all neurons (dim)
        self.spatial_plot.addItem(pg.ScatterPlotItem(
            x1, y1, size=4,
            brush=pg.mkBrush(80, 160, 255, 50),
            pen=pg.mkPen(None),
        ))
        self.spatial_plot.addItem(pg.ScatterPlotItem(
            x2, y2, size=4,
            brush=pg.mkBrush(255, 120, 80, 50),
            pen=pg.mkPen(None),
        ))

        # Sample matches
        n_sample = min(self.n_matches_spin.value(), n_matches)
        if n_sample > 0 and n_matches > 0:
            sample_idx = np.random.choice(n_matches, size=n_sample, replace=False)

            ref_pen = pg.mkPen(0, 220, 220, 120, width=1.5)    # cyan
            tgt_pen = pg.mkPen(255, 100, 80, 120, width=1.5)   # coral
            link_pen = pg.mkPen(180, 180, 180, 60, width=1)     # grey

            for si in sample_idx:
                m = match_indices[si]
                ref_q = m[:4]
                tgt_q = m[4:]

                # Reference quad polygon (closed)
                rx = np.array([x1[ref_q[j]] for j in range(4)] + [x1[ref_q[0]]])
                ry = np.array([y1[ref_q[j]] for j in range(4)] + [y1[ref_q[0]]])
                self.spatial_plot.addItem(
                    pg.PlotDataItem(rx, ry, pen=ref_pen),
                )

                # Target quad polygon (closed)
                tx = np.array([x2[tgt_q[j]] for j in range(4)] + [x2[tgt_q[0]]])
                ty = np.array([y2[tgt_q[j]] for j in range(4)] + [y2[tgt_q[0]]])
                self.spatial_plot.addItem(
                    pg.PlotDataItem(tx, ty, pen=tgt_pen),
                )

                # Vertex correspondence lines
                for j in range(4):
                    self.spatial_plot.addItem(
                        pg.PlotDataItem(
                            [x1[ref_q[j]], x2[tgt_q[j]]],
                            [y1[ref_q[j]], y2[tgt_q[j]]],
                            pen=link_pen,
                        ),
                    )

        self.spatial_plot.setTitle(
            f'Quad matches: {n_sample} / {n_matches:,} sampled  |  '
            f'cyan = ref, coral = target, grey = vertex links',
        )

    def plot_score_distribution(self, distances):
        self.score_figure.clear()
        ax = self.score_figure.add_subplot(111)

        if len(distances) > 0:
            plot_histogram_with_stats(
                ax, distances, bins=50,
                title='Distribution of Descriptor Distances',
                xlabel='Descriptor Distance',
                ylabel='Frequency',
                show_mean=True,
                show_median=True,
            )

        self.score_canvas.draw()

    def plot_spatial_participation(self, x1, y1, x2, y2, match_indices):
        self.participation_figure.clear()

        if len(match_indices) == 0 or len(x1) == 0 or len(x2) == 0:
            return

        ref_counts = np.zeros(len(x1), dtype=int)
        tgt_counts = np.zeros(len(x2), dtype=int)

        for m in match_indices:
            ref_counts[m[:4]] += 1
            tgt_counts[m[4:]] += 1

        fig = self.participation_figure

        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_participation_heatmap(ax1, x1, y1, ref_counts, 'Reference')

        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_participation_heatmap(ax2, x2, y2, tgt_counts, 'Target')

        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_radial_distribution(ax3, x1, y1, ref_counts, 'Reference')

        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_radial_distribution(ax4, x2, y2, tgt_counts, 'Target')

        fig.tight_layout()
        self.participation_canvas.draw()

    def _plot_participation_heatmap(self, ax, x, y, counts, title):
        bins = 30
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=counts)
        H_total, _, _ = np.histogram2d(x, y, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.nan_to_num(H / H_total)

        im = ax.imshow(
            rate.T, origin='lower', cmap='hot',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto', interpolation='bilinear',
        )
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.set_title(f'{title}\nAvg Matches / Neuron', fontsize=11,
                     fontweight='bold')
        cbar = self.participation_figure.colorbar(im, ax=ax)
        cbar.set_label('Avg Matches/Neuron', fontsize=9)

    def _plot_radial_distribution(self, ax, x, y, counts, title):
        cx, cy = np.mean(x), np.mean(y)
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        n_bins = 20
        max_d = np.max(dists)
        bins = np.linspace(0, max_d, n_bins + 1)
        bin_idx = np.digitize(dists, bins)

        avg, centers = [], []
        for i in range(1, n_bins + 1):
            mask = bin_idx == i
            if np.sum(mask) > 0:
                avg.append(np.mean(counts[mask]))
                centers.append((bins[i - 1] + bins[i]) / 2)

        if centers:
            ax.plot(centers, avg, 'o-', linewidth=2, markersize=6,
                    color='steelblue')
            ax.fill_between(centers, 0, avg, alpha=0.3, color='steelblue')
            ax.set_xlabel('Distance from Center (px)', fontsize=10)
            ax.set_ylabel('Avg Matches / Neuron', fontsize=10)
            ax.set_title(f'{title}\nCenter vs Periphery', fontsize=11,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_d)
            ax.set_ylim(0, None)

    # ──────────────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────────────

    def reset_params(self):
        self.threshold_spin.setValue(0.15)

    def rerun_current_pair(self):
        if not self.current_pair or not self.current_animal:
            return
        reply = QMessageBox.question(
            self, 'Confirm Re-run',
            f'Re-run matching for {self.current_animal}: {self.current_pair}?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.config.threshold = self.threshold_spin.value()
            QMessageBox.information(
                self, 'Re-run',
                'Re-running matching...\n(Implementation needed)',
            )

    def rerun_all_pairs(self):
        reply = QMessageBox.question(
            self, 'Confirm Re-run All',
            'Re-run matching for ALL pairs?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.config.threshold = self.threshold_spin.value()
            QMessageBox.information(
                self, 'Re-run All',
                'Re-running all matching...\n(Implementation needed)',
            )

    def export_summary(self):
        QMessageBox.information(
            self, 'Export', 'Export functionality coming soon!',
        )