"""
Step 1 Results Viewer: Quad Generation Inspector
(v2 — diagonal-first pipeline)
"""

import json
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QFormLayout,
                             QDoubleSpinBox, QSpinBox, QMessageBox, QLabel,
                             QSplitter, QTabWidget, QWidget, QPushButton)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

from utilities import *


class Step1Viewer(QDialog):
    """Viewer for Step 1: Quad Generation Results (diagonal-first pipeline)"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_session = None
        self.session_data = {}
        self.animal_sessions = {}

        self.setWindowTitle('Step 1: Quad Generation Results')
        self.resize(1400, 900)

        self.init_ui()
        self.load_results()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def init_ui(self):
        layout = QVBoxLayout()

        # Top controls
        self.controls = create_top_controls(
            'Session Selection',
            combos=[
                ('Animal', self.on_animal_changed),
                ('Session', self.on_session_changed),
            ],
            refresh_callback=self.load_results,
        )
        self.animal_combo, self.session_combo = self.controls.combos
        layout.addWidget(self.controls)

        # Sample-quad controls
        quad_ctl = QHBoxLayout()
        quad_ctl.addWidget(QLabel('Sample quads to overlay:'))
        self.n_quads_spin = QSpinBox()
        self.n_quads_spin.setRange(0, 50000)
        self.n_quads_spin.setValue(500)
        self.n_quads_spin.setSingleStep(100)
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(400)
        self._redraw_timer.timeout.connect(self._redraw_spatial)
        self.n_quads_spin.valueChanged.connect(lambda: self._redraw_timer.start())
        quad_ctl.addWidget(self.n_quads_spin)
        self.reshuffle_btn = QPushButton('🔀 Reshuffle')
        self.reshuffle_btn.clicked.connect(self._redraw_spatial)
        quad_ctl.addWidget(self.reshuffle_btn)
        quad_ctl.addStretch()
        layout.addLayout(quad_ctl)

        # Main content
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        # Bottom buttons
        buttons = create_action_buttons({
            '💾 Export Summary': self.export_summary,
            '✖ Close': self.accept,
        })
        layout.addWidget(buttons)

        self.setLayout(layout)

    # ── Left panel (stats only, no table) ─────────────────────────────────────

    def _create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        stat_labels = [
            ('n_rois',          'Neurons'),
            ('n_quads',         'Quads'),
            ('quads_per_neuron','Quads / Neuron (median)'),
            ('zero_neurons',    'Zero-Quad Neurons'),
            ('coverage_min',    'Coverage Min'),
            ('coverage_p5',     'Coverage 5th %ile'),
            ('gen_method',      'Method'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Session Statistics', stat_labels,
        )
        layout.addWidget(self.stats_group)
        layout.addStretch()
        return widget

    # ── Right panel (tabs) ────────────────────────────────────────────────────

    def _create_right_panel(self):
        tabs = QTabWidget()

        # Tab 1 — Heatmap + sample quads
        spatial_tab, self.spatial_plot = create_pyqtgraph_tab('Spatial View')
        setup_pyqtgraph_plot(
            self.spatial_plot,
            'Neuron Quad Coverage',
            'X (pixels)', 'Y (pixels)',
        )
        tabs.addTab(spatial_tab, '🗺️ Spatial')

        # Tab 2 — Coverage histogram (replaces area + aspect)
        cov_tab, self.cov_figure, self.cov_canvas = create_matplotlib_tab(
            'Coverage Distribution',
        )
        tabs.addTab(cov_tab, '📊 Coverage')

        # Tab 3 — Descriptor scatter (quick health check)
        desc_tab, self.desc_figure, self.desc_canvas = create_matplotlib_tab(
            'Descriptor Space',
        )
        tabs.addTab(desc_tab, '🔬 Descriptors')

        # Tab 4 — Saturation (unchanged)
        sat_tab, self.sat_figure, self.sat_canvas = create_matplotlib_tab(
            'Saturation Check',
        )
        tabs.addTab(sat_tab, '🔵 Saturation')

        # Tab 5 — Config
        tabs.addTab(self._create_config_tab(), '⚙️ Parameters')

        # Tab 6 — Pipeline stats
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(
            1, self.config.output_dir,
        )
        tabs.addTab(stats_tab, '📋 Statistics')

        return tabs

    # ── Config tab (updated for diagonal-first params) ────────────────────────

    def _create_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel('Diagonal-first pipeline parameters')
        info.setStyleSheet('font-weight: bold; padding: 10px;')
        layout.addWidget(info)

        form = QFormLayout()

        self.knn_k_spin = QSpinBox()
        self.knn_k_spin.setRange(2, 100)
        self.knn_k_spin.setValue(getattr(self.config, 'knn_k', 15))
        form.addRow('KNN k (local neighbors):', self.knn_k_spin)

        self.max_K_spin = QSpinBox()
        self.max_K_spin.setRange(2, 200)
        self.max_K_spin.setValue(
            getattr(self.config, 'max_triangles_per_diagonal', 25),
        )
        form.addRow('Max third-points / diagonal (K):', self.max_K_spin)

        self.keep_frac_spin = QDoubleSpinBox()
        self.keep_frac_spin.setRange(0.01, 1.0)
        self.keep_frac_spin.setSingleStep(0.05)
        self.keep_frac_spin.setDecimals(2)
        self.keep_frac_spin.setValue(
            getattr(self.config, 'quad_keep_fraction', 1.0),
        )
        form.addRow('Quad keep fraction:', self.keep_frac_spin)

        self.min_cov_spin = QDoubleSpinBox()
        self.min_cov_spin.setRange(0.0, 1.0)
        self.min_cov_spin.setSingleStep(0.05)
        self.min_cov_spin.setDecimals(2)
        self.min_cov_spin.setValue(
            getattr(self.config, 'min_coverage_fraction', 0.4),
        )
        form.addRow('Min coverage fraction:', self.min_cov_spin)

        layout.addLayout(form)

        buttons = create_action_buttons({
            '↺ Reset to Default': self.reset_params,
            '▶️ Re-run Current Session': self.rerun_current_session,
            '▶️ Re-run All Sessions': self.rerun_all_sessions,
        }, add_stretch=False)
        layout.addWidget(buttons)
        layout.addStretch()
        return widget

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_results(self):
        step1_dir = get_step_results_dir(self.config.output_dir, 1)

        print(f"\n{'='*60}")
        print(f"Loading Step 1 results from: {step1_dir}")

        if not step1_dir.exists():
            show_no_results_error(self, step1_dir, 'Step 1')
            return

        quad_files = scan_results_directory(
            step1_dir,
            get_step_file_pattern(1),
            verbose=True,
        )

        if not quad_files:
            show_no_files_error(
                self, step1_dir, '*_centroids_quads.npz', 'Step 1',
            )
            return

        print(f"Found {len(quad_files)} quad files")

        self.session_data.clear()
        self.animal_sessions.clear()
        self.animal_combo.clear()
        self.session_combo.clear()

        loaded_count = 0
        for quad_file in quad_files:
            data = load_npz_safely(quad_file, verbose=True)
            if data is None:
                continue

            result = extract_animal_session_from_filename(
                quad_file.stem, suffix='_centroids_quads',
            )
            if result:
                animal_id, session_id = result
                self.animal_sessions.setdefault(animal_id, []).append(session_id)
                session_key = f"{animal_id}_{session_id}"
                self.session_data[session_key] = {
                    'file': quad_file,
                    'data': data,
                    'animal': animal_id,
                    'session': session_id,
                }
                loaded_count += 1
                print(f"  ✓ Loaded as {session_key}")

        print(f"{'='*60}")
        print(f"Successfully loaded {loaded_count} sessions")
        print(f"{'='*60}\n")

        if loaded_count == 0:
            QMessageBox.warning(
                self, 'Load Failed',
                f'Could not load any quad files from:\n{step1_dir}',
            )
            return

        animals = sorted(self.animal_sessions.keys())
        self.animal_combo.addItems(animals)
        if animals:
            self.animal_combo.setCurrentIndex(0)

        self._load_and_plot_saturation()
        self.load_pipeline_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # Session selection
    # ──────────────────────────────────────────────────────────────────────────

    def on_animal_changed(self, animal_id):
        if not animal_id or animal_id not in self.animal_sessions:
            return
        self.current_animal = animal_id
        self.session_combo.clear()
        sessions = sorted(self.animal_sessions[animal_id])
        self.session_combo.addItems(sessions)
        if sessions:
            self.session_combo.setCurrentIndex(0)

    def on_session_changed(self, session_id):
        if not session_id or not self.current_animal:
            return
        self.current_session = session_id
        session_key = f"{self.current_animal}_{session_id}"
        if session_key in self.session_data:
            self.display_session(session_key)

    # ──────────────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────────────

    def display_session(self, session_key):
        info = self.session_data[session_key]
        data = info['data']

        centroids = data['centroids']            # (N, 2) as (y, x)
        quad_idx  = data.get('quad_idx', np.array([]))
        quad_desc = data.get('quad_desc', np.array([]))
        n_neurons = len(centroids)
        n_quads   = len(quad_idx)

        # Per-neuron coverage
        coverage = np.zeros(n_neurons, dtype=np.int32)
        if n_quads > 0:
            for nid in np.asarray(quad_idx).ravel():
                if 0 <= nid < n_neurons:
                    coverage[nid] += 1

        gen_method = str(data.get('generation_method', 'diagonal_first'))

        stats = {
            'n_rois':           n_neurons,
            'n_quads':          n_quads,
            'quads_per_neuron': f"{np.median(coverage):.0f}",
            'zero_neurons':     f"{int(np.sum(coverage == 0))} / {n_neurons}",
            'coverage_min':     int(coverage.min()) if n_neurons else 0,
            'coverage_p5':      f"{np.percentile(coverage, 5):.0f}" if n_neurons else 0,
            'gen_method':       gen_method,
        }
        update_stat_labels(self.stat_labels, stats)

        cx = centroids[:, 1]
        cy = centroids[:, 0]

        self.plot_spatial(cx, cy, quad_idx, coverage)
        self.plot_coverage(coverage)
        self.plot_descriptors(quad_desc)

    def _redraw_spatial(self):
        if self.current_animal and self.current_session:
            session_key = f"{self.current_animal}_{self.current_session}"
            if session_key in self.session_data:
                data = self.session_data[session_key]['data']
                centroids = data['centroids']
                quad_idx  = data.get('quad_idx', np.array([]))
                n = len(centroids)
                coverage = np.zeros(n, dtype=np.int32)
                if len(quad_idx) > 0:
                    for nid in np.asarray(quad_idx).ravel():
                        if 0 <= nid < n:
                            coverage[nid] += 1
                self.plot_spatial(
                    centroids[:, 1], centroids[:, 0], quad_idx, coverage,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────────

    def plot_spatial(self, x, y, quad_idx, coverage):
        """Heatmap of per-neuron coverage + random sample of quads."""
        self.spatial_plot.clear()

        n_quads = len(quad_idx)

        # Colour neurons by coverage (log-scale for visibility)
        log_cov = np.log1p(coverage).astype(np.float64)
        if log_cov.max() > 0:
            normed = log_cov / log_cov.max()
        else:
            normed = np.zeros_like(log_cov)

        brushes = [
            pg.mkBrush(
                int(255 * v),           # R — hot
                int(80 * (1 - v)),      # G — dim for low coverage
                int(255 * (1 - v)),     # B — cool for low coverage
                180,
            )
            for v in normed
        ]

        scatter = pg.ScatterPlotItem(
            x, y,
            size=7,
            brush=brushes,
            pen=pg.mkPen(None),
        )
        self.spatial_plot.addItem(scatter)

        # Highlight zero-coverage neurons
        zero_mask = coverage == 0
        if np.any(zero_mask):
            zero_scatter = pg.ScatterPlotItem(
                x[zero_mask], y[zero_mask],
                size=10,
                symbol='x',
                brush=pg.mkBrush(255, 255, 0, 220),
                pen=pg.mkPen('y', width=1.5),
            )
            self.spatial_plot.addItem(zero_scatter)

        # Random sample of quads
        n_sample = min(self.n_quads_spin.value(), n_quads)
        if n_sample > 0 and n_quads > 0:
            sample_idx = np.random.choice(n_quads, size=n_sample, replace=False)
            for si in sample_idx:
                quad = quad_idx[si]
                if len(quad) == 4:
                    pts_x = [x[quad[j]] for j in range(4)] + [x[quad[0]]]
                    pts_y = [y[quad[j]] for j in range(4)] + [y[quad[0]]]
                    line = pg.PlotDataItem(
                        pts_x, pts_y,
                        pen=pg.mkPen(255, 80, 80, 100, width=1),
                    )
                    self.spatial_plot.addItem(line)

        n_zero = int(np.sum(zero_mask))
        self.spatial_plot.setTitle(
            f'Coverage heatmap  |  {n_sample} / {n_quads:,} quads sampled  '
            f'|  {n_zero} zero-coverage ✕'
        )

    def plot_coverage(self, coverage):
        """Histogram of quads-per-neuron."""
        self.cov_figure.clear()

        ax1 = self.cov_figure.add_subplot(1, 2, 1)
        if coverage.size > 0:
            ax1.hist(
                coverage, bins=min(60, max(10, int(coverage.max()))),
                color='steelblue', edgecolor='white', linewidth=0.4,
            )
            ax1.axvline(
                np.median(coverage), color='orange', linewidth=1.5,
                linestyle='-', label=f'median = {np.median(coverage):.0f}',
            )
            ax1.axvline(
                np.percentile(coverage, 5), color='red', linewidth=1,
                linestyle='--', label=f'5th %ile = {np.percentile(coverage, 5):.0f}',
            )
            ax1.legend(fontsize=8)
        ax1.set_xlabel('Quads per Neuron', fontsize=9)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.set_title('Coverage Distribution', fontsize=10, fontweight='bold')

        ax2 = self.cov_figure.add_subplot(1, 2, 2)
        if coverage.size > 0:
            sorted_cov = np.sort(coverage)
            ax2.plot(
                np.arange(len(sorted_cov)), sorted_cov,
                color='steelblue', linewidth=1.5,
            )
            ax2.axhline(
                np.median(coverage) * 0.4, color='red', linewidth=1,
                linestyle='--', label='remediation threshold (0.4 × median)',
            )
            ax2.legend(fontsize=8)
        ax2.set_xlabel('Neuron (sorted)', fontsize=9)
        ax2.set_ylabel('Quad Count', fontsize=9)
        ax2.set_title('Sorted Coverage Curve', fontsize=10, fontweight='bold')

        self.cov_figure.tight_layout(pad=1.5)
        self.cov_canvas.draw()

    def plot_descriptors(self, quad_desc):
        """Quick scatter of descriptor space (xC vs xD, yC vs yD)."""
        self.desc_figure.clear()

        if quad_desc is None or len(quad_desc) == 0:
            ax = self.desc_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No descriptors', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.axis('off')
            self.desc_canvas.draw()
            return

        # Subsample for speed
        n = len(quad_desc)
        max_pts = 20_000
        if n > max_pts:
            idx = np.random.choice(n, size=max_pts, replace=False)
            desc = quad_desc[idx]
        else:
            desc = quad_desc

        xC, yC, xD, yD = desc[:, 0], desc[:, 1], desc[:, 2], desc[:, 3]

        ax1 = self.desc_figure.add_subplot(1, 2, 1)
        ax1.scatter(xC, yC, s=1, alpha=0.15, c='steelblue', edgecolors='none')
        ax1.set_xlabel('xC', fontsize=9)
        ax1.set_ylabel('yC', fontsize=9)
        ax1.set_title('C coordinates', fontsize=10, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.2)

        ax2 = self.desc_figure.add_subplot(1, 2, 2)
        ax2.scatter(xD, yD, s=1, alpha=0.15, c='coral', edgecolors='none')
        ax2.set_xlabel('xD', fontsize=9)
        ax2.set_ylabel('yD', fontsize=9)
        ax2.set_title('D coordinates', fontsize=10, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.2)

        self.desc_figure.tight_layout(pad=1.5)
        self.desc_canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Saturation (unchanged logic)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_and_plot_saturation(self):
        sat_json = Path(self.config.output_dir) / "saturation" / "saturation_results.json"
        if not sat_json.exists():
            ax = self.sat_figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f'No saturation results found.\nRun Step 1 to generate.\n\n{sat_json}',
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='gray',
            )
            ax.axis('off')
            self.sat_canvas.draw()
            return

        with open(sat_json) as f:
            results = json.load(f)
        if not results:
            return

        ratios     = np.array([r["saturation_ratio"]   for r in results])
        is_sat     = np.array([r["is_saturated"]        for r in results])
        conditions = [r.get("condition", "unknown")     for r in results]

        self.sat_figure.clear()

        # ── Bar chart (sorted) ────────────────────────────────────────────────
        ax1 = self.sat_figure.add_subplot(2, 2, 1)
        order = np.argsort(ratios)
        colors = ['#e74c3c' if is_sat[i] else '#2ecc71' for i in order]
        ax1.bar(range(len(ratios)), ratios[order], color=colors,
                width=1.0, edgecolor='none')
        ax1.axhline(1.0, color='black', linewidth=1.5, linestyle='--',
                     label='Saturation threshold')
        ax1.set_xlabel('File (sorted)', fontsize=9)
        ax1.set_ylabel('Saturation Ratio', fontsize=9)
        ax1.set_title(
            f'Saturation — {is_sat.sum()}/{len(results)} saturated',
            fontsize=10, fontweight='bold',
        )
        ax1.legend(fontsize=8)

        # ── Histogram ─────────────────────────────────────────────────────────
        ax2 = self.sat_figure.add_subplot(2, 2, 2)
        ax2.hist(ratios, bins=30, color='steelblue', edgecolor='white',
                 linewidth=0.5)
        ax2.axvline(1.0, color='red', linewidth=1.5, linestyle='--',
                     label='Saturation = 1.0')
        ax2.axvline(float(np.median(ratios)), color='orange', linewidth=1.5,
                     linestyle='-', label=f'Median = {np.median(ratios):.2f}')
        ax2.set_xlabel('Saturation Ratio', fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_title('Ratio Distribution', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)

        # ── Per-condition box plot ────────────────────────────────────────────
        ax3 = self.sat_figure.add_subplot(2, 2, 3)
        unique_conds = sorted(set(conditions))
        if len(unique_conds) > 1 or unique_conds != ['unknown']:
            cond_ratios = [
                [r for r, c in zip(ratios, conditions) if c == cond]
                for cond in unique_conds
            ]
            bp = ax3.boxplot(cond_ratios, labels=unique_conds,
                             patch_artist=True, notch=False)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)
            ax3.axhline(1.0, color='red', linewidth=1.2, linestyle='--')
            ax3.set_xlabel('Condition', fontsize=9)
            ax3.set_ylabel('Saturation Ratio', fontsize=9)
            ax3.set_title('Per-Condition Ratios', fontsize=10, fontweight='bold')
            ax3.tick_params(axis='x', rotation=30)
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, 'No condition labels found',
                     ha='center', va='center', transform=ax3.transAxes,
                     color='gray')

        # ── NN vs sigma_d scatter ─────────────────────────────────────────────
        ax4 = self.sat_figure.add_subplot(2, 2, 4)
        nn_dists = np.array([r["median_nn_distance"] for r in results])
        sig_ds   = np.array([r["median_sigma_d"]     for r in results])
        ax4.scatter(
            sig_ds, nn_dists,
            c=['#e74c3c' if s else '#2ecc71' for s in is_sat],
            alpha=0.7, s=30, edgecolors='none',
        )
        lim = max(nn_dists.max(), sig_ds.max()) * 1.05
        ax4.plot([0, lim], [0, lim], 'k--', linewidth=1,
                 label='nn = σ_d (threshold)')
        ax4.set_xlabel('Median σ_d (descriptor blur)', fontsize=9)
        ax4.set_ylabel('Median NN distance', fontsize=9)
        ax4.set_title(
            'NN Distance vs $\\sigma_d$\n(green = ok, red = saturated)',
            fontsize=10, fontweight='bold',
        )
        ax4.legend(fontsize=8)

        self.sat_figure.tight_layout(pad=2.0)
        self.sat_canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────────────

    def reset_params(self):
        self.knn_k_spin.setValue(15)
        self.max_K_spin.setValue(25)
        self.keep_frac_spin.setValue(1.0)
        self.min_cov_spin.setValue(0.4)

    def rerun_current_session(self):
        if not self.current_session or not self.current_animal:
            return
        reply = QMessageBox.question(
            self, 'Confirm Re-run',
            f'Re-run quad generation for {self.current_animal} / '
            f'{self.current_session}?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.config.knn_k = self.knn_k_spin.value()
            self.config.max_triangles_per_diagonal = self.max_K_spin.value()
            self.config.quad_keep_fraction = self.keep_frac_spin.value()
            self.config.min_coverage_fraction = self.min_cov_spin.value()
            QMessageBox.information(
                self, 'Re-run',
                'Re-running quad generation…\n(Implementation needed)',
            )

    def rerun_all_sessions(self):
        reply = QMessageBox.question(
            self, 'Confirm Re-run All',
            'Re-run quad generation for ALL sessions?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.config.knn_k = self.knn_k_spin.value()
            self.config.max_triangles_per_diagonal = self.max_K_spin.value()
            self.config.quad_keep_fraction = self.keep_frac_spin.value()
            self.config.min_coverage_fraction = self.min_cov_spin.value()
            QMessageBox.information(
                self, 'Re-run All',
                'Re-running quad generation for all sessions…\n'
                '(Implementation needed)',
            )

    def export_summary(self):
        QMessageBox.information(
            self, 'Export', 'Export functionality coming soon!',
        )