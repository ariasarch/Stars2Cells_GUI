"""
Step 3 Results Viewer: RANSAC-Informed Matching Analysis

V3: Updated for padded Hungarian with post-filter + second-pass recovery.
    The cost-threshold sweep has been removed — matching is now self-calibrating
    via asymmetric dummy padding. The viewer focuses on match quality diagnostics.
"""

import numpy as np
import json
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QGroupBox, QMessageBox,
                             QSplitter, QTabWidget, QWidget, QLabel, QComboBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

from utilities import *


class Step3Viewer(QDialog):
    """Viewer for Step 3: RANSAC-Informed Hungarian Matching + Consolidated Tracking"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_pair = None
        self.sweep_data = {}
        self.tracking_data = {}
        self.pair_data_cache = {}

        self.setWindowTitle('Step 3: Neuron Matching & Tracking Analysis')
        self.resize(1800, 1000)

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

        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        # Bottom buttons
        buttons = create_action_buttons({
            '💾 Export Data': self.export_data,
            '✖ Close': self.accept
        })
        layout.addWidget(buttons)

        self.setLayout(layout)

    def create_left_panel(self):
        """Create left panel with stats and pair table"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tracking header
        self.tracking_header = self.create_tracking_header()
        layout.addWidget(self.tracking_header)

        # Summary statistics
        stat_labels = [
            ('n_sessions', 'Sessions'),
            ('n_pairs', 'Session Pairs'),
            ('total_matches', 'Total Matched Pairs'),
            ('avg_match_rate', 'Avg Match Rate'),
            ('n_tracks', 'Total Tracks'),
            ('full_tracks', 'Full-Length Tracks'),
            ('avg_track_len', 'Avg Track Length'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Summary', stat_labels
        )
        layout.addWidget(self.stats_group)

        # Session pair table
        table_group = QGroupBox('Session Pairs')
        table_layout = QVBoxLayout()

        self.pair_table = QTableWidget()
        self.pair_table.setColumnCount(4)
        self.pair_table.setHorizontalHeaderLabels([
            'Pair Name', 'Ref Neurons', 'Matches', 'Match Rate'
        ])
        self.pair_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pair_table.setSelectionMode(QTableWidget.SingleSelection)
        self.pair_table.itemSelectionChanged.connect(self.on_pair_selected)
        table_layout.addWidget(self.pair_table)

        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        return widget

    def create_tracking_header(self):
        """Create prominent tracking percentage display"""
        header = QLabel()
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet('''
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border: 2px solid #00AAFF;
                border-radius: 8px;
                margin: 5px;
            }
        ''')
        header.setText('📊 No animal selected')
        return header

    def create_right_panel(self):
        """Create right panel with visualizations"""
        tabs = QTabWidget()

        # Tab 1: All Pairs Overview
        overview_tab, self.overview_figure, self.overview_canvas = create_matplotlib_tab(
            'All Pairs Overview'
        )
        tabs.addTab(overview_tab, '📊 All Pairs')

        # Tab 2: Geometric Coherence
        geom_tab, self.geom_figure, self.geom_canvas = create_matplotlib_tab(
            'Geometric Coherence'
        )
        tabs.addTab(geom_tab, '🔗 Geometric Coherence')

        # Tab 3: Vote Support
        vote_tab, self.vote_figure, self.vote_canvas = create_matplotlib_tab(
            'Vote Support'
        )
        tabs.addTab(vote_tab, '🗳️ Vote Support')

        # Tab 4: Match Quality
        quality_tab, self.quality_figure, self.quality_canvas = create_matplotlib_tab(
            'Match Quality'
        )
        tabs.addTab(quality_tab, '⭐ Match Quality')

        # Tab 5: Match Alternatives
        alt_tab = self.create_alternatives_tab()
        tabs.addTab(alt_tab, '🎲 Match Alternatives')

        # Tab 6: Statistics
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(3, self.config.output_dir)
        tabs.addTab(stats_tab, '📋 Statistics')

        return tabs

    def create_alternatives_tab(self):
        """Create tab for match alternatives view"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Neuron selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel('Select Neuron:'))
        self.neuron_selector = QComboBox()
        self.neuron_selector.currentIndexChanged.connect(self.on_neuron_selected)
        selector_layout.addWidget(self.neuron_selector)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        # Alternatives plot
        self.alt_widget, self.alt_figure, self.alt_canvas = create_matplotlib_tab('')
        layout.addWidget(self.alt_widget)

        return tab

    def load_results(self):
        """Load Step 3 results"""
        step3_dir = get_step_results_dir(self.config.output_dir, 3)

        print(f"\n{'='*60}")
        print(f"Loading Step 3 results from: {step3_dir}")

        if not step3_dir.exists():
            show_no_results_error(self, step3_dir, 3)
            return

        summary_file = step3_dir / 'step3_summary.json'

        if not summary_file.exists():
            QMessageBox.warning(
                self, 'No Summary',
                f'Summary file not found:\n{summary_file}\n\n'
                f'Please run Step 3 first.'
            )
            return

        print(f"Loading: {summary_file}")

        summary = load_json_safely(summary_file, verbose=True)
        if summary is None:
            return

        animals_data = summary['animals']

        # Clear previous data
        self.sweep_data.clear()
        self.tracking_data.clear()
        self.pair_data_cache.clear()
        self.animal_combo.clear()

        # Load data for each animal
        for animal_result in animals_data:
            animal_id = str(animal_result['animal_id'])

            # Load sweep NPZ files (format kept for compatibility)
            pattern = f"{animal_id}_*_sweep.npz"
            sweep_files = sorted(step3_dir.glob(pattern))

            pairs = []
            for npz_file in sweep_files:
                data = load_npz_safely(npz_file, verbose=False)
                if data:
                    pair_name = str(data.get('pair_name', npz_file.stem.replace('_sweep', '')))
                    pair_info = {
                        'pair_name': pair_name,
                        'file': npz_file,
                        'ref_session': str(data.get('ref_session', '')),
                        'target_session': str(data.get('target_session', '')),
                        'n_ref_neurons': int(data.get('n_ref_neurons', 0)),
                        'n_target_neurons': int(data.get('n_target_neurons', 0)),
                        'optimal_matches': int(data.get('optimal_matches', 0)),
                        'optimal_rate': float(data.get('optimal_rate', 0.0)),
                    }
                    pairs.append(pair_info)

            self.sweep_data[animal_id] = {
                'pairs': pairs,
                'n_pairs': len(pairs),
                'avg_optimal_rate': animal_result.get('avg_optimal_rate', 0.0),
                'total_optimal_matches': animal_result.get('total_optimal_matches', 0),
            }

            # Load consolidated tracking
            tracking_file = step3_dir / f"{animal_id}_consolidated_tracking.npz"
            if tracking_file.exists():
                tracking = load_npz_safely(tracking_file, verbose=True)
                if tracking:
                    self.tracking_data[animal_id] = tracking
                    print(f"  {animal_id}: {len(pairs)} pairs, "
                          f"{tracking['n_total_tracks']} tracks, "
                          f"{tracking['full_length_tracks']} full-length")
            else:
                print(f"  {animal_id}: {len(pairs)} pairs, no consolidated tracking")

        print(f"{'='*60}")
        print(f"Loaded {len(self.sweep_data)} animals")
        print(f"{'='*60}\n")

        # Populate combo
        animals = sorted(self.sweep_data.keys())
        self.animal_combo.addItems(animals)

        if animals:
            self.animal_combo.setCurrentIndex(0)

        self.load_pipeline_stats()

    def on_animal_changed(self, animal_id):
        """Handle animal selection"""
        if not animal_id or animal_id not in self.sweep_data:
            return

        self.current_animal = animal_id
        self.display_animal(animal_id)

    def display_animal(self, animal_id):
        """Display results for selected animal"""
        if animal_id not in self.sweep_data:
            return

        sweep_info = self.sweep_data[animal_id]
        tracking = self.tracking_data.get(animal_id)
        pairs = sweep_info['pairs']

        total_matched = sweep_info['total_optimal_matches']
        avg_rate = sweep_info['avg_optimal_rate']

        # Update tracking header
        if tracking is not None:
            n_tracks = int(tracking['n_total_tracks'])
            full_tracks = int(tracking['full_length_tracks'])
            n_sessions = int(tracking['n_sessions'])
            full_pct = full_tracks / n_tracks * 100 if n_tracks > 0 else 0

            header_text = (
                f'🌟 {animal_id}: {total_matched:,} matched pairs | '
                f'{avg_rate*100:.1f}% avg match rate | '
                f'{full_tracks:,}/{n_tracks:,} full-length tracks ({full_pct:.1f}%)'
            )

            if full_pct >= 70:
                color = '#00FF00'
            elif full_pct >= 50:
                color = '#FFAA00'
            else:
                color = '#FF6600'
        else:
            header_text = (
                f'📊 {animal_id}: {total_matched:,} matched pairs | '
                f'{avg_rate*100:.1f}% avg match rate | tracking not available'
            )
            color = '#AAAAAA'

        self.tracking_header.setStyleSheet(f'''
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: {color};
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                border: 2px solid {color};
                border-radius: 8px;
                margin: 5px;
            }}
        ''')
        self.tracking_header.setText(header_text)

        # Update statistics
        stats = {
            'n_pairs': sweep_info['n_pairs'],
            'total_matches': f"{total_matched:,}",
            'avg_match_rate': f"{avg_rate*100:.1f}%",
        }

        if tracking is not None:
            stats['n_sessions'] = int(tracking['n_sessions'])
            stats['n_tracks'] = int(tracking['n_total_tracks'])
            stats['full_tracks'] = int(tracking['full_length_tracks'])
            stats['avg_track_len'] = f"{tracking['avg_track_length']:.1f}"
        else:
            stats['n_sessions'] = len(pairs)
            stats['n_tracks'] = 'N/A'
            stats['full_tracks'] = 'N/A'
            stats['avg_track_len'] = 'N/A'

        update_stat_labels(self.stat_labels, stats)

        self._update_pair_table(pairs)
        self.plot_all_pairs_overview(sweep_info)

        if pairs:
            self.pair_table.selectRow(0)

    def _update_pair_table(self, pairs):
        """Update pair table"""
        self.pair_table.setRowCount(len(pairs))

        for i, pair in enumerate(pairs):
            self.pair_table.setItem(i, 0, QTableWidgetItem(pair['pair_name']))
            self.pair_table.setItem(i, 1, QTableWidgetItem(str(pair['n_ref_neurons'])))
            self.pair_table.setItem(i, 2, QTableWidgetItem(str(pair['optimal_matches'])))
            self.pair_table.setItem(i, 3, QTableWidgetItem(f"{pair['optimal_rate']*100:.1f}%"))

        self.pair_table.resizeColumnsToContents()

    def on_pair_selected(self):
        """Handle pair selection"""
        selected_rows = self.pair_table.selectedItems()
        if not selected_rows:
            return

        row = selected_rows[0].row()

        if not self.current_animal:
            return

        pairs = self.sweep_data[self.current_animal]['pairs']
        if row >= len(pairs):
            return

        self.current_pair = pairs[row]

        # Load full pair data for detailed plots
        pair_data = self.load_pair_data(self.current_pair)
        if pair_data:
            self.plot_geometric_coherence(pair_data)
            self.plot_vote_support(pair_data)
            self.plot_match_quality(pair_data)
            self.update_neuron_selector(pair_data)

    def load_pair_data(self, pair):
        """Load full data for a session pair"""
        pair_name = pair['pair_name']

        if pair_name in self.pair_data_cache:
            return self.pair_data_cache[pair_name]

        npz_file = pair['file']
        data = load_npz_safely(npz_file, verbose=False)

        if data is None:
            return None

        ref_centroids = data['ref_centroids']
        tgt_centroids = data['tgt_centroids']

        matched_ref_indices = data.get('matched_ref_indices', np.array([]))
        matched_tgt_indices = data.get('matched_tgt_indices', np.array([]))
        matched_costs = data.get('matched_costs', np.array([]))

        # Reconstruct vote and cost matrices from Step 2.5
        step2_5_dir = Path(self.config.output_dir) / 'step_2_5_results'
        if not step2_5_dir.exists():
            step2_5_dir = Path(self.config.output_dir) / 'step_2_5'

        vote_matrix = None
        cost_matrix = None

        filter_file = step2_5_dir / f"{pair_name}_filtered_matches.npz"
        if filter_file.exists():
            filter_data = load_npz_safely(filter_file, verbose=False)
            if filter_data:
                match_indices = filter_data.get('match_indices')

                if match_indices is not None and len(match_indices) > 0:
                    n_ref = len(ref_centroids)
                    n_tgt = len(tgt_centroids)
                    vote_matrix = np.zeros((n_ref, n_tgt), dtype=np.int32)

                    for quad_match in match_indices:
                        ref_neurons = quad_match[:4].astype(int)
                        tgt_neurons = quad_match[4:].astype(int)

                        for i_ref in ref_neurons:
                            for i_tgt in tgt_neurons:
                                if i_ref < n_ref and i_tgt < n_tgt:
                                    vote_matrix[i_ref, i_tgt] += 1

                    max_votes = vote_matrix.max() if vote_matrix.size > 0 else 1
                    cost_matrix = (max_votes - vote_matrix).astype(np.float32)
                    cost_matrix[vote_matrix == 0] = 1e6

        pair_data = {
            'pair_name': pair_name,
            'ref_centroids': ref_centroids,
            'tgt_centroids': tgt_centroids,
            'matched_ref_indices': matched_ref_indices,
            'matched_tgt_indices': matched_tgt_indices,
            'matched_costs': matched_costs,
            'vote_matrix': vote_matrix,
            'cost_matrix': cost_matrix,
            'n_ref': len(ref_centroids),
            'n_tgt': len(tgt_centroids),
        }

        self.pair_data_cache[pair_name] = pair_data
        return pair_data

    # =========================================================================
    # ALL PAIRS OVERVIEW
    # =========================================================================

    def plot_all_pairs_overview(self, sweep_info):
        """Plot overview of all pairs — match rates and neuron counts"""
        self.overview_figure.clear()
        fig = self.overview_figure

        pairs = sweep_info['pairs']

        if not pairs:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No pairs available', ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.overview_canvas.draw()
            return

        pair_names = [p['pair_name'] for p in pairs]
        match_rates = [p['optimal_rate'] * 100 for p in pairs]
        match_counts = [p['optimal_matches'] for p in pairs]
        ref_neurons = [p['n_ref_neurons'] for p in pairs]
        tgt_neurons = [p['n_target_neurons'] for p in pairs]

        # Match rate bar chart
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['#2ecc71' if r >= 70 else '#f39c12' if r >= 50 else '#e74c3c'
                  for r in match_rates]
        bars = ax1.barh(range(len(pairs)), match_rates, color=colors,
                        edgecolor='black', alpha=0.8)
        ax1.set_yticks(range(len(pairs)))
        ax1.set_yticklabels(pair_names, fontsize=8)
        ax1.set_xlabel('Match Rate (%)', fontsize=10)
        ax1.set_title('Match Rate by Pair', fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 105)
        ax1.axvline(sweep_info['avg_optimal_rate'] * 100, color='red',
                    linestyle='--', linewidth=2, alpha=0.7,
                    label=f"Mean: {sweep_info['avg_optimal_rate']*100:.1f}%")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        # Match rate distribution
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(match_rates, bins=min(15, len(pairs)),
                 edgecolor='black', alpha=0.7, color='blue')
        ax2.axvline(sweep_info['avg_optimal_rate'] * 100, color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {sweep_info['avg_optimal_rate']*100:.1f}%")
        ax2.set_xlabel('Match Rate (%)', fontsize=10)
        ax2.set_ylabel('# Pairs', fontsize=10)
        ax2.set_title('Match Rate Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()

        # Neuron counts per pair
        ax3 = fig.add_subplot(2, 2, 3)
        x = np.arange(len(pairs))
        width = 0.35
        ax3.bar(x - width/2, ref_neurons, width, label='Ref', color='steelblue',
                edgecolor='black', alpha=0.8)
        ax3.bar(x + width/2, tgt_neurons, width, label='Tgt', color='coral',
                edgecolor='black', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=7)
        ax3.set_ylabel('# Neurons', fontsize=10)
        ax3.set_title('Neuron Counts per Pair', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # Summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        summary_text = f"""
        {self.current_animal}
        {'='*30}

        Pairs: {sweep_info['n_pairs']}

        Match Rates:
        • Mean: {sweep_info['avg_optimal_rate']*100:.1f}%
        • Range: {min(match_rates):.1f}% – {max(match_rates):.1f}%

        Total: {sweep_info['total_optimal_matches']} matches

        Method: Padded Hungarian
        + RANSAC post-filter
        + 2nd-pass recovery
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        fig.tight_layout()
        self.overview_canvas.draw()

    # =========================================================================
    # GEOMETRIC COHERENCE
    # =========================================================================

    def plot_geometric_coherence(self, pair_data):
        """Plot geometric coherence — matched pairs preserve spatial structure"""
        self.geom_figure.clear()
        fig = self.geom_figure

        ref_centroids = pair_data['ref_centroids']
        tgt_centroids = pair_data['tgt_centroids']
        matched_ref = pair_data['matched_ref_indices']
        matched_tgt = pair_data['matched_tgt_indices']

        if len(matched_ref) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No matches', ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.geom_canvas.draw()
            return

        matched_ref_set = set(matched_ref)
        matched_tgt_set = set(matched_tgt)

        n_matches = len(matched_ref)
        if n_matches <= 20:
            match_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_matches)))
        else:
            match_colors = plt.cm.hsv(np.linspace(0, 0.9, n_matches))

        # Reference session
        ax1 = fig.add_subplot(1, 2, 1)

        unmatched_ref = [i for i in range(len(ref_centroids)) if i not in matched_ref_set]
        if unmatched_ref:
            ax1.scatter(ref_centroids[unmatched_ref, 0], ref_centroids[unmatched_ref, 1],
                        s=20, c='lightgray', alpha=0.3, label='Unmatched')

        for i, ref_idx in enumerate(matched_ref):
            color = match_colors[i % len(match_colors)]
            ax1.scatter(ref_centroids[ref_idx, 0], ref_centroids[ref_idx, 1],
                        s=50, c=[color], edgecolors='black', linewidth=1, zorder=10)

        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        ax1.set_title('Reference Session\n(colored by match ID)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend()

        # Target session
        ax2 = fig.add_subplot(1, 2, 2)

        unmatched_tgt = [i for i in range(len(tgt_centroids)) if i not in matched_tgt_set]
        if unmatched_tgt:
            ax2.scatter(tgt_centroids[unmatched_tgt, 0], tgt_centroids[unmatched_tgt, 1],
                        s=20, c='lightgray', alpha=0.3, label='Unmatched')

        for i, tgt_idx in enumerate(matched_tgt):
            color = match_colors[i % len(match_colors)]
            ax2.scatter(tgt_centroids[tgt_idx, 0], tgt_centroids[tgt_idx, 1],
                        s=50, c=[color], edgecolors='black', linewidth=1, zorder=10)

        ax2.set_xlabel('X (pixels)', fontsize=10)
        ax2.set_ylabel('Y (pixels)', fontsize=10)
        ax2.set_title('Target Session\n(same colors = matched)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.legend()

        fig.suptitle(f'Geometric Coherence: {pair_data["pair_name"]}\n'
                     f'{len(matched_ref)} matches preserve spatial structure',
                     fontsize=12, fontweight='bold')

        fig.tight_layout()
        self.geom_canvas.draw()

    # =========================================================================
    # VOTE SUPPORT
    # =========================================================================

    def plot_vote_support(self, pair_data):
        """Plot vote support for matches"""
        self.vote_figure.clear()
        fig = self.vote_figure

        vote_matrix = pair_data.get('vote_matrix')
        matched_ref = pair_data['matched_ref_indices']
        matched_tgt = pair_data['matched_tgt_indices']

        if vote_matrix is None or len(matched_ref) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Vote data not available', ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.vote_canvas.draw()
            return

        match_votes = np.array([vote_matrix[r, t] for r, t in zip(matched_ref, matched_tgt)])

        # Vote distribution
        ax1 = fig.add_subplot(2, 2, 1)
        plot_histogram_with_stats(
            ax1, match_votes, bins=range(0, max(match_votes) + 2),
            title='Vote Distribution', xlabel='# Quads Voting',
            ylabel='# Matches', show_mean=True, show_median=True
        )

        # Sorted votes
        ax2 = fig.add_subplot(2, 2, 2)
        sorted_votes = np.sort(match_votes)[::-1]
        colors = plt.cm.RdYlGn(sorted_votes / max(sorted_votes))
        ax2.bar(range(len(sorted_votes)), sorted_votes, color=colors, edgecolor='black')
        ax2.set_xlabel('Match Rank', fontsize=10)
        ax2.set_ylabel('Votes', fontsize=10)
        ax2.set_title('Vote Support by Match', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Cumulative
        ax3 = fig.add_subplot(2, 2, 3)
        cumsum = np.cumsum(sorted_votes)
        ax3.plot(range(len(cumsum)), cumsum / cumsum[-1] * 100, 'b-', linewidth=2)
        ax3.axhline(y=80, color='r', linestyle='--', label='80%')
        ax3.set_xlabel('Top N Matches', fontsize=10)
        ax3.set_ylabel('Cumulative %', fontsize=10)
        ax3.set_title('Cumulative Votes', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Categories
        ax4 = fig.add_subplot(2, 2, 4)
        high = np.sum(match_votes >= 5)
        med = np.sum((match_votes >= 2) & (match_votes < 5))
        low = np.sum(match_votes < 2)

        categories = ['High\n(≥5)', 'Med\n(2-4)', 'Low\n(<2)']
        counts = [high, med, low]
        colors_cat = ['green', 'orange', 'red']

        bars = ax4.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('# Matches', fontsize=10)
        ax4.set_title('Confidence Categories', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            pct = 100 * count / len(match_votes) if len(match_votes) > 0 else 0
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

        fig.suptitle(f'Vote Support: {pair_data["pair_name"]}', fontsize=13, fontweight='bold')
        fig.tight_layout()
        self.vote_canvas.draw()

    # =========================================================================
    # MATCH QUALITY
    # =========================================================================

    def plot_match_quality(self, pair_data):
        """Plot match quality metrics"""
        self.quality_figure.clear()
        fig = self.quality_figure

        ref_centroids = pair_data['ref_centroids']
        tgt_centroids = pair_data['tgt_centroids']
        matched_ref = pair_data['matched_ref_indices']
        matched_tgt = pair_data['matched_tgt_indices']
        matched_costs = pair_data['matched_costs']
        vote_matrix = pair_data.get('vote_matrix')

        if len(matched_ref) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No matches', ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.quality_canvas.draw()
            return

        distances = []
        votes = []

        for ref_idx, tgt_idx in zip(matched_ref, matched_tgt):
            dist = np.linalg.norm(ref_centroids[ref_idx] - tgt_centroids[tgt_idx])
            distances.append(dist)
            if vote_matrix is not None:
                votes.append(vote_matrix[ref_idx, tgt_idx])
            else:
                votes.append(0)

        distances = np.array(distances)
        votes = np.array(votes)

        # Distance vs Votes
        ax1 = fig.add_subplot(2, 2, 1)
        if vote_matrix is not None and np.max(votes) > 0:
            scatter = ax1.scatter(distances, votes, c=matched_costs,
                                  cmap='RdYlGn_r', s=50, alpha=0.6,
                                  edgecolors='black', linewidth=0.5)
            ax1.set_ylabel('Votes', fontsize=10)
            fig.colorbar(scatter, ax=ax1, label='Cost')
        else:
            ax1.scatter(distances, matched_costs, s=50, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
            ax1.set_ylabel('Cost', fontsize=10)

        ax1.set_xlabel('Distance (px)', fontsize=10)
        ax1.set_title('Distance vs Evidence', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Distance distribution
        ax2 = fig.add_subplot(2, 2, 2)
        plot_histogram_with_stats(
            ax2, distances, bins=30, title='Distance Distribution',
            xlabel='Distance (px)', ylabel='Frequency',
            show_mean=True, show_median=True
        )

        # Cost distribution
        ax3 = fig.add_subplot(2, 2, 3)
        finite_costs = matched_costs[matched_costs < 1e5]
        if len(finite_costs) > 0:
            plot_histogram_with_stats(
                ax3, finite_costs, bins=30, title='Cost Distribution',
                xlabel='Cost', ylabel='Frequency',
                show_mean=True, show_median=True
            )

        # Summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        metrics_text = f"""
        Quality Metrics
        {'='*30}

        Matches: {len(matched_ref)}

        Distance:
        • Mean: {np.mean(distances):.2f} px
        • Median: {np.median(distances):.2f} px
        • Std: {np.std(distances):.2f} px
        """

        if vote_matrix is not None and np.max(votes) > 0:
            metrics_text += f"""
        Votes:
        • Mean: {np.mean(votes):.1f}
        • Median: {np.median(votes):.0f}
        • Max: {np.max(votes)}

        Strong (≥5): {np.sum(votes >= 5)}
        Weak (<2): {np.sum(votes < 2)}
        """

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'Match Quality: {pair_data["pair_name"]}', fontsize=13, fontweight='bold')
        fig.tight_layout()
        self.quality_canvas.draw()

    # =========================================================================
    # MATCH ALTERNATIVES
    # =========================================================================

    def update_neuron_selector(self, pair_data):
        """Update neuron selector"""
        self.neuron_selector.clear()

        matched_ref = pair_data['matched_ref_indices']
        matched_tgt = pair_data['matched_tgt_indices']

        for i, (ref_idx, tgt_idx) in enumerate(zip(matched_ref, matched_tgt)):
            self.neuron_selector.addItem(
                f"Match {i}: Ref#{ref_idx} → Tgt#{tgt_idx}",
                (ref_idx, tgt_idx, i)
            )

        if len(matched_ref) > 0:
            self.neuron_selector.setCurrentIndex(0)

    def on_neuron_selected(self, index):
        """Handle neuron selection"""
        if index < 0 or self.current_pair is None:
            return

        pair_data = self.pair_data_cache.get(self.current_pair['pair_name'])
        if pair_data is None:
            return

        data = self.neuron_selector.itemData(index)
        if data is None:
            return

        ref_idx, tgt_idx, match_idx = data
        self.plot_match_alternatives(pair_data, ref_idx, tgt_idx)

    def plot_match_alternatives(self, pair_data, ref_idx, tgt_idx):
        """Plot alternative match candidates"""
        self.alt_figure.clear()
        fig = self.alt_figure

        cost_matrix = pair_data.get('cost_matrix')
        vote_matrix = pair_data.get('vote_matrix')
        ref_centroids = pair_data['ref_centroids']
        tgt_centroids = pair_data['tgt_centroids']

        if cost_matrix is None:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Cost matrix not available',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.alt_canvas.draw()
            return

        costs = cost_matrix[ref_idx, :]
        valid_costs = costs < 1e5

        if not np.any(valid_costs):
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No valid alternatives', ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.alt_canvas.draw()
            return

        top_indices = np.argsort(costs)[:5]
        top_costs = costs[top_indices]

        # Spatial view
        ax1 = fig.add_subplot(1, 2, 1)

        ax1.scatter(tgt_centroids[:, 0], tgt_centroids[:, 1],
                    s=20, c='lightgray', alpha=0.3, label='All')

        ax1.scatter([ref_centroids[ref_idx, 0]], [ref_centroids[ref_idx, 1]],
                    s=100, c='blue', marker='*', edgecolors='black',
                    linewidth=2, label=f'Ref#{ref_idx}', zorder=10)

        alt_colors = ['green', 'yellow', 'orange', 'red', 'purple']
        for rank, (alt_idx, alt_cost) in enumerate(zip(top_indices, top_costs)):
            if alt_cost >= 1e5:
                continue

            size = 150 if alt_idx == tgt_idx else 80
            marker = 'o' if alt_idx == tgt_idx else 's'

            ax1.scatter([tgt_centroids[alt_idx, 0]], [tgt_centroids[alt_idx, 1]],
                        s=size, c=alt_colors[rank], marker=marker,
                        edgecolors='black', linewidth=2,
                        label=f'#{rank+1}: Tgt#{alt_idx}' + (' ✓' if alt_idx == tgt_idx else ''),
                        zorder=9 - rank)

            ax1.plot([ref_centroids[ref_idx, 0], tgt_centroids[alt_idx, 0]],
                     [ref_centroids[ref_idx, 1], tgt_centroids[alt_idx, 1]],
                     '--', color=alt_colors[rank], alpha=0.5, linewidth=1.5)

        ax1.set_xlabel('X (px)', fontsize=10)
        ax1.set_ylabel('Y (px)', fontsize=10)
        ax1.set_title('Alternatives (Circle = chosen)', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Comparison table
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')

        table_data = []
        for rank, (alt_idx, alt_cost) in enumerate(zip(top_indices, top_costs)):
            if alt_cost >= 1e5:
                continue

            dist = np.linalg.norm(ref_centroids[ref_idx] - tgt_centroids[alt_idx])
            votes = vote_matrix[ref_idx, alt_idx] if vote_matrix is not None else 0
            chosen = '✓' if alt_idx == tgt_idx else ''

            table_data.append([
                f"{rank + 1}",
                f"Tgt#{alt_idx}",
                f"{alt_cost:.1f}",
                f"{dist:.1f}px",
                f"{votes}",
                chosen
            ])

        if table_data:
            table = ax2.table(
                cellText=table_data,
                colLabels=['Rank', 'Target', 'Cost', 'Dist', 'Votes', '✓'],
                cellLoc='center', loc='center',
                bbox=[0.1, 0.3, 0.8, 0.6]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for i, row in enumerate(table_data):
                if row[5] == '✓':
                    for j in range(6):
                        table[(i + 1, j)].set_facecolor('lightgreen')

        ax2.set_title(f'Alternatives for Ref#{ref_idx}', fontsize=11, fontweight='bold', pad=20)

        fig.tight_layout()
        self.alt_canvas.draw()

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_data(self):
        """Export tracking data"""
        if not self.current_animal:
            QMessageBox.information(self, 'Export', 'Please select an animal first.')
            return

        tracking = self.tracking_data.get(self.current_animal)
        sweep_info = self.sweep_data[self.current_animal]

        export_text = f'Step 3 Results: {self.current_animal}\n\n'

        if tracking:
            export_text += f'TRACKING:\n'
            export_text += f'  Total tracks: {tracking["n_total_tracks"]}\n'
            export_text += f'  Full-length: {tracking["full_length_tracks"]}\n'
            export_text += f'  Sessions: {tracking["n_sessions"]}\n'
            export_text += f'  Avg length: {tracking["avg_track_length"]:.1f}\n\n'

        export_text += f'MATCHING:\n'
        export_text += f'  Pairs: {sweep_info["n_pairs"]}\n'
        export_text += f'  Avg match rate: {sweep_info["avg_optimal_rate"]*100:.1f}%\n'
        export_text += f'  Total matches: {sweep_info["total_optimal_matches"]}\n'

        QMessageBox.information(self, 'Export', export_text)