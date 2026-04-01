"""
Step 3 Results Viewer: RANSAC-Informed Matching Analysis

V3.1: Two plot tabs with per-match confidence coloring and per-track
      confidence distributions.
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QGroupBox, QMessageBox,
                             QSplitter, QTabWidget, QWidget, QLabel)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

from utilities import *


class Step3Viewer(QDialog):

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_pair = None
        self.sweep_data = {}
        self.tracking_data = {}
        self.pair_data_cache = {}

        self.setWindowTitle('Step 3: Neuron Matching & Tracking')
        self.resize(1600, 900)
        self.init_ui()
        self.load_results()

    def init_ui(self):
        layout = QVBoxLayout()

        self.controls = create_top_controls(
            'Animal Selection',
            combos=[('Animal', self.on_animal_changed)],
            refresh_callback=self.load_results
        )
        self.animal_combo = self.controls.combos[0]
        layout.addWidget(self.controls)

        self.tracking_header = QLabel()
        self.tracking_header.setAlignment(Qt.AlignCenter)
        self.tracking_header.setStyleSheet('''
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white; font-size: 16px; font-weight: bold;
                padding: 15px; border: 2px solid #00AAFF;
                border-radius: 8px; margin: 5px;
            }
        ''')
        self.tracking_header.setText('Select an animal above')
        layout.addWidget(self.tracking_header)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        table_group = QGroupBox('Session Pairs')
        table_layout = QVBoxLayout()
        self.pair_table = QTableWidget()
        self.pair_table.setColumnCount(4)
        self.pair_table.setHorizontalHeaderLabels(['Pair', 'Ref N', 'Matches', 'Rate'])
        self.pair_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pair_table.setSelectionMode(QTableWidget.SingleSelection)
        self.pair_table.itemSelectionChanged.connect(self.on_pair_selected)
        table_layout.addWidget(self.pair_table)
        table_group.setLayout(table_layout)
        left_layout.addWidget(table_group)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        ov_tab, self.ov_fig, self.ov_canvas = create_matplotlib_tab('Overview')
        self.tabs.addTab(ov_tab, '📊 Overview')
        dt_tab, self.dt_fig, self.dt_canvas = create_matplotlib_tab('Pair Detail')
        self.tabs.addTab(dt_tab, '🔗 Pair Detail')
        st_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(3, self.config.output_dir)
        self.tabs.addTab(st_tab, '📋 Statistics')
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)
        layout.addWidget(create_action_buttons({
            '💾 Export': self.export_data, '✖ Close': self.accept
        }))
        self.setLayout(layout)

    # =====================================================================
    # DATA LOADING
    # =====================================================================

    def load_results(self):
        step3_dir = get_step_results_dir(self.config.output_dir, 3)
        if not step3_dir.exists():
            show_no_results_error(self, step3_dir, 3)
            return

        summary_file = step3_dir / 'step3_summary.json'
        if not summary_file.exists():
            QMessageBox.warning(self, 'No Summary',
                                f'Summary not found:\n{summary_file}\nRun Step 3 first.')
            return

        summary = load_json_safely(summary_file, verbose=True)
        if summary is None:
            return

        self.sweep_data.clear()
        self.tracking_data.clear()
        self.pair_data_cache.clear()
        self.animal_combo.clear()

        for ar in summary['animals']:
            aid = str(ar['animal_id'])
            pairs = []
            for npz_file in sorted(step3_dir.glob(f"{aid}_*_sweep.npz")):
                data = load_npz_safely(npz_file, verbose=False)
                if data:
                    pairs.append({
                        'pair_name': str(data.get('pair_name', npz_file.stem.replace('_sweep', ''))),
                        'file': npz_file,
                        'ref_session': str(data.get('ref_session', '')),
                        'target_session': str(data.get('target_session', '')),
                        'n_ref_neurons': int(data.get('n_ref_neurons', 0)),
                        'n_target_neurons': int(data.get('n_target_neurons', 0)),
                        'optimal_matches': int(data.get('optimal_matches', 0)),
                        'optimal_rate': float(data.get('optimal_rate', 0.0)),
                    })

            self.sweep_data[aid] = {
                'pairs': pairs, 'n_pairs': len(pairs),
                'avg_optimal_rate': ar.get('avg_optimal_rate', 0.0),
                'total_optimal_matches': ar.get('total_optimal_matches', 0),
            }

            tf = step3_dir / f"{aid}_consolidated_tracking.npz"
            if tf.exists():
                tracking = load_npz_safely(tf, verbose=False)
                if tracking:
                    self.tracking_data[aid] = tracking

        animals = sorted(self.sweep_data.keys())
        self.animal_combo.addItems(animals)
        if animals:
            self.animal_combo.setCurrentIndex(0)
        self.load_pipeline_stats()

    def load_pair_data(self, pair):
        pn = pair['pair_name']
        if pn in self.pair_data_cache:
            return self.pair_data_cache[pn]

        data = load_npz_safely(pair['file'], verbose=False)
        if data is None:
            return None

        ref_c = data['ref_centroids']
        tgt_c = data['tgt_centroids']

        # Reconstruct vote matrix from Step 2.5
        vote_matrix = None
        for subdir in ['step_2_5_results', 'step_2_5']:
            ff = Path(self.config.output_dir) / subdir / f"{pn}_filtered_matches.npz"
            if ff.exists():
                fd = load_npz_safely(ff, verbose=False)
                if fd is not None:
                    mi = fd.get('match_indices')
                    if mi is not None and len(mi) > 0:
                        nr, nt = len(ref_c), len(tgt_c)
                        vote_matrix = np.zeros((nr, nt), dtype=np.int32)
                        for qm in mi:
                            for ri in qm[:4].astype(int):
                                for ti in qm[4:].astype(int):
                                    if ri < nr and ti < nt:
                                        vote_matrix[ri, ti] += 1
                break

        pd = {
            'pair_name': pn,
            'ref_centroids': ref_c, 'tgt_centroids': tgt_c,
            'matched_ref_indices': data.get('matched_ref_indices', np.array([])),
            'matched_tgt_indices': data.get('matched_tgt_indices', np.array([])),
            'matched_costs': data.get('matched_costs', np.array([])),
            'match_confidence': data.get('match_confidence', None),
            'match_pass': data.get('match_pass', None),
            'vote_matrix': vote_matrix,
            'n_ref': len(ref_c), 'n_tgt': len(tgt_c),
        }
        self.pair_data_cache[pn] = pd
        return pd

    # =====================================================================
    # SELECTION
    # =====================================================================

    def on_animal_changed(self, aid):
        if not aid or aid not in self.sweep_data:
            return
        self.current_animal = aid
        self.display_animal(aid)

    def on_pair_selected(self):
        sel = self.pair_table.selectedItems()
        if not sel or not self.current_animal:
            return
        row = sel[0].row()
        pairs = self.sweep_data[self.current_animal]['pairs']
        if row >= len(pairs):
            return
        self.current_pair = pairs[row]
        pd = self.load_pair_data(self.current_pair)
        if pd:
            self.plot_pair_detail(pd)
            self.tabs.setCurrentIndex(1)

    # =====================================================================
    # DISPLAY
    # =====================================================================

    def display_animal(self, aid):
        si = self.sweep_data[aid]
        tr = self.tracking_data.get(aid)
        pairs = si['pairs']
        total = si['total_optimal_matches']
        avg_r = si['avg_optimal_rate']

        if tr is not None:
            nt = int(tr['n_total_tracks'])
            ft = int(tr['full_length_tracks'])
            fp = ft / nt * 100 if nt > 0 else 0
            tc = tr.get('track_confidence')
            if tc is not None and len(tc) > 0:
                full_mask = np.array(tr['track_lengths']) == int(tr['n_sessions'])
                if np.any(full_mask):
                    avg_conf = float(np.mean(np.array(tc)[full_mask]))
                    conf_str = f'  |  avg conf: {avg_conf:.2f}'
                else:
                    conf_str = ''
            else:
                conf_str = ''

            txt = (f'{aid}:  {total:,} matched  |  {avg_r*100:.1f}% rate  |  '
                   f'{ft}/{nt} full tracks ({fp:.0f}%){conf_str}')
            color = '#00FF00' if fp >= 70 else '#FFAA00' if fp >= 50 else '#FF6600'
        else:
            txt = f'{aid}:  {total:,} matched  |  {avg_r*100:.1f}% rate  |  no tracking'
            color = '#AAAAAA'

        self.tracking_header.setStyleSheet(f'''
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: {color}; font-size: 18px; font-weight: bold;
                padding: 15px; border: 2px solid {color};
                border-radius: 8px; margin: 5px;
            }}
        ''')
        self.tracking_header.setText(txt)

        self.pair_table.setRowCount(len(pairs))
        for i, p in enumerate(pairs):
            self.pair_table.setItem(i, 0, QTableWidgetItem(p['pair_name']))
            self.pair_table.setItem(i, 1, QTableWidgetItem(str(p['n_ref_neurons'])))
            self.pair_table.setItem(i, 2, QTableWidgetItem(str(p['optimal_matches'])))
            self.pair_table.setItem(i, 3, QTableWidgetItem(f"{p['optimal_rate']*100:.1f}%"))
        self.pair_table.resizeColumnsToContents()

        self.plot_overview(si, tr)
        if pairs:
            self.pair_table.selectRow(0)

    # =====================================================================
    # OVERVIEW
    # =====================================================================

    def plot_overview(self, si, tr):
        fig = self.ov_fig
        fig.clear()
        pairs = si['pairs']
        if not pairs:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No pairs', ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.ov_canvas.draw()
            return

        has_tr = tr is not None
        match_rates = [p['optimal_rate'] * 100 for p in pairs]
        pair_names = [p['pair_name'] for p in pairs]
        mean_r = si['avg_optimal_rate'] * 100

        n_cols = 2 if has_tr else 1
        gs = fig.add_gridspec(2, n_cols, hspace=0.45, wspace=0.35)

        # Match rate bars
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2ecc71' if r >= 70 else '#f39c12' if r >= 50 else '#e74c3c'
                  for r in match_rates]
        ax1.barh(range(len(pairs)), match_rates, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_yticks(range(len(pairs)))
        ax1.set_yticklabels(pair_names, fontsize=7)
        ax1.set_xlabel('Match Rate (%)')
        ax1.set_title('Match Rate by Pair', fontweight='bold')
        ax1.set_xlim(0, 105)
        ax1.axvline(mean_r, color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_r:.1f}%')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        # Match rate histogram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(match_rates, bins=min(15, len(pairs)), edgecolor='black',
                 alpha=0.7, color='steelblue')
        ax2.axvline(mean_r, color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_r:.1f}%')
        ax2.set_xlabel('Match Rate (%)')
        ax2.set_ylabel('# Pairs')
        ax2.set_title('Match Rate Distribution', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        if has_tr:
            tl = np.array(tr['track_lengths'])
            ns = int(tr['n_sessions'])
            tc = tr.get('track_confidence')

            # Track length histogram
            ax3 = fig.add_subplot(gs[0, 1])
            bins = np.arange(0.5, ns + 1.5, 1)
            ax3.hist(tl, bins=bins, edgecolor='black', alpha=0.7, color='coral')
            ax3.axvline(ns, color='green', linestyle='--', linewidth=2,
                        label=f'Full ({ns})')
            ax3.set_xlabel('Track Length (sessions)')
            ax3.set_ylabel('# Tracks')
            ax3.set_title('Track Length Distribution', fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

            # Track confidence distribution (full-length only)
            ax4 = fig.add_subplot(gs[1, 1])
            if tc is not None and len(tc) > 0:
                tc_arr = np.array(tc)
                full_mask = tl == ns

                if np.any(full_mask):
                    full_conf = tc_arr[full_mask]
                    c_colors = plt.cm.RdYlGn(full_conf)
                    ax4.hist(full_conf, bins=20, edgecolor='black', alpha=0.7,
                             color='teal')
                    ax4.axvline(np.median(full_conf), color='red', linestyle='--',
                                linewidth=1.5,
                                label=f'Median: {np.median(full_conf):.2f}')
                    ax4.axvline(0.5, color='orange', linestyle=':', linewidth=1.5,
                                label='High/low boundary')
                    high = int(np.sum(full_conf >= 0.5))
                    low = int(np.sum(full_conf < 0.5))
                    ax4.set_title(f'Full-Track Confidence ({high} high / {low} low)',
                                  fontweight='bold')
                else:
                    ax4.text(0.5, 0.5, 'No full-length tracks', ha='center',
                             va='center', fontsize=11, color='gray')
                    ax4.set_title('Track Confidence', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No confidence data\n(re-run Step 3)',
                         ha='center', va='center', fontsize=11, color='gray')
                ax4.set_title('Track Confidence', fontweight='bold')

            ax4.set_xlabel('Min Confidence')
            ax4.set_ylabel('# Tracks')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')

        fig.suptitle(self.current_animal, fontsize=14, fontweight='bold')
        self.ov_canvas.draw()

    # =====================================================================
    # PAIR DETAIL
    # =====================================================================

    def plot_pair_detail(self, pd):
        fig = self.dt_fig
        fig.clear()

        ref_c = pd['ref_centroids']
        tgt_c = pd['tgt_centroids']
        m_ref = pd['matched_ref_indices']
        m_tgt = pd['matched_tgt_indices']
        m_costs = pd['matched_costs']
        confidence = pd.get('match_confidence')
        match_pass = pd.get('match_pass')
        vote_matrix = pd.get('vote_matrix')

        if len(m_ref) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No matches', ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.dt_canvas.draw()
            return

        n_matches = len(m_ref)
        has_conf = confidence is not None and len(confidence) == n_matches

        distances = np.array([
            np.linalg.norm(ref_c[r] - tgt_c[t]) for r, t in zip(m_ref, m_tgt)
        ])
        votes = np.array([
            vote_matrix[r, t] if vote_matrix is not None else 0
            for r, t in zip(m_ref, m_tgt)
        ])

        m_ref_set = set(m_ref.tolist() if hasattr(m_ref, 'tolist') else list(m_ref))
        m_tgt_set = set(m_tgt.tolist() if hasattr(m_tgt, 'tolist') else list(m_tgt))

        # Use confidence for coloring if available, else match index colors
        if has_conf:
            conf_arr = np.array(confidence)
            match_colors = plt.cm.RdYlGn(conf_arr)
        elif n_matches <= 20:
            match_colors = plt.cm.tab20(np.linspace(0, 1, max(n_matches, 1)))
        else:
            match_colors = plt.cm.hsv(np.linspace(0, 0.9, n_matches))

        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        # Reference spatial map
        ax_ref = fig.add_subplot(gs[0, 0])
        unmatched = [i for i in range(len(ref_c)) if i not in m_ref_set]
        if unmatched:
            ax_ref.scatter(ref_c[unmatched, 0], ref_c[unmatched, 1],
                           s=15, c='lightgray', alpha=0.3)
        for i, ri in enumerate(m_ref):
            ax_ref.scatter(ref_c[ri, 0], ref_c[ri, 1], s=40,
                           c=[match_colors[i]], edgecolors='black',
                           linewidth=0.5, zorder=5)
        label = 'Reference (colored by confidence)' if has_conf else 'Reference'
        ax_ref.set_title(label, fontweight='bold', fontsize=10)
        ax_ref.set_aspect('equal')
        ax_ref.grid(True, alpha=0.2)
        ax_ref.tick_params(labelsize=7)

        # Target spatial map
        ax_tgt = fig.add_subplot(gs[0, 1])
        unmatched = [i for i in range(len(tgt_c)) if i not in m_tgt_set]
        if unmatched:
            ax_tgt.scatter(tgt_c[unmatched, 0], tgt_c[unmatched, 1],
                           s=15, c='lightgray', alpha=0.3)
        for i, ti in enumerate(m_tgt):
            ax_tgt.scatter(tgt_c[ti, 0], tgt_c[ti, 1], s=40,
                           c=[match_colors[i]], edgecolors='black',
                           linewidth=0.5, zorder=5)
        ax_tgt.set_title('Target (same colors)', fontweight='bold', fontsize=10)
        ax_tgt.set_aspect('equal')
        ax_tgt.grid(True, alpha=0.2)
        ax_tgt.tick_params(labelsize=7)

        # Confidence histogram or distance-vs-votes scatter
        ax_top_r = fig.add_subplot(gs[0, 2])
        if has_conf:
            ax_top_r.hist(conf_arr, bins=20, edgecolor='black', alpha=0.7, color='teal')
            ax_top_r.axvline(np.median(conf_arr), color='red', linestyle='--',
                             linewidth=1.5, label=f'Median: {np.median(conf_arr):.2f}')
            ax_top_r.axvline(0.5, color='orange', linestyle=':', linewidth=1.5,
                             label='High/low')
            high = int(np.sum(conf_arr >= 0.5))
            low = int(np.sum(conf_arr < 0.5))
            ax_top_r.set_title(f'Confidence ({high} high / {low} low)',
                               fontweight='bold', fontsize=10)
            ax_top_r.set_xlabel('Confidence')
            ax_top_r.set_ylabel('Count')
            ax_top_r.legend(fontsize=7)
        else:
            has_votes = vote_matrix is not None and np.max(votes) > 0
            if has_votes:
                sc = ax_top_r.scatter(distances, votes, c=m_costs, cmap='RdYlGn_r',
                                       s=35, alpha=0.7, edgecolors='black', linewidth=0.3)
                fig.colorbar(sc, ax=ax_top_r, label='Cost', fraction=0.046, pad=0.04)
                ax_top_r.set_ylabel('Votes')
            else:
                ax_top_r.scatter(distances, m_costs, s=35, alpha=0.7,
                                 edgecolors='black', linewidth=0.3, color='steelblue')
                ax_top_r.set_ylabel('Cost')
            ax_top_r.set_xlabel('Distance (px)')
            ax_top_r.set_title('Distance vs Evidence', fontweight='bold', fontsize=10)
        ax_top_r.grid(True, alpha=0.3)
        ax_top_r.tick_params(labelsize=7)

        # Distance histogram
        ax_dist = fig.add_subplot(gs[1, 0])
        if has_conf:
            # Color bars by average confidence in each bin
            n_bins = 25
            counts, bin_edges = np.histogram(distances, bins=n_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_colors = []
            for b_lo, b_hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (distances >= b_lo) & (distances < b_hi)
                if np.any(mask):
                    bin_colors.append(plt.cm.RdYlGn(np.mean(conf_arr[mask])))
                else:
                    bin_colors.append('lightgray')
            ax_dist.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
                        color=bin_colors, edgecolor='black', alpha=0.8)
        else:
            ax_dist.hist(distances, bins=25, edgecolor='black', alpha=0.7,
                         color='steelblue')
        ax_dist.axvline(np.median(distances), color='red', linestyle='--',
                        linewidth=1.5, label=f'Median: {np.median(distances):.1f}px')
        ax_dist.set_xlabel('Distance (px)')
        ax_dist.set_ylabel('Count')
        ax_dist.set_title('Match Distances', fontweight='bold', fontsize=10)
        ax_dist.legend(fontsize=7)
        ax_dist.grid(True, alpha=0.3, axis='y')
        ax_dist.tick_params(labelsize=7)

        # Vote histogram
        ax_vote = fig.add_subplot(gs[1, 1])
        has_votes = vote_matrix is not None and np.max(votes) > 0
        if has_votes:
            max_v = int(np.max(votes))
            ax_vote.hist(votes, bins=range(0, max_v + 2), edgecolor='black',
                         alpha=0.7, color='coral')
            ax_vote.axvline(np.median(votes), color='red', linestyle='--',
                            linewidth=1.5, label=f'Median: {np.median(votes):.0f}')
            ax_vote.set_xlabel('Quad Votes')
            ax_vote.set_ylabel('Count')
            ax_vote.set_title('Vote Support', fontweight='bold', fontsize=10)
            ax_vote.legend(fontsize=7)
        else:
            ax_vote.text(0.5, 0.5, 'Vote data\nnot available',
                         ha='center', va='center', fontsize=11, color='gray')
            ax_vote.axis('off')
        ax_vote.grid(True, alpha=0.3, axis='y')
        ax_vote.tick_params(labelsize=7)

        # Summary text
        ax_sum = fig.add_subplot(gs[1, 2])
        ax_sum.axis('off')

        rate_pct = n_matches / pd['n_ref'] * 100 if pd['n_ref'] > 0 else 0

        lines = [
            f"{'Match Summary':^28}",
            f"{'─'*28}",
            "",
            f"  Ref neurons:    {pd['n_ref']}",
            f"  Tgt neurons:    {pd['n_tgt']}",
            f"  Matches:        {n_matches}  ({rate_pct:.1f}%)",
            "",
            f"  Distance (px):",
            f"    mean:  {np.mean(distances):.2f}",
            f"    med:   {np.median(distances):.2f}",
            f"    max:   {np.max(distances):.2f}",
        ]

        if has_conf:
            p1 = int(np.sum(np.array(match_pass) == 1)) if match_pass is not None else n_matches
            p2 = n_matches - p1
            lines += [
                "",
                f"  Confidence:",
                f"    mean:  {np.mean(conf_arr):.3f}",
                f"    med:   {np.median(conf_arr):.3f}",
                f"    min:   {np.min(conf_arr):.3f}",
                "",
                f"  Pass 1: {p1}  Pass 2: {p2}",
            ]

        if has_votes:
            high = int(np.sum(votes >= 5))
            med_v = int(np.sum((votes >= 2) & (votes < 5)))
            low = int(np.sum(votes < 2))
            lines += [
                "",
                f"  Votes:",
                f"    high (>=5): {high}",
                f"    med  (2-4): {med_v}",
                f"    low  (<2):  {low}",
            ]

        ax_sum.text(0.05, 0.95, '\n'.join(lines), transform=ax_sum.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        fig.suptitle(pd['pair_name'], fontsize=12, fontweight='bold')
        self.dt_canvas.draw()

    # =====================================================================
    # EXPORT
    # =====================================================================

    def export_data(self):
        if not self.current_animal:
            QMessageBox.information(self, 'Export', 'Select an animal first.')
            return

        tr = self.tracking_data.get(self.current_animal)
        si = self.sweep_data[self.current_animal]

        lines = [f'Step 3 Results: {self.current_animal}\n']

        if tr:
            lines.append('TRACKING:')
            lines.append(f'  Total tracks:   {tr["n_total_tracks"]}')
            lines.append(f'  Full-length:    {tr["full_length_tracks"]}')
            lines.append(f'  Sessions:       {tr["n_sessions"]}')
            lines.append(f'  Avg length:     {tr["avg_track_length"]:.1f}')

            tc = tr.get('track_confidence')
            if tc is not None and len(tc) > 0:
                tc_arr = np.array(tc)
                full_mask = np.array(tr['track_lengths']) == int(tr['n_sessions'])
                if np.any(full_mask):
                    fc = tc_arr[full_mask]
                    lines.append(f'  Full-track confidence:')
                    lines.append(f'    mean: {np.mean(fc):.3f}')
                    lines.append(f'    med:  {np.median(fc):.3f}')
                    lines.append(f'    high (>=0.5): {int(np.sum(fc >= 0.5))}')
            lines.append('')

        lines.append('MATCHING:')
        lines.append(f'  Pairs:          {si["n_pairs"]}')
        lines.append(f'  Avg match rate: {si["avg_optimal_rate"]*100:.1f}%')
        lines.append(f'  Total matches:  {si["total_optimal_matches"]}')
        lines.append('')

        for p in si['pairs']:
            lines.append(f'  {p["pair_name"]}: {p["optimal_matches"]} '
                         f'({p["optimal_rate"]*100:.1f}%)')

        QMessageBox.information(self, 'Export', '\n'.join(lines))