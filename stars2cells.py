"""
Stars2Cells Unified Viewer 
Single interface for data loading, visualization, and pipeline execution
"""

import numpy as np
import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QLabel,
                             QColorDialog, QCheckBox, QMessageBox, QFileDialog,
                             QSplitter, QGroupBox, QListWidgetItem, QDialog,
                             QAction, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg

# Import pipeline config
from utilities import *

class Stars2CellsViewer(QMainWindow):
    """Unified viewer for Stars2Cells pipeline"""
    
    def __init__(self):
        super().__init__()
        set_application_icon(self, icon_path="S2C_logo.png")
        
        # Session data
        self.sessions = {}
        self.session_list = []
        self.session_file_paths = {}
        self.current_session = None
        self.data_folder = None
        
        # Pipeline
        self.config = None
        self.pipeline_worker = None
        
        # Visualization
        self.point_colors = {}
        self.default_color = '#00FFFF'
        self.default_edge_color = '#FFFFFF'
        self.selection_color = '#FFFF00'
        self.current_point_size = 10
        
        # Selection state
        self.selected_indices = set()
        self.deleted_indices = {}
        self.region_select_mode = False
        self.click_select_mode = False
        self.region_start = None
        self.region_rect = None
        
        # Temporary refs for pipeline callbacks
        self._current_progress = None
        self._current_log_text = None
        
        self.init_ui()
        apply_dark_theme_to_widget(self)
        self.create_menu_bar()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle('Stars2Cells Unified Viewer')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        
        main_layout.addWidget(splitter)
        
    def create_left_panel(self):
        """Create left sidebar with sessions and pipeline controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data section
        layout.addWidget(self._create_data_header())
        layout.addWidget(self._create_load_button())
        layout.addWidget(self._create_session_list())
        
        # Pipeline section
        layout.addWidget(self._create_pipeline_group())
        
        # Session info
        layout.addWidget(self._create_info_label())
        
        return panel
    
    def _create_data_header(self):
        """Create data section header"""
        header = QLabel('📁 Sessions')
        header.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; color: white;')
        return header
    
    def _create_load_button(self):
        """Create load data button"""
        self.load_btn = QPushButton('🗂️  Load Data Folder')
        self.load_btn.clicked.connect(self.load_folder)
        apply_button_style(self.load_btn, 'primary')
        return self.load_btn
    
    def _create_session_list(self):
        """Create session list widget"""
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_session_selected)
        self.file_list.setMinimumHeight(200)  # Set minimum height for scrolling
        self.file_list.setStyleSheet('''
            QListWidget {
                background-color: #1E1E1E;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
                color: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #00AAFF;
                color: white;
            }
        ''')
        return self.file_list
    
    def _create_pipeline_group(self):
        """Create pipeline controls group"""
        group = QGroupBox('🌟 Stars2Cells Pipeline')
        group.setStyleSheet('''
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #FF00FF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        
        layout = QVBoxLayout()
        layout.addWidget(self._create_config_button())
        layout.addLayout(self._create_step_controls(1, 'Step 1: Quad Generation'))
        layout.addLayout(self._create_step_controls(1.5, 'Step 1.5: Quad Matching Sweep'))
        layout.addLayout(self._create_step_controls(2, 'Step 2: Quad Matching'))
        layout.addLayout(self._create_step_controls(2.5, 'Step 2.5: RANSAC Filtering'))
        layout.addLayout(self._create_step_controls(3, 'Step 3: Neuron Matching'))
        layout.addWidget(self._create_run_all_button())
        
        group.setLayout(layout)
        return group
    
    def _create_config_button(self):
        """Create configuration button"""
        config_btn = QPushButton('⚙️  Configure Pipeline')
        config_btn.clicked.connect(self.configure_pipeline)
        apply_button_style(config_btn, 'config')
        return config_btn
    
    def _create_step_controls(self, step, label):
        """Create controls for a pipeline step"""
        layout = QHBoxLayout()
        
        # Main run button
        run_btn = QPushButton(f'▶  {label}')
        run_btn.clicked.connect(lambda: self.run_pipeline_step(step))
        run_btn.setEnabled(False)
        apply_button_style(run_btn, 'pipeline')
        if sys.platform == 'darwin':
            from PyQt5.QtGui import QFont
            run_btn.setFont(QFont(".AppleSystemUIFont", 13))
        layout.addWidget(run_btn, stretch=3)
        
        # Load button
        load_btn = QPushButton('📂')
        load_btn.setToolTip(f'Load Existing Step {step} Data')
        load_btn.clicked.connect(lambda: self.load_existing_step(step))
        load_btn.setMaximumWidth(40)
        load_btn.setStyleSheet('''
            QPushButton {
                background-color: #2A2A2A;
                color: white;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #00AAFF;
                font-family: ".AppleSystemUIFont", "Apple Color Emoji";
            }
            QPushButton:hover {
                background-color: #3A3A3A;
            }
        ''')
        if sys.platform == 'darwin':
            load_btn.setFont(QFont(".AppleSystemUIFont", 13))
        layout.addWidget(load_btn)
        
        # Inspect button
        inspect_btn = QPushButton('👁️')
        inspect_btn.setToolTip(f'Inspect Step {step} Results')
        inspect_btn.clicked.connect(lambda: self.inspect_results(step))
        inspect_btn.setEnabled(False)
        inspect_btn.setMaximumWidth(40)
        inspect_btn.setStyleSheet('''
            QPushButton {
                background-color: #2A2A2A;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-family: ".AppleSystemUIFont", "Apple Color Emoji";
            }
            QPushButton:hover:enabled {
                background-color: #3A3A3A;
            }
            QPushButton:disabled {
                background-color: #1A1A1A;
                color: #555;
            }
        ''')
        if sys.platform == 'darwin':
            inspect_btn.setFont(QFont(".AppleSystemUIFont", 13))
        layout.addWidget(inspect_btn)
        
        # Store references
        step_key = str(step).replace('.', '_')
        setattr(self, f'step{step_key}_btn', run_btn)
        setattr(self, f'step{step_key}_load_btn', load_btn)
        setattr(self, f'step{step_key}_inspect_btn', inspect_btn)
        
        return layout
    
    def _create_run_all_button(self):
        """Create run all button"""
        self.run_all_btn = QPushButton('▶▶  Run Full Pipeline (Steps 1-3)')
        self.run_all_btn.clicked.connect(self.run_full_pipeline)
        self.run_all_btn.setEnabled(False)
        apply_button_style(self.run_all_btn, 'success')
        if sys.platform == 'darwin':
            from PyQt5.QtGui import QFont
            self.run_all_btn.setFont(QFont(".AppleSystemUIFont", 13))
        return self.run_all_btn
    
    def _create_info_label(self):
        """Create session info label with scroll area"""
        # Create the actual label
        self.info_label = QLabel('No sessions loaded')
        self.info_label.setStyleSheet('''
            padding: 10px;
            background-color: #1E1E1E;
            border: none;
            font-family: monospace;
            font-size: 11px;
            color: white;
        ''')
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.info_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)  # Set minimum height
        scroll_area.setStyleSheet('''
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background-color: #1E1E1E;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #00AAFF;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00DDFF;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        ''')
        
        return scroll_area
    
    def _update_aggregate_stats(self, tracking_map=None):
        """Update info label with aggregate statistics for all loaded sessions"""
        if not self.sessions:
            self.info_label.setText('No sessions loaded')
            return
        
        # Compute aggregate statistics
        total_sessions = len(self.sessions)
        total_rois = 0
        total_deleted = 0
        all_subjects = set()
        all_x = []
        all_y = []
        
        for session_key, data in self.sessions.items():
            n_rois = len(data['centroids_x'])
            n_deleted = len(self.deleted_indices.get(session_key, set()))
            
            total_rois += n_rois
            total_deleted += n_deleted
            all_subjects.add(data.get('subject_id', 'Unknown'))
            
            all_x.extend(data['centroids_x'])
            all_y.extend(data['centroids_y'])
        
        active_rois = total_rois - total_deleted
        
        # Build info string
        info = f"📊 GROUP STATISTICS\n"
        info += f"{'='*30}\n"
        info += f"🗂️  Total Sessions: {total_sessions}\n"
        info += f"👥 Unique Subjects: {len(all_subjects)}\n"
        info += f"\n"
        info += f"🎯 Total ROIs: {total_rois:,}\n"
        info += f"✅ Active ROIs: {active_rois:,}\n"
        if total_deleted > 0:
            info += f"🗑️  Deleted: {total_deleted}\n"
        info += f"\n"
        if all_x and all_y:
            info += f"📍 X Range: [{min(all_x):.1f}, {max(all_x):.1f}]\n"
            info += f"📍 Y Range: [{min(all_y):.1f}, {max(all_y):.1f}]\n"
        
        # Add current session info
        if self.current_session:
            info += f"\n{'─'*30}\n"
            info += f"📌 CURRENT SESSION\n"
            info += f"{'─'*30}\n"
            
            data = self.sessions[self.current_session]
            x, y = data['centroids_x'], data['centroids_y']
            deleted = self.deleted_indices.get(self.current_session, set())
            n_rois = len(x) - len(deleted)
            n_selected = len(self.selected_indices)
            
            info += f"🔖 {self.current_session}\n"
            info += f"👤 Subject: {data.get('subject_id', 'Unknown')}\n"
            info += f"🎯 ROIs: {n_rois}\n"
            
            if tracking_map:
                n_tracked = len(tracking_map)
                n_untracked = n_rois - n_tracked
                tracking_pct = (n_tracked / n_rois * 100) if n_rois > 0 else 0
                
                info += f"\n🌟 TRACKING (Step 3)\n"
                info += f"✅ Tracked: {n_tracked} ({tracking_pct:.1f}%)\n"
                info += f"❌ Untracked: {n_untracked}\n"
                
                # Show track length distribution
                track_lengths = [t['track_length'] for t in tracking_map.values()]
                if track_lengths:
                    info += f"📊 Avg track length: {np.mean(track_lengths):.1f} sessions\n"
                    info += f"📊 Max track length: {max(track_lengths)} sessions\n"
            
            if len(deleted) > 0:
                info += f"🗑️  Deleted: {len(deleted)}\n"
            if n_selected > 0:
                info += f"✨ Selected: {n_selected}\n"
        
        self.info_label.setText(info)

    def create_right_panel(self):
        """Create right panel with plot and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(self.create_display_controls())
        layout.addWidget(self._create_plot_widget())
        
        return panel
        
    def create_display_controls(self):
        """Create display controls"""
        group = QGroupBox('Display Options')
        group.setStyleSheet('''
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        
        layout = QHBoxLayout()
        
        self.grid_check = QCheckBox('Show Grid')
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self.toggle_grid)
        self.grid_check.setStyleSheet('font-size: 12px; padding: 5px; color: white;')
        layout.addWidget(self.grid_check)
        
        layout.addStretch()
        
        self.mode_label = QLabel('Mode: View')
        self.mode_label.setStyleSheet('''
            font-size: 12px; padding: 8px 15px; 
            background-color: #2A2A2A;
            border: 1px solid #444;
            border-radius: 4px;
            color: white;
        ''')
        layout.addWidget(self.mode_label)
        
        group.setLayout(layout)
        return group
    
    def _create_plot_widget(self):
        """Create and configure plot widget"""
        self.plot_widget = InteractivePlot()
        self.plot_widget.setBackground('#0D0D0D')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', 'X Coordinate', color='white', size='12pt')
        self.plot_widget.setLabel('left', 'Y Coordinate', color='white', size='12pt')
        self.plot_widget.setAspectLocked(True)
        pg.setConfigOptions(antialias=True)
        
        self.plot_widget.pointClicked.connect(self.on_point_clicked)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        
        self._setup_empty_plot()
        return self.plot_widget
    
    def _setup_empty_plot(self):
        """Setup initial empty plot"""
        self.plot_widget.clear()
        text = pg.TextItem('← Load a folder to begin', color='white', anchor=(0.5, 0.5))
        text.setPos(0, 0)
        self.plot_widget.addItem(text)
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet('''
            QMenuBar {
                background-color: #1E1E1E;
                color: white;
                padding: 5px;
            }
            QMenuBar::item {
                background-color: #1E1E1E;
                color: white;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #00AAFF;
            }
            QMenu {
                background-color: #2A2A2A;
                color: white;
                border: 1px solid #00AAFF;
            }
            QMenu::item {
                padding: 8px 30px;
                color: white;
            }
            QMenu::item:selected {
                background-color: #00AAFF;
            }
        ''')
        
        # File menu
        file_menu = menubar.addMenu('📁 File')
        load_action = QAction('🗂️  Load Folder', self)
        load_action.triggered.connect(self.load_folder)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        exit_action = QAction('❌ Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('🎨 Edit')
        color_menu = edit_menu.addMenu('🎨 Colors')
        
        default_color_action = QAction('Default Point Color', self)
        default_color_action.triggered.connect(self.change_default_color)
        color_menu.addAction(default_color_action)
        
        edge_color_action = QAction('Edge Color', self)
        edge_color_action.triggered.connect(self.change_edge_color)
        color_menu.addAction(edge_color_action)
        
        color_menu.addSeparator()
        reset_colors_action = QAction('🔄 Reset All Colors', self)
        reset_colors_action.triggered.connect(self.reset_all_colors)
        color_menu.addAction(reset_colors_action)
        
        # Selection menu
        selection_menu = menubar.addMenu('✨ Selection')
        
        click_select_action = QAction('👆 Click Select Mode', self)
        click_select_action.setCheckable(True)
        click_select_action.triggered.connect(self.toggle_click_select)
        selection_menu.addAction(click_select_action)
        self.click_select_action = click_select_action
        
        region_select_action = QAction('🔲 Region Select Mode', self)
        region_select_action.setCheckable(True)
        region_select_action.triggered.connect(self.toggle_region_select)
        selection_menu.addAction(region_select_action)
        self.region_select_action = region_select_action
        
        selection_menu.addSeparator()
        
        color_selected_action = QAction('🖌️ Color Selected Points', self)
        color_selected_action.triggered.connect(self.color_selected_points)
        selection_menu.addAction(color_selected_action)
        
        delete_selected_action = QAction('🗑️ Delete Selected Points', self)
        delete_selected_action.triggered.connect(self.delete_selected_points)
        selection_menu.addAction(delete_selected_action)
        
        selection_menu.addSeparator()
        clear_selection_action = QAction('⭕ Clear Selection', self)
        clear_selection_action.triggered.connect(self.clear_selection)
        selection_menu.addAction(clear_selection_action)

        # View menu
        view_menu = menubar.addMenu('👁️ View')

        self.show_tracking_action = QAction('🌟 Show Step 3 Tracking', self)
        self.show_tracking_action.setCheckable(True)
        self.show_tracking_action.setChecked(True)  # On by default
        self.show_tracking_action.triggered.connect(self.toggle_tracking_view)
        view_menu.addAction(self.show_tracking_action)

        view_menu.addSeparator()

        legend_action = QAction('📋 Show Color Legend', self)
        legend_action.triggered.connect(self.show_tracking_legend)
        view_menu.addAction(legend_action)
    
    def toggle_tracking_view(self):
        """Toggle between tracking colors and default colors"""
        if not self.current_session:
            return
        
        if self.show_tracking_action.isChecked():
            # Apply tracking colors
            self.display_session()  # Will check for tracking and apply
        else:
            # Revert to default colors
            n_points = len(self.point_colors[self.current_session])
            self.point_colors[self.current_session] = [self.default_color] * n_points
            self.display_session()  # Will skip tracking colors

    def show_tracking_legend(self):
        """Show legend explaining tracking colors"""
        legend_text = (
            "🌟 STEP 3 TRACKING COLOR LEGEND\n\n"
            "🟢 Green: Tracked in 80%+ sessions\n"
            "🟡 Yellow: Tracked in 50-80% sessions\n"
            "🟠 Orange: Tracked in <50% sessions\n"
            "🔴 Red: Not tracked (unmatched)\n\n"
            "Neurons are colored based on how many\n"
            "sessions they were successfully tracked across."
        )
        QMessageBox.information(self, 'Tracking Color Legend', legend_text)

    def closeEvent(self, event):
        """Handle window close event with confirmation dialog"""
        reply = QMessageBox.question(
            self, 
            'Confirm Exit',
            'Are you sure you want to exit Stars2Cells Viewer?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        
    def load_folder(self):
        """Load sessions from folder using utility function"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if not folder:
            return
        
        folder = Path(folder)
        self.data_folder = folder
        
        # Initialize or update config
        if self.config is None:
            self.config = PipelineConfig(
                input_dir=str(folder),
                output_dir=str(folder.parent / "Stars2Cells_Results"),
                verbose=True,
                skip_existing=True,
                threshold=0.15,
            )
        else:
            self.config.input_dir = str(folder)
            self.config.input_path = Path(folder)
            if self.config.threshold is None:
                self.config.threshold = 0.15
        
        # Clear previous data
        self._clear_session_data()
        
        # Load sessions using utility function
        loaded_data = load_sessions_from_folder(folder)
        
        if 'error' in loaded_data:
            QMessageBox.warning(self, 'No Files', loaded_data['error'])
            return
        
        # Store loaded data
        self.sessions = loaded_data['sessions']
        self.session_list = loaded_data['session_list']
        self.session_file_paths = loaded_data['file_paths']
        
        # Initialize UI data
        for meta in loaded_data['metadata']:
            session_key = meta['session_key']
            self.point_colors[session_key] = [self.default_color] * meta['num_rois']
            self.deleted_indices[session_key] = set()
            
            # Add to list
            item_text = (f"{session_key}\n  └─ {meta['subject_id']} | "
                        f"Session {meta['session_id']} | {meta['num_rois']} ROIs")
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, session_key)
            self.file_list.addItem(item)
        
        if len(self.sessions) > 0:
            self.step1_btn.setEnabled(True)
            self.step1_5_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)
            self._auto_detect_and_enable_steps(folder)
            
            # Update aggregate statistics
            self._update_aggregate_stats()
            
            # Select first session
            self.file_list.setCurrentRow(0)
            self.on_session_selected(self.file_list.item(0))
    
    def _clear_session_data(self):
        """Clear all session data"""
        self.sessions.clear()
        self.session_list.clear()
        self.session_file_paths.clear()
        self.point_colors.clear()
        self.deleted_indices.clear()
        self.file_list.clear()
        self.current_session = None
        self.selected_indices.clear()
        self.info_label.setText('No sessions loaded')
        
    def on_session_selected(self, item):
        """Handle session selection"""
        session_key = item.data(Qt.UserRole)
        print(f"\n[SESSION] Selected: {session_key}")
        print(f"[SESSION] Type: {type(session_key)}")
        
        self.current_session = session_key
        self.selected_indices.clear()
        
        if hasattr(self, 'region_select_action'):
            self.region_select_action.setChecked(False)
            self.region_select_mode = False
        
        if hasattr(self, 'click_select_action'):
            self.click_select_action.setChecked(False)
            self.click_select_mode = False
            self.plot_widget.click_select_enabled = False
        
        self.update_mode_label()
        self.display_session()

    def _load_step3_tracking_for_session(self, session_key):
        """Load Step 3 tracking data for current session if available"""
        print(f"\n[TRACKING] Attempting to load tracking for: {session_key}")
        
        if not self.config or not self.config.output_dir:
            print(f"[TRACKING] No config or output_dir")
            return None
        
        # Extract animal_id from session_key
        # Assuming session_key format: "animalID_sessionID"
        parts = session_key.split('_')
        if len(parts) < 2:
            print(f"[TRACKING] Invalid session_key format: {session_key}")
            return None
        
        animal_id = parts[0]
        print(f"[TRACKING] Animal ID: {animal_id}")
        
        # Try to load consolidated tracking
        step3_dir = Path(self.config.output_dir) / "step_3_results"
        tracking_file = step3_dir / f"{animal_id}_consolidated_tracking.npz"
        
        print(f"[TRACKING] Looking for: {tracking_file}")
        print(f"[TRACKING] File exists: {tracking_file.exists()}")
        
        if not tracking_file.exists():
            return None
        
        try:
            tracking_data = np.load(tracking_file, allow_pickle=True)
            
            # Find session index
            sessions = tracking_data['sessions']
            print(f"[TRACKING] Sessions in tracking file: {list(sessions)}")
            
            if session_key not in sessions:
                print(f"[TRACKING] Session {session_key} not in tracking data")
                return None
            
            session_idx = list(sessions).index(session_key)
            print(f"[TRACKING] Session index: {session_idx}")
            
            # Get neuron tracks
            neuron_tracks = tracking_data['neuron_tracks'].item()
            track_lengths = tracking_data['track_lengths']
            n_sessions = int(tracking_data['n_sessions'])
            
            print(f"[TRACKING] Total tracked neurons: {len(neuron_tracks)}")
            print(f"[TRACKING] Number of sessions: {n_sessions}")
            
            # Build mapping: local_neuron_id -> (global_id, track_length)
            tracking_map = {}
            for global_id, track in neuron_tracks.items():
                if session_idx in track:
                    local_idx = track[session_idx]
                    track_length = len(track)
                    tracking_map[local_idx] = {
                        'global_id': global_id,
                        'track_length': track_length,
                        'max_possible': n_sessions
                    }
            
            print(f"[TRACKING] Mapped {len(tracking_map)} neurons for this session")
            
            return tracking_map
            
        except Exception as e:
            print(f"[TRACKING] ERROR loading tracking data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_tracking_colors(self, session_key, tracking_map):
        """Apply colors based on tracking status"""
        if tracking_map is None:
            return
        
        n_rois = len(self.point_colors[session_key])
        
        for i in range(n_rois):
            if i in tracking_map:
                track_info = tracking_map[i]
                track_length = track_info['track_length']
                max_sessions = track_info['max_possible']
                
                # Color based on tracking quality
                if track_length >= max_sessions * 0.8:  # Tracked in 80%+ sessions
                    self.point_colors[session_key][i] = '#00FF00'  # Bright green
                elif track_length >= max_sessions * 0.5:  # Tracked in 50-80% sessions
                    self.point_colors[session_key][i] = '#FFFF00'  # Yellow
                else:  # Tracked in <50% sessions
                    self.point_colors[session_key][i] = '#FFA500'  # Orange
            else:
                # Not tracked at all
                self.point_colors[session_key][i] = '#FF0000'  # Red

    def _update_display_with_tracking(self):
        """Update display with Step 3 tracking colors if available"""
        if not self.current_session:
            return
        
        tracking_map = self._load_step3_tracking_for_session(self.current_session)
        
        if tracking_map:
            self._apply_tracking_colors(self.current_session, tracking_map)
            print(f"✓ Applied Step 3 tracking colors to {self.current_session}")
            print(f"  - Tracked neurons: {len(tracking_map)}")
            print(f"  - Untracked neurons: {len(self.point_colors[self.current_session]) - len(tracking_map)}")
            
    def display_session(self):
        """Display current session"""
        if not self.current_session:
            self._update_aggregate_stats()
            return
        
        print(f"\n[DISPLAY] Showing session: {self.current_session}")
        print(f"[DISPLAY] Tracking toggle checked: {self.show_tracking_action.isChecked()}")
        
        data = self.sessions[self.current_session]
        x, y = data['centroids_x'], data['centroids_y']
        
        # CHECK FOR STEP 3 TRACKING DATA AND APPLY COLORS
        tracking_map = None
        if hasattr(self, 'show_tracking_action') and self.show_tracking_action.isChecked():
            tracking_map = self._load_step3_tracking_for_session(self.current_session)
            if tracking_map:
                self._apply_tracking_colors(self.current_session, tracking_map)
                print(f"[DISPLAY] ✓ Applied tracking colors: {len(tracking_map)} tracked")
            else:
                print(f"[DISPLAY] ✗ No tracking data found")
        
        colors = self.point_colors[self.current_session]
        deleted = self.deleted_indices.get(self.current_session, set())
        
        print(f"[DISPLAY] Clearing plot...")  
        self.plot_widget.clear()
        valid_indices = [i for i in range(len(x)) if i not in deleted]
        
        print(f"[DISPLAY] Valid indices: {len(valid_indices)}")  
        
        if len(valid_indices) == 0:
            self._update_aggregate_stats()
            return
        
        pos_array = np.column_stack([x, y])
        self.plot_widget.point_positions = pos_array
        self.plot_widget.deleted_indices = deleted
        
        print(f"[DISPLAY] Plotting {len(valid_indices)} points...") 
        
        for i in valid_indices:
            xi, yi = x[i], y[i]
            color = colors[i]
            
            if i in self.selected_indices:
                brush = pg.mkBrush(color)
                pen = pg.mkPen(color=self.selection_color, width=3)
                size = self.current_point_size * 1.5
            else:
                brush = pg.mkBrush(color)
                pen = pg.mkPen(color=self.default_edge_color, width=1.5)
                size = self.current_point_size
            
            scatter = pg.ScatterPlotItem([xi], [yi], size=size, brush=brush, 
                                        pen=pen, symbol='o')
            scatter.setData([xi], [yi], data=[i])
            self.plot_widget.addItem(scatter)
        
        print(f"[DISPLAY] Plot updated!") 
        
        self._update_plot_title(data, len(x), len(deleted))
        
        # PASS TRACKING MAP TO STATS
        self._update_aggregate_stats(tracking_map)
        
        print(f"[DISPLAY] Display complete!\n")  

    def _update_plot_title(self, data, total_rois, n_deleted):
        """Update plot title only"""
        subject = data.get('subject_id', 'Unknown')
        session_id = data.get('session_id', 'Unknown')
        n_rois = total_rois - n_deleted
        n_selected = len(self.selected_indices)
        
        title = f"{self.current_session} | Subject: {subject} | Session: {session_id} | ROIs: {n_rois}"
        if n_deleted > 0:
            title += f" | Deleted: {n_deleted}"
        if n_selected > 0:
            title += f" | Selected: {n_selected}"
        
        self.plot_widget.setTitle(title, color='white', size='14pt')
    
    def toggle_grid(self):
        """Toggle grid visibility"""
        show_grid = self.grid_check.isChecked()
        self.plot_widget.showGrid(x=show_grid, y=show_grid, alpha=0.2)
        
    def update_mode_label(self):
        """Update mode label"""
        if self.click_select_mode:
            self.mode_label.setText('Mode: Click Select')
            self.mode_label.setStyleSheet('''
                font-size: 12px; padding: 8px 15px; 
                background-color: #00AAFF; border: 1px solid #00DDFF;
                border-radius: 4px; color: white; font-weight: bold;
            ''')
        elif self.region_select_mode:
            self.mode_label.setText('Mode: Region Select')
            self.mode_label.setStyleSheet('''
                font-size: 12px; padding: 8px 15px; 
                background-color: #00AAFF; border: 1px solid #00DDFF;
                border-radius: 4px; color: white; font-weight: bold;
            ''')
        else:
            self.mode_label.setText('Mode: View')
            self.mode_label.setStyleSheet('''
                font-size: 12px; padding: 8px 15px; 
                background-color: #2A2A2A; border: 1px solid #444;
                border-radius: 4px; color: white;
            ''')
    
    def toggle_click_select(self):
        """Toggle click selection mode"""
        self.click_select_mode = self.click_select_action.isChecked()
        self.plot_widget.click_select_enabled = self.click_select_mode
        
        if self.click_select_mode and self.region_select_action.isChecked():
            self.region_select_action.setChecked(False)
            self.region_select_mode = False
        
        self.update_mode_label()
    
    def toggle_region_select(self):
        """Toggle region selection mode"""
        self.region_select_mode = self.region_select_action.isChecked()
        
        if self.region_select_mode and self.click_select_action.isChecked():
            self.click_select_action.setChecked(False)
            self.click_select_mode = False
            self.plot_widget.click_select_enabled = False
        
        self.update_mode_label()
        
        if not self.region_select_mode:
            self.region_start = None
            if self.region_rect is not None:
                self.plot_widget.removeItem(self.region_rect)
                self.region_rect = None
    
    def on_mouse_clicked(self, event):
        """Handle mouse click for region selection"""
        if not self.region_select_mode or not self.current_session:
            return
        
        pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        
        if event.button() == Qt.LeftButton:
            if event.double():
                return
            
            if self.region_start is None:
                self.region_start = (pos.x(), pos.y())
                self.region_rect = pg.RectROI([pos.x(), pos.y()], [0, 0],
                                              pen=pg.mkPen(color='cyan', width=2))
                self.plot_widget.addItem(self.region_rect)
            else:
                x1, y1 = self.region_start
                x2, y2 = pos.x(), pos.y()
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                
                data = self.sessions[self.current_session]
                x, y = data['centroids_x'], data['centroids_y']
                deleted = self.deleted_indices.get(self.current_session, set())
                
                newly_selected = {i for i, (xi, yi) in enumerate(zip(x, y))
                                if i not in deleted and x_min <= xi <= x_max and y_min <= yi <= y_max}
                
                self.selected_indices.update(newly_selected)
                print(f"Selected {len(newly_selected)} new points (total: {len(self.selected_indices)})")
                
                self.region_start = None
                if self.region_rect is not None:
                    self.plot_widget.removeItem(self.region_rect)
                    self.region_rect = None
                
                self.display_session()
    
    def on_point_clicked(self, index):
        """Handle point click"""
        if not self.current_session:
            return
        
        data = self.sessions[self.current_session]
        x, y = data['centroids_x'][index], data['centroids_y'][index]
        
        if self.click_select_mode:
            if index in self.selected_indices:
                self.selected_indices.remove(index)
            else:
                self.selected_indices.add(index)
            self.display_session()
        else:
            info_msg = (f"🎯 Point #{index}\n\n"
                       f"📍 Position: ({x:.2f}, {y:.2f})\n"
                       f"📊 Session: {self.current_session}\n"
                       f"👤 Subject: {data.get('subject_id', 'Unknown')}\n"
                       f"🔬 Session ID: {data.get('session_id', 'Unknown')}")
            QMessageBox.information(self, 'Point Info', info_msg)
    
    def clear_selection(self):
        """Clear selection"""
        self.selected_indices.clear()
        if self.current_session:
            self.display_session()  # This will update aggregate stats
    
    def delete_selected_points(self):
        """Delete selected points"""
        if not self.selected_indices or not self.current_session:
            QMessageBox.information(self, 'No Selection', 'Please select points first.')
            return
        
        reply = QMessageBox.question(self, 'Confirm Delete',
                                    f'Delete {len(self.selected_indices)} selected points?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.deleted_indices[self.current_session].update(self.selected_indices)
            self.selected_indices.clear()
            self.display_session()  # This will call _update_aggregate_stats()
    
    def color_selected_points(self):
        """Color selected points"""
        if not self.selected_indices or not self.current_session:
            QMessageBox.information(self, 'No Selection', 'Please select points first.')
            return
        
        color = QColorDialog.getColor(QColor(self.default_color), self, 'Select Color')
        if color.isValid():
            for idx in self.selected_indices:
                self.point_colors[self.current_session][idx] = color.name()
            self.display_session()
    
    def change_default_color(self):
        """Change default color"""
        color = QColorDialog.getColor(QColor(self.default_color), self, 'Select Default Color')
        if color.isValid():
            self.default_color = color.name()
            if self.current_session:
                num_points = len(self.point_colors[self.current_session])
                self.point_colors[self.current_session] = [self.default_color] * num_points
                self.display_session()
    
    def change_edge_color(self):
        """Change edge color"""
        color = QColorDialog.getColor(QColor(self.default_edge_color), self, 'Select Edge Color')
        if color.isValid():
            self.default_edge_color = color.name()
            if self.current_session:
                self.display_session()
    
    def reset_all_colors(self):
        """Reset all colors"""
        if not self.current_session:
            return
        num_points = len(self.point_colors[self.current_session])
        self.point_colors[self.current_session] = [self.default_color] * num_points
        self.display_session()
    
    # ===== PIPELINE METHODS =====
    
    def configure_pipeline(self):
        """Configure pipeline"""
        if self.config is None:
            self.config = PipelineConfig(threshold=0.15)
        elif self.config.threshold is None:
            self.config.threshold = 0.15
        
        dialog = PipelineConfigDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            self.config = dialog.get_config()
            
            if self.config.output_dir:
                from pathlib import Path
                self._auto_detect_and_enable_steps(Path(self.config.output_dir))
            
            QMessageBox.information(self, 'Configuration Updated', 
                                'Pipeline configuration has been updated.')
        
    def run_pipeline_step(self, step):
        """Run a specific pipeline step"""
        if self.config is None:
            QMessageBox.warning(self, 'No Configuration', 
                            'Please configure the pipeline first.')
            return
        
        if step == 1 and len(self.session_file_paths) == 0:
            QMessageBox.warning(self, 'No Sessions', 'Please load sessions first.')
            return
        
        if step == 2.5 and not self._validate_step2_requirements():
            return
        
        # Show confirmation dialog with editable parameters
        if not confirm_step_execution(step, self.config, self):
            return
        
        # Determine maximum for progress bar
        maximum = 0
        if step == 1:
            maximum = len(self.session_file_paths)
        elif step == 1.5:
            maximum = count_items_for_step(1.5, self.config.output_dir)
        elif step == 2:
            step2_dir = Path(self.config.output_dir) / "step_2_results"
            step1_dir = Path(self.config.output_dir) / "step_1_results"
            if step1_dir.exists():
                n_sessions = len(list(step1_dir.glob("*_centroids_quads.npz")))
                maximum = n_sessions - 1  # consecutive pairs ≈ sessions - 1 per animal
            else:
                maximum = 0
        elif step == 2.5:
            step2_dir = Path(self.config.output_dir) / "step_2_results"
            if step2_dir.exists():
                maximum = len(list(step2_dir.glob("*_matches_light.npz")))
        elif step == 3:
            # Count animals for Step 3
            step2_5_dir = Path(self.config.output_dir) / "step_2_5_results"
            if step2_5_dir.exists():
                animals = []
                for npz_file in sorted(step2_5_dir.glob("*_filtered_matches.npz")):
                    animal_id = npz_file.stem.replace("_filtered_matches", "")
                    animals.append(animal_id)
                maximum = len(set(animals))
        
        # Create progress dialog
        progress, log_window, log_text = create_progress_dialog_with_log(
            self, f'Step {step} Execution', step, maximum
        )
        import time
        self._current_progress = progress
        self._current_log_text = log_text
        self._step_wall_start = time.time()
        progress.starttime = time.time()

        # For Step 1: convert dict to list of file paths
        loaded_session_list = None
        if step == 1 and self.session_file_paths:
            loaded_session_list = list(self.session_file_paths.values())  # Extract the file paths
            print(f"[DEBUG] Passing {len(loaded_session_list)} file paths to Step 1")

        # Create worker
        self.pipeline_worker = PipelineWorker(
            step, self.config,
            loaded_sessions=loaded_session_list
        )
        
        # Create callbacks with step-specific post-processing
        on_progress, on_finished, on_error = create_pipeline_callbacks(self, step, self.config)
        
        # WRAP on_finished TO ADD STEP 3 DISPLAY REFRESH
        original_on_finished = on_finished
        
        def on_finished_with_refresh(result):
            """Wrapper to refresh display after Step 3"""
            # Call original callback
            original_on_finished(result)
            
            # IF STEP 3 COMPLETED, REFRESH DISPLAY WITH TRACKING COLORS
            if step == 3 and self.current_session:
                print("\n✓ Step 3 complete - refreshing display with tracking colors...")
                self.display_session()  # Will auto-detect and apply tracking colors
        
        self.pipeline_worker.progress.connect(on_progress)
        
        # Connect appropriate progress signal based on step
        if step == 1:
            self.pipeline_worker.session_progress.connect(self._on_session_progress)
        elif step == 1.5:                                                              # ADD THIS
            self.pipeline_worker.animal_progress.connect(self._on_pair_progress)   
        elif step == 2:
            self.pipeline_worker.animal_progress.connect(self._on_animal_progress)
        elif step == 2.5 or step == 3:
            self.pipeline_worker.animal_progress.connect(self._on_animal_progress)
        
        self.pipeline_worker.finished.connect(on_finished_with_refresh) 
        self.pipeline_worker.error.connect(on_error)
        print(f"[RUN_STEP] About to start worker for step={step}")
        print(f"[RUN_STEP] Worker signals connected: progress={self.pipeline_worker.progress}, animal_progress={self.pipeline_worker.animal_progress}")
        self.pipeline_worker.start()
        
    def _fmt_elapsed(self, seconds: float) -> str:
        """Format elapsed seconds as Xh Xm Xs."""
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _on_session_progress(self, current, total, sessiontime):
        """Step 1 — one worker per session, wall-clock ETA."""
        if not hasattr(self, '_current_progress') or self._current_progress is None:
            return

        progress = self._current_progress
        progress.setValue(current)

        wall_elapsed = time.time() - getattr(self, '_step_wall_start', time.time())
        if current > 0:
            avg_wall = wall_elapsed / current
            eta_s = avg_wall * (total - current)
            mins, secs = divmod(int(eta_s), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                eta_str = f"{hours}h {mins}m"
            elif mins:
                eta_str = f"{mins}m {secs}s"
            else:
                eta_str = f"{secs}s"
            rate_str = f"{avg_wall:.1f}s/session"
        else:
            eta_str = "calculating…"
            rate_str = "—"

        progress.setLabelText(
            f"Session {current}/{total}\n"
            f"Rate: {rate_str}    ETA: {eta_str}\n"
            f"Elapsed: {self._fmt_elapsed(wall_elapsed)}"
        )

    # def _on_pair_progress(self, current, total, animal_time):
    #     """
    #     Step 1.5 progress — throttled, never calls processEvents() which can
    #     cause re-entrant event handling while multiprocessing workers are active.
    #     """
    #     import time
    #     if not hasattr(self, '_current_progress') or self._current_progress is None:
    #         return

    #     progress = self._current_progress

    #     if progress.maximum() == 0 and total > 0:
    #         progress.setMaximum(total)

    #     progress.setValue(current)

    #     wall_elapsed = time.time() - getattr(self, '_step_wall_start', time.time())

    #     if current > 0:
    #         avg_wall = wall_elapsed / current
    #         eta_s = avg_wall * (total - current)
    #         mins, secs = divmod(int(eta_s), 60)
    #         hours, mins = divmod(mins, 60)
    #         if hours:
    #             eta_str = f"{hours}h {mins}m"
    #         elif mins:
    #             eta_str = f"{mins}m {secs}s"
    #         else:
    #             eta_str = f"{secs}s"
    #         rate_str = f"{avg_wall:.1f}s/animal"
    #     else:
    #         eta_str = "calculating…"
    #         rate_str = "—"

    #     progress.setLabelText(
    #         f"Animals completed: {current}/{total}\n"
    #         f"Rate: {rate_str}    ETA: {eta_str}\n"
    #         f"Elapsed: {self._fmt_elapsed(wall_elapsed)}"
    #     )
    #     # NOTE: No QApplication.processEvents() here — Qt handles repaints naturally
    #     # via the event loop. Calling processEvents() during multiprocessing callbacks
    #     # causes re-entrant event handling which freezes or crashes the GUI.

    def _on_pair_progress(self, current, total, animal_time):
        import time
        print(f"[ON_PAIR_PROGRESS] Qt received signal: current={current}/{total}")
        if not hasattr(self, '_current_progress') or self._current_progress is None:
            print(f"[ON_PAIR_PROGRESS] WARNING: _current_progress is None — returning early")
            return

        progress = self._current_progress

        if progress.maximum() == 0 and total > 0:
            print(f"[ON_PAIR_PROGRESS] Setting progress maximum to {total}")
            progress.setMaximum(total)

        progress.setValue(current)
        print(f"[ON_PAIR_PROGRESS] progress.setValue({current}) done")

        wall_elapsed = time.time() - getattr(self, '_step_wall_start', time.time())

        if current > 0:
            avg_wall = wall_elapsed / current
            eta_s = avg_wall * (total - current)
            mins, secs = divmod(int(eta_s), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                eta_str = f"{hours}h {mins}m"
            elif mins:
                eta_str = f"{mins}m {secs}s"
            else:
                eta_str = f"{secs}s"
            rate_str = f"{avg_wall:.1f}s/animal"
        else:
            eta_str = "calculating…"
            rate_str = "—"

        progress.setLabelText(
            f"Animals completed: {current}/{total}\n"
            f"Rate: {rate_str}    ETA: {eta_str}\n"
            f"Elapsed: {self._fmt_elapsed(wall_elapsed)}"
        )
        print(f"[ON_PAIR_PROGRESS] setLabelText done")

    def _on_animal_progress(self, current, total, animaltime):
        """Steps 2.5 and 3 — sequential per-animal processing."""
        if not hasattr(self, '_current_progress') or self._current_progress is None:
            return

        progress = self._current_progress
        progress.setValue(current)

        wall_elapsed = time.time() - getattr(self, '_step_wall_start', time.time())

        if current > 0:
            avg_wall = wall_elapsed / current
            eta_s = avg_wall * (total - current)
            mins, secs = divmod(int(eta_s), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                eta_str = f"{hours}h {mins}m"
            elif mins:
                eta_str = f"{mins}m {secs}s"
            else:
                eta_str = f"{secs}s"
            rate_str = f"{avg_wall:.1f}s/animal"
        else:
            eta_str = "calculating…"
            rate_str = "—"

        progress.setLabelText(
            f"{current}/{total}\n"
            f"Rate: {rate_str}    ETA: {eta_str}\n"
            f"Elapsed: {self._fmt_elapsed(wall_elapsed)}"
        )

    def _validate_step2_requirements(self):
        """Validate that Step 2 results exist for Step 2.5"""        
        # Use step_info.py to get the correct directory
        step2_dir = get_step_output_dir(2, self.config.output_dir)
        
        if not step2_dir.exists():
            QMessageBox.warning(
                self, 'Step 2 Required',
                f'Step 2.5 requires Step 2 results.\n\n'
                f'Expected directory:\n{step2_dir}\n\n'
                f'Please either:\n  • Run Step 2 first, or\n  • Use the 📂 button'
            )
            return False
        
        # Use step_info.py to get the correct file pattern
        pattern = get_step_file_pattern(2)  # Returns '*_matches_light.npz'
        match_files = list(step2_dir.glob(pattern))
        
        if not match_files:
            QMessageBox.warning(
                self, 'No Step 2 Results',
                f'No Step 2 match files found in:\n{step2_dir}\n\n'
                f'Expected pattern: {pattern}\n\n'
                f'Please either:\n  • Run Step 2 first, or\n  • Use the 📂 button'
            )
            return False
        
        print(f"\n✓ Found {len(match_files)} Step 2 match files in {step2_dir}")
        return True

    def run_full_pipeline(self):
        """Run all pipeline steps"""
        reply = QMessageBox.question(
            self, 'Run Full Pipeline',
            'This will run all steps of the Stars2Cells pipeline:\n\n'
            'This may take some time. Continue?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.run_pipeline_step(1)
    
    def inspect_results(self, step):
        """Inspect results for a specific step"""
        if self.config is None:
            return
        open_results_inspector(step, self.config, self)
    
    def load_existing_step(self, step):
        """Load existing results for a specific step by browsing"""
        if self.config is None:
            self.config = PipelineConfig(
                input_dir=".", output_dir=".", threshold=0.15, verbose=True
            )
        
        step_names = {1: 'Quad Generation', 1.5: 'Threshold Generation', 2: 'Matching', 
                    2.5: 'RANSAC', 3: 'Pruning'}
        step_patterns = {1: '*_centroids_quads.npz', 1.5: '*_threshold_calibration', 2: '*_matches_light.npz',
                        2.5: '*_filtered_matches.npz', 3: '*_final_matches.npz'}
        
        folder = QFileDialog.getExistingDirectory(
            self, f"Select Folder with Step {step} Results ({step_names[step]})"
        )
        
        if not folder:
            return
        
        folder = Path(folder)
        result_files = list(folder.glob(step_patterns[step]))
        if not result_files:
            result_files = list(folder.glob(f'*/{step_patterns[step]}'))
        
        if not result_files:
            QMessageBox.warning(
                self, 'No Results Found',
                f'No Step {step} result files found in:\n{folder}\n\n'
                f'Expected file pattern: {step_patterns[step]}'
            )
            return
        
        # Update config using utility function
        self.config = update_config_from_step_path(self.config, step, folder)
        
        # Enable inspect button
        step_key = str(step).replace('.', '_')
        getattr(self, f'step{step_key}_inspect_btn').setEnabled(True)
        if step == 1:
            self.step2_btn.setEnabled(True)
        elif step == 2:
            self.step2_5_btn.setEnabled(True)
        elif step == 2.5:
            self.step3_btn.setEnabled(True)
        
        QMessageBox.information(
            self, 'Results Loaded',
            f'✓ Step {step} ({step_names[step]}) results loaded!\n\n'
            f'Folder: {folder}\n'
            f'Files found: {len(result_files)}'
        )
        
        print(f"\n{'='*60}")
        print(f"LOADED EXISTING STEP {step} RESULTS")
        print(f"Folder: {folder}")
        print(f"Files found: {len(result_files)}")
        for f in result_files[:5]:
            print(f"  - {f.name}")
        if len(result_files) > 5:
            print(f"  ... and {len(result_files) - 5} more")
        print(f"{'='*60}\n")
        
        self._auto_detect_and_enable_steps(folder)
        
        # IF STEP 3 WAS LOADED, REFRESH DISPLAY WITH TRACKING COLORS
        if step == 3 and self.current_session:
            print("✓ Step 3 results loaded - refreshing display with tracking colors...")
            self.display_session()
            
    def _auto_detect_and_enable_steps(self, folder: Path):
        """Auto-detect and enable available pipeline steps"""
        print(f"\n{'='*60}")
        print(f"AUTO-DETECTING PIPELINE RESULTS")
        print(f"Searching in and around: {folder}")
        print(f"{'='*60}")
        
        results = detect_pipeline_results(folder)
        
        # Check if there is a found project
        if results['project_root']:
            print(f"✓ Project root: {results['project_root']}")
            # Update config to use this project root
            self.config.output_dir = str(results['project_root'])
            self.config.output_path = results['project_root']
        else:
            print(f"⚠️  No Stars2Cells project structure found")
            print(f"{'='*60}\n")
            return
        
        found_any = False
        
        step_map = {
            'step_1': (1, '1', 'Quad Generation'),
            'step_1_5': (1.5, '1_5', 'Calibration'),
            'step_2': (2, '2', 'Matching'),
            'step_2_5': (2.5, '2_5', 'RANSAC'),
            'step_3': (3, '3', 'Final Matching')
        }
        
        for result_key, (step_num, step_str, step_name) in step_map.items():
            if results[result_key]['found']:
                found_any = True
                path = results[result_key]['path']
                files = results[result_key]['files']
                print(f"✓ Step {step_num} ({step_name}): {len(files)} files in {path}")
                
                # Update config
                self.config = update_config_from_step_path(self.config, step_num, path)
                
                # Enable inspect button
                inspect_btn_name = f'step{step_str}_inspect_btn'
                if hasattr(self, inspect_btn_name):
                    btn = getattr(self, inspect_btn_name)
                    btn.setEnabled(True)
                    print(f"  ✓ Enabled {inspect_btn_name}")
                else:
                    print(f"  ✗ WARNING: {inspect_btn_name} not found!")
                
                # Enable next step's run button
                if step_num == 1:
                    self.step2_btn.setEnabled(True)
                elif step_num == 1.5:
                    self.step2_btn.setEnabled(True)
                elif step_num == 2:
                    self.step2_5_btn.setEnabled(True)
                elif step_num == 2.5:
                    self.step3_btn.setEnabled(True)
        
        if not found_any:
            print(f"⚠️  No pipeline results found")
        
        print(f"{'='*60}\n")               

def main():
    print("\n" + "="*70)
    print("STARS2CELLS UNIFIED VIEWER")
    print("="*70)
    print("Features:")
    print("  🗂️  Load and browse neuron data")
    print("  🎨  Visualize and edit sessions")
    print("  ⚙️  Configure pipeline parameters")
    print("  🌟  Run Stars2Cells pipeline (Steps 1-3)")
    print("  👁️  Inspect results at each step")
    print("="*70 + "\n")
    
    app = QApplication(sys.argv)
    # Fix emoji rendering on macOS (Windows handles it natively)
    if sys.platform == 'darwin':
        from PyQt5.QtGui import QFont
        app.setFont(QFont(".AppleSystemUIFont", 13))
    set_application_icon(app, icon_path="S2C_logo.png")
    splash = show_splash_screen(app, logo_path="S2C_logo.png", duration=2500)
    
    viewer = Stars2CellsViewer()
    if splash is not None:
        splash.finish(viewer)
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()