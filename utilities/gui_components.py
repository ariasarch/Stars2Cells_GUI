"""
GUI Components for Stars2Cells Viewer 
"""

import numpy as np
import time
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QFormLayout, 
                             QDialogButtonBox, QTextEdit, QMessageBox, QComboBox,
                             QPushButton, QProgressDialog, QLineEdit, QCheckBox,
                             QFileDialog, QApplication, QScrollArea, QWidget, QGroupBox) 
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import logging
import threading
import importlib
import sys

# Import centralized step info
from .step_info import *

from .config import PipelineConfig

class QTextEditLogger(logging.Handler):
    """Custom logging handler that emits log messages as Qt signals"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        
    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class InteractivePlot(pg.PlotWidget):
    """Custom plot widget with click selection support"""
    pointClicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = None
        self.click_select_enabled = False
        self.point_positions = None
        self.deleted_indices = set()
        
    def mousePressEvent(self, event):
        """Handle mouse press for point selection"""
        if self.point_positions is not None and event.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            x_click, y_click = pos.x(), pos.y()
            
            if len(self.point_positions) > 0:
                distances = np.sqrt((self.point_positions[:, 0] - x_click)**2 + 
                                   (self.point_positions[:, 1] - y_click)**2)
                
                for idx in self.deleted_indices:
                    if idx < len(distances):
                        distances[idx] = np.inf
                
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < 10:
                    self.pointClicked.emit(int(nearest_idx))
                    return
        
        super().mousePressEvent(event)

# ============================================================================
# UNIFIED PIPELINE WORKER 
# ============================================================================

class PipelineWorker(QThread):
    """Worker thread for running pipeline steps - completely step-agnostic"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    session_progress = pyqtSignal(int, int, float)  # For Step 1
    animal_progress = pyqtSignal(int, int, float)   # For other steps
    
    def __init__(self, step, config, loaded_sessions=None):
        super().__init__()
        self.step = step
        self.config = config
        self.loaded_sessions = loaded_sessions
        self._step_result = None
        self._step_error = None
        # Thread-safe message queue instead of direct signal emission from threads
        self._msg_queue = []
        self._msg_lock = threading.Lock()

    def run(self):
        """Run the pipeline step — guarantees finished or error is always emitted."""
        _signal_emitted = False
        print(f"\n[WORKER.run] START  step={self.step}  thread={self.currentThread()}")

        try:
            step_info = get_step_info(self.step)
            if not step_info:
                print(f"[WORKER.run] Unknown step {self.step} — emitting error")
                self.error.emit(f"Unknown step: {self.step}")
                _signal_emitted = True
                return

            print(f"[WORKER.run] step_info found: label={step_info.get('label')}")
            self.progress.emit(f"Starting {step_info['label']}...")

            gui_handler = self._setup_logging(step_info['logger_name'])
            print(f"[WORKER.run] Logging handler set up for logger: {step_info['logger_name']}")

            try:
                print(f"[WORKER.run] Calling _run_with_monitoring...")
                self._run_with_monitoring(step_info, gui_handler)
                print(f"[WORKER.run] _run_with_monitoring returned normally")
                _signal_emitted = True
            finally:
                self._cleanup_logging(step_info['logger_name'], gui_handler)
                print(f"[WORKER.run] Logging cleaned up")

        except BaseException as e:
            import traceback
            error_msg = (
                f"Error in Step {self.step}:\n"
                f"{type(e).__name__}: {str(e)}\n\n"
                f"{traceback.format_exc()}"
            )
            print(f"[WORKER.run] EXCEPTION caught: {type(e).__name__}: {e}")
            if not _signal_emitted:
                self.error.emit(error_msg)
                _signal_emitted = True

        finally:
            print(f"[WORKER.run] FINALLY block  _signal_emitted={_signal_emitted}")
            if not _signal_emitted:
                print(f"[WORKER.run] No signal emitted — emitting fallback error")
                self.error.emit(
                    f"Step {self.step} worker terminated unexpectedly "
                    f"without emitting a result. Check the console for details."
                )
            print(f"[WORKER.run] END\n")

    def _setup_logging(self, logger_name):
        """Setup logging handler for step - routes through thread-safe queue"""
        worker_ref = self  # Capture self for use in handler
        
        class GUIHandler(logging.Handler):
            def __init__(self):
                super().__init__()
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Queue the message instead of emitting signal directly from logging thread
                    with worker_ref._msg_lock:
                        worker_ref._msg_queue.append(msg)
                except:
                    pass
        
        gui_handler = GUIHandler()
        gui_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger = logging.getLogger(logger_name)
        logger.addHandler(gui_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent double-logging via root logger propagation
        
        return gui_handler
    
    def _cleanup_logging(self, logger_name, gui_handler):
        """Cleanup logging handlers"""
        logger = logging.getLogger(logger_name)
        logger.removeHandler(gui_handler)
        logger.propagate = True  # Restore default propagation
    
    def _flush_msg_queue(self):
        """Flush queued log messages - safe to call from QThread's run()"""
        with self._msg_lock:
            messages = self._msg_queue[:]
            self._msg_queue.clear()
        for msg in messages:
            self.progress.emit(msg)

    def _run_with_monitoring(self, step_info, gui_handler):
        import importlib
        import threading
        import time
        import numpy as np

        print(f"\n[RUN_WITH_MON] step={self.step}  has_callback={step_info['run_kwargs'].get('needs_callback', False)}")

        total_items = count_items_for_step(self.step, self.config.output_dir, self.loaded_sessions)
        print(f"[RUN_WITH_MON] total_items={total_items}  count_unit={step_info['count_unit']}")
        self.progress.emit(f"Found {total_items} {step_info['count_unit']} to process")

        output_dir = get_step_output_dir(self.step, self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN_WITH_MON] output_dir={output_dir}")

        file_pattern = get_step_file_pattern(self.step)
        print(f"[RUN_WITH_MON] file_pattern={file_pattern}")

        initial_mod_times = {}
        for f in output_dir.glob(file_pattern):
            try:
                initial_mod_times[f.name] = f.stat().st_mtime
            except:
                pass
        print(f"[RUN_WITH_MON] pre-existing files matching pattern: {len(initial_mod_times)}")

        run_kwargs_info = step_info['run_kwargs']
        has_callback = run_kwargs_info.get('needs_callback', False)

        if has_callback:
            print(f"[RUN_WITH_MON] Taking DIRECT EXECUTION path (step {self.step})")
            try:
                module = importlib.import_module(f"steps.{step_info['run_module']}")
                run_func = getattr(module, step_info['run_function'])
                kwargs = build_run_kwargs(self.step, self.config)
                print(f"[RUN_WITH_MON] Loaded module={step_info['run_module']}  func={step_info['run_function']}")

                if run_kwargs_info.get('needs_sessions'):
                    print(f"[RUN_WITH_MON] Calling run_func with loaded_sessions + session_callback")
                    self._step_result = run_func(
                        self.config,
                        loaded_sessions=self.loaded_sessions,
                        session_callback=self._emit_session_progress
                    )
                else:
                    print(f"[RUN_WITH_MON] Calling run_func with session_callback (Step 1.5 path)")
                    self._step_result = run_func(
                        **{**kwargs, 'session_callback': self._emit_animal_progress}
                    )

                print(f"[RUN_WITH_MON] run_func returned. _step_result type={type(self._step_result)}")

            except Exception as e:
                import traceback
                print(f"[RUN_WITH_MON] EXCEPTION in run_func: {e}")
                traceback.print_exc()
                self._step_error = str(e)

            print(f"[RUN_WITH_MON] Flushing msg queue after direct execution...")
            self._flush_msg_queue()
            print(f"[RUN_WITH_MON] Flush done. _step_error={self._step_error}  _step_result={self._step_result is not None}")

            if self._step_error:
                print(f"[RUN_WITH_MON] Emitting ERROR signal")
                self.error.emit(self._step_error)
            elif self._step_result is not None:
                print(f"[RUN_WITH_MON] Emitting FINISHED signal with result")
                self.finished.emit(self._step_result)
            else:
                print(f"[RUN_WITH_MON] _step_result is None — checking output files...")
                output_files = list(output_dir.glob(file_pattern))
                new_files = [f for f in output_files
                            if f.name not in initial_mod_times or
                            f.stat().st_mtime > initial_mod_times.get(f.name, 0)]
                print(f"[RUN_WITH_MON] new output files found: {len(new_files)}")
                if new_files:
                    self.progress.emit(f"Step completed — {len(new_files)} output file(s) written")
                    print(f"[RUN_WITH_MON] Emitting FINISHED signal (None result, files found)")
                    self.finished.emit(None)
                else:
                    print(f"[RUN_WITH_MON] NO output files — emitting ERROR signal")
                    self.error.emit(f"Step {self.step} completed but no output files were generated")
        else:
                    processing_complete = threading.Event()

                    def run_processing():
                        try:
                            module = importlib.import_module(f"steps.{step_info['run_module']}")
                            run_func = getattr(module, step_info['run_function'])
                            kwargs = build_run_kwargs(self.step, self.config)
                            print(f"[RUN_WITH_MON] THREADED PATH: calling {step_info['run_function']}...", file=sys.stderr, flush=True)
                            self._step_result = run_func(**kwargs)
                            print(f"[RUN_WITH_MON] THREADED PATH: run_func returned", file=sys.stderr, flush=True)
                        except Exception as e:
                            self._step_error = str(e)
                            import traceback
                            traceback.print_exc()
                        finally:
                            processing_complete.set()

                    thread = threading.Thread(target=run_processing)
                    thread.daemon = True
                    thread.start()

                    self._monitor_files(
                        output_dir, file_pattern, initial_mod_times,
                        total_items, processing_complete,
                        step_info['progress_signal']
                    )

                    processing_complete.wait()
                    thread.join(timeout=30.0)
                    self._flush_msg_queue()

                    if self._step_error:
                        self.error.emit(self._step_error)
                    elif self._step_result is not None:
                        self.finished.emit(self._step_result)
                    else:
                        output_files = list(output_dir.glob(file_pattern))
                        new_files = [f for f in output_files
                                    if f.name not in initial_mod_times or
                                    f.stat().st_mtime > initial_mod_times.get(f.name, 0)]
                        if new_files:
                            self.progress.emit(f"Step completed — {len(new_files)} output file(s) written")
                            self.finished.emit(None)
                        else:
                            self.error.emit(f"Step {self.step} completed but no output files were generated")

    def _monitor_files(self, output_dir, file_pattern, initial_mod_times, 
                      total_items, processing_complete, progress_signal):
        """Monitor file creation and emit progress — runs in QThread so signals are safe"""
        completed_items = 0
        item_times = []
        last_count = 0
        last_completion_time = time.time()
        
        while not processing_complete.is_set():
            # Flush any log messages queued by the background thread
            self._flush_msg_queue()
            
            current_files = list(output_dir.glob(file_pattern))
            
            updated_files = []
            for f in current_files:
                try:
                    current_mtime = f.stat().st_mtime
                    if f.name not in initial_mod_times or current_mtime > initial_mod_times[f.name]:
                        updated_files.append(f)
                except (FileNotFoundError, OSError):
                    continue
            
            completed_items = len(updated_files)
            
            if completed_items > last_count:
                current_time = time.time()
                time_since_last = current_time - last_completion_time
                item_times.append(time_since_last)
                last_completion_time = current_time
                last_count = completed_items
                
                avg_time = np.mean(item_times) if item_times else time_since_last
                
                if progress_signal == 'animal_progress':
                    self.animal_progress.emit(completed_items, total_items, avg_time)
            
            # Use wait() with timeout so we check processing_complete regularly
            processing_complete.wait(timeout=0.5)
        
        # One final flush after processing completes
        self._flush_msg_queue()
    
    def _emit_session_progress(self, current, total, session_time):
        self._flush_msg_queue()  # ← drain log queue so window updates live
        self.session_progress.emit(current, total, session_time)

    def _emit_animal_progress(self, current, total, avg_time):
        """DEBUG VERSION — no throttle, prints every call."""
        import time
        print(f"[ANIMAL_PROGRESS] current={current}/{total}  avg_time={avg_time:.3f}s")
        self._flush_msg_queue()
        self.animal_progress.emit(current, total, avg_time)

    def _emit_pair_progress(self, current, total, pair_time):
        """Emit pair-level progress for Step 1.5."""
        self.animal_progress.emit(current, total, pair_time)

# ============================================================================
# PIPELINE CONFIG DIALOG - FULLY SCHEMA DRIVEN
# ============================================================================

class PipelineConfigDialog(QDialog):
    """Dialog for configuring pipeline parameters - 100% SCHEMA DRIVEN"""
    
    def __init__(self, config, parent=None):
        super().__init__()
        self.config = config
        self.param_widgets = {}
        from utilities import apply_dark_theme_to_widget
        apply_dark_theme_to_widget(self)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI dynamically from PARAMETER_SCHEMAS"""
        self.setWindowTitle('Pipeline Configuration')
        self.setMinimumWidth(700)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel('⚙️ Pipeline Configuration')
        title.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; color: white;')
        layout.addWidget(title)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(500)
        scroll.setStyleSheet('''
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #00AAFF;
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
        ''')
        
        params_widget = QWidget()
        params_widget.setStyleSheet("background-color: #1E1E1E;")
        params_layout = QVBoxLayout(params_widget)
        
        # Directories group
        self._add_directories_group(params_layout)
        
        # General parameters group
        self._add_parameter_group(params_layout, "General", [
            'n_workers', 'verbose', 'skip_existing'
        ])
        
        # Step 1 parameters
        self._add_parameter_group(params_layout, "Step 1: Quad Generation", [
            'knn_k', 'diagonal_rng_seed', 'max_triangles_per_diagonal',
            'quad_keep_fraction', 'min_pairwise_distance',
        ])
        
        # Step 1.5 parameters
        self._add_parameter_group(params_layout, "Step 1.5: Calibration", [
            'sample_size', 'target_quality', 'threshold_min', 'threshold_max', 'n_threshold_points'
        ])
        
        # Step 2 parameters
        self._add_parameter_group(params_layout, "Step 2: Quad Matching", [
            'threshold', 'distance_metric', 'consistency_threshold'
        ])
        
        # Step 2.5 parameters
        self._add_parameter_group(params_layout, "Step 2.5: RANSAC", [
            'ransac_max_residual', 'ransac_iterations', 'ransac_min_inlier_ratio',
            'ransac_max_rotation_deg', 'ransac_max_translation_px',
        ])
        
        # Step 3 parameters
        self._add_parameter_group(params_layout, "Step 3: Neuron Matching", [
            'use_quad_voting',
            'use_asymmetric_dummy_costs',
            'block_zero_vote_pairs',
            'dist_cutoff_multiplier',
            'postfilter_residual_multiplier',
            'pass2_cutoff_multiplier',
            'pass2_dummy_percentile',
        ])
        
        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton('🔄 Reset All to Defaults')
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setStyleSheet('''
            QPushButton {
                background-color: #555555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        ''')
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton('❌ Cancel')
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet('''
            QPushButton {
                background-color: #555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        ''')
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton('💾 Save Configuration')
        save_btn.clicked.connect(self.accept_and_apply)
        save_btn.setStyleSheet('''
            QPushButton {
                background-color: #00CC00;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00DD00;
            }
        ''')
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _add_directories_group(self, parent_layout):
        """Add input/output directory controls"""
        group = QGroupBox("📁 Directories")
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
        ''')
        
        layout = QVBoxLayout()
        
        # Input directory (read-only)
        input_row = QHBoxLayout()
        input_label = QLabel("Input Directory:")
        input_label.setStyleSheet("color: white; font-size: 12px;")
        input_label.setMinimumWidth(150)
        input_row.addWidget(input_label)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setText(self.config.input_dir if self.config.input_dir else "Not set")
        self.input_dir_edit.setReadOnly(True)
        self.input_dir_edit.setStyleSheet("background-color: #1A1A1A; color: #888; border: 1px solid #555; padding: 3px;")
        input_row.addWidget(self.input_dir_edit)
        
        layout.addLayout(input_row)
        
        # Output directory (editable)
        output_row = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setStyleSheet("color: white; font-size: 12px;")
        output_label.setMinimumWidth(150)
        output_row.addWidget(output_label)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.config.output_dir if self.config.output_dir else "")
        self.output_dir_edit.setStyleSheet("background-color: #1A1A1A; color: white; border: 1px solid #555; padding: 3px;")
        output_row.addWidget(self.output_dir_edit)
        
        browse_btn = QPushButton('📁 Browse')
        browse_btn.clicked.connect(self.browse_output_dir)
        browse_btn.setMaximumWidth(100)
        browse_btn.setStyleSheet('''
            QPushButton {
                background-color: #4400AA;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5500BB;
            }
        ''')
        output_row.addWidget(browse_btn)
        
        layout.addLayout(output_row)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _add_parameter_group(self, parent_layout, title, param_names):
        """Add a group of parameters from schemas"""
        group = QGroupBox(f"⚙️ {title}")
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
            QLabel {
                background-color: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1A1A1A;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
            QCheckBox {
                color: white;
            }
        ''')
        
        layout = QVBoxLayout()
        
        for param_name in param_names:
            self._add_schema_param(layout, param_name)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _add_schema_param(self, layout, param_name):
        """Add a parameter widget from PARAMETER_SCHEMAS"""
        schema = get_parameter_schema(param_name)
        if not schema:
            return
        
        # Get current value from config
        value = getattr(self.config, param_name, schema['default'])
        
        row = QHBoxLayout()
        
        # Label
        label_text = schema.get('description', param_name.replace('_', ' ').title())
        label_widget = QLabel(f"{label_text}:")
        label_widget.setStyleSheet("color: white; font-size: 12px;")
        label_widget.setMinimumWidth(250)
        row.addWidget(label_widget)
        
        # Widget based on schema
        widget_type = schema.get('widget', 'lineedit')
        
        if widget_type == 'spinbox':
            widget = QSpinBox()
            widget.setRange(schema.get('min', 0), schema.get('max', 999999))
            widget.setValue(int(value) if value is not None else schema['default'])
            
        elif widget_type == 'doublespinbox':
            widget = QDoubleSpinBox()
            decimals = schema.get('decimals', 2)
            step_size = schema.get('step', 10.0 ** (-decimals))
            widget.setDecimals(decimals)
            widget.setSingleStep(step_size)

            real_min = schema.get('min', 0.0)
            real_max = schema.get('max', 999999.0)

            if schema.get('nullable'):
                # Put sentinel one step below real min so None is distinct from min
                sentinel = real_min - step_size
                widget.setRange(sentinel, real_max)
                widget.setSpecialValueText("None (no limit)")
                if value is None:
                    widget.setValue(sentinel)
                else:
                    widget.setValue(float(value))
            else:
                widget.setRange(real_min, real_max)
                widget.setValue(float(value) if value is not None else schema['default'])

        elif widget_type == 'checkbox':
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else schema['default'])
            
        elif widget_type == 'combobox':
            widget = QComboBox()
            options = schema.get('options', [])
            widget.addItems(options)
            if value in options:
                widget.setCurrentText(value)
            
        else:
            widget = QLineEdit(str(value) if value is not None else str(schema['default']))
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        current_dir = self.output_dir_edit.text()
        if not current_dir:
            current_dir = self.config.input_dir if self.config.input_dir else "."
        
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", current_dir)
        if folder:
            self.output_dir_edit.setText(folder)
    
    def reset_to_defaults(self):
        """Reset all parameters to defaults from PARAMETER_SCHEMAS"""
        for param_name, widget in self.param_widgets.items():
            schema = get_parameter_schema(param_name)
            if not schema:
                continue
            
            default = schema['default']
            
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default) if default is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                if default is None and schema.get('nullable'):
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default) if default is not None else 0.0)
            elif isinstance(widget, QComboBox):
                if default in schema.get('options', []):
                    widget.setCurrentText(default)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(default) if default is not None else "")
        
        QMessageBox.information(self, 'Reset', 'All parameters reset to defaults from step_info.py')
    
    def accept_and_apply(self):
        """Apply changes to config and accept"""
        # Apply all parameter widgets
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(self.config, param_name, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                setattr(self.config, param_name, widget.text())
            elif isinstance(widget, QSpinBox):
                setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                schema = get_parameter_schema(param_name)
                if schema and schema.get('nullable') and widget.value() == widget.minimum():
                    setattr(self.config, param_name, None)
                else:
                    setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QComboBox):
                setattr(self.config, param_name, widget.currentText())
        
        # Update output directory
        new_output = self.output_dir_edit.text()
        if new_output and new_output != self.config.output_dir:
            self.config.output_dir = new_output
            self.config.output_path = Path(new_output)
            self.config.intermediate_path = Path(new_output) / "intermediate"
        
        self.accept()
    
    def get_config(self):
        """Get updated configuration"""
        return self.config

def _compute_quad_estimate(
    n_neurons, n_sessions, knn_k,
    keep_frac, max_K,
):
    _Q_CAP = 1_314_266

    k_random    = max(2, knn_k // 2)
    n_diag_eff  = n_neurons * (knn_k + k_random) / 2.0
    M_eff       = min(float(max_K), float(max(0, n_neurons - 2)))
    q_per_diag  = M_eff * (M_eff - 1) / 2.0
    est_q_final = int(n_diag_eff * q_per_diag * keep_frac)

    over_cap  = est_q_final > _Q_CAP
    overshoot = max(0.0, 100.0 * (est_q_final - _Q_CAP) / _Q_CAP)

    suggested_kf = keep_frac
    suggested_K  = max_K
    if over_cap and est_q_final > 0:
        suggested_kf = keep_frac * (_Q_CAP / est_q_final)
        suggested_kf = max(0.001, round(suggested_kf, 3))
        import math
        suggested_K = max(2, int(math.floor(
            0.5 + 0.5 * math.sqrt(1 + 8 * _Q_CAP / max(n_diag_eff * keep_frac, 1))
        )))

    return {
        'n_sessions':    n_sessions,
        'n_neurons':     n_neurons,
        'knn_k':         knn_k,
        'k_random':      k_random,
        'n_diag_eff':    n_diag_eff,
        'keep_frac':     keep_frac,
        'max_K':         max_K,
        'M_eff':         M_eff,
        'q_per_diag':    q_per_diag,
        'est_q_final':   est_q_final,
        'cap':           _Q_CAP,
        'over_cap':      over_cap,
        'suggested_kf':  suggested_kf,
        'suggested_K':   suggested_K,
        'overshoot_pct': overshoot,
    }

# ============================================================================
# REST OF COMPONENTS 
# ============================================================================

class StepConfirmationDialog(QDialog):
    """Step-specific confirmation dialog - uses step_info.py for metadata"""
    
    def __init__(self, step, config, parent=None):
        super().__init__()
        self.step = step
        self.config = config
        self.param_widgets = {}
        self._step1_est = None  # populated after show via QTimer
        from utilities import apply_dark_theme_to_widget
        apply_dark_theme_to_widget(self)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI — estimate panel is deferred to after show()"""
        step_info = get_step_info(self.step)
        step_name = step_info.get('name', f'Step {self.step}')
        step_icon = step_info.get('icon', '▶')

        self.setWindowTitle(f'Confirm {step_info.get("label", f"Step {self.step}")}')
        self.setMinimumWidth(600)

        layout = QVBoxLayout()

        title = QLabel(f'{step_icon} Run {step_info.get("label", f"Step {self.step}")}')
        title.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; color: white;')
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        scroll.setMaximumHeight(500)
        scroll.setStyleSheet('''
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #00AAFF;
                border-radius: 5px;
            }
            QScrollArea > QWidget > QWidget { background-color: #1E1E1E; }
            QScrollBar:vertical {
                background-color: #1E1E1E; width: 12px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #00AAFF; border-radius: 6px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background-color: #00DDFF; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none; background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        ''')

        params_widget = QWidget()
        params_widget.setStyleSheet("background-color: #1E1E1E;")
        self._params_layout = QVBoxLayout(params_widget)
        self._params_layout.setContentsMargins(5, 5, 5, 5)

        # Add non-estimate params immediately (fast, no I/O)
        self._add_step_parameters_static(self._params_layout, step_info)

        scroll.setWidget(params_widget)
        layout.addWidget(scroll)

        message = QLabel(self._get_step_message(step_info))
        message.setWordWrap(True)
        message.setStyleSheet('padding: 10px; color: #FFD700; font-size: 12px;')
        layout.addWidget(message)

        button_layout = QHBoxLayout()

        reset_btn = QPushButton('🔄 Reset to Defaults')
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setStyleSheet('''
            QPushButton {
                background-color: #555555; color: white;
                padding: 8px 15px; border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #666666; }
        ''')
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()

        cancel_btn = QPushButton('❌ Cancel')
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet('''
            QPushButton {
                background-color: #555; color: white;
                padding: 8px 15px; border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #666; }
        ''')
        button_layout.addWidget(cancel_btn)

        # Run button — starts enabled; may be disabled after estimate loads
        self._run_btn = QPushButton('▶  Run Step')
        self._run_btn.clicked.connect(self.accept_and_apply)
        self._run_btn.setStyleSheet('''
            QPushButton {
                background-color: #00CC00; color: white;
                font-weight: bold; padding: 8px 20px;
                border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover:enabled { background-color: #00DD00; }
            QPushButton:disabled {
                background-color: #444; color: #888;
            }
        ''')
        button_layout.addWidget(self._run_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # ── Defer the Step 1 estimate panel to after the dialog is visible ──
        if self.step == 1:
            self._est_placeholder = None  # will be set in _add_estimate_panel_deferred
            QTimer.singleShot(0, self._load_estimate_deferred)

    def _load_estimate_deferred(self):
        """
        Called via QTimer.singleShot(0) — runs after the event loop processes
        the show() call, so the dialog is already fully visible on screen.

        Loads one random .npy file, computes the estimate, caches n_neurons so
        all subsequent live refreshes are pure arithmetic (no file I/O), then
        wires up valueChanged signals on the relevant widgets.
        """
        step_info = get_step_info(self.step)

        placeholder = QLabel("  📊 Computing quad estimate from a sample session…")
        placeholder.setStyleSheet(
            "color: #888; font-size: 11px; padding: 6px; background: transparent;"
        )
        self._params_layout.insertWidget(1, placeholder)
        QApplication.processEvents()

        # Compute initial estimate (reads ONE random file)
        est = self._compute_step1_estimate()
        self._step1_est       = est
        self._n_neurons_cached  = est['n_neurons']   # reused on every refresh
        self._n_sessions_cached = est['n_sessions']

        placeholder.setParent(None)

        self._est_group = self._build_estimate_group(est, step_info)
        self._params_layout.insertWidget(1, self._est_group)

        if est.get('over_cap', False):
            self._run_btn.setEnabled(False)
            self._run_btn.setText('⚠️  Fix overshoot above to enable Run')

        # ── Wire up live refresh on all parameters that affect the estimate ──
        live_params = [
            'quad_keep_fraction',
            'max_triangles_per_diagonal',
            'knn_k',
        ]
        for param_name in live_params:
            widget = self.param_widgets.get(param_name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(self._refresh_estimate)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._refresh_estimate)

        QApplication.processEvents()

    def _refresh_estimate(self):
        """Re-compute estimate from current widget values and rebuild the panel in-place."""
        step_info = get_step_info(self.step)

        # ── Read current widget values — no file I/O ──
        knn_k     = 15
        keep_frac = 1.0
        max_K     = 25

        w = self.param_widgets.get('knn_k')
        if isinstance(w, QSpinBox):
            knn_k = w.value()

        w = self.param_widgets.get('quad_keep_fraction')
        if isinstance(w, QDoubleSpinBox):
            keep_frac = w.value()

        w = self.param_widgets.get('max_triangles_per_diagonal')
        if isinstance(w, QSpinBox):
            max_K = w.value()

        n_neurons  = getattr(self, '_n_neurons_cached',  250)
        n_sessions = getattr(self, '_n_sessions_cached', 0)

        est = _compute_quad_estimate(
            n_neurons, n_sessions, knn_k,
            keep_frac, max_K,
        )
        self._step1_est = est

        # Swap old estimate group for the new one
        if hasattr(self, '_est_group') and self._est_group is not None:
            self._params_layout.removeWidget(self._est_group)
            self._est_group.setParent(None)

        self._est_group = self._build_estimate_group(est, step_info)
        self._params_layout.insertWidget(1, self._est_group)

        if est['over_cap']:
            self._run_btn.setEnabled(False)
            self._run_btn.setText('⚠️  Fix overshoot above to enable Run')
        else:
            self._run_btn.setEnabled(True)
            self._run_btn.setText('▶  Run Step')

        QApplication.processEvents()

    def _compute_step1_estimate(self):
        """
        Quick quad estimate by peeking at ONE randomly-chosen .npy file.
        Delegates arithmetic to the module-level _compute_quad_estimate()
        so _refresh_estimate can reuse the same logic without file I/O.
        """
        import random

        n_sessions = 0
        sample_npy = None
        for attr in ('input_path', 'input_dir'):
            val = getattr(self.config, attr, None)
            if val:
                npy_files = list(Path(val).glob('**/*.npy'))
                n_sessions = len(npy_files)
                if npy_files:
                    sample_npy = random.choice(npy_files)
                break

        n_neurons = 250
        if sample_npy is not None:
            try:
                raw = np.load(str(sample_npy), allow_pickle=True)
                if isinstance(raw, np.ndarray) and raw.ndim == 0:
                    data = raw.item()
                    if isinstance(data, dict) and 'centroids_x' in data:
                        n_neurons = len(data['centroids_x'])
                elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                    n_neurons = raw.shape[2]
            except Exception:
                pass

        return _compute_quad_estimate(
            n_neurons  = n_neurons,
            n_sessions = n_sessions,
            knn_k      = getattr(self.config, 'knn_k', 15),
            keep_frac  = getattr(self.config, 'quad_keep_fraction', 1.0),
            max_K      = getattr(self.config, 'max_triangles_per_diagonal', 25),
        )

    def _build_estimate_group(self, est, step_info):
        """Build the quad estimate QGroupBox from a completed estimate dict."""
        status_color = '#FF4444' if est['over_cap'] else '#00CC44'
        status_icon  = '⚠️' if est['over_cap'] else '✅'

        est_group = self._create_param_group("📊 Quad Estimate (sampled from 1 random session)")
        est_layout = QVBoxLayout()
        est_layout.setSpacing(4)

        def _info_row(label, value, color='#CCCCCC'):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888888; font-size: 11px; background: transparent;")
            lbl.setMinimumWidth(220)
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; background: transparent;")
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            return row

        est_layout.addLayout(_info_row("Sessions to process",        f"{est['n_sessions']:,}"))
        est_layout.addLayout(_info_row("Neurons / session",          f"~{est['n_neurons']}"))
        est_layout.addLayout(_info_row("Diagonals sampled (est)",    f"~{int(est['n_diag_eff']):,}  (k={est['knn_k']} local + {est['k_random']} random)"))
        est_layout.addLayout(_info_row("Third-points / diagonal (K)", f"{est['max_K']}  →  {int(est['q_per_diag'])} quads/diag"))
        est_layout.addLayout(_info_row("quad_keep_fraction",          f"{est['keep_frac']:.3f}"))
        est_layout.addLayout(_info_row("Quads / session (est)",       f"~{est['est_q_final']:,}"))
        est_layout.addLayout(_info_row("Empirical descriptor cap",    f"{est['cap']:,}"))

        # Status line
        if est['over_cap']:
            status_text = f"{status_icon}  Overshoots cap by {est['overshoot_pct']:.1f}%"
        else:
            status_text = f"{status_icon}  Within cap — no adjustment needed"
        status_lbl = QLabel(status_text)
        status_lbl.setStyleSheet(
            f"color: {status_color}; font-size: 12px; font-weight: bold; "
            f"padding: 6px; background: transparent;"
        )
        est_layout.addWidget(status_lbl)

        # Suggestions
        if est['over_cap']:
            sep = QLabel("─" * 40)
            sep.setStyleSheet("color: #444; background: transparent;")
            est_layout.addWidget(sep)

            sugg_title = QLabel("💡 Suggested parameters to stay within cap:")
            sugg_title.setStyleSheet(
                "color: #FFD700; font-size: 11px; font-weight: bold; "
                "background: transparent; padding-top: 4px;"
            )
            est_layout.addWidget(sugg_title)

            est_layout.addLayout(_info_row(
                "  Option A — quad_keep_fraction",
                f"{est['suggested_kf']:.3f}  (trim after generation, zero extra compute)",
                '#00AAFF'
            ))
            est_layout.addLayout(_info_row(
                "  Option B — max_triangles_per_diagonal",
                f"{est['suggested_K']}  (generate fewer quads, faster)",
                '#00AAFF'
            ))

            apply_btn = QPushButton(
                "⚡  Apply Option A  (set keep_fraction = {:.3f})".format(est['suggested_kf'])
            )
            apply_btn.setStyleSheet("""
                QPushButton {
                    background-color: #005599; color: white;
                    font-weight: bold; padding: 6px 12px;
                    border-radius: 4px; font-size: 12px; margin-top: 6px;
                }
                QPushButton:hover { background-color: #0077BB; }
            """)
            apply_btn.clicked.connect(self._apply_option_a)
            est_layout.addWidget(apply_btn)

            apply_b_btn = QPushButton(
                            "⚡  Apply Option B  (max_triangles_per_diagonal = {})".format(
                                est['suggested_K']
                            )
                        )
            apply_b_btn.setStyleSheet("""
                QPushButton {
                    background-color: #005599; color: white;
                    font-weight: bold; padding: 6px 12px;
                    border-radius: 4px; font-size: 12px;
                }
                QPushButton:hover { background-color: #0077BB; }
            """)
            apply_b_btn.clicked.connect(self._apply_option_b)
            est_layout.addWidget(apply_b_btn)

        est_group.setLayout(est_layout)
        return est_group

    def _add_schema_param(self, layout, param_name):
        """Add a parameter row using PARAMETER_SCHEMAS"""
        schema = get_parameter_schema(param_name)
        if not schema:
            return
        
        value = getattr(self.config, param_name, schema['default'])
        
        row = QHBoxLayout()
        
        label_text = schema.get('description', param_name.replace('_', ' ').title())
        label_widget = QLabel(f"{label_text}:")
        label_widget.setStyleSheet("color: white; font-size: 12px; background-color: transparent;")
        label_widget.setMinimumWidth(200)
        row.addWidget(label_widget)
        
        widget_type = schema.get('widget', 'lineedit')
        
        if widget_type == 'spinbox':
            widget = QSpinBox()
            widget.setRange(schema.get('min', 0), schema.get('max', 999999))
            widget.setValue(int(value) if value is not None else schema['default'])
            
        elif widget_type == 'doublespinbox':
            widget = QDoubleSpinBox()
            decimals = schema.get('decimals', 2)
            step_size = schema.get('step', 10.0 ** (-decimals))
            widget.setDecimals(decimals)
            widget.setSingleStep(step_size)

            real_min = schema.get('min', 0.0)
            real_max = schema.get('max', 999999.0)

            if schema.get('nullable'):
                sentinel = real_min - step_size
                widget.setRange(sentinel, real_max)
                widget.setSpecialValueText("None (no limit)")
                if value is None:
                    widget.setValue(sentinel)
                else:
                    widget.setValue(float(value))
            else:
                widget.setRange(real_min, real_max)
                widget.setValue(float(value) if value is not None else schema['default'])
                
        elif widget_type == 'checkbox':
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else schema['default'])
            
        elif widget_type == 'combobox':
            widget = QComboBox()
            options = schema.get('options', [])
            widget.addItems(options)
            if value in options:
                widget.setCurrentText(value)
            
        else:
            widget = QLineEdit(str(value) if value is not None else str(schema['default']))
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)

    def _apply_option_a(self):
        """Apply suggested keep_fraction and re-enable Run button."""
        est = getattr(self, '_step1_est', None)
        if est is None:
            return
        widget = self.param_widgets.get('quad_keep_fraction')
        if widget is not None:
            widget.setValue(est['suggested_kf'])
        if hasattr(self, '_run_btn'):
            self._run_btn.setEnabled(True)
            self._run_btn.setText('▶  Run Step')

    def _apply_option_b(self):
        """Apply suggested knn_k + keep_fraction=1.0 and re-enable Run button."""
        est = getattr(self, '_step1_est', None)
        if est is None or est['suggested_knn_k'] is None:
            return
        knn_widget     = self.param_widgets.get('knn_k')
        kf_widget      = self.param_widgets.get('quad_keep_fraction')
        use_knn_widget = self.param_widgets.get('use_knn_triangles')
        if knn_widget is not None:
            knn_widget.setValue(est['suggested_knn_k'])
        if kf_widget is not None:
            kf_widget.setValue(1.0)
        if use_knn_widget is not None:
            use_knn_widget.setChecked(True)
        if hasattr(self, '_run_btn'):
            self._run_btn.setEnabled(True)
            self._run_btn.setText('▶ Run Step')

    def _add_step_parameters_static(self, layout, step_info):
        """
        Add everything EXCEPT the Step 1 estimate panel — this runs synchronously
        at dialog init so it's instant (no file I/O).  The estimate panel is
        inserted at position 1 via _load_estimate_deferred().
        """
        # ── Common params ──
        common_group = self._create_param_group("📁 Common Parameters")
        common_layout = QVBoxLayout()
        self._add_param_row(common_layout, "Output Directory", "output_dir",
                            self.config.output_dir, "readonly")
        self._add_schema_param(common_layout, "n_workers")
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)

        # ── Step-specific editable params ──
        step_params = get_step_parameters(self.step)
        if step_params:
            step_group = self._create_param_group(f"⚙️ {step_info['name']} Parameters")
            step_layout = QVBoxLayout()
            for param_name in step_params:
                self._add_schema_param(step_layout, param_name)
            step_group.setLayout(step_layout)
            layout.addWidget(step_group)

    # Keep _add_step_parameters as an alias so nothing else breaks
    def _add_step_parameters(self, layout, step_info):
        self._add_step_parameters_static(layout, step_info)

    def _create_param_group(self, title):
        """Create a styled parameter group"""
        group = QGroupBox(title)
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
            QLabel {
                background-color: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1A1A1A;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #00AAFF;
            }
            QLineEdit {
                background-color: #1A1A1A;
                color: #888;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
        ''')
        return group

    def _add_param_row(self, layout, label, param_name, value, widget_type, 
                    min_val=None, max_val=None, step=None, decimals=2, tooltip=None):
        """Add a parameter row with label and editable widget"""
        row = QHBoxLayout()
        
        label_widget = QLabel(f"{label}:")
        label_widget.setStyleSheet("color: white; font-size: 12px; background-color: transparent;")
        label_widget.setMinimumWidth(200)
        if tooltip:
            label_widget.setToolTip(tooltip)
        row.addWidget(label_widget)
        
        if widget_type == "readonly":
            widget = QLineEdit(str(value))
            widget.setReadOnly(True)
        elif widget_type == "spinbox":
            widget = QSpinBox()
            widget.setRange(min_val or 0, max_val or 999999)
            widget.setValue(int(value))
        elif widget_type == "doublespinbox":
            widget = QDoubleSpinBox()
            widget.setRange(min_val or 0.0, max_val or 999999.0)
            widget.setSingleStep(step or 0.1)
            widget.setDecimals(decimals)
            widget.setValue(float(value))
        elif widget_type == "checkbox":
            widget = QCheckBox()
            widget.setChecked(bool(value))
        else:
            widget = QLineEdit(str(value))
        
        if tooltip:
            widget.setToolTip(tooltip)
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)

    def reset_to_defaults(self):
        """Reset all parameters to defaults from PARAMETER_SCHEMAS"""
        for param_name, widget in self.param_widgets.items():
            schema = get_parameter_schema(param_name)
            if not schema:
                continue
                
            default = schema['default']
            
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default) if default is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                if default is None and schema.get('nullable'):
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default) if default is not None else 0.0)
            elif isinstance(widget, QComboBox):
                if default in schema.get('options', []):
                    widget.setCurrentText(default)
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                widget.setText(str(default) if default is not None else "")
        
        QMessageBox.information(self, 'Reset', 'Parameters reset to defaults from step_info.py')

    def accept_and_apply(self):
        """Apply changes to config and accept dialog"""
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(self.config, param_name, widget.isChecked())
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                setattr(self.config, param_name, widget.text())
            elif isinstance(widget, QSpinBox):
                setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                schema = get_parameter_schema(param_name)
                if schema and schema.get('nullable') and widget.value() == widget.minimum():
                    setattr(self.config, param_name, None)
                else:
                    setattr(self.config, param_name, widget.value())
        
        self.accept()
    
    def _get_step_message(self, step_info) -> str:
        """Minimal prereq message — estimate is shown in the group above."""
        prereqs = step_info.get('prerequisites', [])
        prereq_text = (
            f"Requires Step {', '.join(map(str, prereqs))} results"
            if prereqs else "No prerequisites"
        )
        return f"⚠️  {step_info.get('description', '')}\n{prereq_text}"

def open_results_inspector(step, config, parent=None):
    """Open the appropriate results inspector for a pipeline step"""
    from viewers import (
        Step1Viewer, Step1_5Viewer, Step2Viewer, Step2_5Viewer, Step3Viewer,
    )
    
    # Parse step if it's a directory name string
    if isinstance(step, str):
        from .step_info import parse_step_from_dirname
        parsed = parse_step_from_dirname(step)
        if parsed:
            step = parsed
        else:
            QMessageBox.critical(parent, 'Error', f'Unknown step: {step}')
            return
    
    # Map steps to viewers
    viewers = {
        1: Step1Viewer,
        1.5: Step1_5Viewer,
        2: Step2Viewer,
        2.5: Step2_5Viewer,
        3: Step3Viewer,
    }
    
    try:
        viewer_class = viewers.get(step)
        if viewer_class:
            viewer = viewer_class(config, parent)
            viewer.exec_()
        else:
            QMessageBox.critical(parent, 'Error', f'No viewer for step {step}')
    except ImportError as e:
        QMessageBox.warning(parent, 'Import Error', f'Could not import viewer:\n{e}')
    except Exception as e:
        import traceback
        QMessageBox.critical(parent, 'Error', f'Error launching viewer:\n{e}\n\n{traceback.format_exc()}')

def create_progress_dialog_with_log(parent, title, step, maximum=0):
    """Create a progress dialog with embedded log window"""
    step_info = get_step_info(step)
    label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
    
    progress = QProgressDialog(f'Running {label}...', 'Cancel', 0, maximum, parent)
    progress.setWindowModality(Qt.WindowModal)
    progress.setWindowTitle(title)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    
    progress.start_time = time.time()
    progress.session_times = []
    
    log_window = QDialog(parent)
    log_window.setWindowTitle(f'{label} Log')
    log_window.resize(600, 400)
    log_layout = QVBoxLayout()
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setFont(QFont('Courier', 9))
    log_layout.addWidget(log_text)
    log_window.setLayout(log_layout)
    log_window.show()
    
    return progress, log_window, log_text

def apply_button_style(button, style_type='primary'):
    """Apply consistent button styling"""
    if sys.platform == 'darwin':
        button.setFont(QFont(".AppleSystemUIFont", 13))  # enables emoji fallback

    styles = {
        'primary': '''
            QPushButton {
                background-color: #00AAFF;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0088DD;
            }
        ''',
        'pipeline': '''
            QPushButton {
                background-color: #4400AA;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5500BB;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        ''',
        'success': '''
            QPushButton {
                background-color: #00CC00;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00DD00;
            }
        ''',
        'config': '''
            QPushButton {
                background-color: #6600CC;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7700DD;
            }
        '''
    }
    
    style = styles.get(style_type, styles['primary'])
    if sys.platform == 'darwin':
        style = style.replace('QPushButton {', 'QPushButton { font-family: ".AppleSystemUIFont", "Apple Color Emoji";')
    button.setStyleSheet(style)

def create_pipeline_callbacks(parent_viewer, step, config):
    """Create standardized callbacks for pipeline step execution"""
    from PyQt5.QtWidgets import QMessageBox, QApplication
    from .step_info import get_step_enables
    
    def on_progress(msg):
        """Handle progress updates from worker thread"""
        if hasattr(parent_viewer, '_current_log_text'):
            parent_viewer._current_log_text.append(msg)
            parent_viewer._current_log_text.verticalScrollBar().setValue(
                parent_viewer._current_log_text.verticalScrollBar().maximum()
            )
        print(msg)
        QApplication.processEvents()
    
    def on_finished(result):
        """Handle successful completion"""
        if hasattr(parent_viewer, '_current_progress'):
            parent_viewer._current_progress.close()
        
        # Enable next steps based on metadata
        step_key = str(step).replace('.', '_')
        inspect_btn_name = f'step{step_key}_inspect_btn'
        if hasattr(parent_viewer, inspect_btn_name):
            getattr(parent_viewer, inspect_btn_name).setEnabled(True)
        
        # Enable steps that this step enables
        for enabled_step in get_step_enables(step):
            enabled_key = str(enabled_step).replace('.', '_')
            btn_name = f'step{enabled_key}_btn'
            if hasattr(parent_viewer, btn_name):
                getattr(parent_viewer, btn_name).setEnabled(True)
        
        step_info = get_step_info(step)
        label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
        
        QMessageBox.information(
            parent_viewer, 'Success', 
            f'{label} completed successfully!\n\nResults saved to:\n{config.output_dir}'
        )
    
    def on_error(error_msg):
        """Handle errors"""
        if hasattr(parent_viewer, '_current_progress'):
            parent_viewer._current_progress.close()
        
        import traceback
        if hasattr(parent_viewer, '_current_log_text'):
            parent_viewer._current_log_text.append(f'\n{"="*60}')
            parent_viewer._current_log_text.append(f'ERROR: {error_msg}')
            parent_viewer._current_log_text.append(f'{"="*60}')
            parent_viewer._current_log_text.append('\nTraceback:')
            parent_viewer._current_log_text.append(traceback.format_exc())
        
        print(f'\nERROR: {error_msg}')
        print(traceback.format_exc())
        
        step_info = get_step_info(step)
        label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
        
        QMessageBox.critical(parent_viewer, 'Error', f'{label} failed:\n{error_msg}')
    
    return on_progress, on_finished, on_error

def confirm_step_execution(step, config, parent=None) -> bool:
    """Show confirmation dialog before running a pipeline step"""
    dialog = StepConfirmationDialog(step, config, parent)
    return dialog.exec_() == QDialog.Accepted