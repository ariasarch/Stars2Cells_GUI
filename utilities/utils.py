"""
Utility functions for the neuron mapping pipeline.
"""

from pathlib import Path

def show_splash_screen(app, logo_path: str = "S2C_logo.png", 
                                   duration: int = 2000, message: str = "Loading..."):
    """
    Display a splash screen with logo and loading message.
    """
    from PyQt5.QtWidgets import QSplashScreen
    from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont, QPen, QBrush
    from PyQt5.QtCore import Qt, QRect
    from pathlib import Path
    
    # Find logo (same logic as above)
    logo_file = Path(logo_path)
    if not logo_file.exists():
        search_paths = [
            Path(__file__).parent / logo_path,
            Path(__file__).parent.parent / logo_path,
            Path.cwd() / logo_path,
            Path('/mnt/project') / logo_path,
        ]
        for search_path in search_paths:
            if search_path.exists():
                logo_file = search_path
                break
    
    if not logo_file.exists():
        print(f"Warning: Logo not found at {logo_path}")
        return None
    
    pixmap = QPixmap(str(logo_file))
    
    if pixmap.width() > 800 or pixmap.height() > 600:
        pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    # Add creator name to pixmap
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Set up font for measuring text
    name_font = QFont("Arial", 11)
    name_font.setBold(True)
    painter.setFont(name_font)
    
    # Measure text to create background rectangle
    text = "Created by the Neumaier Lab"
    font_metrics = painter.fontMetrics()
    text_width = font_metrics.width(text)
    text_height = font_metrics.height()
    
    # Create rectangle for background (with padding)
    padding = 8
    rect_width = text_width + (padding * 2)
    rect_height = text_height + (padding * 2)
    
    # Position rectangle at bottom center
    rect_x = (pixmap.width() - rect_width) // 2
    rect_y = pixmap.height() - rect_height - 45  # 45px from bottom
    
    background_rect = QRect(rect_x, rect_y, rect_width, rect_height)
    
    # Draw black background with white outline
    painter.setPen(QPen(QColor(255, 255, 255), 2))  # White outline, 2px thick
    painter.setBrush(QBrush(QColor(0, 0, 0)))  # Black fill
    painter.drawRect(background_rect)
    
    # Draw creator name in white
    painter.setPen(QColor(255, 255, 255))  # White text
    painter.drawText(
        background_rect,
        Qt.AlignCenter,
        text
    )
    
    painter.end()
    
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    
    # Add loading message (above the creator name)
    splash.showMessage(
        message,
        Qt.AlignBottom | Qt.AlignCenter,
        QColor(0, 170, 255)  # Cyan color
    )
    
    splash.show()
    app.processEvents()
    
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(duration, splash.close)
    
    return splash

def set_application_icon(app_or_window, icon_path: str = "S2C_logo.png"):
    """
    Set the application icon for window title bar and taskbar.
    
    Args:
        app_or_window: QApplication or QMainWindow instance
        icon_path: Path to icon image file
    
    Returns:
        QIcon instance (or None if image not found)
    """
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from pathlib import Path
    
    # Try to find the icon
    icon_file = Path(icon_path)
    
    # If not found, try in common locations
    if not icon_file.exists():
        search_paths = [
            Path(__file__).parent / icon_path,
            Path(__file__).parent.parent / icon_path,
            Path.cwd() / icon_path,
            Path('/mnt/project') / icon_path,
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                icon_file = search_path
                break
    
    # Check if there is the icon
    if not icon_file.exists():
        print(f"Warning: Icon not found at {icon_path}")
        return None
    
    # Create icon
    icon = QIcon(str(icon_file))
    
    # Set on application and/or window
    if isinstance(app_or_window, QApplication):
        app_or_window.setWindowIcon(icon)
    elif isinstance(app_or_window, QMainWindow):
        app_or_window.setWindowIcon(icon)
    
    return icon

def apply_dark_theme_to_widget(widget):
    """
    Apply dark theme palette to a Qt widget or application.
    
    Args:
        widget: QWidget or QApplication to apply theme to
    """
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtCore import Qt
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(13, 13, 13))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(0, 170, 255))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    widget.setPalette(palette)

def detect_pipeline_results(folder: Path) -> dict:
    """
    Auto-detect Stars2Cells pipeline results by searching for the project structure.
    
    Searches up (parent directories), down (subdirectories), and sideways (siblings)
    to find a folder containing step_X_results directories.
    
    Parameters
    ----------
    folder : Path
        Starting folder to search from
        
    Returns
    -------
    dict
        Dictionary with detected steps and project root
    """
    folder = Path(folder)
    
    results = {
        'project_root': None,
        'step_1': {'found': False, 'path': None, 'files': []},
        'step_1_5': {'found': False, 'path': None, 'files': []},
        'step_2': {'found': False, 'path': None, 'files': []},
        'step_2_5': {'found': False, 'path': None, 'files': []},
        'step_3': {'found': False, 'path': None, 'files': []}
    }
    
    def looks_like_stars2cells_project(directory: Path) -> bool:
        """Check if a directory looks like a Stars2Cells project."""
        step_folders = [
            'step_1_results', 'step_1_5_results', 'step_2_results',
            'step_2_5_results', 'step_3_results'
        ]
        # If 2+ step folders found, it's probably a Stars2Cells project
        found_count = sum(1 for name in step_folders if (directory / name).exists())
        return found_count >= 2
    
    def scan_for_project_root(start_dir: Path, max_depth: int = 3) -> Path:
        """Find the Stars2Cells project root directory."""
        
        print(f"[SCAN] Starting search from: {start_dir}")
        
        # 1. Check current directory first
        if looks_like_stars2cells_project(start_dir):
            print(f"[SCAN] ✓ Found project at current location: {start_dir}")
            return start_dir
        
        # 2. Search UP (parent directories) - up to 3 levels
        print(f"[SCAN] Searching parent directories...")
        current = start_dir
        for level in range(3):
            if current.parent == current:  # Reached filesystem root
                print(f"[SCAN]   Reached filesystem root")
                break
            current = current.parent
            print(f"[SCAN]   Checking parent {level+1}: {current}")
            if looks_like_stars2cells_project(current):
                print(f"[SCAN] ✓ Found project in parent: {current}")
                return current
        
        # 3. Search SIBLINGS (if there is a parent)
        if start_dir.parent != start_dir:
            print(f"[SCAN] Searching sibling directories of {start_dir.name}...")
            try:
                for sibling in start_dir.parent.iterdir():
                    if sibling == start_dir or not sibling.is_dir():
                        continue
                    
                    # Skip hidden folders and common system folders
                    if sibling.name.startswith('.') or sibling.name.startswith('$'):
                        continue
                    
                    print(f"[SCAN]   Checking sibling: {sibling.name}")
                    if looks_like_stars2cells_project(sibling):
                        print(f"[SCAN] ✓ Found project in sibling: {sibling}")
                        return sibling
                    
                    # Also check one level deep in siblings (e.g., sibling/Stars2Cells_Results)
                    try:
                        for sub in sibling.iterdir():
                            if sub.is_dir() and looks_like_stars2cells_project(sub):
                                print(f"[SCAN] ✓ Found project in sibling subfolder: {sub}")
                                return sub
                    except (PermissionError, OSError):
                        pass
                        
            except (PermissionError, OSError) as e:
                print(f"[SCAN] ⚠️  Cannot access parent directory: {e}")
        
        # 4. Search DOWN (subdirectories) - recursive with depth limit
        print(f"[SCAN] Searching subdirectories...")
        def search_subdirs(directory: Path, depth: int = 0):
            if depth > max_depth:
                return None
            
            try:
                for item in directory.iterdir():
                    if not item.is_dir() or item.name.startswith('.'):
                        continue
                    
                    print(f"[SCAN]   {'  '*depth}Checking: {item.name}")
                    
                    # Check if this subdirectory is the project
                    if looks_like_stars2cells_project(item):
                        print(f"[SCAN] ✓ Found project in subdirectory: {item}")
                        return item
                    
                    # Recurse deeper
                    found = search_subdirs(item, depth + 1)
                    if found:
                        return found
            except (PermissionError, OSError):
                pass
            
            return None
        
        found = search_subdirs(start_dir)
        if found:
            return found
        
        print(f"[SCAN] ✗ No Stars2Cells project found")
        return None
    
    # Find the project root
    project_root = scan_for_project_root(folder)
    
    if project_root is None:
        print(f"⚠️  No Stars2Cells project found near {folder}")
        return results
    
    print(f"\n✓ Found Stars2Cells project at: {project_root}")
    results['project_root'] = project_root
    
    # Now scan each step directory for results
    step_configs = {
        'step_1': {
            'folder': 'step_1_results',
            'patterns': ['*_centroids_quads.npz']
        },
        'step_1_5': {
            'folder': 'step_1_5_results',
            'patterns': ['*_threshold_calibration.npz', 'all_animals_summary.json']
        },
        'step_2': {
            'folder': 'step_2_results',
            'patterns': ['*_matches_light.npz']
        },
        'step_2_5': {
            'folder': 'step_2_5_results',
            'patterns': ['*_filtered_matches.npz', 'all_pairs_summary.json']
        },
        'step_3': {
            'folders': ['step_3_results'], 
            'patterns': ['*_final_matches.npz', 'all_animals_summary.json'] 
        }
    }

    print(f"\nScanning for step results...")
    for step_name, config in step_configs.items():
        # Handle step 3 special case with multiple possible folder names
        if step_name == 'step_3':
            step_dir = None
            for folder_name in config['folders']:
                candidate = project_root / folder_name
                if candidate.exists():
                    step_dir = candidate
                    break
            if step_dir is None:
                print(f"  ✗ {step_name}: Directory not found (tried {config['folders']})")
                continue
        else:
            step_dir = project_root / config['folder']
            if not step_dir.exists():
                print(f"  ✗ {step_name}: Directory not found ({config['folder']})")
                continue
        
        # Look for files matching any of the patterns
        found_files = []
        for pattern in config['patterns']:
            found_files.extend(step_dir.glob(pattern))
        
        if found_files:
            results[step_name] = {
                'found': True,
                'path': step_dir,
                'files': sorted(set(found_files))
            }
            print(f"  ✓ {step_name}: {len(results[step_name]['files'])} files in {step_dir.name}")
        else:
            print(f"  ✗ {step_name}: No matching files in {step_dir.name}")
            
    return results
