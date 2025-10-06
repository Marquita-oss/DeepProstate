"""
infrastructure/ui/widgets/collapsible_sidebar.py

Collapsible Sidebar Menu for Medical Imaging Workstation.
Replaces cluttered right panel with clean, organized menu system.
Each panel can be accessed independently via dropdown menu.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget,
    QLabel, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QEasingCurve, QPropertyAnimation, QRect, pyqtSignal, QSize, QMutex, QMutexLocker
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QFont
from typing import Dict, Optional, List, Tuple, Union
import logging

from src.frameworks.infrastructure.ui.themes import theme_service, ComponentType


class CollapsibleSidebar(QWidget):
    """
    Modern collapsible sidebar menu for medical imaging panels.
    
    Features:
    - Clean toggle button with medical icons
    - Smooth animations for expand/collapse
    - Independent panel access via menu items
    - Responsive design that adapts to content
    - Medical-grade styling with professional appearance
    """
    
    # Signal emitted when a panel is selected
    panel_selected = pyqtSignal(str)  # panel_name
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._logger = logging.getLogger(__name__)
        
        # Thread safety for shared state modifications
        self._state_mutex = QMutex()
        
        # State management
        self._is_expanded = False
        self._current_panel = None
        self._panels: Dict[str, QWidget] = {}
        self._menu_items: Dict[str, QPushButton] = {}
        
        # UI state tracking
        self._ui_state = "icons"  # "icons", "panel" - showing icons initially
        
        # Animation properties
        self._collapsed_width = 50
        self._menu_width = None  # Will be calculated dynamically
        self._panel_width = None  # Will be calculated dynamically
        self._animation_duration = 300
        
        self._setup_ui()
        self._apply_medical_styling()
        
        self._logger.info("CollapsibleSidebar initialized")
    
    def sizeHint(self):
        """Provide size hint to parent layout."""
        from PyQt6.QtCore import QSize
        # Use compact icon width or expanded panel width
        if self._ui_state == "icons":
            width = self._calculate_icon_width()
            height = self._calculate_icon_container_height()
        else:
            width = self._calculate_panel_width()
            height = 500  # Increased height for expanded panels to accommodate more content
        return QSize(width, height)
    
    def minimumSizeHint(self):
        """Provide minimum size hint."""
        from PyQt6.QtCore import QSize
        # Minimum should be based on actual content
        if self._ui_state == "icons":
            return QSize(self._calculate_icon_width(), 50)  # At least one button height
        else:
            return QSize(200, 100)  # Minimum for expanded panels
    
    def _setup_ui(self) -> None:
        """Initialize the always-visible sidebar UI structure."""
        # Use size policy that adapts to content
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        # Default width for always-visible sidebar (will be updated when buttons are added)
        self._desired_width = 250
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create icon buttons area (compact view)
        self._icons_area = QWidget()
        self._icons_area.setObjectName("icons-area")
        self._icons_area.setVisible(True)  # Always visible initially
        self._setup_icons_layout()
        main_layout.addWidget(self._icons_area)
        
        # Create expanded panel area (with header + content)
        self._expanded_area = QWidget()
        self._expanded_area.setObjectName("expanded-area")
        self._expanded_area.setVisible(False)
        self._setup_expanded_layout()
        main_layout.addWidget(self._expanded_area)
        
        # Don't add stretch - let containers size themselves based on content
    
    def _setup_icons_layout(self) -> None:
        """Setup the compact icon layout."""
        # Set size policy to be compact
        self._icons_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Use a vertical layout for compact icon arrangement
        self._icons_layout = QVBoxLayout(self._icons_area)
        self._icons_layout.setContentsMargins(4, 4, 4, 4)
        self._icons_layout.setSpacing(8)  # Increased spacing for better visual separation
        self._icons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    
    def _setup_expanded_layout(self) -> None:
        """Setup the expanded panel layout with header + content."""
        # Set size policy to be more flexible but prefer minimum size
        self._expanded_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Use a vertical layout: header at top, content below
        self._expanded_layout = QVBoxLayout(self._expanded_area)
        self._expanded_layout.setContentsMargins(0, 0, 0, 0)
        self._expanded_layout.setSpacing(0)
        
        # Create header button (shows selected tool name)
        self._panel_header = QPushButton()
        self._panel_header.setProperty("class", "panel-header")
        self._panel_header.setFixedHeight(44)
        self._panel_header.clicked.connect(self._return_to_icons)
        self._expanded_layout.addWidget(self._panel_header)
        
        # Create panel container for content
        self._panel_container = QStackedWidget()
        self._expanded_layout.addWidget(self._panel_container)
    
    
    def _apply_medical_styling(self) -> None:
        """
        REFACTORED: Now uses centralized ThemeService to eliminate styling code duplication.
        Apply medical-grade styling to sidebar components using centralized theme service.
        """
        # Use centralized sidebar styling
        sidebar_style = theme_service.get_component_style(ComponentType.SIDEBAR)
        self.setStyleSheet(sidebar_style)
    
    def add_panel(self, name: str, display_name: str, widget: QWidget, icon: Union[str, QIcon] = "") -> None:
        """
        Add a panel to the sidebar with icon-only display.
        
        Args:
            name: Internal name for the panel
            display_name: Display name for tooltips and header
            widget: The panel widget to show
            icon: Icon/emoji to show in compact view (can be str for emoji or QIcon for SVG/images)
        """
        with QMutexLocker(self._state_mutex):
            # Store panel and display name
            self._panels[name] = widget
            self._panel_container.addWidget(widget)
        
        # Store display name and icon for header (also needs mutex protection)
        with QMutexLocker(self._state_mutex):
            if not hasattr(self, '_panel_names'):
                self._panel_names = {}
            if not hasattr(self, '_panel_icons'):
                self._panel_icons = {}
            self._panel_names[name] = display_name
            # Store the original icon (QIwith or string) for header use
            self._panel_icons[name] = icon
        
        # Create compact icon button
        icon_button = QPushButton()
        if isinstance(icon, QIcon):
            icon_button.setIcon(icon)
            # Set icon size to 60% of button size (24x24 for 40x40 button)
            icon_button.setIconSize(QSize(24, 24))
        else:
            icon_button.setText(str(icon))
        
        icon_button.setProperty("class", "icon-button")
        icon_button.setCheckable(True)
        icon_button.setFixedSize(40, 40)
        icon_button.setToolTip(f"{display_name}")
        icon_button.setCursor(Qt.CursorShape.PointingHandCursor)
        icon_button.setFont(QFont("Arial", 18))
        icon_button.clicked.connect(lambda checked, n=name: self._select_panel(n))
        
        with QMutexLocker(self._state_mutex):
            self._menu_items[name] = icon_button
        
        # Add to icons layout
        self._icons_layout.addWidget(icon_button)
        
        # Update geometry to reflect new layout and size requirements
        self.updateGeometry()
        self._icons_area.updateGeometry()
        
        # Adjust icons area size to fit content
        self._icons_area.setFixedSize(
            self._calculate_icon_width(),
            self._calculate_icon_container_height()
        )
        
        icon_type = "QIcon" if isinstance(icon, QIcon) else f"Text: {icon}"
        self._logger.debug(f"Added panel icon: {name} - {display_name} ({icon_type})")
    
    def _show_panel(self, panel_name: str) -> None:
        """Show specific panel (icons -> expanded panel state)."""
        with QMutexLocker(self._state_mutex):
            self._ui_state = "panel" 
            self._current_panel = panel_name
        
        # Hide icons area, show expanded area
        self._icons_area.setVisible(False)
        self._expanded_area.setVisible(True)
        
        # Update header with tool name and icon
        if hasattr(self, '_panel_names') and panel_name in self._panel_names:
            display_name = self._panel_names[panel_name]
            stored_icon = self._panel_icons.get(panel_name, "")
            
            if isinstance(stored_icon, QIcon):
                # For QIcon: set icon on button and text separately
                self._panel_header.setIcon(stored_icon)
                self._panel_header.setIconSize(QSize(20, 20))  # Smaller icon for header
                self._panel_header.setText(f"  {display_name}")  # Space for icon + text
            else:
                # For emoji/text: clear any icon and show emoji + text
                self._panel_header.setIcon(QIcon())  # Clear icon
                self._panel_header.setText(f"{stored_icon}  {display_name}")
        
        # Show selected panel content
        panel_widget = self._panels[panel_name]
        self._panel_container.setCurrentWidget(panel_widget)
        
        # Update layout geometry for width change with animation
        self._animate_size_change()
        
        self._logger.debug(f"Showing panel: {panel_name}")
    
    def _return_to_icons(self) -> None:
        """Return to icon-only view (panel -> icons state)."""
        with QMutexLocker(self._state_mutex):
            self._ui_state = "icons"
            self._current_panel = None
        
        # Show icons area, hide expanded area
        self._icons_area.setVisible(True)
        self._expanded_area.setVisible(False)
        
        # Clear panel selection
        for button in self._menu_items.values():
            button.setChecked(False)
        
        # Update layout geometry for width change with animation
        self._animate_size_change()
        
        self._logger.debug("Returning to icon view")
    
    def _select_panel(self, panel_name: str) -> None:
        """
        Select and display a specific panel with toggle behavior.
        
        Args:
            panel_name: Name of the panel to display
        """
        with QMutexLocker(self._state_mutex):
            if panel_name not in self._panels:
                self._logger.warning(f"Panel not found: {panel_name}")
                return
            
            # If same panel is selected again, return to icons
            current_panel = self._current_panel
            current_ui_state = self._ui_state
        
        if current_panel == panel_name and current_ui_state == "panel":
            self._return_to_icons()
            return
        
        # Show the selected panel
        self._show_panel(panel_name)
        
        # Update menu button states - only selected one is checked
        for name, button in self._menu_items.items():
            button.setChecked(name == panel_name)
        
        # Emit signal
        self.panel_selected.emit(panel_name)
        
        self._logger.debug(f"Selected panel: {panel_name}")
    
    
    def get_current_panel(self) -> Optional[str]:
        """Get the currently selected panel name."""
        with QMutexLocker(self._state_mutex):
            return self._current_panel
    
    def get_ui_state(self) -> str:
        """Get the current UI state for debugging."""
        with QMutexLocker(self._state_mutex):
            return self._ui_state
    
    def expand_and_select_panel(self, panel_name: str) -> None:
        """
        Select specific panel (sidebar is always expanded).
        
        Args:
            panel_name: Panel to select
        """
        # Sidebar is always expanded, just select panel
        self._select_panel(panel_name)
    
    def get_panel_names(self) -> List[str]:
        """Get list of all available panel names."""
        with QMutexLocker(self._state_mutex):
            return list(self._panels.keys())
    
    def remove_panel(self, panel_name: str) -> None:
        """
        Remove a panel from the sidebar.
        
        Args:
            panel_name: Panel to remove
        """
        with QMutexLocker(self._state_mutex):
            if panel_name not in self._panels:
                return
            
            # Remove from container
            widget = self._panels[panel_name]
            self._panel_container.removeWidget(widget)
            
            # Remove icon button
            if panel_name in self._menu_items:
                button = self._menu_items[panel_name]
                self._icons_layout.removeWidget(button)
                button.deleteLater()
                del self._menu_items[panel_name]

            # Remove from panels dict
            del self._panels[panel_name]
            
            # Update current panel if needed
            if self._current_panel == panel_name:
                self._current_panel = None
        
        self._logger.debug(f"Removed panel: {panel_name}")
    
    def set_panel_visibility(self, panel_name: str, visible: bool) -> None:
        """
        Show or hide a specific panel menu item.
        
        Args:
            panel_name: Panel name
            visible: Whether to show the panel
        """
        with QMutexLocker(self._state_mutex):
            if panel_name in self._menu_items:
                self._menu_items[panel_name].setVisible(visible)
    
    def _calculate_icon_width(self) -> int:
        """Calculate required width for icon-only layout."""
        # Icons are 40px wiof + margins
        icon_width = 40
        margins = 8  # 4px left + 4px right
        return icon_width + margins
    
    def _calculate_icon_container_height(self) -> int:
        """Calculate required height for icon container based on number of buttons."""
        if not self._menu_items:
            return 50  # Minimum height when no buttons
        
        # Calculate height based on number of icon buttons
        button_height = 40  # Each icon button is 40px tall
        button_spacing = 2   # 2px spacing between buttons
        top_bottom_margins = 8  # 4px top + 4px bottom
        
        num_buttons = len(self._menu_items)
        total_height = (button_height * num_buttons) + (button_spacing * (num_buttons - 1)) + top_bottom_margins
        
        return total_height
    
    def _calculate_buttons_width(self) -> int:
        """Calculate required width for buttons area (legacy method)."""
        return self._calculate_icon_width()

    def _calculate_menu_width(self) -> int:
        """Calculate required width for menu buttons (now uses buttons width)."""
        return self._calculate_buttons_width()

    def _calculate_panel_width(self) -> int:
        """Calculate required width for expanded panel view based on content."""
        if not self._current_panel or self._current_panel not in self._panels:
            return 300  # Default minimum width

        # Get the current panel widget
        current_widget = self._panels[self._current_panel]

        # Calculate optimal width based on content
        if hasattr(current_widget, 'sizeHint'):
            content_size = current_widget.sizeHint()
            optimal_width = content_size.width()
        else:
            optimal_width = current_widget.minimumSizeHint().width() if hasattr(current_widget, 'minimumSizeHint') else 300

        # Apply smart width adjustment based on panel type
        if self._current_panel == "ai_analysis":
            # AI Analysis panel needs more space for forms and controls
            optimal_width = max(optimal_width, 380)
        elif self._current_panel == "manual_editing":
            # Manual editing needs moderate space
            optimal_width = max(optimal_width, 320)
        elif self._current_panel == "quantitative":
            # Quantitative analysis needs space for charts and data
            optimal_width = max(optimal_width, 360)
        elif self._current_panel == "image_info":
            # Image Information panel needs space for detailed metadata display
            # Use the panel's own sizeHint() for optimal sizing
            optimal_width = max(optimal_width, 280)  # Minimum, but prefer panel's sizeHint
        else:
            # Default panels get reasonable space
            optimal_width = max(optimal_width, 300)

        # Ensure reasonable bounds (not too narrow, not too wide)
        # Allow Image Information panel to be wider for detailed metadata
        max_width = 600 if self._current_panel == "image_info" else 500
        optimal_width = max(280, min(optimal_width, max_width))
        
        self._logger.debug(f"Calculated panel width for '{self._current_panel}': {optimal_width}px")
        return optimal_width
    
    def _animate_size_change(self):
        """Animate the size change of the sidebar for smooth transitions."""
        # For QSplitter-based layouts, we use updateGeometry() and rely on the parent splitter
        # to handle the smooth resize animation
        
        # Use a timer to ensure the geometry update is processed smoothly
        QTimer.singleShot(50, self._delayed_geometry_update)
    
    def _delayed_geometry_update(self):
        """Delayed geometry update to ensure smooth resize."""
        self.updateGeometry()
        
        # Force parent layout to recalculate
        if self.parent():
            parent = self.parent()
            if hasattr(parent, 'updateGeometry'):
                parent.updateGeometry()
            
            # If parent is a splitter, adjust sizes smoothly
            if hasattr(parent, 'setSizes') and hasattr(parent, 'sizes'):
                # Let the splitter handle the resize naturally
                pass
    
