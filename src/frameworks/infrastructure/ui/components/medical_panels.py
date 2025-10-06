import logging
from typing import Optional, List, Dict, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QSizePolicy, QTabWidget, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont

from src.frameworks.infrastructure.ui.themes import theme_service, ComponentType, StyleVariant


class MedicalInfoPanel(QWidget):
    action_clicked = pyqtSignal(str)  # action_name
    
    def __init__(
        self,
        title: str,
        collapsible: bool = False,
        scrollable: bool = False,
        actions: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._title = title
        self._collapsible = collapsible
        self._scrollable = scrollable
        self._actions = actions or []
        
        self._setup_panel()
        self._apply_medical_styling()
    
    def _setup_panel(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        header_frame = QFrame()
        header_frame.setObjectName("info-panel-header")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(6, 4, 6, 4)
        
        self._title_label = QLabel(self._title)
        self._title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(self._title_label)
        
        header_layout.addStretch()
        
        for action in self._actions:
            from .common_controls import MedicalButton
            button = MedicalButton(
                text=action.get("text", action["name"]),
                button_type=action.get("type", "default"),
                size="compact"
            )
            button.clicked.connect(
                lambda checked, name=action["name"]: self.action_clicked.emit(name)
            )
            header_layout.addWidget(button)
        
        layout.addWidget(header_frame)
        
        if self._scrollable:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            self._content_widget = QWidget()
            self._content_layout = QVBoxLayout(self._content_widget)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            
            scroll_area.setWidget(self._content_widget)
            layout.addWidget(scroll_area)
        else:
            self._content_widget = QWidget()
            self._content_layout = QVBoxLayout(self._content_widget)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._content_widget)
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.PANEL)
        self.setStyleSheet(style)
    
    def add_content_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)
    
    def clear_content(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(title)
    
    def add_info_row(self, label: str, value: str) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 2, 0, 2)
        
        label_widget = QLabel(f"{label}:")
        label_widget.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        label_widget.setMinimumWidth(100)
        row_layout.addWidget(label_widget)
        
        value_widget = QLabel(value)
        value_widget.setFont(QFont("Arial", 9))
        value_widget.setWordWrap(True)
        row_layout.addWidget(value_widget)
        
        row_layout.addStretch()
        
        self.add_content_widget(row_widget)


class MedicalControlPanel(QWidget):
    control_changed = pyqtSignal(str, object) 
    
    def __init__(
        self,
        title: str = "",
        orientation: str = "vertical",  
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._title = title
        self._orientation = orientation
        self._control_groups: Dict[str, QGroupBox] = {}
        
        self._setup_panel()
        self._apply_medical_styling()
    
    def _setup_panel(self) -> None:
        if self._orientation == "horizontal":
            self._main_layout = QHBoxLayout(self)
        else:
            self._main_layout = QVBoxLayout(self)
        
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self._main_layout.setSpacing(8)
        
        if self._title:
            title_label = QLabel(self._title)
            title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            self._main_layout.addWidget(title_label)
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.PANEL)
        self.setStyleSheet(style)
    
    def add_control_group(self, group_name: str, title: str) -> QGroupBox:
        group_box = QGroupBox(title)
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(8, 12, 8, 8)
        group_layout.setSpacing(6)
        
        self._control_groups[group_name] = group_box
        self._main_layout.addWidget(group_box)
        
        return group_box
    
    def add_control_to_group(
        self,
        group_name: str,
        control_name: str,
        control_widget: QWidget
    ) -> None:
        if group_name in self._control_groups:
            group = self._control_groups[group_name]
            group.layout().addWidget(control_widget)
            
            if hasattr(control_widget, 'valueChanged'):
                control_widget.valueChanged.connect(
                    lambda value, name=control_name: self.control_changed.emit(name, value)
                )
    
    def get_control_group(self, group_name: str) -> Optional[QGroupBox]:
        return self._control_groups.get(group_name)


class MedicalStatusPanel(QWidget):
    status_clicked = pyqtSignal(str) 
    
    def __init__(
        self,
        show_progress: bool = True,
        show_log: bool = True,
        max_log_entries: int = 100,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._show_progress = show_progress
        self._show_log = show_log
        self._max_log_entries = max_log_entries
        self._log_entries: List[Dict[str, Any]] = []
        
        self._setup_panel()
        self._apply_medical_styling()
    
    def _setup_panel(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        from .common_controls import MedicalProgressBar
        self._status_progress = MedicalProgressBar(show_percentage=False)
        self._status_progress.setVisible(self._show_progress)
        status_layout.addWidget(self._status_progress)
        
        self._status_label = QLabel("Ready")
        self._status_label.setFont(QFont("Arial", 9))
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)
        
        layout.addWidget(status_group)
        
        if self._show_log:
            log_group = QGroupBox("Activity Log")
            log_layout = QVBoxLayout(log_group)
            
            from PyQt6.QtWidgets import QTextEdit
            self._log_text = QTextEdit()
            self._log_text.setReadOnly(True)
            self._log_text.setMaximumHeight(120)
            self._log_text.setFont(QFont("Consolas", 8))
            log_layout.addWidget(self._log_text)
            
            layout.addWidget(log_group)
        
        layout.addStretch()
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.PANEL)
        self.setStyleSheet(style)
    
    def set_status(
        self,
        message: str,
        progress: Optional[int] = None,
        status_type: str = "info"
    ) -> None:
        self._status_label.setText(message)
        
        if self._show_progress and progress is not None:
            self._status_progress.setValue(progress)
            
            color_map = {
                "info": "blue",
                "success": "green",
                "warning": "orange",
                "error": "red"
            }
         
        if self._show_log:
            self.add_log_entry(message, status_type)
    
    def add_log_entry(self, message: str, entry_type: str = "info") -> None:
        if not self._show_log:
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "info": "#333333",
            "success": "#006600",
            "warning": "#CC6600", 
            "error": "#CC0000"
        }
        
        color = color_map.get(entry_type, "#333333")
        html_entry = f'<span style="color: {color};">[{timestamp}] {message}</span>'
        
        self._log_text.append(html_entry)
        
        self._log_entries.append({
            "timestamp": timestamp,
            "message": message,
            "type": entry_type
        })
        
        if len(self._log_entries) > self._max_log_entries:
            self._log_entries.pop(0)
    
    def clear_log(self) -> None:
        if self._show_log:
            self._log_text.clear()
            self._log_entries.clear()
    
    def get_log_entries(self) -> List[Dict[str, Any]]:
        return self._log_entries.copy()


class MedicalTabWidget(QTabWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._setup_tab_widget()
        self._apply_medical_styling()
    
    def _setup_tab_widget(self) -> None:
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setTabsClosable(False)
        self.setMovable(True)        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.TAB_WIDGET)
        self.setStyleSheet(style)
    
    def add_medical_tab(
        self,
        widget: QWidget,
        title: str,
        icon: Optional[str] = None,
        closable: bool = False
    ) -> int:
        if icon:
            tab_text = f"{icon} {title}"
        else:
            tab_text = title
            
        index = self.addTab(widget, tab_text)
        
        if closable:
            self.setTabsClosable(True)
        
        return index
    
    def set_tab_badge(self, index: int, count: int) -> None:
        if count > 0:
            current_text = self.tabText(index)
            if "(" in current_text:
                current_text = current_text.split("(")[0].strip()
            
            new_text = f"{current_text} ({count})"
            self.setTabText(index, new_text)
        else:
            current_text = self.tabText(index)
            if "(" in current_text:
                new_text = current_text.split("(")[0].strip()
                self.setTabText(index, new_text)