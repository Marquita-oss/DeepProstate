import logging
from typing import Optional, Callable, Any, List, Union
from PyQt6.QtWidgets import (
    QPushButton, QProgressBar, QLineEdit, QSpinBox, QDoubleSpinBox,
    QSlider, QGroupBox, QLabel, QVBoxLayout, QHBoxLayout, QComboBox,
    QCheckBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QPixmap

from deepprostate.frameworks.infrastructure.ui.themes import theme_service, ComponentType, StyleVariant


class MedicalButton(QPushButton):
    _logger: logging.Logger
    _button_type: str
    _size: str
    _requires_confirmation: bool
    _confirmation_text: str
    
    def __init__(
        self,
        text: str = "",
        icon: Optional[Union[QIcon, str]] = None,
        button_type: str = "default",
        size: str = "normal",
        requires_confirmation: bool = False,
        confirmation_text: str = "Are you sure?",
        parent: Optional[QWidget] = None
    ):
        super().__init__(text, parent)
        
        self._logger = logging.getLogger(__name__)
        self._button_type = button_type
        self._size = size
        self._requires_confirmation = requires_confirmation
        self._confirmation_text = confirmation_text
        
        self._setup_button(icon)
        self._apply_medical_styling()
        
        if requires_confirmation:
            self.clicked.connect(self._handle_confirmation_click)
    
    def _setup_button(self, icon: Optional[Union[QIcon, str]]) -> None:
        if icon:
            if isinstance(icon, str):
                self.setText(f"{icon} {self.text()}".strip())
            else:
                self.setIcon(icon)
                if self._size == "compact":
                    self.setIconSize(QSize(16, 16))
                elif self._size == "large":
                    self.setIconSize(QSize(24, 24))
                else:
                    self.setIconSize(QSize(20, 20))
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)        
        self.setEnabled(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        
    
    def _apply_medical_styling(self) -> None:
        try:
            style = theme_service.get_medical_button_style(self._button_type, self._size)
            self.setStyleSheet(style)
        except Exception as e:
            self._logger.error(f"Failed to apply medical styling: {e}")
            fallback_style = """
                QPushButton {
                    background-color: #f0f0f0;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #4A90E2;
                    border-color: #357abd;
                    color: red;
                }
                QPushButton:pressed {
                    background-color: #357abd;
                    border-color: #2868a3;
                    color: white;
                }
            """
            self.setStyleSheet(fallback_style)
        
    def _handle_confirmation_click(self) -> None:
        from .medical_dialogs import MedicalConfirmDialog
        
        dialog = MedicalConfirmDialog(
            title="Confirmation",
            message=self._confirmation_text,
            parent=self.parent()
        )
        
        if dialog.exec():
            self.clicked.disconnect(self._handle_confirmation_click)
            self.clicked.emit(True)
            self.clicked.connect(self._handle_confirmation_click)
    
    def set_medical_state(self, state: str) -> None:
        if state == "processing":
            self.setEnabled(False)
            self.setText("Processing...")
        elif state == "success":
            self.setEnabled(True)
        elif state == "error":
            self.setEnabled(True)
        else: 
            self.setEnabled(True)


class MedicalDeleteButton(MedicalButton):    
    def __init__(
        self,
        size: str = "small",
        confirmation_text: str = "Are you sure you want to delete this item?",
        parent: Optional[QWidget] = None
    ):
        """Initialize medical delete button."""
        super().__init__(
            text="×",
            button_type="danger",
            size=size,
            requires_confirmation=True,
            confirmation_text=confirmation_text,
            parent=parent
        )
        
        delete_style = theme_service.get_component_style(ComponentType.DELETE_BUTTON)
        self.setStyleSheet(delete_style)
        
        size_map = {"small": 20, "medium": 28, "large": 36}
        button_size = size_map.get(size, 20)
        self.setFixedSize(button_size, button_size)


class MedicalProgressBar(QProgressBar):    
    def __init__(
        self,
        color: str = "blue",
        show_percentage: bool = True,
        show_eta: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._color = color
        self._show_percentage = show_percentage
        self._show_eta = show_eta
        self._start_time = None
        
        self._setup_progress_bar()
        self._apply_medical_styling()
    
    def _setup_progress_bar(self) -> None:
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        
        if not self._show_percentage:
            self.setTextVisible(False)
    
    def _apply_medical_styling(self) -> None:
        if self._color == "blue":
            variant = StyleVariant.DEFAULT
        elif self._color == "green":
            variant = StyleVariant.SUCCESS
        elif self._color == "orange":
            variant = StyleVariant.WARNING
        elif self._color == "red":
            variant = StyleVariant.DANGER
        else:
            variant = StyleVariant.DEFAULT
        
        style = theme_service.get_component_style(ComponentType.PROGRESS_BAR, variant)
        self.setStyleSheet(style)
    
    def set_medical_progress(self, value: int, message: str = "") -> None:
        self.setValue(value)
        
        if message and self._show_percentage:
            self.setFormat(f"{value}% - {message}")
        elif message:
            self.setFormat(message)
    
    def start_operation(self, operation_name: str = "Operation") -> None:
        import time
        self._start_time = time.time()
        self.setValue(0)
        self.setFormat(f"{operation_name}...")
    
    def complete_operation(self, success_message: str = "Complete") -> None:
        self.setValue(100)
        self.setFormat(success_message)


class MedicalInputField(QWidget):
    
    valueChanged = pyqtSignal(object)
    
    def __init__(
        self,
        label: str,
        input_type: str = "text", 
        default_value: Any = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        options: Optional[List[str]] = None,
        validator: Optional[Callable] = None,
        required: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._label_text = label
        self._input_type = input_type
        self._validator = validator
        self._required = required
        
        self._setup_layout()
        self._create_input_widget(default_value, min_value, max_value, options)
        self._apply_medical_styling()
    
    def _setup_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Create label
        self._label = QLabel(self._label_text)
        if self._required:
            self._label.setText(f"{self._label_text} *")
        layout.addWidget(self._label)
    
    def _create_input_widget(
        self,
        default_value: Any,
        min_value: Optional[float],
        max_value: Optional[float],
        options: Optional[List[str]]
    ) -> None:
        if self._input_type == "text":
            self._input_widget = QLineEdit()
            if default_value:
                self._input_widget.setText(str(default_value))
            self._input_widget.textChanged.connect(lambda x: self.valueChanged.emit(x))
            
        elif self._input_type == "int":
            self._input_widget = QSpinBox()
            if min_value is not None:
                self._input_widget.setMinimum(int(min_value))
            if max_value is not None:
                self._input_widget.setMaximum(int(max_value))
            if default_value is not None:
                self._input_widget.setValue(int(default_value))
            self._input_widget.valueChanged.connect(lambda x: self.valueChanged.emit(x))
            
        elif self._input_type == "float":
            self._input_widget = QDoubleSpinBox()
            if min_value is not None:
                self._input_widget.setMinimum(min_value)
            if max_value is not None:
                self._input_widget.setMaximum(max_value)
            if default_value is not None:
                self._input_widget.setValue(float(default_value))
            self._input_widget.valueChanged.connect(lambda x: self.valueChanged.emit(x))
            
        elif self._input_type == "combo":
            self._input_widget = QComboBox()
            if options:
                self._input_widget.addItems(options)
            if default_value and default_value in (options or []):
                self._input_widget.setCurrentText(str(default_value))
            self._input_widget.currentTextChanged.connect(lambda x: self.valueChanged.emit(x))
            
        self.layout().addWidget(self._input_widget)
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_medical_input_style()
        self.setStyleSheet(style)
    
    def get_value(self) -> Any:
        if self._input_type == "text":
            return self._input_widget.text()
        elif self._input_type in ["int", "float"]:
            return self._input_widget.value()
        elif self._input_type == "combo":
            return self._input_widget.currentText()
    
    def set_value(self, value: Any) -> None:
        if self._input_type == "text":
            self._input_widget.setText(str(value))
        elif self._input_type in ["int", "float"]:
            self._input_widget.setValue(value)
        elif self._input_type == "combo":
            self._input_widget.setCurrentText(str(value))
    
    def validate(self) -> tuple[bool, str]:
        value = self.get_value()
        
        if self._required and not value:
            return False, f"{self._label_text} is required"
        
        if self._validator:
            try:
                result = self._validator(value)
                if isinstance(result, bool):
                    return result, "" if result else f"Invalid {self._label_text}"
                elif isinstance(result, tuple):
                    return result
            except Exception as e:
                return False, str(e)
        
        return True, ""


class MedicalSlider(QWidget):    
    valueChanged = pyqtSignal(float)
    
    def __init__(
        self,
        label: str,
        min_value: float = 0.0,
        max_value: float = 100.0,
        default_value: float = 50.0,
        decimals: int = 0,
        show_value: bool = True,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._label_text = label
        self._min_value = min_value
        self._max_value = max_value
        self._decimals = decimals
        self._multiplier = 10 ** decimals
        
        self._setup_layout(orientation, show_value)
        self._setup_slider(default_value)
        self._apply_medical_styling()
    
    def _setup_layout(self, orientation: Qt.Orientation, show_value: bool) -> None:
        if orientation == Qt.Orientation.Horizontal:
            layout = QVBoxLayout(self)
            
            # Label row
            label_layout = QHBoxLayout()
            self._label = QLabel(self._label_text)
            label_layout.addWidget(self._label)
            
            if show_value:
                self._value_label = QLabel()
                label_layout.addWidget(self._value_label)
            
            layout.addLayout(label_layout)
        else:
            layout = QHBoxLayout(self)
            self._label = QLabel(self._label_text)
            layout.addWidget(self._label)
            
            if show_value:
                self._value_label = QLabel()
                layout.addWidget(self._value_label)
    
    def _setup_slider(self, default_value: float) -> None:
        self._slider = QSlider()
        self._slider.setMinimum(int(self._min_value * self._multiplier))
        self._slider.setMaximum(int(self._max_value * self._multiplier))
        self._slider.setValue(int(default_value * self._multiplier))
        
        self._slider.valueChanged.connect(self._on_slider_changed)
        self.layout().addWidget(self._slider)
        
        self._update_value_display(default_value)
    
    def _apply_medical_styling(self) -> None:
        pass
    
    def _on_slider_changed(self, value: int) -> None:
        real_value = value / self._multiplier
        self._update_value_display(real_value)
        self.valueChanged.emit(real_value)
    
    def _update_value_display(self, value: float) -> None:
        if hasattr(self, '_value_label'):
            if self._decimals == 0:
                self._value_label.setText(f"{int(value)}")
            else:
                self._value_label.setText(f"{value:.{self._decimals}f}")
    
    def get_value(self) -> float:
        return self._slider.value() / self._multiplier
    
    def set_value(self, value: float) -> None:
        self._slider.setValue(int(value * self._multiplier))


class MedicalGroupBox(QGroupBox):    
    def __init__(
        self,
        title: str,
        collapsible: bool = False,
        collapsed: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(title, parent)
        
        self._collapsible = collapsible
        self._collapsed = collapsed
        
        self._setup_group_box()
        self._apply_medical_styling()
    
    def _setup_group_box(self) -> None:
        if self._collapsible:
            self.setCheckable(True)
            self.setChecked(not self._collapsed)
            self.toggled.connect(self._on_toggle)
    
    def _apply_medical_styling(self) -> None:
        pass
    
    def _on_toggle(self, checked: bool) -> None:
        for child in self.findChildren(QWidget):
            child.setVisible(checked)
    
    def add_medical_content(self, widget: QWidget) -> None:
        if not self.layout():
            layout = QVBoxLayout(self)
            layout.setContentsMargins(10, 20, 10, 10)
            layout.setSpacing(8)
        
        self.layout().addWidget(widget)