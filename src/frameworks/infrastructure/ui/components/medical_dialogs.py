import logging
from typing import Optional, List, Dict, Any, Callable
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QProgressBar, QTextEdit, QDialogButtonBox, QWidget, QCheckBox,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPalette

from src.frameworks.infrastructure.ui.themes import theme_service, ComponentType, StyleVariant


class MedicalMessageDialog(QDialog):
    def __init__(
        self,
        title: str,
        message: str,
        dialog_type: str = "info", 
        details: Optional[str] = None,
        buttons: Optional[List[str]] = None,
        default_button: Optional[str] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._dialog_type = dialog_type
        self._details = details
        self._buttons = buttons or self._get_default_buttons(dialog_type)
        self._default_button = default_button
        
        self._setup_dialog(title, message)
        self._apply_medical_styling()
        
        self._center_dialog()
    
    def _get_default_buttons(self, dialog_type: str) -> List[str]:
        button_map = {
            "info": ["OK"],
            "warning": ["OK"], 
            "error": ["OK"],
            "question": ["Yes", "No"],
            "success": ["OK"]
        }
        return button_map.get(dialog_type, ["OK"])
    
    def _get_dialog_icon(self) -> str:
        icon_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "question": "❓",
            "success": "✅"
        }
        return icon_map.get(self._dialog_type, "ℹ️")
    
    def _setup_dialog(self, title: str, message: str) -> None:
        """Set up dialog layout and content."""
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(self._get_dialog_icon())
        icon_label.setFont(QFont("Arial", 24))
        icon_label.setFixedSize(32, 32)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setWordWrap(True)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        message_label = QLabel(message)
        message_label.setFont(QFont("Arial", 10))
        message_label.setWordWrap(True)
        message_label.setMinimumHeight(40)
        layout.addWidget(message_label)
        
        if self._details:
            self._setup_details_section(layout)
        
        self._setup_buttons(layout)
        
        if self._default_button:
            self._set_default_button()
    
    def _setup_details_section(self, layout: QVBoxLayout) -> None:
        self._details_button = QPushButton("Show Details")
        self._details_button.setCheckable(True)
        self._details_button.toggled.connect(self._toggle_details)
        layout.addWidget(self._details_button)
        
        self._details_text = QTextEdit()
        self._details_text.setPlainText(self._details)
        self._details_text.setReadOnly(True)
        self._details_text.setMaximumHeight(120)
        self._details_text.setVisible(False)
        layout.addWidget(self._details_text)
    
    def _setup_buttons(self, layout: QVBoxLayout) -> None:
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._button_widgets = {}
        
        for button_text in self._buttons:
            from .common_controls import MedicalButton
            
            button_type = "default"
            if button_text.lower() in ["ok", "yes", "accept", "save"]:
                button_type = "primary"
            elif button_text.lower() in ["cancel", "no", "reject"]:
                button_type = "default"
            elif button_text.lower() in ["delete", "remove"]:
                button_type = "danger"
            
            button = MedicalButton(
                text=button_text,
                button_type=button_type,
                size="normal"
            )
            
            button.clicked.connect(lambda checked, text=button_text: self._button_clicked(text))
            button_layout.addWidget(button)
            
            self._button_widgets[button_text] = button
        
        layout.addLayout(button_layout)
    
    def _apply_medical_styling(self) -> None:
        dialog_style = theme_service.get_component_style(ComponentType.DIALOG)
        self.setStyleSheet(dialog_style)
        
        if self._dialog_type == "error":
            self.setProperty("dialog-type", "error")
        elif self._dialog_type == "warning":
            self.setProperty("dialog-type", "warning")
        elif self._dialog_type == "success":
            self.setProperty("dialog-type", "success")
    
    def _center_dialog(self) -> None:
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def _toggle_details(self, show: bool) -> None:
        if hasattr(self, '_details_text'):
            self._details_text.setVisible(show)
            self._details_button.setText("Hiof Details" if show else "Show Details")
            
            # Adjust dialog size
            if show:
                self.resize(self.width(), self.height() + 120)
            else:
                self.resize(self.width(), self.height() - 120)
    
    def _set_default_button(self) -> None:
        if self._default_button in self._button_widgets:
            button = self._button_widgets[self._default_button]
            button.setDefault(True)
            button.setFocus()
    
    def _button_clicked(self, button_text: str) -> None:
        if button_text.lower() in ["ok", "yes", "accept", "save"]:
            self.accept()
        else:
            self.reject()
    
    @classmethod
    def show_info(
        cls,
        title: str,
        message: str,
        details: Optional[str] = None,
        parent: Optional[QWidget] = None
    ) -> bool:
        dialog = cls(title, message, "info", details, parent=parent)
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    @classmethod
    def show_warning(
        cls,
        title: str,
        message: str,
        details: Optional[str] = None,
        parent: Optional[QWidget] = None
    ) -> bool:
        dialog = cls(title, message, "warning", details, parent=parent)
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    @classmethod
    def show_error(
        cls,
        title: str,
        message: str,
        details: Optional[str] = None,
        parent: Optional[QWidget] = None
    ) -> bool:
        dialog = cls(title, message, "error", details, parent=parent)
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    @classmethod
    def show_question(
        cls,
        title: str,
        message: str,
        parent: Optional[QWidget] = None
    ) -> bool:
        dialog = cls(title, message, "question", parent=parent)
        return dialog.exec() == QDialog.DialogCode.Accepted


class MedicalProgressDialog(QDialog):
    cancel_requested = pyqtSignal()
    
    def __init__(
        self,
        title: str = "Processing",
        message: str = "Please wait...",
        cancelable: bool = True,
        show_details: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._cancelable = cancelable
        self._show_details = show_details
        self._cancelled = False
        
        self._setup_dialog(title, message)
        self._apply_medical_styling()
        self._center_dialog()
    
    def _setup_dialog(self, title: str, message: str) -> None:
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlags(
            Qt.WindowType.Dialog | 
            Qt.WindowType.WindowTitleHint |
            (Qt.WindowType.WindowSystemMenuHint if self._cancelable else Qt.WindowType.Widget)
        )
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        self._message_label = QLabel(message)
        self._message_label.setFont(QFont("Arial", 10))
        self._message_label.setWordWrap(True)
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._message_label)
        
        from .common_controls import MedicalProgressBar
        self._progress_bar = MedicalProgressBar(
            color="blue",
            show_percentage=True
        )
        layout.addWidget(self._progress_bar)
        
        if self._show_details:
            self._details_text = QTextEdit()
            self._details_text.setReadOnly(True)
            self._details_text.setMaximumHeight(100)
            self._details_text.setFont(QFont("Consolas", 8))
            layout.addWidget(self._details_text)
        
        if self._cancelable:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            from .common_controls import MedicalButton
            self._cancel_button = MedicalButton(
                text="Cancel",
                button_type="default",
                size="normal"
            )
            self._cancel_button.clicked.connect(self._cancel_clicked)
            button_layout.addWidget(self._cancel_button)
            
            layout.addLayout(button_layout)
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.PROGRESS_DIALOG)
        self.setStyleSheet(style)
    
    def _center_dialog(self) -> None:
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def _cancel_clicked(self) -> None:
        self._cancelled = True
        self.cancel_requested.emit()
        self.reject()
    
    def set_progress(self, value: int, message: Optional[str] = None) -> None:
        self._progress_bar.setValue(value)
        
        if message:
            self._message_label.setText(message)
        
        QApplication.processEvents()
    
    def add_detail(self, text: str) -> None:
        if self._show_details and hasattr(self, '_details_text'):
            self._details_text.append(text)
            QApplication.processEvents()
    
    def set_complete(self, message: str = "Complete") -> None:
        self._progress_bar.setValue(100)
        self._message_label.setText(message)
        
        if self._cancelable and hasattr(self, '_cancel_button'):
            self._cancel_button.setText("Close")
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def closeEvent(self, event) -> None:
        if self._cancelable and not self._cancelled:
            self._cancel_clicked()
        event.accept()


class MedicalConfirmDialog(QDialog):
    def __init__(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
        danger_action: bool = False,
        checkbox_text: Optional[str] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._logger = logging.getLogger(__name__)
        self._danger_action = danger_action
        self._checkbox_checked = False
        
        self._setup_dialog(title, message, confirm_text, cancel_text, checkbox_text)
        self._apply_medical_styling()
        self._center_dialog()
    
    def _setup_dialog(
        self,
        title: str,
        message: str,
        confirm_text: str,
        cancel_text: str,
        checkbox_text: Optional[str]
    ) -> None:
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(380)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        content_layout = QHBoxLayout()

        icon = "⚠️" if self._danger_action else "❓"
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setFixedSize(24, 24)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(icon_label)
        
        message_label = QLabel(message)
        message_label.setFont(QFont("Arial", 10))
        message_label.setWordWrap(True)
        content_layout.addWidget(message_label)
        
        layout.addLayout(content_layout)
        
        if checkbox_text:
            self._checkbox = QCheckBox(checkbox_text)
            self._checkbox.setFont(QFont("Arial", 9))
            self._checkbox.toggled.connect(self._checkbox_toggled)
            layout.addWidget(self._checkbox)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        from .common_controls import MedicalButton
        
        self._cancel_button = MedicalButton(
            text=cancel_text,
            button_type="default",
            size="normal"
        )
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)
        
        confirm_type = "danger" if self._danger_action else "primary"
        self._confirm_button = MedicalButton(
            text=confirm_text,
            button_type=confirm_type,
            size="normal"
        )
        self._confirm_button.clicked.connect(self.accept)
        self._confirm_button.setDefault(True)
        button_layout.addWidget(self._confirm_button)
        
        layout.addLayout(button_layout)
        
        if self._danger_action:
            self._cancel_button.setFocus()
        else:
            self._confirm_button.setFocus()
    
    def _apply_medical_styling(self) -> None:
        style = theme_service.get_component_style(ComponentType.CONFIRM_DIALOG)
        self.setStyleSheet(style)
        
        if self._danger_action:
            self.setProperty("dialog-type", "danger")
    
    def _center_dialog(self) -> None:
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def _checkbox_toggled(self, checked: bool) -> None:
        self._checkbox_checked = checked
    
    def is_checkbox_checked(self) -> bool:
        return self._checkbox_checked
    
    @classmethod
    def ask_confirmation(
        cls,
        title: str,
        message: str,
        parent: Optional[QWidget] = None,
        danger: bool = False
    ) -> bool:
        dialog = cls(
            title=title,
            message=message,
            danger_action=danger,
            parent=parent
        )
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    @classmethod
    def ask_delete_confirmation(
        cls,
        item_name: str,
        parent: Optional[QWidget] = None
    ) -> bool:
        return cls.ask_confirmation(
            title="Delete Confirmation",
            message=f"Are you sure you want to delete '{item_name}'?\n\nThis action cannot be undone.",
            parent=parent,
            danger=True
        )