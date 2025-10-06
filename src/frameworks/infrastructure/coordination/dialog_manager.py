import logging
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QDialogButtonBox, 
    QHBoxLayout, QFileDialog, QMessageBox, QProgressDialog, QWidget
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QObject
from PyQt6.QtGui import QIcon


class DialogResult:    
    def __init__(self, accepted: bool = False, data: Any = None, dialog_type: str = "unknown"):
        self.accepted = accepted
        self.data = data  
        self.dialog_type = dialog_type


class DialogManager(QObject):
    dialog_completed = pyqtSignal(object)  
    progress_updated = pyqtSignal(int, str)  
    
    def __init__(self, parent_window: Optional[QWidget] = None):
        super().__init__()
        self._parent_window = parent_window
        self._logger = logging.getLogger(__name__)
        self._current_progress_dialog: Optional[QProgressDialog] = None
    
    def set_parent_window(self, parent_window: QWidget) -> None:
        self._parent_window = parent_window
        self._logger.debug("Parent window set for dialogs")
    
    def show_file_selection_dialog(self) -> DialogResult:
        try:
            self._logger.debug("Creating file selection dialog (original implementation)...")
            
            from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, 
                                       QLabel, QDialogButtonBox, QHBoxLayout)
            from PyQt6.QtCore import Qt, QSize
            from PyQt6.QtGui import QIcon
            
            dialog = QDialog(self._parent_window)
            dialog.setWindowTitle("Upload Medical Image")
            dialog.setModal(True)
            dialog.resize(340, 140)
            
            layout = QVBoxLayout()
            
            title = QLabel("Select the type of load:")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
            layout.addWidget(title)
            
            single_btn = QPushButton("      Individual DICOM File")
            single_btn.setIcon(QIcon("resources/icons/upload_file.svg"))
            single_btn.setIconSize(QSize(24, 24))
            single_btn.setToolTip("Select a single DICOM file")
            single_btn.clicked.connect(lambda: self._set_selection_result(dialog, "single_file"))
            
            folder_btn = QPushButton("      Folder with DICOM series")
            folder_btn.setIcon(QIcon("resources/icons/upload_folder.svg"))
            folder_btn.setIconSize(QSize(24, 24))
            folder_btn.setToolTip("Select folder containing a complete DICOM series")
            folder_btn.clicked.connect(lambda: self._set_selection_result(dialog, "folder"))
            
            button_style = """
                QPushButton {
                    text-align: left;
                    padding: 10px;
                    margin: 5px;
                    border: 1px solid #555;
                    border-radius: 5px;
                    background-color: #2b2b2b;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #3b3b3b;
                    border-color: #777;
                }
            """
            
            for btn in [single_btn, folder_btn]:
                btn.setStyleSheet(button_style)
                layout.addWidget(btn)
            
            cancel_btn = QPushButton("Cancelar")
            cancel_btn.setFixedWidth(150)
            cancel_btn.setStyleSheet("text-align: center;")
            cancel_btn.clicked.connect(dialog.reject)
            
            h_layout = QHBoxLayout()
            h_layout.addStretch()
            h_layout.addWidget(cancel_btn)
            h_layout.addStretch()
            
            layout.addLayout(h_layout)
            
            dialog.setLayout(layout)
            
            dialog.selection_result = None
            
            self._logger.info("Showing dialog...")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                result = getattr(dialog, 'selection_result', None)
                self._logger.debug(f"Dialog accepted with result: {result}")
                return DialogResult(accepted=True, data=result, dialog_type="file_selection")
            else:
                self._logger.info("Dialog cancelled")
                return DialogResult(accepted=False, dialog_type="file_selection")
                
        except Exception as e:
            self._logger.error(f"Error showing file selection dialog: {e}")
            return DialogResult(accepted=False, dialog_type="file_selection")
    
    
    def _set_selection_result(self, dialog: QDialog, selection: str) -> None:
        self._logger.info(f"User selected: {selection}")
        dialog.selection_result = selection
        dialog.accept()
    
    def show_single_file_dialog(self) -> DialogResult:
        try:
            file_dialog = QFileDialog(self._parent_window)
            file_dialog.setWindowTitle("Select DICOM File")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)            
            file_dialog.setNameFilter(
                "DICOM Files (*.dcm *.dicom *.ima *.IMA);;All Files (*)"
            )
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    file_path = selected_files[0]
                    
                    if self._validate_file_path(file_path):
                        return DialogResult(
                            accepted=True, 
                            data=file_path, 
                            dialog_type="single_file"
                        )
                    else:
                        self.show_error_message(
                            "Invalid File",
                            "The selected file path is invalid or inaccessible."
                        )
            
            return DialogResult(accepted=False, dialog_type="single_file")
            
        except Exception as e:
            self._logger.error(f"Error showing single file dialog: {e}")
            self.show_error_message("Error", f"Error selecting file: {str(e)}")
            return DialogResult(accepted=False, dialog_type="single_file")
    
    def show_folder_dialog(self) -> DialogResult:
        try:
            file_dialog = QFileDialog(self._parent_window)
            file_dialog.setWindowTitle("Select Folder with DICOM series")
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                selected_folders = file_dialog.selectedUrls()
                if selected_folders:
                    folder_path = selected_folders[0].toLocalFile()
                    
                    # SECURITY FIX: Validate folder path
                    if self._validate_folder_path(folder_path):
                        return DialogResult(
                            accepted=True,
                            data=folder_path,
                            dialog_type="folder"
                        )
                    else:
                        self.show_error_message(
                            "Invalid Folder",
                            "The selected folder path is invalid or inaccessible."
                        )
            
            return DialogResult(accepted=False, dialog_type="folder")
            
        except Exception as e:
            self._logger.error(f"Error showing folder dialog: {e}")
            self.show_error_message("Error", f"Error selecting folder: {str(e)}")
            return DialogResult(accepted=False, dialog_type="folder")
    
    def _validate_file_path(self, file_path: str) -> bool:
        try:
            path_obj = Path(file_path).resolve()
            
            if not path_obj.exists():
                self._logger.warning(f"File does not exist: {file_path}")
                return False
            
            if not path_obj.is_file():
                self._logger.warning(f"Path is not a file: {file_path}")
                return False
            
            str_path = str(path_obj).lower()
            forbidden_paths = ['/etc/', '/sys/', '/proc/', 'system32', 'windows/system32']
            
            if any(forbiddin in str_path for forbiddin in forbidden_paths):
                self._logger.warning(f"Access to system path denied: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error validating file path {file_path}: {e}")
            return False
    
    def _validate_folder_path(self, folder_path: str) -> bool:
        try:
            path_obj = Path(folder_path).resolve()
            
            if not path_obj.exists():
                self._logger.warning(f"Folder does not exist: {folder_path}")
                return False
            
            if not path_obj.is_dir():
                self._logger.warning(f"Path is not a directory: {folder_path}")
                return False
            
            str_path = str(path_obj).lower()
            forbidden_paths = ['/etc/', '/sys/', '/proc/', 'system32', 'windows/system32']
            
            if any(forbiddin in str_path for forbiddin in forbidden_paths):
                self._logger.warning(f"Access to system path denied: {folder_path}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error validating folder path {folder_path}: {e}")
            return False
    
    def show_progress_dialog(self, title: str = "Processing", 
                           message: str = "Please wait...",
                           maximum: int = 100,
                           cancelable: bool = True) -> QProgressDialog:
        try:
            if self._current_progress_dialog:
                self._current_progress_dialog.close()
            
            self._current_progress_dialog = QProgressDialog(
                message, "Cancel" if cancelable else "", 0, maximum, self._parent_window
            )
            
            self._current_progress_dialog.setWindowTitle(title)
            self._current_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._current_progress_dialog.setMinimumSize(420, 160)
            self._current_progress_dialog.resize(420, 160)
            
            medical_progress_style = '''
                QProgressDialog {
                    background-color: #1A1A1A;
                    border: 2px solid #606060;
                    border-radius: 8px;
                    color: #E8E8E8;
                    font-family: "Segoe UI", Arial, sans-serif;
                    padding: 12px;
                    border: none;
                }
                
                QProgressDialog QLabel {
                    background-color: transparent;
                    color: #E8E8E8;
                    font-size: 14px;
                    font-weight: 500;
                    padding: 8px 12px;
                    margin: 4px 0px;
                    border: none;
                }
                
                QProgressBar {
                    background-color: #2A2A2A;
                    border: 2px solid #404040;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 12px;
                    color: #E8E8E8;
                    min-height: 28px;
                    max-height: 28px;
                    padding: 2px;
                }
                
                QProgressBar::chunk {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #5A9FE2,
                        stop: 0.3 #4A90E2,
                        stop: 0.7 #3A80D2,
                        stop: 1 #2A70C2
                    );
                    border-radius: 6px;
                    margin: 2px;
                }
                
                QPushButton {
                    background-color: #2A2A2A;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: #E8E8E8;
                    font-weight: 500;
                    font-size: 12px;
                    min-width: 80px;
                    min-height: 24px;
                }
                
                QPushButton:hover {
                    background-color: #4A90E2;
                    border-color: #4A90E2;
                    color: white;
                }
                
                QPushButton:pressed {
                    background-color: #3A80D2;
                    border-color: #3A80D2;
                    color: white;
                }
            '''
            
            self._current_progress_dialog.setStyleSheet(medical_progress_style)
            
            if not cancelable:
                self._current_progress_dialog.setCancelButton(None)
            
            if self._parent_window:
                parent_geometry = self._parent_window.geometry()
                dialog_width = 420
                dialog_height = 160
                x = parent_geometry.x() + (parent_geometry.width() - dialog_width) // 2
                y = parent_geometry.y() + (parent_geometry.height() - dialog_height) // 2
                self._current_progress_dialog.move(x, y)
            
            self._current_progress_dialog.show()
            return self._current_progress_dialog
            
        except Exception as e:
            self._logger.error(f"Error showing progress dialog: {e}")
            return None
    
    def update_progress(self, value: int, message: str = None) -> None:
        try:
            if self._current_progress_dialog and not self._current_progress_dialog.wasCanceled():
                self._current_progress_dialog.setValue(value)
                if message:
                    self._current_progress_dialog.setLabelText(message)
                
                self.progress_updated.emit(value, message or "")
                
        except Exception as e:
            self._logger.error(f"Error updating progress: {e}")
    
    def close_progress_dialog(self) -> None:
        try:
            if self._current_progress_dialog:
                self._current_progress_dialog.close()
                self._current_progress_dialog = None
                
        except Exception as e:
            self._logger.error(f"Error closing progress dialog: {e}")
    
    def show_error_message(self, title: str, message: str) -> None:
        try:
            msg_box = QMessageBox(self._parent_window)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
        except Exception as e:
            self._logger.error(f"Error showing error message: {e}")
    
    def show_warning_message(self, title: str, message: str) -> bool:
        try:
            msg_box = QMessageBox(self._parent_window)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            
            result = msg_box.exec()
            return result == QMessageBox.StandardButton.Ok
            
        except Exception as e:
            self._logger.error(f"Error showing warning message: {e}")
            return False
    
    def show_information_message(self, title: str, message: str) -> None:
        try:
            msg_box = QMessageBox(self._parent_window)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
        except Exception as e:
            self._logger.error(f"Error showing information message: {e}")
    
    def show_confirmation_dialog(self, title: str, message: str) -> bool:
        try:
            msg_box = QMessageBox(self._parent_window)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            result = msg_box.exec()
            return result == QMessageBox.StandardButton.Yes
            
        except Exception as e:
            self._logger.error(f"Error showing confirmation dialog: {e}")
            return False