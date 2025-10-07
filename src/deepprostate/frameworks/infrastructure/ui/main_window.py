import logging
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QStatusBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QCloseEvent

from deepprostate.frameworks.infrastructure.di.medical_service_container import MedicalServiceContainer
from deepprostate.frameworks.infrastructure.coordination.workflow_orchestrator import WorkflowOrchestrator
from deepprostate.core.domain.utils.universal_mask_visualizer import UniversalMaskVisualizer
from deepprostate.frameworks.infrastructure.data.unified_data_loader import UnifiedDataLoader
from deepprostate.frameworks.infrastructure.ui.widgets.collapsible_sidebar import CollapsibleSidebar
from deepprostate.frameworks.infrastructure.ui.widgets import ImageViewer2D
from deepprostate.frameworks.infrastructure.ui.widgets import PatientBrowserPanel
from deepprostate.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository
from deepprostate.core.domain.entities.medical_image import MedicalImage


class SimpleUIComponents:
    def __init__(self):
        self.central_viewer = None
        self.right_panel = None
        self.left_panel = None
        self.patient_browser = None
        self.segmentation_panel = None


class MedicalMainWindow(QMainWindow):
    application_closing = pyqtSignal()
    patient_context_changed = pyqtSignal(str)
    
    def __init__(
        self,
        service_container: MedicalServiceContainer,
        workflow_coordinator: WorkflowOrchestrator
    ):
        super().__init__()
        self.hide()

        self._services = service_container
        self._coordinator = workflow_coordinator
        self._logger = logging.getLogger(__name__)
        self._ui_components = self._create_child_widgets()

        self._current_patient_id: Optional[str] = None
        self._current_image: Optional[MedicalImage] = None
        self._application_state = "idle"
        self._workflow_history: list = []
        self._segmentations_by_series: Dict[str, list] = {}

        self.mask_visualizer = UniversalMaskVisualizer()
        self.data_loader = UnifiedDataLoader()

        self._setup_main_window()
        self._setup_menu_system()
        self._setup_status_system()
        self._setup_high_level_coordination()
        

    def _create_child_widgets(self) -> SimpleUIComponents:
        components = SimpleUIComponents()

        components.central_viewer = ImageViewer2D()
        components.central_viewer.setParent(self)

        components.right_panel = CollapsibleSidebar()
        components.right_panel.setParent(self)
        components.segmentation_panel = components.right_panel

        dicom_repo = DICOMImageRepository("./medical_data/dicom_storage")
        components.left_panel = PatientBrowserPanel(dicom_repo)
        components.left_panel.setParent(self)
        components.patient_browser = components.left_panel

        self._setup_collapsible_sidebar(components.right_panel, components.patient_browser, components)

        return components

    def _setup_collapsible_sidebar(self, sidebar: CollapsibleSidebar, patient_browser, components):
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PyQt6.QtGui import QIcon
        from deepprostate.frameworks.infrastructure.ui.widgets import AIAnalysisPanel
        from deepprostate.use_cases.application.services.ai_analysis_orchestrator import AIAnalysisOrchestrator

        ai_orchestrator = self._services.ai_analysis_orchestrator
        dynamic_config = self._services.dynamic_model_config_service

        ai_panel = AIAnalysisPanel(ai_orchestrator, dynamic_config)
        ai_panel.setParent(self)
        ai_panel.set_patient_browser(patient_browser)
        ai_panel.analysis_completed.connect(self._on_ai_analysis_completed)
        ai_panel.overlay_visibility_changed.connect(self._on_ai_overlay_visibility_changed)

        from deepprostate.frameworks.infrastructure.ui.widgets.manual_editing_panel import ManualEditingPanel
        manual_panel = ManualEditingPanel()
        manual_panel.setParent(self)

        from deepprostate.frameworks.infrastructure.ui.widgets.quantitative_analysis_panel import QuantitativeAnalysisPanel
        quant_panel = QuantitativeAnalysisPanel()
        quant_panel.setParent(self)

        from deepprostate.frameworks.infrastructure.ui.widgets.image_information_panel import ImageInformationPanel
        image_info_panel = ImageInformationPanel()
        image_info_panel.setParent(self)

        sidebar.image_information_panel = image_info_panel

        sidebar.add_panel("ai_analysis", "AI Analysis", ai_panel, QIcon("src/resources/icons/ai_analysis.svg"))
        sidebar.add_panel("manual_editing", "Manual Editing", manual_panel, QIcon("src/resources/icons/manual_editing.svg"))
        sidebar.add_panel("quantitative", "Quantitative Analysis", quant_panel, QIcon("src/resources/icons/quantitative_analysis.svg"))
        sidebar.add_panel("image_info", "Image Information", image_info_panel, QIcon("src/resources/icons/image_information.svg"))

        if hasattr(quant_panel, 'connect_to_mask_selector'):
            quant_panel.connect_to_mask_selector(components.central_viewer)

        if hasattr(quant_panel, 'connect_to_image_viewer') and hasattr(components, 'central_viewer'):
            quant_panel.connect_to_image_viewer(components.central_viewer)

    def setup_image_information_connections(self) -> None:
        if hasattr(self, 'image_information_panel') and self.image_information_panel:
            self._coordinator.image_loaded.connect(self.image_information_panel.update_image_information)
        else:
            self._logger.info("Image Information panel not found - connection skipped")
    
    def _setup_main_window(self) -> None:
        self.setWindowTitle("Deep Prostate")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        self._create_main_layout()
        self._apply_medical_window_theme()
    
    def _create_main_layout(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        if self._ui_components.left_panel:
            main_splitter.addWidget(self._ui_components.left_panel)

        if self._ui_components.central_viewer:
            main_splitter.addWidget(self._ui_components.central_viewer)

        if self._ui_components.right_panel:
            main_splitter.addWidget(self._ui_components.right_panel)

        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

        if self._ui_components.left_panel:
            self._ui_components.left_panel.setMinimumWidth(250)
            self._ui_components.left_panel.setMaximumWidth(400)

        if self._ui_components.central_viewer:
            self._ui_components.central_viewer.setMinimumWidth(400)
    
    def _setup_menu_system(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        open_dicom_action = QAction("Opin &DICOM File(s)...", self)
        open_dicom_action.setShortcut("Ctrl+O")
        open_dicom_action.setToolTip("Opin single or multiple DICOM files (supports multi-modal studies)")
        open_dicom_action.triggered.connect(self._on_open_dicom_file)
        file_menu.addAction(open_dicom_action)

        open_study_action = QAction("Opin Multi-Modal &Study Directory...", self)
        open_study_action.setShortcut("Ctrl+Shift+O")
        open_study_action.setToolTip("Opin directory containing multi-modal study (CT, MRI sequences, etc.)")
        open_study_action.triggered.connect(self._on_open_medical_study)
        file_menu.addAction(open_study_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        analysis_menu = menubar.addMenu("&Analysis")

        full_ai_analysis_action = QAction("&Run Full AI Analysis", self)
        full_ai_analysis_action.setShortcut("Ctrl+A")
        full_ai_analysis_action.triggered.connect(self._on_run_full_ai_analysis)
        analysis_menu.addAction(full_ai_analysis_action)

        view_menu = menubar.addMenu("&View")

        dev_view_action = QAction("&Development View", self)
        dev_view_action.setCheckable(True)
        dev_view_action.triggered.connect(self._on_toggle_development_view)
        view_menu.addAction(dev_view_action)

        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_system(self) -> None:
        status_bar = self.statusBar()

        self._status_update_timer = QTimer()
        self._status_update_timer.timeout.connect(self._update_high_level_status)
        self._status_update_timer.start(2000)

        self._update_high_level_status()
    
    def _setup_high_level_coordination(self) -> None:
        self._coordinator.workflow_completed.connect(self._on_workflow_completed)
        self._coordinator.workflow_error.connect(self._on_workflow_error)
        self._coordinator.workflow_progress.connect(self._on_workflow_progress)
        self._coordinator.medical_validation_required.connect(self._on_medical_validation_required)
        self._coordinator.image_loaded.connect(self._on_image_loaded_strategically)

        self._ui_components.central_viewer.current_image_changed.connect(self._on_current_image_changed_for_metrics)

        if hasattr(self._ui_components, 'right_panel'):
            QTimer.singleShot(2000, self._send_current_image_to_quantitative_panel)

        if self._ui_components.patient_browser:
            self._ui_components.patient_browser.patient_changed.connect(self._on_patient_context_changed)

        if self._ui_components.patient_browser:
            self._ui_components.patient_browser.image_selected.connect(self._on_image_selected_from_browser)
            self._ui_components.patient_browser.cached_image_selected.connect(self._on_cached_image_selected_from_browser)

        self._setup_segmentation_synchronization()
        

    def _apply_medical_window_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
            }
            QMenuBar::item:selected {
                background-color: palette(highlight);
            }
        """)

    def _on_open_medical_study(self) -> None:
        try:
            workflow_id = self._coordinator.start_image_loading_workflow()
        except Exception as e:
            self._logger.error(f"Error starting interactive study loading: {e}")
            workflow_id = "error"

        self._workflow_history.append({
            'action': 'open_study_requested',
            'timestamp': datetime.now(),
            'workflow_id': workflow_id,
            'user_initiated': True
        })

        self._application_state = "loading_study"
        self._update_high_level_status()

    def _on_open_dicom_file(self) -> None:
        try:
            workflow_id = self._coordinator.start_image_loading_workflow()
        except Exception as e:
            self._logger.error(f"Error starting interactive image loading: {e}")
            workflow_id = "error"

        self._workflow_history.append({
            'action': 'open_dicom_file_requested',
            'timestamp': datetime.now(),
            'workflow_id': workflow_id,
            'user_initiated': True
        })

        self._application_state = "loading_dicom_file"
        self._update_high_level_status()

    def _on_run_full_ai_analysis(self) -> None:
        if not self._current_image:
            self._show_user_message(
                "No Image Loaded",
                "Please load a medical image before running AI analysis.",
                "warning"
            )
            return

        if not self._validate_ai_analysis_prerequisites():
            return

        workflow_id = self._coordinator.start_ai_analysis_workflow(
            self._current_image,
            analysis_type="full"
        )

        self._workflow_history.append({
            'action': 'ai_analysis_requested',
            'timestamp': datetime.now(),
            'workflow_id': workflow_id,
            'image_id': self._current_image.series_instance_uid if self._current_image else None,
            'analysis_type': 'full'
        })

        self._application_state = "running_ai_analysis"
        self._update_high_level_status()

    def _on_toggle_development_view(self, checked: bool) -> None:
        if checked:
            self._show_development_info()
        else:
            self._hide_development_info()

    def _on_show_about(self) -> None:
        from PyQt6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QLabel
        from PyQt6.QtGui import QPixmap, QFont
        from PyQt6.QtSvg import QSvgRenderer
        from PyQt6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("About Deep Prostate")
        dialog.setFixedSize(580, 300)
        dialog.setModal(True)

        dialog.move(
            self.x() + (self.width() - dialog.width()) // 2,
            self.y() + (self.height() - dialog.height()) // 2
        )

        main_layout = QHBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        logo_label = QLabel()
        logo_label.setFixedSize(190, 190)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        import os
        logo_path = "src/resources/image/logo2.svg"

        if Path(logo_path).exists():
            try:
                svg_renderer = QSvgRenderer(logo_path)

                if svg_renderer.isValid():
                    logo_size = 190
                    pixmap = QPixmap(logo_size, logo_size)
                    pixmap.fill(Qt.GlobalColor.transparent)

                    from PyQt6.QtGui import QPainter
                    from PyQt6.QtCore import QRectF
                    painter = QPainter(pixmap)
                    svg_renderer.render(painter, QRectF(0, 0, logo_size, logo_size))
                    painter.end()

                    logo_label.setPixmap(pixmap)
                else:
                    logo_label.setText("\nDP")
                    logo_label.setStyleSheet("""
                        QLabel {
                            font-size: 20pt;
                            font-weight: bold;
                            color: #2980b9;
                            text-align: center;
                            border: 2px solid #2980b9;
                            border-radius: 50px;
                            background-color: #ecf0f1;
                        }
                    """)
            except Exception as e:
                logo_label.setText("\nDP")
                logo_label.setStyleSheet("""
                    QLabel {
                        font-size: 20pt;
                        font-weight: bold;
                        color: #2980b9;
                        text-align: center;
                        border: 2px solid #2980b9;
                        border-radius: 50px;
                        background-color: #ecf0f1;
                    }
                """)
        else:
            logo_label.setText("\nDP")
            logo_label.setStyleSheet("""
                QLabel {
                    font-size: 20pt;
                    font-weight: bold;
                    color: #2980b9;
                    text-align: center;
                    border: 2px solid #2980b9;
                    border-radius: 50px;
                    background-color: #ecf0f1;
                }
            """)

        text_widget = QLabel()
        text_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        text_widget.setWordWrap(True)

        formatted_text = """
        <div style='font-family: Arial, sans-serif;'>
        <h2 style='margin: 0 0 10px 0; color: #2980b9; font-size: 18pt;'>Deep Prostate v2.0</h2>
        <p style='margin: 0 0 8px 0; font-style: italic; font-size: 12pt; color: #34495e;'>Sistema of Análisis of Próstata with IA</p>
        <p style='margin: 0 0 15px 0; font-size: 10pt; color: #7f8c8d;'>desarrollado by Ronald Marca (rnldmarca@gmail.com)</p>

        <h4 style='margin: 0 0 8px 0; color: #27ae60; font-size: 12pt;'>Características:</h4>
        <ul style='margin: 0 0 15px 0; padding-left: 20px; font-size: 10pt; line-height: 1.4;'>
        <li>Processesmiento of imágenes DICOM</li>
        <li>Segmentación automática with nnU-Net</li>
        <li>Análisis cuantitativo of próstata</li>
        <li>Soporte T2W, ADC, DWI</li>
        </ul>

        <p style='margin: 0; font-style: italic; color: #e74c3c; font-size: 10pt;'>Para investigación y educación médica</p>
        </div>
        """

        text_widget.setText(formatted_text)
        main_layout.addWidget(logo_label, 0)  
        main_layout.addWidget(text_widget, 1) 

        dialog.exec()

    def _on_workflow_completed(self, workflow_id: str, results: Dict[str, Any]) -> None:
        workflow_info = next(
            (w for w in self._workflow_history if w.get('workflow_id') == workflow_id),
            None
        )

        if workflow_info:
            if workflow_info['action'] == 'open_study_requested':
                self._application_state = "study_loaded"
            elif workflow_info['action'] == 'ai_analysis_requested':
                self._application_state = "ai_analysis_completed"

        self._update_high_level_status()

    def _on_workflow_error(self, workflow_id: str, error_message: str) -> None:
        self._logger.error(f"Error in flujo of trabajo {workflow_id}: {error_message}")

        self._show_user_message(
            "Medical Workflow Error",
            f"An error occurred during medical workflow:\n{error_message}\n\n"
            f"Please check the logs for detailed information.",
            "error"
        )

        self._application_state = "error"
        self._update_high_level_status()

    def _on_workflow_progress(self, workflow_id: str, progress: int, message: str) -> None:
        if not self.isVisible():
            return

        if not hasattr(self, '_progress_dialog') or self._progress_dialog is None:
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt

            self._progress_dialog = QProgressDialog(self)
            self._progress_dialog.setWindowTitle("Processing...")
            self._progress_dialog.setModal(True)
            self._progress_dialog.setAutoClose(True)
            self._progress_dialog.setAutoReset(False)
            self._progress_dialog.setCancelButton(None)
            self._progress_dialog.setWindowFlags(
                Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint
            )
            self._progress_dialog.setMinimumWidth(400)

        self._progress_dialog.setLabelText(message)
        self._progress_dialog.setValue(progress)
        self._progress_dialog.setMaximum(100)

        if not self._progress_dialog.isVisible():
            self._progress_dialog.show()

        if progress >= 100:
            QTimer.singleShot(1000, self._close_progress_dialog)

    def _close_progress_dialog(self):
        if hasattr(self, '_progress_dialog') and self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

    def _on_medical_validation_required(self, validation_type: str, data: Dict[str, Any]) -> None:
        self._logger.info(f"Validation médica requerida: {validation_type}")


        if validation_type == "low_quality_ai_results":
            self._handle_low_quality_ai_results(data)
        elif validation_type == "ambiguous_findings":
            self._handle_ambiguous_medical_findings(data)
        else:
            self._handle_generic_medical_validation(validation_type, data)

    def _on_image_loaded_strategically(self, image: MedicalImage) -> None:
        self._current_image = image

        self._application_state = "image_ready"
        self._update_high_level_status()

        if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
            self._ui_components.central_viewer.set_image(image)

        if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
            quantitative_panel = self._ui_components.right_panel._panels.get('quantitative')
            if quantitative_panel and hasattr(quantitative_panel, 'set_image_data'):
                quantitative_panel.set_image_data(image.image_data)

        if hasattr(self._ui_components, 'patient_browser') and self._ui_components.patient_browser:

            if hasattr(self._ui_components.patient_browser, 'add_medical_image_to_cache'):
                self._ui_components.patient_browser.add_medical_image_to_cache(
                    image.series_instance_uid, image
                )

            patient_data = {
                "patients": {
                    image.patient_id: {
                        "patient_name": getattr(image, 'patient_name', image.patient_id),
                        "studies": {
                            image.study_instance_uid: {
                                "study_description": getattr(image, 'study_description', 'Loaded Study'),
                                "modality": image.modality.value if hasattr(image, 'modality') and image.modality else '',
                                "series": {
                                    image.series_instance_uid: {
                                        "series_description": getattr(image, 'series_description', self._generate_fallback_series_description(image)),
                                        "modality": image.modality.value if hasattr(image, 'modality') and image.modality else '',
                                        "images_count": getattr(image, 'number_of_slices', 1),
                                        "has_masks": self._extract_mask_info(image, 'has_associated_masks'),
                                        "masks_count": self._extract_mask_info(image, 'masks_count'),
                                        "associated_masks": self._extract_mask_info(image, 'associated_masks')
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            current_data = getattr(self._ui_components.patient_browser, 'current_patient_data', {"patients": {}})
            
            if not current_data or not current_data.get("patients"):
                self._ui_components.patient_browser.current_patient_data = patient_data
            else:
                patient_id = image.patient_id
                study_uid = image.study_instance_uid
                series_uid = image.series_instance_uid
                
                
                if patient_id not in current_data["patients"]:
                    current_data["patients"][patient_id] = patient_data["patients"][patient_id]
                else:
                    existing_patient = current_data["patients"][patient_id]
                    existing_studies = existing_patient.get("studies", {})
                    
                    if "studies" not in existing_patient:
                        existing_patient["studies"] = {}
                    
                    if study_uid not in existing_patient["studies"]:
                        existing_patient["studies"][study_uid] = patient_data["patients"][patient_id]["studies"][study_uid]
                    else:
                        existing_study = existing_patient["studies"][study_uid]
                        existing_series = existing_study.get("series", {})
                        
                        if "series" not in existing_study:
                            existing_study["series"] = {}
                        
                        new_series_data = patient_data["patients"][patient_id]["studies"][study_uid]["series"][series_uid]
                        existing_study["series"][series_uid] = new_series_data
                
                self._ui_components.patient_browser.current_patient_data = current_data                
                final_patients = len(current_data.get("patients", {}))
                final_studies = sum(len(p.get("studies", {})) for p in current_data["patients"].values())
                final_series = sum(len(s.get("series", {})) for p in current_data["patients"].values() for s in p.get("studies", {}).values())
            
            if hasattr(self._ui_components.patient_browser, 'patient_tree'):
                final_data = getattr(self._ui_components.patient_browser, 'current_patient_data', patient_data)
                self._ui_components.patient_browser.patient_tree.populate_tree(final_data)
                    
        self._update_segmentation_dropdowns_for_current_image(image)
        self._handle_automatic_mask_loading(image)       
        self._workflow_history.append({
            'action': 'image_loaded',
            'timestamp': datetime.now(),
            'image_id': image.series_instance_uid,
            'patient_id': image.patient_id,
            'modality': image.modality.value
        })

    def _on_current_image_changed_for_metrics(self, image: MedicalImage) -> None:
        if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
            quantitative_panel = self._ui_components.right_panel._panels.get('quantitative')
            if quantitative_panel and hasattr(quantitative_panel, 'set_image_data'):
                quantitative_panel.set_image_data(image.image_data)
            else:
                self._logger.error("Quantitative Analysis panel not found or missing set_image_data method")
        else:
            self._logger.error("Right panel or panels not found")

    def _send_current_image_to_quantitative_panel(self) -> None:
        try:
            if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
                current_image = getattr(self._ui_components.central_viewer, '_current_image', None)
                if current_image:
                    if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
                        quantitative_panel = self._ui_components.right_panel._panels.get('quantitative')
                        if quantitative_panel and hasattr(quantitative_panel, 'set_image_data'):
                            quantitative_panel.set_image_data(current_image.image_data)
                        else:
                            self._logger.error("FALLBACK: Quantitative Analysis panel not found")
                    else:
                        self._logger.error("FALLBACK: Right panel not found")
            else:
                self._logger.error("FALLBACK: Central viewer not found")
        except Exception as e:
            self._logger.error(f"FALLBACK: Error sending current image to quantitative panel: {e}")

    def _handle_automatic_mask_loading(self, medical_image):
        try:
            metadata = getattr(medical_image, '_dicom_metadata', {}) or getattr(medical_image, 'dicom_metadata', {})
            
            associated_masks = metadata.get('associated_masks', [])
            
            if not associated_masks:
                return
                        
            for mask_path_str in associated_masks:
                try:
                    mask_path = Path(mask_path_str)                    
                    mask_medical_image = self.data_loader.load_mask(mask_path, reference_image=medical_image)                    
                    mask_overlays = self.mask_visualizer.create_visualization_overlays(
                        mask_array=mask_medical_image.image_data,
                        mask_type=None
                    )
                    
                    if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer and mask_overlays:
                        for i, overlay_data in enumerate(mask_overlays):
                            overlay_id = f"auto_mask_{mask_path.stem}_{i}"
                            
                            from PyQt6.QtGui import QColor
                            r, g, b, a = overlay_data.color_rgba
                            qcolor = QColor(int(r*255), int(g*255), int(b*255), int(a*255))
                            
                            if hasattr(self._ui_components.central_viewer, 'add_segmentation_overlay'):
                                color_tuple = (qcolor.red(), qcolor.green(), qcolor.blue())
                                
                                self._ui_components.central_viewer.add_segmentation_overlay(
                                    overlay_data=overlay_data.mask_array,
                                    overlay_id=overlay_id,
                                    color=color_tuple
                                )
                                
                            else:
                                self._logger.error("Central viewer does not support segmentation overlays")
                    
                    self._add_mask_to_patient_tree(mask_medical_image, medical_image, mask_path)
                    
                except Exception as e:
                    self._logger.error(f"Failed to load mask {mask_path_str}: {e}")
                    continue
                    
            self._update_segmentation_tab_with_masks(medical_image, associated_masks)
            
            
        except Exception as e:
            self._logger.error(f"Error in automatic mask loading: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_mask_to_patient_tree(self, mask_image, reference_image, mask_path):
        try:
            metadata = getattr(mask_image, '_dicom_metadata', {})
            mask_type = metadata.get('detected_sequence_type', 'unknown')
            mask_description = f"Mask ({mask_path.stem})"
            
            patient_id = reference_image.patient_id
            study_uid = reference_image.study_instance_uid
            
            mask_series_data = {
                'series_description': f"{mask_description}",
                'modality': 'SEG',  
                'images_count': 1,
                'medical_image_object': mask_image,
                'is_mask': True,
                'mask_type': mask_type,
                'reference_series': reference_image.series_instance_uid
            }            
            if hasattr(self._ui_components, 'patient_browser') and self._ui_components.patient_browser:
                current_data = getattr(self._ui_components.patient_browser, 'current_patient_data', {})
                
                if patient_id in current_data.get('patients', {}):
                    if study_uid in current_data['patients'][patient_id].get('studies', {}):
                        study_data = current_data['patients'][patient_id]['studies'][study_uid]

                        if 'series' not in study_data:
                            study_data['series'] = {}
                        
                        study_data['series'][mask_image.series_instance_uid] = mask_series_data
                        
                        if hasattr(self._ui_components.patient_browser, 'add_medical_image_to_cache'):
                            self._ui_components.patient_browser.add_medical_image_to_cache(
                                mask_image.series_instance_uid, mask_image
                            )
                                                
                        if hasattr(self._ui_components.patient_browser, 'patient_tree'):
                            self._ui_components.patient_browser.patient_tree.populate_tree(current_data)
            
        except Exception as e:
            self._logger.error(f"Error adding mask to patient tree: {e}")

    def _update_segmentation_tab_with_masks(self, medical_image, associated_masks):
        try:
            if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
                manual_panel = self._ui_components.right_panel._panels.get('manual_editing')
                
                if manual_panel and hasattr(manual_panel, 'update_available_segmentations'):
                    segmentations = []
                    series_uid = medical_image.series_instance_uid
                    
                    for mask_path_str in associated_masks:
                        mask_path = Path(mask_path_str)                        
                        try:
                            mask_medical_image = self.data_loader.load_mask(mask_path, reference_image=medical_image)
                            mask_overlays = self.mask_visualizer.create_visualization_overlays(
                                mask_array=mask_medical_image.image_data,
                                mask_type=None
                            )                            
                            if len(mask_overlays) > 1:
                                for i, overlay_data in enumerate(mask_overlays):
                                    region_name = f"{mask_path.stem}_{overlay_data.label_name if hasattr(overlay_data, 'label_name') else f'Region_{i+1}'}"
                                    segmentation_entry = {
                                        'name': region_name,
                                        'type': '',
                                        'confidence': 0.95,
                                        'file_path': mask_path_str,
                                        'medical_image_reference': series_uid,
                                        'overlay_index': i  
                                    }
                                    segmentations.append(segmentation_entry)
                            else:
                                segmentation_entry = {
                                    'name': mask_path.stem,
                                    'type': 'Auto-detected',
                                    'confidence': 0.95,
                                    'file_path': mask_path_str,
                                    'medical_image_reference': series_uid,
                                    'overlay_index': 0
                                }
                                segmentations.append(segmentation_entry)
                        except Exception as e:
                            self._logger.error(f"Failed to process mask {mask_path}: {e}")
                            segmentation_entry = {
                                'name': mask_path.stem,
                                'type': 'Auto-detected',
                                'confidence': 0.95,
                                'file_path': mask_path_str,
                                'medical_image_reference': series_uid,
                                'overlay_index': 0
                            }
                            segmentations.append(segmentation_entry)
                    
                    self._segmentations_by_series[series_uid] = segmentations                    
                    manual_panel.update_available_segmentations(segmentations)

                    if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
                        if hasattr(self._ui_components.central_viewer, 'update_available_segmentations'):
                            self._ui_components.central_viewer.update_available_segmentations(segmentations)                    
                else:
                    self._logger.error("Manual editing panel or update_available_segmentations method not found")
            else:
                self._logger.error("Right panel structure not found for segmentation update")
                
        except Exception as e:
            self._logger.error(f"Error updating segmentation tab: {e}")
    
    def _setup_segmentation_synchronization(self) -> None:
        try:
            if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
                if hasattr(self._ui_components.central_viewer, 'segmentation_selection_changed'):
                    self._ui_components.central_viewer.segmentation_selection_changed.connect(
                        self._on_image_viewer_segmentation_changed
                    )

            if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
                manual_panel = self._ui_components.right_panel._panels.get('manual_editing')
                if manual_panel and hasattr(manual_panel, 'connect_segmentation_selector_change'):
                    manual_panel.connect_segmentation_selector_change(
                        self._on_segmentation_tab_selection_changed
                    )
                    
        except Exception as e:
            self._logger.error(f"Error setting up segmentation synchronization: {e}")
    
    def _on_image_viewer_segmentation_changed(self, selection: str) -> None:
        try:
            if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
                manual_panel = self._ui_components.right_panel._panels.get('manual_editing')
                if manual_panel and hasattr(manual_panel, 'sync_with_image_viewer_segmentation'):
                    manual_panel.sync_with_image_viewer_segmentation(selection)
                    
        except Exception as e:
            self._logger.error(f"Error syncing segmentation tab with image viewer: {e}")
    
    def _on_segmentation_tab_selection_changed(self, selection: str) -> None:
        try:
            if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
                if hasattr(self._ui_components.central_viewer, 'set_active_segmentation'):
                    self._ui_components.central_viewer.set_active_segmentation(selection)
                    
        except Exception as e:
            self._logger.error(f"Error syncing image viewer with segmentation tab: {e}")

    def _on_patient_context_changed(self, patient_id: str) -> None:
        self._current_patient_id = patient_id        
        self.patient_context_changed.emit(patient_id)
    
    def _on_image_selected_from_browser(self, series_uid: str) -> None:
        try:
            self._coordinator.start_image_loading_workflow(series_uid=series_uid)
        except Exception as e:
            self._logger.error(f"Error loading image from browser: {e}")
            self._show_error_message("Loading Error", f"Failed to load image: {str(e)}")
    
    def _on_cached_image_selected_from_browser(self, medical_image) -> None:
        try:
            self._coordinator.load_cached_medical_image(medical_image)
        except Exception as e:
            self._logger.error(f"Error loading cached image: {e}")
            self._show_error_message("Loading Error", f"Failed to load cached image: {str(e)}")
    
    def _on_batch_series_load_requested(self, series_uids: list) -> None:
        if not series_uids:
            self._logger.info("No series UIDs provided for batch loading")
            return
        
        try:
            for i, series_uid in enumerate(series_uids, 1):
                self._coordinator.start_image_loading_workflow(series_uid=series_uid)
        except Exception as e:
            self._logger.error(f"Error during batch series loading: {e}")
            self._show_error_message("Batch Loading Error", 
                                   f"Failed to load {len(series_uids)} series: {str(e)}")
    
    def _on_ai_analysis_completed(self, result) -> None:
        try:
            self._logger.info(f"AI Analysis completed: {len(result.segmentations)} segmentations")

            viewer = self._ui_components.central_viewer
            if not viewer:
                self._logger.error("No central viewer available")
                return

            overlay_service = getattr(viewer, '_overlay_service', None)
            if not overlay_service:
                self._logger.error("No overlay service found in viewer")
                return

            from PyQt6.QtGui import QColor
            from deepprostate.core.domain.entities.segmentation import AnatomicalRegion

            region_colors = {
                AnatomicalRegion.PROSTATE_WHOLE: QColor(204, 153, 102, 150),      # Brown
                AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: QColor(102, 204, 102, 150),  # Green
                AnatomicalRegion.PROSTATE_TRANSITION_ZONE: QColor(102, 102, 204, 150),  # Blue
                AnatomicalRegion.CONFIRMED_CANCER: QColor(255, 51, 51, 150),      # Red
                AnatomicalRegion.SUSPICIOUS_LESION: QColor(255, 204, 0, 150),     # Yellow
            }

            for i, segmentation in enumerate(result.segmentations):
                region_name = segmentation.anatomical_region.value
                overlay_id = f"ai_{result.analysis_type.value}_{region_name}_{i}"

                if hasattr(segmentation, 'mask_3d'):
                    mask_data = segmentation.mask_3d
                elif hasattr(segmentation, 'mask_data'):
                    mask_data = segmentation.mask_data
                else:
                    self._logger.error(f"No mask data found for segmentation {i}")
                    continue

                color = region_colors.get(segmentation.anatomical_region, QColor(255, 0, 255, 150))

                overlay_service.add_segmentation_overlay(
                    segmentation_id=overlay_id,
                    mask_data=mask_data,
                    color=color,
                    region_type=region_name
                )

                self._logger.debug(f"Added overlay {overlay_id} with shape {mask_data.shape}")

            if hasattr(viewer, 'update_overlays'):
                viewer.update_overlays()
            elif hasattr(viewer, 'refresh'):
                viewer.refresh()

            self._logger.info(f"Successfully added {len(result.segmentations)} overlays to viewer")

        except Exception as e:
            self._logger.error(f"Error adding AI segmentations to viewer: {e}", exc_info=True)

    def _on_ai_overlay_visibility_changed(self, overlay_id: str, visible: bool) -> None:
        try:
            viewer = self._ui_components.central_viewer
            if not viewer:
                return

            overlay_service = getattr(viewer, '_overlay_service', None)
            if not overlay_service:
                self._logger.error("No overlay service found")
                return

            overlay_service.set_overlay_visibility(overlay_id, visible)

            if hasattr(viewer, 'update_overlays'):
                viewer.update_overlays()
            elif hasattr(viewer, 'refresh'):
                viewer.refresh()

            self._logger.debug(f"Overlay {overlay_id} visibility: {visible}")

        except Exception as e:
            self._logger.error(f"Error changing overlay visibility: {e}")
    
    def _generate_fallback_series_description(self, image) -> str:
        metadata = getattr(image, '_dicom_metadata', {})
        if metadata:
            detected_sequence = metadata.get('detected_sequence_type')
            if detected_sequence and detected_sequence != 'UNKNOWN':
                return f"{detected_sequence} Series"
        
        if hasattr(image, 'series_instance_uid') and image.series_instance_uid:
            series_uid = image.series_instance_uid
            if '_' in series_uid:
                parts = series_uid.split('_')
                if len(parts) >= 3: 
                    sequence_part = '_'.join(parts[2:]) 
                    if sequence_part and sequence_part != 'UNKNOWN':
                        return f"{sequence_part} Series"
        
        if hasattr(image, 'file_path') and image.file_path:
            filename = Path(image.file_path).stem.upper()
            
            if 'T2W' in filename or 'T2-WEIGHTED' in filename:
                return "T2W Series"
            elif 'ADC' in filename:
                return "ADC Series" 
            elif 'DWI' in filename:
                return "DWI Series"
            elif 'T1W' in filename or 'T1-WEIGHTED' in filename:
                return "T1W Series"
            elif 'FLAIR' in filename:
                return "FLAIR Series"
            else:
                if '_' in filename:
                    sequence_part = filename.split('_')[-1]
                    if sequence_part and len(sequence_part) > 1:
                        return f"{sequence_part} Series"
                else:
                    return f"{filename} Series"
        
        return "Medical Series"
    
    def _extract_mask_info(self, image, key: str):
        metadata = getattr(image, '_dicom_metadata', {})
        if metadata:
            return metadata.get(key, False if 'has_' in key else 0 if 'count' in key else [])
        return False if 'has_' in key else 0 if 'count' in key else []
    
    def _update_segmentation_dropdowns_for_current_image(self, medical_image: MedicalImage) -> None:
        try:
            series_uid = medical_image.series_instance_uid
            self._current_image = medical_image  
            current_segmentations = self._segmentations_by_series.get(series_uid, [])
            
            if hasattr(self._ui_components, 'right_panel') and hasattr(self._ui_components.right_panel, '_panels'):
                manual_panel = self._ui_components.right_panel._panels.get('manual_editing')
                if manual_panel and hasattr(manual_panel, 'update_available_segmentations'):
                    manual_panel.update_available_segmentations(current_segmentations)
            
            if hasattr(self._ui_components, 'central_viewer') and self._ui_components.central_viewer:
                if hasattr(self._ui_components.central_viewer, 'update_available_segmentations'):
                    self._ui_components.central_viewer.update_available_segmentations(current_segmentations)
            
            self._hide_non_current_overlays(series_uid)
            
        except Exception as e:
            self._logger.error(f"Error updating segmentation dropdowns for current image: {e}")
            import traceback
            traceback.print_exc()
    
    def _hide_non_current_overlays(self, current_series_uid: str) -> None:
        try:
            if not hasattr(self._ui_components, 'central_viewer') or not self._ui_components.central_viewer:
                return
                
            overlay_service = getattr(self._ui_components.central_viewer, '_overlay_service', None)
            if not overlay_service:
                return
            
            all_overlay_ids = overlay_service.get_all_overlay_ids()
            hidden_count = 0
            
            for overlay_id in all_overlay_ids:
                current_segmentations = self._segmentations_by_series.get(current_series_uid, [])
                belongs_to_current = any(
                    overlay_id.startswith(f"auto_mask_{Path(seg['file_path']).stem}")
                    for seg in current_segmentations
                )
                
                if not belongs_to_current:
                    overlay_service.set_overlay_visibility(overlay_id, False)
                    hidden_count += 1
        except Exception as e:
            self._logger.error(f"Error hiding non-current overlays: {e}")
        
    def _validate_ai_analysis_prerequisites(self) -> bool:
        if not self._current_image:
            return False
        
        if not self._current_patient_id:
            self._show_user_message(
                "Missing Patient Context",
                "Patient context is required for AI analysis.\n"
                "Please ensure proper patient selection.",
                "warning"
            )
            return False
        
        return True
    
    def _update_high_level_status(self) -> None:
        status_parts = []        
        status_parts.append(f"Status: {self._application_state.replace('_', ' ').title()}")
        
        if self._current_patient_id:
            status_parts.append(f"Patient: {self._current_patient_id}")
        else:
            status_parts.append("No patient selected")
        
        if self._current_image:
            modality = self._current_image.modality.value
            status_parts.append(f"Image: {modality}")
        
        active_workflows = self._coordinator.get_active_workflows()
        if active_workflows:
            status_parts.append(f"Active workflows: {len(active_workflows)}")
        
        self.statusBar().showMessage(" | ".join(status_parts))
    
    def _show_user_message(self, title: str, message: str, message_type: str = "info") -> None:
        if message_type == "error":
            QMessageBox.critical(self, title, message)
        elif message_type == "warning":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)
    
    def _handle_low_quality_ai_results(self, data: Dict[str, Any]) -> None:
        message = (
            "The AI analysis has produced results with lower confidence scores.\n\n"
            "Medical recommendation:\n"
            "• Review results carefully with clinical context\n"
            "• Consider manual verification of segmentations\n"
            "• Evaluate need for additional imaging\n\n"
            "Would you like to proceed with these results?"
        )
        self._show_user_message("AI Quality Alert", message, "warning")
    
    def _handle_ambiguous_medical_findings(self, data: Dict[str, Any]) -> None:
        self._show_user_message(
            "Medical Review Required",
            "Ambiguous findings detected. Medical review recommended.",
            "warning"
        )
    
    def _handle_generic_medical_validation(self, validation_type: str, data: Dict[str, Any]) -> None:
        self._show_user_message(
            "Medical Validation",
            f"Medical validation required: {validation_type}",
            "info"
        )
    
    def _show_development_info(self) -> None:
        dev_info = f"""
        Development Information:
        
        Service Container: {type(self._services).__name__}
        Workflow Coordinator: {type(self._coordinator).__name__}
        UI Components: {type(self._ui_components).__name__}
        
        Current State: {self._application_state}
        Patient ID: {self._current_patient_id or 'None'}
        Current Image: {self._current_image.series_instance_uid if self._current_image else 'None'}
        
        Active Workflows: {len(self._coordinator.get_active_workflows())}
        Workflow History: {len(self._workflow_history)} events
        """
        
        QMessageBox.information(self, "Development View", dev_info.strip())
    
    def _hide_development_info(self) -> None:
        """Oculta información of desarrollo."""
  
    def closeEvent(self, event: QCloseEvent) -> None:
        active_workflows = self._coordinator.get_active_workflows()
        if active_workflows:
            reply = QMessageBox.question(
                self,
                "Active Medical Workflows",
                f"There are {len(active_workflows)} active medical workflows.\n\n"
                "Closing now may interrupt medical processes.\n"
                "Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

