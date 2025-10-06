"""
infrastructure/ui/widgets/ai_analysis_panel.py

AI Analysis Panel for medical image analysis with nnUNet models.
Provides UI controls for selecting analysis type, managing input sequences,
and controlling visualization overlays.
"""

import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QTreeWidget,
    QTreeWidgetItem, QComboBox, QProgressBar, QCheckBox, QFileDialog,
    QMessageBox, QTextEdit, QSplitter, QScrollArea, QSlider, QFrame,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QPixmap

from src.core.domain.entities.ai_analysis import (
    AIAnalysisType, AIAnalysisRequest, AIAnalysisResult, AISequenceRequirement
)
from src.core.domain.entities.medical_image import MedicalImage
from src.use_cases.application.services.ai_analysis_orchestrator import AIAnalysisOrchestrator
from src.core.domain.services.patient_browser_integration_service import PatientBrowserIntegrationService
from src.core.domain.services.ai_analysis_persistence_service import AIAnalysisPersistenceService
from src.core.domain.services.ai_analysis_export_service import AIAnalysisExportService
from src.core.domain.services.dynamic_model_config_service import DynamicModelConfigService

# Specialized services
from src.core.domain.services.sequence_detection_service import SequenceDetectionService
from src.adapters.image_conversion.nifti_converter import NIfTIConverter
from src.adapters.image_conversion.temp_file_manager import TempFileManager
from src.core.domain.services.analysis_validation_service import AnalysisValidationService


class AIAnalysisWorkerThread(QThread):
    """Background thread for AI analysis to prevent UI blocking."""
    
    analysis_completed = pyqtSignal(object)  # AIAnalysisResult
    analysis_failed = pyqtSignal(str)        # Error message
    progress_update = pyqtSignal(str)        # Status message
    
    def __init__(self, orchestrator: AIAnalysisOrchestrator, request: AIAnalysisRequest):
        super().__init__()
        self.orchestrator = orchestrator
        self.request = request
        
    def run(self):
        """Execute AI analysis in background thread."""
        try:
            # Run async analysis in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.orchestrator.run_ai_analysis(self.request)
                )
                self.analysis_completed.emit(result)
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"WORKER ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_failed.emit(str(e))


class AIAnalysisPanel(QWidget):
    """
    Complete UI panel for AI analysis functionality.
    
    Provides controls for:
    - Selecting analysis type (Prostate Gland, TZ/PZ Zones, csPCa Detection)
    - Managing required input sequences
    - Running analysis with progress tracking
    - Displaying results and controlling overlays
    - Saving generated masks
    """
    
    analysis_requested = pyqtSignal(object)      # AIAnalysisRequest
    overlay_visibility_changed = pyqtSignal(str, bool)  # overlay_id, visible
    mask_save_requested = pyqtSignal(list, str)  # segmentations, format
    analysis_completed = pyqtSignal(object)      # AIAnalysisResult
    case_selection_changed = pyqtSignal(object)  # MedicalImage - for viewer synchronization
    
    def __init__(self, orchestrator: AIAnalysisOrchestrator, dynamic_config_service: Optional[DynamicModelConfigService] = None):
        super().__init__()
        self._orchestrator = orchestrator
        self._dynamic_config = dynamic_config_service
        self._logger = logging.getLogger(__name__)

        self._patient_browser_service = PatientBrowserIntegrationService()
        self._persistence_service = AIAnalysisPersistenceService()
        self._export_service = AIAnalysisExportService()

        # Specialized services
        self._sequence_detector = SequenceDetectionService()
        self._temp_file_manager = TempFileManager()
        self._nifti_converter = NIfTIConverter(temp_file_manager=self._temp_file_manager)
        self._validation_service = AnalysisValidationService(
            sequence_detection_service=self._sequence_detector
        )

        # State management
        self._current_analysis_result: Optional[AIAnalysisResult] = None
        self._selected_sequences: Dict[str, Path] = {}
        self._worker_thread: Optional[AIAnalysisWorkerThread] = None
        self._current_medical_image = None  # Currently loaded medical image from Patient Studies
        self._current_image_file_path = None  # File path of the currently loaded image
        self._patient_browser = None  # Reference to Patient Browser for accessing loaded cases
        self._case_selected_from_input_sequences = False  # Track if case was explicitly selected from Input Sequences
        self._temp_files_created = []  # List of temporary files created for cleanup
        self._temp_dir = Path("temp")  # Temporary directory for .nii.gz files

        self._models_available = False

        self._setup_ui()
        self._setup_connections()

        if self._dynamic_config:
            self._dynamic_config.add_config_listener(self._on_model_config_changed)
            self._check_initial_model_availability()


    def set_patient_browser_reference(self, patient_browser):
        """Set reference to Patient Browser for Clean Architecture integration."""
        self._patient_browser = patient_browser
        self._patient_browser_service.set_patient_browser_reference(patient_browser)

    def sizeHint(self):
        """Provide size hint for optimal AI analysis panel layout."""
        from PyQt6.QtCore import QSize
        optimal_width = 400  # Generous width for all content
        optimal_height = 600  # Height to accommodate all sections
        return QSize(optimal_width, optimal_height)
    
    def minimumSizeHint(self):
        """Provide minimum size hint."""
        from PyQt6.QtCore import QSize
        return QSize(350, 400)  # Minimum usable size for AI analysis
    
    def _setup_ui(self):
        """Set up clean vertical navigation UI layout."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Main content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(1)  # Minimal spacing between sections
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self._create_navigation_sections(content_layout)
        
        content_layout.addStretch(1)
        
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        self._apply_navigation_styling()
    
    def _create_navigation_sections(self, parent_layout):
        """Create clean vertical navigation sections."""
        # 1. Analysis Type Selection - Always visible
        analysis_section = self._create_collapsible_section(
            "🔍 Analysis Type", 
            self._create_analysis_type_content(),
            initially_expanded=True
        )
        parent_layout.addWidget(analysis_section)
        
        # 2. Input Sequences - Expandable
        sequences_section = self._create_collapsible_section(
            "📁 Input Sequences",
            self._create_sequences_content(),
            initially_expanded=False
        )
        parent_layout.addWidget(sequences_section)
        
        # 3. Advanced Settings - HIDDEN (reserved for future implementation)
        #   - Confidence threshold (currently at line 1275)
        #   - Post-processing presets (Conservative/Aggressive/Minimal)
        #   - Expert mode toggles
        # settings_section = self._create_collapsible_section(
        #     "⚙️ Advanced Settings",
        #     self._create_settings_content(),
        #     initially_expanded=False
        # )
        # parent_layout.addWidget(settings_section)
        
        # 4. Analysis Control - Always visible when ready
        control_section = self._create_collapsible_section(
            "▶️ Run Analysis",
            self._create_control_content(),
            initially_expanded=True
        )
        parent_layout.addWidget(control_section)
        
        # 5. Results & Overlays - Shown after analysis
        results_section = self._create_collapsible_section(
            "📊 Results & Overlays", 
            self._create_results_section(),
            initially_expanded=False
        )
        parent_layout.addWidget(results_section)
        
        # 6. Prediction History - Collapsed by default
        history_section = self._create_collapsible_section(
            "📈 Prediction History",
            self._create_history_section(),
            initially_expanded=False
        )
        parent_layout.addWidget(history_section)
        self._results_section = results_section
        results_section.setVisible(False)  # Hiddin until analysis completes
    
    def _create_collapsible_section(self, title: str, content_widget: QWidget, initially_expanded: bool = True) -> QWidget:
        """Create a collapsible section with professional medical styling."""
        section_widget = QWidget()
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)
        
        # Header button for expand/collapse
        header_button = QPushButton(title)
        header_button.setCheckable(True)
        header_button.setChecked(initially_expanded)
        header_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header_button.setMinimumHeight(40)
        
        content_container = QWidget()
        content_container_layout = QVBoxLayout()
        content_container_layout.setContentsMargins(8, 8, 8, 12)
        content_container_layout.addWidget(content_widget)
        content_container.setLayout(content_container_layout)
        content_container.setVisible(initially_expanded)
        
        header_button.toggled.connect(content_container.setVisible)
        
        section_layout.addWidget(header_button)
        section_layout.addWidget(content_container)
        section_widget.setLayout(section_layout)
        
        return section_widget
    
    def _create_analysis_type_content(self) -> QWidget:
        """Create clean analysis type selection content."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        self._analysis_buttons = {}
        
        analysis_types = [
            {
                'type': AIAnalysisType.PROSTATE_GLAND,
                'name': 'Prostate Gland',
                'description': 'Complete prostate segmentation',
                'accuracy': 'DICE: 95%',
                'sequences': ['T2W']
            },
            {
                'type': AIAnalysisType.ZONES_TZ_PZ, 
                'name': 'Prostate Zones',
                'description': 'Transition & Peripheral zones',
                'accuracy': 'DICE: 80%',
                'sequences': ['T2W']
            },
            {
                'type': AIAnalysisType.CSPCA_DETECTION,
                'name': 'csPCa Detection', 
                'description': 'Clinically significant PCa detection',
                'accuracy': 'DICE: 75%',
                'sequences': ['T2W', 'ADC', 'HBV']
            }
        ]
        
        from PyQt6.QtWidgets import QButtonGroup
        self._analysis_button_group = QButtonGroup()
        
        for analysis in analysis_types:
            option_widget = self._create_analysis_option(analysis)
            layout.addWidget(option_widget)
            
        # No default selection - user must choose
        
        widget.setLayout(layout)
        return widget
    
    def _create_analysis_option(self, analysis_data: dict) -> QWidget:
        """Create a single analysis option with professional styling."""
        option_widget = QWidget()
        option_layout = QHBoxLayout()
        option_layout.setContentsMargins(4, 4, 4, 4)
        
        # Radio button
        radio_button = QRadioButton()
        radio_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._analysis_buttons[analysis_data['type']] = radio_button
        
        self._analysis_button_group.addButton(radio_button)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)
        
        # Title
        title_label = QLabel(analysis_data['name'])
        title_label.setFont(QFont("", 11, QFont.Weight.DemiBold))
        
        # Description and accuracy in one line
        desc_accuracy = f"{analysis_data['description']} • {analysis_data['accuracy']}"
        desc_label = QLabel(desc_accuracy)
        desc_label.setFont(QFont("", 9))
        desc_label.setStyleSheet("color: #666666;")
        
        # Sequences required
        sequences_text = "Requires: " + " + ".join(analysis_data['sequences'])
        seq_label = QLabel(sequences_text)
        seq_label.setFont(QFont("", 8))
        seq_label.setStyleSheet("color: #888888; font-style: italic;")
        
        content_layout.addWidget(title_label)
        content_layout.addWidget(desc_label)
        content_layout.addWidget(seq_label)
        content_widget.setLayout(content_layout)
        
        option_layout.addWidget(radio_button)
        option_layout.addWidget(content_widget, 1)
        option_widget.setLayout(option_layout)
        
        # Make entire option clickable
        def select_option():
            radio_button.setChecked(True)
        content_widget.mousePressEvent = lambda e: select_option()
        
        return option_widget
    
    def _create_sequences_content(self) -> QWidget:
        """Create clean input sequences content."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        
        # Sequences will be populated based on selected analysis type
        self._sequence_widgets = {}
        
        info_label = QLabel("Select input sequences for AI analysis:")
        info_label.setFont(QFont("", 9))
        info_label.setStyleSheet("color: #555; margin-bottom: 6px;")
        layout.addWidget(info_label)
        
        # Placeholder that will be updated based on analysis type
        self._sequences_container = QWidget()
        self._sequences_layout = QVBoxLayout()
        self._sequences_container.setLayout(self._sequences_layout)
        layout.addWidget(self._sequences_container)
        
        widget.setLayout(layout)
        return widget
        
    #   - Confidence threshold slider (move from hardcoded 0.75)
    #   - Post-processing presets (Conservative/Aggressive/Minimal)
    #   - Expert mode toggles
    # The removed code created dead controls that were never read:
    #   - self._smooth_boundaries_cb (NEVER USED)
    #   - self._fill_holes_cb (NEVER USED)
    #   - self._remove_small_cb (NEVER USED)
        
    def _create_control_content(self) -> QWidget:
        """Create clean analysis control content."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        self._status_label = QLabel("AI models not loaded")
        self._status_label.setFont(QFont("", 9))
        self._status_label.setStyleSheet("color: #d32f2f; padding: 4px;")
        layout.addWidget(self._status_label)

        self._load_models_button = QPushButton("📁 Load AI Models Path")
        self._load_models_button.setMinimumHeight(36)
        self._load_models_button.setFont(QFont("", 10, QFont.Weight.DemiBold))
        self._load_models_button.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: 1px solid #1976d2;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
                border-color: #1565c0;
            }
        """)
        layout.addWidget(self._load_models_button)

        # Run Analysis button (hiddin initially)
        self._run_button = QPushButton("🚀 Run AI Analysis")
        self._run_button.setMinimumHeight(36)
        self._run_button.setFont(QFont("", 10, QFont.Weight.DemiBold))
        self._run_button.setEnabled(False)
        self._run_button.setVisible(False)  # Hiddin until models are loaded
        self._run_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: 1px solid #45a049;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #3e8e41;
            }
        """)
        layout.addWidget(self._run_button)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        widget.setLayout(layout)
        return widget
        

    def _apply_navigation_styling(self):
        """Apply professional medical styling to the navigation."""
        self.setStyleSheet("""
            /* Main panel styling */
            AIAnalysisPanel {
                background-color: #fafafa;
                border: none;
            }
            
            /* Section headers */
            QPushButton[checkable="true"] {
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 6px;
                text-align: left;
                padding-left: 12px;
                font-weight: 600;
                color: #1565c0;
            }
            
            QPushButton[checkable="true"]:hover {
                background-color: #bbdefb;
                border-color: #90caf9;
            }
            
            QPushButton[checkable="true"]:checked {
                background-color: #2196f3;
                color: white;
                border-color: #1976d2;
            }
            
            /* Content areas */
            QWidget {
                background-color: transparent;
            }
            
            /* Radio buttons */
            QRadioButton {
                font-size: 11px;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #1976d2;
                background-color: white;
            }
            
            QRadioButton::indicator:checked {
                background-color: #2196f3;
                border-color: #1976d2;
            }
            
            /* Sliders */
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #f0f0f0;
                height: 6px;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #2196f3;
                border: 1px solid #1976d2;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            /* Checkboxes */
            QCheckBox {
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #1976d2;
                border-radius: 3px;
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2196f3;
                border-color: #1976d2;
            }
            
            /* Progress bar */
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
                text-align: center;
                height: 18px;
            }
            
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
            
            /* Run button - REMOVED: was causing button to always be green
               Now button styling is controlled dynamically via _update_analysis_status()
            QPushButton[text*="Run"] {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            
            QPushButton[text*="Run"]:hover {
                background-color: #45a049;
            }
            
            QPushButton[text*="Run"]:pressed {
                background-color: #3d8b40;
            }
            */
            
            /* Scroll area */
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 8px;
                border-radius: 4px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 4px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
        """)
    
    def _create_analysis_configuration_section(self) -> QWidget:
        """Create the analysis configuration controls."""
        config_widget = QWidget()
        layout = QVBoxLayout()
        
        analysis_group = QGroupBox("AI Analysis Type")
        analysis_layout = QVBoxLayout()
        
        self._analysis_buttons = {}
        analysis_info = [
            (AIAnalysisType.PROSTATE_GLAND, "Prostate Gland Segmentation", 
             "T2W single-label model (DICE: 95%)", "Single T2W sequence required"),
            (AIAnalysisType.ZONES_TZ_PZ, "Prostate Zones (TZ/PZ)", 
             "T2W dual-label model (DICE: 80%)", "Single T2W sequence required"),
            (AIAnalysisType.CSPCA_DETECTION, "csPCa Detection", 
             "Multi-sequence model (DICE: 75%)", "T2W + ADC + HBV sequences required")
        ]
        
        for analysis_type, title, subtitle, description in analysis_info:
            radio_widget = QWidget()
            radio_layout = QVBoxLayout()
            radio_layout.setSpacing(2)
            
            radio_button = QRadioButton(title)
            radio_button.setFont(QFont("", 10, QFont.Weight.Bold))
            
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #666; font-size: 9pt;")
            
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888; font-size: 8pt; font-style: italic;")
            desc_label.setWordWrap(True)
            
            radio_layout.addWidget(radio_button)
            radio_layout.addWidget(subtitle_label)
            radio_layout.addWidget(desc_label)
            radio_widget.setLayout(radio_layout)
            
            analysis_layout.addWidget(radio_widget)
            
            self._analysis_buttons[analysis_type] = radio_button
            radio_button.setEnabled(False)  # Disabled until models are loaded
            radio_button.toggled.connect(
                lambda checked, at=analysis_type: self._on_analysis_type_changed(at, checked)
            )
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Sequence File Selection
        files_group = QGroupBox("Required Image Sequences")
        files_layout = QVBoxLayout()
        
        # Sequence requirements display
        self._requirements_label = QLabel("Select an analysis type to see requirements.")
        self._requirements_label.setWordWrap(True)
        self._requirements_label.setStyleSheet("color: #666; font-style: italic; padding: 8px;")
        files_layout.addWidget(self._requirements_label)
        
        # File selection list
        self._sequence_list = QListWidget()
        self._sequence_list.setMaximumHeight(150)
        self._sequence_list.setAlternatingRowColors(True)
        files_layout.addWidget(self._sequence_list)
        
        # File selection buttons
        file_buttons_layout = QHBoxLayout()
        
        self._add_sequence_btn = QPushButton("Add Sequence File...")
        self._add_sequence_btn.clicked.connect(self._add_sequence_file)
        self._add_sequence_btn.setEnabled(False)
        file_buttons_layout.addWidget(self._add_sequence_btn)
        
        self._remove_sequence_btn = QPushButton("Remove Selected")
        self._remove_sequence_btn.clicked.connect(self._remove_selected_sequence)
        self._remove_sequence_btn.setEnabled(False)
        file_buttons_layout.addWidget(self._remove_sequence_btn)
        
        file_buttons_layout.addStretch()
        files_layout.addLayout(file_buttons_layout)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        execution_group = QGroupBox("Analysis Execution")
        execution_layout = QVBoxLayout()
        
        # Options
        options_layout = QHBoxLayout()
        
        self._apply_refinement_cb = QCheckBox("Apply post-processing refinement")
        self._apply_refinement_cb.setChecked(True)
        self._apply_refinement_cb.setToolTip("Apply conservative morphological operations to improve mask quality")
        options_layout.addWidget(self._apply_refinement_cb)
        
        self._save_temp_files_cb = QCheckBox("Keep temporary files")
        self._save_temp_files_cb.setToolTip("Keep intermediate .nii.gz files for debugging")
        options_layout.addWidget(self._save_temp_files_cb)
        
        options_layout.addStretch()
        execution_layout.addLayout(options_layout)
        
        # Execution controls
        controls_layout = QHBoxLayout()
        
        self._run_button.setEnabled(False)
        controls_layout.addWidget(self._run_button)
        
        self._cancel_analysis_btn = QPushButton("Cancel")
        self._cancel_analysis_btn.clicked.connect(self._cancel_analysis)
        self._cancel_analysis_btn.setEnabled(False)
        controls_layout.addWidget(self._cancel_analysis_btn)
        
        execution_layout.addLayout(controls_layout)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        execution_layout.addWidget(self._progress_bar)
        
        self._status_label = QLabel("Ready - Select analysis type and add required sequences")
        self._status_label.setStyleSheet("color: #666; font-style: italic;")
        execution_layout.addWidget(self._status_label)
        
        execution_group.setLayout(execution_layout)
        layout.addWidget(execution_group)
        
        config_widget.setLayout(layout)
        return config_widget
    
    def _create_results_section(self) -> QWidget:
        """Create the results display and overlay controls."""
        results_widget = QWidget()
        layout = QVBoxLayout()
        
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self._results_tree = QTreeWidget()
        self._results_tree.setHeaderLabels(["Structure", "Confidence", "Volume (mm³)", "Visible"])
        self._results_tree.setAlternatingRowColors(True)
        self._results_tree.setMaximumHeight(200)
        self._results_tree.itemChanged.connect(self._on_overlay_visibility_changed)
        results_layout.addWidget(self._results_tree)
        
        self._summary_text = QTextEdit()
        self._summary_text.setMaximumHeight(80)
        self._summary_text.setReadOnly(True)
        self._summary_text.setPlainText("No analysis performed yet.")
        results_layout.addWidget(self._summary_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Overlay Controls
        overlay_group = QGroupBox("Overlay Controls")
        overlay_layout = QVBoxLayout()
        
        # Global overlay controls
        global_controls_layout = QHBoxLayout()
        
        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.clicked.connect(lambda: self._toggle_all_overlays(True))
        self._show_all_btn.setEnabled(False)
        global_controls_layout.addWidget(self._show_all_btn)
        
        self._hide_all_btn = QPushButton("Hiof All")
        self._hide_all_btn.clicked.connect(lambda: self._toggle_all_overlays(False))
        self._hide_all_btn.setEnabled(False)
        global_controls_layout.addWidget(self._hide_all_btn)
        
        global_controls_layout.addStretch()
        
        opacity_label = QLabel("Opacity:")
        global_controls_layout.addWidget(opacity_label)
        
        # Opacity slider for future enhancement
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(75)
        self._opacity_slider.setEnabled(False)  # Disabled until overlay service integration
        global_controls_layout.addWidget(self._opacity_slider)
        
        overlay_layout.addLayout(global_controls_layout)
        
        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)
        
        save_group = QGroupBox("Save Analysis Results")
        save_layout = QHBoxLayout()
        
        save_layout.addWidget(QLabel("Format:"))
        
        self._save_format_combo = QComboBox()
        self._save_format_combo.addItems([
            ".nii.gz (NIfTI Compressed)", 
            ".dcm (DICOM Segmentation)", 
            ".stl (3D Surface Mesh)",
            ".mha (MetaImage)"
        ])
        save_layout.addWidget(self._save_format_combo)
        
        self._save_masks_btn = QPushButton("Save Masks...")
        self._save_masks_btn.clicked.connect(self._save_masks)
        self._save_masks_btn.setEnabled(False)
        save_layout.addWidget(self._save_masks_btn)
        
        save_layout.addStretch()
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        results_widget.setLayout(layout)
        return results_widget
    
    def _setup_connections(self):
        """Set up signal connections for new clean navigation."""
        for analysis_type, radio_button in self._analysis_buttons.items():
            radio_button.toggled.connect(
                lambda checked, at=analysis_type: self._on_analysis_type_changed(at, checked)
            )
        
        self._load_models_button.clicked.connect(self._on_load_models_clicked)

        self._run_button.clicked.connect(self._on_run_analysis_clicked)

        self._update_analysis_status()
        
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_progress_animation)
    
    def _on_analysis_type_changed(self, analysis_type: AIAnalysisType, checked: bool):
        """Handle analysis type selection change."""
        if not checked:
            return

        # Preserve case selection if user already has a valid case selected
        # Only reset if no case is currently selected
        if not self._current_medical_image:
            self._case_selected_from_input_sequences = False

        self._update_sequence_requirements(analysis_type)

        self._update_analysis_status()
        
    def _update_sequence_requirements(self, analysis_type: AIAnalysisType):
        """Update the sequences section to show all loaded cases."""
        for i in reversed(range(self._sequences_layout.count())):
            child = self._sequences_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Store analysis type for later use
        self._selected_analysis_type = analysis_type
        
        self._display_loaded_cases()
        
        if analysis_type == AIAnalysisType.CSPCA_DETECTION:
            self._add_multi_sequence_requirements()
    
    def _create_sequence_input(self, requirement: AISequenceRequirement) -> QWidget:
        """Create a sequence input widget."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Sequence label
        label = QLabel(f"{requirement.sequence_name}:")
        label.setFont(QFont("", 9, QFont.Weight.DemiBold))
        label.setMinimumWidth(60)
        
        # File path display
        path_label = QLabel("No file selected")
        path_label.setFont(QFont("", 8))
        path_label.setStyleSheet("color: #666; padding: 6px; border: 1px solid #ddd; border-radius: 4px; background: #fafafa;")
        
        # Browse button
        browse_button = QPushButton("Browse...")
        browse_button.setFont(QFont("", 8))
        browse_button.setMaximumWidth(80)
        
        # Store references for later access
        if not hasattr(self, '_sequence_inputs'):
            self._sequence_inputs = {}
        self._sequence_inputs[requirement.sequence_name] = {
            'path_label': path_label,
            'requirement': requirement,
            'selected_path': None
        }
        
        browse_button.clicked.connect(
            lambda: self._browse_sequence_file(requirement.sequence_name)
        )
        
        layout.addWidget(label)
        layout.addWidget(path_label, 1)
        layout.addWidget(browse_button)
        widget.setLayout(layout)
        
        return widget
    
    def _browse_sequence_file(self, sequence_name: str):
        """Browse for a sequence file."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Medical Images (*.dcm *.nii *.nii.gz *.mha *.nrrd)")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = Path(selected_files[0])
                
                if sequence_name in self._sequence_inputs:
                    self._sequence_inputs[sequence_name]['path_label'].setText(file_path.name)
                    self._sequence_inputs[sequence_name]['selected_path'] = file_path
                    
                    self._sequence_inputs[sequence_name]['path_label'].setStyleSheet(
                        "color: #2e7d32; padding: 4px; border: 1px solid #4caf50; border-radius: 3px; background-color: #e8f5e8;"
                    )
                
                # Note: Do not update analysis status here - only from Input Sequences radio buttons
    
    def _update_analysis_status(self):
        """Update the analysis status based on current selections."""
        # PRIMERA VERIFICACIÓN: Comprobar disponibilidad of modelos AI
        if not self._check_model_availability():
            self._status_label.setText("AI models not loaded - Please load models first")
            self._status_label.setStyleSheet("color: #d32f2f; padding: 4px; font-weight: 500;")
            self._run_button.setEnabled(False)
            self._set_models_not_loaded_button_style()
            return

        selected_type = self._get_selected_analysis_type()
        if not selected_type:
            self._status_label.setText("Please select an analysis type")
            self._status_label.setStyleSheet("color: #666; padding: 4px;")
            self._run_button.setEnabled(False)
            self._run_button.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    color: #666;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            return
        
        if self._current_medical_image and self._case_selected_from_input_sequences:
            if hasattr(self, '_selected_analysis_type') and self._selected_analysis_type == AIAnalysisType.CSPCA_DETECTION:
                missing_sequences = self._get_missing_multi_sequences()
                if missing_sequences:
                    self._status_label.setText(f"Missing sequences: {', '.join(missing_sequences)}")
                    self._status_label.setStyleSheet("color: #f57c00; padding: 4px;")
                    self._run_button.setEnabled(False)
                    self._set_warning_button_style()
                    return
            
            self._status_label.setText("Ready for analysis")
            self._status_label.setStyleSheet("color: #2e7d32; font-weight: 500; padding: 4px;")
            self._run_button.setEnabled(True)
            self._run_button.setStyleSheet("""
                QPushButton {
                    background-color: #4caf50;
                    color: white;
                    border: 1px solid #45a049;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    border-color: #3e8e41;
                }
                QPushButton:pressed {
                    background-color: #3e8e41;
                }
            """)
            return
        
        # No case selected
        self._status_label.setText("Select a case from the list above")
        self._status_label.setStyleSheet("color: #666; padding: 4px;")
        self._run_button.setEnabled(False)
        self._run_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #666;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
    
    def _get_missing_multi_sequences(self):
        """
        Get list of missing sequences for multi-sequence analysis (REFACTORED - delegated to service).

        This method now delegates to AnalysisValidationService for intelligent detection
        of missing sequences based on the current analysis type.
        """
        if not self._current_medical_image:
            return ["ADC", "HBV"]

        selected_type = getattr(self, '_selected_analysis_type', None)
        if not selected_type:
            return ["ADC", "HBV"]

        # Detect related sequences
        detected_sequences = self._detect_related_sequences(self._current_medical_image)

        # Delegate to AnalysisValidationService
        missing = self._validation_service.get_missing_sequences(
            analysis_type=selected_type,
            current_image=self._current_medical_image,
            available_sequences=detected_sequences
        )

        return missing
    
    def _detect_related_sequences(self, primary_image):
        """Detect related sequences using multiple criteria (REFACTORED - delegated to service)."""
        if not self._patient_browser:
            return {}

        loaded_images = self._patient_browser_service.get_loaded_cases()

        # Delegate to SequenceDetectionService
        detected_sequences = self._sequence_detector.detect_related_sequences(
            primary_image=primary_image,
            all_loaded_images=loaded_images
        )

        return detected_sequences
    
    
    def _get_selected_analysis_type(self) -> Optional[AIAnalysisType]:
        """Get the currently selected analysis type."""
        for analysis_type, radio_button in self._analysis_buttons.items():
            if radio_button.isChecked():
                return analysis_type
        return None

    def _on_load_models_clicked(self):
        """Handle load AI models button click."""
        try:
            # Opin file dialog to select models directory
            suggested_dir = str(Path.home() / "Downloads")

            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setDirectory(suggested_dir)
            dialog.setWindowTitle("Select AI Models Directory")

            if dialog.exec():
                selected_paths = dialog.selectedFiles()
                if selected_paths:
                    selected_path = Path(selected_paths[0])
                    self._logger.info(f"User selected model directory: {selected_path}")

                    self._load_models_button.setText("🔄 Loading...")
                    self._load_models_button.setEnabled(False)
                    self._status_label.setText("Validating AI models...")

                    if not self._dynamic_config:
                        self._dynamic_config = DynamicModelConfigService()
                        self._dynamic_config.add_config_listener(self._on_model_config_changed)

                    success = self._dynamic_config.set_base_model_path(selected_path)

                    # Restore button state
                    self._load_models_button.setText("📁 Load AI Models Path")
                    self._load_models_button.setEnabled(True)

                    if success:
                        status = self._dynamic_config.get_model_status()
                        available_count = status.get("available_count", 0)
                        total_count = status.get("total_models", 0)

                        QMessageBox.information(
                            self,
                            "Models Loaded Successfully",
                            f"Successfully loaded AI models from:\n{selected_path}\n\n"
                            f"Available models: {available_count}/{total_count}\n\n"
                            f"The AI Analysis panel is now ready for use."
                        )

                        # Switch to analysis mode
                        self._switch_to_analysis_mode()

                    else:
                        QMessageBox.warning(
                            self,
                            "Invalid Model Directory",
                            f"The selected directory does not contain valid AI models:\n{selected_path}\n\n"
                            f"Please ensure you have downloaded the correct model files from ZENODO "
                            f"and extracted them properly.\n\n"
                            f"Expected structure:\n"
                            f"• Dataset998_PICAI_Prostate/\n"
                            f"• Dataset600_PICAI_PZ_TZ_T2W/\n"
                            f"• Dataset500_PICAI_csPCa/"
                        )

        except Exception as e:
            self._logger.error(f"Error loading models: {e}")
            # Restore button state
            self._load_models_button.setText("📁 Load AI Models Path")
            self._load_models_button.setEnabled(True)
            self._status_label.setText("Error loading models")

            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading models:\n{str(e)}"
            )

    def _switch_to_analysis_mode(self):
        """Switch the panel from model loading mode to analysis mode."""
        self._load_models_button.setVisible(False)

        self._run_button.setVisible(True)
        self._run_button.setEnabled(True)

        self._status_label.setText("Models loaded - Ready for analysis")
        self._status_label.setStyleSheet("color: #2e7d32; padding: 4px; font-weight: 500;")

        if hasattr(self, '_analysis_buttons'):
            for button in self._analysis_buttons.values():
                button.setEnabled(True)

        self._update_analysis_status()

        self._logger.info("Switched to analysis mode")

    def _on_run_analysis_clicked(self):
        """Handle run analysis button click."""
        selected_type = self._get_selected_analysis_type()
        if not selected_type:
            return
        
        if not self._show_case_confirmation_dialog(selected_type):
            return
            
        # Collect selected sequences
        sequences = {}
        if hasattr(self, '_sequence_inputs'):
            for seq_name, seq_info in self._sequence_inputs.items():
                if seq_info['selected_path']:
                    sequences[seq_name] = seq_info['selected_path']
        
        
        self._status_label.setText("Running analysis...")
        self._status_label.setStyleSheet("color: #1976d2; padding: 4px;")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # Indeterminate progress
        self._run_button.setEnabled(False)
        
        # Start progress animation
        self._progress_timer.start(100)
        
        try:
            # Use current medical image as primary source
            if not self._current_medical_image:
                self._show_error("No medical image loaded from Patient Studies")
                return
            
            primary_image_path = self._nifti_converter.convert_to_nifti(
                medical_image=self._current_medical_image,
                temp_dir=self._temp_dir
            )
            
            request = AIAnalysisRequest(
                analysis_type=selected_type,
                primary_image_path=primary_image_path,
                additional_sequences={},  # Will be populated if multi-modal analysis
                confidence_threshold=0.75  # Default confidence threshold (75%)
            )
            
            self.analysis_requested.emit(request)
            
        except Exception as e:
            self._logger.error(f"Error creating analysis request: {e}")
            self._show_error(f"Error creating analysis request: {str(e)}")
    
    def _show_error(self, message: str):
        """Show error message and reset UI state."""
        self._status_label.setText(f"Error: {message}")
        self._status_label.setStyleSheet("color: #d32f2f; padding: 4px;")
        self._progress_bar.setVisible(False)
        self._progress_timer.stop()
        # Use proper logic instead of forcing button enabled
        self._update_analysis_status()
    
    def _update_requirements_display(self, analysis_type: AIAnalysisType):
        """Update the requirements display for selected analysis type."""
        requirements = AISequenceRequirement.get_requirements_for_analysis(analysis_type)
        
        if not requirements:
            self._requirements_label.setText("No specific requirements defined.")
            return
        
        req_text = f"<b>Required sequences for {analysis_type.value.replace('_', ' ').title()}:</b><br>"
        for req in requirements:
            status = "✓ Required" if req.is_required else "○ Optional"
            req_text += f"• <b>{req.sequence_name}</b> - {req.description} [{status}]<br>"
        
        self._requirements_label.setText(req_text)
    
    def _add_sequence_file(self):
        """Opin file dialog to add a sequence file."""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Medical Image Sequence")
        file_dialog.setNameFilters([
            "Medical Images (*.dcm *.nii *.nii.gz *.mha *.mhd)",
            "DICOM Files (*.dcm)",
            "NIfTI Files (*.nii *.nii.gz)", 
            "MetaImage Files (*.mha *.mhd)",
            "All Files (*)"
        ])
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = Path(selected_files[0])
                self._add_sequence_to_list(file_path)
    
    def _add_sequence_to_list(self, file_path: Path):
        """Add a sequence file to the list."""
        # Determine sequence type (for multi-sequence analysis)
        sequence_name = self._determine_sequence_type(file_path)
        
        self._selected_sequences[sequence_name] = file_path
        
        item = QListWidgetItem(f"{sequence_name}: {file_path.name}")
        item.setData(Qt.ItemDataRole.UserRole, sequence_name)
        item.setToolTip(str(file_path))
        self._sequence_list.addItem(item)
        
        
        self._validate_analysis_readiness()
    
    def _determine_sequence_type(self, file_path: Path) -> str:
        """Determine sequence type from filename or user selection."""
        filename_lower = file_path.name.lower()
        
        if 'adc' in filename_lower:
            return 'ADC'
        elif 'hbv' in filename_lower or 'dwi' in filename_lower:
            return 'HBV'
        elif 't2' in filename_lower:
            return 'T2W'
        else:
            return 'T2W'
    
    def _remove_selected_sequence(self):
        """Remove selected sequence from list."""
        current_item = self._sequence_list.currentItem()
        if current_item:
            sequence_name = current_item.data(Qt.ItemDataRole.UserRole)
            
            self._selected_sequences.pop(sequence_name, None)
            
            row = self._sequence_list.row(current_item)
            self._sequence_list.takeItem(row)
            
            
            self._validate_analysis_readiness()
    
    def _on_sequence_selection_changed(self):
        """Handle sequence list selection changes."""
        has_selection = self._sequence_list.currentItem() is not None
        self._remove_sequence_btn.setEnabled(has_selection)
    
    def _validate_analysis_readiness(self):
        """
        Check if all requirements are met for analysis (REFACTORED - delegated to service).

        This method now delegates to AnalysisValidationService for comprehensive validation.
        Returns True if analysis is ready, False otherwise (for backward compatibility).
        """
        selected_type = getattr(self, '_selected_analysis_type', None)
        current_image = getattr(self, '_current_medical_image', None)

        # Detect related sequences if needed
        available_sequences = {}
        if current_image and selected_type == AIAnalysisType.CSPCA_DETECTION:
            available_sequences = self._detect_related_sequences(current_image)

        # Delegate to AnalysisValidationService
        validation_result = self._validation_service.validate_analysis_readiness(
            analysis_type=selected_type,
            current_image=current_image,
            available_sequences=available_sequences,
            models_available=self._models_available,
            case_explicitly_selected=self._case_selected_from_input_sequences
        )

        return validation_result.is_valid
    
    def _run_analysis(self):
        """Execute AI analysis."""
        selected_type = None
        for analysis_type, button in self._analysis_buttons.items():
            if button.isChecked():
                selected_type = analysis_type
                break
        
        if not selected_type:
            QMessageBox.warning(self, "Analysis Error", "No analysis type selected.")
            return
        
        if not self._selected_sequences:
            QMessageBox.warning(self, "Analysis Error", "No sequence files selected.")
            return
        
        # Use first sequence as primary, rest as additional
        primary_sequence = next(iter(self._selected_sequences.values()))
        additional_sequences = {k: v for k, v in list(self._selected_sequences.items())[1:]}
        
        request = AIAnalysisRequest(
            analysis_type=selected_type,
            primary_image_path=primary_sequence,
            additional_sequences=additional_sequences,
            apply_refinement=self._apply_refinement_cb.isChecked(),
            save_intermediate_files=self._save_temp_files_cb.isChecked(),
            requested_by="user_interface"
        )
        
        is_valid, errors = request.validate_requirements()
        if not is_valid:
            error_msg = "Analysis request validation failed:\\n" + "\\n".join(errors)
            QMessageBox.critical(self, "Validation Error", error_msg)
            return
        
        # Start analysis in background thread
        self._start_background_analysis(request)
    
    def _start_background_analysis(self, request: AIAnalysisRequest):
        """Start analysis in background thread."""
        self._run_button.setEnabled(False)
        self._cancel_analysis_btn.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # Indeterminate progress
        self._status_label.setText("Starting AI analysis...")
        
        # Start progress animation
        self._progress_timer.start(100)
        
        self._worker_thread = AIAnalysisWorkerThread(self._orchestrator, request)
        self._worker_thread.analysis_completed.connect(self._on_analysis_completed)
        self._worker_thread.analysis_failed.connect(self._on_analysis_failed)
        self._worker_thread.progress_update.connect(self._on_progress_update)
        self._worker_thread.finished.connect(self._on_worker_finished)
        
        self._worker_thread.start()
        
    
    def _cancel_analysis(self):
        """Cancel running analysis."""
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.terminate()
            self._worker_thread.wait(3000)  # Wait up to 3 seconds
            
        self._on_worker_finished()
        self._status_label.setText("Analysis cancelled by user")
        
    
    def _update_progress_animation(self):
        """Update progress bar animation during analysis."""
        if hasattr(self, '_progress_bar') and self._progress_bar:
            self._progress_bar.setRange(0, 0)
    
    def _on_progress_update(self, message: str):
        """Handle progress updates from worker thread."""
        self._status_label.setText(message)
    
    def _on_analysis_completed(self, result: AIAnalysisResult):
        """Handle successful analysis completion."""
        self._current_analysis_result = result
        
        self._display_analysis_results(result)
        
        if hasattr(self, '_show_all_btn'):
            self._show_all_btn.setEnabled(True)
        if hasattr(self, '_hide_all_btn'):
            self._hide_all_btn.setEnabled(True)
        if hasattr(self, '_save_masks_btn'):
            self._save_masks_btn.setEnabled(True)
        
        self._persistence_service.add_prediction_to_history(result)
        
        # Refresh history display
        if hasattr(self, '_history_tree'):
            self._refresh_history_display()
        
        if hasattr(self, '_status_label'):
            self._status_label.setText(
                f"Analysis completed in {result.processing_time_seconds:.1f}s - "
                f"{len(result.segmentations)} structures found"
            )

        self.analysis_completed.emit(result)
        
    
    def _on_analysis_failed(self, error_message: str):
        """Handle analysis failure."""
        logger.error(f"PANEL: _on_analysis_failed called: {error_message}")
        if hasattr(self, '_status_label'):
            self._status_label.setText(f"Analysis failed: {error_message}")
        
        if hasattr(self, '_summary_text'):
            self._summary_text.setPlainText(f"Analysis Failed\n\nError: {error_message}")
        elif hasattr(self, '_results_summary'):
            self._results_summary.setStyleSheet("color: #d32f2f; background-color: #ffebee; padding: 12px; border-radius: 6px;")
            self._results_summary.setText(f"<b>Analysis Failed</b><br><br>Error: {error_message}")
        
        QMessageBox.critical(
            self, 
            "Analysis Failed", 
            f"The AI analysis could not be completed:\\n\\n{error_message}"
        )
        
        self._logger.error(f"Analysis failed: {error_message}")
    
    def _on_worker_finished(self):
        """Clean up after worker thread finishes."""
        if hasattr(self, '_cancel_analysis_btn'):
            self._cancel_analysis_btn.setEnabled(False)
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setVisible(False)
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        
        # Clean up worker thread
        if self._worker_thread:
            self._worker_thread.deleteLater()
            self._worker_thread = None
        
        self._update_analysis_status()
    
    def _display_analysis_results(self, result: AIAnalysisResult):
        """Display analysis results in the simplified UI."""
        try:
            summary = result.get_analysis_summary()
            
            summary_text = f"<b>Analysis Complete!</b><br><br>"
            summary_text += f"<b>Type:</b> {summary['analysis_type'].replace('_', ' ').title()}<br>"
            summary_text += f"<b>Structures Found:</b> {summary['segmentations_count']}<br>"
            summary_text += f"<b>Overall Confidence:</b> {summary['overall_confidence']}<br>"
            summary_text += f"<b>Processing Time:</b> {summary['processing_time']}<br>"
            
            if summary['total_volume_mm3'] > 0:
                summary_text += f"<b>Total Volume:</b> {summary['total_volume_mm3']:.1f} mm³<br>"
            
            if summary['structures_detected']:
                summary_text += f"<br><b>Detected Structures:</b><br>"
                for structure in summary['structures_detected']:
                    structure_name = structure.replace('_', ' ').title()
                    summary_text += f"• {structure_name}<br>"
            
            if hasattr(self, '_summary_text'):
                import re
                plain_text = re.sub('<[^<]+?>', '', summary_text).replace('<br>', '\n')
                self._summary_text.setPlainText(plain_text)
            elif hasattr(self, '_results_summary'):
                # Fallback to simplified interface
                self._results_summary.setStyleSheet("color: #2e7d32; background-color: #e8f5e8; padding: 12px; border-radius: 6px;")
                self._results_summary.setText(summary_text)
            
            if hasattr(self, '_results_section'):
                self._results_section.setVisible(True)
            
            if hasattr(self, '_results_tree'):
                # Original interface - populate the tree widget
                self._populate_results_tree(result)
            elif hasattr(self, '_overlay_controls_layout'):
                self._create_overlay_controls(result)
            
            
        except Exception as e:
            self._logger.error(f"Error displaying analysis results: {e}")
            if hasattr(self, '_summary_text'):
                self._summary_text.setPlainText(f"Error displaying results: {str(e)}")
            elif hasattr(self, '_results_summary'):
                self._results_summary.setStyleSheet("color: #d32f2f; background-color: #ffebee; padding: 12px; border-radius: 6px;")
                self._results_summary.setText(f"Error displaying results: {str(e)}")
    
    def _populate_results_tree(self, result: AIAnalysisResult):
        """Populate the results tree widget with analysis results (original interface)."""
        try:
            from PyQt6.QtWidgets import QTreeWidgetItem
            from PyQt6.QtCore import Qt
            
            self._results_tree.clear()
            
            for i, segmentation in enumerate(result.segmentations):
                item = QTreeWidgetItem(self._results_tree)
                
                # Structure name
                structure_name = segmentation.anatomical_region.value.replace('_', ' ').title()
                item.setText(0, structure_name)
                
                confidence = getattr(segmentation, 'confidence_score', 0.0)
                item.setText(1, f"{confidence:.2f}")
                
                # Volume (from overlay data if available)
                volume_text = "N/A"
                if i < len(result.overlay_data):
                    volume_mm3 = result.overlay_data[i].volume_mm3
                    if volume_mm3 > 0:
                        volume_text = f"{volume_mm3:.1f}"
                item.setText(2, volume_text)
                
                # Visibility checkbox
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(3, Qt.CheckState.Checked)
                
                # Store overlay ID for visibility control
                overlay_id = f"ai_{result.analysis_type.value}_{segmentation.anatomical_region.value}"
                item.setData(0, Qt.ItemDataRole.UserRole, overlay_id)
            
            
        except Exception as e:
            self._logger.error(f"Error populating results tree: {e}")
    
    def _create_overlay_controls(self, result: AIAnalysisResult):
        """Create dynamic overlay controls for each segmentation."""
        try:
            for i in reversed(range(self._overlay_controls_layout.count())):
                child = self._overlay_controls_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            for i, overlay in enumerate(result.overlay_data):
                overlay_control = self._create_single_overlay_control(overlay, result.analysis_type)
                self._overlay_controls_layout.addWidget(overlay_control)
            
            
        except Exception as e:
            self._logger.error(f"Error creating overlay controls: {e}")
    
    def _create_single_overlay_control(self, overlay, analysis_type):
        """Create control widget for a single overlay."""
        from PyQt6.QtWidgets import QFrame, QCheckBox, QHBoxLayout, QVBoxLayout, QLabel
        from PyQt6.QtGui import QFont
        
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid #ddd; border-radius: 4px; margin: 2px; background: #fafafa; }")
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        
        # Visibility checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        
        checkbox.stateChanged.connect(lambda state: self._on_overlay_checkbox_changed(checkbox, overlay))
        
        # Structure info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        name_label = QLabel(overlay.name)
        name_label.setFont(QFont("", 10, QFont.Weight.DemiBold))
        
        details_text = f"Confidence: {overlay.confidence_score:.2f}"
        if overlay.volume_mm3 > 0:
            details_text += f" | Volume: {overlay.volume_mm3:.1f} mm³"
        
        details_label = QLabel(details_text)
        details_label.setFont(QFont("", 8))
        details_label.setStyleSheet("color: #666;")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(details_label)
        
        layout.addWidget(checkbox)
        layout.addLayout(info_layout, 1)
        
        # Store overlay data for future reference
        frame._overlay_data = overlay
        frame._checkbox = checkbox
        
        return frame
    
    def _on_overlay_visibility_changed(self, item, column: int):
        """Handle overlay visibility checkbox changes in QTreeWidget (original interface)."""
        if column == 3:  # Visibility column
            try:
                from PyQt6.QtCore import Qt
                overlay_id = item.data(0, Qt.ItemDataRole.UserRole)
                is_visible = item.checkState(3) == Qt.CheckState.Checked
                
                if overlay_id:
                    self.overlay_visibility_changed.emit(overlay_id, is_visible)
                    
            except Exception as e:
                self._logger.error(f"Error handling tree overlay visibility change: {e}")
    
    def _on_overlay_checkbox_changed(self, checkbox, overlay_data):
        """Handle overlay visibility checkbox changes in simplified UI."""
        try:
            overlay_id = f"ai_{overlay_data.anatomical_region.value}_{overlay_data.name}"
            is_visible = checkbox.isChecked()
            
            self.overlay_visibility_changed.emit(overlay_id, is_visible)
            
            
        except Exception as e:
            self._logger.error(f"Error handling overlay visibility change: {e}")
    
    def _toggle_all_overlays(self, visible: bool):
        """Show or hide all overlays - works with both interface types."""
        try:
            if hasattr(self, '_results_tree'):
                # Original interface - toggle tree checkboxes
                from PyQt6.QtCore import Qt
                for i in range(self._results_tree.topLevelItemCount()):
                    item = self._results_tree.topLevelItem(i)
                    item.setCheckState(3, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
                    
            elif hasattr(self, '_overlay_controls_layout'):
                for i in range(self._overlay_controls_layout.count()):
                    widget = self._overlay_controls_layout.itemAt(i).widget()
                    if widget and hasattr(widget, '_checkbox'):
                        widget._checkbox.setChecked(visible)
            
            
        except Exception as e:
            self._logger.error(f"Error toggling all overlays: {e}")
    
    def _save_masks(self):
        """Save generated masks to file."""
        if not self._current_analysis_result:
            return
        
        format_text = self._save_format_combo.currentText()
        if ".nii.gz" in format_text:
            extension = ".nii.gz"
        elif ".dcm" in format_text:
            extension = ".dcm"
        elif ".stl" in format_text:
            extension = ".stl"
        elif ".mha" in format_text:
            extension = ".mha"
        else:
            extension = ".nii.gz"
        
        # Opin save dialog
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save AI Analysis Masks")
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setDefaultSuffix(extension.replace(".", ""))
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            save_path = file_dialog.selectedFiles()[0]
            
            self.mask_save_requested.emit(
                self._current_analysis_result.segmentations,
                extension
            )
            
    
    def get_current_result(self) -> Optional[AIAnalysisResult]:
        """Get the current analysis result."""
        return self._current_analysis_result
    
    def clear_results(self):
        """Clear current analysis results."""
        self._current_analysis_result = None
        self._results_tree.clear()
        self._summary_text.setPlainText("No analysis performed yet.")
        
        self._show_all_btn.setEnabled(False)
        self._hide_all_btn.setEnabled(False)
        self._save_masks_btn.setEnabled(False)
    
    def set_current_medical_image(self, medical_image, file_path=None):
        """Set the currently loaded medical image from Patient Studies."""
        # Clean up temporary files from previous case
        if self._current_medical_image != medical_image:
            self._cleanup_temp_files()
        
        self._current_medical_image = medical_image
        self._current_image_file_path = file_path
        self._patient_browser_service.set_current_medical_image(medical_image, file_path)
        self._update_current_case_display()
        
        # User must explicitly select from Input Sequences to enable analysis
        self._case_selected_from_input_sequences = False
        
    
    def set_current_image_file_path(self, file_path):
        """Set the file path of the currently loaded image."""
        self._current_image_file_path = file_path
        if self._current_medical_image:
            self._update_current_case_display()
    
    def _refresh_sequences_display(self):
        """Refresh the Input Sequences display to show newly loaded cases."""
        try:
            if hasattr(self, '_selected_analysis_type') and self._selected_analysis_type:
                self._update_sequence_requirements(self._selected_analysis_type)
        except Exception as e:
            self._logger.error(f"Error refreshing sequences display: {e}")
    
    def get_current_medical_image(self):
        """Get the currently loaded medical image."""
        return self._current_medical_image
    
    def set_patient_browser(self, patient_browser):
        """Set reference to Patient Browser for accessing loaded cases."""
        self._patient_browser = patient_browser
        self._patient_browser_service.set_patient_browser_reference(patient_browser)
    
    def _update_current_case_display(self):
        """Update the display to show information about the currently selected case."""
        if not self._current_medical_image:
            self._status_label.setText("No case selected from Patient Studies")
            self._clear_sequence_inputs()
            return
        
        case_info = f"Current case: {self._current_medical_image.patient_id}"
        if hasattr(self._current_medical_image, 'series_description'):
            case_info += f" - {self._current_medical_image.series_description}"
        
        self._status_label.setText(case_info)
        self._status_label.setStyleSheet("color: #1565c0; font-weight: 500; padding: 4px;")
        
        # Auto-populate sequences from current medical image
        self._auto_populate_sequences_from_current_image()
    
    def _show_case_confirmation_dialog(self, analysis_type):
        """Show confirmation dialog for the case to be analyzed."""
        if not self._current_medical_image:
            QMessageBox.warning(
                self,
                "No Case Selected",
                "Please select and load a case from Patient Studies before running AI analysis."
            )
            return False
        
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm AI Analysis")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        # Case information
        case_info = f"""
        <b>Patient ID:</b> {self._current_medical_image.patient_id}<br>
        <b>Series:</b> {getattr(self._current_medical_image, 'series_description', 'N/A')}<br>
        <b>Analysis Type:</b> {analysis_type.value.replace('_', ' ').title()}<br><br>
        <b>Do you want to proceed with AI analysis for this case?</b>
        """
        
        info_label = QLabel(case_info)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dialog
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    def _clear_sequence_inputs(self):
        """Clear all sequence input displays."""
        if hasattr(self, '_sequence_inputs'):
            for seq_name, seq_info in self._sequence_inputs.items():
                seq_info['path_label'].setText("No file selected")
                seq_info['path_label'].setStyleSheet("color: #666; padding: 6px; border: 1px solid #ddd; border-radius: 4px; background: #fafafa;")
                seq_info['selected_path'] = None
    
    def _auto_populate_sequences_from_current_image(self):
        """Auto-populate sequence inputs from current medical image."""
        if not self._current_medical_image:
            return
        
        if not hasattr(self, '_sequence_inputs') or not self._sequence_inputs:
            return
            
        # Use stored file path if available, otherwise create a placeholder
        if self._current_image_file_path:
            current_image_path = Path(self._current_image_file_path)
        else:
            current_image_path = Path(f"temp_{self._current_medical_image.series_instance_uid}.dcm")
        
        for seq_name, seq_info in self._sequence_inputs.items():
            requirement = seq_info['requirement']
            
            # the DICOM metadata to determine the actual sequence type
            if seq_name == "T2W" or len(self._sequence_inputs) == 1:
                # Auto-populate with current image
                seq_info['selected_path'] = current_image_path
                
                display_path = str(current_image_path.name) if len(str(current_image_path)) > 50 else str(current_image_path)
                seq_info['path_label'].setText(f"✓ {display_path}")
                seq_info['path_label'].setStyleSheet("color: #2d5016; padding: 6px; border: 1px solid #81c784; border-radius: 4px; background: #f1f8e9;")
                
            else:
                # Future enhancement: search for related sequences in the same study
                seq_info['path_label'].setText("Required - Please select file")
                seq_info['path_label'].setStyleSheet("color: #b71c1c; padding: 6px; border: 1px solid #e57373; border-radius: 4px; background: #ffebee;")
                seq_info['selected_path'] = None
    
    def _display_available_cases(self):
        """Display all available cases from Patient Studies."""
        for i in reversed(range(self._sequences_layout.count())):
            child = self._sequences_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if not self._patient_browser:
            no_browser_label = QLabel("Patient Browser not available")
            no_browser_label.setStyleSheet("color: #d32f2f; padding: 8px; font-style: italic;")
            self._sequences_layout.addWidget(no_browser_label)
            return
        
        loaded_images = self._patient_browser_service.get_loaded_cases()
        
        if not loaded_images:
            no_cases_label = QLabel("No cases loaded in Patient Studies")
            no_cases_label.setStyleSheet("color: #666666; padding: 8px; font-style: italic;")
            self._sequences_layout.addWidget(no_cases_label)
            return
        
        for series_uid, medical_image in loaded_images.items():
            case_widget = self._create_case_display_widget(medical_image, series_uid)
            self._sequences_layout.addWidget(case_widget)
    
    def _create_case_display_widget(self, medical_image, series_uid):
        """Create a widget to display a case from Patient Studies."""
        from PyQt6.QtWidgets import QFrame, QRadioButton, QCheckBox
        
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setLineWidth(1)
        layout = QVBoxLayout(frame)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 6, 8, 6)
        
        # Case header
        header_layout = QHBoxLayout()
        
        # Selection radio button
        radio_button = QRadioButton()
        radio_button.setObjectName(f"case_radio_{series_uid}")
        
        # Case info
        case_info = f"Patient: {medical_image.patient_id}"
        if hasattr(medical_image, 'series_description'):
            case_info += f" | Series: {medical_image.series_description}"
        case_info += f" | Modality: {medical_image.modality.value}"
        
        case_label = QLabel(case_info)
        case_label.setFont(QFont("", 9, QFont.Weight.DemiBold))
        
        header_layout.addWidget(radio_button)
        header_layout.addWidget(case_label, 1)
        layout.addLayout(header_layout)
        
        details_text = f"Series UID: {series_uid[:20]}..."
        if hasattr(medical_image, 'acquisition_date'):
            details_text += f" | Date: {medical_image.acquisition_date.strftime('%Y-%m-%d')}"
        
        details_label = QLabel(details_text)
        details_label.setFont(QFont("", 8))
        details_label.setStyleSheet("color: #666666;")
        layout.addWidget(details_label)
        
        # Store reference for selection handling
        radio_button.toggled.connect(
            lambda checked, uid=series_uid: self._on_case_selected(uid, checked)
        )
        
        # Highlight current case if it matches
        if (self._current_medical_image and 
            hasattr(self._current_medical_image, 'series_instance_uid') and
            self._current_medical_image.series_instance_uid == series_uid):
            radio_button.setChecked(True)
            frame.setStyleSheet("QFrame { background-color: #e8f5e8; border: 2px solid #4caf50; }")
        
        return frame
    
    def _on_case_selected(self, series_uid, checked):
        """Handle case selection from the list."""
        if not checked or not self._patient_browser:
            return
        
        loaded_images = self._patient_browser_service.get_loaded_cases()
        if series_uid in loaded_images:
            medical_image = loaded_images[series_uid]
            self.set_current_medical_image(medical_image)
            
            # Note: Do not update analysis status here - only from Input Sequences radio buttons
    
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created for previous analyses (REFACTORED - delegated to service)."""
        # Delegate to NIfTIConverter service
        deleted_count = self._nifti_converter.cleanup_temp_files()

        # Also cleanup the temp directory if empty
        self._nifti_converter.cleanup_temp_directory(
            temp_dir=self._temp_dir,
            remove_if_empty=True
        )

        self._temp_files_created.clear()

        self._logger.info(f"Cleaned up {deleted_count} temporary files")
    
    def _display_loaded_cases(self):
        """Display only T2W sequences from Patient Studies with radio buttons."""
        if not self._patient_browser:
            self._logger.warning("Patient Browser not connected - cannot display loaded cases")
            no_browser_label = QLabel("Patient Browser not available")
            no_browser_label.setStyleSheet("color: #999; padding: 8px; font-style: italic;")
            self._sequences_layout.addWidget(no_browser_label)
            return

        loaded_images = self._patient_browser_service.get_loaded_cases()
        self._logger.info(f"Retrieved {len(loaded_images)} loaded cases from Patient Browser")

        if not loaded_images:
            self._logger.warning("No cases loaded in Patient Studies")
            no_cases_label = QLabel("No cases loaded in Patient Studies")
            no_cases_label.setStyleSheet("color: #999; padding: 8px; font-style: italic;")
            self._sequences_layout.addWidget(no_cases_label)
            return

        # Filter only T2W sequences
        t2w_images = self._filter_t2w_sequences(loaded_images)
        self._logger.info(f"Filtered to {len(t2w_images)} T2W sequences from {len(loaded_images)} total cases")

        if not t2w_images:
            self._logger.warning("No T2W sequences found in loaded cases")
            no_t2w_label = QLabel("No T2W sequences found in loaded cases")
            no_t2w_label.setStyleSheet("color: #999; padding: 8px; font-style: italic;")
            self._sequences_layout.addWidget(no_t2w_label)
            return
        
        # Header
        header_label = QLabel("Select T2W sequence for analysis:")
        header_label.setFont(QFont("", 10, QFont.Weight.DemiBold))
        header_label.setStyleSheet("color: #222; margin-bottom: 8px;")
        self._sequences_layout.addWidget(header_label)
        
        # Radio button group for case selection
        from PyQt6.QtWidgets import QButtonGroup
        self._case_button_group = QButtonGroup()
        
        for i, (series_uid, medical_image) in enumerate(t2w_images.items()):
            case_widget = self._create_case_radio_widget(medical_image, series_uid)
            self._sequences_layout.addWidget(case_widget)
            
            # Auto-select current medical image if it matches
            if (self._current_medical_image and 
                hasattr(self._current_medical_image, 'series_instance_uid') and
                self._current_medical_image.series_instance_uid == series_uid):
                radio_button = case_widget.findChild(QRadioButton)
                if radio_button:
                    radio_button.setChecked(True)
    
    def _filter_t2w_sequences(self, loaded_images):
        """Filter loaded images to show only T2W sequences (REFACTORED - delegated to service)."""
        # Delegate to SequenceDetectionService
        return self._sequence_detector.filter_t2w_sequences(loaded_images)

    
    def _create_case_radio_widget(self, medical_image, series_uid):
        """Create a radio button widget for case selection."""
        from PyQt6.QtWidgets import QFrame, QRadioButton
        
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setLineWidth(1)
        frame.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 4px; margin: 2px; background: #fefefe; }")
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        
        # Radio button
        radio_button = QRadioButton()
        radio_button.setObjectName(f"case_radio_{series_uid}")
        self._case_button_group.addButton(radio_button)
        
        # Case information - emphasize T2W sequence
        case_info = f"Patient: {medical_image.patient_id} | T2W Sequence"
        if hasattr(medical_image, 'series_description') and medical_image.series_description:
            case_info += f" | {medical_image.series_description}"
        
        case_label = QLabel(case_info)
        case_label.setFont(QFont("", 10))
        case_label.setStyleSheet("color: #111;")
        
        radio_button.toggled.connect(
            lambda checked, uid=series_uid: self._on_case_radio_selected(uid, checked)
        )
        
        layout.addWidget(radio_button)
        layout.addWidget(case_label, 1)
        
        return frame
    
    def _on_case_radio_selected(self, series_uid, checked):
        """Handle case radio button selection."""
        if not checked or not self._patient_browser:
            if not checked:
                self._case_selected_from_input_sequences = False
                self._update_analysis_status()
            return
        
        loaded_images = self._patient_browser_service.get_loaded_cases()
        if series_uid in loaded_images:
            medical_image = loaded_images[series_uid]
            self.set_current_medical_image(medical_image)
            
            # Mark that a case was selected from Input Sequences
            self._case_selected_from_input_sequences = True
            
            self._update_analysis_status()
            
            self._sync_viewer_with_selected_case(medical_image)
    
    def _sync_viewer_with_selected_case(self, medical_image):
        """Synchronize the main viewer with the selected case."""
        try:
            if self._patient_browser and medical_image:
                self._patient_browser.cached_image_selected.emit(medical_image)
            else:
                self._logger.debug("Cannot sync viewer: missing patient_browser or medical_image")
        except Exception as e:
            self._logger.error(f"Error synchronizing viewer with selected case: {e}")
    
    def _add_multi_sequence_requirements(self):
        """Add additional sequence requirements for multi-sequence analysis."""
        if not self._current_medical_image:
            return
        
        separator = QLabel("Additional sequences required for csPCa Detection:")
        separator.setFont(QFont("", 9, QFont.Weight.DemiBold))
        separator.setStyleSheet("color: #333; margin-top: 16px; margin-bottom: 8px;")
        self._sequences_layout.addWidget(separator)
        
        info_widget = QFrame()
        info_widget.setFrameStyle(QFrame.Shape.StyledPanel)
        info_widget.setLineWidth(1)
        info_widget.setStyleSheet("QFrame { border: 1px solid #2196f3; border-radius: 4px; background: #e3f2fd; margin: 2px; }")
        
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(10)

        info_icon = QLabel("ℹ")
        info_icon.setStyleSheet("color: #1976d2; font-weight: bold; font-size: 16px;")
        
        info_text = QLabel("ADC and HBV sequences will be loaded automatically when available in Patient Studies")
        info_text.setFont(QFont("", 10))
        info_text.setStyleSheet("color: #1565c0;")
        info_text.setWordWrap(True)
        
        info_layout.addWidget(info_icon)
        info_layout.addWidget(info_text)
        
        self._sequences_layout.addWidget(info_widget)
    
    def _create_detected_sequence_widget(self, seq_name, medical_image):
        """Create widget for automatically detected sequence."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setLineWidth(1)
        frame.setStyleSheet("QFrame { border: 1px solid #4caf50; border-radius: 4px; background: #f8fff8; margin: 2px; }")
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        
        status_label = QLabel("✓")
        status_label.setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 16px;")
        
        # Sequence info
        info_text = f"{seq_name}: {medical_image.patient_id}"
        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            series_desc = medical_image.dicom_metadata.get('SeriesDescription', '')
            if series_desc:
                info_text += f" | {series_desc}"
        
        info_label = QLabel(info_text)
        info_label.setFont(QFont("", 10))
        info_label.setStyleSheet("color: #1b5e20;")
        
        layout.addWidget(status_label)
        layout.addWidget(info_label, 1)
        
        return frame
    
    
    
    def _create_history_section(self) -> QWidget:
        """Create the prediction history section."""
        from PyQt6.QtWidgets import QScrollArea, QTreeWidget, QTreeWidgetItem
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Header
        header_label = QLabel("Prediction History:")
        header_label.setFont(QFont("", 9, QFont.Weight.DemiBold))
        header_label.setStyleSheet("color: #333; margin-bottom: 8px;")
        layout.addWidget(header_label)
        
        # History tree
        self._history_tree = QTreeWidget()
        self._history_tree.setHeaderLabels(["Date", "Patient", "Analysis", "Results", "Confidence"])
        self._history_tree.setMaximumHeight(200)
        self._history_tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #fafafa;
            }
            QTreeWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QTreeWidget::item:selected {
                background: #e3f2fd;
            }
        """)
        
        self._history_tree.itemDoubleClicked.connect(self._on_history_item_double_clicked)
        
        layout.addWidget(self._history_tree)
        
        buttons_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_history_display)
        refresh_btn.setMaximumWidth(80)
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_prediction_history)
        clear_btn.setMaximumWidth(100)
        clear_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        
        buttons_layout.addWidget(refresh_btn)
        buttons_layout.addWidget(clear_btn)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        widget.setLayout(layout)
        
        self._refresh_history_display()
        
        return widget
    
    def _refresh_history_display(self):
        """Refresh the history display."""
        self._history_tree.clear()
        
        for record in self._persistence_service.get_prediction_history(limit=20):  # Show last 20 records
            item = QTreeWidgetItem(self._history_tree)
            
            timestamp = record.get('timestamp')
            if isinstance(timestamp, str):
                from datetime import datetime
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            
            item.setText(0, timestamp.strftime('%Y-%m-%d %H:%M'))
            item.setText(1, record.get('patient_id', 'Unknown'))
            item.setText(2, record.get('analysis_type', 'Unknown').replace('_', ' ').title())
            item.setText(3, f"{record.get('segmentations_count', 0)} structures")
            item.setText(4, f"{record.get('overall_confidence', 0):.2f}")
            
            # Store full record data
            item.setData(0, Qt.ItemDataRole.UserRole, record)
    
    def _on_history_item_double_clicked(self, item, column):
        """Handle double-click on history item."""
        record = item.data(0, Qt.ItemDataRole.UserRole)
        if record:
            self._show_history_details(record)
    
    def _show_history_details(self, record):
        """Show detailed information about a historical prediction."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Prediction Details - {record.get('result_id', 'Unknown')}")
        dialog.setModal(True)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Details text
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        
        details_content = f"""
        <h3>Prediction Details</h3>
        <p><b>Date:</b> {record.get('timestamp', 'Unknown')}</p>
        <p><b>Patient ID:</b> {record.get('patient_id', 'Unknown')}</p>
        <p><b>Series UID:</b> {record.get('series_uid', 'Unknown')}</p>
        <p><b>Analysis Type:</b> {record.get('analysis_type', 'Unknown').replace('_', ' ').title()}</p>
        <p><b>Processing Time:</b> {record.get('processing_time', 0):.1f} seconds</p>
        <p><b>Overall Confidence:</b> {record.get('overall_confidence', 0):.2f}</p>
        <p><b>Structures Detected:</b> {len(record.get('structures_detected', []))}</p>
        <p><b>Clinically Significant:</b> {'Yes' if record.get('clinically_significant', False) else 'No'}</p>
        
        <h4>Detected Structures:</h4>
        <ul>
        """
        
        for structure in record.get('structures_detected', []):
            details_content += f"<li>{structure.replace('_', ' ').title()}</li>"
        
        details_content += "</ul>"
        
        details_text.setHtml(details_content)
        layout.addWidget(details_text)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dialog)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.exec()
    
    def _clear_prediction_history(self):
        """Clear the prediction history."""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all prediction history?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._persistence_service.clear_history()
            self._refresh_history_display()

    def _set_warning_button_style(self):
        """Helper method to consolidate warning button styling."""
        self._run_button.setStyleSheet("""
            QPushButton {
                background-color: #fff3e0;
                color: #f57c00;
                border: 1px solid #ffb74d;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #ffe0b2;
            }
        """)

    def _set_models_not_loaded_button_style(self):
        """Style for when AI models are not loaded."""
        self._run_button.setStyleSheet("""
            QPushButton {
                background-color: #ffebee;
                color: #c62828;
                border: 2px solid #ef5350;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
                border-color: #d32f2f;
            }
        """)

    def _check_model_availability(self) -> bool:
        """Check if AI models are available for analysis."""
        try:
            if self._dynamic_config:
                status = self._dynamic_config.get_model_status()
                return status.get("available_count", 0) > 0

            # Fallback: usar el orchestrator existente for verificar modelos
            if hasattr(self._orchestrator, '_segmentation_service'):
                model_status = self._orchestrator._segmentation_service.get_system_health()
                model_availability = model_status.get("model_availability", {})
                return any(model_availability.values())

            # Último fallback: asumir que no hay modelos disponibles
            return False

        except Exception as e:
            self._logger.error(f"Could not check model availability: {e}")
            return False

    def set_models_available(self, available: bool):
        """Enable/disable the entire panel based on model availability."""
        self._models_available = available

        model_dependent_controls = []

        if hasattr(self, '_run_button'):
            model_dependent_controls.append(self._run_button)
        if hasattr(self, '_show_all_btn'):
            model_dependent_controls.append(self._show_all_btn)
        if hasattr(self, '_hide_all_btn'):
            model_dependent_controls.append(self._hide_all_btn)
        if hasattr(self, '_save_masks_btn'):
            model_dependent_controls.append(self._save_masks_btn)

        if hasattr(self, '_analysis_buttons'):
            for button in self._analysis_buttons.values():
                button.setEnabled(available)
                if not available:
                    button.setStyleSheet("""
                        QRadioButton {
                            color: #999;
                        }
                        QRadioButton::indicator {
                            background-color: #f0f0f0;
                            border: 1px solid #ccc;
                        }
                    """)
                else:
                    button.setStyleSheet("")  # Reset to default styling

        self._update_analysis_status()

    def _check_initial_model_availability(self):
        """Check model availability during initialization."""
        try:
            available = self._check_model_availability()
            self.set_models_available(available)
            self._logger.info(f"Initial model availability check: {available}")
        except Exception as e:
            self._logger.error(f"Error during initial model availability check: {e}")
            self.set_models_available(False)

    def _on_model_config_changed(self, new_status: Dict[str, Any]):
        """Handle configuration changes from dynamic config service."""
        try:
            available_count = new_status.get("available_count", 0)
            models_available = available_count > 0

            self._logger.info(f"Model config changed: {available_count} models available")
            self.set_models_available(models_available)

        except Exception as e:
            self._logger.error(f"Error handling model config change: {e}")
            self.set_models_available(False)

    def closeEvent(self, event):
        """Handle widget close event and cleanup resources."""
        try:
            if self._dynamic_config:
                self._dynamic_config.remove_config_listener(self._on_model_config_changed)

            # Cancel any running analysis
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.terminate()
                self._worker_thread.wait(3000)  # Wait up to 3 seconds

            # Clean up temporary files
            self._cleanup_temp_files()

            self._logger.info("AI Analysis Panel closed and cleaned up")

        except Exception as e:
            self._logger.error(f"Error during AI Analysis Panel cleanup: {e}")

        super().closeEvent(event)
