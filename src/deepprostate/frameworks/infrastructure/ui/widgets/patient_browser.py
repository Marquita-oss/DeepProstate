"""
infrastructure/ui/widgets/patient_browser.py

Patient Browser Panel - Clean architecture implementation with specialized services.
"""

import logging
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QLineEdit, QComboBox, QGroupBox, QSplitter,
    QTextEdit, QProgressBar, QMessageBox, QMenu, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QFont

from deepprostate.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository
from deepprostate.core.domain.services.patient_data_management_service import PatientDataManagementService
from deepprostate.core.domain.services.medical_image_cache_service import MedicalImageCacheService
from deepprostate.core.domain.services.patient_search_service import PatientSearchService, SearchCriteria
from deepprostate.core.domain.services.batch_operation_service import BatchOperationService
from deepprostate.core.domain.services.sequence_analysis_service import SequenceAnalysisService


class PatientTreeWidget(QTreeWidget):
    """Widget of árbol especializado for navegación of pacientes."""
    
    patient_selected = pyqtSignal(str)  # patient_id
    study_selected = pyqtSignal(str)    # study_uid
    series_selected = pyqtSignal(str)   # series_uid
    batch_series_load_requested = pyqtSignal(list)  # List of series_uids for batch loading
    
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_tree()
        self._current_data = {}
        
    def _setup_tree(self):
        """Configura el widget of árbol."""
        # Configure headers
        self.setHeaderLabels(["Name", "Date", "Modality/Sequence", "Description"])
        self.setAlternatingRowColors(True)
        self.setExpandsOnDoubleClick(True)
        
        # Configure selección
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Menú contextual
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Configure columnas
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    
    def populate_tree(self, patient_data: Dict[str, Any]):
        """Puebla el árbol with datos of pacientes."""
        self._current_data = patient_data
        self.clear()
        
        patients = patient_data.get("patients", {})
        
        # Debug logging
        total_patients = len(patients)
        total_studies = sum(len(p.get("studies", {})) for p in patients.values())
        total_series = sum(
            len(s.get("series", {})) 
            for p in patients.values()
            for s in p.get("studies", {}).values()
        )
        
        for patient_id, patient_info in patients.items():
            self._add_patient_to_tree(patient_id, patient_info)
        
        # Expandir todos los nodos by defecto
        self.expandAll()
    
    def _add_patient_to_tree(self, patient_id: str, patient_info: Dict[str, Any]):
        """Añaof un paciente al árbol."""
        patient_item = QTreeWidgetItem(self)
        
        # Datos of the paciente
        patient_name = patient_info.get('patient_name', 'Unknown Patient')
        birth_date = patient_info.get('birth_date', '')
        sex = patient_info.get('sex', '')
        
        # Configure elementos of the paciente
        patient_item.setText(0, f"👤 {patient_name} ({patient_id})")
        patient_item.setText(1, birth_date)
        patient_item.setText(2, sex)
        patient_item.setText(3, f"{len(patient_info.get('studies', {}))} studies")
        
        # Metadata for identificación
        patient_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'patient',
            'patient_id': patient_id,
            'data': patient_info
        })
        
        # Añadir estudios
        studies = patient_info.get('studies', {})
        for study_uid, study_info in studies.items():
            self._add_study_to_tree(patient_item, study_uid, study_info, patient_id)
    
    def _add_study_to_tree(self, parent_item: QTreeWidgetItem, study_uid: str, 
                          study_info: Dict[str, Any], patient_id: str):
        """Añaof un estudio al árbol."""
        study_item = QTreeWidgetItem(parent_item)
        
        # Datos of the estudio
        study_desc = study_info.get('study_description', 'Unknown Study')
        study_date = study_info.get('study_date', '')
        study_time = study_info.get('study_time', '')
        modality = study_info.get('modality', '')
        
        # Formatear fecha y hora
        datetime_str = study_date
        if study_time:
            datetime_str += f" {study_time[:5]}"
        
        # Configure elementos of the estudio
        study_item.setText(0, f"📁 {study_desc}")
        study_item.setText(1, datetime_str)
        study_item.setText(2, modality)
        study_item.setText(3, f"{len(study_info.get('series', {}))} series")
        
        # Metadata for identificación
        study_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'study',
            'patient_id': patient_id,
            'study_uid': study_uid,
            'data': study_info
        })
        
        # Añadir series
        series = study_info.get('series', {})
        for series_uid, series_info in series.items():
            self._add_series_to_tree(study_item, series_uid, series_info, patient_id, study_uid)
    
    def _add_series_to_tree(self, parent_item: QTreeWidgetItem, series_uid: str,
                           series_info: Dict[str, Any], patient_id: str, study_uid: str):
        """Añaof una serie al árbol."""
        series_item = QTreeWidgetItem(parent_item)
        
        # Datos of la serie
        series_desc = series_info.get('series_description', 'Unknown Series')
        series_number = series_info.get('series_number', '')
        modality = series_info.get('modality', '')
        images_count = series_info.get('images_count', 0)
        slice_thickness = series_info.get('slice_thickness', '')
        
        # Formatear descripción
        display_desc = f"{series_desc}"
        if series_number:
            display_desc += f" #{series_number}"
        
        # Configure elementos of la serie
        series_item.setText(0, display_desc)
        series_item.setText(1, f"{images_count} imgs")
        series_item.setText(2, modality)
        
        # Detalles técnicos
        details = []
        if slice_thickness:
            details.append(f"{slice_thickness}mm")
        if images_count:
            details.append(f"{images_count} images")
        
        series_item.setText(3, " | ".join(details))
        
        # Add mask information as child items
        masks_count = series_info.get('masks_count', 0)
        has_masks = series_info.get('has_masks', False)
        
        if has_masks and masks_count > 0:
            mask_item = QTreeWidgetItem(series_item)
            mask_item.setText(0, f"{masks_count} Associated Mask(s)")
            mask_item.setText(1, "Auto-detected")
            mask_item.setText(2, "MASK")
            
            # Store mask paths for potential loading
            associated_masks = series_info.get('associated_masks', [])
            mask_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'mask_collection',
                'mask_paths': associated_masks,
                'parent_series_uid': series_uid,
                'parent_patient_id': patient_id,
                'parent_study_uid': study_uid
            })
        
        # Metadata for identificación
        series_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'series',
            'patient_id': patient_id,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'data': series_info
        })
    
    def _on_selection_changed(self):
        """Handles cambios in la selección."""
        current_item = self.currentItem()
        if not current_item:
            return
        
        item_data = current_item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return
        
        item_type = item_data.get('type')

        if item_type == 'patient':
            self.patient_selected.emit(item_data.get('patient_id', ''))
        elif item_type == 'study':
            self.study_selected.emit(item_data.get('study_uid', ''))
        # NOTE: series_selected se emite solo in doble-click, no in selección simple for evitar duplicación
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handles doble click in elementos."""
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return

        # Emitir señal apropiada basada in the tipo
        item_type = item_data.get('type')
        if item_type == 'series':
            series_uid = item_data.get('series_uid', '')
            self.series_selected.emit(series_uid)
        elif item_type == 'study':
            # Para estudios, cargar la primera serie disponible o expandir el estudio
            study_uid = item_data.get('study_uid', '')
            if study_uid:
                self._load_first_series_from_study(item, study_uid)
        elif item_type == 'patient':
            # Para pacientes, expandir/colapsar el nodo
            if item.isExpanded():
                item.setExpanded(False)
            else:
                item.setExpanded(True)

    def _load_first_series_from_study(self, study_item: QTreeWidgetItem, study_uid: str):
        """Carga la primera serie disponible of un estudio."""
        try:
            # Buscar la primera serie hijo
            for i in range(study_item.childCount()):
                child_item = study_item.child(i)
                child_data = child_item.data(0, Qt.ItemDataRole.UserRole)

                if child_data and child_data.get('type') == 'series':
                    series_uid = child_data.get('series_uid', '')
                    if series_uid:
                        self._logger.debug(f"Loading first series from study: {series_uid}")
                        self.series_selected.emit(series_uid)
                        return

            # Si no hay series directas, expandir el estudio for mostrar las series
            if not study_item.isExpanded():
                study_item.setExpanded(True)

        except Exception as e:
            self._logger.error(f"Error loading first series from study {study_uid}: {e}")

    def _show_context_menu(self, position):
        """Muestra menú contextual."""
        item = self.itemAt(position)
        if not item:
            return
        
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return
        
        menu = QMenu(self)
        item_type = item_data.get('type')
        
        if item_type == 'patient':
            # Patient level: No actions needed (info available in Image Information panel)
            menu.addAction("ℹ️ Info available in Image Information panel").setEnabled(False)
        elif item_type == 'study':
            # Study level: Only functional loading action
            menu.addAction("⚡ Load All Series", lambda: self._load_all_series(item_data))
        elif item_type == 'series':
            # Series level: Only functional loading action (info available in Image Information panel)
            menu.addAction("📁 Load Series", lambda: self._load_series(item_data))
        
        if menu.actions():
            menu.exec(self.mapToGlobal(position))
    
    
    
    def _load_all_series(self, item_data):
        """
        Carga todas las series of un estudio - Nueva funcionalidad valiosa.
        Reutiliza señal batch_series_load_requested existente.
        """
        try:
            if not item_data:
                self._logger.error("No item data provided for batch series loading")
                return
                
            study_data = item_data.get('data', {})
            series_dict = study_data.get('series', {})
            
            if not series_dict:
                self._logger.warning("No series found in study for batch loading")
                return
            
            # Extraer series_uids of todas las series of the estudio
            series_uids = []
            for series_uid, series_info in series_dict.items():
                if series_uid and isinstance(series_uid, str):
                    series_uids.append(series_uid)
            
            if series_uids:
                self._logger.info(f"Loading {len(series_uids)} series from study via context menu")
                self.batch_series_load_requested.emit(series_uids)
            else:
                self._logger.warning("No valid series UIDs found for batch loading")
                
        except Exception as e:
            self._logger.error(f"Error in batch series loading: {e}")
    
    def _load_series(self, item_data):
        """
        Carga serie individual - Reutiliza lógica of the doble-click.
        Mantiene consistencia entre doble-click y menú contextual.
        """
        try:
            if not item_data:
                self._logger.error("No item data provided for series loading")
                return
                
            series_uid = item_data.get('series_uid', '')
            if series_uid:
                self._logger.debug(f"Loading series via context menu: {series_uid}")
                self.series_selected.emit(series_uid)
            else:
                self._logger.warning("No series_uid found in item_data for context menu load")
                
        except Exception as e:
            self._logger.error(f"Error in series loading: {e}")


class PatientBrowserPanel(QWidget):
    """
    Patient Browser Panel refactored with Clean Architecture principles.
    Maintains EXACT original visual appearance while delegating to specialized services.
    """
    
    # Signals for main window integration - EXACT same as original
    image_selected = pyqtSignal(str)  # series_uid (for file-based loading)
    cached_image_selected = pyqtSignal(object)  # MedicalImage object (for cached reloading without patient browser updates)
    multi_modal_sequence_selected = pyqtSignal(str, str)  # study_analysis_path, modality_key (for multi-modal reloading)
    filename_based_sequence_selected = pyqtSignal(str, str)  # file_path, sequence_type (for filename-based studies)
    patient_changed = pyqtSignal(str)  # patient_id
    batch_series_load_requested = pyqtSignal(list)  # List of series_uids to load
    batch_studies_load_requested = pyqtSignal(list)  # List of study_uids to load
    
    def __init__(self, repository: DICOMImageRepository):
        super().__init__()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize specialized services (Clean Architecture)
        self._data_service = PatientDataManagementService(repository)
        self._cache_service = MedicalImageCacheService(max_cached_images=50, max_memory_mb=2048)
        self._search_service = PatientSearchService()
        self._batch_service = BatchOperationService(repository)
        self._sequence_service = SequenceAnalysisService()
        
        # UI Components - EXACT same names as original
        self.search_edit: Optional[QLineEdit] = None
        self.search_button: Optional[QPushButton] = None
        self.modality_filter: Optional[QComboBox] = None
        self.clear_filters_button: Optional[QPushButton] = None
        self.filter_status_label: Optional[QLabel] = None
        self.refresh_button: Optional[QPushButton] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.progress_detail_label: Optional[QLabel] = None
        self.cache_status_label: Optional[QLabel] = None
        self.patient_tree: Optional[PatientTreeWidget] = None
        self.clear_button: Optional[QPushButton] = None
        
        # State management
        self.current_patient_data = {}
        self._current_search_criteria: Optional[SearchCriteria] = None
        
        # Setup UI and connections - EXACT same as original
        self._setup_ui()
        self._setup_connections()
        
        # Setup timers
        self._setup_timers()
        
        # Load initial data
        self._load_initial_data()
    
    def _setup_ui(self):
        """Configura la interfaz of the panel - EXACT same as original."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Grupo of búsqueda y filtros - EXACT same styling
        search_group = QGroupBox("Search & Filters ")
        layout.addWidget(search_group)
        
        search_layout = QVBoxLayout(search_group)
        
        # Barra of búsqueda
        search_row = QHBoxLayout()
        search_layout.addLayout(search_row)
        
        search_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Patient name, ID, ...")
        search_row.addWidget(self.search_edit)
        
        # Import MedicalButton for EXACT same appearance
        try:
            from deepprostate.frameworks.infrastructure.ui.components import MedicalButton
            self.search_button = MedicalButton(
                text="🔍", 
                button_type="default", 
                size="compact"
            )
            self.search_button.setFixedWidth(40)
        except ImportError:
            # Fallback if MedicalButton not available
            self.search_button = QPushButton("🔍")
            self.search_button.setFixedWidth(40)
        
        search_row.addWidget(self.search_button)
        
        # Filtros - EXACT same as original
        filter_row = QHBoxLayout()
        search_layout.addLayout(filter_row)
        
        filter_row.addWidget(QLabel("Modality:"))
        self.modality_filter = QComboBox()
        self.modality_filter.addItems(["All", "CT", "MRI", "US", "XR", "PT"])
        self.modality_filter.setToolTip("Filter cases by imaging modality.\nNote: Single cases may contain multiple modalities/sequences\n(e.g., T1, T2, FLAIR, ADC for MRI studies)")
        filter_row.addWidget(self.modality_filter)
        
        # Clear filters button - EXACT same
        self.clear_filters_button = QPushButton(" Clear")
        self.clear_filters_button.setToolTip("Clear all search and filter criteria")
        self.clear_filters_button.setFixedWidth(80)
        filter_row.addWidget(self.clear_filters_button)
        
        # Filter status label - EXACT same styling
        self.filter_status_label = QLabel(" All cases shown")
        self.filter_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        search_layout.addWidget(self.filter_status_label)
        
        # Multi-modal help text - EXACT same
        help_text = QLabel(" Cases may contain multiple modalities/sequences\n(e.g., MRI with T1, T2, FLAIR, ADC, DWI)")
        help_text.setStyleSheet("color: #888; font-size: 9px; padding: 3px; font-style: italic;")
        help_text.setWordWrap(True)
        search_layout.addWidget(help_text)
        
        # Botón of actualización
        self.refresh_button = QPushButton(" Refresh")
        search_layout.addWidget(self.refresh_button)
        
        # Árbol of navegación - EXACT same
        tree_group = QGroupBox("Patient Studies")
        layout.addWidget(tree_group)
        
        tree_layout = QVBoxLayout(tree_group)
        
        # Enhanced progress feedback - EXACT same styling
        progress_container = QVBoxLayout()
        
        # Main progress bar with EXACT same styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_container.addWidget(self.progress_bar)
        
        # Detailed progress label - EXACT same styling
        self.progress_detail_label = QLabel()
        self.progress_detail_label.setVisible(False)
        self.progress_detail_label.setStyleSheet("""
            QLabel {
                color: #bbb;
                font-size: 11px;
                padding: 2px 5px;
                background-color: rgba(45, 45, 45, 0.8);
                border-radius: 3px;
            }
        """)
        self.progress_detail_label.setWordWrap(True)
        progress_container.addWidget(self.progress_detail_label)
        
        # Cache status indicator - EXACT same styling
        self.cache_status_label = QLabel()
        self.cache_status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 10px;
                padding: 2px 5px;
            }
        """)
        self._update_cache_status_display()
        progress_container.addWidget(self.cache_status_label)
        
        tree_layout.addLayout(progress_container)
        
        # Widget of árbol
        self.patient_tree = PatientTreeWidget()
        tree_layout.addWidget(self.patient_tree)
        
        # Botón of acción - Solo Clear All
        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout)
        
        # Botón for limpiar casos
        self.clear_button = QPushButton(" Clear All")
        self.clear_button.setToolTip("Clear all loaded cases")
        buttons_layout.addWidget(self.clear_button)
    
    def _setup_connections(self):
        """Configura las conexiones entre widgets - EXACT same as original."""
        # Conexiones of búsqueda
        self.search_button.clicked.connect(self._perform_search)
        self.search_edit.returnPressed.connect(self._perform_search)
        self.refresh_button.clicked.connect(self._refresh_view)
        
        # Conexiones of filtros
        self.modality_filter.currentTextChanged.connect(self._apply_filters)
        self.clear_filters_button.clicked.connect(self._clear_all_filters)
        
        # Conexiones of the árbol
        self.patient_tree.patient_selected.connect(self._on_patient_selected)
        self.patient_tree.study_selected.connect(self._on_study_selected)
        # Connect series_selected to cache-first handler (v17 compatible)
        self.patient_tree.series_selected.connect(self._on_series_selected)
        self.patient_tree.batch_series_load_requested.connect(self._on_batch_series_load_requested)
        
        # Conexión of the botón Clear All
        self.clear_button.clicked.connect(self._confirm_clear_all_cases)
        
        # Batch operation signals
        self._batch_service.operation_started.connect(self._on_batch_operation_started)
        self._batch_service.progress_updated.connect(self._on_batch_progress_updated)
        self._batch_service.operation_completed.connect(self._on_batch_operation_completed)
        self._batch_service.operation_failed.connect(self._on_batch_operation_failed)
    
    def _setup_timers(self):
        """Setup periodic update timers."""
        # Cache status update timer
        self._cache_status_timer = QTimer()
        self._cache_status_timer.timeout.connect(self._update_cache_status_display)
        self._cache_status_timer.start(10000)  # Update every 10 seconds
        
        # Memory cleanup timer
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._periodic_memory_cleanup)
        self._cleanup_timer.start(60000)  # Cleanup every minute
    
    def _load_initial_data(self):
        """Carga datos iniciales of the repositorio - EXACT same as original."""
        # No cargar datos demo by defecto - solo mostrar tree vacío
        empty_data = {"patients": {}}
        self.current_patient_data = empty_data
        self.patient_tree.populate_tree(empty_data)
        self._logger.info("Patient Studies initialized - waiting for case to be loaded")
    
    # Delegate all business logic to services while maintaining exact UI behavior
    @pyqtSlot()
    def _perform_search(self):
        """Perform search using PatientSearchService - maintains exact original behavior."""
        try:
            search_text = self.search_edit.text().strip()
            modality = self.modality_filter.currentText()
            
            # Create search criteria using service
            criteria = self._search_service.create_search_criteria(
                free_text=search_text if search_text else None,
                modality=modality if modality != "All" else None,
                case_insensitive=True
            )
            
            # Apply search filters using service
            filtered_data, filter_result = self._search_service.apply_search_filters(
                self.current_patient_data, criteria
            )
            
            # Update tree with filtered data
            self.patient_tree.populate_tree(filtered_data)
            
            # Update filter status - exact same format as original
            if filter_result.filter_applied:
                self.filter_status_label.setText(
                    f" Showing {filter_result.filtered_patients} of {filter_result.total_patients} patients"
                )
            else:
                self.filter_status_label.setText(" All cases shown")
            
            self._current_search_criteria = criteria
            
        except Exception as e:
            self._logger.error(f"Search error: {e}")
            self.filter_status_label.setText(f" Search error: {e}")
    
    @pyqtSlot()
    def _apply_filters(self):
        """Apply filters - delegates to service."""
        if self._current_search_criteria:
            self._perform_search()
    
    @pyqtSlot()
    def _clear_all_filters(self):
        """Clear all filters - exact same behavior as original."""
        self.search_edit.clear()
        self.modality_filter.setCurrentText("All")
        self._current_search_criteria = None
        
        # Show all data
        self.patient_tree.populate_tree(self.current_patient_data)
        self.filter_status_label.setText(" All cases shown")
    
    @pyqtSlot()
    def _refresh_view(self):
        """Refresh view - delegates to data service."""
        try:
            # Use data service to refresh
            self._show_loading("Refreshing patient data...")
            
            # Reload data
            patient_data = self._data_service.get_current_patient_data()
            self.current_patient_data = patient_data
            
            # Update tree
            self.patient_tree.populate_tree(patient_data)
            self._update_modality_filter()
            
            self._hide_loading()
            
        except Exception as e:
            self._logger.error(f"Refresh error: {e}")
            self._hide_loading()
    
    def _show_loading(self, message: str, detailed_message: str = None):
        """Show loading - exact same as original."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        if detailed_message:
            self.progress_detail_label.setText(detailed_message)
            self.progress_detail_label.setVisible(True)
        else:
            self.progress_detail_label.setVisible(False)
    
    def _hide_loading(self):
        """Hiof loading - exact same as original."""
        self.progress_bar.setVisible(False)
        self.progress_detail_label.setVisible(False)
        self.progress_bar.setRange(0, 100)
    
    def _update_cache_status_display(self):
        """Update cache status - delegates to cache service."""
        stats = self._cache_service.get_cache_statistics()
        self.cache_status_label.setText(
            f"Cache: {stats['cached_images_count']}/{stats['max_cached_images']} | "
            f"{stats['total_memory_mb']:.0f}MB ({stats['memory_usage_percent']:.0f}%)"
        )
    
    def _periodic_memory_cleanup(self):
        """Perform periodic memory cleanup using cache service."""
        if self._cache_service.should_cleanup_memory():
            cleaned = self._cache_service.cleanup_old_images(max_age_minutes=30)
            if cleaned > 0:
                self._logger.info(f"Cleaned up {cleaned} old cached images")
    
    def _update_modality_filter(self):
        """Update modality filter - delegates to search service."""
        modalities = self._search_service.get_available_modalities(self.current_patient_data)
        
        current_text = self.modality_filter.currentText()
        self.modality_filter.clear()
        self.modality_filter.addItem("All")
        
        for modality in sorted(modalities):
            self.modality_filter.addItem(modality)
        
        # Restore previous selection
        index = self.modality_filter.findText(current_text)
        if index >= 0:
            self.modality_filter.setCurrentIndex(index)
    
    # Signal handlers - exact same behavior as original
    @pyqtSlot(str)
    def _on_patient_selected(self, patient_id: str):
        """Handle patient selection."""
        self.patient_changed.emit(patient_id)
        self._update_button_states(True, 'patient')
    
    @pyqtSlot(str)
    def _on_study_selected(self, study_uid: str):
        """Handle study selection."""
        self._update_button_states(True, 'study')
    
    
    def _on_batch_series_load_requested(self, series_uids: list):
        """
        Handle batch series loading request from tree widget.
        Forward to main window via existing signal.
        """
        if series_uids:
            self._logger.info(f"Forwarding batch load request for {len(series_uids)} series to main window")
            self.batch_series_load_requested.emit(series_uids)
        else:
            self._logger.warning("Empty series list received for batch loading")
    
    def _update_button_states(self, enabled: bool, item_type: str):
        """Update button states based on selection. Clear All button always enabled."""
        # Clear All button always stays enabled - no state change needed
        pass
    
    @pyqtSlot()
    def _confirm_clear_all_cases(self):
        """Confirm and clear all cases."""
        reply = QMessageBox.question(
            self, "Clear All Cases", 
            "Are you sure you want to clear all loaded cases?\nThis will free memory but you'll need to reload cases.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._clear_all_cases()
    
    def _clear_all_cases(self):
        """Clear all cases - delegates to services."""
        # Clear data using data service
        self._data_service.clear_patient_data()
        
        # Clear cache using cache service  
        cleared_count = self._cache_service.clear_cache()
        
        # Update UI
        empty_data = {"patients": {}}
        self.current_patient_data = empty_data
        self.patient_tree.populate_tree(empty_data)
        
        self.filter_status_label.setText(f" Cleared all cases (freed {cleared_count} cached images)")
        self._update_button_states(False, '')
    
    # Batch operation handlers
    @pyqtSlot(str, str)
    def _on_batch_operation_started(self, operation_id: str, description: str):
        """Handle batch operation started."""
        self._show_loading("Batch loading...", description)
    
    @pyqtSlot(str, int, str)
    def _on_batch_progress_updated(self, operation_id: str, progress: int, message: str):
        """Handle batch progress update."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(progress)
        self.progress_detail_label.setText(message)
        self.progress_detail_label.setVisible(True)
    
    @pyqtSlot(str, dict)
    def _on_batch_operation_completed(self, operation_id: str, results: Dict[str, Any]):
        """Handle batch operation completion."""
        self._hide_loading()
        
        loaded_count = len(results.get('loaded_series', [])) + len(results.get('loaded_studies', []))
        failed_count = len(results.get('failed_series', [])) + len(results.get('failed_studies', []))
        
        self.filter_status_label.setText(f" Batch complete: {loaded_count} loaded, {failed_count} failed")
    
    @pyqtSlot(str, str)
    def _on_batch_operation_failed(self, operation_id: str, error_message: str):
        """Handle batch operation failure."""
        self._hide_loading()
        self.filter_status_label.setText(f" Batch failed: {error_message}")
    
    # Public interface methods - exact same as original for compatibility
    def get_loaded_cases_summary(self) -> Dict[str, int]:
        """Get summary of loaded cases."""
        return self._data_service.get_data_statistics()
    
    def refresh_data(self):
        """Refresh patient data."""
        self._refresh_view()
    
    def get_cached_medical_image(self, series_uid: str):
        """Get cached medical image."""
        return self._cache_service.get_cached_image(series_uid)
    
    def add_medical_image_to_cache(self, series_uid: str, medical_image) -> None:
        """Add medical image to cache."""
        try:
            success = self._cache_service.cache_image(series_uid, medical_image)
            if success:
                self._logger.debug(f"Cached image for series_uid: {series_uid}")
            else:
                self._logger.error(f"Failed to cache image for series_uid: {series_uid}")
        except Exception as e:
            self._logger.error(f"Error caching image for series_uid {series_uid}: {e}")

    @pyqtSlot(str)
    def _on_series_selected(self, series_uid: str):
        """Handle series selection with cache-first logic (v17 compatible)."""
        self._logger.debug(f"Series selected with cache-first logic: {series_uid}")

        # Check cache first using cache service
        cached_image = self._cache_service.get_cached_image(series_uid)
        if cached_image:
            self._logger.debug(f"Found cached image for series: {series_uid}")
            self.cached_image_selected.emit(cached_image)
        else:
            self._logger.debug(f"Image not cached, starting async load for series: {series_uid}")
            self.image_selected.emit(series_uid)

    def shutdown(self):
        """Shutdown the panel and cleanup resources."""
        if hasattr(self, '_cache_status_timer'):
            self._cache_status_timer.stop()
        if hasattr(self, '_cleanup_timer'):
            self._cleanup_timer.stop()
        self._batch_service.shutdown()
        self._logger.debug("Patient browser panel shut down")