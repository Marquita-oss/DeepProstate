"""
infrastructure/ui/widgets/manual_editing_panel.py

Panel orquestador for herramientas of edición manual of imágenes médicas.
Coordinates herramientas modulares manteniendo arquitectura limpia.

RESPONSABILIDAD: Orquestación y coordinación of herramientas especializadas.
NO implementa lógica of herramientas, solo las coordina.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QFrame, QGroupBox, QSplitter, QPushButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QSlider, QGridLayout
)
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
import logging
from typing import Optional, Dict, Any, List

# Importar herramientas modulares
from .tools import (
    DistanceMeasurementTool, AngleMeasurementTool, ROITool,
    BrushTool, EraserTool, FillTool
)
from .shared import BaseTool


class ManualEditingPanel(QWidget):
    """
    Panel orquestador for herramientas of edición manual modularizadas.
    
    ARQUITECTURA LIMPIA:
    - Coordinates herramientas especializadas
    - Mantiene separación entre herramientas of image base y segmentación
    - Proporciona interfaz unificada without duplicar lógica
    - Emite señales consolidadas for comunicación with ImageViewer
    
    RESPONSABILIDADES:
    - Orquestación of herramientas
    - Gestión of UI y layout
    - Consolidación of señales
    - Coordinatesción of estado entre herramientas
    """
    
    measurement_mode_changed = pyqtSignal(str)
    segmentation_tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    clear_measurements_requested = pyqtSignal()
    clear_segmentations_requested = pyqtSignal()
    measurement_selected = pyqtSignal(dict)
    measurement_delete_requested = pyqtSignal(dict)
    active_segmentation_changed = pyqtSignal(str, str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self._measurement_tools: Dict[str, BaseTool] = {}
        self._segmentation_tools: Dict[str, BaseTool] = {}

        self._active_measurement_tool: Optional[str] = None
        self._active_segmentation_tool: Optional[str] = None
        self._active_segmentation_name: Optional[str] = None
        self._active_overlay_id: Optional[str] = None

        self._history_stack: List[Dict[str, Any]] = []
        self._history_index: int = -1
        self._max_history_size: int = 50
        self._original_segmentation_data: Optional[Dict[str, Any]] = None

        self._current_slice_info: Dict[str, Any] = {
            "plane": "axial",
            "index": 0,
            "total_slices": 1,
            "position_mm": 0.0
        }

        self._selected_measurement_row = -1
        self._selected_measurement_data = None
        self._current_measurements_data = []
        
        self._setup_tools()
        self._setup_ui()
        self._setup_connections()
        

    def _setup_tools(self) -> None:
        """Inicializa todas las herramientas modulares."""
        self._measurement_tools = {
            "distance": DistanceMeasurementTool(self),
            "angle": AngleMeasurementTool(self),
            "roi": ROITool(self)
        }

        self._segmentation_tools = {
            "brush": BrushTool(self),
            "eraser": EraserTool(self),
            "fill": FillTool(self)
        }
        
    
    def _setup_ui(self) -> None:
        """Configura la interfaz orquestadora with tabs organizados."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Título of the panel
        title_label = QLabel("🛠️ Manual Editing Tools")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        layout.addWidget(self._create_separator())

        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
        """)

        measurement_tab = self._create_measurement_tab()
        self._tab_widget.addTab(measurement_tab, "📏 Measurements")

        segmentation_tab = self._create_segmentation_tab()
        self._tab_widget.addTab(segmentation_tab, "🎨 Segmentation")

        layout.addWidget(self._tab_widget)
        layout.addWidget(self._create_separator())

        actions_layout = self._create_global_actions()
        layout.addLayout(actions_layout)

        self._deactivate_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._deactivate_shortcut.activated.connect(self.deactivate_all_tools)
        self._deactivate_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    
    def _create_separator(self) -> QFrame:
        """Crea una línea separadora."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator
    
    def _create_slice_info_widget(self) -> QLabel:
        """Crea widget simple with información of the slice actual."""
        self._measurements_slice_info_label = QLabel("Axial 0/1 (0.0mm)")
        self._measurements_slice_info_label.setFont(QFont("monospace", 10, QFont.Weight.Bold))
        self._measurements_slice_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._measurements_slice_info_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background-color: #ECF0F1;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #BDC3C7;
                margin: 5px;
            }
        """)

        return self._measurements_slice_info_label
    
    def _create_unified_measurement_interface(self) -> QWidget:
        """Crea la interfaz unificada for selección of herramientas of medición."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Selector of herramienta with botones of radio
        tools_layout = QHBoxLayout()
        self._tool_button_group = QButtonGroup()
        
        self._distance_radio = QPushButton("📏 Distance")
        self._distance_radio.setCheckable(True)
        self._distance_radio.clicked.connect(lambda: self._on_tool_selected("distance"))
        self._tool_button_group.addButton(self._distance_radio, 0)
        tools_layout.addWidget(self._distance_radio)
        
        self._angle_radio = QPushButton("📐 Angle") 
        self._angle_radio.setCheckable(True)
        self._angle_radio.clicked.connect(lambda: self._on_tool_selected("angle"))
        self._tool_button_group.addButton(self._angle_radio, 1)
        tools_layout.addWidget(self._angle_radio)
        
        self._roi_radio = QPushButton("⬛ ROI")
        self._roi_radio.setCheckable(True) 
        self._roi_radio.clicked.connect(lambda: self._on_tool_selected("roi"))
        self._tool_button_group.addButton(self._roi_radio, 2)
        tools_layout.addWidget(self._roi_radio)
        
        layout.addLayout(tools_layout)
        
        # Label dinámico of instrucciones
        self._dynamic_status_label = QLabel("Select a measurement tool above")
        self._dynamic_status_label.setStyleSheet("color: #666; font-style: italic; padding: 8px; text-align: center;")
        self._dynamic_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._dynamic_status_label)
        
        # Tabla of mediciones
        self._measurements_table = QTableWidget()
        self._measurements_table.setColumnCount(4)
        self._measurements_table.setHorizontalHeaderLabels(["Type", "Value", "Unit", "Slice"])
        
        # Configure tabla
        header = self._measurements_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Type
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)          # Value
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Unit
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Slice
        
        self._measurements_table.setAlternatingRowColors(True)
        self._measurements_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._measurements_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        # Habilitar foco for eventos of teclado
        self._measurements_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        layout.addWidget(self._measurements_table)
        
        return widget
    
    def _on_tool_selected(self, tool_name: str) -> None:
        """Handles la selección of herramienta unificada."""
        for name, tool in self._measurement_tools.items():
            if name != tool_name and tool.is_active:
                tool.deactivate()

        selected_tool = self._measurement_tools.get(tool_name)
        if selected_tool:
            if not selected_tool.is_active:
                selected_tool.activate()

                instructions = {
                    "distance": "Click two points to measure distance",
                    "angle": "Click three points to measure angle",
                    "roi": "Draw a region of interest on the image"
                }
                self._dynamic_status_label.setText(instructions.get(tool_name, "Tool active"))
                self._dynamic_status_label.setStyleSheet("color: #27AE60; font-weight: bold; padding: 8px; text-align: center;")
            else:
                selected_tool.deactivate()
                self._dynamic_status_label.setText("Select a measurement tool above")
                self._dynamic_status_label.setStyleSheet("color: #666; font-style: italic; padding: 8px; text-align: center;")

                button = {
                    "distance": self._distance_radio,
                    "angle": self._angle_radio,
                    "roi": self._roi_radio
                }.get(tool_name)
                if button:
                    button.setChecked(False)
    
    def _create_measurement_tab(self) -> QWidget:
        """Crea el tab of herramientas of medición."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Información of the tab
        info_label = QLabel("Tools for measuring on the base image")
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(info_label)

        self._slice_info_widget = self._create_slice_info_widget()
        layout.addWidget(self._slice_info_widget)

        self._unified_tool_widget = self._create_unified_measurement_interface()
        layout.addWidget(self._unified_tool_widget)
        
        return widget
    
    def _create_segmentation_tab(self) -> QWidget:
        """Crea el tab of herramientas of segmentación unificado."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        active_seg_group = self._create_active_segmentation_selector()
        layout.addWidget(active_seg_group)
        layout.addWidget(self._create_separator())

        tools_group = self._create_grouped_editing_tools()
        layout.addWidget(tools_group)
        layout.addWidget(self._create_separator())

        self._tool_settings_widget = self._create_contextual_tool_settings()
        layout.addWidget(self._tool_settings_widget)
        layout.addWidget(self._create_separator())

        actions_group = self._create_segmentation_actions()
        layout.addWidget(actions_group)
        layout.addWidget(self._create_separator())

        info_group = self._create_live_segmentation_info()
        layout.addWidget(info_group)
        
        layout.addStretch()
        return widget
    
    def _create_global_actions(self) -> QHBoxLayout:
        """Crea botones of acción global."""
        layout = QHBoxLayout()
        
        # Botón for desactivar herramientas inteligentemente
        deactivate_all_button = QPushButton("⏹️ Deactivate All")
        deactivate_all_button.setToolTip(
            "Smart deactivation:\n"
            "• In Measurements tab: deactivates measurement tools\n"
            "• In Segmentation tab: deactivates segmentation tools + resets view\n"
            "• If both types active: deactivates all tools\n"
            "• Hotkey: Escape"
        )
        deactivate_all_button.clicked.connect(self.deactivate_all_tools)
        layout.addWidget(deactivate_all_button)
        
        layout.addStretch()
        
        # Botón for borrar datos permanentemente
        clear_all_button = QPushButton("🗑️ Clear All")
        clear_all_button.setToolTip(
            "⚠️ PERMANENTLY delete all data:\n"
            "• All measurements (distances, angles, ROIs)\n"
            "• All segmentation edits\n"
            "• Requires confirmation - action cannot be undone"
        )
        clear_all_button.clicked.connect(self._clear_all)
        layout.addWidget(clear_all_button)
        
        return layout
    
    def _create_active_segmentation_selector(self) -> QWidget:
        """Crea el selector of segmentación activa."""
        group = QGroupBox("📋 Active Segmentation")
        layout = QHBoxLayout(group)

        self._segmentation_selector = QComboBox()
        self._segmentation_selector.addItem("No segmentation loaded")
        self._segmentation_selector.setMinimumWidth(180)
        self._segmentation_selector.currentTextChanged.connect(self._on_segmentation_selection_changed)
        layout.addWidget(self._segmentation_selector)

        self._visibility_button = QPushButton("👁️")
        self._visibility_button.setCheckable(True)
        self._visibility_button.setChecked(False)
        self._visibility_button.setFixedSize(55, 25)
        self._visibility_button.setToolTip("Toggle segmentation visibility")
        self._visibility_button.clicked.connect(self._toggle_visibility)
        layout.addWidget(self._visibility_button)
        
        self._color_button = QPushButton("🎨")
        self._color_button.setFixedSize(55, 25)
        self._color_button.setToolTip("Change segmentation color")
        self._color_button.clicked.connect(self._change_color)
        layout.addWidget(self._color_button)
        
        return group
    
    def _create_grouped_editing_tools(self) -> QWidget:
        """Crea las herramientas of edición agrupadas."""
        group = QGroupBox("🛠️ Editing Tools")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        brush_tool = self._segmentation_tools["brush"]
        eraser_tool = self._segmentation_tools["eraser"]
        fill_tool = self._segmentation_tools["fill"]

        self._brush_button = QPushButton("🖌️ Brush")
        self._brush_button.setCheckable(True)
        self._brush_button.clicked.connect(lambda: self._toggle_tool_activation("brush"))
        layout.addWidget(self._brush_button, 0, 0)
        
        self._eraser_button = QPushButton("🧽 Eraser")
        self._eraser_button.setCheckable(True)
        self._eraser_button.clicked.connect(lambda: self._toggle_tool_activation("eraser"))
        layout.addWidget(self._eraser_button, 0, 1)
        
        self._fill_button = QPushButton("🪣 Fill")
        self._fill_button.setCheckable(True)
        self._fill_button.clicked.connect(lambda: self._toggle_tool_activation("fill"))
        layout.addWidget(self._fill_button, 1, 0, 1, 2)
        
        return group
    
    def _create_contextual_tool_settings(self) -> QWidget:
        """Crea configuración contextual basada in herramienta activa."""
        group = QGroupBox("⚙️ Tool Settings")
        layout = QVBoxLayout(group)
        
        # Configuración of tamaño (común for brush y eraser)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 50)
        self._size_slider.setValue(5)
        self._size_slider.valueChanged.connect(self._on_tool_size_changed)
        size_layout.addWidget(self._size_slider)
        
        self._size_label = QLabel("5px")
        self._size_label.setMinimumWidth(35)
        size_layout.addWidget(self._size_label)
        layout.addLayout(size_layout)
        
        # Configuración específica (hardness, tolerance, etc.)
        self._specific_settings_widget = QWidget()
        self._specific_settings_layout = QVBoxLayout(self._specific_settings_widget)
        layout.addWidget(self._specific_settings_widget)
        
        # Inicialmente vacío, se actualiza según herramienta activa
        self._update_contextual_settings("brush")
        
        return group
    
    def _create_segmentation_actions(self) -> QWidget:
        """Crea las acciones of control of segmentación."""
        group = QGroupBox("🔄 Actions")
        group.setVisible(True)  # Force visible
        layout = QHBoxLayout(group)
        
        # Historial
        self._undo_button = QPushButton("↶")
        self._undo_button.setToolTip("Undo last edit")
        self._undo_button.setFixedSize(35, 25)
        self._undo_button.clicked.connect(self._undo_action)
        self._undo_button.setVisible(True)  # Force visible
        layout.addWidget(self._undo_button)
        
        self._redo_button = QPushButton("↷")
        self._redo_button.setToolTip("Redo last edit")
        self._redo_button.setFixedSize(35, 25)
        self._redo_button.clicked.connect(self._redo_action)
        self._redo_button.setVisible(True)  # Force visible
        layout.addWidget(self._redo_button)
        
        layout.addWidget(self._create_mini_separator())
        
        # Aplicar/Cancelar cambios
        self._apply_button = QPushButton("✓ Apply")
        self._apply_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self._apply_button.clicked.connect(self._apply_changes)
        self._apply_button.setVisible(True)  # Force visible
        layout.addWidget(self._apply_button)
        
        self._cancel_button = QPushButton("✗ Cancel")
        self._cancel_button.setStyleSheet("background-color: #f44336; color: white;")
        self._cancel_button.clicked.connect(self._cancel_changes)
        self._cancel_button.setVisible(True)  # Force visible
        layout.addWidget(self._cancel_button)
        
        layout.addWidget(self._create_mini_separator())
        
        # Guardar
        self._save_segmentation_button = QPushButton("💾 Save")
        self._save_segmentation_button.setToolTip("Save current segmentation to file")
        self._save_segmentation_button.clicked.connect(self._save_segmentation_to_file)
        layout.addWidget(self._save_segmentation_button)
        
        return group
    
    def _create_live_segmentation_info(self) -> QWidget:
        """Crea información in tiempo real of la segmentación."""
        group = QGroupBox("📊 Live Info")
        layout = QVBoxLayout(group)
        
        # Info principal
        self._slice_info_label = QLabel("Slice: 1/1 (Axial)")
        self._slice_info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._slice_info_label)
        
        # Métricas in tiempo real
        metrics_layout = QHBoxLayout()
        self._volume_label = QLabel("Vol: 0.0 cm³")
        self._voxels_label = QLabel("Voxels: 0")
        self._changes_label = QLabel("Modified")
        
        metrics_layout.addWidget(self._volume_label)
        metrics_layout.addWidget(self._voxels_label)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self._changes_label)
        
        layout.addLayout(metrics_layout)
        
        return group
    
    def _create_mini_separator(self) -> QFrame:
        """Crea un separador mini vertical."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedWidth(1)
        return separator
    
    
    def _on_tool_size_changed(self, value: int) -> None:
        """Handles cambio of tamaño of herramienta of forma centralizada."""
        self._size_label.setText(f"{value}px")

        if self._active_segmentation_tool and self._active_segmentation_tool in ["brush", "eraser"]:
            tool = self._segmentation_tools[self._active_segmentation_tool]
            if hasattr(tool, 'set_brush_size'):
                tool.set_brush_size(value)

        self.brush_size_changed.emit(value)

    
    def _update_contextual_settings(self, tool_name: str) -> None:
        """Actualiza configuración específica según herramienta activa."""
        for i in reversed(range(self._specific_settings_layout.count())):
            child = self._specific_settings_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        if tool_name == "brush":
            hardness_layout = QHBoxLayout()
            hardness_layout.addWidget(QLabel("Hardness:"))
            hardness_slider = QSlider(Qt.Orientation.Horizontal)
            hardness_slider.setRange(0, 100)
            hardness_slider.setValue(100)
            hardness_layout.addWidget(hardness_slider)
            self._specific_settings_layout.addLayout(hardness_layout)
            
        elif tool_name == "fill":
            tolerance_layout = QHBoxLayout()
            tolerance_layout.addWidget(QLabel("Tolerance:"))
            tolerance_slider = QSlider(Qt.Orientation.Horizontal)
            tolerance_slider.setRange(0, 100)
            tolerance_slider.setValue(10)
            tolerance_layout.addWidget(tolerance_slider)
            self._specific_settings_layout.addLayout(tolerance_layout)
    
    def _update_tool_buttons_state(self, active_tool: Optional[str]) -> None:
        """Actualiza el estado visual of the botones of herramientas."""
        if hasattr(self, '_brush_button'):
            self._brush_button.setChecked(False)
        if hasattr(self, '_eraser_button'):
            self._eraser_button.setChecked(False)
        if hasattr(self, '_fill_button'):
            self._fill_button.setChecked(False)

        if active_tool == "brush" and hasattr(self, '_brush_button'):
            self._brush_button.setChecked(True)
        elif active_tool == "eraser" and hasattr(self, '_eraser_button'):
            self._eraser_button.setChecked(True)
        elif active_tool == "fill" and hasattr(self, '_fill_button'):
            self._fill_button.setChecked(True)
    
    def _setup_connections(self) -> None:
        """Configura las conexiones entre herramientas y señales of the orquestador."""
        for tool_name, tool in self._measurement_tools.items():
            tool.tool_activated.connect(lambda name=tool_name: self._on_measurement_tool_activated(name))
            tool.tool_deactivated.connect(lambda name=tool_name: self._on_measurement_tool_deactivated(name))
            tool.measurement_created.connect(self._on_measurement_created)

        for tool_name, tool in self._segmentation_tools.items():
            tool.tool_activated.connect(lambda name=tool_name: self._on_segmentation_tool_activated(name))
            tool.tool_deactivated.connect(lambda name=tool_name: self._on_segmentation_tool_deactivated(name))

            if hasattr(tool, 'brush_size_changed'):
                tool.brush_size_changed.connect(self.brush_size_changed.emit)

        self._measurements_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._measurements_table.keyPressEvent = self._on_table_key_press
    
    def _on_measurement_tool_activated(self, tool_name: str) -> None:
        """Handles la activación of una herramienta of medición."""
        for name, tool in self._measurement_tools.items():
            if name != tool_name and tool.is_active:
                tool.deactivate()

        for tool in self._segmentation_tools.values():
            if tool.is_active:
                tool.deactivate()

        self._active_measurement_tool = tool_name
        self._active_segmentation_tool = None

        self.measurement_mode_changed.emit(tool_name)
    
    def _on_measurement_tool_deactivated(self, tool_name: str) -> None:
        """Handles la desactivación of una herramienta of medición."""
        if self._active_measurement_tool == tool_name:
            self._active_measurement_tool = None
            self.measurement_mode_changed.emit("")
    
    def _on_measurement_created(self, measurement: Dict[str, Any]) -> None:
        """Handles la creación of nuevas mediciones y actualiza la tabla."""
        measurement['slice_info'] = self._current_slice_info.copy()
    
    def _on_segmentation_tool_activated(self, tool_name: str) -> None:
        """Handles la activación of una herramienta of segmentación."""
        for name, tool in self._segmentation_tools.items():
            if name != tool_name and tool.is_active:
                tool.deactivate()

        for tool in self._measurement_tools.values():
            if tool.is_active:
                tool.deactivate()

        self._active_segmentation_tool = tool_name
        self._active_measurement_tool = None

        self._update_tool_buttons_state(tool_name)
        self._update_contextual_settings(tool_name)

        self.segmentation_tool_changed.emit(tool_name)
    
    def _on_segmentation_tool_deactivated(self, tool_name: str) -> None:
        """Handles la desactivación of una herramienta of segmentación."""
        if self._active_segmentation_tool == tool_name:
            self._active_segmentation_tool = None
            
            # Update UI of botones
            self._update_tool_buttons_state(None)
            
            self.segmentation_tool_changed.emit("")
    
    def deactivate_all_tools(self) -> None:
        """
        Desactiva herramientas of forma inteligente según el contexto actual.
        """
        current_tab_index = self._tab_widget.currentIndex()
        current_tab_name = self._tab_widget.tabText(current_tab_index)

        active_measurement_tools = [name for name, tool in self._measurement_tools.items() if tool.is_active]
        active_segmentation_tools = [name for name, tool in self._segmentation_tools.items() if tool.is_active]

        deactivated_types = []

        if current_tab_name == "📏 Measurements":
            if active_measurement_tools:
                for tool in self._measurement_tools.values():
                    if tool.is_active:
                        tool.deactivate()
                self._active_measurement_tool = None
                deactivated_types.append("measurement")

        elif current_tab_name == "🎨 Segmentation":
            if active_segmentation_tools:
                for tool in self._segmentation_tools.values():
                    if tool.is_active:
                        tool.deactivate()
                self._active_segmentation_tool = None
                deactivated_types.append("segmentation")

                if hasattr(self, '_segmentation_selector'):
                    self._segmentation_selector.blockSignals(True)
                    self._segmentation_selector.setCurrentText("None")
                    self._segmentation_selector.blockSignals(False)
                    self._on_segmentation_selection_changed("None")

        if active_measurement_tools and active_segmentation_tools:
            for tool in self._measurement_tools.values():
                if tool.is_active:
                    tool.deactivate()
            for tool in self._segmentation_tools.values():
                if tool.is_active:
                    tool.deactivate()

            self._active_measurement_tool = None
            self._active_segmentation_tool = None

            if hasattr(self, '_segmentation_selector'):
                self._segmentation_selector.blockSignals(True)
                self._segmentation_selector.setCurrentText("None")
                self._segmentation_selector.blockSignals(False)
                self._on_segmentation_selection_changed("None")

            deactivated_types = ["measurement", "segmentation"]

        self._update_tool_buttons_state(None)

        if deactivated_types:
            types_str = " & ".join(deactivated_types)
            self.logger.debug(f"Deactivated tools: {types_str}")

    def _save_segmentation_to_file(self) -> None:
        """Guarda la segmentación actual a un archivo."""
        try:
            active_overlay_id = self._get_active_overlay_id()
            if not active_overlay_id:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Active Segmentation",
                                  "Please select an active segmentation to save.")
                return

            overlay_service = self._get_overlay_service()
            if not overlay_service:
                raise Exception("Cannot access overlay service")

            mask_data = overlay_service.get_overlay_mask_data(active_overlay_id)
            if mask_data is None:
                raise Exception(f"No mask data found for {active_overlay_id}")

            from PyQt6.QtWidgets import QFileDialog
            current_selection = self.get_current_segmentation_selection()

            suggested_name = f"{current_selection}_edited.nii.gz"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Segmentation",
                suggested_name,
                "NIfTI files (*.nii *.nii.gz);;All files (*.*)"
            )

            if not file_path:
                return

            success = overlay_service.save_segmentation_to_file(active_overlay_id, file_path)

            if success:
                self.logger.info(f"Successfully saved segmentation to: {file_path}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success",
                                      f"Segmentation saved successfully to:\n{file_path}")
            else:
                raise Exception("Failed to save segmentation")

        except Exception as e:
            self.logger.error(f"Error saving segmentation: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", f"Failed to save segmentation:\n{str(e)}")

    def _clear_all(self) -> None:
        """Borra permanentemente todas las mediciones y segmentaciones with confirmación."""
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "⚠️ This will permanently delete ALL measurements and segmentation edits.\n\n"
            "This action cannot be undone.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.deactivate_all_tools()
            self.clear_measurements_requested.emit()
            self.clear_segmentations_requested.emit()

            QMessageBox.information(
                self,
                "Data Cleared",
                "✅ All measurements and segmentation edits have been cleared."
            )

    def get_active_measurement_tool(self) -> Optional[str]:
        """Obtiene la herramienta of medición activa."""
        return self._active_measurement_tool
    
    def get_active_segmentation_tool(self) -> Optional[str]:
        """Obtiene la herramienta of segmentación activa."""
        return self._active_segmentation_tool
    
    def get_measurement_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Obtiene una herramienta of medición específica."""
        return self._measurement_tools.get(tool_name)
    
    def get_segmentation_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Obtiene una herramienta of segmentación específica."""
        return self._segmentation_tools.get(tool_name)
    
    def update_available_segmentations(self, segmentations: List[Dict[str, Any]]) -> None:
        """
        Actualiza las segmentaciones disponibles in the selector.
        
        Args:
            segmentations: Lista of diccionarios with info of segmentaciones
                         Cada dict debe tener: {'name', 'path', 'type', 'confidence'}
        """
        if not hasattr(self, '_segmentation_selector'):
            return
            
        # Limpiar selector actual
        self._segmentation_selector.clear()
        
        if not segmentations:
            self._segmentation_selector.addItem("No segmentation loaded")
            return
        
        # Add segmentaciones disponibles
        for seg in segmentations:
            name = seg.get('name', 'Unknown')
            seg_type = seg.get('type', 'Unknown')
            confidence = seg.get('confidence', 0)
            
            # Show only the name, remove type and confidence info
            display_name = name
                
            # Guardar datos completos in the item
            self._segmentation_selector.addItem(display_name)
            self._segmentation_selector.setItemData(
                self._segmentation_selector.count() - 1, 
                seg
            )
        
        # Seleccionar la primera by defecto
        if segmentations:
            self._segmentation_selector.setCurrentIndex(0)
        
    
    
    def get_current_segmentation(self) -> Optional[Dict[str, Any]]:
        """Obtiene la segmentación actualmente seleccionada."""
        if not hasattr(self, '_segmentation_selector'):
            return None
            
        current_index = self._segmentation_selector.currentIndex()
        if current_index < 0:
            return None
            
        return self._segmentation_selector.itemData(current_index)
    
    def update_slice_info(self, plane: str, index: int, total_slices: int, position_mm: float = 0.0) -> None:
        """Actualiza la información of the slice actual."""
        self._current_slice_info.update({
            "plane": plane,
            "index": index, 
            "total_slices": total_slices,
            "position_mm": position_mm
        })
        
        # Update display
        slice_text = f"{plane.title()} {index + 1}/{total_slices}"
        if position_mm != 0.0:
            slice_text += f" ({position_mm:.1f}mm)"
        
        # Update ambos labels si existen
        if hasattr(self, '_measurements_slice_info_label'):
            self._measurements_slice_info_label.setText(slice_text)
        if hasattr(self, '_slice_info_label'):
            self._slice_info_label.setText(slice_text)
        # Note: No llamamos _update_slice_measurements_display() aquí
        # porque será llamado by update_measurements_for_slice() with los datos correctos
    
    def update_measurements_for_slice(self, measurements_data: Dict[str, List[Dict]]) -> None:
        """Actualiza las herramientas of medición with datos of the slice actual."""
        # Get la clave of the slice actual
        current_slice_key = f"{self._current_slice_info['plane']}_{self._current_slice_info['index']:03d}"
        current_measurements = measurements_data.get(current_slice_key, [])
        
        # Solo actualizar la tabla - las herramientas ya no tienin displays individuales
        self._update_slice_measurements_display(current_measurements)
    
    def _update_slice_measurements_display(self, measurements: Optional[List[Dict]] = None) -> None:
        """Actualiza la tabla of mediciones of the slice actual."""
        # Limpiar tabla
        self._measurements_table.setRowCount(0)
        
        if measurements is None or not measurements:
            return
        
        # Configure tabla with the número correcto of filas
        self._measurements_table.setRowCount(len(measurements))
        
        # Colores médicos profesionales - consistentes with visualizador
        type_colors = {
            'distance': '#2E7D32',  # Verof médico
            'angle': '#0D47A1',     # Azul médico profundo  
            'roi': '#9C27B0'        # Púrpura médico
        }
        
        # Llenar tabla
        for row, measurement in enumerate(measurements):
            m_type = measurement.get('type', 'unknown')
            color = type_colors.get(m_type, '#333333')
            
            # Columna Type
            type_item = QTableWidgetItem(m_type.capitalize())
            type_item.setForeground(QBrush(QColor(color)))
            font = type_item.font()
            font.setBold(True)
            type_item.setFont(font)
            self._measurements_table.setItem(row, 0, type_item)
            
            # Columna Value
            if m_type == 'distance':
                distance_px = measurement.get('distance', 0)
                distance_mm = measurement.get('distance_mm', 0)
                value_text = f"{distance_px:.1f}px ({distance_mm:.1f}mm)"
            elif m_type == 'angle':
                angle = measurement.get('angle', 0)
                value_text = f"{angle:.1f}°"
            elif m_type == 'roi':
                area = measurement.get('area', 0)
                value_text = f"Area: {area:.1f}px²"
            else:
                value_text = "Unknown"
            
            value_item = QTableWidgetItem(value_text)
            value_item.setForeground(QBrush(QColor(color)))
            self._measurements_table.setItem(row, 1, value_item)
            
            # Columna Unit
            unit = measurement.get('unit', '')
            if m_type == 'distance':
                unit = 'mm/px'
            elif m_type == 'angle':
                unit = 'degrees'
            elif m_type == 'roi':
                unit = 'pixels²'
            
            unit_item = QTableWidgetItem(unit)
            self._measurements_table.setItem(row, 2, unit_item)
            
            # Columna Slice
            slice_info = measurement.get('slice_info', {})
            slice_text = slice_info.get('slice_key', 'Current')
            slice_item = QTableWidgetItem(slice_text)
            self._measurements_table.setItem(row, 3, slice_item)
        
        # Guardar datos actuales for referencias posteriores
        self._current_measurements_data = measurements if measurements else []
    
    def _on_table_selection_changed(self) -> None:
        """Handles el cambio of selección in la tabla of mediciones."""
        try:
            # Limpiar resaltado previo
            self._clear_table_highlighting()
            
            selected_items = self._measurements_table.selectedItems()
            if not selected_items:
                # No hay selección
                self._selected_measurement_row = -1
                self._selected_measurement_data = None
                return
            
            # Get la fila seleccionada
            selected_row = selected_items[0].row()
            
            # Verify que la fila es válida
            if selected_row < 0 or selected_row >= len(self._current_measurements_data):
                return
            
            # Update estado of selección
            self._selected_measurement_row = selected_row
            self._selected_measurement_data = self._current_measurements_data[selected_row].copy()
            
            # Add información of resaltado visual
            for col in range(self._measurements_table.columnCount()):
                item = self._measurements_table.item(selected_row, col)
                if item:
                    # Resaltar fila seleccionada with fondo suave
                    item.setBackground(QBrush(QColor("#E3F2FD")))  # Azul muy claro
            
            # Emitir señal for resaltar in canvas
            if self._selected_measurement_data:
                self.measurement_selected.emit(self._selected_measurement_data)
        
        except Exception as e:
            self.logger.error(f"Error in table selection handler: {e}")
    
    def _clear_table_highlighting(self) -> None:
        """Limpia el resaltado visual of todas las filas of la tabla."""
        try:
            for row in range(self._measurements_table.rowCount()):
                for col in range(self._measurements_table.columnCount()):
                    item = self._measurements_table.item(row, col)
                    if item:
                        # Restaurar fondo by defecto
                        item.setBackground(QBrush(QColor("#FFFFFF")))
        except Exception as e:
            self.logger.error(f"Error clearing table highlighting: {e}")
    
    def _on_table_key_press(self, event) -> None:
        """Handles eventos of teclado in la tabla of mediciones."""
        try:
            # Verify si es la tecla Delete o Suprimir
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._selected_measurement_data is not None:
                    # Emitir señal for borrar medición
                    measurement_type = self._selected_measurement_data.get('type', 'unknown') if isinstance(self._selected_measurement_data, dict) else 'unknown'
                    self.measurement_delete_requested.emit(self._selected_measurement_data)

                    # Limpiar selección local
                    self._selected_measurement_row = -1
                    self._selected_measurement_data = None
                # No measurement selected for deletion
            else:
                # Para otras teclas, llamar al handler original
                super(QTableWidget, self._measurements_table).keyPressEvent(event)
        
        except Exception as e:
            self.logger.error(f"Error in table key press handler: {e}")
            # Fallback al handler original
            super(QTableWidget, self._measurements_table).keyPressEvent(event)
    
    def sync_with_image_viewer_segmentation(self, selection: str) -> None:
        """
        Synchronize segmentation tab selector with image viewer dropdown.
        
        Args:
            selection: Selected segmentation name from image viewer
        """
        if hasattr(self, '_segmentation_selector') and self._segmentation_selector:
            # Block signals to avoid circular updates
            self._segmentation_selector.blockSignals(True)
            
            # Find and set the matching item
            index = self._segmentation_selector.findText(selection, Qt.MatchFlag.MatchContains)
            if index >= 0:
                self._segmentation_selector.setCurrentIndex(index)
            else:
                # Default to first item (usually "No segmentation loaded" or "None")
                self._segmentation_selector.setCurrentIndex(0)
            
            # Re-enable signals
            self._segmentation_selector.blockSignals(False)
            
    
    def get_current_segmentation_selection(self) -> str:
        """
        Get current segmentation selection from tab.
        
        Returns:
            Currently selected segmentation name
        """
        if hasattr(self, '_segmentation_selector') and self._segmentation_selector:
            return self._segmentation_selector.currentText()
        return "None"
    
    def connect_segmentation_selector_change(self, callback) -> None:
        """
        Connect a callback to segmentation selector changes.

        Args:
            callback: Function to call when selection changes
        """
        if hasattr(self, '_segmentation_selector') and self._segmentation_selector:
            self._segmentation_selector.currentTextChanged.connect(callback)
    
    # === FUNCIONALIDADES DEL TAB ===
    
    def _toggle_visibility(self) -> None:
        """Toggle visibility of the currently selected segmentation."""
        current_selection = self.get_current_segmentation_selection()
        if not current_selection or current_selection == "None":
            return
            
        is_visible = self._visibility_button.isChecked()
        
        # Buscar la main window a través of la jerarquía
        main_window = self.window()
        if main_window and hasattr(main_window, '_ui_components'):
            # Acceder al image viewer directamente
            image_viewer = main_window._ui_components.central_viewer
            if image_viewer and hasattr(image_viewer, '_overlay_service'):
                # Construir el overlay ID basado in the nombre of la selección
                base_name = current_selection.split(" (")[0].strip()
                
                # Lógica similar a la of the método of selección original
                overlay_index = 0
                if "_Region_" in base_name:
                    region_num = int(base_name.split("_Region_")[-1])
                    overlay_index = region_num - 1
                    base_name = base_name.split("_Region_")[0]
                elif base_name.count("_") >= 2:
                    parts = base_name.split("_")
                    if len(parts) >= 3:
                        possible_region = parts[-1]
                        if possible_region in ['PZ', 'TZ', 'WG']:
                            if possible_region == 'TZ':
                                overlay_index = 0
                            elif possible_region == 'PZ':
                                overlay_index = 1
                            base_name = "_".join(parts[:-1])
                
                target_overlay_id = f"auto_mask_{base_name}_{overlay_index}"
                
                # Cambiar solo la visibilidad of este overlay específico
                image_viewer._overlay_service.set_overlay_visibility(target_overlay_id, is_visible)
                
                # Forzar actualización of the canvas
                image_viewer._update_canvas_for_plane(image_viewer._active_view_plane)
        
    
    def _change_color(self) -> None:
        """Opin color picker to change segmentation color."""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        
        current_selection = self.get_current_segmentation_selection()
        if not current_selection or current_selection == "None":
            self.logger.warning("No segmentation selected for color change")
            return
            
        # Opin color picker
        color = QColorDialog.getColor(QColor(255, 0, 0), self, "Choose Segmentation Color")
        if color.isValid():
            # Implementar cambio of color real
            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                image_viewer = main_window._ui_components.central_viewer
                if image_viewer and hasattr(image_viewer, '_overlay_service'):
                    # Construir el overlay ID basado in the nombre of la selección
                    base_name = current_selection.split(" (")[0].strip()
                    
                    # Lógica similar a la of the método of visibilidad
                    overlay_index = 0
                    if "_Region_" in base_name:
                        region_num = int(base_name.split("_Region_")[-1])
                        overlay_index = region_num - 1
                        base_name = base_name.split("_Region_")[0]
                    elif base_name.count("_") >= 2:
                        parts = base_name.split("_")
                        if len(parts) >= 3:
                            possible_region = parts[-1]
                            if possible_region in ['PZ', 'TZ', 'WG']:
                                if possible_region == 'TZ':
                                    overlay_index = 0
                                elif possible_region == 'PZ':
                                    overlay_index = 1
                                base_name = "_".join(parts[:-1])
                    
                    target_overlay_id = f"auto_mask_{base_name}_{overlay_index}"
                    
                    # Cambiar color of the overlay específico - pasar QColor directamente
                    image_viewer._overlay_service.set_overlay_color(target_overlay_id, color)
                    
                    # Forzar actualización of the canvas
                    image_viewer._update_canvas_for_plane(image_viewer._active_view_plane)
                    
            else:
                self.logger.error("Could not access overlay service for color change")
    
    def _toggle_tool_activation(self, tool_name: str) -> None:
        """Toggle activation of editing tools."""
        # Deactivate all other tools first
        for name, button in [("brush", getattr(self, '_brush_button', None)),
                            ("eraser", getattr(self, '_eraser_button', None)),
                            ("fill", getattr(self, '_fill_button', None))]:
            if button and name != tool_name:
                button.setChecked(False)
        
        # Get the tool and activate/deactivate
        tool = self._segmentation_tools.get(tool_name)
        if tool:
            button = getattr(self, f'_{tool_name}_button', None)
            if button and button.isChecked():
                self._active_segmentation_tool = tool_name
                tool.activate()
                
                # Save current state to history before first tool action
                self._save_current_state_to_history()
                
                # Apply current size if applicable
                current_size = 5
                if tool_name in ["brush", "eraser"] and hasattr(self, '_size_slider'):
                    current_size = self._size_slider.value()
                    if hasattr(tool, 'set_brush_size'):
                        tool.set_brush_size(current_size)
                
                # Emit signal to connect with canvas
                self.segmentation_tool_changed.emit(tool_name)
                if hasattr(self, '_size_slider'):
                    self.brush_size_changed.emit(self._size_slider.value())
                
            else:
                self._active_segmentation_tool = None
                tool.deactivate()
                # Emit empty signal to deactivate
                self.segmentation_tool_changed.emit("")
    
    def _apply_changes(self) -> None:
        """Apply all current editing changes to the segmentation."""

        # Get the active overlay ID
        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Active Segmentation", "Please select an active segmentation to apply changes to.")
            return

        try:
            # Get overlay service
            overlay_service = self._get_overlay_service()
            if not overlay_service:
                raise Exception("Cannot access overlay service")

            # Apply segmentation changes
            success = overlay_service.save_segmentation_changes(active_overlay_id)

            if success:
                # Clear history since changes are now applied
                self._clear_history()

                # Update UI state
                self._update_action_buttons()

                # Force canvas update
                self._force_canvas_update()

                current_selection = self.get_current_segmentation_selection()
                self.logger.info(f"Successfully applied changes to {current_selection}")

                # Show success feedback
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Changes applied to {current_selection}")
            else:
                raise Exception("Failed to save segmentation changes")

        except Exception as e:
            self.logger.error(f"Error applying changes: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to apply changes: {str(e)}")
    
    def _cancel_changes(self) -> None:
        """Cancel all current editing changes and revert to original."""

        # Get the active overlay ID
        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Active Segmentation", "Please select an active segmentation to cancel changes.")
            return

        try:
            # Get overlay service
            overlay_service = self._get_overlay_service()
            if not overlay_service:
                raise Exception("Cannot access overlay service")

            # Revert to original segmentation data
            if self._original_segmentation_data:
                success = overlay_service.restore_segmentation_data(active_overlay_id, self._original_segmentation_data)

                if success:
                    # Clear history
                    self._clear_history()

                    # Update UI state
                    self._update_action_buttons()

                    # Force canvas update
                    self._force_canvas_update()

                    current_selection = self.get_current_segmentation_selection()
                    self.logger.info(f"Successfully cancelled changes for {current_selection}")

                    # Show cancellation feedback
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Changes Cancelled", f"Changes cancelled for {current_selection}")
                else:
                    raise Exception("Failed to restore original segmentation data")
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "No Changes", "No original data found to restore. There may be no changes to cancel.")

        except Exception as e:
            self.logger.error(f"Error cancelling changes: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to cancel changes: {str(e)}")
    
    def _undo_action(self) -> None:
        """Undo the last editing action."""
        self.logger.debug("UNDO: Undo button clicked")
        if not self._can_undo():
            self.logger.error("Cannot undo: no actions in history")
            return
            
        try:
            # Move back in history
            self._history_index -= 1
            
            # Get the state to restore
            state_to_restore = self._history_stack[self._history_index]

            # Apply the state
            success = self._apply_history_state(state_to_restore)

            if not success:
                # Revert history index on failure
                self._history_index += 1
                raise Exception("Failed to apply undo state")
                
        except Exception as e:
            self.logger.error(f"Error during undo: {e}")
            
        # Update button states
        self._update_action_buttons()
    
    def _redo_action(self) -> None:
        """Redo the last undone action."""
        self.logger.debug("REDO: Redo button clicked")
        if not self._can_redo():
            self.logger.error("Cannot redo: no actions to redo")
            return
            
        try:
            # Move forward in history
            self._history_index += 1
            
            # Get the state to restore
            state_to_restore = self._history_stack[self._history_index]

            # Apply the state
            success = self._apply_history_state(state_to_restore)

            if not success:
                # Revert history index on failure
                self._history_index -= 1
                raise Exception("Failed to apply redo state")
                
        except Exception as e:
            self.logger.error(f"Error during redo: {e}")
            
        # Update button states
        self._update_action_buttons()
    
    def _update_action_buttons(self) -> None:
        """Update the enabled state of action buttons based on history."""
        self._undo_button.setEnabled(self._can_undo())
        self._redo_button.setEnabled(self._can_redo())
        
        # Enable apply/cancel only if there are changes
        has_changes = len(self._history_stack) > 0
        self._apply_button.setEnabled(has_changes)
        self._cancel_button.setEnabled(has_changes)
    
    def _can_undo(self) -> bool:
        """Check if undo is possible."""
        return self._history_index >= 0
    
    def _can_redo(self) -> bool:
        """Check if redo is possible."""
        return self._history_index < len(self._history_stack) - 1
    
    def _add_to_history(self, state_data: Dict[str, Any]) -> None:
        """MEJORADO: Add a new state to the history stack with real mask data."""
        try:
            # Remove any states after current index (when new action after undo)
            if self._history_index < len(self._history_stack) - 1:
                self._history_stack = self._history_stack[:self._history_index + 1]

            # NUEVO: Asegurar que tenemos datos reales of máscara
            if 'mask_data' not in state_data:
                # Get datos reales si no están incluidos
                overlay_id = state_data.get('overlay_id') or self._get_active_overlay_id()
                if overlay_id:
                    overlay_service = self._get_overlay_service()
                    if overlay_service:
                        mask_data = overlay_service.get_overlay_mask_data(overlay_id)
                        if mask_data is not None:
                            state_data = state_data.copy()
                            state_data['mask_data'] = mask_data.copy()
                            state_data['overlay_id'] = overlay_id

            # Add new state with deep copy
            state_copy = {
                'overlay_id': state_data.get('overlay_id'),
                'mask_data': state_data.get('mask_data').copy() if state_data.get('mask_data') is not None else None,
                'timestamp': state_data.get('timestamp', self._get_current_timestamp()),
                'action': state_data.get('action', 'user_action')
            }

            self._history_stack.append(state_copy)
            self._history_index = len(self._history_stack) - 1

            # Limit history size
            max_history = 20
            if len(self._history_stack) > max_history:
                self._history_stack.pop(0)
                self._history_index -= 1


            # Update buttons
            self._update_action_buttons()

        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")
    
    def _apply_history_state(self, state_data: Dict[str, Any]) -> bool:
        """MEJORADO: Apply a history state to the current segmentation using real mask data."""
        try:
            overlay_service = self._get_overlay_service()
            if not overlay_service:
                self.logger.error("No overlay service available for applying history state")
                return False

            # Get overlay_id of the estado o usar el activo
            overlay_id = state_data.get('overlay_id') or self._get_active_overlay_id()
            if not overlay_id:
                self.logger.error("No overlay ID available for applying history state")
                return False

            # NUEVO: Aplicar datos reales of máscara
            if 'mask_data' in state_data and state_data['mask_data'] is not None:
                # Restaurar la máscara directamente
                try:
                    overlay_service._segmentation_overlays[overlay_id] = state_data['mask_data'].copy()

                    # Forzar actualización of visualización
                    self._force_canvas_update()

                    # Update métricas in tiempo real
                    self._update_live_metrics(overlay_id)

                    return True

                except Exception as restore_error:
                    self.logger.error(f"Error directly restoring mask data: {restore_error}")
                    # Fallback al método original
                    pass

            # Fallback: Usar método original of the overlay service
            current_selection = self.get_current_segmentation_selection()
            success = overlay_service.apply_segmentation_state(current_selection, state_data)

            if success:
                # Forzar actualización
                self._force_canvas_update()
                self._update_live_metrics(overlay_id)

            return success

        except Exception as e:
            self.logger.error(f"Error applying history state: {e}")
            return False
    
    def _save_current_state_to_history(self) -> None:
        """Save the current segmentation state to history before making changes."""
        try:
            # Get overlay service with fallback
            overlay_service = self._get_overlay_service()
            if not overlay_service:
                return
            
            # Get active overlay ID using same logic as editing tools
            active_overlay_id = self._get_active_overlay_id()
            if not active_overlay_id:
                return
            
            # Get current state using active overlay ID
            current_state = overlay_service.get_segmentation_state(active_overlay_id)
            
            if current_state:
                # Save as original data if first time
                if self._original_segmentation_data is None:
                    self._original_segmentation_data = current_state.copy()
                
                # Add to history
                self._add_to_history(current_state)
            
        except Exception as e:
            self.logger.error(f"Error saving state to history: {e}")
    
    def _get_active_overlay_id(self):
        """
        MEJORADO: Get the currently active overlay ID using synchronized state.

        Returns:
            Active overlay ID or None if not available
        """
        # NUEVA ESTRATEGIA: Usar estado sincronizado primero
        if self._active_overlay_id:
            return self._active_overlay_id

        # Fallback a la resolución manual si no hay estado sincronizado
        try:
            current_selection = self.get_current_segmentation_selection()
            if not current_selection or current_selection == "None" or current_selection == "No segmentation loaded":
                return None

            # Usar el método of resolución mejorado
            overlay_id = self._resolve_overlay_id_for_selection(current_selection)
            if overlay_id:
                # Update estado sincronizado for futuras llamadas
                self._active_overlay_id = overlay_id
                self._active_segmentation_name = current_selection

            return overlay_id

        except Exception as e:
            self.logger.error(f"Error getting active overlay ID: {e}")
            return None

    def _get_overlay_service(self):
        """
        Get overlay service with multiple fallback strategies.
        
        Returns:
            Overlay service instance or None if not found
        """
        try:
            # Strategy 1: Through main window
            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                # Try standard path
                if hasattr(main_window._ui_components, 'central_viewer'):
                    image_viewer = main_window._ui_components.central_viewer
                    if image_viewer and hasattr(image_viewer, '_overlay_service'):
                        return image_viewer._overlay_service
                
                # Try right panel path 
                if hasattr(main_window._ui_components, 'right_panel'):
                    right_panel = main_window._ui_components.right_panel
                    if hasattr(right_panel, '_panels'):
                        manual_panel = right_panel._panels.get('manual_editing')
                        if manual_panel and hasattr(manual_panel, '_overlay_service_ref'):
                            return manual_panel._overlay_service_ref
            
            # Strategy 2: Check if stored as instance variable
            if hasattr(self, '_overlay_service_ref'):
                return self._overlay_service_ref

            # Strategy 3: Global reference (if available)
            if hasattr(self, '_parent_image_viewer'):
                if hasattr(self._parent_image_viewer, '_overlay_service'):
                    return self._parent_image_viewer._overlay_service

            self.logger.error("Could not access overlay service - actions may not work correctly")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting overlay service: {e}")
            return None
    
    def set_overlay_service_reference(self, overlay_service):
        """
        Set a reference to the overlay service for fallback access.
        This should be called during setup to ensure actions work.
        
        Args:
            overlay_service: The overlay service instance
        """
        self._overlay_service_ref = overlay_service
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for history tracking."""
        import datetime
        return datetime.datetime.now().isoformat()

    def _on_segmentation_selection_changed(self, selection_text: str) -> None:
        """
        Handles cambio of selección of segmentación y sincroniza with viewer.

        Este es el método crítico que resuelve la comunicación Panel ↔ Viewer.
        """
        try:

            # Limpiar selección anterior
            self._active_segmentation_name = None
            self._active_overlay_id = None

            # Validate selección
            if not selection_text or selection_text == "No segmentation loaded" or selection_text == "None":
                # Emisión of señal with valores vacíos for desactivar
                self.logger.debug(f"No segmentation selected: '{selection_text}'")
                self.active_segmentation_changed.emit("", "")
                return

            # Get datos of la selección
            current_index = self._segmentation_selector.currentIndex()
            segmentation_data = None
            if current_index >= 0:
                segmentation_data = self._segmentation_selector.itemData(current_index)

            # Resolver overlay ID usando múltiples estrategias
            overlay_id = self._resolve_overlay_id_for_selection(selection_text, segmentation_data)

            if overlay_id:
                # Update estado interno
                self._active_segmentation_name = selection_text
                self._active_overlay_id = overlay_id

                # Informar al viewer
                self.active_segmentation_changed.emit(selection_text, overlay_id)

                # Update info in tiempo real si hay métricas disponibles
                self._update_live_metrics(overlay_id)
            else:
                self.logger.error(f"Could not resolve overlay ID for selection: {selection_text}")
                # Emitir señal vacía como fallback
                self.active_segmentation_changed.emit("", "")

        except Exception as e:
            self.logger.error(f"Error handling segmentation selection change: {e}")

    def _resolve_overlay_id_for_selection(self, selection_text: str, segmentation_data: dict = None) -> Optional[str]:
        """
        Resuelve el overlay ID for una selección específica usando múltiples estrategias.

        Args:
            selection_text: Texto of la selección actual
            segmentation_data: Datos adicionales of the item seleccionado

        Returns:
            Overlay ID correspondiente o None si no se encuentra
        """
        try:
            # Direct mapping for Region_ masks
            if "_Region_" in selection_text:
                # Extraer base name y region number
                parts = selection_text.split("_Region_")
                base_name = parts[0]
                region_num = parts[1]

                # Mapeo directo: Region_1 -> _0, Region_2 -> _1
                if region_num == "1":
                    overlay_suffix = "_0"
                elif region_num == "2":
                    overlay_suffix = "_1"
                else:
                    overlay_suffix = f"_{int(region_num) - 1}"

                target_overlay_id = f"auto_mask_{base_name}{overlay_suffix}"
                self.logger.debug(f"DIRECT SIMPLE MAPPING: '{selection_text}' -> '{target_overlay_id}'")
                return target_overlay_id

            overlay_service = self._get_overlay_service()
            if not overlay_service:
                return None

            available_overlays = overlay_service.get_all_overlay_ids()
            if not available_overlays:
                return None

            # Limpiar el texto of selección
            clean_selection = selection_text.split(" (")[0].strip()

            # Estrategia 1: Usar datos of the item si están disponibles
            if segmentation_data and isinstance(segmentation_data, dict):
                if 'overlay_id' in segmentation_data:
                    return segmentation_data['overlay_id']
                if 'path' in segmentation_data:
                    # Intentar construir overlay_id from path
                    for overlay_id in available_overlays:
                        if segmentation_data['path'] in overlay_id or overlay_id in segmentation_data['path']:
                            return overlay_id

            # Estrategia 2: Coincidencia exacta
            for overlay_id in available_overlays:
                if overlay_id == clean_selection:
                    return overlay_id

            # Estrategia 3: Coincidencia que contiene el texto (case insensitive)
            for overlay_id in available_overlays:
                if clean_selection.lower() in overlay_id.lower() or overlay_id.lower() in clean_selection.lower():
                    return overlay_id

            # Estrategia 4: Coincidencia parcial with patrones comunes
            for overlay_id in available_overlays:
                # Remover prefijos comunes for comparación
                clean_overlay = overlay_id.replace("auto_mask_", "").replace("segmentation_", "")
                if clean_selection.lower() in clean_overlay.lower():
                    return overlay_id

            # Estrategia 5: Fallback al primer overlay disponible with warning
            if available_overlays:
                self.logger.warning(f"No specific match for '{clean_selection}', using first available: {available_overlays[0]}")
                return available_overlays[0]

            return None

        except Exception as e:
            self.logger.error(f"Error resolving overlay ID: {e}")
            return None

    def _update_live_metrics(self, overlay_id: str) -> None:
        """
        Actualiza métricas in tiempo real for la segmentación activa.

        Args:
            overlay_id: ID of the overlay for calcular métricas
        """
        try:
            if not (hasattr(self, '_volume_label') and hasattr(self, '_voxels_label')):
                return

            overlay_service = self._get_overlay_service()
            if not overlay_service:
                return

            # Get full 3D mask data
            mask_data = overlay_service.get_overlay_mask_data(overlay_id)
            if mask_data is None:
                return

            # Get current slice info from panel's own state
            current_slice = self._current_slice_info.get('index', 0)
            current_plane = self._current_slice_info.get('plane', 'axial')

            # Extract 2D slice from 3D mask based on plane
            # Mask shape is (depth, height, width) = (z, y, x)
            if len(mask_data.shape) == 3:
                if current_plane == 'axial':
                    slice_mask = mask_data[current_slice, :, :] if current_slice < mask_data.shape[0] else None
                elif current_plane == 'sagittal':
                    slice_mask = mask_data[:, :, current_slice] if current_slice < mask_data.shape[2] else None
                elif current_plane == 'coronal':
                    slice_mask = mask_data[:, current_slice, :] if current_slice < mask_data.shape[1] else None
                else:
                    slice_mask = None
            else:
                slice_mask = mask_data

            if slice_mask is not None:
                # Count voxels in current slice only
                slice_voxels = int(slice_mask.sum()) if hasattr(slice_mask, 'sum') else 0

                # Calculate total volume (for reference)
                total_voxels = int(mask_data.sum()) if hasattr(mask_data, 'sum') else 0
                total_volume = total_voxels * 0.001  # mm³ to cm³

                # Update labels
                self._volume_label.setText(f"Vol: {total_volume:.1f} cm³")
                self._voxels_label.setText(f"Voxels: {slice_voxels}")

                if hasattr(self, '_changes_label'):
                    self._changes_label.setText("Current")

        except Exception as e:
            self.logger.warning(f"Could not update live metrics: {e}")

    def _on_slice_changed_from_viewer(self, slice_index: int) -> None:
        """
        Handler for slice changes from the image viewer.
        Updates live metrics when the slice changes.

        Args:
            slice_index: Index of the new slice
        """
        try:
            # Update metrics if there's an active segmentation
            if self._active_overlay_id:
                self._update_live_metrics(self._active_overlay_id)
        except Exception as e:
            self.logger.warning(f"Error updating metrics on slice change: {e}")

    def _clear_history(self) -> None:
        """Clear the undo/redo history."""
        self._history_stack.clear()
        self._history_index = -1
        self._original_segmentation_data = None

    def _force_canvas_update(self) -> None:
        """Force update of the image canvas."""
        try:
            # Get main window and image viewer
            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                image_viewer = main_window._ui_components.central_viewer
                if image_viewer and hasattr(image_viewer, '_update_canvas_for_plane'):
                    # Update canvas for current plane
                    if hasattr(image_viewer, '_active_view_plane'):
                        image_viewer._update_canvas_for_plane(image_viewer._active_view_plane)
                    else:
                        # Fallback: update all canvases
                        if hasattr(image_viewer, '_update_all_canvases'):
                            image_viewer._update_all_canvases()
        except Exception as e:
            self.logger.error(f"Could not force canvas update: {e}")

    def connect_to_image_viewer(self, image_viewer) -> None:
        """
        Connect this panel to the image viewer for full functionality.

        Args:
            image_viewer: The ImageViewer2D instance to connect to
        """
        try:
            # Store reference for future use
            self._image_viewer_ref = image_viewer

            # Connect segmentation tool signals
            self.segmentation_tool_changed.connect(self._on_segmentation_tool_selected_by_viewer)
            self.brush_size_changed.connect(self._on_brush_size_changed_by_viewer)

            # Set overlay service reference if available
            if hasattr(image_viewer, '_overlay_service'):
                self.set_overlay_service_reference(image_viewer._overlay_service)

            # Connect tool activation signals
            for tool_name, tool in self._segmentation_tools.items():
                tool.tool_activated.connect(lambda name=tool_name: self._activate_tool_in_viewer(name))
                tool.tool_deactivated.connect(lambda name=tool_name: self._deactivate_tool_in_viewer(name))


        except Exception as e:
            self.logger.error(f"Error connecting to image viewer: {e}")

    def _on_segmentation_tool_selected_by_viewer(self, tool_name: str) -> None:
        """Handle segmentation tool selection from viewer."""
        try:
            if hasattr(self, '_image_viewer_ref') and self._image_viewer_ref:
                # Activate the tool in the image viewer
                if hasattr(self._image_viewer_ref, '_on_segmentation_tool_changed'):
                    self._image_viewer_ref._on_segmentation_tool_changed(tool_name)
        except Exception as e:
            self.logger.error(f"Could not activate tool {tool_name} in viewer: {e}")

    def _on_brush_size_changed_by_viewer(self, size: int) -> None:
        """Handle brush size change from viewer."""
        try:
            if hasattr(self, '_image_viewer_ref') and self._image_viewer_ref:
                # Update brush size in the image viewer
                if hasattr(self._image_viewer_ref, '_on_brush_size_changed'):
                    self._image_viewer_ref._on_brush_size_changed(size)
        except Exception as e:
            self.logger.error(f"Could not update brush size in viewer: {e}")

    def _activate_tool_in_viewer(self, tool_name: str) -> None:
        """Activate a tool in the connected image viewer."""
        try:
            if hasattr(self, '_image_viewer_ref') and self._image_viewer_ref:
                # Set active tool in viewer
                if hasattr(self._image_viewer_ref, 'set_active_segmentation_tool'):
                    self._image_viewer_ref.set_active_segmentation_tool(tool_name)
        except Exception as e:
            self.logger.error(f"Could not activate {tool_name} in image viewer: {e}")

    def _deactivate_tool_in_viewer(self, tool_name: str) -> None:
        """Deactivate a tool in the connected image viewer."""
        try:
            if hasattr(self, '_image_viewer_ref') and self._image_viewer_ref:
                # Deactivate tool in viewer
                if hasattr(self._image_viewer_ref, 'set_active_segmentation_tool'):
                    self._image_viewer_ref.set_active_segmentation_tool("")
        except Exception as e:
            self.logger.error(f"Could not deactivate {tool_name} in image viewer: {e}")

    def clear_user_edits_only(self) -> bool:
        """
        Limpia solo las ediciones of usuario, preservando las segmentaciones originales.

        Esto restaura las máscaras editadas a su estado original, eliminando
        solo las modificaciones hechas with herramientas of edición (brush, eraser, fill).

        Returns:
            bool: True si se limpiaron ediciones exitosamente, False si no había ediciones o hubo error
        """
        try:
            # Solo proceder si hay historial of ediciones
            if not self._history_stack:
                return True

            overlay_service = self._get_overlay_service()
            if not overlay_service:
                self.logger.error("Cannot access overlay service for clearing edits")
                return False

            # Si tenemos datos originales guardados, restaurar a ese estado
            if self._original_segmentation_data:
                active_overlay_id = self._get_active_overlay_id()
                if active_overlay_id:
                    success = overlay_service.restore_segmentation_data(active_overlay_id, self._original_segmentation_data)
                    if not success:
                        self.logger.error(f"Failed to restore {active_overlay_id}")
                        return False
                else:
                    self.logger.error("No active overlay found for restoration")
                    return False
            else:
                self.logger.error("No original segmentation data found - cannot restore to original state")
                return False

            # Limpiar solo el historial of ediciones, pero mantener los datos originales
            self._history_stack.clear()
            self._history_index = -1

            # NO resetear _original_segmentation_data - lo mantenemos for futuras restauraciones

            # Update botones y UI
            self._update_action_buttons()

            # Forzar actualización of canvas
            self._force_canvas_update()

            return True

        except Exception as e:
            self.logger.error(f"Error clearing user edits: {e}")
            return False