"""
infrastructure/ui/widgets/tools/segmentation_tools.py

Herramientas of edición of segmentación for máscaras predichas/cargadas.
Responsabilidad única: modificación of datos of segmentación.
"""

import logging
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QButtonGroup, QLabel, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..shared.tool_interfaces import SegmentationToolInterface


class BrushTool(SegmentationToolInterface):
    """
    Herramienta of pincel for agregar áreas a la segmentación.
    
    Permite al usuario dibujar about la image for agregar píxeles
    a la máscara of segmentación actual.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("brush", parent)
        self.logger = logging.getLogger(__name__)
        
        # Configuración of the pincel
        self._brush_size = 5
        self._brush_hardness = 1.0  # 0.0 = suave, 1.0 = duro
        
        self._settings = {
            "size": self._brush_size,
            "hardness": self._brush_hardness,
            "color": "#FF0000",
            "opacity": 0.7
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configura la interfaz of la herramienta of pincel."""
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        
        # Botón of activación
        self._activate_button = QPushButton("🖌️ Brush Tool")
        self._activate_button.setCheckable(True)
        self._activate_button.setToolTip("Draw to add areas to segmentation")
        self._activate_button.clicked.connect(self._toggle_activation)
        layout.addWidget(self._activate_button)
        
        # Estado
        self._status_label = QLabel("Click to activate brush tool")
        self._status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._status_label)
        
        # Controles of pincel
        controls_group = QGroupBox("Brush Settings")
        controls_layout = QGridLayout(controls_group)
        
        # Tamaño of the pincel
        controls_layout.addWidget(QLabel("Size:"), 0, 0)
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 50)
        self._size_slider.setValue(self._brush_size)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        controls_layout.addWidget(self._size_slider, 0, 1)
        
        self._size_spinbox = QSpinBox()
        self._size_spinbox.setRange(1, 50)
        self._size_spinbox.setValue(self._brush_size)
        self._size_spinbox.setSuffix("px")
        self._size_spinbox.valueChanged.connect(self._on_size_changed)
        controls_layout.addWidget(self._size_spinbox, 0, 2)
        
        # Dureza of the pincel
        controls_layout.addWidget(QLabel("Hardness:"), 1, 0)
        self._hardness_slider = QSlider(Qt.Orientation.Horizontal)
        self._hardness_slider.setRange(0, 100)
        self._hardness_slider.setValue(int(self._brush_hardness * 100))
        self._hardness_slider.valueChanged.connect(self._on_hardness_changed)
        controls_layout.addWidget(self._hardness_slider, 1, 1, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Información of uso
        usage_label = QLabel("💡 Tip: Click and drag to add areas to segmentation")
        usage_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        usage_label.setWordWrap(True)
        layout.addWidget(usage_label)
    
    def activate(self) -> None:
        """Activa la herramienta of pincel."""
        if not self._is_active:
            self._is_active = True
            self._activate_button.setChecked(True)
            self._status_label.setText("Active: Click and drag to paint")
            self._status_label.setStyleSheet("color: #E91E63; font-weight: bold;")
            self.tool_activated.emit(self._name)
            self.logger.info("Brush tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of pincel."""
        if self._is_active:
            self._is_active = False
            self._activate_button.setChecked(False)
            self._status_label.setText("Click to activate brush tool")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            self.tool_deactivated.emit(self._name)
            self.logger.info("Brush tool deactivated")
    
    def _toggle_activation(self) -> None:
        """Alterna el estado of activación."""
        if self._is_active:
            self.deactivate()
        else:
            self.activate()
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def set_brush_size(self, size: int) -> None:
        """Establece el tamaño of the pincel."""
        self._brush_size = max(1, min(50, size))
        self._size_slider.setValue(self._brush_size)
        self._size_spinbox.setValue(self._brush_size)
        self._settings["size"] = self._brush_size
        self.brush_size_changed.emit(self._brush_size)
        self.settings_changed.emit(self._settings.copy())
    
    def get_brush_size(self) -> int:
        """Obtiene el tamaño actual of the pincel."""
        return self._brush_size
    
    def _on_size_changed(self, value: int) -> None:
        """Handles el cambio of tamaño of the pincel."""
        if value != self._brush_size:
            self._brush_size = value
            self._size_slider.setValue(value)
            self._size_spinbox.setValue(value)
            self._settings["size"] = value
            self.brush_size_changed.emit(value)
            self.settings_changed.emit(self._settings.copy())
            self.logger.debug(f"Brush size changed to {value}px")
    
    def _on_hardness_changed(self, value: int) -> None:
        """Handles el cambio of dureza of the pincel."""
        self._brush_hardness = value / 100.0
        self._settings["hardness"] = self._brush_hardness
        self.settings_changed.emit(self._settings.copy())
        self.logger.debug(f"Brush hardness changed to {self._brush_hardness}")
    
    def clear_segmentation(self) -> None:
        """Limpia la segmentación actual (implementación placeholder)."""
        self.logger.info("Clear segmentation requested from brush tool")
        # En implementación real, esto emitiría señal for limpiar máscara


class EraserTool(SegmentationToolInterface):
    """
    Herramienta of borrador for remover áreas of la segmentación.
    
    Permite al usuario borrar píxeles of la máscara of segmentación
    estableciéndolos como fondo.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("eraser", parent)
        self.logger = logging.getLogger(__name__)
        
        # Configuración of the borrador
        self._brush_size = 8  # Ligeramente más granof by defecto
        self._eraser_strength = 1.0  # 0.0 = suave, 1.0 = completo
        
        self._settings = {
            "size": self._brush_size,
            "strength": self._eraser_strength,
            "mode": "hard"  # "hard" o "soft"
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configura la interfaz of la herramienta of borrador."""
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        
        # Botón of activación
        self._activate_button = QPushButton("🧽 Eraser Tool")
        self._activate_button.setCheckable(True)
        self._activate_button.setToolTip("Erase areas from segmentation")
        self._activate_button.clicked.connect(self._toggle_activation)
        layout.addWidget(self._activate_button)
        
        # Estado
        self._status_label = QLabel("Click to activate eraser tool")
        self._status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._status_label)
        
        # Controles of borrador
        controls_group = QGroupBox("Eraser Settings")
        controls_layout = QGridLayout(controls_group)
        
        # Tamaño of the borrador
        controls_layout.addWidget(QLabel("Size:"), 0, 0)
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 50)
        self._size_slider.setValue(self._brush_size)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        controls_layout.addWidget(self._size_slider, 0, 1)
        
        self._size_spinbox = QSpinBox()
        self._size_spinbox.setRange(1, 50)
        self._size_spinbox.setValue(self._brush_size)
        self._size_spinbox.setSuffix("px")
        self._size_spinbox.valueChanged.connect(self._on_size_changed)
        controls_layout.addWidget(self._size_spinbox, 0, 2)
        
        # Fuerza of the borrador
        controls_layout.addWidget(QLabel("Strength:"), 1, 0)
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(0, 100)
        self._strength_slider.setValue(int(self._eraser_strength * 100))
        self._strength_slider.valueChanged.connect(self._on_strength_changed)
        controls_layout.addWidget(self._strength_slider, 1, 1, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Información of uso
        usage_label = QLabel("💡 Tip: Click and drag to erase areas from segmentation")
        usage_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        usage_label.setWordWrap(True)
        layout.addWidget(usage_label)
    
    def activate(self) -> None:
        """Activa la herramienta of borrador."""
        if not self._is_active:
            self._is_active = True
            self._activate_button.setChecked(True)
            self._status_label.setText("Active: Click and drag to erase")
            self._status_label.setStyleSheet("color: #FF5722; font-weight: bold;")
            self.tool_activated.emit(self._name)
            self.logger.info("Eraser tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of borrador."""
        if self._is_active:
            self._is_active = False
            self._activate_button.setChecked(False)
            self._status_label.setText("Click to activate eraser tool")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            self.tool_deactivated.emit(self._name)
            self.logger.info("Eraser tool deactivated")
    
    def _toggle_activation(self) -> None:
        """Alterna el estado of activación."""
        if self._is_active:
            self.deactivate()
        else:
            self.activate()
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def set_brush_size(self, size: int) -> None:
        """Establece el tamaño of the borrador."""
        self._brush_size = max(1, min(50, size))
        self._size_slider.setValue(self._brush_size)
        self._size_spinbox.setValue(self._brush_size)
        self._settings["size"] = self._brush_size
        self.brush_size_changed.emit(self._brush_size)
        self.settings_changed.emit(self._settings.copy())
    
    def get_brush_size(self) -> int:
        """Obtiene el tamaño actual of the borrador."""
        return self._brush_size
    
    def _on_size_changed(self, value: int) -> None:
        """Handles el cambio of tamaño of the borrador."""
        if value != self._brush_size:
            self._brush_size = value
            self._size_slider.setValue(value)
            self._size_spinbox.setValue(value)
            self._settings["size"] = value
            self.brush_size_changed.emit(value)
            self.settings_changed.emit(self._settings.copy())
            self.logger.debug(f"Eraser size changed to {value}px")
    
    def _on_strength_changed(self, value: int) -> None:
        """Handles el cambio of fuerza of the borrador."""
        self._eraser_strength = value / 100.0
        self._settings["strength"] = self._eraser_strength
        self.settings_changed.emit(self._settings.copy())
        self.logger.debug(f"Eraser strength changed to {self._eraser_strength}")
    
    def clear_segmentation(self) -> None:
        """Limpia la segmentación actual completamente."""
        self.logger.info("Clear entire segmentation requested from eraser tool")
        # En implementación real, esto emitiría señal for limpiar toda la máscara


class FillTool(SegmentationToolInterface):
    """
    Herramienta of relleno for llenar áreas cerradas in la segmentación.
    
    Utiliza algoritmo flood-fill for rellenar áreas conectadas
    basándose in tolerancia of color/intensidad.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("fill", parent)
        self.logger = logging.getLogger(__name__)
        
        # Configuración of the relleno
        self._tolerance = 10  # Tolerancia for flood-fill
        self._fill_mode = "add"  # "add" o "remove"
        
        self._settings = {
            "tolerance": self._tolerance,
            "mode": self._fill_mode,
            "connectivity": 8  # 4 o 8 conectividad
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configura la interfaz of la herramienta of relleno."""
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        
        # Botón of activación
        self._activate_button = QPushButton("🪣 Fill Tool")
        self._activate_button.setCheckable(True)
        self._activate_button.setToolTip("Fill enclosed areas in segmentation")
        self._activate_button.clicked.connect(self._toggle_activation)
        layout.addWidget(self._activate_button)
        
        # Estado
        self._status_label = QLabel("Click to activate fill tool")
        self._status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._status_label)
        
        # Controles of relleno
        controls_group = QGroupBox("Fill Settings")
        controls_layout = QGridLayout(controls_group)
        
        # Tolerancia
        controls_layout.addWidget(QLabel("Tolerance:"), 0, 0)
        self._tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self._tolerance_slider.setRange(0, 100)
        self._tolerance_slider.setValue(self._tolerance)
        self._tolerance_slider.valueChanged.connect(self._on_tolerance_changed)
        controls_layout.addWidget(self._tolerance_slider, 0, 1)
        
        self._tolerance_spinbox = QSpinBox()
        self._tolerance_spinbox.setRange(0, 100)
        self._tolerance_spinbox.setValue(self._tolerance)
        self._tolerance_spinbox.valueChanged.connect(self._on_tolerance_changed)
        controls_layout.addWidget(self._tolerance_spinbox, 0, 2)
        
        # Modo of relleno
        controls_layout.addWidget(QLabel("Mode:"), 1, 0)
        self._mode_add_button = QPushButton("➕ Add")
        self._mode_add_button.setCheckable(True)
        self._mode_add_button.setChecked(True)
        self._mode_add_button.clicked.connect(lambda: self._set_mode("add"))
        controls_layout.addWidget(self._mode_add_button, 1, 1)
        
        self._mode_remove_button = QPushButton("➖ Remove")
        self._mode_remove_button.setCheckable(True)
        self._mode_remove_button.clicked.connect(lambda: self._set_mode("remove"))
        controls_layout.addWidget(self._mode_remove_button, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Información of uso
        usage_label = QLabel("💡 Tip: Click on an area to fill similar connected pixels")
        usage_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        usage_label.setWordWrap(True)
        layout.addWidget(usage_label)
    
    def activate(self) -> None:
        """Activa la herramienta of relleno."""
        if not self._is_active:
            self._is_active = True
            self._activate_button.setChecked(True)
            self._status_label.setText("Active: Click to fill areas")
            self._status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
            self.tool_activated.emit(self._name)
            self.logger.info("Fill tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of relleno."""
        if self._is_active:
            self._is_active = False
            self._activate_button.setChecked(False)
            self._status_label.setText("Click to activate fill tool")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            self.tool_deactivated.emit(self._name)
            self.logger.info("Fill tool deactivated")
    
    def _toggle_activation(self) -> None:
        """Alterna el estado of activación."""
        if self._is_active:
            self.deactivate()
        else:
            self.activate()
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def set_brush_size(self, size: int) -> None:
        """El fill tool no usa brush size, pero implementa la interfaz."""
        self.logger.debug("Fill tool doesn't use brush size")
    
    def get_brush_size(self) -> int:
        """El fill tool no usa brush size."""
        return 1
    
    def _on_tolerance_changed(self, value: int) -> None:
        """Handles el cambio of tolerancia."""
        self._tolerance = value
        self._tolerance_slider.setValue(value)
        self._tolerance_spinbox.setValue(value)
        self._settings["tolerance"] = value
        self.settings_changed.emit(self._settings.copy())
        self.logger.debug(f"Fill tolerance changed to {value}")
    
    def _set_mode(self, mode: str) -> None:
        """Establece el modo of relleno (add/remove)."""
        self._fill_mode = mode
        self._settings["mode"] = mode
        
        # Update botones
        self._mode_add_button.setChecked(mode == "add")
        self._mode_remove_button.setChecked(mode == "remove")
        
        self.settings_changed.emit(self._settings.copy())
        self.logger.debug(f"Fill mode changed to {mode}")
    
    def clear_segmentation(self) -> None:
        """Limpia la segmentación actual."""
        self.logger.info("Clear segmentation requested from fill tool")
        # En implementación real, esto emitiría señal for limpiar máscara