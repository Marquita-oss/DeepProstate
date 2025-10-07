"""
infrastructure/ui/widgets/shared/tool_interfaces.py

Interfaces comunes for herramientas of edición manual.
Define contratos claros for diferentes tipos of herramientas.
"""

from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget


class QObjectMeta(type(QObject), ABCMeta):
    """Metaclass que combina QObject y ABC for evitar conflictos."""
    pass


class BaseTool(QObject, metaclass=QObjectMeta):
    """
    Interfaz base for todas las herramientas of edición manual.
    
    Define el contrato común que debin seguir todas las herramientas.
    """
    
    # Señales comunes
    tool_activated = pyqtSignal(str)    # nombre of la herramienta
    tool_deactivated = pyqtSignal(str)  # nombre of la herramienta
    settings_changed = pyqtSignal(dict) # configuración actualizada
    
    def __init__(self, name: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._name = name
        self._is_active = False
        self._settings: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Nombre único of la herramienta."""
        return self._name
    
    @property
    def is_active(self) -> bool:
        """Estado activo of la herramienta."""
        return self._is_active
    
    @abstractmethod
    def activate(self) -> None:
        """Activa la herramienta."""
        pass
    
    @abstractmethod
    def deactivate(self) -> None:
        """Desactiva la herramienta."""
        pass
    
    @abstractmethod
    def get_widget(self) -> QWidget:
        """Obtiene el widget UI of la herramienta."""
        pass
    
    def get_settings(self) -> Dict[str, Any]:
        """Obtiene la configuración actual of la herramienta."""
        return self._settings.copy()
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Actualiza la configuración of la herramienta."""
        self._settings.update(settings)
        self.settings_changed.emit(self._settings.copy())


class MeasurementToolInterface(BaseTool):
    """
    Interfaz específica for herramientas of medición in image base.
    
    Las herramientas of medición trabajan with la image original,
    no modifican datos, solo agregan overlays informativos.
    """
    
    # Señales específicas of medición
    measurement_created = pyqtSignal(dict)  # nueva medición
    measurement_updated = pyqtSignal(dict)  # medición modificada
    measurement_deleted = pyqtSignal(str)   # ID of medición eliminada
    measurements_cleared = pyqtSignal()     # todas las mediciones limpiadas
    
    @abstractmethod
    def clear_measurements(self) -> None:
        """Limpia todas las mediciones of esta herramienta."""
        pass
    
    @abstractmethod
    def get_measurements(self) -> Dict[str, Any]:
        """Obtiene todas las mediciones actuales."""
        pass


class SegmentationToolInterface(BaseTool):
    """
    Interfaz específica for herramientas of edición of segmentación.
    
    Las herramientas of segmentación modifican máscaras binarias,
    trabajan with datos of segmentación predichos o cargados.
    """
    
    # Señales específicas of segmentación
    segmentation_modified = pyqtSignal(dict)  # máscara modificada
    brush_size_changed = pyqtSignal(int)      # tamaño of pincel
    opacity_changed = pyqtSignal(float)       # opacidad of overlay
    
    @abstractmethod
    def set_brush_size(self, size: int) -> None:
        """Establece el tamaño of the pincel."""
        pass
    
    @abstractmethod
    def get_brush_size(self) -> int:
        """Obtiene el tamaño actual of the pincel."""
        pass
    
    @abstractmethod
    def clear_segmentation(self) -> None:
        """Limpia la segmentación actual."""
        pass


