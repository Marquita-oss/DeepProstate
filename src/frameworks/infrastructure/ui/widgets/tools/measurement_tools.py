"""
infrastructure/ui/widgets/tools/measurement_tools.py

Herramientas of medición for image base.
Responsabilidad única: medición in image original without modificar datos.
"""

import logging
from typing import Optional, Dict, Any, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QButtonGroup, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..shared.tool_interfaces import MeasurementToolInterface


class DistanceMeasurementTool(MeasurementToolInterface):
    """
    Herramienta for medir distancias in image base.
    
    Permite al usuario hacer clic in dos puntos for medir distancia
    in píxeles o unidades calibradas.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("distance", parent)
        self.logger = logging.getLogger(__name__)
        self._measurements: Dict[str, Dict[str, Any]] = {}
        self._measurement_counter = 0
        
        # Configuración by defecto
        self._settings = {
            "unit": "pixels",
            "show_labels": True,
            "line_color": "#FF0000",
            "line_width": 2
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configuración mínima - UI manejada by panel unificado."""
        self._widget = QWidget()
    
    def activate(self) -> None:
        """Activa la herramienta of medición of distancia."""
        if not self._is_active:
            self._is_active = True
            self.tool_activated.emit(self._name)
            self.logger.info("Distance measurement tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of medición of distancia."""
        if self._is_active:
            self._is_active = False
            self.tool_deactivated.emit(self._name)
            self.logger.info("Distance measurement tool deactivated")
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def add_measurement(self, point1: tuple, point2: tuple, distance: float) -> str:
        """
        Agrega una nueva medición of distancia.
        
        Args:
            point1: (x, y) primer punto
            point2: (x, y) segundo punto  
            distance: distancia calculada
            
        Returns:
            ID único of la medición
        """
        measurement_id = f"dist_{self._measurement_counter}"
        self._measurement_counter += 1
        
        measurement = {
            "id": measurement_id,
            "type": "distance",
            "point1": point1,
            "point2": point2,
            "distance": distance,
            "unit": self._settings["unit"],
            "timestamp": "now"  # En implementación real usar datetime
        }
        
        self._measurements[measurement_id] = measurement
        self.measurement_created.emit(measurement)
        self.logger.debug(f"Distance measurement added: {distance} {self._settings['unit']}")
        
        return measurement_id
    
    def clear_measurements(self) -> None:
        """Limpia todas las mediciones of distancia."""
        self._measurements.clear()
        self._measurement_counter = 0
        self.measurements_cleared.emit()
        self.logger.info("All distance measurements cleared")
    
    def get_measurements(self) -> Dict[str, Any]:
        """Obtiene todas las mediciones actuales."""
        return self._measurements.copy()
    


class AngleMeasurementTool(MeasurementToolInterface):
    """
    Herramienta for medir ángulos in image base.
    
    Permite al usuario hacer clic in tres puntos for medir el ángulo
    formado entre dos líneas.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("angle", parent)
        self.logger = logging.getLogger(__name__)
        self._measurements: Dict[str, Dict[str, Any]] = {}
        self._measurement_counter = 0
        
        self._settings = {
            "unit": "degrees",
            "show_labels": True,
            "line_color": "#0000FF",
            "line_width": 2
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configuración mínima - UI manejada by panel unificado."""
        self._widget = QWidget()
    
    def activate(self) -> None:
        """Activa la herramienta of medición of ángulos."""
        if not self._is_active:
            self._is_active = True
            self.tool_activated.emit(self._name)
            self.logger.info("Angle measurement tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of medición of ángulos."""
        if self._is_active:
            self._is_active = False
            self.tool_deactivated.emit(self._name)
            self.logger.info("Angle measurement tool deactivated")
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def add_measurement(self, point1: tuple, vertex: tuple, point3: tuple, angle: float) -> str:
        """Agrega una nueva medición of ángulo."""
        measurement_id = f"angle_{self._measurement_counter}"
        self._measurement_counter += 1
        
        measurement = {
            "id": measurement_id,
            "type": "angle",
            "point1": point1,
            "vertex": vertex,
            "point3": point3,
            "angle": angle,
            "unit": self._settings["unit"],
            "timestamp": "now"
        }
        
        self._measurements[measurement_id] = measurement
        self.measurement_created.emit(measurement)
        self.logger.debug(f"Angle measurement added: {angle} {self._settings['unit']}")
        
        return measurement_id
    
    def clear_measurements(self) -> None:
        """Limpia todas las mediciones of ángulos."""
        self._measurements.clear()
        self._measurement_counter = 0
        self.measurements_cleared.emit()
        self.logger.info("All angle measurements cleared")
    
    def get_measurements(self) -> Dict[str, Any]:
        """Obtiene todas las mediciones of ángulos."""
        return self._measurements.copy()


class ROITool(MeasurementToolInterface):
    """
    Herramienta for crear regiones of interés (ROI) in image base.
    
    Permite definir áreas rectangulares, circulares o of forma libre
    for análisis posterior.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("roi", parent)
        self.logger = logging.getLogger(__name__)
        self._measurements: Dict[str, Dict[str, Any]] = {}  # ROIs son tipo especial of "medición"
        self._measurement_counter = 0
        
        self._settings = {
            "shape": "rectangle",
            "fill_color": "#00FF0050",
            "border_color": "#00FF00",
            "border_width": 2
        }
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configuración mínima - UI manejada by panel unificado."""
        self._widget = QWidget()
    
    def activate(self) -> None:
        """Activa la herramienta of ROI."""
        if not self._is_active:
            self._is_active = True
            self.tool_activated.emit(self._name)
            self.logger.info("ROI tool activated")
    
    def deactivate(self) -> None:
        """Desactiva la herramienta of ROI."""
        if self._is_active:
            self._is_active = False
            self.tool_deactivated.emit(self._name)
            self.logger.info("ROI tool deactivated")
    
    def get_widget(self) -> QWidget:
        """Obtiene el widget of la herramienta."""
        return self._widget
    
    def add_roi(self, shape_type: str, coordinates: Dict[str, Any], area: float) -> str:
        """Agrega una nueva ROI."""
        roi_id = f"roi_{self._measurement_counter}"
        self._measurement_counter += 1
        
        roi = {
            "id": roi_id,
            "type": "roi",
            "shape": shape_type,
            "coordinates": coordinates,
            "area": area,
            "unit": "pixels²",
            "timestamp": "now"
        }
        
        self._measurements[roi_id] = roi
        self.measurement_created.emit(roi)
        self.logger.debug(f"ROI added: {shape_type} with area {area}")
        
        return roi_id
    
    def clear_measurements(self) -> None:
        """Limpia todas las ROIs."""
        self._measurements.clear()
        self._measurement_counter = 0
        self.measurements_cleared.emit()
        self.logger.info("All ROIs cleared")
    
    def get_measurements(self) -> Dict[str, Any]:
        """Obtiene todas las ROIs."""
        return self._measurements.copy()