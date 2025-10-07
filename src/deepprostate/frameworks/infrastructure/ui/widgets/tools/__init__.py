"""
infrastructure/ui/widgets/tools/__init__.py

Módulo of herramientas of edición manual modularizadas.
Exporta todas las herramientas disponibles with arquitectura limpia.
"""

from .measurement_tools import (
    DistanceMeasurementTool,
    AngleMeasurementTool, 
    ROITool
)

from .segmentation_tools import (
    BrushTool,
    EraserTool,
    FillTool
)

__all__ = [
    # Measurement tools (work on base image)
    "DistanceMeasurementTool",
    "AngleMeasurementTool",
    "ROITool",

    # Segmentation tools (work on masks)
    "BrushTool",
    "EraserTool",
    "FillTool"
]