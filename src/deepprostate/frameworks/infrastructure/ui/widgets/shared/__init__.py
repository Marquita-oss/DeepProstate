"""
infrastructure/ui/widgets/shared/__init__.py

Módulo of interfaces y utilidades compartidas.
Define contratos comunes for herramientas of edición manual.
"""

from .tool_interfaces import (
    BaseTool,
    MeasurementToolInterface,
    SegmentationToolInterface
)

__all__ = [
    "BaseTool",
    "MeasurementToolInterface",
    "SegmentationToolInterface"
]