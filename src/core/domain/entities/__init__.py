"""
Entidades of the dominio médico.
"""

from .medical_image import MedicalImage, ImageSpacing, ImageModalityType, WindowLevel
from .segmentation import MedicalSegmentation, AnatomicalRegion, SegmentationType, ConfidenceLevel
from .analysis_state import AnalysisState, AnalysisStatus

__all__ = [
    'MedicalImage',
    'ImageSpacing',
    'ImageModalityType',
    'WindowLevel',
    'MedicalSegmentation',
    'AnatomicalRegion',
    'SegmentationType',
    'ConfidenceLevel',
    'AnalysisState',
    'AnalysisStatus'
]