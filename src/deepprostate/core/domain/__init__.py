"""
Capa of Dominio - Clean Architecture
Contiene entidades médicas y reglas of negocio puras.
"""

# Exportar entidades principales for imports fáciles
from .entities.medical_image import MedicalImage, ImageSpacing, ImageModalityType, WindowLevel
from .entities.segmentation import MedicalSegmentation, AnatomicalRegion, SegmentationType

# Exportar interfaces of repositorio
from .repositories.repositories import (
    MedicalImageRepository, 
    SegmentationRepository,
    ProjectRepository,
    RepositoryError,
    ImageNotFoundError
)

__all__ = [
    # Entidades
    'MedicalImage',
    'ImageSpacing', 
    'ImageModalityType',
    'WindowLevel',
    'MedicalSegmentation',
    'AnatomicalRegion',
    'SegmentationType',
    
    # Repositorios
    'MedicalImageRepository',
    'SegmentationRepository', 
    'ProjectRepository',
    'RepositoryError',
    'ImageNotFoundError'
]