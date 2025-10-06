"""
Interfaces of repositorio of the dominio.
"""

from .repositories import (
    MedicalImageRepository,
    SegmentationRepository,
    ProjectRepository,
    ConfigurationRepository,
    RepositoryError,
    ImageNotFoundError,
    SegmentationNotFoundError,
    ProjectNotFoundError,
    DuplicateEntityError,
    InvalidQueryError
)

__all__ = [
    'MedicalImageRepository',
    'SegmentationRepository',
    'ProjectRepository', 
    'ConfigurationRepository',
    'RepositoryError',
    'ImageNotFoundError',
    'SegmentationNotFoundError',
    'ProjectNotFoundError',
    'DuplicateEntityError',
    'InvalidQueryError'
]