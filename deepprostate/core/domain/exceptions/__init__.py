"""
domain/exceptions/__init__.py

Domain-specific exceptions for the medical imaging workstation.
"""

from .medical_exceptions import (
    # Base exception
    MedicalImagingError,
    
    # Domain-specific exceptions
    DicomProcessingError,
    ImageLoadingError,
    AIAnalysisError,
    SegmentationError,
    DataValidationError,
    WorkflowExecutionError,
    StorageError,
    ConfigurationError,
    SecurityError,
    MemoryError,
    UIError,
    
    # Utility functions
    create_error_context,
    handle_exception_with_context
)

__all__ = [
    # Base exception
    'MedicalImagingError',
    
    # Domain-specific exceptions
    'DicomProcessingError',
    'ImageLoadingError', 
    'AIAnalysisError',
    'SegmentationError',
    'DataValidationError',
    'WorkflowExecutionError',
    'StorageError',
    'ConfigurationError',
    'SecurityError',
    'MemoryError',
    'UIError',
    
    # Utility functions
    'create_error_context',
    'handle_exception_with_context'
]