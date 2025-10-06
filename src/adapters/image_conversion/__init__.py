"""
adapters/image_conversion

Module for converting medical image formats.

This package provides adapters for bridging domain entities with external
image formats like NIfTI, DICOM, etc.
"""

from .nifti_converter import NIfTIConverter, NIfTIConversionError
from .temp_file_manager import TempFileManager, get_temp_file_manager

__all__ = [
    'NIfTIConverter',
    'NIfTIConversionError',
    'TempFileManager',
    'get_temp_file_manager',
]
