#!/usr/bin/env python3
"""
domain/utils/__init__.py

Utilidades for the dominio médico.
"""

from .medical_shape_handler import (
    MedicalShapeHandler,
    MedicalDimensionOrder,
    validate_medical_image_array
)

__all__ = [
    'MedicalShapeHandler',
    'MedicalDimensionOrder', 
    'validate_medical_image_array'
]