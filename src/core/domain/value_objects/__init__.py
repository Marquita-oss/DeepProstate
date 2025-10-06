"""
core/domain/value_objects/__init__.py

Value Objects package for the domain layer.
Value Objects are immutable objects that represent descriptive aspects
of the domain with no conceptual identity.
"""

from .validation_result import ValidationResult

__all__ = [
    'ValidationResult',
]
