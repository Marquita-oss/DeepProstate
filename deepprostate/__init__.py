"""
DeepProstate - AI-powered prostate MRI analysis

A comprehensive medical imaging application for prostate analysis
using deep learning and Clean Architecture principles.
"""

__version__ = "1.4.0"
__author__ = "Ronald Marca"
__email__ = "rnldmarca@gmail.com"

# Import main entities for easy access
from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.core.domain.services.ai_model_service import AIModelService

__all__ = [
    "MedicalImage",
    "AIModelService",
    "__version__",
    "__author__",
    "__email__",
]
