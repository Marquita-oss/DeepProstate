"""
infrastructure/ui/components/__init__.py

Reusable UI component library for medical imaging applications.
Contains common UI components that eliminate code duplication across the application.
"""

from .common_controls import (
    MedicalButton,
    MedicalDeleteButton,
    MedicalProgressBar,
    MedicalInputField,
    MedicalSlider,
    MedicalGroupBox
)
from .medical_panels import (
    MedicalInfoPanel,
    MedicalControlPanel,
    MedicalStatusPanel,
    MedicalTabWidget
)
from .medical_dialogs import (
    MedicalMessageDialog,
    MedicalProgressDialog,
    MedicalConfirmDialog
)

__all__ = [
    # Common Controls
    'MedicalButton',
    'MedicalDeleteButton', 
    'MedicalProgressBar',
    'MedicalInputField',
    'MedicalSlider',
    'MedicalGroupBox',
    
    # Medical Panels
    'MedicalInfoPanel',
    'MedicalControlPanel',
    'MedicalStatusPanel',
    'MedicalTabWidget',
    
    # Medical Dialogs
    'MedicalMessageDialog',
    'MedicalProgressDialog',
    'MedicalConfirmDialog'
]