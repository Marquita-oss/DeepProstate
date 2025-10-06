import logging
import numpy as np
import pydicom
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from src.core.domain.entities.medical_image import MedicalImage, ImageModalityType
from .dicom_file_handler import DicomFileInfo


@dataclass
class ValidationResult:
    is_valid: bool
    error_messages: List[str]
    warnings: List[str]
    validated_data: Optional[Dict[str, Any]] = None


class ImageValidator:
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("ImageValidator initialized")
        
        self.MIN_MEDICAL_IMAGE_SIZE = 64  
        self.MAX_MEDICAL_IMAGE_SIZE = 2048 
        self.MIN_SPACING = 0.01  
        self.MAX_SPACING = 50.0  
    
    def validate_medical_image_data(self, image_data: np.ndarray, 
                                  modality: Optional[ImageModalityType] = None) -> ValidationResult:
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            if image_data is None:
                errors.append("Image data is None")
                return ValidationResult(False, errors, warnings)
            
            if not isinstance(image_data, np.ndarray):
                errors.append(f"Image data must be numpy array, got {type(image_data)}")
                return ValidationResult(False, errors, warnings)
            
            dimensions = image_data.shape
            validated_data['dimensions'] = dimensions
            
            if len(dimensions) < 2:
                errors.append(f"Image must be at least 2D, got {len(dimensions)}D")
            elif len(dimensions) > 4:
                errors.append(f"Image dimensions too high ({len(dimensions)}D), max 4D supported")
            
            for i, size in enumerate(dimensions):
                if size < self.MIN_MEDICAL_IMAGE_SIZE:
                    if len(dimensions) == 2 or i < 2:  
                        errors.append(f"Dimension {i} too small for medical imaging: {size} < {self.MIN_MEDICAL_IMAGE_SIZE}")
                    else:
                        warnings.append(f"Dimension {i} small but may be valid for medical data: {size}")
                
                if size > self.MAX_MEDICAL_IMAGE_SIZE:
                    warnings.append(f"Dimension {i} very large, may cause memory issues: {size}")
            
            validated_data['dtype'] = str(image_data.dtype)
            
            if image_data.dtype == np.bool_:
                warnings.append("Boolean image data - may be segmentation mask")
            elif image_data.dtype in [np.int8, np.uint8]:
                warnings.append("8-bit image data - may have limited dynamic range for medical imaging")
            
            min_val = float(np.min(image_data))
            max_val = float(np.max(image_data))
            validated_data['value_range'] = (min_val, max_val)
            
            self._validate_modality_specific(image_data, modality, errors, warnings)
            
            memory_mb = image_data.nbytes / (1024 * 1024)
            validated_data['memory_usage_mb'] = memory_mb
            
            if memory_mb > 1024:  # > 1GB
                warnings.append(f"Large image data may cause memory issues: {memory_mb:.1f} MB")
            
            is_valid = len(errors) == 0
            self._logger.debug(f"Image validation result: {'PASSED' if is_valid else 'FAILED'}")
            
            return ValidationResult(is_valid, errors, warnings, validated_data)
            
        except Exception as e:
            self._logger.error(f"Error during image validation: {e}")
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def _validate_modality_specific(self, image_data: np.ndarray, 
                                  modality: Optional[ImageModalityType],
                                  errors: List[str], 
                                  warnings: List[str]) -> None:
        if modality is None:
            return
            
        min_val = float(np.min(image_data))
        max_val = float(np.max(image_data))
        
        if modality == ImageModalityType.CT:
            if min_val < -2000 or max_val > 5000:
                warnings.append(f"CT values outsiof typical range: [{min_val:.1f}, {max_val:.1f}]")
            
        elif modality == ImageModalityType.MRI:
            if min_val < 0:
                warnings.append("MRI image contains negative values - unusual but may be valid")
            if max_val > 20000:
                warnings.append(f"MRI values very high: max={max_val:.1f}")
        
        elif modality == ImageModalityType.ULTRASOUND:
            if max_val > 255 and image_data.dtype in [np.uint8]:
                errors.append("Ultrasound 8-bit image has values > 255")
    
    def validate_dicom_series(self, dicom_files: List[DicomFileInfo]) -> ValidationResult:
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            if not dicom_files:
                errors.append("No DICOM files provided")
                return ValidationResult(False, errors, warnings)
            
            series_uids = set(file_info.series_uid for file_info in dicom_files)
            if len(series_uids) > 1:
                warnings.append(f"Multiple series UIDs found: {len(series_uids)}")
            
            validated_data['series_count'] = len(series_uids)
            validated_data['file_count'] = len(dicom_files)
            
            modalities = set(file_info.modality for file_info in dicom_files)
            if len(modalities) > 1:
                warnings.append(f"Multiple modalities in series: {modalities}")
            
            validated_data['modalities'] = list(modalities)
            valid_files = [f for f in dicom_files if f.is_valid]
            invalid_files = [f for f in dicom_files if not f.is_valid]
            
            if invalid_files:
                errors.append(f"{len(invalid_files)} invalid DICOM files found")
                for invalid_file in invalid_files[:5]: 
                    errors.append(f"  - {invalid_file.file_path.name}: {invalid_file.error_message}")
            
            if len(valid_files) < len(dicom_files) * 0.8:  
                errors.append("Too many invalid files in series")
            
            validated_data['valid_files'] = len(valid_files)
            validated_data['invalid_files'] = len(invalid_files)
            
            if len(valid_files) < 5:
                warnings.append(f"Small series size: {len(valid_files)} files")
            elif len(valid_files) > 1000:
                warnings.append(f"Very large series: {len(valid_files)} files - may cause performance issues")
            
            is_valid = len(errors) == 0 and len(valid_files) > 0
            
            return ValidationResult(is_valid, errors, warnings, validated_data)
            
        except Exception as e:
            self._logger.error(f"Error validating DICOM series: {e}")
            errors.append(f"Series validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def validate_spacing_values(self, spacing: Tuple[float, float, float]) -> ValidationResult:
        errors = []
        warnings = []
        
        try:
            if len(spacing) != 3:
                errors.append(f"Spacing must have 3 values, got {len(spacing)}")
                return ValidationResult(False, errors, warnings)
            
            for i, value in enumerate(spacing):
                axis = ['X', 'Y', 'Z'][i]
                
                if value <= 0:
                    errors.append(f"{axis} spacing must be positive, got {value}")
                elif value < self.MIN_SPACING:
                    errors.append(f"{axis} spacing too small for medical imaging: {value} mm < {self.MIN_SPACING} mm")
                elif value > self.MAX_SPACING:
                    warnings.append(f"{axis} spacing very large: {value} mm - check if correct")
                
                if value < 0.1:
                    warnings.append(f"{axis} spacing very fine: {value} mm - high resolution imaging")
                elif value > 10.0:
                    warnings.append(f"{axis} spacing coarse: {value} mm - low resolution")
            
            if not (abs(spacing[0] - spacing[1]) < 0.01):
                warnings.append("Anisotropic in-plane spacing detected")
            
            if abs(spacing[2] - spacing[0]) > spacing[0]:
                warnings.append("Thick slice spacing - anisotropic volume")
            
            is_valid = len(errors) == 0
            validated_data = {
                'spacing': spacing,
                'is_isotropic': all(abs(s - spacing[0]) < 0.01 for s in spacing),
                'min_spacing': min(spacing),
                'max_spacing': max(spacing)
            }
            
            return ValidationResult(is_valid, errors, warnings, validated_data)
            
        except Exception as e:
            self._logger.error(f"Error validating spacing: {e}")
            errors.append(f"Spacing validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def validate_window_level(self, window: float, level: float, 
                            modality: Optional[ImageModalityType] = None) -> ValidationResult:
        errors = []
        warnings = []
        
        try:
            if window <= 0:
                errors.append(f"Window width must be positive, got {window}")
            
            if modality == ImageModalityType.CT:
                if window > 4000:
                    warnings.append(f"Very wiof CT window: {window} HU")
                if abs(level) > 3000:
                    warnings.append(f"CT level outsiof typical range: {level} HU")
                    
            elif modality == ImageModalityType.MRI:
                if level < 0:
                    warnings.append("Negative level unusual for MRI imaging")
                if window > 10000:
                    warnings.append(f"Very wiof MRI window: {window}")
            
            is_valid = len(errors) == 0
            validated_data = {
                'window': window,
                'level': level,
                'min_value': level - window/2,
                'max_value': level + window/2
            }
            
            return ValidationResult(is_valid, errors, warnings, validated_data)
            
        except Exception as e:
            self._logger.error(f"Error validating window/level: {e}")
            errors.append(f"Window/level validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def validate_medical_image_entity(self, medical_image: MedicalImage) -> ValidationResult:
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            image_result = self.validate_medical_image_data(
                medical_image.image_data, 
                medical_image.modality
            )
            
            errors.extend(image_result.error_messages)
            warnings.extend(image_result.warnings)
            if image_result.validated_data:
                validated_data['image_data'] = image_result.validated_data
            
            if medical_image.spacing:
                spacing_tuple = (medical_image.spacing.x, medical_image.spacing.y, medical_image.spacing.z)
                spacing_result = self.validate_spacing_values(spacing_tuple)
                
                errors.extend(spacing_result.error_messages)
                warnings.extend(spacing_result.warnings)
                if spacing_result.validated_data:
                    validated_data['spacing'] = spacing_result.validated_data
            
            if not medical_image.series_uid or medical_image.series_uid == "UNKNOWN":
                warnings.append("Missing or unknown series UID")
            
            if not medical_image.study_uid or medical_image.study_uid == "UNKNOWN":
                warnings.append("Missing or unknown study UID")
            
            validated_data['entity_info'] = {
                'modality': medical_image.modality.value,
                'series_uid': medical_image.series_uid,
                'study_uid': medical_image.study_uid,
                'has_spacing': medical_image.spacing is not None,
                'has_metadata': bool(medical_image.metadata)
            }
            
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, validated_data)
            
        except Exception as e:
            self._logger.error(f"Error validating medical image entity: {e}")
            errors.append(f"Entity validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)