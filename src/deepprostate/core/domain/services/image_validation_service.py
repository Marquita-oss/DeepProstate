import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

from deepprostate.core.domain.entities.medical_image import MedicalImage, ImageSpacing


class ImageValidationService:
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        self._min_dimensions = (64, 64, 16) 
        self._max_dimensions = (1024, 1024, 512)
        self._preferred_spacing = (0.5, 0.5, 3.0) 
        self._max_spacing_deviation = 2.0
        
        self._logger.info("ImageValidationService initialized with Clean Architecture")
    
    async def validate_for_prostate_segmentation(self, image: MedicalImage) -> None:
        try:
            self._validate_basic_requirements(image)
            
            self._validate_prostate_specific_requirements(image)
            
            self._validate_format_requirements(image)
            
            self._logger.info(f"Image validation passed for {image.image_id}")
            
        except Exception as e:
            self._logger.error(f"Image validation failed for {image.image_id}: {e}")
            raise ImageValidationError(f"Validation failed: {e}")
    
    def _validate_basic_requirements(self, image: MedicalImage) -> None:
        if image.image_data is None:
            raise ImageValidationError("Image data is None")
        
        if image.image_data.size == 0:
            raise ImageValidationError("Image data is empty")
        
        shape = image.image_data.shape
        if len(shape) != 3:
            raise ImageValidationError(f"Expected 3D image, got {len(shape)}D")
        
        for i, (dim, min_dim, max_dim) in enumerate(zip(shape, self._min_dimensions, self._max_dimensions)):
            if dim < min_dim:
                raise ImageValidationError(
                    f"Dimension {i} too small: {dim} < {min_dim}"
                )
            if dim > max_dim:
                raise ImageValidationError(
                    f"Dimension {i} too large: {dim} > {max_dim}"
                )
    
    def _validate_prostate_specific_requirements(self, image: MedicalImage) -> None:
        if image.pixel_spacing:
            self._validate_pixel_spacing(image.pixel_spacing)
        
        self._validate_intensity_range(image.image_data)
        
        if hasattr(image, 'orientation') and image.orientation:
            self._validate_anatomical_orientation(image.orientation)
    
    def _validate_pixel_spacing(self, spacing: ImageSpacing) -> None:
        spacings = [spacing.x, spacing.y, spacing.z]
        
        for i, (current, preferred) in enumerate(zip(spacings, self._preferred_spacing)):
            if current <= 0:
                raise ImageValidationError(f"Invalid spacing[{i}]: {current} <= 0")
            
            ratio = max(current / preferred, preferred / current)
            if ratio > self._max_spacing_deviation:
                self._logger.warning(
                    f"Spacing[{i}] significantly different from preferred: "
                    f"{current} vs {preferred} (ratio: {ratio:.2f})"
                )
    
    def _validate_intensity_range(self, image_data: np.ndarray) -> None:
        min_val = float(np.min(image_data))
        max_val = float(np.max(image_data))
        
        unique_values = np.unique(image_data)
        
        if len(unique_values) <= 2 and np.array_equal(unique_values, [0, 1]):
            raise ImageValidationError("Image appears to be a binary mask, not medical image")
        
        if max_val - min_val < 10:
            raise ImageValidationError(
                f"Intensity range too small: {max_val - min_val} "
                "(possible corrupted or preprocessed image)"
            )
        
        mean_val = float(np.mean(image_data))
        std_val = float(np.std(image_data))
        self._logger.debug(
            f"Intensity stats - Min: {min_val:.2f}, Max: {max_val:.2f}, "
            f"Mean: {mean_val:.2f}, Std: {std_val:.2f}"
        )
    
    def _validate_anatomical_orientation(self, orientation: str) -> None:
        valid_orientations = [
            'RAS', 'LAS', 'RPS', 'LPS',  # Axial
            'RSA', 'LSA', 'RPA', 'LPA',  # Coronal
            'ASR', 'ASL', 'PSR', 'PSL'   # Sagital
        ]
        
        if orientation not in valid_orientations:
            self._logger.warning(f"Unusual anatomical orientation: {orientation}")
    
    def _validate_format_requirements(self, image: MedicalImage) -> None:
        if not image.image_id or not image.image_id.strip():
            raise ImageValidationError("Image must have a valid image_id")
        
        if not np.issubdtype(image.image_data.dtype, np.number):
            raise ImageValidationError(
                f"Image data must be numeric, got {image.image_data.dtype}"
            )
        
        if np.any(np.isnan(image.image_data)):
            raise ImageValidationError("Image contains NaN values")
        
        if np.any(np.isinf(image.image_data)):
            raise ImageValidationError("Image contains infinite values")
    
    def get_validation_report(self, image: MedicalImage) -> Dict[str, Any]:
        report = {
            "image_id": image.image_id,
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        try:
            shape = image.image_data.shape
            report["stats"] = {
                "dimensions": shape,
                "total_voxels": int(np.prod(shape)),
                "data_type": str(image.image_data.dtype),
                "memory_usage_mb": image.image_data.nbytes / (1024 * 1024)
            }
            
            if image.image_data.size > 0:
                report["stats"]["intensity"] = {
                    "min": float(np.min(image.image_data)),
                    "max": float(np.max(image.image_data)),
                    "mean": float(np.mean(image.image_data)),
                    "std": float(np.std(image.image_data))
                }
            
            if image.pixel_spacing:
                report["stats"]["spacing"] = {
                    "x": image.pixel_spacing.x,
                    "y": image.pixel_spacing.y,
                    "z": image.pixel_spacing.z
                }
            
            self.validate_for_prostate_segmentation(image)
            
        except ImageValidationError as e:
            report["valid"] = False
            report["errors"].append(str(e))
        except Exception as e:
            report["valid"] = False
            report["errors"].append(f"Unexpected error: {e}")
        
        return report
    
    def is_suitable_for_ai_processing(self, image: MedicalImage) -> bool:
        try:
            self.validate_for_prostate_segmentation(image)
            return True
        except (ImageValidationError, Exception):
            return False


class ImageValidationError(Exception):
    pass