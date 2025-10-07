import numpy as np
from typing import Tuple, Optional
import logging

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    zoom = None


class SliceInterpolationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.thin_slice_threshold = 50  
        self.target_height = 150
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available, slice interpolation disabled")
    
    def should_interpolate_slice(self, slice_data: np.ndarray) -> bool:
        if not SCIPY_AVAILABLE:
            return False
            
        if slice_data.ndim != 2:
            return False
            
        return slice_data.shape[0] < self.thin_slice_threshold
    
    def interpolate_thin_slice(
        self, 
        slice_data: np.ndarray, 
        original_spacing: Tuple[float, float, float],
        plane: str
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        if not self.should_interpolate_slice(slice_data):
            return slice_data, original_spacing
        
        try:
            current_height = slice_data.shape[0]
            scale_factor = self.target_height / current_height
            
            if SCIPY_AVAILABLE:
                interpolated_slice = zoom(
                    slice_data, 
                    (scale_factor, 1.0), 
                    order=1, 
                    mode='nearest'
                )
            else:
                repeat_factor = int(np.ceil(scale_factor))
                interpolated_slice = np.repeat(slice_data, repeat_factor, axis=0)
                target_shape = int(current_height * scale_factor)
                if interpolated_slice.shape[0] > target_shape:
                    interpolated_slice = interpolated_slice[:target_shape, :]
            
            adjusted_spacing = self._adjust_spacing_for_interpolation(
                original_spacing, scale_factor, plane
            )
            
            self.logger.debug(
                f"Interpolated {plane} slice: {slice_data.shape} -> "
                f"{interpolated_slice.shape}, scale: {scale_factor:.2f}"
            )
            
            return interpolated_slice, adjusted_spacing
            
        except Exception as e:
            self.logger.error(f"Error interpolating slice: {e}")
            return slice_data, original_spacing
    
    def _adjust_spacing_for_interpolation(
        self, 
        original_spacing: Tuple[float, float, float],
        scale_factor: float,
        plane: str
    ) -> Tuple[float, float, float]:
        x_spacing, y_spacing, z_spacing = original_spacing

        if plane == 'sagittal':
            adjusted_z_spacing = z_spacing / scale_factor
            return (x_spacing, y_spacing, adjusted_z_spacing)
        elif plane == 'coronal':
            adjusted_y_spacing = y_spacing / scale_factor
            return (x_spacing, adjusted_y_spacing, z_spacing)
        else:  
            adjusted_y_spacing = y_spacing / scale_factor
            return (x_spacing, adjusted_y_spacing, z_spacing)
    
    def process_slice_with_interpolation(
        self,
        slice_data: np.ndarray,
        spacing: Tuple[float, float, float],
        plane: str
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        processed_slice, final_spacing = self.interpolate_thin_slice(
            slice_data, spacing, plane
        )
        
        return processed_slice, final_spacing