import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from src.core.domain.entities.medical_image import MedicalImage, ImagePlaneType
from src.core.domain.services.slice_interpolation_service import SliceInterpolationService


class ViewLayoutMode(Enum):
    SINGLE = "single"
    QUAD = "quad"


class MultiViewManagerService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._interpolation_service = SliceInterpolationService()
        
        self._layout_mode = ViewLayoutMode.SINGLE
        self._active_view_plane = "axial"
        
        self._slice_indices = {
            "axial": 0,
            "sagittal": 0,
            "coronal": 0
        }
        
        self._slice_ranges = {
            "axial": (0, 0),
            "sagittal": (0, 0),
            "coronal": (0, 0)
        }
                
        self._current_image: Optional[MedicalImage] = None
        
        self.logger.info("MultiViewManagerService initialized")
    
    def set_layout_mode(self, mode: ViewLayoutMode) -> None:
        self._layout_mode = mode
        self.logger.info(f"Layout mode set to: {mode.value}")
    
    def get_layout_mode(self) -> ViewLayoutMode:
        return self._layout_mode
    
    def set_active_view_plane(self, plane: str) -> None:
        if plane in self._slice_indices:
            self._active_view_plane = plane
            self.logger.debug(f"Active view plane set to: {plane}")
        else:
            self.logger.warning(f"Invalid view plane: {plane}")
    
    def get_active_view_plane(self) -> str:
        return self._active_view_plane
    
    def set_image(self, image: MedicalImage) -> None:
        self._current_image = image
        
        if image and image.image_data is not None:
            shape = image.image_data.shape
            
            if len(shape) >= 3:
                self._slice_ranges["axial"] = (0, shape[0] - 1)
                self._slice_ranges["sagittal"] = (0, shape[2] - 1)
                self._slice_ranges["coronal"] = (0, shape[1] - 1)
            else:
                self._slice_ranges["axial"] = (0, 0)
                self._slice_ranges["sagittal"] = (0, 0)
                self._slice_ranges["coronal"] = (0, 0)
            
            for plane in self._slice_indices:
                min_slice, max_slice = self._slice_ranges[plane]
                self._slice_indices[plane] = min(self._slice_indices[plane], max_slice)
                self._slice_indices[plane] = max(self._slice_indices[plane], min_slice)
            
            self.logger.info(f"Image set with slice ranges: {self._slice_ranges}")
        else:
            self.logger.warning("No image data provided")
    
    def set_slice_index(self, plane: str, slice_index: int) -> bool:
        if plane not in self._slice_indices:
            self.logger.warning(f"Invalid plane: {plane}")
            return False
        
        min_slice, max_slice = self._slice_ranges[plane]

        clamped_index = max(min_slice, min(slice_index, max_slice))

        if clamped_index != slice_index:
            self.logger.debug(f"Slice index {slice_index} clamped to valid range [{min_slice}, {max_slice}] -> {clamped_index}")

        self._slice_indices[plane] = clamped_index
        return True
    
    def get_slice_index(self, plane: str) -> int:
        return self._slice_indices.get(plane, 0)
    
    def get_slice_range(self, plane: str) -> Tuple[int, int]:
        return self._slice_ranges.get(plane, (0, 0))
    
    def get_active_slice_index(self) -> int:
        return self.get_slice_index(self._active_view_plane)
    
    def get_active_slice_range(self) -> Tuple[int, int]:
        return self.get_slice_range(self._active_view_plane)
    
    def next_slice(self, plane: Optional[str] = None) -> bool:
        target_plane = plane or self._active_view_plane
        current_index = self.get_slice_index(target_plane)
        return self.set_slice_index(target_plane, current_index + 1)
    
    def previous_slice(self, plane: Optional[str] = None) -> bool:
        target_plane = plane or self._active_view_plane
        current_index = self.get_slice_index(target_plane)
        return self.set_slice_index(target_plane, current_index - 1)
    
    def get_slice_data_for_plane(self, plane: str) -> Optional[Dict[str, Any]]:
        if not self._current_image or self._current_image.image_data is None:
            return None
        
        slice_index = self.get_slice_index(plane)        
        image_data = self._current_image.image_data
        
        try:
            if plane == "axial":
                slice_data = image_data[slice_index, :, :] if len(image_data.shape) >= 3 else image_data
            elif plane == "sagittal":
                slice_data = image_data[:, :, slice_index] if len(image_data.shape) >= 3 else image_data
            elif plane == "coronal":
                slice_data = image_data[:, slice_index, :] if len(image_data.shape) >= 3 else image_data
            else:
                self.logger.warning(f"Unknown plane: {plane}")
                return None
            
            original_spacing = (
                self._current_image.spacing.x,
                self._current_image.spacing.y, 
                self._current_image.spacing.z
            )
            
            interpolated_slice, adjusted_spacing = self._interpolation_service.process_slice_with_interpolation(
                slice_data, original_spacing, plane
            )
            
            if plane == "axial":
                slice_spacing = (adjusted_spacing[0], adjusted_spacing[1])
            elif plane == "sagittal":
                slice_spacing = (adjusted_spacing[2], adjusted_spacing[1])
            elif plane == "coronal":
                slice_spacing = (adjusted_spacing[2], adjusted_spacing[0])
            else:
                slice_spacing = (adjusted_spacing[0], adjusted_spacing[1])
            
            self.logger.debug(f"Final spacing for {plane}: {slice_spacing}")
            slice_data = interpolated_slice
            
            return {
                'plane': plane,
                'slice_index': slice_index,
                'slice_data': slice_data,
                'image_spacing': slice_spacing,
                'slice_range': self.get_slice_range(plane)
            }
            
        except IndexError as e:
            self.logger.error(f"Error extracting slice for {plane} at index {slice_index}: {e}")
            return None
    
    def get_all_view_states(self) -> Dict[str, Any]:
        return {
            'layout_mode': self._layout_mode.value,
            'active_view_plane': self._active_view_plane,
            'slice_indices': self._slice_indices.copy(),
            'slice_ranges': self._slice_ranges.copy(),
            'has_image': self._current_image is not None
        }
    
    def restore_view_states(self, states: Dict[str, Any]) -> None:
        try:
            if 'layout_mode' in states:
                mode_str = states['layout_mode']
                self._layout_mode = ViewLayoutMode(mode_str)
            
            if 'active_view_plane' in states:
                self._active_view_plane = states['active_view_plane']
            
            if 'slice_indices' in states:
                self._slice_indices.update(states['slice_indices'])
            
            self.logger.debug("View states restored")
            
        except Exception as e:
            self.logger.error(f"Error restoring view states: {e}")
    
    def get_available_planes(self) -> List[str]:
        return list(self._slice_indices.keys())
    
    def get_current_image(self):
        return self._current_image
    
    def is_3d_image(self) -> bool:
        if not self._current_image or self._current_image.image_data is None:
            return False
        
        return len(self._current_image.image_data.shape) >= 3
    
    def can_navigate_slice(self, plane: str, direction: int) -> bool:
        current_index = self.get_slice_index(plane)
        min_slice, max_slice = self.get_slice_range(plane)
        
        if direction > 0:
            return current_index < max_slice
        elif direction < 0:
            return current_index > min_slice
        else:
            return True