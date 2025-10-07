import numpy as np
import logging
from typing import Tuple, Union, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MedicalDimensionOrder(Enum):
    DHW = "depth_height_width" 
    HWD = "height_width_depth" 
    WHD = "width_height_depth" 


class MedicalShapeHandler:    
    STANDARD_ORDER = MedicalDimensionOrder.DHW
    
    @staticmethod
    def validate_medical_shape(
        array: np.ndarray, 
        expected_dims: int = 3,
        min_size: Optional[Tuple[int, ...]] = None
    ) -> bool:
        if len(array.shape) != expected_dims:
            raise ValueError(f"Expected {expected_dims}D array, got {len(array.shape)}D: {array.shape}")
        
        if expected_dims == 3:
            depth, height, width = array.shape
            min_d, min_h, min_w = min_size or (1, 8, 8)
            
            if depth < min_d or height < min_h or width < min_w:
                raise ValueError(f"3D shape too small: {array.shape}, minimum: ({min_d}, {min_h}, {min_w})")
                
            if depth > 1000 or height > 2048 or width > 2048:
                logger.warning(f"Unusually large 3D shape: {array.shape}")
                
        elif expected_dims == 2:
            height, width = array.shape  
            min_h, min_w = min_size or (8, 8)
            
            if height < min_h or width < min_w:
                raise ValueError(f"2D shape too small: {array.shape}, minimum: ({min_h}, {min_w})")
                
        return True
    
    @staticmethod
    def get_slice_count(array: np.ndarray, plane: str = 'axial') -> int:
        MedicalShapeHandler.validate_medical_shape(array, expected_dims=3)
        depth, height, width = array.shape
        
        if plane.lower() == 'axial':
            return depth 
        elif plane.lower() == 'sagittal':
            return width  
        elif plane.lower() == 'coronal':
            return height  
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    @staticmethod
    def extract_slice(
        array: np.ndarray, 
        plane: str, 
        index: int
    ) -> np.ndarray:
        MedicalShapeHandler.validate_medical_shape(array, expected_dims=3)
        depth, height, width = array.shape
        
        if plane.lower() == 'axial':
            if not 0 <= index < depth:
                raise ValueError(f"Axial slice index {index} out of range [0, {depth-1}]")
            return array[index, :, :]  
            
        elif plane.lower() == 'sagittal':
            if not 0 <= index < width:
                raise ValueError(f"Sagittal slice index {index} out of range [0, {width-1}]")
            return array[:, :, index] 
        
        elif plane.lower() == 'coronal':
            if not 0 <= index < height:
                raise ValueError(f"Coronal slice index {index} out of range [0, {height-1}]")
            return array[:, index, :] 
            
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    @staticmethod
    def get_middle_slice_index(array: np.ndarray, plane: str = 'axial') -> int:
        slice_count = MedicalShapeHandler.get_slice_count(array, plane)
        return slice_count // 2
    
    @staticmethod
    def ensure_3d_shape(array: np.ndarray) -> np.ndarray:
        if len(array.shape) == 2:
            return array[np.newaxis, :, :]  
        elif len(array.shape) == 3:
            MedicalShapeHandler.validate_medical_shape(array, expected_dims=3)
            return array
        else:
            raise ValueError(f"Cannot convert {len(array.shape)}D array to 3D")
    
    @staticmethod
    def get_spatial_dimensions(array: np.ndarray) -> Tuple[int, int, int]:
        if len(array.shape) == 3:
            return array.shape 
        elif len(array.shape) == 2:
            height, width = array.shape
            return (1, height, width) 
        else:
            raise ValueError(f"Cannot get spatial dimensions from {len(array.shape)}D array")
    
    @staticmethod
    def format_shape_info(array: np.ndarray) -> str:
        shape = array.shape
        
        if len(shape) == 3:
            depth, height, width = shape
            return f"3D: {depth}×{height}×{width} (D×H×W)"
        elif len(shape) == 2:
            height, width = shape
            return f"2D: {height}×{width} (H×W)"
        else:
            return f"{len(shape)}D: {shape}"
    
    @staticmethod
    def are_shapes_compatible(
        shape1: Tuple[int, ...], 
        shape2: Tuple[int, ...]
    ) -> bool:
        if len(shape1) != len(shape2):
            return False
            
        return shape1 == shape2
    
    @staticmethod
    def get_dimension_info(array: np.ndarray) -> dict:
        shape = array.shape
        
        info = {
            'shape': shape,
            'ndim': len(shape),
            'formatted': MedicalShapeHandler.format_shape_info(array),
            'total_voxels': int(np.prod(shape)),
            'memory_mb': array.nbytes / (1024 * 1024)
        }
        
        if len(shape) == 3:
            depth, height, width = shape
            info.update({
                'depth': depth,
                'height': height, 
                'width': width,
                'axial_slices': depth,
                'sagittal_slices': width,
                'coronal_slices': height,
                'middle_axial': depth // 2,
                'middle_sagittal': width // 2,
                'middle_coronal': height // 2
            })
        elif len(shape) == 2:
            height, width = shape
            info.update({
                'height': height,
                'width': width
            })
            
        return info


def validate_medical_image_array(array: np.ndarray, context: str = "medical_image") -> np.ndarray:
    if array is None:
        raise ValueError(f"Array is None in context: {context}")
        
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Expected numpy array in context: {context}, got {type(array)}")
    
    if len(array.shape) < 2 or len(array.shape) > 3:
        raise ValueError(f"Medical images must be 2D or 3D, got {len(array.shape)}D in context: {context}")
    
    MedicalShapeHandler.validate_medical_shape(
        array, 
        expected_dims=len(array.shape)
    )
    
    logger.debug(f"Validated {context}: {MedicalShapeHandler.format_shape_info(array)}")
    return array