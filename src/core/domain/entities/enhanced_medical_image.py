import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..utils.medical_shape_handler import MedicalShapeHandler, validate_medical_image_array

from .medical_dimension_analyzer import (
    MedicalDimensionAnalyzer, MedicalImageStructure, DimensionType, ImageOrientation
)
from .medical_image import ImageSpacing, WindowLevel, ImageModalityType


class EnhancedImagePlaneType(Enum):
    AXIAL = "axial"         
    SAGITTAL = "sagittal"   
    CORONAL = "coronal"    
    OBLIQUE = "oblique"    


class SliceExtractionMode(Enum):
    SPATIAL_ONLY = "spatial"     
    TEMPORAL_FRAME = "temporal"  
    CHANNEL_SPECIFIC = "channel"
    AVERAGE_TEMPORAL = "avg_temp" 
    AVERAGE_CHANNEL = "avg_chan"  

@dataclass
class SliceInfo:
    plane: EnhancedImagePlaneType
    index: int
    temporal_index: Optional[int] = None
    channel_index: Optional[int] = None
    extraction_mode: SliceExtractionMode = SliceExtractionMode.SPATIAL_ONLY
    physical_position: Optional[float] = None 


class EnhancedMedicalImage:
    def __init__(
        self,
        image_data: np.ndarray,
        spacing: ImageSpacing,
        modality: ImageModalityType,
        patient_id: str,
        study_instance_uid: str,
        series_instance_uid: str,
        acquisition_date: datetime,
        dicom_metadata: Optional[Dict[str, Any]] = None,
        image_id: Optional[str] = None,
        auto_analyze_dimensions: bool = True
    ):
        self._validate_image_data(image_data)        
        self._image_data = image_data.copy() if image_data.flags.writeable else image_data
        self._spacing = spacing
        self._modality = modality
        self._patient_id = patient_id
        self._study_instance_uid = study_instance_uid
        self._series_instance_uid = series_instance_uid
        self._acquisition_date = acquisition_date
        self._dicom_metadata = dicom_metadata or {}
        self._image_id = image_id or self._generate_image_id()
        
        self._dimension_analyzer = MedicalDimensionAnalyzer()
        
        if auto_analyze_dimensions:
            self._analyze_image_structure()
        else:
            self._structure = None
        
        self._default_window_level = self._get_default_window_level()
        self._current_window_level = self._default_window_level
    
    def _analyze_image_structure(self) -> None:
        try:
            spacing_tuple = None
            if self._spacing:
                spacing_tuple = (self._spacing.x, self._spacing.y, self._spacing.z)
            
            self._structure = self._dimension_analyzer.analyze_image_structure(
                image_array=self._image_data,
                spacing=spacing_tuple,
                modality=self._modality.value if self._modality else None,
                metadata=self._dicom_metadata
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to analyze image structure: {e}")
            self._structure = None
    
    @property
    def image_data(self) -> np.ndarray:
        return self._image_data
    
    @property
    def structure(self) -> Optional[MedicalImageStructure]:
        return self._structure
    
    @property
    def dimensions(self) -> Tuple[int, ...]:
        return self._image_data.shape
    
    @property
    def spatial_dimensions(self) -> Tuple[int, int, int]:
        if not self._structure:
            return MedicalShapeHandler.get_spatial_dimensions(self._image_data)
        
        z_size = self._get_dimension_size(DimensionType.SPATIAL_Z)  
        y_size = self._get_dimension_size(DimensionType.SPATIAL_Y)
        x_size = self._get_dimension_size(DimensionType.SPATIAL_X)  
        
        return (z_size, y_size, x_size)  
    
    @property
    def temporal_size(self) -> int:
        if not self._structure or not self._structure.temporal_dimension:
            return 1
        return self._structure.temporal_dimension.size
    
    @property
    def is_4d(self) -> bool:
        return self._structure and self._structure.is_4d if self._structure else False
    
    @property
    def is_temporal(self) -> bool:
        return self._structure and self._structure.is_temporal if self._structure else False
    
    @property
    def is_multi_channel(self) -> bool:
        return self._structure and self._structure.is_multi_channel if self._structure else False
    
    def get_spatial_slice(
        self,
        plane: EnhancedImagePlaneType,
        index: int,
        temporal_index: int = 0,
        channel_index: int = 0,
        extraction_mode: SliceExtractionMode = SliceExtractionMode.SPATIAL_ONLY
    ) -> Tuple[np.ndarray, SliceInfo]:
        if not self._structure:
            return self._basic_slice_extraction(plane, index)
        
        slice_array = self._extract_slice_with_structure(
            plane, index, temporal_index, channel_index, extraction_mode
        )
        
        slice_info = SliceInfo(
            plane=plane,
            index=index,
            temporal_index=temporal_index if self.is_4d else None,
            channel_index=channel_index if self.is_multi_channel else None,
            extraction_mode=extraction_mode,
            physical_position=self._calculate_physical_position(plane, index)
        )
        
        return slice_array, slice_info
    
    def get_temporal_series(
        self, 
        plane: EnhancedImagePlaneType, 
        spatial_index: int,
        channel_index: int = 0
    ) -> np.ndarray:
        if not self.is_temporal:
            raise ValueError("Image does not have temporal dimension")
        
        temporal_series = []
        for t in range(self.temporal_size):
            slice_array, _ = self.get_spatial_slice(
                plane, spatial_index, temporal_index=t, channel_index=channel_index
            )
            temporal_series.append(slice_array)
        
        return np.stack(temporal_series, axis=0)
    
    def get_channel_comparison(
        self, 
        plane: EnhancedImagePlaneType, 
        spatial_index: int,
        temporal_index: int = 0
    ) -> np.ndarray:
        if not self.is_multi_channel:
            raise ValueError("Image does not have multiple channels")
        
        channel_data = []
        num_channels = self._get_dimension_size(DimensionType.CHANNEL)
        
        for c in range(num_channels):
            slice_array, _ = self.get_spatial_slice(
                plane, spatial_index, temporal_index=temporal_index, channel_index=c
            )
            channel_data.append(slice_array)
        
        return np.stack(channel_data, axis=0)
    
    def get_volume_at_timepoint(self, temporal_index: int = 0, channel_index: int = 0) -> np.ndarray:
        if not self._structure:
            return self._image_data
        
        actual_dims = len(self._image_data.shape)
        
        if actual_dims == 4 and self.is_4d and self._structure.t_index is not None:
            indexing = [slice(None)] * len(self._image_data.shape)
            
            if self.is_temporal:
                # Validate temporal index
                if temporal_index >= self.temporal_size:
                    raise IndexError(f"Temporal index {temporal_index} out of range (0-{self.temporal_size-1})")
                indexing[self._structure.t_index] = temporal_index
            else:  # multi-channel
                # Validate channel index
                channel_size = self._get_dimension_size(DimensionType.CHANNEL)
                if channel_index >= channel_size:
                    raise IndexError(f"Channel index {channel_index} out of range (0-{channel_size-1})")
                indexing[self._structure.t_index] = channel_index
            
            volume = self._image_data[tuple(indexing)]
        elif actual_dims >= 3:
            volume = self._image_data
        else:
            raise ValueError(f"Cannot extract volume from {actual_dims}D data")
        
        return self._reorient_to_medical_convention(volume)
    
    def get_slice_count(self, plane) -> int:
        if hasattr(plane, 'value'):
            plane_str = plane.value
        elif isinstance(plane, str):
            plane_str = plane
        else:
            plane_str = str(plane).split('.')[-1].lower()
        
        if plane_str == 'axial':
            enhanced_plane = EnhancedImagePlaneType.AXIAL
        elif plane_str == 'sagittal':
            enhanced_plane = EnhancedImagePlaneType.SAGITTAL
        elif plane_str == 'coronal':
            enhanced_plane = EnhancedImagePlaneType.CORONAL
        else:
            return 1
        
        if not self._structure:
            if len(self._image_data.shape) >= 3:
                if enhanced_plane == EnhancedImagePlaneType.AXIAL:
                    return self._image_data.shape[0]
                elif enhanced_plane == EnhancedImagePlaneType.SAGITTAL:
                    return self._image_data.shape[2]
                elif enhanced_plane == EnhancedImagePlaneType.CORONAL:
                    return self._image_data.shape[1]
            return 1
        
        if enhanced_plane == EnhancedImagePlaneType.AXIAL:
            return self._get_dimension_size(DimensionType.SPATIAL_Z)
        elif enhanced_plane == EnhancedImagePlaneType.SAGITTAL:
            return self._get_dimension_size(DimensionType.SPATIAL_X)
        elif enhanced_plane == EnhancedImagePlaneType.CORONAL:
            return self._get_dimension_size(DimensionType.SPATIAL_Y)
        else:
            return 1
    
    def is_multi_slice_series(self) -> bool:
        if len(self._image_data.shape) >= 3:
            return (self._image_data.shape[0] > 1 or 
                   self._image_data.shape[1] > 1 or 
                   self._image_data.shape[2] > 1)
        return False
    
    def get_total_slices_all_planes(self) -> Dict[str, int]:
        from .medical_image import ImagePlaneType
        return {
            "axial": self.get_slice_count(ImagePlaneType.AXIAL),
            "sagittal": self.get_slice_count(ImagePlaneType.SAGITTAL), 
            "coronal": self.get_slice_count(ImagePlaneType.CORONAL)
        }
    
    def _extract_slice_with_structure(
        self, 
        plane: EnhancedImagePlaneType, 
        index: int,
        temporal_index: int,
        channel_index: int,
        extraction_mode: SliceExtractionMode
    ) -> np.ndarray:        
        indexing = [slice(None)] * len(self._image_data.shape)
        
        if self._structure.t_index is not None:
            if extraction_mode == SliceExtractionMode.TEMPORAL_FRAME:
                indexing[self._structure.t_index] = temporal_index
            elif extraction_mode == SliceExtractionMode.CHANNEL_SPECIFIC:
                indexing[self._structure.t_index] = channel_index
            elif extraction_mode == SliceExtractionMode.AVERAGE_TEMPORAL:
                pass
            elif extraction_mode == SliceExtractionMode.AVERAGE_CHANNEL:
                pass
            else:  
                indexing[self._structure.t_index] = 0 
        
        if plane == EnhancedImagePlaneType.AXIAL and self._structure.z_index is not None:
            indexing[self._structure.z_index] = index
        elif plane == EnhancedImagePlaneType.SAGITTAL and self._structure.x_index is not None:
            indexing[self._structure.x_index] = index
        elif plane == EnhancedImagePlaneType.CORONAL and self._structure.y_index is not None:
            indexing[self._structure.y_index] = index
        
        if extraction_mode in [SliceExtractionMode.AVERAGE_TEMPORAL, SliceExtractionMode.AVERAGE_CHANNEL]:
            if self._structure.t_index is not None:
                indexing[self._structure.t_index] = slice(None)
            
            slice_data = self._image_data[tuple(indexing)]
            
            if len(slice_data.shape) > 2:
                axis = 0 if self._structure.t_index == 0 else -1
                slice_data = np.mean(slice_data, axis=axis)
        else:
            slice_data = self._image_data[tuple(indexing)]
        
        return slice_data
    
    def _basic_slice_extraction(self, plane: EnhancedImagePlaneType, index: int) -> Tuple[np.ndarray, SliceInfo]:
        if len(self._image_data.shape) < 3:
            raise ValueError("Slice extraction requires 3D or higher dimensional data")
        
        depth, height, width = self._image_data.shape[:3]
        
        if plane == EnhancedImagePlaneType.AXIAL:
            slice_array = self._image_data[index, :, :]
        elif plane == EnhancedImagePlaneType.SAGITTAL:
            slice_array = self._image_data[:, :, index]
        elif plane == EnhancedImagePlaneType.CORONAL:
            slice_array = self._image_data[:, index, :]
        else:
            raise ValueError(f"Unsupported plane: {plane}")
        
        slice_info = SliceInfo(plane=plane, index=index)
        return slice_array, slice_info
    
    def _get_dimension_size(self, dim_type: DimensionType) -> int:
        if not self._structure:
            return 1
        
        for dim in self._structure.dimensions:
            if dim.type == dim_type:
                return dim.size
        return 1
    
    def _calculate_physical_position(self, plane: EnhancedImagePlaneType, index: int) -> Optional[float]:
        if not self._structure:
            return None
        
        if plane == EnhancedImagePlaneType.AXIAL:
            spacing = self._spacing.z
        elif plane == EnhancedImagePlaneType.SAGITTAL:
            spacing = self._spacing.x
        elif plane == EnhancedImagePlaneType.CORONAL:
            spacing = self._spacing.y
        else:
            return None
        
        return index * spacing
    
    def _reorient_to_medical_convention(self, volume: np.ndarray) -> np.ndarray:
        if not self._structure or len(volume.shape) != 3:
            return volume
        
        spatial_dims = self._structure.spatial_dimensions
        if len(spatial_dims) != 3:
            return volume
        
        volume_shape_mapping = {}
        volume_axis = 0
        
        for dim in spatial_dims:
            volume_shape_mapping[dim.type] = volume_axis
            volume_axis += 1
        
        target_indices = [None, None, None]
        
        if DimensionType.SPATIAL_Z in volume_shape_mapping:
            target_indices[0] = volume_shape_mapping[DimensionType.SPATIAL_Z]
        if DimensionType.SPATIAL_Y in volume_shape_mapping:
            target_indices[1] = volume_shape_mapping[DimensionType.SPATIAL_Y]
        if DimensionType.SPATIAL_X in volume_shape_mapping:
            target_indices[2] = volume_shape_mapping[DimensionType.SPATIAL_X]
        
        if target_indices == [0, 1, 2]:
            return volume
        
        if all(idx is not None for idx in target_indices):
            return np.transpose(volume, target_indices)
        
        return volume
    
    def _validate_image_data(self, image_data: np.ndarray) -> None:
        if not isinstance(image_data, np.ndarray):
            raise TypeError("Image data must be a numpy array")
        
        if image_data.size == 0:
            raise ValueError("Image data cannot be empty")
        
        try:
            if len(image_data.shape) == 2:
                MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=2)
            elif len(image_data.shape) == 3:
                MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=3)
            elif len(image_data.shape) == 4:
                MedicalShapeHandler.validate_medical_shape(image_data[0], expected_dims=3)
            else:
                raise ValueError(f"Unsupported image dimensions: {len(image_data.shape)}D")
        except ValueError as e:
            raise ValueError(f"Invalid medical image shape: {e}")
    
    def _generate_image_id(self) -> str:
        import hashlib
        base_string = f"{self._series_instance_uid}_{self._study_instance_uid}_{self._patient_id}"
        dims_string = "_".join(str(d) for d in self._image_data.shape)
        base_string += f"_{dims_string}"
        hash_obj = hashlib.md5(base_string.encode())
        return f"enhanced_img_{hash_obj.hexdigest()[:8]}"
    
    def _get_default_window_level(self) -> WindowLevel:
        if self._modality == ImageModalityType.CT:
            min_intensity = np.percentile(self._image_data, 2.0)
            max_intensity = np.percentile(self._image_data, 98.0)
            
            if (max_intensity - min_intensity) < 50:
                return WindowLevel(window=400, level=40)
            
            window = max_intensity - min_intensity
            level = (max_intensity + min_intensity) / 2.0
            return WindowLevel(window=window, level=level)
        
        elif self._modality == ImageModalityType.MRI:
            min_intensity = np.percentile(self._image_data, 5.0)
            max_intensity = np.percentile(self._image_data, 95.0)
            
            window = max_intensity - min_intensity
            level = (max_intensity + min_intensity) / 2.0
            
            if window < 1.0:
                window = float(np.std(self._image_data) * 3.0)
                level = float(np.mean(self._image_data))
            
            return WindowLevel(window=window, level=level)
        
        else:
            min_intensity = np.percentile(self._image_data, 5.0)
            max_intensity = np.percentile(self._image_data, 95.0)
            
            window = max_intensity - min_intensity
            level = (max_intensity + min_intensity) / 2.0
            
            if window < 1.0:
                window = float(np.std(self._image_data) * 3.0)
                level = float(np.mean(self._image_data))
            
            return WindowLevel(window=window, level=level)
    
    @property
    def spacing(self) -> ImageSpacing:
        return self._spacing
    
    @property
    def modality(self) -> ImageModalityType:
        return self._modality
    
    @property
    def patient_id(self) -> str:
        return self._patient_id
    
    @property
    def study_instance_uid(self) -> str:
        return self._study_instance_uid
    
    @property
    def series_instance_uid(self) -> str:
        return self._series_instance_uid
    
    @property
    def acquisition_date(self):
        return self._acquisition_date
    
    @property
    def image_id(self) -> str:
        return self._image_id
    
    @property
    def current_window_level(self) -> WindowLevel:
        return self._current_window_level