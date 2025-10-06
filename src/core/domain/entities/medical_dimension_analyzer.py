import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

class DimensionType(Enum):
    SPATIAL_X = "spatial_x"      
    SPATIAL_Y = "spatial_y"     
    SPATIAL_Z = "spatial_z"    
    TEMPORAL = "temporal"     
    CHANNEL = "channel"     
    UNKNOWN = "unknown"         

class ImageOrientation(Enum):
    AXIAL = "axial"          
    SAGITTAL = "sagittal"    
    CORONAL = "coronal"      
    OBLIQUE = "oblique"      

@dataclass
class DimensionInfo:
    index: int                 
    type: DimensionType        
    size: int                  
    spacing: float              
    label: str                 
    is_spatial: bool = False   
    is_temporal: bool = False  

@dataclass
class MedicalImageStructure:
    dimensions: List[DimensionInfo]
    spatial_dimensions: List[DimensionInfo] 
    temporal_dimension: Optional[DimensionInfo]  
    total_dims: int
    is_2d: bool
    is_3d: bool 
    is_4d: bool
    is_temporal: bool
    is_multi_channel: bool
    
    x_index: Optional[int] = None  
    y_index: Optional[int] = None
    z_index: Optional[int] = None
    t_index: Optional[int] = None  

    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0) 
    temporal_spacing: Optional[float] = None


class MedicalDimensionAnalyzer:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def analyze_image_structure(
        self, 
        image_array: np.ndarray, 
        spacing: Optional[Tuple[float, ...]] = None,
        modality: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MedicalImageStructure:
        self._logger.debug(f"Analyzing image structure: shape={image_array.shape}")
        
        shape = image_array.shape
        ndims = len(shape)
        
        structure = self._initialize_structure(shape, spacing, modality, metadata)
        
        if ndims == 2:
            structure = self._analyze_2d_structure(structure, shape, spacing)
        elif ndims == 3:
            structure = self._analyze_3d_structure(structure, shape, spacing, modality, metadata)
        elif ndims == 4:
            structure = self._analyze_4d_structure(structure, shape, spacing, modality, metadata)
        elif ndims >= 5:
            structure = self._analyze_nd_structure(structure, shape, spacing, modality, metadata)
        else:
            raise ValueError(f"Cannot analyze {ndims}D medical image")
        
        structure = self._finalize_structure(structure)
        
        self._logger.info(f"Medical image structure: {self._structure_summary(structure)}")
        return structure
    
    def _initialize_structure(
        self, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> MedicalImageStructure:
        ndims = len(shape)
        
        return MedicalImageStructure(
            dimensions=[],
            spatial_dimensions=[],
            temporal_dimension=None,
            total_dims=ndims,
            is_2d=(ndims == 2),
            is_3d=(ndims == 3),
            is_4d=(ndims == 4),
            is_temporal=False,
            is_multi_channel=False
        )
    
    def _analyze_2d_structure(
        self, 
        structure: MedicalImageStructure, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]]
    ) -> MedicalImageStructure:
        height, width = shape
        
        x_spacing = spacing[0] if spacing and len(spacing) > 0 else 1.0
        y_spacing = spacing[1] if spacing and len(spacing) > 1 else 1.0
        
        y_dim = DimensionInfo(
            index=0, type=DimensionType.SPATIAL_Y, size=height,
            spacing=y_spacing, label="Height (Y)", is_spatial=True
        )
        x_dim = DimensionInfo(
            index=1, type=DimensionType.SPATIAL_X, size=width,
            spacing=x_spacing, label="Width (X)", is_spatial=True
        )
        
        structure.dimensions = [y_dim, x_dim]
        structure.spatial_dimensions = [y_dim, x_dim]
        structure.y_index = 0
        structure.x_index = 1
        structure.voxel_size = (x_spacing, y_spacing, 1.0)
        
        return structure
    
    def _analyze_3d_structure(
        self, 
        structure: MedicalImageStructure, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> MedicalImageStructure:        
        dim_order = self._detect_3d_dimension_order(shape, spacing, modality, metadata)
        dimensions = []

        for i, (dim_type, label) in enumerate(dim_order):
            dim_spacing = spacing[i] if spacing and len(spacing) > i else 1.0
            
            dim_info = DimensionInfo(
                index=i,
                type=dim_type,
                size=shape[i],
                spacing=dim_spacing,
                label=label,
                is_spatial=(dim_type in [DimensionType.SPATIAL_X, DimensionType.SPATIAL_Y, DimensionType.SPATIAL_Z])
            )
            dimensions.append(dim_info)
        
        structure.dimensions = dimensions
        structure.spatial_dimensions = [d for d in dimensions if d.is_spatial]
        
        for dim in dimensions:
            if dim.type == DimensionType.SPATIAL_X:
                structure.x_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Y:
                structure.y_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Z:
                structure.z_index = dim.index
        
        x_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_X), 1.0)
        y_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_Y), 1.0)
        z_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_Z), 1.0)
        structure.voxel_size = (x_spacing, y_spacing, z_spacing)
        
        return structure
    
    def _analyze_4d_structure(
        self, 
        structure: MedicalImageStructure, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> MedicalImageStructure:        
        dim_order = self._detect_4d_dimension_order(shape, spacing, modality, metadata)
        dimensions = []

        for i, (dim_type, label) in enumerate(dim_order):
            dim_spacing = spacing[i] if spacing and len(spacing) > i else 1.0
            
            dim_info = DimensionInfo(
                index=i,
                type=dim_type,
                size=shape[i],
                spacing=dim_spacing,
                label=label,
                is_spatial=(dim_type in [DimensionType.SPATIAL_X, DimensionType.SPATIAL_Y, DimensionType.SPATIAL_Z]),
                is_temporal=(dim_type in [DimensionType.TEMPORAL, DimensionType.CHANNEL])
            )
            dimensions.append(dim_info)
        
        structure.dimensions = dimensions
        structure.spatial_dimensions = [d for d in dimensions if d.is_spatial]
        structure.temporal_dimension = next((d for d in dimensions if d.is_temporal), None)
        
        structure.is_temporal = any(d.type == DimensionType.TEMPORAL for d in dimensions)
        structure.is_multi_channel = any(d.type == DimensionType.CHANNEL for d in dimensions)
        
        for dim in dimensions:
            if dim.type == DimensionType.SPATIAL_X:
                structure.x_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Y:
                structure.y_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Z:
                structure.z_index = dim.index
            elif dim.type in [DimensionType.TEMPORAL, DimensionType.CHANNEL]:
                structure.t_index = dim.index
        
        x_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_X), 1.0)
        y_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_Y), 1.0)
        z_spacing = next((d.spacing for d in dimensions if d.type == DimensionType.SPATIAL_Z), 1.0)
        structure.voxel_size = (x_spacing, y_spacing, z_spacing)
        
        if structure.temporal_dimension:
            structure.temporal_spacing = structure.temporal_dimension.spacing
        
        return structure
    
    def _analyze_nd_structure(
        self, 
        structure: MedicalImageStructure, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> MedicalImageStructure:        
        dimensions = []
        spatial_count = 0
        
        for i, size in enumerate(shape):
            dim_spacing = spacing[i] if spacing and len(spacing) > i else 1.0
            
            if spatial_count < 3:
                if spatial_count == 0:
                    dim_type = DimensionType.SPATIAL_Z 
                    label = "Depth (Z)"
                elif spatial_count == 1:
                    dim_type = DimensionType.SPATIAL_Y
                    label = "Height (Y)"
                else:
                    dim_type = DimensionType.SPATIAL_X
                    label = "Width (X)"
                spatial_count += 1
                is_spatial = True
            else:
                dim_type = DimensionType.CHANNEL
                label = f"Channel {i-2}"
                is_spatial = False
            
            dim_info = DimensionInfo(
                index=i, type=dim_type, size=size,
                spacing=dim_spacing, label=label, is_spatial=is_spatial
            )
            dimensions.append(dim_info)
        
        structure.dimensions = dimensions
        structure.spatial_dimensions = [d for d in dimensions if d.is_spatial]
        structure.is_multi_channel = True
        
        for dim in dimensions:
            if dim.type == DimensionType.SPATIAL_X:
                structure.x_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Y:
                structure.y_index = dim.index
            elif dim.type == DimensionType.SPATIAL_Z:
                structure.z_index = dim.index
        
        return structure
    
    def _detect_3d_dimension_order(
        self, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[DimensionType, str]]:
        if metadata:
            if 'format' in metadata and metadata['format'] == 'DICOM':
                return self._detect_dicom_3d_order(shape, spacing, metadata)
            
            elif 'format' in metadata and metadata['format'] == 'NIfTI':
                return self._detect_nifti_3d_order(shape, spacing, metadata)
        
        return self._detect_heuristic_3d_order(shape, spacing, modality)
    
    def _detect_4d_dimension_order(
        self, 
        shape: Tuple[int, ...], 
        spacing: Optional[Tuple[float, ...]], 
        modality: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[DimensionType, str]]:        
        temporal_index, is_temporal = self._identify_temporal_dimension(shape, metadata)
        
        if temporal_index == 0:
            dim_type = DimensionType.TEMPORAL if is_temporal else DimensionType.CHANNEL
            label = "Time" if is_temporal else "Channel"
            return [
                (dim_type, label),
                (DimensionType.SPATIAL_Z, "Depth (Z)"),
                (DimensionType.SPATIAL_Y, "Height (Y)"),
                (DimensionType.SPATIAL_X, "Width (X)")
            ]
        elif temporal_index == 3:
            dim_type = DimensionType.TEMPORAL if is_temporal else DimensionType.CHANNEL
            label = "Time" if is_temporal else "Channel"
            return [
                (DimensionType.SPATIAL_Z, "Depth (Z)"),
                (DimensionType.SPATIAL_Y, "Height (Y)"),
                (DimensionType.SPATIAL_X, "Width (X)"),
                (dim_type, label)
            ]
        else:
            return [
                (DimensionType.SPATIAL_Z, "Depth (Z)"),
                (DimensionType.SPATIAL_Y, "Height (Y)"),
                (DimensionType.SPATIAL_X, "Width (X)"),
                (DimensionType.CHANNEL, "Channel")
            ]
    
    def _detect_dicom_3d_order(self, shape, spacing, metadata) -> List[Tuple[DimensionType, str]]:
        return [
            (DimensionType.SPATIAL_Z, "Depth (Z) - Slices"),
            (DimensionType.SPATIAL_Y, "Height (Y) - Rows"), 
            (DimensionType.SPATIAL_X, "Width (X) - Columns")
        ]
    
    def _detect_nifti_3d_order(self, shape, spacing, metadata) -> List[Tuple[DimensionType, str]]:
        return [
            (DimensionType.SPATIAL_Z, "Depth (Z)"),
            (DimensionType.SPATIAL_Y, "Height (Y)"),
            (DimensionType.SPATIAL_X, "Width (X)")
        ]
    
    def _detect_heuristic_3d_order(self, shape, spacing, modality) -> List[Tuple[DimensionType, str]]:        
        sizes = list(shape)
        sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
        
        if sizes[sorted_indices[0]] < min(sizes[sorted_indices[1]], sizes[sorted_indices[2]]) * 0.5:
            z_index = sorted_indices[0]
            remaining = [i for i in range(len(sizes)) if i != z_index]
            
            order = [None, None, None]
            order[z_index] = (DimensionType.SPATIAL_Z, "Depth (Z)")
            order[remaining[0]] = (DimensionType.SPATIAL_Y, "Height (Y)")
            order[remaining[1]] = (DimensionType.SPATIAL_X, "Width (X)")
            
            return order
        
        return [
            (DimensionType.SPATIAL_Z, "Depth (Z)"),
            (DimensionType.SPATIAL_Y, "Height (Y)"),
            (DimensionType.SPATIAL_X, "Width (X)")
        ]
    
    def _identify_temporal_dimension(self, shape, metadata) -> Tuple[int, bool]:
        if metadata:
            temporal_hints = ['temporal', 'time', 'dynamic', 'fmri', 'bold', 'sequence']
            channel_hints = ['channel', 'multi_parametric', 'T1', 'T2', 'FLAIR', 'DWI']
            
            metadata_str = str(metadata).lower()
            has_temporal_hint = any(hint in metadata_str for hint in temporal_hints)
            has_channel_hint = any(hint in metadata_str for hint in channel_hints)
            
            if has_temporal_hint and not has_channel_hint:
                index = 0 if shape[0] < shape[-1] else 3
                return index, True
            elif has_channel_hint and not has_temporal_hint:
                return 3, False
        
        candidates = []
        
        for i, size in enumerate(shape):
            if i < 3: 
                spatial_sizes = [s for j, s in enumerate(shape[:3]) if j != i]
                if size > 16 and any(abs(size - s) < size * 0.5 for s in spatial_sizes):
                    continue 
            
            candidates.append((i, size))
        
        if not candidates:
            return len(shape) - 1, False
        
        candidates.sort(key=lambda x: (x[1], x[0]))
        
        best_index, best_size = candidates[0]
        if best_size >= 10:
            is_temporal = True
        elif best_size <= 8:
            is_temporal = False
        else:
            is_temporal = (best_index == 0)
        
        return best_index, is_temporal
    
    def _finalize_structure(self, structure: MedicalImageStructure) -> MedicalImageStructure:        
        if structure.total_dims >= 3:
            spatial_types = [d.type for d in structure.spatial_dimensions]
            required_spatial = {DimensionType.SPATIAL_X, DimensionType.SPATIAL_Y, DimensionType.SPATIAL_Z}
            
            if not required_spatial.issubset(set(spatial_types)):
                self._logger.warning("Missing required spatial dimensions - using fallback assignment")
        
        return structure
    
    def _structure_summary(self, structure: MedicalImageStructure) -> str:
        dim_summary = []
        for dim in structure.dimensions:
            dim_summary.append(f"{dim.label}:{dim.size}")
        
        summary = f"{structure.total_dims}D image ({', '.join(dim_summary)})"
        
        if structure.is_temporal:
            summary += " [TEMPORAL]"
        if structure.is_multi_channel:
            summary += " [MULTI-CHANNEL]"
            
        return summary