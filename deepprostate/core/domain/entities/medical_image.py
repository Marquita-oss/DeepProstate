import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..utils.medical_shape_handler import MedicalShapeHandler, validate_medical_image_array


class ImageModalityType(Enum):
    CT = "CT"
    MRI = "MRI"
    ULTRASOUND = "US"
    XRAY = "XR"
    PET = "PT"


class ImagePlaneType(Enum):
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    OBLIQUE = "oblique"


@dataclass
class ImageSpacing:
    x: float
    y: float
    z: float
    
    def get_voxel_volume(self) -> float:
        return self.x * self.y * self.z
    
    def is_isotropic(self, tolerance: float = 0.1) -> bool:
        return (abs(self.x - self.y) < tolerance and 
                abs(self.y - self.z) < tolerance and 
                abs(self.x - self.z) < tolerance)


@dataclass
class WindowLevel:
    window: float
    level: float
    
    def get_display_range(self) -> Tuple[float, float]:
        min_val = self.level - (self.window / 2.0)
        max_val = self.level + (self.window / 2.0)
        return min_val, max_val
    
    def apply_to_array(self, image_array: np.ndarray) -> np.ndarray:
        min_val, max_val = self.get_display_range()
        clipped = np.clip(image_array, min_val, max_val)
        if max_val != min_val:
            normalized = (clipped - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(clipped)
        return normalized


class MedicalImage:
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
        image_id: Optional[str] = None
    ):
        self._validate_image_data(image_data)
        validated_patient_id = self._validate_patient_id(patient_id)
        self._validate_uid(study_instance_uid, "Study Instance UID")
        self._validate_uid(series_instance_uid, "Series Instance UID")
        
        if image_data.flags.owndata and image_data.flags.c_contiguous:
            self._image_data = image_data 
            self._owns_data = False 
        else:
            self._image_data = image_data.copy() 
            self._owns_data = True 
        self._spacing = spacing
        self._modality = modality
        self._patient_id = validated_patient_id
        self._study_instance_uid = study_instance_uid
        self._series_instance_uid = series_instance_uid
        self._acquisition_date = acquisition_date
        self._dicom_metadata = dicom_metadata or {}
        self._image_id = image_id or self._generate_image_id()

        self._default_window_level = None
        self._current_window_level = None
    
    def __del__(self):
        try:
            if hasattr(self, '_owns_data') and self._owns_data and hasattr(self, '_image_data'):
                self._image_data = None
        except:
            pass
    
    @property
    def image_data(self) -> np.ndarray:
        view = self._image_data.view()
        view.flags.writeable = False
        return view
    
    def get_image_data_copy(self) -> np.ndarray:
        return self._image_data.copy()
    
    def _replace_image_data(self, new_image_data: np.ndarray) -> None:
        if hasattr(self, '_owns_data') and self._owns_data:
            self._image_data = None 
        
        if new_image_data.flags.owndata and new_image_data.flags.c_contiguous:
            self._image_data = new_image_data
            self._owns_data = False
        else:
            self._image_data = new_image_data.copy()
            self._owns_data = True
    
    @property
    def original_data_type(self) -> np.dtype:
        return self._image_data.dtype
    
    @property
    def dimensions(self) -> Tuple[int, ...]:
        return self._image_data.shape
    
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
    def acquisition_date(self) -> datetime:
        return self._acquisition_date
    
    @property
    def image_id(self) -> str:
        return self._image_id
    
    @property
    def current_window_level(self) -> WindowLevel:
        if self._current_window_level is None:
            if self._default_window_level is None:
                self._default_window_level = self._get_default_window_level()
            self._current_window_level = self._default_window_level
        return self._current_window_level
    
    def get_slice(self, plane: ImagePlaneType, index: int) -> np.ndarray:
        if plane == ImagePlaneType.AXIAL:
            return MedicalShapeHandler.extract_slice(self._image_data, 'axial', index)
        elif plane == ImagePlaneType.SAGITTAL:
            return MedicalShapeHandler.extract_slice(self._image_data, 'sagittal', index)
        elif plane == ImagePlaneType.CORONAL:
            return MedicalShapeHandler.extract_slice(self._image_data, 'coronal', index)
        else:
            raise ValueError(f"Plano {plane} no soportado for extracción simple")
    
    def get_slice_count(self, plane: ImagePlaneType) -> int:
        if len(self._image_data.shape) < 3:
            return 1
            
        if plane == ImagePlaneType.AXIAL:
            return MedicalShapeHandler.get_slice_count(self._image_data, 'axial')
        elif plane == ImagePlaneType.SAGITTAL:
            return MedicalShapeHandler.get_slice_count(self._image_data, 'sagittal')
        elif plane == ImagePlaneType.CORONAL:
            return MedicalShapeHandler.get_slice_count(self._image_data, 'coronal')
        else:
            return 1
    
    def get_total_slices_all_planes(self) -> Dict[str, int]:
        if len(self._image_data.shape) < 3:
            return {
                'axial': 1,
                'sagittal': 1, 
                'coronal': 1
            }
        
        depth, height, width = self._image_data.shape
        
        return {
            'axial': depth,
            'sagittal': width,
            'coronal': height
        }
    
    def is_3d_volume(self) -> bool:
        return len(self._image_data.shape) >= 3 and any(dim > 1 for dim in self._image_data.shape)
    
    def is_multi_slice_series(self) -> bool:
        if 'is_series' in self._dicom_metadata:
            return self._dicom_metadata['is_series']
        
        if len(self._image_data.shape) >= 3:
            depth, _, _ = self._image_data.shape
            return depth > 1
            
        return False
    
    def get_physical_dimensions(self) -> Tuple[float, float, float]:
        if len(self._image_data.shape) == 3:
            depth, height, width = self._image_data.shape
            return (
                width * self._spacing.x,
                height * self._spacing.y,
                depth * self._spacing.z
            )
        else:
            height, width = self._image_data.shape
            return (width * self._spacing.x, height * self._spacing.y, 0.0)
    
    def get_intensity_statistics(self, use_cache: bool = True) -> Dict[str, float]:
        if use_cache and hasattr(self, '_cached_statistics') and self._cached_statistics is not None:
            return self._cached_statistics

        stats = {
            'min': float(np.min(self._image_data)),
            'max': float(np.max(self._image_data)),
            'mean': float(np.mean(self._image_data)),
            'std': float(np.std(self._image_data)),
            'median': float(np.median(self._image_data))
        }

        self._cached_statistics = stats
        return stats
    
    def set_window_level(self, window: float, level: float) -> None:
        if window <= 0:
            raise ValueError("El ancho of ventana debe ser positivo")
        
        self._current_window_level = WindowLevel(window=window, level=level)
    
    def reset_window_level(self) -> None:
        if self._default_window_level is None:
            self._default_window_level = self._get_default_window_level()
        self._current_window_level = self._default_window_level
    
    def get_dicom_tag(self, tag_name: str) -> Any:
        return self._dicom_metadata.get(tag_name)
    
    def _validate_image_data(self, image_data: np.ndarray) -> None:
        if not isinstance(image_data, np.ndarray):
            raise TypeError("Los datos of image debin ser un numpy array")
        
        if image_data.size == 0:
            raise ValueError("Los datos of image no puedin estar vacíos")
        
        ndim = len(image_data.shape)
        
        if ndim == 1:
            size = image_data.shape[0]
            if size >= 64:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Imagin 1D detectada with {size} elementos - tratando como válida")
            else:
                raise ValueError(f"Imagin 1D demasiado pequeña: {size} elementos")
                
        elif ndim == 2:
            MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=2)
                
        elif ndim == 3:
            MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=3)
                
        elif ndim == 4:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Imagin 4D detectada with shape {image_data.shape} - usando primera componente")
            
        elif ndim >= 5:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Imagin with {ndim} dimensiones detectada - shape: {image_data.shape}")
            
        else:
            raise ValueError(f"Formato of image no soportado: {ndim} dimensiones")
    
    def _validate_patient_id(self, patient_id: str) -> str:
        if not isinstance(patient_id, str):
            patient_id = str(patient_id) if patient_id is not None else ""
        
        if not patient_id.strip():
            import uuid
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generated_id = f"ANONYMOUS_{timestamp}_{str(uuid.uuid4())[:8]}"
            
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Patient ID vacío detectado. Generando ID temporal: {generated_id}")
            
            return generated_id
        
        return patient_id.strip()
    
    def _validate_uid(self, uid: str, uid_type: str) -> None:
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"{uid_type} debe ser una cadena no vacía")
        
        uid_cleaned = uid.strip()
        
        if len(uid_cleaned) < 5 or len(uid_cleaned) > 64:
            raise ValueError(f"{uid_type} tiene longitud inválida: {len(uid_cleaned)}")
        
        invalid_chars = set(' \t\n\r\0')
        if any(c in invalid_chars for c in uid_cleaned):
            raise ValueError(f"{uid_type} contiene caracteres inválidos")
            
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
    
    def _generate_image_id(self) -> str:
        import hashlib
        
        base_string = f"{self._series_instance_uid}_{self._study_instance_uid}_{self._patient_id}"
        
        dims_string = "_".join(str(d) for d in self._image_data.shape)
        base_string += f"_{dims_string}"
        
        hash_obj = hashlib.md5(base_string.encode())
        image_id = f"img_{hash_obj.hexdigest()[:8]}"
        
        return image_id
