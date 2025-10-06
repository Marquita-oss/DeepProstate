import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

from src.core.domain.entities.medical_image import (
    MedicalImage, ImageSpacing, WindowLevel, 
    ImageModalityType, ImagePlaneType
)
from src.core.domain.entities.segmentation import MedicalSegmentation
from src.core.domain.repositories.repositories import (
    MedicalImageRepository, SegmentationRepository, ProjectRepository,
    ImageNotFoundError, RepositoryError
)

class ImageLoadingService:    
    def __init__(self, image_repository: MedicalImageRepository) -> None:
        self._image_repository = image_repository
        self._supported_modalities = {
            ImageModalityType.CT,
            ImageModalityType.MRI,
            ImageModalityType.ULTRASOUND
        }
    
    def load_image_by_series_uid_sync(self, series_uid: str) -> Optional[MedicalImage]:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                image = loop.run_until_complete(
                    self._image_repository.find_by_series_uid(series_uid)
                )
                
                if image is None:
                    return None
                
                loop.run_until_complete(self._validate_image_integrity(image))
                
                return image
                
            finally:
                loop.close()
            
        except RepositoryError as e:
            raise ImageLoadingError(f"Error accediendo al repositorio: {e}") from e
        except Exception as e:
            raise ImageLoadingError(f"Error inesperado cargando imagen: {e}") from e
    
    async def _load_series_as_volume(self, file_paths: List[Path], base_image: MedicalImage) -> Optional[MedicalImage]:
        try:
            import SimpleITK as sitk
            import numpy as np
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(path) for path in file_paths])
            
            sitk_image = reader.Execute()
            volume_array = sitk.GetArrayFromImage(sitk_image)
            
            volume_image = MedicalImage(
                image_data=volume_array,
                spacing=base_image.spacing,
                modality=base_image.modality,
                patient_id=base_image.patient_id,
                study_instance_uid=base_image.study_instance_uid,
                series_instance_uid=base_image.series_instance_uid,
                acquisition_date=base_image.acquisition_date,
                dicom_metadata=base_image._dicom_metadata.copy(),
                image_id=f"{base_image.image_id}_series"
            )
            
            volume_image._dicom_metadata['original_shape'] = volume_array.shape
            volume_image._dicom_metadata['is_3d_volume'] = True
            volume_image._dicom_metadata['slice_count'] = len(file_paths)
            
            return volume_image
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error cargando serie como volume 3D: {e}. Usando primer archivo.")
            return base_image
    
    async def load_image_by_series_uid(self, series_uid: str) -> Optional[MedicalImage]:
        try:
            image = await self._image_repository.find_by_series_uid(series_uid)
            
            if image is None:
                return None
            
            await self._validate_image_integrity(image)
            
            return image
            
        except RepositoryError as e:
            raise ImageLoadingError(f"Error accediendo al repositorio: {e}") from e
        except Exception as e:
            raise ImageLoadingError(f"Error inesperado cargando imagen: {e}") from e
    
    async def load_study_images(self, study_uid: str) -> List[MedicalImage]:
        try:
            images = await self._image_repository.find_by_study_uid(study_uid)
            
            if not images:
                raise ImageLoadingError(f"No se encontraron imágenes for the estudio {study_uid}")
            
            validation_tasks = [self._validate_image_integrity(img) for img in images]
            await asyncio.gather(*validation_tasks)
            
            sorted_images = self._sort_images_by_series(images)
            
            return sorted_images
            
        except RepositoryError as e:
            raise ImageLoadingError(f"Error accediendo al repositorio: {e}") from e
        except Exception as e:
            raise ImageLoadingError(f"Error inesperado cargando estudio: {e}") from e
    
    async def load_patient_images(
        self, 
        patient_id: str,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[MedicalImage]:
        try:
            if date_range:
                start_date, end_date = date_range
                images = await self._image_repository.find_by_date_range(start_date, end_date)
                images = [img for img in images if img.patient_id == patient_id]
            else:
                images = await self._image_repository.find_by_patient_id(patient_id)
            
            sorted_images = sorted(images, key=lambda img: img.acquisition_date)
            
            return sorted_images
            
        except RepositoryError as e:
            raise ImageLoadingError(f"Error accediendo al repositorio: {e}") from e
    
    async def validate_modality_support(self, image: MedicalImage) -> bool:
        return image.modality in self._supported_modalities
    
    async def _validate_image_integrity(self, image: MedicalImage) -> None:
        dims = image.dimensions
        if len(dims) == 3:
            depth, height, width = dims
            if depth < 1 or height < 16 or width < 16:
                raise ImageValidationError(
                    f"Dimensiones of image demasiado pequeñas: {dims}. "
                    f"Se requiere al menos 1x16x16 for volúmenes 3D"
                )
        elif len(dims) == 2:
            height, width = dims
            if height < 16 or width < 16:
                raise ImageValidationError(
                    f"Dimensiones of image 2D demasiado pequeñas: {dims}. "
                    f"Se requiere al menos 16x16"
                )
        elif len(dims) == 1:
            if dims[0] < 64:
                raise ImageValidationError(
                    f"Array 1D demasiado pequeño: {dims[0]} elementos. "
                    f"Se requierin al menos 64 elementos"
                )
        
        spacing = image.spacing
        if spacing.x <= 0 or spacing.y <= 0 or spacing.z <= 0:
            raise ImageValidationError(
                f"Espaciado of image inválido: {spacing}"
            )
        
        if not await self.validate_modality_support(image):
            raise ImageValidationError(
                f"Modalidad {image.modality} no soportada"
            )
        
        stats = image.get_intensity_statistics()
        if stats['min'] == stats['max']:
            raise ImageValidationError("Imagin with intensidad uniforme detectada")
    
    def _sort_images_by_series(self, images: List[MedicalImage]) -> List[MedicalImage]:
        def get_series_number(image: MedicalImage) -> int:
            series_number = image.get_dicom_tag("SeriesNumber")
            return int(series_number) if series_number else 0
        
        return sorted(images, key=get_series_number)


class ImageVisualizationService:
    def __init__(self):
        self._window_level_presets = {
            ImageModalityType.CT: {
                "soft_tissue": WindowLevel(window=400, level=40),
                "bone": WindowLevel(window=1500, level=300),
                "lung": WindowLevel(window=1600, level=-600),
                "brain": WindowLevel(window=100, level=40),
                "liver": WindowLevel(window=150, level=60)
            },
            ImageModalityType.MRI: {
                "t1": WindowLevel(window=600, level=300),
                "t2": WindowLevel(window=1000, level=500),
                "flair": WindowLevel(window=800, level=400),
                "dwi": WindowLevel(window=1200, level=600)
            }
        }
    
    async def prepare_slice_for_display(
        self,
        image: MedicalImage,
        plane: ImagePlaneType,
        slice_index: int,
        window_level: Optional[WindowLevel] = None
    ) -> Dict[str, Any]:
        try:
            slice_data = image.get_slice(plane, slice_index)            
            wl = window_level or image.current_window_level            
            normalized_slice = wl.apply_to_array(slice_data)            
            spatial_info = self._calculate_slice_spatial_info(image, plane, slice_index)
            
            slice_metadata = {
                "plane": plane.value,
                "slice_index": slice_index,
                "total_slices": self._get_total_slices(image, plane),
                "window_level": {"window": wl.window, "level": wl.level},
                "spatial_info": spatial_info,
                "intensity_range": wl.get_display_range()
            }
            
            return {
                "image_data": normalized_slice,
                "metadata": slice_metadata,
                "original_data": slice_data 
            }
            
        except Exception as e:
            raise ImageVisualizationError(f"Error preparando corte for visualización: {e}")
    
    async def prepare_volume_for_3d(
        self,
        image: MedicalImage,
        downsample_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            volume_data = image.image_data
            
            if downsample_factor and downsample_factor > 1:
                volume_data = self._downsample_volume(volume_data, downsample_factor)
            
            wl = image.current_window_level
            normalized_volume = wl.apply_to_array(volume_data)
            
            spacing = image.spacing
            if downsample_factor:
                spacing = ImageSpacing(
                    x=spacing.x * downsample_factor,
                    y=spacing.y * downsample_factor,
                    z=spacing.z * downsample_factor
                )
            
            volume_metadata = {
                "original_dimensions": image.dimensions,
                "current_dimensions": volume_data.shape,
                "spacing": {
                    "x": spacing.x,
                    "y": spacing.y,
                    "z": spacing.z
                },
                "downsample_factor": downsample_factor or 1,
                "window_level": {"window": wl.window, "level": wl.level},
                "modality": image.modality.value,
                "intensity_statistics": image.get_intensity_statistics()
            }
            
            return {
                "volume_data": normalized_volume,
                "metadata": volume_metadata,
                "original_spacing": image.spacing
            }
            
        except Exception as e:
            raise ImageVisualizationError(f"Error preparando volume for 3D: {e}")
    
    async def apply_window_level_preset(
        self,
        image: MedicalImage,
        preset_name: str
    ) -> WindowLevel:
        modality_presets = self._window_level_presets.get(image.modality)
        
        if not modality_presets:
            raise PresetNotFoundError(
                f"No hay presets disponibles for modalidad {image.modality}"
            )
        
        if preset_name not in modality_presets:
            available_presets = list(modality_presets.keys())
            raise PresetNotFoundError(
                f"Preset '{preset_name}' no encontrado. "
                f"Presets disponibles: {available_presets}"
            )
        
        preset_wl = modality_presets[preset_name]
        image.set_window_level(preset_wl.window, preset_wl.level)
        
        return preset_wl
    
    async def get_available_presets(self, modality: ImageModalityType) -> List[str]:
        presets = self._window_level_presets.get(modality, {})
        return list(presets.keys())
    
    async def calculate_optimal_window_level(
        self,
        image: MedicalImage,
        percentile_range: Tuple[float, float] = (5.0, 95.0)
    ) -> WindowLevel:
        image_data = image.image_data
        
        low_percentile, high_percentile = percentile_range
        min_intensity = np.percentile(image_data, low_percentile)
        max_intensity = np.percentile(image_data, high_percentile)
        
        window = max_intensity - min_intensity
        level = (max_intensity + min_intensity) / 2.0
        
        if window < 1.0:
            window = np.std(image_data) * 3.0
        
        return WindowLevel(window=window, level=level)
    
    def _calculate_slice_spatial_info(
        self,
        image: MedicalImage,
        plane: ImagePlaneType,
        slice_index: int
    ) -> Dict[str, Any]:
        spacing = image.spacing
        dimensions = image.dimensions
        
        if plane == ImagePlaneType.AXIAL:
            pixel_spacing = (spacing.x, spacing.y)
            slice_thickness = spacing.z
            slice_dimensions = (dimensions[2], dimensions[1])  
            physical_position = slice_index * spacing.z
            
        elif plane == ImagePlaneType.SAGITTAL:
            pixel_spacing = (spacing.y, spacing.z)
            slice_thickness = spacing.x
            slice_dimensions = (dimensions[1], dimensions[0])  
            physical_position = slice_index * spacing.x
            
        elif plane == ImagePlaneType.CORONAL:
            pixel_spacing = (spacing.x, spacing.z)
            slice_thickness = spacing.y
            slice_dimensions = (dimensions[2], dimensions[0])  
            physical_position = slice_index * spacing.y
            
        else:
            raise ValueError(f"Plano {plane} no soportado for cálculo espacial")
        
        return {
            "pixel_spacing_mm": pixel_spacing,
            "slice_thickness_mm": slice_thickness,
            "slice_dimensions": slice_dimensions,
            "physical_position_mm": physical_position,
            "plane_normal": plane.value
        }
    
    def _get_total_slices(self, image: MedicalImage, plane: ImagePlaneType) -> int:
        dimensions = image.dimensions
        
        if plane == ImagePlaneType.AXIAL:
            return dimensions[0]  # depth
        elif plane == ImagePlaneType.SAGITTAL:
            return dimensions[2]  # width
        elif plane == ImagePlaneType.CORONAL:
            return dimensions[1]  # height
        else:
            raise ValueError(f"Plano {plane} no soportado")
    
    def _downsample_volume(
        self,
        volume: np.ndarray,
        factor: int
    ) -> np.ndarray:
        if len(volume.shape) != 3:
            raise ValueError("El submuestreo requiere un volume 3D")
        
        return volume[::factor, ::factor, ::factor]


class ImageLoadingError(Exception):
    pass


class ImageValidationError(Exception):
    pass


class ImageVisualizationError(Exception):
    pass


class PresetNotFoundError(Exception):
    pass