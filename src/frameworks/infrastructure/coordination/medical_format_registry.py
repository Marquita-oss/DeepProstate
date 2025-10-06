import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np

from .medical_format_handler import FormatDetectionService, MedicalFormatHandler
from .dicom_format_handler import DICOMFormatHandler
from .nifti_format_handler import NIfTIFormatHandler
from .metaimage_format_handler import MetaImageFormatHandler
from .common_image_format_handler import CommonImageFormatHandler
from src.core.domain.entities.medical_image import MedicalImage, ImageSpacing
from src.frameworks.infrastructure.data.unified_data_loader import UnifiedDataLoader


class MedicalFormatRegistry:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._detection_service = FormatDetectionService()
        
        from src.core.domain.services.mask_detection_service import MaskDetectionConfig
        mask_config = MaskDetectionConfig()
        self._unified_loader = UnifiedDataLoader(mask_config)
        
        self._initialize_handlers()        
        self._logger.info("Medical format registry initialized with unified loader and all handlers")
    
    def _initialize_handlers(self) -> None:
        try:
            dicom_handler = DICOMFormatHandler()
            self._detection_service.register_handler(dicom_handler)
            
            nifti_handler = NIfTIFormatHandler()
            self._detection_service.register_handler(nifti_handler)
            
            metaimage_handler = MetaImageFormatHandler()
            self._detection_service.register_handler(metaimage_handler)
            
            common_handler = CommonImageFormatHandler()
            self._detection_service.register_handler(common_handler)
            
            self._logger.info("All format handlers registered successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize format handlers: {e}")
            raise
    
    def can_load_file(self, file_path: Path) -> bool:
        handler = self._detection_service.detect_format(file_path)
        return handler is not None
    
    def get_file_format(self, file_path: Path) -> Optional[str]:
        handler = self._detection_service.detect_format(file_path)
        return handler.get_format_name() if handler else None
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str], Optional[str]]:
        handler = self._detection_service.detect_format(file_path)
        if not handler:
            return False, "No suitable format handler found", None
        
        is_valid, error_msg = handler.validate_file(file_path)
        return is_valid, error_msg, handler.get_format_name()
    
    def load_medical_image(self, file_path: Path) -> Optional[MedicalImage]:
        try:
            medical_image = self._unified_loader.load_medical_image(file_path, force_3d=True)

            if medical_image:
                self._logger.info(f"Successfully loaded image: {file_path.name} -> {medical_image.image_data.shape}")
            else:
                self._logger.error(f"Failed to load image: {file_path}")

            return medical_image

        except Exception as e:
            from src.core.domain.exceptions.medical_exceptions import MaskFileDetectedError
            if isinstance(e, MaskFileDetectedError):
                self._logger.debug(f"Skipped mask file: {file_path.name} - {e.message}")
                raise e
            else:
                self._logger.error(f"Error loading medical image {file_path}: {e}")
                return None

    def load_medical_image_by_series_uid(self, series_uid: str) -> Optional[MedicalImage]:
        try:
            self._logger.info(f"Loading image by series UID: {series_uid}")
            self._logger.warning(f"Series UID loading not fully implemented: {series_uid}")
            return None

        except Exception as e:
            self._logger.error(f"Error in load_medical_image_by_series_uid {series_uid}: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        return self._detection_service.get_all_supported_extensions()
    
    def get_format_capabilities(self) -> Dict[str, Dict[str, Any]]:
        capabilities = self._detection_service.get_format_capabilities()
        return {
            format_name: {
                'can_read': caps.can_read,
                'can_write': caps.can_write,
                'supports_metadata': caps.supports_metadata,
                'supports_series': caps.supports_series,
                'supports_3d': caps.supports_3d,
                'extensions': caps.typical_extensions
            }
            for format_name, caps in capabilities.items()
        }
    
    def get_format_handler(self, format_name: str) -> Optional[MedicalFormatHandler]:
        for handler in self._detection_service._handlers:
            if handler.get_format_name() == format_name:
                return handler
        return None
    
    def save_medical_image(self, medical_image: MedicalImage, output_path: Path, 
                          format_name: str = 'PNG') -> bool:
        try:
            handler = self.get_format_handler('CommonImage') 
            
            if not handler or not handler.get_capabilities().can_write:
                self._logger.error(f"Format {format_name} does not support writing")
                return False
            
            if hasattr(handler, 'save_image'):
                return handler.save_image(medical_image, output_path, format_name)
            else:
                self._logger.error(f"Handler {handler.get_format_name()} does not support saving")
                return False
                
        except Exception as e:
            self._logger.error(f"Error saving medical image: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        capabilities = self.get_format_capabilities()
        
        return {
            'total_formats': len(capabilities),
            'readable_formats': sum(1 for caps in capabilities.values() if caps['can_read']),
            'writable_formats': sum(1 for caps in capabilities.values() if caps['can_write']),
            'total_extensions': len(self.get_supported_extensions()),
            'formats': list(capabilities.keys()),
            'extensions': self.get_supported_extensions()
        }
    
    def scan_folder(self, folder_path: Path) -> Dict[str, List[Path]]:
        format_groups = defaultdict(list)
        
        if not folder_path.exists() or not folder_path.is_dir():
            return dict(format_groups)
        
        # Scan all files in folder
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                format_name = self.get_file_format(file_path)
                if format_name:
                    format_groups[format_name].append(file_path)
        
        return dict(format_groups)
    
    def load_folder_as_series(self, folder_path: Path) -> Optional[MedicalImage]:
        try:
            format_groups = self.scan_folder(folder_path)
            
            if not format_groups:
                self._logger.warning(f"No supported medical image files found in {folder_path}")
                return None
            
            for format_name, files in format_groups.items():
                self._logger.info(f"Found {len(files)} {format_name} files")
            
            format_priority = ['DICOM', 'NIfTI', 'MetaImage']
            
            selected_format = None
            selected_files = None
            
            for format_name in format_priority:
                if format_name in format_groups:
                    selected_format = format_name
                    selected_files = format_groups[format_name]
                    break
            
            if not selected_format:
                selected_format = next(iter(format_groups.keys()))
                selected_files = format_groups[selected_format]
            
            self._logger.info(f"Loading {len(selected_files)} {selected_format} files as series")
            
            if selected_format == 'DICOM':
                return self._load_dicom_series(selected_files)
            else:
                return self._load_non_dicom_series(selected_files, selected_format)
                
        except Exception as e:
            self._logger.error(f"Error loading folder as series {folder_path}: {e}")
            return None
    
    def _load_dicom_series(self, dicom_files: List[Path]) -> Optional[MedicalImage]:
        try:
            if len(dicom_files) == 1:
                return self.load_medical_image(dicom_files[0])
            
            self._logger.info(f"Loading {len(dicom_files)} DICOM files as series using DICOMImageRepository")
            
            try:
                import SimpleITK as sitk
                from src.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository
                
                sorted_files = self._sort_dicom_files_by_location(dicom_files)
                
                reader = sitk.ImageSeriesReader()
                dicom_file_names = [str(f) for f in sorted_files]
                reader.SetFileNames(dicom_file_names)
                
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()
                
                try:
                    sitk_image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(sitk_image)
                    
                    if len(image_array.shape) == 4 and image_array.shape[1] == 1:
                        image_array = np.squeeze(image_array, axis=1)
                        self._logger.debug(f"Squeezed singleton dimension: {image_array.shape}")
                    
                    spacing_sitk = sitk_image.GetSpacing()
                    spacing = ImageSpacing(
                        x=float(spacing_sitk[0]),
                        y=float(spacing_sitk[1]),
                        z=float(spacing_sitk[2]) if len(spacing_sitk) > 2 else 1.0
                    )
                    
                    self._logger.debug(f"Series loaded with SimpleITK - Shape: {image_array.shape}, Spacing: x={spacing.x:.3f}, y={spacing.y:.3f}, z={spacing.z:.3f}")
                    
                except Exception as sitk_series_error:
                    self._logger.warning(f"SimpleITK series reader failed: {sitk_series_error}")
                    self._logger.info("Attempting individual slice loading and reconstruction...")
                    
                    slices = []
                    first_spacing = None
                    
                    for file_path in sorted_files:
                        try:
                            slice_image = sitk.ReadImage(str(file_path))
                            slice_array = sitk.GetArrayFromImage(slice_image)
                            
                            if len(slice_array.shape) == 3 and slice_array.shape[0] == 1:
                                slice_array = slice_array[0]
                            elif len(slice_array.shape) == 4 and slice_array.shape[0] == 1 and slice_array.shape[1] == 1:
                                slice_array = slice_array[0, 0]
                            
                            slices.append(slice_array)
                            
                            if first_spacing is None:
                                slice_spacing = slice_image.GetSpacing()
                                first_spacing = ImageSpacing(
                                    x=float(slice_spacing[0]),
                                    y=float(slice_spacing[1]),
                                    z=1.0 
                                )
                                
                        except Exception as slice_error:
                            self._logger.error(f"Failed to load slice {file_path}: {slice_error}")
                            continue
                    
                    if not slices:
                        raise ValueError("No slices could be loaded")
                    
                    image_array = np.stack(slices, axis=0)                    
                    z_spacing = self._calculate_z_spacing_from_files(sorted_files)
                    spacing = ImageSpacing(
                        x=first_spacing.x,
                        y=first_spacing.y,
                        z=z_spacing
                    )
                    
                    self._logger.debug(f"Manual reconstruction completed - Shape: {image_array.shape}, Spacing: x={spacing.x:.3f}, y={spacing.y:.3f}, z={spacing.z:.3f}")
                
                import pydicom
                first_ds = pydicom.dcmread(str(sorted_files[0]))
                
                temp_repo = DICOMImageRepository("/tmp")
                modality = temp_repo._parse_modality(getattr(first_ds, 'Modality', 'CT'))
                acquisition_date = temp_repo._parse_dicom_date(
                    getattr(first_ds, 'AcquisitionDate', None),
                    getattr(first_ds, 'AcquisitionTime', None)
                )                
                medical_image = MedicalImage(
                    image_data=image_array,
                    spacing=spacing,
                    modality=modality,
                    patient_id=getattr(first_ds, 'PatientID', 'UNKNOWN'),
                    study_instance_uid=getattr(first_ds, 'StudyInstanceUID', ''),
                    series_instance_uid=getattr(first_ds, 'SeriesInstanceUID', ''),
                    acquisition_date=acquisition_date,
                    dicom_metadata=temp_repo._extract_dicom_metadata(first_ds)
                )
                
                return medical_image
                
            except Exception as series_error:
                self._logger.error(f"SimpleITK series loading failed: {series_error}")
                dicom_handler = self.get_format_handler('DICOM')
                if dicom_handler:
                    return self._reconstruct_dicom_volume(dicom_files, dicom_handler)
                return None
            
        except Exception as e:
            self._logger.error(f"Error loading DICOM series: {e}")
            return None
    
    def _reconstruct_dicom_volume(self, dicom_files: List[Path], dicom_handler) -> Optional[MedicalImage]:
        try:
            dicom_slices = []
            first_image = None
            
            for i, file_path in enumerate(dicom_files):
                image = self.load_medical_image(file_path)
                if image is None:
                    self._logger.error(f"Failed to load DICOM file {i+1}/{len(dicom_files)}: {file_path.name}")
                    continue
                
                if first_image is None:
                    first_image = image
                
                # Store slice with metadata for sorting
                dicom_slices.append({
                    'image': image,
                    'data': image.image_data,
                    'file_path': file_path,
                    'slice_location': self._extract_slice_location(file_path)
                })
                
                self._logger.debug(f"Loaded DICOM slice {i+1}/{len(dicom_files)}: {image.image_data.shape}")
            
            if not dicom_slices or first_image is None:
                self._logger.error("No DICOM slices were successfully loaded")
                return None
            
            try:
                dicom_slices.sort(key=lambda s: s['slice_location'])
                self._logger.info(f"Sorted {len(dicom_slices)} DICOM slices by slice location")
            except Exception as e:
                self._logger.warning(f"Could not sort slices by location: {e}, using fallback file order")
            
            try:
                base_shape = dicom_slices[0]['data'].shape
                compatible_slices = []
                
                for slice_data in dicom_slices:
                    if slice_data['data'].shape == base_shape:
                        compatible_slices.append(slice_data['data'])
                    else:
                        self._logger.warning(f"Slice has incompatible shape {slice_data['data'].shape}, expected {base_shape}")
                
                if len(compatible_slices) > 1:
                    volume_data = np.stack(compatible_slices, axis=0)
                    self._logger.info(f"Created 3D DICOM volume with shape: {volume_data.shape}")
                    
                    from src.core.domain.entities.enhanced_medical_image import EnhancedMedicalImage
                    
                    enhanced_image = EnhancedMedicalImage(
                        image_data=volume_data,
                        spacing=first_image.spacing,
                        modality=first_image.modality,
                        patient_id=first_image.patient_id,
                        study_instance_uid=first_image.study_instance_uid,
                        series_instance_uid=first_image.series_instance_uid,
                        acquisition_date=first_image.acquisition_date,
                        dicom_metadata={
                            **getattr(first_image, 'dicom_metadata', {}),
                            'series_files': [f.name for f in dicom_files],
                            'dicom_series': True,
                            'slice_count': len(compatible_slices),
                            'reconstructed_volume': True
                        },
                        auto_analyze_dimensions=True
                    )
                    
                    return enhanced_image
                else:
                    self._logger.warning("Only one compatible slice found, returning as 2D image")
                    return first_image
                    
            except Exception as e:
                self._logger.error(f"Failed to stack DICOM slices into volume: {e}")
                return first_image
            
        except Exception as e:
            self._logger.error(f"Error reconstructing DICOM volume: {e}")
            return None
    
    def _extract_slice_location(self, dicom_file: Path) -> float:
        try:
            import pydicom
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
            
            if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
                return float(ds.SliceLocation)
            elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                return float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'InstanceNumber') and ds.InstanceNumber:
                return float(ds.InstanceNumber)
            else:
                import re
                numbers = re.findall(r'\d+', dicom_file.stem)
                return float(numbers[-1]) if numbers else 0.0
                
        except Exception as e:
            self._logger.debug(f"Could not extract slice location from {dicom_file.name}: {e}")
            import re
            numbers = re.findall(r'\d+', dicom_file.stem)
            return float(numbers[-1]) if numbers else 0.0
    
    def _load_non_dicom_series(self, files: List[Path], format_name: str) -> Optional[MedicalImage]:
        try:
            sorted_files = sorted(files, key=lambda f: f.name)
            
            if len(sorted_files) == 1:
                return self.load_medical_image(sorted_files[0])
            else:
                self._logger.info(f"Attempting to reconstruct {len(sorted_files)} {format_name} files as series")
                
                if self._detect_temporal_series(sorted_files):
                    return self._reconstruct_temporal_series(sorted_files, format_name)
                else:
                    primary_image = self.load_medical_image(sorted_files[0])
                    if primary_image:
                        primary_image = self._enhance_with_series_info(primary_image, sorted_files, format_name)
                    return primary_image
                
        except Exception as e:
            self._logger.error(f"Error loading {format_name} series: {e}")
            return None
    
    def _detect_temporal_series(self, files: List[Path]) -> bool:
        filenames = [f.stem.lower() for f in files]
        
        time_indicators = ['t1', 't2', 't3', 'time', 'frame', '_001', '_002', 'vol_0', 'vol_1']
        has_time_indicators = any(
            any(indicator in filename for indicator in time_indicators)
            for filename in filenames
        )
        
        has_sequential = self._has_sequential_numbering(filenames)
        
        return has_time_indicators or has_sequential
    
    def _has_sequential_numbering(self, filenames: List[str]) -> bool:
        import re
        numbers = []
        
        for filename in filenames:
            nums = re.findall(r'\d+', filename)
            if nums:
                numbers.append(int(nums[-1])) 
        
        if len(numbers) < 2:
            return False
        
        numbers.sort()
        return all(numbers[i] + 1 == numbers[i + 1] for i in range(len(numbers) - 1))
    
    def _reconstruct_temporal_series(self, files: List[Path], format_name: str) -> Optional[MedicalImage]:
        try:
            self._logger.info(f"Reconstructing {len(files)} files as temporal {format_name} series")
            
            # Load all files
            volumes = []
            first_image = None
            
            for i, file_path in enumerate(files):
                image = self.load_medical_image(file_path)
                if image is None:
                    self._logger.error(f"Failed to load file {i+1}/{len(files)}: {file_path.name}")
                    continue
                
                if first_image is None:
                    first_image = image
                
                volumes.append(image.image_data)
                self._logger.debug(f"Loaded volume {i+1}/{len(files)}: {image.image_data.shape}")
            
            if not volumes or first_image is None:
                self._logger.error("No volumes were successfully loaded")
                return None
            
            if len(volumes) > 1:
                try:
                    base_shape = volumes[0].shape
                    compatible_volumes = []
                    
                    for i, vol in enumerate(volumes):
                        if vol.shape == base_shape:
                            compatible_volumes.append(vol)
                        else:
                            self._logger.warning(f"Volume {i} has incompatible shape {vol.shape}, expected {base_shape}")
                    
                    if len(compatible_volumes) > 1:
                        temporal_data = np.stack(compatible_volumes, axis=-1)
                        self._logger.info(f"Created temporal series with shape: {temporal_data.shape}")
                        
                        from src.core.domain.entities.enhanced_medical_image import EnhancedMedicalImage
                        
                        enhanced_image = EnhancedMedicalImage(
                            image_data=temporal_data,
                            spacing=first_image.spacing,
                            modality=first_image.modality,
                            patient_id=first_image.patient_id,
                            study_instance_uid=first_image.study_instance_uid,
                            series_instance_uid=first_image.series_instance_uid,
                            acquisition_date=first_image.acquisition_date,
                            dicom_metadata={
                                **getattr(first_image, 'dicom_metadata', {}),
                                'series_files': [f.name for f in files],
                                'temporal_series': True,
                                'temporal_frames': len(compatible_volumes)
                            },
                            auto_analyze_dimensions=True
                        )
                        
                        return enhanced_image
                    
                except Exception as e:
                        self._logger.error(f"Failed to stack volumes as temporal series: {e}")
            
            return self._enhance_with_series_info(first_image, files, format_name)
            
        except Exception as e:
            self._logger.error(f"Error reconstructing temporal series: {e}")
            return None
    
    def _sort_dicom_files_by_location(self, dicom_files: List[Path]) -> List[Path]:
        try:
            files_with_location = []
            for file_path in dicom_files:
                location = self._extract_slice_location(file_path)
                files_with_location.append((file_path, location))
            
            # Sort by location
            files_with_location.sort(key=lambda x: x[1])
            return [f[0] for f in files_with_location]
            
        except Exception as e:
            self._logger.warning(f"Failed to sort DICOM files by location: {e}, using fallback file order")
            return dicom_files
    
    def _calculate_z_spacing_from_files(self, sorted_files: List[Path]) -> float:
        try:
            if len(sorted_files) < 2:
                return 1.0
            
            positions = []
            for file_path in sorted_files[:10]: 
                location = self._extract_slice_location(file_path)
                if location is not None:
                    positions.append(location)
            
            if len(positions) < 2:
                import pydicom
                first_ds = pydicom.dcmread(str(sorted_files[0]), stop_before_pixels=True)
                if hasattr(first_ds, 'SliceThickness') and first_ds.SliceThickness:
                    return float(first_ds.SliceThickness)
                elif hasattr(first_ds, 'SpacingBetweenSlices') and first_ds.SpacingBetweenSlices:
                    return float(first_ds.SpacingBetweenSlices)
                return 1.0
            
            spacings = []
            for i in range(1, len(positions)):
                spacing = abs(positions[i] - positions[i-1])
                if spacing > 0:
                    spacings.append(spacing)
            
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                self._logger.debug(f"Calculated Z spacing: {avg_spacing:.3f} from {len(spacings)} intervals")
                return avg_spacing
            
            return 1.0
            
        except Exception as e:
            self._logger.warning(f"Failed to calculate Z spacing: {e}, using fallback value 1.0")
            return 1.0

    def _enhance_with_series_info(self, primary_image: MedicalImage, all_files: List[Path], format_name: str) -> MedicalImage:
        try:
            if hasattr(primary_image, 'dicom_metadata'):
                metadata = primary_image.dicom_metadata.copy()
            else:
                metadata = {}
            
            metadata.update({
                'series_files': [f.name for f in all_files],
                'series_format': format_name,
                'series_count': len(all_files),
                'primary_file': all_files[0].name,
                'available_files': [f.name for f in all_files[1:]]
            })
            
            if hasattr(primary_image, '_dicom_metadata'):
                primary_image._dicom_metadata = metadata
            
            self._logger.info(f"Enhanced image with series info: {len(all_files)} files available")
            return primary_image
            
        except Exception as e:
            self._logger.error(f"Error inhancing image with series info: {e}")
            return primary_image


_global_registry: Optional[MedicalFormatRegistry] = None


def get_medical_format_registry() -> MedicalFormatRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = MedicalFormatRegistry()
    return _global_registry


def load_medical_image_from_file(file_path: Path) -> Optional[MedicalImage]:
    registry = get_medical_format_registry()
    return registry.load_medical_image(file_path)