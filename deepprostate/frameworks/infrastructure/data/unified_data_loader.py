import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

from deepprostate.core.domain.utils.unified_orientation_handler import UnifiedOrientationHandler
from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.core.domain.services.mask_detection_service import MaskDetectionService, MaskDetectionConfig


class UnifiedDataLoader:
    def __init__(self, mask_detection_config: Optional[MaskDetectionConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.orientation_handler = UnifiedOrientationHandler()
        
        self._mask_detection_service = MaskDetectionService(mask_detection_config)
        
    def load_medical_image(
        self,
        image_path: Union[str, Path],
        force_3d: bool = True,
        apply_physical_reorientation: bool = False
    ) -> MedicalImage:
        """
        Load medical image with optional physical reorientation.

        Args:
            image_path: Path to image file
            force_3d: Force 2D images to 3D
            apply_physical_reorientation: If True, physically reorient to RAS
        """
        image_path = Path(image_path)

        if self._is_likely_mask_file(image_path):
            self.logger.info(f"MASK DETECTION: Skipping {image_path.name} - detected as mask/segmentation file")
            from deepprostate.core.domain.exceptions.medical_exceptions import MaskFileDetectedError
            raise MaskFileDetectedError(
                f"File {image_path.name} appears to be a mask/segmentation file, not a medical image",
                file_path=str(image_path)
            )

        try:
            normalized_array, metadata = self.orientation_handler.normalize_to_ras(
                image_path,
                force_3d=force_3d,
                apply_physical_reorientation=apply_physical_reorientation
            )

            from deepprostate.core.domain.entities.medical_image import ImageSpacing, ImageModalityType
            from datetime import datetime
            import hashlib

            file_hash = hashlib.md5(str(image_path.parent).encode()).hexdigest()[:8]
            sequence_type = self._detect_sequence_type(image_path)

            if sequence_type == 'UNKNOWN':
                unique_part = image_path.stem
                default_series_uid = f'series_{file_hash}_{unique_part}'
            else:
                default_series_uid = f'series_{file_hash}_{sequence_type}'


            spacing_tuple = metadata.get('spacing', (1.0, 1.0, 1.0))

            # Autodetect associated masks (IMPORTANT for program functionality)
            associated_masks = self._mask_detection_service.find_associated_masks(image_path)
            if associated_masks:
                self.logger.info(f"Found {len(associated_masks)} associated masks for {image_path.name}: {[p.name for p in associated_masks]}")
            
            enhanced_metadata = {
                **metadata, 
                'detected_sequence_type': sequence_type,
                'associated_masks': [str(mask_path) for mask_path in associated_masks],
                'has_associated_masks': len(associated_masks) > 0,
                'masks_count': len(associated_masks)
            }
            
            medical_image = MedicalImage(
                image_data=normalized_array,
                spacing=ImageSpacing(x=spacing_tuple[0], y=spacing_tuple[1], z=spacing_tuple[2]),
                modality=ImageModalityType.MRI,
                patient_id=metadata.get('patient_id', f'PATIENT_{file_hash}'),
                study_instance_uid=metadata.get('study_uid', f'study_{file_hash}'),
                series_instance_uid=metadata.get('series_uid', default_series_uid),
                acquisition_date=datetime.now(),
                dicom_metadata=enhanced_metadata
            )

            if associated_masks:
                self.logger.info(f"Auto-detected {len(associated_masks)} associated masks for {image_path.name}")

            self.logger.info(f"Loaded unified image: {image_path.name} -> {normalized_array.shape}")
            return medical_image
            
        except Exception as e:
            self.logger.error(f"Error loading medical image: {e}")
            raise
    
    def load_mask(
        self,
        mask_path: Union[str, Path],
        reference_image: Optional[MedicalImage] = None,
        apply_physical_reorientation: bool = False
    ) -> MedicalImage:
        """Load mask with optional physical reorientation."""
        mask_path = Path(mask_path)

        try:
            reference_metadata = None
            if reference_image and hasattr(reference_image, 'metadata'):
                reference_metadata = reference_image.metadata

            normalized_mask, metadata = self.orientation_handler.normalize_mask_to_ras(
                mask_path,
                reference_metadata,
                apply_physical_reorientation=apply_physical_reorientation
            )
            
            from deepprostate.core.domain.entities.medical_image import ImageSpacing, ImageModalityType
            from datetime import datetime
            import hashlib
            
            dir_hash = hashlib.md5(str(mask_path.parent).encode()).hexdigest()[:8]
            mask_hash = hashlib.md5(str(mask_path).encode()).hexdigest()[:8]
            
            spacing_tuple = metadata.get('spacing', (1.0, 1.0, 1.0))
            mask_image = MedicalImage(
                image_data=normalized_mask,
                spacing=ImageSpacing(x=spacing_tuple[0], y=spacing_tuple[1], z=spacing_tuple[2]),
                modality=ImageModalityType.MRI,  # Asumiendo MRI by defecto
                patient_id=metadata.get('patient_id', f'PATIENT_{dir_hash}'),
                study_instance_uid=metadata.get('study_uid', f'study_{dir_hash}'),
                series_instance_uid=metadata.get('series_uid', f'mask_{mask_hash}_{mask_path.stem}'),
                acquisition_date=datetime.now(),
                dicom_metadata=metadata
            )
            
            self.logger.debug(f"Loaded unified mask: {mask_path.name} -> {normalized_mask.shape}")
            return mask_image
            
        except Exception as e:
            self.logger.error(f"Error loading mask: {e}")
            raise
    
    def load_study_series(
        self,
        study_path: Union[str, Path],
        sequence_filter: Optional[List[str]] = None,
        apply_physical_reorientation: bool = False
    ) -> Dict[str, MedicalImage]:
        """Load all series in a study directory with optional physical reorientation."""
        study_path = Path(study_path)
        series_images = {}

        if not study_path.exists() or not study_path.is_dir():
            raise FileNotFoundError(f"Study directory not found: {study_path}")

        image_files = []
        for pattern in ['*.dcm', '*.nii', '*.nii.gz', '*.mha', '*.mhd']:
            image_files.extend(study_path.glob(pattern))

        if not image_files:
            for pattern in ['**/*.dcm', '**/*.nii', '**/*.nii.gz', '**/*.mha', '**/*.mhd']:
                image_files.extend(study_path.glob(pattern))

        self.logger.info(f"Found {len(image_files)} image files in study")

        for image_file in image_files:
            try:
                sequence_type = self._detect_sequence_type(image_file)

                if sequence_filter and sequence_type not in sequence_filter:
                    continue

                medical_image = self.load_medical_image(
                    image_file,
                    apply_physical_reorientation=apply_physical_reorientation
                )
                series_images[sequence_type] = medical_image

            except Exception as e:
                self.logger.warning(f"Skipped {image_file.name}: {e}")
                continue

        self.logger.info(f"Loaded {len(series_images)} series with unified orientation")
        return series_images
    
    def _detect_sequence_type(self, image_path: Path) -> str:
        filename = image_path.name.upper()
        
        if any(pattern in filename for pattern in ['T2W', 'T2-WEIGHTED', 'T2_WEIGHTED']):
            return 'T2W'
        elif any(pattern in filename for pattern in ['ADC', 'APPARENT_DIFFUSION', 'DIFFUSION']):
            return 'ADC'
        elif any(pattern in filename for pattern in ['HBV', 'HIGH_B', 'HIGHB']):
            return 'HBV'
        elif any(pattern in filename for pattern in ['DWI', 'DIFFUSION_WEIGHTED']):
            return 'DWI'
        else:
            stem = image_path.stem.upper()
            
            parts = stem.split('_')
            for part in parts:
                if part in ['T1', 'T1W', 'T1WEIGHTED']:
                    return 'T1W'
                elif part in ['FLAIR', 'T2FLAIR']:
                    return 'FLAIR'
                elif 'CONTRAST' in part or 'GAD' in part:
                    return 'T1C'
                elif part in ['B0', 'B50', 'B100', 'B800', 'B1000']:
                    return f'DWI_{part}'
            
            if len(parts) > 1:
                last_part = parts[-1]
                if last_part and last_part != 'IMAGE' and len(last_part) > 1:
                    return last_part
            
            return 'UNKNOWN'
    
    def _is_likely_mask_file(self, file_path: Path) -> bool:
        filename_lower = file_path.name.lower()
        
        mask_indicators = ['_mask', 'mask_', '_seg', 'seg_', '_label', 'label_', 
                          '_segmentation', 'segmentation_', '_annotation', 'annotation_',
                          '_contour', 'contour_', '_tz', '_pz', '_zones']
        
        strong_match = any(indicator in filename_lower for indicator in mask_indicators)
        if strong_match:
            return True
        
        import re
        mask_patterns = [
            r'.*_\d+_mask\.',    
            r'.*_seg\.',         
            r'.*_label\.',       
            r'.*zones\.',        
        ]
        
        for pattern in mask_patterns:
            if re.match(pattern, filename_lower):
                return True
        
        return False
    
    def get_supported_formats(self) -> List[str]:
        return ['.dcm', '.nii', '.nii.gz', '.mha', '.mhd', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    def validate_image_compatibility(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        
        compatibility_info = {
            'supported': False,
            'format': 'unknown',
            'estimated_shape': None,
            'needs_conversion': False,
            'error': None
        }
        
        try:
            file_format = self.orientation_handler._detect_format(image_path)
            compatibility_info['format'] = file_format
            compatibility_info['supported'] = file_format != 'unknown'
            
            if not compatibility_info['supported']:
                compatibility_info['error'] = f"Unsupported format: {image_path.suffix}"
            
        except Exception as e:
            compatibility_info['error'] = str(e)
        
        return compatibility_info