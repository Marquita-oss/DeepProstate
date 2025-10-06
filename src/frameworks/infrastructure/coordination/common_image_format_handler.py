import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
from PIL import Image

from .medical_format_handler import (
    MedicalFormatHandler, FormatCapabilities, LoadedImageData
)
from src.core.domain.entities.medical_image import ImageSpacing, ImageModalityType


class CommonImageFormatHandler(MedicalFormatHandler):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        self._default_spacing = ImageSpacing(x=1.0, y=1.0, z=1.0)
        self._default_modality = ImageModalityType.XRAY 
    
    def get_format_name(self) -> str:
        return "CommonImage"
    
    def get_capabilities(self) -> FormatCapabilities:
        return FormatCapabilities(
            can_read=True,
            can_write=True,
            supports_metadata=False,
            supports_series=False,
            supports_3d=False,
            typical_extensions=['.png', '.jpg', '.jpeg', '.tiff', '.tif']
        )
    
    def can_handle_file(self, file_path: Path) -> bool:
        extension = file_path.suffix.lower()
        return extension in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        try:
            with Image.open(file_path) as img:
                if img.size[0] < 1 or img.size[1] < 1:
                    return False, "Invalid image dimensions"
                
                if img.size[0] < 32 or img.size[1] < 32:
                    return False, "Image too small for medical use (minimum 32x32)"
                
                return True, None
                
        except Exception as e:
            return False, f"Image validation error: {e}"
    
    def load_image(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            with Image.open(file_path) as img:
                image_array = self._convert_to_medical_array(img)
                metadata = self._extract_image_metadata(img, file_path)                
                spacing = self._extract_or_estimate_spacing(img, metadata)
                modality = self._guess_modality_from_context(file_path, metadata)
                patient_id = self._generate_patient_id(file_path)
                study_uid = f"image_study_{file_path.stem}"
                series_uid = f"image_series_{file_path.stem}"
                
                return LoadedImageData(
                    image_array=image_array,
                    spacing=spacing,
                    modality=modality,
                    metadata=metadata,
                    patient_id=patient_id,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    acquisition_date=self._get_file_date(file_path)
                )
                
        except Exception as e:
            self._logger.error(f"Failed to load image {file_path}: {e}")
            return None
    
    def _convert_to_medical_array(self, img: Image.Image) -> np.ndarray:
        if img.mode == 'RGB' or img.mode == 'RGBA':
            img_gray = img.convert('L')
            array = np.array(img_gray, dtype=np.float32)
        elif img.mode == 'L':
            array = np.array(img, dtype=np.float32)
        else:
            img_gray = img.convert('L')
            array = np.array(img_gray, dtype=np.float32)
        
        if len(array.shape) == 2:
            return array 
        else:
            self._logger.warning(f"Unexpected array shape: {array.shape}")
            return array.squeeze()  
    
    def _extract_image_metadata(self, img: Image.Image, file_path: Path) -> Dict[str, Any]:
        metadata = {
            'format': img.format or 'Unknown',
            'mode': img.mode,
            'size': img.size,
            'file_path': str(file_path),
            'filename': file_path.name
        }
        
        if hasattr(img, '_getexif') and img._getexif():
            try:
                exif = img._getexif()
                metadata['exif'] = {str(k): str(v) for k, v in exif.items()}
            except:
                pass
        
        if hasattr(img, 'info') and img.info:
            metadata['info'] = {str(k): str(v) for k, v in img.info.items()}
        
        return metadata
    
    def _extract_or_estimate_spacing(self, img: Image.Image, metadata: Dict[str, Any]) -> ImageSpacing:
        try:
            if hasattr(img, 'info') and 'dpi' in img.info:
                dpi_x, dpi_y = img.info['dpi']
                spacing_x = 25.4 / dpi_x if dpi_x > 0 else 1.0
                spacing_y = 25.4 / dpi_y if dpi_y > 0 else 1.0
                return ImageSpacing(x=spacing_x, y=spacing_y, z=1.0)
            
            if img.format == 'TIFF' and hasattr(img, 'tag_v2'):
                resolution_unit = img.tag_v2.get(296, 2)  # 2 = inches, 3 = cm
                x_resolution = img.tag_v2.get(282)
                y_resolution = img.tag_v2.get(283)
                
                if x_resolution and y_resolution:
                    if resolution_unit == 2:  # inches
                        spacing_x = 25.4 / x_resolution
                        spacing_y = 25.4 / y_resolution
                    elif resolution_unit == 3:  # cm
                        spacing_x = 10.0 / x_resolution
                        spacing_y = 10.0 / y_resolution
                    else:
                        spacing_x = spacing_y = 1.0
                    
                    return ImageSpacing(x=spacing_x, y=spacing_y, z=1.0)
        
        except Exception as e:
            self._logger.debug(f"Could not extract spacing: {e}")
        
        return self._default_spacing
    
    def _guess_modality_from_context(self, file_path: Path, metadata: Dict[str, Any]) -> ImageModalityType:
        filename = file_path.name.lower()
        
        if any(keyword in filename for keyword in ['xray', 'x-ray', 'radiograph']):
            return ImageModalityType.XRAY
        elif any(keyword in filename for keyword in ['ultrasound', 'us', 'echo']):
            return ImageModalityType.ULTRASOUND
        elif any(keyword in filename for keyword in ['ct', 'computed']):
            return ImageModalityType.CT
        elif any(keyword in filename for keyword in ['mri', 'magnetic', 't1', 't2']):
            return ImageModalityType.MRI
        else:
            size = metadata.get('size', (0, 0))
            if size[0] > 2000 or size[1] > 2000:
                return ImageModalityType.XRAY  
            else:
                return self._default_modality
    
    def _generate_patient_id(self, file_path: Path) -> str:
        stem = file_path.stem
        
        if 'patient' in stem.lower():
            return stem
        elif any(char.isdigit() for char in stem):
            return f"IMG_PATIENT_{stem}"
        else:
            return f"IMG_PATIENT_{stem}"
    
    def _get_file_date(self, file_path: Path) -> datetime:
        try:
            stat = file_path.stat()
            return datetime.fromtimestamp(stat.st_mtime)
        except:
            return datetime.now()
    
    def save_image(self, medical_image: 'MedicalImage', output_path: Path, 
                   format_type: str = 'PNG') -> bool:
        try:
            image_data = medical_image.image_data
            
            if len(image_data.shape) == 3:
                middle_slice = image_data.shape[0] // 2
                image_data = image_data[middle_slice, :, :]
                self._logger.info(f"Saving middle slice ({middle_slice}) of 3D volume")
            
            if image_data.max() > 255 or image_data.min() < 0:
                image_data = image_data - image_data.min()
                image_data = (image_data / image_data.max() * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
            
            pil_image = Image.fromarray(image_data, mode='L')
            pil_image.save(output_path, format=format_type)
            self._logger.info(f"Saved medical image to {output_path}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save image: {e}")
            return False