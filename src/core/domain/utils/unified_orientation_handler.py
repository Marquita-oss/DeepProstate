import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    sitk = None

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


class UnifiedOrientationHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._supported_formats = {
            '.dcm': 'dicom',
            '.nii': 'nifti', 
            '.nii.gz': 'nifti',
            '.mha': 'itk',
            '.mhd': 'itk',
            '.png': 'image_2d',
            '.jpg': 'image_2d', 
            '.jpeg': 'image_2d',
            '.tif': 'image_2d',
            '.tiff': 'image_2d'
        }
        
    def normalize_to_ras(
        self,
        image_path: Union[str, Path],
        force_3d: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        file_format = self._detect_format(image_path)
        self.logger.info(f"Normalizing {file_format} image to RAS: {image_path.name}")
        
        if file_format in ['dicom', 'nifti', 'itk']:
            return self._normalize_medical_image(image_path, file_format, force_3d)
        elif file_format == 'image_2d':
            return self._normalize_2d_image(image_path, force_3d)
        else:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
    
    def _detect_format(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        if suffix == '.gz' and image_path.stem.endswith('.nii'):
            suffix = '.nii.gz'
        return self._supported_formats.get(suffix, 'unknown')
    
    def _normalize_medical_image(
        self,
        image_path: Path,
        file_format: str,
        force_3d: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not HAS_SITK:
            raise ImportError("SimpleITK required for medical image formats")
            
        try:
            sitk_image = sitk.ReadImage(str(image_path))
            sitk_image = self._ensure_ras_orientation(sitk_image)
            numpy_array = sitk.GetArrayFromImage(sitk_image)
            numpy_array = self._ensure_3d_shape(numpy_array, force_3d)
            metadata = self._extract_metadata(sitk_image, file_format)
            
            self.logger.info(f"Normalized to RAS: {numpy_array.shape}, spacing: {metadata.get('spacing', 'unknown')}")
            
            return numpy_array, metadata
            
        except Exception as e:
            self.logger.error(f"Error normalizing medical image: {e}")
            raise
    
    def _ensure_ras_orientation(self, sitk_image):
        try:
            current_direction = sitk_image.GetDirection()
            reorienter = sitk.DICOMOrientImageFilter()
            reorienter.SetDesiredCoordinateOrientation("RAS")
            ras_image = reorienter.Execute(sitk_image)
            self.logger.debug(f"Reoriented to RAS from direction: {current_direction}")
            return ras_image
            
        except Exception as e:
            self.logger.info(f"Could not reorient to RAS, using original: {e}")
            return sitk_image
    
    def _normalize_2d_image(
        self,
        image_path: Path,
        force_3d: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for 2D image formats")
            
        try:
            pil_image = Image.open(image_path)
            numpy_array = np.array(pil_image)

            if len(numpy_array.shape) == 3:
                numpy_array = np.mean(numpy_array, axis=2).astype(numpy_array.dtype)
                self.logger.info("Converted color image to grayscale")
            
            numpy_array = self._ensure_3d_shape(numpy_array, force_3d)
            
            metadata = {
                'spacing': (1.0, 1.0, 1.0),
                'origin': (0.0, 0.0, 0.0),
                'direction': np.eye(3).flatten().tolist(),
                'format': 'image_2d',
                'original_shape': numpy_array.shape
            }
            
            self.logger.info(f"Normalized 2D image: {numpy_array.shape}")
            return numpy_array, metadata
            
        except Exception as e:
            self.logger.error(f"Error normalizing 2D image: {e}")
            raise
    
    def _ensure_3d_shape(self, array: np.ndarray, force_3d: bool) -> np.ndarray:
        if len(array.shape) == 2 and force_3d:
            array = array[np.newaxis, :, :]
            self.logger.debug("Converted 2D to 3D (single slice)")
            
        elif len(array.shape) == 3:
            pass  
            
        elif len(array.shape) > 3:
            array = array[0] if array.shape[0] == 1 else array
            self.logger.warning(f"Reduced {len(array.shape)}D to 3D")
            
        return array
    
    def _extract_metadata(self, sitk_image, file_format: str) -> Dict[str, Any]:
        try:
            metadata = {
                'spacing': sitk_image.GetSpacing(),
                'origin': sitk_image.GetOrigin(),
                'direction': sitk_image.GetDirection(),
                'size': sitk_image.GetSize(),
                'format': file_format,
                'pixel_type': sitk_image.GetPixelIDTypeAsString(),
                'dimension': sitk_image.GetDimension()
            }
            
            if file_format == 'dicom':
                metadata.update(self._extract_dicom_metadata(sitk_image))
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Could not extract full metadata: {e}")
            return {
                'spacing': (1.0, 1.0, 1.0),
                'origin': (0.0, 0.0, 0.0),
                'direction': np.eye(3).flatten().tolist(),
                'format': file_format
            }
    
    def _extract_dicom_metadata(self, sitk_image) -> Dict[str, Any]:
        dicom_metadata = {}
        
        important_tags = {
            '0008|0060': 'Modality',
            '0018|0050': 'SliceThickness', 
            '0020|0032': 'ImagePositionPatient',
            '0020|0037': 'ImageOrientationPatient',
            '0008|103e': 'SeriesDescription',
            '0018|0015': 'BodyPartExamined'
        }
        
        for tag, name in important_tags.items():
            try:
                if sitk_image.HasMetaDataKey(tag):
                    dicom_metadata[name] = sitk_image.GetMetaData(tag)
            except:
                continue
                
        return {'dicom_metadata': dicom_metadata}
    
    def normalize_mask_to_ras(
        self,
        mask_path: Union[str, Path],
        reference_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        mask_array, mask_metadata = self.normalize_to_ras(mask_path, force_3d=True)
        
        if mask_array.dtype not in [np.uint8, np.uint16]:
            mask_array = mask_array.astype(np.uint8)
            self.logger.debug("Converted mask to uint8")
        
        unique_values = np.unique(mask_array)
        self.logger.debug(f"Mask unique values: {unique_values}")
        
        if reference_metadata:
            mask_array = self._align_mask_with_reference(mask_array, mask_metadata, reference_metadata)
        
        return mask_array, mask_metadata
    
    def _align_mask_with_reference(
        self,
        mask_array: np.ndarray,
        mask_metadata: Dict[str, Any],
        reference_metadata: Dict[str, Any]
    ) -> np.ndarray:
        ref_spacing = reference_metadata.get('spacing')
        mask_spacing = mask_metadata.get('spacing')
        
        if ref_spacing and mask_spacing and ref_spacing != mask_spacing:
            self.logger.info("Spacing differs, might need resampling (not implemented yet)")

        return mask_array
    
    @staticmethod
    def get_standard_shape_info() -> Dict[str, str]:
        return {
            'orientation': 'RAS (Right-Anterior-Superior)',
            'shape_format': '(Z, Y, X) = (Slices, Height, Width)',
            'coordinate_system': 'Medical standard',
            'z_axis': 'Inferior to Superior',
            'y_axis': 'Posterior to Anterior', 
            'x_axis': 'Left to Right'
        }