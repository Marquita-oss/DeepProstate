import logging
import gzip
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.core.domain.utils.medical_shape_handler import MedicalShapeHandler
from datetime import datetime

import numpy as np

from .medical_format_handler import (
    MedicalFormatHandler, FormatCapabilities, LoadedImageData
)
from src.core.domain.entities.medical_image import ImageSpacing, ImageModalityType
from src.core.domain.entities.enhanced_medical_image import EnhancedMedicalImage


class NIfTIFormatHandler(MedicalFormatHandler):    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        from src.core.domain.services.filename_study_grouper import FilenameStudyGrouper
        self._filename_grouper = FilenameStudyGrouper()
    
    def get_format_name(self) -> str:
        return "NIfTI"
    
    def get_capabilities(self) -> FormatCapabilities:
        return FormatCapabilities(
            can_read=True,
            can_write=False,  
            supports_metadata=True,
            supports_series=False,  
            supports_3d=True,
            typical_extensions=['.nii', '.nii.gz']
        )
    
    def can_handle_file(self, file_path: Path) -> bool:
        if file_path.suffix.lower() == '.nii':
            return True
        if file_path.suffixes[-2:] == ['.nii', '.gz']:
            return True
        return False
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        try:
            header = self._read_nifti_header(file_path)
            if header is None:
                return False, "Invalid NIfTI header"
            
            if header.get('sizeof_hdr', 0) not in [348, 540]:  
                return False, "Invalid NIfTI header size"
            
            magic = header.get('magic', b'').decode('ascii', errors='ignore')
            if not magic.startswith('ni1') and not magic.startswith('n+1'):
                return False, "Invalid NIfTI magic number"
            
            return True, None
            
        except Exception as e:
            return False, f"NIfTI validation error: {e}"
    
    def load_image(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            try:
                import nibabel as nib
                return self._load_with_nibabel(file_path)
            except ImportError:
                self._logger.info("nibabel not available, using basic NIfTI loader")
                return self._load_basic_nifti(file_path)
                
        except Exception as e:
            self._logger.error(f"Failed to load NIfTI file {file_path}: {e}")
            return None
    
    def _load_with_nibabel(self, file_path: Path) -> Optional[LoadedImageData]:
        import nibabel as nib
        
        try:
            nii_img = nib.load(str(file_path))
            image_data = nii_img.get_fdata()
            
            if len(image_data.shape) == 3:
                image_data = np.transpose(image_data, (2, 1, 0))
                MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=3)
            
            header = nii_img.header
            pixdim = header['pixdim'][1:8] 

            spacing = ImageSpacing(
                x=float(pixdim[0]) if len(pixdim) > 0 and pixdim[0] > 0 else 1.0,  # X spacing
                y=float(pixdim[1]) if len(pixdim) > 1 and pixdim[1] > 0 else 1.0,  # Y spacing  
                z=float(pixdim[2]) if len(pixdim) > 2 and pixdim[2] > 0 else 1.0   # Z spacing
            )
            
            metadata = self._extract_nifti_metadata(header, image_data.shape)            
            metadata['format'] = 'NIfTI'
            
            if len(image_data.shape) == 4:
                metadata['is_4d'] = True
                metadata['temporal_size'] = image_data.shape[3]
                if len(pixdim) > 3 and pixdim[3] > 0:
                    metadata['temporal_spacing'] = float(pixdim[3])
                
                if image_data.shape[3] > 10:
                    metadata['likely_temporal'] = True
                else:
                    metadata['likely_multi_channel'] = True
            
            modality = self._guess_modality_from_metadata(metadata, file_path)
            
            patient_id = self._generate_patient_id(file_path)
            study_uid, series_uid = self._generate_study_and_series_ids(file_path)
            
            return LoadedImageData(
                image_array=image_data.astype(np.float32),
                spacing=spacing,
                modality=modality,
                metadata=metadata,
                patient_id=patient_id,
                study_uid=study_uid,
                series_uid=series_uid,
                acquisition_date=datetime.now()
            )
            
        except Exception as e:
            self._logger.error(f"nibabel loading failed: {e}")
            return None
    
    def _load_basic_nifti(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            header = self._read_nifti_header(file_path)
            if not header:
                return None
            
            image_data = self._read_nifti_data(file_path, header)
            if image_data is None:
                return None
            
            pixdim = header.get('pixdim', [0, 1, 1, 1])
            spacing = ImageSpacing(
                x=float(pixdim[1]) if len(pixdim) > 1 and pixdim[1] > 0 else 1.0,
                y=float(pixdim[2]) if len(pixdim) > 2 and pixdim[2] > 0 else 1.0,
                z=float(pixdim[3]) if len(pixdim) > 3 and pixdim[3] > 0 else 1.0
            )
            
            metadata = {
                'format': 'NIfTI',
                'datatype': header.get('datatype', 0),
                'dimensions': header.get('dim', []),
                'file_path': str(file_path)
            }
            
            modality = ImageModalityType.MRI 
            patient_id = self._generate_patient_id(file_path)
            study_uid, series_uid = self._generate_study_and_series_ids(file_path)
            
            return LoadedImageData(
                image_array=image_data,
                spacing=spacing,
                modality=modality,
                metadata=metadata,
                patient_id=patient_id,
                study_uid=study_uid,
                series_uid=series_uid,
                acquisition_date=datetime.now()
            )
            
        except Exception as e:
            self._logger.error(f"Basic NIfTI loading failed: {e}")
            return None
    
    def _read_nifti_header(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            if file_path.suffixes[-2:] == ['.nii', '.gz']:
                with gzip.open(file_path, 'rb') as f:
                    header_bytes = f.read(348) 
            else:
                with open(file_path, 'rb') as f:
                    header_bytes = f.read(348)
            
            if len(header_bytes) < 348:
                return None
            
            header = {}
            header['sizeof_hdr'] = int.from_bytes(header_bytes[0:4], 'little')
            header['dim'] = [int.from_bytes(header_bytes[40+i*2:42+i*2], 'little') 
                            for i in range(8)]
            header['pixdim'] = [float(np.frombuffer(header_bytes[76+i*4:80+i*4], 'float32')[0]) 
                               for i in range(8)]
            header['datatype'] = int.from_bytes(header_bytes[70:72], 'little')            
            header['magic'] = header_bytes[344:348]
            
            return header
            
        except Exception as e:
            self._logger.error(f"Failed to read NIfTI header: {e}")
            return None
    
    def _read_nifti_data(self, file_path: Path, header: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            dims = header['dim'][1:4] 
            if any(d <= 0 for d in dims[:3]):
                self._logger.error("Invalid NIfTI dimensions")
                return None
            
            datatype = header['datatype']
            dtype_map = {
                2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64
            }
            
            if datatype not in dtype_map:
                self._logger.info(f"Unsupported datatype {datatype}, using float32")
                dtype = np.float32
            else:
                dtype = dtype_map[datatype]
            
            if file_path.suffixes[-2:] == ['.nii', '.gz']:
                with gzip.open(file_path, 'rb') as f:
                    f.seek(352) 
                    data_bytes = f.read()
            else:
                with open(file_path, 'rb') as f:
                    f.seek(352)  
                    data_bytes = f.read()
            
            data = np.frombuffer(data_bytes, dtype=dtype)
            
            if len(dims) >= 3:
                expected_size = dims[0] * dims[1] * dims[2]
                if data.size >= expected_size:
                    data = data[:expected_size].reshape(dims[2], dims[1], dims[0])  # Z, Y, X order
                else:
                    self._logger.error(f"Not enough data: got {data.size}, expected {expected_size}")
                    return None
            
            return data.astype(np.float32)
            
        except Exception as e:
            self._logger.error(f"Failed to read NIfTI data: {e}")
            return None
    
    def _extract_nifti_metadata(self, header, image_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
        metadata = {
            'format': 'NIfTI',
            'datatype': str(header.get('datatype', 'unknown')),
            'dimensions': str(header.get('dim', [])),
            'voxel_size': str(header.get('pixdim', [])),
        }
        
        if image_shape:
            metadata['image_shape'] = image_shape
            metadata['ndim'] = len(image_shape)
        
        if hasattr(header, 'get_data_dtype'):
            metadata['data_dtype'] = str(header.get_data_dtype())
        
        if hasattr(header, 'get_slope'):
            try:
                slope = header.get_slope()
                if slope != 1.0:
                    metadata['slope'] = slope
            except:
                pass
                
        if hasattr(header, 'get_intercept'):
            try:
                intercept = header.get_intercept()
                if intercept != 0.0:
                    metadata['intercept'] = intercept
            except:
                pass
        
        return metadata
    
    def _guess_modality_from_metadata(self, metadata: Dict[str, Any], file_path: Path) -> ImageModalityType:
        filename = file_path.name.lower()
        
        if 't1' in filename or 't2' in filename or 'flair' in filename:
            return ImageModalityType.MRI
        elif 'ct' in filename:
            return ImageModalityType.CT
        else:
            return ImageModalityType.MRI 
    
    def _generate_patient_id(self, file_path: Path) -> str:
        stem = file_path.stem.replace('.nii', '') 
        
        if 'patient' in stem.lower():
            return stem
        else:
            return f"NIFTI_PATIENT_{stem}"
    
    def _generate_study_and_series_ids(self, file_path: Path) -> Tuple[str, str]:        
        if not file_path:
            raise ValueError("file_path cannot be None")
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        stem = file_path.stem
        
        try:
            analysis = self._filename_grouper._analyze_filename(str(file_path))
            if analysis and analysis.get('study_id') and analysis.get('sequence_type'):
                study_id = analysis['study_id']
                sequence_type = analysis['sequence_type']
                
                study_uid = f"nifti_study_{study_id}"
                
                if sequence_type == 'UNKNOWN':
                    unique_part = stem.split('_')[-1] if '_' in stem else stem
                    series_uid = f"nifti_series_{study_id}_{unique_part}"
                else:
                    series_uid = f"nifti_series_{study_id}_{sequence_type}"
                
                return study_uid, series_uid
                
        except (AttributeError, KeyError, IndexError, ValueError) as e:
            self._logger.debug(f"Could not analyze filename for ID generation: {e}")
        
        study_uid = f"nifti_study_{stem}"
        series_uid = f"nifti_series_{stem}"
        return study_uid, series_uid
    
    def create_enhanced_medical_image(self, file_path: Path) -> Optional[EnhancedMedicalImage]:
        try:
            loaded_data = self.load_image(file_path)
            if not loaded_data:
                return None
            
            enhanced_image = EnhancedMedicalImage(
                image_data=loaded_data.image_array,
                spacing=loaded_data.spacing,
                modality=loaded_data.modality,
                patient_id=loaded_data.patient_id,
                study_instance_uid=loaded_data.study_uid,
                series_instance_uid=loaded_data.series_uid,
                acquisition_date=loaded_data.acquisition_date or datetime.now(),
                dicom_metadata=loaded_data.metadata,
                auto_analyze_dimensions=True 
            )
            
            if enhanced_image.is_4d:
                self._logger.info(f"Created 4D NIfTI image: {enhanced_image.spatial_dimensions} spatial + {enhanced_image.temporal_size} temporal/channel frames")
            else:
                self._logger.info(f"Created 3D NIfTI image with structure: {enhanced_image.structure}")
            
            return enhanced_image
            
        except Exception as e:
            self._logger.error(f"Failed to create enhanced NIfTI image: {e}")
            return None