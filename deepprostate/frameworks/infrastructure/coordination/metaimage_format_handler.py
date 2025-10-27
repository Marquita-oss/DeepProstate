import logging
import struct
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
import numpy as np
from deepprostate.core.domain.services.filename_study_grouper import FilenameStudyGrouper
from deepprostate.frameworks.infrastructure.constants.medical_sequences import (
    SEQUENCE_DESCRIPTIONS, MRI_SEQUENCES, CT_SEQUENCES, 
    get_sequence_description, is_mri_sequence, is_ct_sequence
)

from .medical_format_handler import (
    MedicalFormatHandler, FormatCapabilities, LoadedImageData
)
from deepprostate.core.domain.entities.medical_image import ImageSpacing, ImageModalityType
from deepprostate.core.domain.entities.enhanced_medical_image import EnhancedMedicalImage


class MetaImageFormatHandler(MedicalFormatHandler):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._filename_grouper = FilenameStudyGrouper()
    
    def get_format_name(self) -> str:
        return "MetaImage"
    
    def get_capabilities(self) -> FormatCapabilities:
        return FormatCapabilities(
            can_read=True,
            can_write=False,  
            supports_metadata=True,
            supports_series=False,  
            supports_3d=True,
            typical_extensions=['.mha', '.mhd']
        )
    
    def can_handle_file(self, file_path: Path) -> bool:
        extension = file_path.suffix.lower()
        return extension in ['.mha', '.mhd']
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        try:
            header = self._read_metaimage_header(file_path)
            if not header:
                return False, "Invalid MetaImage header"
            
            if 'NDims' not in header:
                return False, "Missing NDims in MetaImage header"
            
            if 'DimSize' not in header:
                return False, "Missing DimSize in MetaImage header"
            
            if 'ElementType' not in header:
                return False, "Missing ElementType in MetaImage header"
            
            return True, None
            
        except Exception as e:
            return False, f"MetaImage validation error: {e}"
    
    def load_image(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            header = self._read_metaimage_header(file_path)
            if not header:
                return None
            
            image_data = self._read_metaimage_data(file_path, header)
            if image_data is None:
                return None
            
            spacing = self._extract_metaimage_spacing(header)
            modality = self._guess_modality_from_metadata(header, file_path)
            metadata = self._extract_metaimage_metadata(header, image_data.shape)
            metadata['format'] = 'MetaImage'
            sequence_info = self._extract_sequence_information(file_path)
            metadata.update(sequence_info)
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
            self._logger.error(f"Failed to load MetaImage file {file_path}: {e}")
            return None
    
    def _read_metaimage_header(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            header = {}
            
            if file_path.suffix.lower() == '.mha':
                with open(file_path, 'rb') as f:
                    header_lines = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        
                        line_str = line.decode('ascii', errors='ignore').strip()
                        if line_str == 'ElementDataFile = LOCAL':
                            header['_data_offset'] = f.tell()
                            break
                        
                        header_lines.append(line_str)
                        
                        if len(header_lines) > 100:
                            break
            else:
                with open(file_path, 'r', encoding='ascii', errors='ignore') as f:
                    header_lines = [line.strip() for line in f.readlines()]
            
            for line in header_lines:
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'NDims':
                        header[key] = int(value)
                    elif key == 'DimSize':
                        header[key] = [int(x) for x in value.split()]
                    elif key == 'ElementSpacing':
                        header[key] = [float(x) for x in value.split()]
                    elif key == 'ElementSize':
                        header[key] = [float(x) for x in value.split()]
                    elif key == 'ElementType':
                        header[key] = value
                    elif key == 'ElementDataFile':
                        header[key] = value
                    elif key == 'ElementByteOrderMSB':
                        header[key] = value.lower() in ['true', '1']
                    elif key == 'CompressedData':
                        header[key] = value.lower() in ['true', '1']
                    elif key == 'CompressedDataSize':
                        header[key] = int(value)
                    else:
                        header[key] = value
            
            return header
            
        except Exception as e:
            self._logger.error(f"Failed to read MetaImage header: {e}")
            return None
    
    def _read_metaimage_data(self, file_path: Path, header: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            if file_path.suffix.lower() == '.mha':
                data_file = file_path
                data_offset = header.get('_data_offset', 0)
            else:
                data_file_name = header.get('ElementDataFile', f"{file_path.stem}.raw")
                if data_file_name == 'LOCAL':
                    data_file = file_path
                    data_offset = 0
                else:
                    data_file = file_path.parent / data_file_name
                    data_offset = 0
            
            element_type = header.get('ElementType', 'MET_FLOAT')
            dtype = self._metaimage_type_to_numpy(element_type)
            
            dim_size = header.get('DimSize', [])
            if not dim_size:
                self._logger.error("No DimSize found in MetaImage header")
                return None
            
            total_elements = 1
            for size in dim_size:
                total_elements *= size
            
            with open(data_file, 'rb') as f:
                if data_offset > 0:
                    f.seek(data_offset)
                
                is_compressed = header.get('CompressedData', False)
                if is_compressed:
                    compressed_size = header.get('CompressedDataSize', 0)
                    if compressed_size > 0:
                        compressed_data = f.read(compressed_size)
                    else:
                        compressed_data = f.read() 
                    
                    try:
                        import zlib
                        data_bytes = zlib.decompress(compressed_data)
                    except Exception as e:
                        self._logger.error(f"Failed to decompress data: {e}")
                        return None
                else:
                    data_bytes = f.read(total_elements * dtype().itemsize)
                
                expected_size = total_elements * dtype().itemsize
                if len(data_bytes) < expected_size:
                    self._logger.error(f"Not enough data: expected {expected_size}, got {len(data_bytes)}")
                    return None
            
            data = np.frombuffer(data_bytes, dtype=dtype)            
            byte_order_msb = header.get('ElementByteOrderMSB', False)
            if byte_order_msb and data.dtype.byteorder in ['<', '=']:
                data = data.byteswap().newbyteorder()
            elif not byte_order_msb and data.dtype.byteorder == '>':
                data = data.byteswap().newbyteorder()
            
            if len(dim_size) >= 3:
                z_size, y_size, x_size = dim_size[2], dim_size[1], dim_size[0]
                data = data.reshape(z_size, y_size, x_size)
            else:
                data = data.reshape(dim_size)
            
            return data.astype(np.float32)
            
        except Exception as e:
            self._logger.error(f"Failed to read MetaImage data: {e}")
            return None
    
    def _metaimage_type_to_numpy(self, element_type: str) -> np.dtype:
        type_map = {
            'MET_CHAR': np.int8,
            'MET_UCHAR': np.uint8,
            'MET_SHORT': np.int16,
            'MET_USHORT': np.uint16,
            'MET_INT': np.int32,
            'MET_UINT': np.uint32,
            'MET_LONG': np.int64,
            'MET_ULONG': np.uint64,
            'MET_FLOAT': np.float32,
            'MET_DOUBLE': np.float64
        }
        
        return type_map.get(element_type, np.float32)
    
    def _extract_metaimage_spacing(self, header: Dict[str, Any]) -> ImageSpacing:
        try:
            spacing_data = header.get('ElementSpacing')
            if not spacing_data:
                spacing_data = header.get('ElementSize', [1.0, 1.0, 1.0])
            
            while len(spacing_data) < 3:
                spacing_data.append(1.0)
            
            return ImageSpacing(
                x=float(spacing_data[0]),
                y=float(spacing_data[1]),
                z=float(spacing_data[2]) if len(spacing_data) > 2 else 1.0
            )
            
        except Exception as e:
            self._logger.error(f"Could not extract MetaImage spacing: {e}")
            return ImageSpacing(x=1.0, y=1.0, z=1.0)
    
    def _extract_metaimage_metadata(self, header: Dict[str, Any], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        metadata = {
            'format': 'MetaImage',
            'element_type': header.get('ElementType', 'unknown'),
            'dimensions': header.get('DimSize', []),
            'spacing': header.get('ElementSpacing', header.get('ElementSize', [])),
            'byte_order_msb': header.get('ElementByteOrderMSB', False),
            'compressed': header.get('CompressedData', False),
        }
        
        if image_shape:
            metadata['image_shape'] = image_shape
            metadata['ndim'] = len(image_shape)
        
        for key, value in header.items():
            if key not in metadata and not key.startswith('_'):
                metadata[f'metaimage_{key.lower()}'] = str(value)
        
        return metadata
    
    def _guess_modality_from_metadata(self, metadata: Dict[str, Any], file_path: Path) -> ImageModalityType:
        filename = file_path.name.lower()
        
        try:
            analysis = self._filename_grouper._analyze_filename(str(file_path))
            if analysis and analysis.get('sequence_type'):
                sequence_type = analysis['sequence_type'].upper()
                
                if is_mri_sequence(sequence_type):
                    return ImageModalityType.MRI
                elif is_ct_sequence(sequence_type):
                    return ImageModalityType.CT
        except Exception as e:
            self._logger.debug(f"Could not determine modality from sequence analysis: {e}")
        
        if 't1' in filename or 't2' in filename or 'flair' in filename or 'mri' in filename:
            return ImageModalityType.MRI
        elif 'ct' in filename:
            return ImageModalityType.CT
        else:
            return ImageModalityType.MRI
    
    def _generate_patient_id(self, file_path: Path) -> str:
        if not file_path:
            raise ValueError("file_path cannot be None")
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        stem = file_path.stem
        
        try:
            analysis = self._filename_grouper._analyze_filename(str(file_path))
            if analysis and analysis.get('study_id'):
                study_id = analysis['study_id']
                return f"METAIMAGE_PATIENT_{study_id}"
        except (AttributeError, KeyError, IndexError, ValueError) as e:
            self._logger.debug(f"Could not analyze filename with grouper: {e}")
        
        if 'patient' in stem.lower():
            return stem
        else:
            return f"METAIMAGE_PATIENT_{stem}"
    
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
                
                study_uid = f"metaimage_study_{study_id}"
                
                if sequence_type == 'UNKNOWN':
                    unique_part = stem.split('_')[-1] if '_' in stem else stem
                    series_uid = f"metaimage_series_{study_id}_{unique_part}"
                else:
                    series_uid = f"metaimage_series_{study_id}_{sequence_type}"
                
                return study_uid, series_uid
                
        except (AttributeError, KeyError, IndexError, ValueError) as e:
            self._logger.debug(f"Could not analyze filename for ID generation: {e}")
        
        study_uid = f"metaimage_study_{stem}"
        series_uid = f"metaimage_series_{stem}"
        return study_uid, series_uid
    
    def _extract_sequence_information(self, file_path: Path) -> Dict[str, str]:
        if not file_path:
            raise ValueError("file_path cannot be None")
        
        try:
            analysis = self._filename_grouper._analyze_filename(str(file_path))
            if analysis and analysis.get('sequence_type'):
                sequence_type = analysis['sequence_type']
                
                description = get_sequence_description(sequence_type)
                
                return {
                    'SeriesDescription': description,
                    'SequenceType': sequence_type,
                    'ProtocolName': f'{sequence_type} Protocol',
                    'SequenceName': sequence_type
                }
        except (AttributeError, KeyError, IndexError, ValueError) as e:
            self._logger.debug(f"Could not extract sequence information: {e}")
        
        stem = file_path.stem
        sequence_name = 'UNKNOWN'
        series_description = f'MetaImage Series ({stem})'
        
        if '_' in stem:
            sequence_name = stem.split('_')[-1].upper()
            series_description = f'MetaImage {sequence_name} Series'
        
        return {
            'SeriesDescription': series_description,
            'SequenceType': 'UNKNOWN',
            'ProtocolName': f'MetaImage Protocol',
            'SequenceName': sequence_name
        }
    
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
                self._logger.info(f"Created 4D MetaImage: {enhanced_image.spatial_dimensions} spatial + {enhanced_image.temporal_size} temporal/channel frames")
            else:
                self._logger.info(f"Created 3D MetaImage with structure: {enhanced_image.structure}")
            
            return enhanced_image
            
        except Exception as e:
            self._logger.error(f"Failed to create enhanced MetaImage: {e}")
            return None