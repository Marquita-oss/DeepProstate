import logging
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from src.core.domain.exceptions import DicomProcessingError, DataValidationError, create_error_context


class DicomMetadataExtractor:
    _logger: logging.Logger
    _standard_tags: Dict[str, tuple]
    
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        
        # Standard DICOM tags that are commonly extracted
        self._standard_tags = {
            # Patient Information
            'PatientID': (str, ''),
            'PatientName': (str, ''),
            'PatientBirthDate': (str, ''),
            'PatientSex': (str, ''),
            'PatientAge': (str, ''),
            
            # Study Information
            'StudyInstanceUID': (str, ''),
            'StudyDescription': (str, ''),
            'StudyDate': (str, ''),
            'StudyTime': (str, ''),
            'AccessionNumber': (str, ''),
            
            # Series Information
            'SeriesInstanceUID': (str, ''),
            'SeriesDescription': (str, ''),
            'SeriesNumber': (int, 0),
            'Modality': (str, 'UNKNOWN'),
            'ProtocolName': (str, ''),
            'SeriesDate': (str, ''),
            'SeriesTime': (str, ''),
            
            # Image Information
            'ImageType': (list, []),
            'InstanceNumber': (int, 0),
            'SliceThickness': (float, 1.0),
            'PixelSpacing': (list, [1.0, 1.0]),
            'ImageOrientationPatient': (list, [1, 0, 0, 0, 1, 0]),
            'ImagePositionPatient': (list, [0, 0, 0]),
            'Rows': (int, 512),
            'Columns': (int, 512),
            'BitsAllocated': (int, 16),
            'BitsStored': (int, 16),
            'HighBit': (int, 15),
            'PixelRepresentation': (int, 0),
            
            # Equipment Information
            'Manufacturer': (str, ''),
            'ManufacturerModelName': (str, ''),
            'SoftwareVersions': (str, ''),
            'InstitutionName': (str, ''),
            'StationName': (str, ''),
            
            # Acquisition Parameters (MRI)
            'RepetitionTime': (float, 0.0),
            'EchoTime': (float, 0.0),
            'FlipAngle': (float, 0.0),
            'MagneticFieldStrength': (float, 0.0),
            'PixelBandwidth': (float, 0.0),
            
            # Contrast and Window Settings
            'WindowCenter': (Union[float, List[float]], 0.0),
            'WindowWidth': (Union[float, List[float]], 0.0),
            'RescaleIntercept': (float, 0.0),
            'RescaleSlope': (float, 1.0),
        }
    
    def safe_get_dicom_value(
        self, 
        dataset: Dataset, 
        tag_name: str, 
        default_value: Any = None, 
        value_type: Optional[Type] = None
    ) -> Any:
        try:
            if not hasattr(dataset, tag_name):
                self._logger.debug(f"DICOM tag '{tag_name}' not found, using default: {default_value}")
                return default_value
            
            value = getattr(dataset, tag_name)
            
            if value is None:
                return default_value
            
            if hasattr(value, 'value'):
                value = value.value
                if value is None:
                    return default_value
            
            if isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                value = str(value).strip()
                
                if not value:
                    return default_value
            
            if value_type is not None:
                return self._convert_value_type(value, value_type, default_value, tag_name)
            
            return value
            
        except (AttributeError, KeyError, TypeError) as e:
            self._logger.debug(f"DICOM tag '{tag_name}' not accessible: {e}")
            return default_value
        except Exception as e:
            context = create_error_context(
                operation="extract_dicom_metadata",
                component="DicomMetadataExtractor",
                tag_name=tag_name
            )
            raise DicomProcessingError(
                f"Unexpected error extracting DICOM tag '{tag_name}': {e}",
                error_code="METADATA_EXTRACTION_ERROR",
                dicom_tag=tag_name,
                details=context,
                cause=e
            )
    
    def extract_standard_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        metadata = {}
        
        for tag_name, (expected_type, default_value) in self._standard_tags.items():
            metadata[tag_name] = self.safe_get_dicom_value(
                dataset, tag_name, default_value, expected_type
            )
        
        return metadata
    
    def extract_custom_metadata(
        self, 
        dataset: Dataset, 
        tag_definitions: Dict[str, tuple]
    ) -> Dict[str, Any]:
        metadata = {}
        
        for tag_name, (expected_type, default_value) in tag_definitions.items():
            metadata[tag_name] = self.safe_get_dicom_value(
                dataset, tag_name, default_value, expected_type
            )
        
        return metadata
    
    def extract_medical_relevant_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        medical_tags = {
            'PatientID': (str, ''),
            'StudyInstanceUID': (str, ''),
            'SeriesInstanceUID': (str, ''),
            'SOPInstanceUID': (str, ''),
            
            'StudyDescription': (str, ''),
            'SeriesDescription': (str, ''),
            'BodyPartExamined': (str, ''),
            'PatientPosition': (str, ''),
            
            'Modality': (str, 'UNKNOWN'),
            'SliceThickness': (float, 1.0),
            'PixelSpacing': (list, [1.0, 1.0]),
            'ImageOrientationPatient': (list, [1, 0, 0, 0, 1, 0]),
            'ImagePositionPatient': (list, [0, 0, 0]),
            
            'WindowCenter': (Union[float, List[float]], 0.0),
            'WindowWidth': (Union[float, List[float]], 0.0),
            'RescaleIntercept': (float, 0.0),
            'RescaleSlope': (float, 1.0),
        }
        
        return self.extract_custom_metadata(dataset, medical_tags)
    
    def extract_timing_metadata(self, dataset: Dataset) -> Dict[str, Optional[datetime]]:
        timing_data = {}
        
        study_date = self.safe_get_dicom_value(dataset, 'StudyDate', '')
        study_time = self.safe_get_dicom_value(dataset, 'StudyTime', '')
        timing_data['study_datetime'] = self._parse_dicom_datetime(study_date, study_time)
        
        series_date = self.safe_get_dicom_value(dataset, 'SeriesDate', '')
        series_time = self.safe_get_dicom_value(dataset, 'SeriesTime', '')
        timing_data['series_datetime'] = self._parse_dicom_datetime(series_date, series_time)
        
        content_date = self.safe_get_dicom_value(dataset, 'ContentDate', '')
        content_time = self.safe_get_dicom_value(dataset, 'ContentTime', '')
        timing_data['content_datetime'] = self._parse_dicom_datetime(content_date, content_time)
        
        return timing_data
    
    def extract_sequence_classification(self, dataset: Dataset) -> Dict[str, str]:
        modality = self.safe_get_dicom_value(dataset, 'Modality', 'UNKNOWN')
        description = self.safe_get_dicom_value(dataset, 'SeriesDescription', '')
        
        return self._classify_sequence(modality, description)
    
    def validate_required_tags(
        self, 
        dataset: Dataset, 
        required_tags: List[str]
    ) -> Dict[str, bool]:
        validation_results = {}
        
        for tag_name in required_tags:
            value = self.safe_get_dicom_value(dataset, tag_name)            
            is_valid = (
                value is not None and 
                value != '' and 
                (not isinstance(value, (list, tuple)) or len(value) > 0)
            )
            
            validation_results[tag_name] = is_valid
        
        return validation_results
    
    def _convert_value_type(
        self, 
        value: Any, 
        target_type: Type, 
        default_value: Any, 
        tag_name: str
    ) -> Any:
        try:
            if target_type == int:
                if isinstance(value, float):
                    return int(value)
                return int(float(str(value)))
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value)
            elif target_type == list:
                if isinstance(value, (list, tuple)):
                    return list(value)
                else:
                    return [value]
            else:
                return target_type(value)
                
        except (ValueError, TypeError) as e:
            self._logger.warning(f"Type conversion failed for tag '{tag_name}': {e}")
            return default_value
        except Exception as e:
            context = create_error_context(
                operation="convert_dicom_value_type",
                component="DicomMetadataExtractor", 
                tag_name=tag_name,
                target_type=str(target_type)
            )
            raise DataValidationError(
                f"Unexpected error converting DICOM value for tag '{tag_name}' to {target_type}: {e}",
                error_code="TYPE_CONVERSION_ERROR",
                validation_type="type_conversion",
                expected_value=str(target_type),
                actual_value=str(type(value)),
                details=context,
                cause=e
            )
    
    def _parse_dicom_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        try:
            if not date_str:
                return None
            
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                hour = minute = second = microsecond = 0
                if time_str:
                    time_str = time_str.replace('.', '')
                    if len(time_str) >= 2:
                        hour = int(time_str[:2])
                    if len(time_str) >= 4:
                        minute = int(time_str[2:4])
                    if len(time_str) >= 6:
                        second = int(time_str[4:6])
                    if len(time_str) > 6:
                        frac = time_str[6:12].ljust(6, '0')
                        microsecond = int(frac)
                
                return datetime(year, month, day, hour, minute, second, microsecond)
                
        except (ValueError, IndexError) as e:
            self._logger.debug(f"Error parsing DICOM datetime '{date_str}' '{time_str}': {e}")
            return None
        except Exception as e:
            self._logger.warning(f"Unexpected error parsing DICOM datetime '{date_str}' '{time_str}': {e}")
            return None
    
    def _classify_sequence(self, modality: str, description: str) -> Dict[str, str]:
        description_upper = (description or "").upper()
        modality_upper = modality.upper()
        
        classification = {
            'base_modality': modality_upper,
            'sequence_type': 'UNKNOWN',
            'sequence_name': description,
            'contrast_enhanced': False
        }
        
        if modality_upper in ["MR", "MRI"]:
            if any(keyword in description_upper for keyword in ["T1", "T1W"]):
                if any(keyword in description_upper for keyword in ["POST", "CONTRAST", "+C", "GAD"]):
                    classification.update({
                        'sequence_type': 'T1_POST_CONTRAST',
                        'contrast_enhanced': True
                    })
                else:
                    classification['sequence_type'] = 'T1_WEIGHTED'
            elif any(keyword in description_upper for keyword in ["T2", "T2W"]):
                if "FLAIR" in description_upper:
                    classification['sequence_type'] = 'FLAIR'
                else:
                    classification['sequence_type'] = 'T2_WEIGHTED'
            elif "FLAIR" in description_upper:
                classification['sequence_type'] = 'FLAIR'
            elif any(keyword in description_upper for keyword in ["DWI", "DIFFUSION"]):
                classification['sequence_type'] = 'DIFFUSION_WEIGHTED'
            elif "ADC" in description_upper:
                classification['sequence_type'] = 'ADC_MAP'
            elif any(keyword in description_upper for keyword in ["DCE", "DYNAMIC"]):
                classification.update({
                    'sequence_type': 'DYNAMIC_CONTRAST',
                    'contrast_enhanced': True
                })
            elif any(keyword in description_upper for keyword in ["PERFUSION", "PWI"]):
                classification['sequence_type'] = 'PERFUSION'
            else:
                classification['sequence_type'] = 'MR_OTHER'
                
        elif modality_upper == "CT":
            if any(keyword in description_upper for keyword in ["ARTERIAL", "ART"]):
                classification.update({
                    'sequence_type': 'CT_ARTERIAL_PHASE',
                    'contrast_enhanced': True
                })
            elif any(keyword in description_upper for keyword in ["VENOUS", "VEN", "PORTAL"]):
                classification.update({
                    'sequence_type': 'CT_VENOUS_PHASE',
                    'contrast_enhanced': True
                })
            elif any(keyword in description_upper for keyword in ["DELAYED"]):
                classification.update({
                    'sequence_type': 'CT_DELAYED_PHASE',
                    'contrast_enhanced': True
                })
            elif any(keyword in description_upper for keyword in ["NATIVE", "NON", "WITHOUT"]):
                classification['sequence_type'] = 'CT_NATIVE'
            else:
                classification['sequence_type'] = 'CT_UNKNOWN'
                
        elif modality_upper == "US":
            classification['sequence_type'] = 'ULTRASOUND'
            
        elif modality_upper in ["XA", "RF"]:
            classification['sequence_type'] = 'FLUOROSCOPY'
            
        elif modality_upper == "MG":
            classification['sequence_type'] = 'MAMMOGRAPHY'
            
        else:
            classification['sequence_type'] = f"{modality_upper}_OTHER"
        
        return classification

dicom_extractor = DicomMetadataExtractor()