import logging
import pydicom
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from deepprostate.core.domain.entities.medical_image import MedicalImage

from deepprostate.frameworks.infrastructure.utils.dicom_metadata_extractor import dicom_extractor
from deepprostate.frameworks.infrastructure.utils.filesystem_validator import filesystem_validator
from .medical_format_registry import get_medical_format_registry


@dataclass
class DicomFileInfo:
    file_path: Path
    series_uid: str
    study_uid: str
    modality: str
    series_description: str
    patient_id: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None


class DicomFileHandler:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.info("DicomFileHandler initialized")
    
    def validate_dicom_format(self, file_path: str) -> bool:
        is_valid, error = filesystem_validator.validate_dicom_file(file_path)
        if not is_valid:
            self._logger.debug(f"DICOM validation failed for {file_path}: {error}")
        return is_valid
    
    def validate_dicom_directory(self, directory_path: str) -> bool:
        is_valid, error, dicom_files = filesystem_validator.validate_dicom_directory(
            directory_path, min_dicom_files=1, validate_all_files=False
        )
        
        if not is_valid:
            self._logger.debug(f"DICOM directory validation failed for {directory_path}: {error}")
        else:
            self._logger.info(f"Directory analysis: {len(dicom_files)} DICOM files found")
        
        return is_valid
    
    
    def extract_dicom_file_info(self, file_path: str) -> Optional[DicomFileInfo]:
        try:
            if not self.validate_dicom_format(file_path):
                return DicomFileInfo(
                    file_path=Path(file_path),
                    series_uid="UNKNOWN",
                    study_uid="UNKNOWN", 
                    modality="UNKNOWN",
                    series_description="",
                    is_valid=False,
                    error_message="Invalid DICOM format"
                )
            
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            series_uid = dicom_extractor.safe_get_dicom_value(ds, 'SeriesInstanceUID', 'UNKNOWN')
            study_uid = dicom_extractor.safe_get_dicom_value(ds, 'StudyInstanceUID', 'UNKNOWN')
            modality = dicom_extractor.safe_get_dicom_value(ds, 'Modality', 'UNKNOWN')
            series_description = dicom_extractor.safe_get_dicom_value(ds, 'SeriesDescription', '')
            patient_id = dicom_extractor.safe_get_dicom_value(ds, 'PatientID')
            
            return DicomFileInfo(
                file_path=Path(file_path),
                series_uid=series_uid,
                study_uid=study_uid,
                modality=modality,
                series_description=series_description,
                patient_id=patient_id,
                is_valid=True
            )
            
        except Exception as e:
            self._logger.error(f"Error extracting DICOM info from {file_path}: {e}")
            return DicomFileInfo(
                file_path=Path(file_path),
                series_uid="UNKNOWN",
                study_uid="UNKNOWN",
                modality="UNKNOWN", 
                series_description="",
                is_valid=False,
                error_message=str(e)
            )
    
    def analyze_folder_content(self, folder_path: str) -> Tuple[List[DicomFileInfo], Dict[str, List[DicomFileInfo]]]:
        MAX_FILES_TO_PROCESS = 1000  
        
        try:
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                self._logger.error(f"Folder does not exist: {folder_path}")
                return [], {}
            
            dicom_files = []
            processed_count = 0
            
            for file_path in folder.iterdir():
                if processed_count >= MAX_FILES_TO_PROCESS:
                    self._logger.warning(f"Processed maximum {MAX_FILES_TO_PROCESS} files, stopping to prevent UI freeze")
                    break
                
                if file_path.is_file():
                    file_info = self.extract_dicom_file_info(str(file_path))
                    if file_info and file_info.is_valid:
                        dicom_files.append(file_info)
                    processed_count += 1
            
            series_groups = {}
            for file_info in dicom_files:
                series_uid = file_info.series_uid
                if series_uid not in series_groups:
                    series_groups[series_uid] = []
                series_groups[series_uid].append(file_info)
            
            self._logger.info(f"Analyzed {len(dicom_files)} DICOM files in {len(series_groups)} series")
            return dicom_files, series_groups
            
        except Exception as e:
            self._logger.error(f"Error analyzing folder content: {e}")
            return [], {}
    
    def load_dicom_with_fallback(self, dicom_path: Path) -> Optional[pydicom.Dataset]:
        try:
            ds = pydicom.dcmread(str(dicom_path))
            self._logger.debug(f"Successfully loaded DICOM: {dicom_path}")
            return ds
            
        except pydicom.errors.InvalidDicomError as e:
            self._logger.error(f"Invalid DICOM file {dicom_path}: {e}")
            return None
            
        except Exception as e:
            self._logger.error(f"Error loading DICOM file {dicom_path}: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        registry = get_medical_format_registry()
        return registry.get_supported_extensions()
    
    def is_supported_file(self, file_path: str) -> bool:
        registry = get_medical_format_registry()
        return registry.can_load_file(Path(file_path))
    
    def load_medical_image_any_format(self, file_path: str) -> Optional[MedicalImage]:
        registry = get_medical_format_registry()
        return registry.load_medical_image(Path(file_path))
    
    def get_file_format_info(self, file_path: str) -> Dict[str, Any]:
        registry = get_medical_format_registry()
        format_name = registry.get_file_format(Path(file_path))
        
        if not format_name:
            return {'supported': False, 'format': None}
        
        is_valid, error_msg, detected_format = registry.validate_file(Path(file_path))
        
        return {
            'supported': True,
            'format': format_name,
            'valid': is_valid,
            'error': error_msg,
            'capabilities': registry.get_format_capabilities().get(format_name, {})
        }