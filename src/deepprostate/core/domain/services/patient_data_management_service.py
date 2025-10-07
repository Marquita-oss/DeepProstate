import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from deepprostate.core.domain.entities.medical_image import MedicalImage, ImageModalityType
from deepprostate.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository
from deepprostate.core.domain.services.filename_study_grouper import FilenameStudyGrouper, FilenameStudyGroup


class PatientDataManagementService:    
    def __init__(self, repository: DICOMImageRepository):
        self._logger = logging.getLogger(__name__)
        self._repository = repository
        self._filename_grouper = FilenameStudyGrouper()
        
        self._multi_modal_studies: Dict[str, Dict[str, Any]] = {} 
        
        self._current_patient_data: Dict[str, Any] = {}
        
    def get_repository(self) -> DICOMImageRepository:
        return self._repository
    
    async def load_patient_data(self, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            self._logger.debug("Loading patient data from DICOM repository")
            patient_data = {
                "patients": {},
                "loading_time": datetime.now(),
                "search_params": search_params or {},
                "total_patients": 0,
                "total_studies": 0,
                "total_series": 0
            }
            
            self._current_patient_data = patient_data
            self._logger.debug("Patient data loaded successfully")
            
            return patient_data
            
        except Exception as e:
            self._logger.error(f"Error loading patient data: {e}")
            raise
    
    def get_current_patient_data(self) -> Dict[str, Any]:
        return self._current_patient_data
    
    def transform_dicom_to_hierarchy(self, dicom_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        hierarchy = {"patients": {}}
        
        for dicom_item in dicom_data:
            patient_id = dicom_item.get('PatientID', 'UNKNOWN')
            study_uid = dicom_item.get('StudyInstanceUID', 'UNKNOWN_STUDY')
            series_uid = dicom_item.get('SeriesInstanceUID', 'UNKNOWN_SERIES')
            
            if patient_id not in hierarchy["patients"]:
                hierarchy["patients"][patient_id] = {
                    "patient_name": dicom_item.get('PatientName', 'Unknown Patient'),
                    "patient_id": patient_id,
                    "birth_date": dicom_item.get('PatientBirthDate', ''),
                    "sex": dicom_item.get('PatientSex', ''),
                    "studies": {}
                }
            
            patient = hierarchy["patients"][patient_id]
            
            if study_uid not in patient["studies"]:
                patient["studies"][study_uid] = {
                    "study_uid": study_uid,
                    "study_date": dicom_item.get('StudyDate', ''),
                    "study_time": dicom_item.get('StudyTime', ''),
                    "study_description": dicom_item.get('StudyDescription', ''),
                    "modality": dicom_item.get('Modality', ''),
                    "series": {}
                }
            
            study = patient["studies"][study_uid]
            
            if series_uid not in study["series"]:
                study["series"][series_uid] = {
                    "series_uid": series_uid,
                    "series_number": dicom_item.get('SeriesNumber', ''),
                    "series_description": dicom_item.get('SeriesDescription', ''),
                    "modality": dicom_item.get('Modality', ''),
                    "images_count": dicom_item.get('NumberOfSeriesRelatedInstances', 0),
                    "slice_thickness": dicom_item.get('SliceThickness', ''),
                    "image_orientation": dicom_item.get('ImageOrientationPatient', []),
                    "pixel_spacing": dicom_item.get('PixelSpacing', [])
                }
        
        return hierarchy
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        patients = self._current_patient_data.get('patients', {})
        return patients.get(patient_id)
    
    def get_study_info(self, patient_id: str, study_uid: str) -> Optional[Dict[str, Any]]:
        patient_info = self.get_patient_info(patient_id)
        if not patient_info:
            return None
        
        studies = patient_info.get('studies', {})
        return studies.get(study_uid)
    
    def get_series_info(self, patient_id: str, study_uid: str, series_uid: str) -> Optional[Dict[str, Any]]:
        study_info = self.get_study_info(patient_id, study_uid)
        if not study_info:
            return None
        
        series = study_info.get('series', {})
        return series.get(series_uid)
    
    def get_all_series_in_study(self, patient_id: str, study_uid: str) -> List[Dict[str, Any]]:
        study_info = self.get_study_info(patient_id, study_uid)
        if not study_info:
            return []
        
        series_list = []
        for series_uid, series_info in study_info.get('series', {}).items():
            series_info_copy = series_info.copy()
            series_info_copy['patient_id'] = patient_id
            series_info_copy['study_uid'] = study_uid
            series_list.append(series_info_copy)
        
        return series_list
    
    def get_all_studies_for_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        patient_info = self.get_patient_info(patient_id)
        if not patient_info:
            return []
        
        studies_list = []
        for study_uid, study_info in patient_info.get('studies', {}).items():
            study_info_copy = study_info.copy()
            study_info_copy['patient_id'] = patient_id
            studies_list.append(study_info_copy)
        
        return studies_list
    
    def store_multi_modal_study(self, study_uid: str, study_analysis: Dict[str, Any]) -> None:
        self._multi_modal_studies[study_uid] = study_analysis
        self._logger.debug(f"Stored multi-modal study analysis for {study_uid[:8]}")
    
    def get_multi_modal_study(self, study_uid: str) -> Optional[Dict[str, Any]]:
        return self._multi_modal_studies.get(study_uid)
    
    def clear_multi_modal_studies(self) -> None:
        self._multi_modal_studies.clear()
        self._logger.debug("Cleared all multi-modal study analyses")
    
    def group_files_by_study(self, file_paths: List[Path]) -> Dict[str, FilenameStudyGroup]:
        return self._filename_grouper.group_files_by_study(file_paths)
    
    def create_loaded_case_data(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        case_data = {
            'patient_id': patient_info.get('patient_id', 'Unknown'),
            'patient_name': patient_info.get('patient_name', 'Unknown Patient'),
            'birth_date': patient_info.get('birth_date', ''),
            'sex': patient_info.get('sex', ''),
            'studies_count': len(patient_info.get('studies', {})),
            'total_series': 0,
            'modalities': set(),
            'date_range': {'earliest': None, 'latest': None}
        }
        
        for study_info in patient_info.get('studies', {}).values():
            study_date = study_info.get('study_date', '')
            if study_date:
                if not case_data['date_range']['earliest'] or study_date < case_data['date_range']['earliest']:
                    case_data['date_range']['earliest'] = study_date
                if not case_data['date_range']['latest'] or study_date > case_data['date_range']['latest']:
                    case_data['date_range']['latest'] = study_date
            
            series_count = len(study_info.get('series', {}))
            case_data['total_series'] += series_count
            
            for series_info in study_info.get('series', {}).values():
                modality = series_info.get('modality', '')
                if modality:
                    case_data['modalities'].add(modality)
        
        case_data['modalities'] = list(case_data['modalities'])
        
        return case_data
    
    def get_data_statistics(self) -> Dict[str, int]:
        patients = self._current_patient_data.get('patients', {})
        
        total_patients = len(patients)
        total_studies = 0
        total_series = 0
        
        for patient_info in patients.values():
            studies = patient_info.get('studies', {})
            total_studies += len(studies)
            
            for study_info in studies.values():
                series = study_info.get('series', {})
                total_series += len(series)
        
        return {
            'total_patients': total_patients,
            'total_studies': total_studies,
            'total_series': total_series
        }
    
    def validate_patient_data_structure(self, patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        if 'patients' not in patient_data:
            errors.append("Missing 'patients' key in patient data")
            return False, errors
        
        patients = patient_data['patients']
        if not isinstance(patients, dict):
            errors.append("'patients' must be a dictionary")
            return False, errors
        
        for patient_id, patient_info in patients.items():
            if not isinstance(patient_info, dict):
                errors.append(f"Patient {patient_id} info must be a dictionary")
                continue
            
            required_patient_fields = ['patient_id', 'studies']
            for field in required_patient_fields:
                if field not in patient_info:
                    errors.append(f"Patient {patient_id} missing required field: {field}")
            
            studies = patient_info.get('studies', {})
            if not isinstance(studies, dict):
                errors.append(f"Patient {patient_id} studies must be a dictionary")
                continue
            
            for study_uid, study_info in studies.items():
                if not isinstance(study_info, dict):
                    errors.append(f"Study {study_uid} info must be a dictionary")
                    continue
                
                required_study_fields = ['study_uid', 'series']
                for field in required_study_fields:
                    if field not in study_info:
                        errors.append(f"Study {study_uid} missing required field: {field}")
                
                series = study_info.get('series', {})
                if not isinstance(series, dict):
                    errors.append(f"Study {study_uid} series must be a dictionary")
        
        return len(errors) == 0, errors
    
    def clear_patient_data(self) -> None:
        self._current_patient_data = {}
        self._logger.debug("Cleared all patient data")