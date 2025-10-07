import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from deepprostate.core.domain.entities.medical_image import MedicalImage


class PatientBrowserIntegrationService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._patient_browser = None
        self._current_medical_image: Optional[MedicalImage] = None
        self._current_image_file_path: Optional[Path] = None
    
    def set_patient_browser_reference(self, patient_browser) -> None:
        self._patient_browser = patient_browser
        self._logger.info("Patient Browser reference set in integration service")
    
    def set_current_medical_image(self, medical_image: MedicalImage, file_path: Optional[Path] = None) -> None:
        self._current_medical_image = medical_image
        self._current_image_file_path = file_path
        
        if medical_image:
            self._logger.info(f"Current medical image updated: {medical_image.patient_id}")
        else:
            self._logger.info("Current medical image cleared")
    
    def get_current_medical_image(self) -> Optional[MedicalImage]:
        return self._current_medical_image
    
    def get_current_image_file_path(self) -> Optional[Path]:
        return self._current_image_file_path
    
    def get_loaded_cases(self) -> Dict[str, MedicalImage]:
        if not self._patient_browser:
            self._logger.warning("No Patient Browser reference available - cannot retrieve loaded cases")
            return {}

        loaded_images = {}

        if not hasattr(self._patient_browser, '_cache_service'):
            self._logger.error("Patient Browser has no _cache_service attribute!")
            return {}

        cache_service = self._patient_browser._cache_service
        self._logger.debug(f"Found cache service: {type(cache_service).__name__}")

        if not hasattr(cache_service, '_loaded_images'):
            self._logger.error("Cache service has no _loaded_images attribute!")
            return {}

        if not hasattr(cache_service, '_cache_mutex'):
            self._logger.error("Cache service has no _cache_mutex attribute!")
            return {}

        cache_service._cache_mutex.lock()
        try:
            loaded_images = dict(cache_service._loaded_images)  
            self._logger.debug(f"Retrieved {len(loaded_images)} loaded cases from Patient Browser cache")

            if loaded_images:
                sample_uids = list(loaded_images.keys())[:3]
                self._logger.debug(f"Sample series UIDs: {sample_uids}")
            else:
                self._logger.debug("Cache service has _loaded_images but it's empty")
        finally:
            cache_service._cache_mutex.unlock()

        return loaded_images

    def get_medical_image_file_path(self, medical_image: MedicalImage) -> Optional[str]:
        if not medical_image:
            return None

        if hasattr(medical_image, 'file_path') and medical_image.file_path:
            return medical_image.file_path

        if self._patient_browser and hasattr(self._patient_browser, '_cache_service'):
            cache_service = self._patient_browser._cache_service

            if hasattr(cache_service, '_loaded_images'):
                cache_service._cache_mutex.lock()
                try:
                    for series_uid, cached_image in cache_service._loaded_images.items():
                        if cached_image.series_instance_uid == medical_image.series_instance_uid:
                            if hasattr(cached_image, 'file_path'):
                                return cached_image.file_path
                finally:
                    cache_service._cache_mutex.unlock()

        return f"temp_{medical_image.series_instance_uid}.dcm"

    def get_available_cases_with_file_paths(self) -> Dict[str, Dict[str, Any]]:
        cases_with_paths = {}
        loaded_cases = self.get_loaded_cases()

        for series_uid, medical_image in loaded_cases.items():
            file_path = self.get_medical_image_file_path(medical_image)

            cases_with_paths[series_uid] = {
                'medical_image': medical_image,
                'file_path': file_path,
                'patient_id': getattr(medical_image, 'patient_id', 'Unknown'),
                'series_description': getattr(medical_image, 'series_description', 'Unknown'),
                'study_instance_uid': getattr(medical_image, 'study_instance_uid', ''),
                'series_instance_uid': medical_image.series_instance_uid,
                'detected_sequence_type': None  # Will be populated by validation service
            }

        self._logger.debug(f"Prepared {len(cases_with_paths)} cases with file path information")
        return cases_with_paths
    
    def filter_cases_by_study(self, study_uid: str) -> Dict[str, MedicalImage]:
        loaded_cases = self.get_loaded_cases()
        filtered_cases = {}
        
        for series_uid, medical_image in loaded_cases.items():
            if medical_image.study_instance_uid == study_uid:
                filtered_cases[series_uid] = medical_image
        
        self._logger.info(f"Filtered {len(filtered_cases)} cases from study {study_uid[:8]}")
        return filtered_cases
    
    def get_current_study_cases(self) -> Dict[str, MedicalImage]:
        if not self._current_medical_image:
            return {}
        
        return self.filter_cases_by_study(self._current_medical_image.study_instance_uid)
    
    def auto_populate_sequences_from_current_image(self, sequence_inputs: Dict[str, dict]) -> Dict[str, dict]:
        if not self._current_medical_image or not sequence_inputs:
            return sequence_inputs
        
        if self._current_image_file_path:
            current_image_path = self._current_image_file_path
        else:
            current_image_path = Path(f"temp_{self._current_medical_image.series_instance_uid}.dcm")
        
        for seq_name, seq_info in sequence_inputs.items():
            requirement = seq_info.get('requirement')
            
            if not requirement:
                continue
            
            if seq_name == "T2W" or len(sequence_inputs) == 1:
                seq_info['selected_path'] = current_image_path
                
                display_path = str(current_image_path.name) if len(str(current_image_path)) > 50 else str(current_image_path)
                
                if 'path_label' in seq_info:
                    seq_info['path_label'].setText(f"✓ {display_path}")
                    seq_info['path_label'].setStyleSheet(
                        "color: #2d5016; padding: 6px; border: 1px solid #81c784; border-radius: 4px; background: #f1f8e9;"
                    )
                
                self._logger.debug(f"Auto-populated {seq_name} sequence with current image: {current_image_path.name}")
            else:
                if 'path_label' in seq_info:
                    seq_info['path_label'].setText("Required - Please select file")
                    seq_info['path_label'].setStyleSheet(
                        "color: #b71c1c; padding: 6px; border: 1px solid #e57373; border-radius: 4px; background: #ffebee;"
                    )
                seq_info['selected_path'] = None
        
        return sequence_inputs
    
    def get_available_cases_for_display(self) -> List[Tuple[str, MedicalImage]]:
        loaded_cases = self.get_loaded_cases()
        
        sorted_cases = sorted(
            loaded_cases.items(),
            key=lambda x: (x[1].patient_id, getattr(x[1], 'series_description', ''))
        )
        
        self._logger.info(f"Prepared {len(sorted_cases)} cases for display")
        return sorted_cases
    
    def create_case_display_info(self, medical_image: MedicalImage) -> Dict[str, str]:
        display_info = {
            'patient_id': medical_image.patient_id or 'Unknown Patient',
            'series_description': getattr(medical_image, 'series_description', 'Unknown Series'),
            'study_date': getattr(medical_image, 'study_date', 'Unknown Date'),
            'series_uid_short': medical_image.series_instance_uid[:8] if medical_image.series_instance_uid else 'Unknown',
            'image_count': str(getattr(medical_image, 'number_of_frames', 'Unknown')),
            'modality': getattr(medical_image, 'modality', 'Unknown')
        }
        
        display_info['primary_text'] = f"{display_info['patient_id']} - {display_info['series_description']}"
        display_info['secondary_text'] = f"Study: {display_info['study_date']} | Series: {display_info['series_uid_short']} | {display_info['modality']}"
        
        return display_info
    
    def sync_viewer_with_selected_case(self, medical_image: MedicalImage) -> bool:
        try:
            self.set_current_medical_image(medical_image)
            
            self._logger.info(f"Synchronized viewer with case: {medical_image.patient_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to sync viewer with selected case: {e}")
            return False
    
    def get_case_confirmation_info(self, medical_image: MedicalImage, analysis_type_name: str) -> Dict[str, str]:
        return {
            'patient_id': medical_image.patient_id or 'Unknown Patient',
            'series_description': getattr(medical_image, 'series_description', 'N/A'),
            'study_date': getattr(medical_image, 'study_date', 'N/A'),
            'analysis_type': analysis_type_name,
            'series_uid': medical_image.series_instance_uid[:16] if medical_image.series_instance_uid else 'N/A',
            'modality': getattr(medical_image, 'modality', 'N/A')
        }
    
    def validate_current_case_for_analysis(self) -> Tuple[bool, str]:
        if not self._current_medical_image:
            return False, "No case selected from patient browser"
        
        if not self._current_medical_image.patient_id:
            return False, "Selected case has no patient ID"
        
        if not hasattr(self._current_medical_image, 'pixel_array') or self._current_medical_image.pixel_array is None:
            return False, "Selected case has no image data"
        
        return True, "Case is valid for analysis"
    
    def get_study_statistics(self, study_uid: Optional[str] = None) -> Dict[str, int]:
        if study_uid:
            cases = self.filter_cases_by_study(study_uid)
        else:
            cases = self.get_current_study_cases()
        
        modalities = {}
        series_types = {}
        
        for medical_image in cases.values():
            modality = getattr(medical_image, 'modality', 'Unknown')
            series_desc = getattr(medical_image, 'series_description', 'Unknown')
            
            modalities[modality] = modalities.get(modality, 0) + 1
            series_types[series_desc] = series_types.get(series_desc, 0) + 1
        
        return {
            'total_series': len(cases),
            'modalities': modalities,
            'series_types': series_types
        }
    
    def clear_current_case(self) -> None:
        self._current_medical_image = None
        self._current_image_file_path = None
        self._logger.info("Current case cleared")