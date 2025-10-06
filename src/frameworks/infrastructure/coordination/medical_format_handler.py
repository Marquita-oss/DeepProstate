from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

from src.core.domain.entities.medical_image import MedicalImage, ImageSpacing, ImageModalityType


@dataclass
class FormatCapabilities:
    can_read: bool = True
    can_write: bool = False
    supports_metadata: bool = True
    supports_series: bool = False
    supports_3d: bool = True
    typical_extensions: List[str] = None
    
    def __post_init__(self):
        if self.typical_extensions is None:
            self.typical_extensions = []


@dataclass
class LoadedImageData:
    image_array: 'np.ndarray'
    spacing: ImageSpacing
    modality: ImageModalityType
    metadata: Dict[str, Any]
    patient_id: str = "UNKNOWN"
    study_uid: str = ""
    series_uid: str = ""
    acquisition_date: Optional[datetime] = None


class MedicalFormatHandler(ABC):
    @abstractmethod
    def get_format_name(self) -> str:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> FormatCapabilities:
        pass
    
    @abstractmethod
    def can_handle_file(self, file_path: Path) -> bool:
        pass
    
    @abstractmethod
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        pass
    
    @abstractmethod
    def load_image(self, file_path: Path) -> Optional[LoadedImageData]:
        pass
    
    def create_medical_image(self, file_path: Path) -> Optional[MedicalImage]:
        loaded_data = self.load_image(file_path)
        if not loaded_data:
            return None
        
        try:
            return MedicalImage(
                image_data=loaded_data.image_array,
                spacing=loaded_data.spacing,
                modality=loaded_data.modality,
                patient_id=loaded_data.patient_id,
                study_instance_uid=loaded_data.study_uid,
                series_instance_uid=loaded_data.series_uid,
                acquisition_date=loaded_data.acquisition_date or datetime.now(),
                dicom_metadata=loaded_data.metadata
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to create MedicalImage: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        return self.get_capabilities().typical_extensions


class FormatDetectionService:    
    def __init__(self):
        self._handlers: List[MedicalFormatHandler] = []
        self._logger = logging.getLogger(__name__)
    
    def register_handler(self, handler: MedicalFormatHandler) -> None:
        self._handlers.append(handler)
        self._logger.info(f"Registered {handler.get_format_name()} format handler")
    
    def detect_format(self, file_path: Path) -> Optional[MedicalFormatHandler]:
        for handler in self._handlers:
            if handler.can_handle_file(file_path):
                self._logger.debug(f"Detected {handler.get_format_name()} format for {file_path.name}")
                return handler
        
        self._logger.warning(f"No handler found for file: {file_path}")
        return None
    
    def get_all_supported_extensions(self) -> List[str]:
        extensions = []
        for handler in self._handlers:
            extensions.extend(handler.get_supported_extensions())
        return list(set(extensions))  # Remove duplicates
    
    def get_format_capabilities(self) -> Dict[str, FormatCapabilities]:
        return {
            handler.get_format_name(): handler.get_capabilities()
            for handler in self._handlers
        }