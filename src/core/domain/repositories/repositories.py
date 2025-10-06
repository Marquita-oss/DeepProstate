from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.medical_image import MedicalImage
from ..entities.segmentation import MedicalSegmentation


class MedicalImageRepository(ABC):    
    @abstractmethod
    async def save_image(self, image: MedicalImage) -> bool:
        pass
    
    @abstractmethod
    async def find_by_study_uid(self, study_uid: str) -> List[MedicalImage]:
        pass
    
    @abstractmethod
    async def find_by_series_uid(self, series_uid: str) -> Optional[MedicalImage]:
        pass
    
    @abstractmethod
    async def find_by_patient_id(self, patient_id: str) -> List[MedicalImage]:
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[MedicalImage]:
        pass
    
    @abstractmethod
    async def delete_image(self, series_uid: str) -> bool:
        pass
    
    @abstractmethod
    async def update_image_metadata(
        self, 
        series_uid: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        pass
    
    @abstractmethod
    async def exists_image(self, series_uid: str) -> bool:
        pass


class SegmentationRepository(ABC):
    @abstractmethod
    async def save_segmentation(self, segmentation: MedicalSegmentation) -> bool:
        pass
    
    @abstractmethod
    async def find_by_image_uid(self, image_uid: str) -> List[MedicalSegmentation]:
        pass
    
    @abstractmethod
    async def find_by_segmentation_id(self, segmentation_id: str) -> Optional[MedicalSegmentation]:
        pass
    
    @abstractmethod
    async def find_by_anatomical_region(
        self, 
        region: 'AnatomicalRegion',
        image_uid: Optional[str] = None
    ) -> List[MedicalSegmentation]:
        pass
    
    @abstractmethod
    async def find_by_creator(self, creator_id: str) -> List[MedicalSegmentation]:
        pass
    
    @abstractmethod
    async def find_automatic_segmentations(
        self, 
        confidence_threshold: float = 0.5
    ) -> List[MedicalSegmentation]:
        pass
    
    @abstractmethod
    async def update_segmentation(self, segmentation: MedicalSegmentation) -> bool:
        pass
    
    @abstractmethod
    async def delete_segmentation(self, segmentation_id: str) -> bool:
        pass
    
    @abstractmethod
    async def save_segmentation_metrics(
        self, 
        segmentation_id: str, 
        metrics: 'SegmentationMetrics'
    ) -> bool:
        pass
    
    @abstractmethod
    async def get_segmentation_history(
        self, 
        segmentation_id: str
    ) -> List[Dict[str, Any]]:
        pass


class ProjectRepository(ABC):
    @abstractmethod
    async def create_project(
        self, 
        name: str, 
        description: str,
        creator_id: str
    ) -> str:
        pass
    
    @abstractmethod
    async def add_image_to_project(self, project_id: str, image_uid: str) -> bool:
        pass
    
    @abstractmethod
    async def get_project_images(self, project_id: str) -> List[str]:
        pass
    
    @abstractmethod
    async def get_project_segmentations(self, project_id: str) -> List[str]:
        pass
    
    @abstractmethod
    async def delete_project(self, project_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        pass


class ConfigurationRepository(ABC):    
    @abstractmethod
    async def save_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def save_ai_model_config(
        self, 
        model_name: str, 
        config: Dict[str, Any]
    ) -> bool:
        pass
    
    @abstractmethod
    async def get_ai_model_config(self, model_name: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def save_visualization_presets(
        self, 
        preset_name: str, 
        settings: Dict[str, Any]
    ) -> bool:
        pass
    
    @abstractmethod
    async def get_visualization_presets(self) -> Dict[str, Dict[str, Any]]:
        pass


class RepositoryError(Exception):    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class ImageNotFoundError(RepositoryError):
    pass


class SegmentationNotFoundError(RepositoryError):
    pass


class ProjectNotFoundError(RepositoryError):
    pass


class DuplicateEntityError(RepositoryError):
    pass


class InvalidQueryError(RepositoryError):
    pass