from typing import Dict, Any, Optional, TypeVar, Type
from pathlib import Path
import logging

from deepprostate.core.domain.repositories.repositories import (
    MedicalImageRepository, SegmentationRepository
)
from deepprostate.use_cases.application.services.image_services import (
    ImageLoadingService, ImageVisualizationService
)
from deepprostate.use_cases.application.services.segmentation_services import SegmentationEditingService
from deepprostate.use_cases.application.services.ai_segmentation_service import AISegmentationService
from deepprostate.use_cases.application.services.ai_analysis_orchestrator import AIAnalysisOrchestrator

from deepprostate.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository
from deepprostate.frameworks.infrastructure.visualization.vtk_renderer import MedicalVTKRenderer

from deepprostate.core.domain.services.dynamic_model_config_service import DynamicModelConfigService

T = TypeVar('T')


class MedicalServiceContainer:

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._services: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

        self._dynamic_model_config = DynamicModelConfigService()
        self._services['dynamic_model_config'] = self._dynamic_model_config
        self._validate_medical_configuration()
        self._initialize_core_services()
        self._logger.info("Medical services container initialized successfully")
    
    def _validate_medical_configuration(self) -> None:
        required_keys = [
            'storage_path',
            'ai_config', 
            'visualization_config',
            'logging_config'
        ]
        
        missing_keys = [key for key in required_keys if key not in self._config]
        if missing_keys:
            raise ValueError(
                f"Incomplete medical configuration. Missing: {missing_keys}"
            )
        
        storage_path = Path(self._config['storage_path'])
        # Don't create storage directory automatically - only create when actually needed
        # if not storage_path.exists():
        #     storage_path.mkdir(parents=True, exist_ok=True)
        #     self._logger.info(f"Created medical storage directory: {storage_path}")
    
    def _initialize_core_services(self) -> None:
        try:
            self._initialize_repositories()
            self._initialize_application_services()
            self._initialize_infrastructure_services()

        except Exception as e:
            self._logger.error(f"Error initializing medical services: {e}")
            raise RuntimeError(f"Failed to initialize medical services: {e}")
    
    def _initialize_repositories(self) -> None:
        storage_path = self._config['storage_path']
        self._services['image_repository'] = DICOMImageRepository(storage_path)
        self._logger.debug("Medical repositories initialized")
    
    def _initialize_application_services(self) -> None:
        """Initialize medical business logic services."""
        image_repo = self._services['image_repository']
        ai_config = self._config['ai_config']
        self._services['image_loading_service'] = ImageLoadingService(image_repo)
        self._services['image_visualization_service'] = ImageVisualizationService()
        self._services['ai_segmentation_service'] = AISegmentationService(
            image_repo,  
            ai_config,
            self._dynamic_model_config  
        )
        self._services['segmentation_editing_service'] = SegmentationEditingService(
            image_repo 
        )

        from deepprostate.frameworks.infrastructure.coordination.medical_format_registry import MedicalFormatRegistry
        format_registry = MedicalFormatRegistry()
        # Don't pass temp_storage_path - let orchestrator use /tmp/deep_prostate with auto-cleanup
        # temp_storage_path = Path(self._config['storage_path']) / "temp"

        self._services['ai_analysis_orchestrator'] = AIAnalysisOrchestrator(
            segmentation_service=self._services['ai_segmentation_service'],
            format_registry=format_registry,
            temp_storage_path=None  # Use system temp with auto-cleanup
        )
        
        self._logger.debug("Medical application services initialized")

    def _initialize_infrastructure_services(self) -> None:
        viz_config = self._config['visualization_config']
        self._services['vtk_renderer'] = MedicalVTKRenderer()
        self._logger.debug("Infrastructure services initialized")
    
    @property
    def image_repository(self) -> MedicalImageRepository:
        return self._services['image_repository']

    @property
    def image_loading_service(self) -> ImageLoadingService:
        return self._services['image_loading_service']

    @property
    def image_visualization_service(self) -> ImageVisualizationService:
        return self._services['image_visualization_service']

    @property
    def ai_segmentation_service(self) -> AISegmentationService:
        return self._services['ai_segmentation_service']

    @property
    def segmentation_editing_service(self) -> SegmentationEditingService:
        return self._services['segmentation_editing_service']

    @property
    def vtk_renderer(self) -> MedicalVTKRenderer:
        return self._services['vtk_renderer']

    @property
    def ai_analysis_orchestrator(self) -> AIAnalysisOrchestrator:
        return self._services['ai_analysis_orchestrator']

    @property
    def dynamic_model_config_service(self) -> DynamicModelConfigService:
        return self._services['dynamic_model_config']
    
    def get_service(self, service_type: Type[T]) -> T:
        for service in self._services.values():
            if isinstance(service, service_type):
                return service
        
        raise ValueError(f"Servicio {service_type.__name__} no disponible")
    
    def configure_for_testing(self, mock_services: Dict[str, Any]) -> None:
        for service_name, mock_service in mock_services.items():
            if service_name in self._services:
                self._services[service_name] = mock_service
                self._logger.debug(f"Servicio {service_name} configurado for testing")
    
    def shutdown(self) -> None:
        self._logger.info("Iniciando cierre ordenado of servicios médicos")
        
        # Cerrar servicios in ordin inverso al of inicialización
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    service.shutdown()
                    self._logger.debug(f"Servicio {service_name} cerrado correctamente")
            except Exception as e:
                self._logger.error(f"Error cerrando servicio {service_name}: {e}")
        
        self._services.clear()
        self._logger.info("Medical services closed correctamente")


def create_medical_service_container(config_path: Optional[str] = None) -> MedicalServiceContainer:
    from deepprostate.frameworks.infrastructure.utils.secure_config import SecureConfigManager
    
    try:
        config_manager = SecureConfigManager()
        secure_config = config_manager.load_configuration(config_path)
        
        logging.info("Medical service container created with secure configuration")
        return MedicalServiceContainer(secure_config)
        
    except Exception as e:
        logging.error(f"Error creating secure medical service container: {e}")
        fallback_config = {
            'storage_path': './medical_data_fallback',
            'ai_config': {
                'model_path': None, 
                'confidence_threshold': 0.7,
                'preprocessing_params': {
                    'normalize': True,
                    'resample': True,
                    'target_spacing': [1.0, 1.0, 3.0]
                }
            },
            'visualization_config': {
                'default_window_width': 400,
                'default_window_level': 40,
                'enable_gpu_rendering': True, 
                'max_texture_memory_mb': 256
            },
            'logging_config': {
                'level': 'INFO',
                'medical_audit': True,
                'hipaa_compliant': True
            }
        }
        
        logging.warning("Using fallback configuration due to security configuration error")
        return MedicalServiceContainer(fallback_config)