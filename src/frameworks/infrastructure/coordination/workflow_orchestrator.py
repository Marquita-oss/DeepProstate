import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt6.QtWidgets import QWidget

from .dicom_file_handler import DicomFileHandler, DicomFileInfo
from .image_validator import ImageValidator, ValidationResult
from .dialog_manager import DialogManager, DialogResult
from .progress_tracker import ProgressTracker, WorkflowStatus
from .medical_format_registry import get_medical_format_registry
from .memory_manager import create_memory_manager, MemoryManagerInterface
from .loading_interfaces import (
    LoadingProgressCallback, LoadingProgress, LoadingResult, LoadingStatus,
    SeriesLoadRequest, ThreadSafeProgressCallback
)
from .dicom_loader import DicomLoader
from .async_loader import AsyncMedicalImageLoader

from src.core.domain.entities.medical_image import MedicalImage
from src.frameworks.infrastructure.di.medical_service_container import MedicalServiceContainer


class UIProgressBridge(LoadingProgressCallback):

    def __init__(self, orchestrator: 'WorkflowOrchestrator', workflow_id: str):
        self._orchestrator = orchestrator
        self._workflow_id = workflow_id
        self._logger = logging.getLogger(f"{__name__}.UIProgressBridge")

    def on_progress_updated(self, progress: LoadingProgress) -> None:
        try:
            self._orchestrator._dialog_manager.update_progress(
                int(progress.percentage), progress.message
            )
            self._orchestrator._progress_tracker.update_workflow_progress(
                self._workflow_id, current_step=progress.current_step, message=progress.message
            )

        except Exception as e:
            self._logger.error(f"Error updating UI progress: {e}")

    def on_loading_completed(self, result: LoadingResult) -> None:
        try:
            if result.status == LoadingStatus.COMPLETED and result.data:
                if self._orchestrator._validate_medical_image_original(result.data):
                    self._orchestrator._progress_tracker.complete_workflow(self._workflow_id, success=True)
                    self._orchestrator.image_loaded.emit(result.data)
                    QTimer.singleShot(500, lambda: self._orchestrator._dialog_manager.close_progress_dialog())
                else:
                    self._handle_error("Image not valid for medical use")

            elif result.status == LoadingStatus.FAILED:
                self._handle_error(result.error_message or "Unknown loading error")
            else:
                self._handle_error(f"Unexpected loading status: {result.status}")

        except Exception as e:
            self._logger.error(f"Error in completion handler: {e}")
            self._handle_error(str(e))

    def _handle_error(self, error_message: str) -> None:
        self._logger.error(f"Loading error for workflow {self._workflow_id}: {error_message}")
        self._orchestrator._dialog_manager.close_progress_dialog()
        self._orchestrator._progress_tracker.complete_workflow(
            self._workflow_id, success=False, error_message=error_message
        )
        if "not fully implemented" not in error_message.lower() and "series uid loading" not in error_message.lower():
            self._orchestrator._dialog_manager.show_error_message("Loading Error", error_message)

class WorkflowOrchestrator(QObject):
    workflow_started = pyqtSignal(str)
    workflow_completed = pyqtSignal(str, bool, object)
    workflow_error = pyqtSignal(str, str)
    image_loaded = pyqtSignal(object)
    ai_analysis_completed = pyqtSignal(list)
    workflow_progress = pyqtSignal(str, int, str)
    medical_validation_required = pyqtSignal(str, dict)
    
    def __init__(self, services: MedicalServiceContainer, parent_window: Optional[QWidget] = None):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._services = services

        self._file_handler = DicomFileHandler()
        self._validator = ImageValidator()
        self._dialog_manager = DialogManager(parent_window)
        self._progress_tracker = ProgressTracker()

        self._memory_manager = create_memory_manager()
        self._dicom_loader = DicomLoader(self._memory_manager)
        self._async_loader = AsyncMedicalImageLoader(self._dicom_loader, max_concurrent_loads=2)

        self._active_loads: Dict[str, str] = {}
        self._setup_component_connections()


    def cleanup(self) -> None:
        try:
            self._async_loader.shutdown(wait=True)
        except Exception as e:
            self._logger.error(f"Error during WorkflowOrchestrator cleanup: {e}")
    
    def set_parent_window(self, parent_window: QWidget) -> None:
        self._dialog_manager.set_parent_window(parent_window)
    
    def _setup_component_connections(self) -> None:
        self._progress_tracker.workflow_completed.connect(
            lambda workflow_id, success: self.workflow_completed.emit(workflow_id, success, None)
        )
        self._progress_tracker.workflow_progress_updated.connect(
            lambda workflow_id, percentage, message: self.workflow_progress.emit(workflow_id, percentage, message)
        )
    
    async def load_medical_image_interactive(self) -> Optional[MedicalImage]:
        workflow_id = self._progress_tracker.create_workflow(
            "Load Medical Image", 
            total_steps=4
        )
        
        try:
            self.workflow_started.emit(workflow_id)
            self._progress_tracker.start_workflow(workflow_id)

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=1, message="Getting user selection..."
            )
            
            selection_result = self._dialog_manager.show_file_selection_dialog()
            if not selection_result.accepted:
                self._progress_tracker.cancel_workflow(workflow_id, "User cancelled selection")
                return None

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=2, message="Getting file path..."
            )
            
            if selection_result.data == "single_file":
                path_result = self._dialog_manager.show_single_file_dialog()
            else:  
                path_result = self._dialog_manager.show_folder_dialog()
            
            if not path_result.accepted:
                self._progress_tracker.cancel_workflow(workflow_id, "User cancelled path selection")
                return None

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=3, message="Loading and validating image..."
            )
            
            if selection_result.data == "single_file":
                medical_image = await self._load_single_file(path_result.data, workflow_id)
            else:
                medical_image = await self._load_multi_format_folder(path_result.data, workflow_id)
            
            if medical_image is None:
                self._progress_tracker.complete_workflow(
                    workflow_id, success=False, error_message="Failed to load image"
                )
                return None

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=4, message="Final validation..."
            )
            
            validation_result = self._validator.validate_medical_image_entity(medical_image)
            if not validation_result.is_valid:
                error_msg = f"Validation failed: {'; '.join(validation_result.error_messages)}"
                self._progress_tracker.complete_workflow(workflow_id, success=False, error_message=error_msg)
                self._dialog_manager.show_error_message("Validation Error", error_msg)
                return None

            if validation_result.warnings:
                warning_msg = "Image loaded with warnings:\n" + "\n".join(validation_result.warnings)
                self._dialog_manager.show_warning_message("Image Warnings", warning_msg)
            
            self._progress_tracker.complete_workflow(workflow_id, success=True)
            self.workflow_completed.emit(workflow_id, True, medical_image)
            self.image_loaded.emit(medical_image)
            
            return medical_image
            
        except Exception as e:
            error_message = f"Workflow error: {str(e)}"
            self._logger.error(f"Error in load medical image workflow: {e}")
            self._progress_tracker.complete_workflow(workflow_id, success=False, error_message=error_message)
            self.workflow_error.emit(workflow_id, error_message)
            self._dialog_manager.show_error_message("Error", error_message)
            return None
    
    async def _load_single_file(self, file_path: str, workflow_id: str) -> Optional[MedicalImage]:
        try:
            registry = get_medical_format_registry()
            is_valid, error_msg, format_name = registry.validate_file(Path(file_path))
            
            if not is_valid:
                raise ValueError(f"Invalid {format_name or 'medical image'} format: {error_msg}")

            medical_image = registry.load_medical_image(Path(file_path))
            if not medical_image:
                raise ValueError(f"Failed to load {format_name} image")
            
            return medical_image

        except Exception as e:
            from src.core.domain.exceptions.medical_exceptions import MaskFileDetectedError
            if isinstance(e, MaskFileDetectedError):
                return None
            else:
                self._logger.error(f"Error loading single file {file_path}: {e}")
                return None
    
    async def _load_dicom_folder(self, folder_path: str, workflow_id: str) -> Optional[MedicalImage]:
        try:
            if not self._file_handler.validate_dicom_directory(folder_path):
                raise ValueError("Directory does not contain valid DICOM files")

            dicom_files, series_groups = self._file_handler.analyze_folder_content(folder_path)
            
            if not dicom_files:
                raise ValueError("No valid DICOM files found in directory")

            validation_result = self._validator.validate_dicom_series(dicom_files)
            if not validation_result.is_valid:
                error_msg = "; ".join(validation_result.error_messages)
                raise ValueError(f"Series validation failed: {error_msg}")

            first_series_uid = next(iter(series_groups.keys()))
            series_files = series_groups[first_series_uid]

            image_repository = self._services.image_repository
            file_paths = [str(file_info.file_path) for file_info in series_files]
            
            medical_image = await image_repository.load_series_from_paths(file_paths)
            
            return medical_image
            
        except Exception as e:
            self._logger.error(f"Error loading DICOM folder {folder_path}: {e}")
            return None
    
    async def _load_multi_format_folder(self, folder_path: str, workflow_id: str) -> Optional[MedicalImage]:
        try:
            from .medical_format_registry import get_medical_format_registry

            registry = get_medical_format_registry()
            medical_image = registry.load_folder_as_series(Path(folder_path))
            
            if not medical_image:
                self._logger.error(f"No supported medical images found in folder: {folder_path}")
                return None
            
            return medical_image
            
        except Exception as e:
            self._logger.error(f"Error loading multi-format folder {folder_path}: {e}")
            return None
    
    def get_workflow_progress(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        progress = self._progress_tracker.get_workflow_progress(workflow_id)
        if not progress:
            return None
        
        return {
            'workflow_id': progress.workflow_id,
            'workflow_name': progress.workflow_name,
            'status': progress.status.value,
            'progress_percentage': progress.progress_percentage,
            'current_message': progress.current_message,
            'start_time': progress.start_time.isoformat() if progress.start_time else None,
            'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None
        }
    
    def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> bool:
        return self._progress_tracker.cancel_workflow(workflow_id, reason)
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        active_workflows = self._progress_tracker.get_active_workflows()
        
        return [
            {
                'workflow_id': workflow.workflow_id,
                'workflow_name': workflow.workflow_name,
                'status': workflow.status.value,
                'progress_percentage': workflow.progress_percentage,
                'current_message': workflow.current_message
            }
            for workflow in active_workflows
        ]
    
    def cleanup_old_workflows(self, max_age_hours: int = 24) -> int:
        return self._progress_tracker.cleanup_completed_workflows(max_age_hours)

    def start_image_loading_workflow(self, series_uid: Optional[str] = None, file_path: Optional[str] = None) -> str:
        try:
            workflow_id = self._generate_workflow_id("image_loading")

            if not file_path and not series_uid:
                file_path = self._prompt_for_dicom_file()
                if not file_path:
                    return workflow_id

            if series_uid:
                QTimer.singleShot(0, lambda: self._start_loading_series(workflow_id, series_uid))
            elif file_path:
                QTimer.singleShot(0, lambda: self._execute_image_loading_workflow_original(workflow_id, file_path))

            return workflow_id

        except Exception as e:
            self._logger.error(f"Error starting image loading workflow: {e}")
            return "error_workflow"

    def _start_loading_series(self, workflow_id: str, series_uid: str) -> None:
        try:
            if not self._memory_manager.check_availability(estimated_size_mb=512.0):
                freed_mb = self._memory_manager.force_cleanup()

                if not self._memory_manager.check_availability(estimated_size_mb=512.0):
                    self._logger.warning("Proceeding with load despite low memory")

            progress_workflow_id = self._progress_tracker.create_workflow(
                f"Loading DICOM Series {series_uid[:8]}",
                total_steps=3
            )
            self._progress_tracker.start_workflow(progress_workflow_id)

            self._dialog_manager.show_progress_dialog(
                title="Processing... -- Medical Workstation",
                message="Checking memory and initializing...",
                cancelable=False
            )

            progress_callback = UIProgressBridge(self, progress_workflow_id)
            load_request = SeriesLoadRequest(series_uid=series_uid, estimated_size_mb=512.0)
            operation_id = self._async_loader.start_async_load(load_request, progress_callback)
            self._active_loads[workflow_id] = operation_id
            QTimer.singleShot(2000, lambda: self._memory_manager.force_cleanup())

        except Exception as e:
            self._logger.error(f"Error starting clean loading: {e}")
            self._dialog_manager.close_progress_dialog()
            self._progress_tracker.complete_workflow(workflow_id, success=False, error_message=str(e))
    
    def load_cached_medical_image(self, medical_image) -> None:
        try:
            self.image_loaded.emit(medical_image)
            
        except Exception as e:
            self._logger.error(f"Error loading cached medical image: {e}")
    
    def start_ai_analysis_workflow(self, medical_image, analysis_type: str = "full") -> None:
        try:
            workflow_id = self._progress_tracker.create_workflow(
                f"AI Analysis ({analysis_type})", 
                total_steps=3
            )
            self._progress_tracker.start_workflow(workflow_id)

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=1, message="Preparing image for analysis..."
            )

            self._progress_tracker.update_workflow_progress(
                workflow_id, current_step=2, message="Running AI analysis..."
            )

            self._progress_tracker.complete_workflow(workflow_id, success=True)
            self.ai_analysis_completed.emit([])
            
        except Exception as e:
            self._logger.error(f"Error starting AI analysis workflow: {e}")
    
    def _analyze_folder_content(self, folder_path: str):
        try:
            dicom_files, series_groups = self._file_handler.analyze_folder_content(folder_path)

            return {
                'total_files': len(dicom_files),
                'series_count': len(series_groups),
                'series_groups': series_groups
            }
            
        except Exception as e:
            self._logger.error(f"Error analyzing folder content: {e}")
            return {'total_files': 0, 'series_count': 0, 'series_groups': {}}
    
    def _load_specific_modality_from_folder(self, folder_path: str, modality: str):
        try:
            dicom_files, series_groups = self._file_handler.analyze_folder_content(folder_path)
            
            if series_groups:
                first_series = next(iter(series_groups.values()))
                file_paths = [str(f.file_path) for f in first_series]
                image_repository = self._services.image_repository
                return None
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error loading specific modality: {e}")
            return None
    
    def _execute_image_loading_workflow_original(self, workflow_id: str, file_path: str) -> None:
        try:
            self._dialog_manager.show_progress_dialog(
                title="Processing... -- Medical Workstation",
                message="Validating medical image format...",
                cancelable=False
            )

            self._dialog_manager.update_progress(15, "Validating medical image format...")

            file_path_obj = Path(file_path)
            if file_path_obj.is_file() and not self._validate_medical_image_format(file_path):
                self._dialog_manager.close_progress_dialog()
                self._logger.error("Invalid medical image format")
                return
            elif file_path_obj.is_dir() and not self._validate_medical_image_directory(file_path):
                self._dialog_manager.close_progress_dialog()
                self._logger.error("Directory contains no valid medical image files")
                return

            QTimer.singleShot(1000, lambda: self._run_async_completion(workflow_id, file_path))
            
        except Exception as e:
            self._logger.error(f"Error in image loading workflow: {e}")
            self._dialog_manager.close_progress_dialog()
    
    def _run_async_completion(self, workflow_id: str, file_path: str) -> None:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._complete_image_loading_original(workflow_id, file_path))
            finally:
                loop.close()
        except Exception as e:
            self._logger.error(f"Error in async completion: {e}")
            self._dialog_manager.close_progress_dialog()
    
    async def _complete_image_loading_original(self, workflow_id: str, file_path: str) -> None:
        try:
            self._dialog_manager.update_progress(40, "Loading medical image...")

            file_path_obj = Path(file_path)
            if file_path_obj.is_file():
                image = await self._load_single_file(file_path, workflow_id)
            else:
                image = await self._load_multi_format_folder(file_path, workflow_id)
            if not image:
                self._dialog_manager.close_progress_dialog()
                return

            self._dialog_manager.update_progress(60, "Validating medical integrity...")
            
            if not self._validate_medical_image_original(image):
                self._dialog_manager.close_progress_dialog()
                self._logger.error("Image not valid for medical use")
                return

            self._dialog_manager.update_progress(80, "Preparing for display...")
            self._dialog_manager.update_progress(100, "Image loaded successfully")
            self.image_loaded.emit(image)
            QTimer.singleShot(500, lambda: self._dialog_manager.close_progress_dialog())

        except Exception as e:
            self._logger.error(f"Error completing image loading: {e}")
            self._dialog_manager.close_progress_dialog()
    
    def _prompt_for_dicom_file(self) -> Optional[str]:
        try:
            selection_result = self._dialog_manager.show_file_selection_dialog()
            if not selection_result.accepted:
                return None

            if selection_result.data == "single_file":
                path_result = self._dialog_manager.show_single_file_dialog()
            else:  # folder
                path_result = self._dialog_manager.show_folder_dialog()
            
            if path_result.accepted:
                return path_result.data
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error in file selection: {e}")
            return None
    
    def _validate_medical_image_format(self, file_path: str) -> bool:
        try:
            registry = get_medical_format_registry()
            is_valid, error_msg, format_name = registry.validate_file(Path(file_path))
            return is_valid
            
        except Exception as e:
            return False
    
    def _validate_medical_image_directory(self, directory_path: str) -> bool:
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return False

            registry = get_medical_format_registry()
            supported_extensions = set(registry.get_supported_extensions())
            supported_extensions.add('')

            for file_path in directory.iterdir():
                if file_path.is_file():
                    if (file_path.suffix.lower() in supported_extensions and 
                        self._validate_medical_image_format(str(file_path))):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_and_load_dicom_series_original(self, file_path: str) -> Optional[MedicalImage]:
        try:
            file_path_obj = Path(file_path)

            if file_path_obj.is_dir():
                return self._load_folder_with_repository(file_path)
            else:
                return self._load_single_file_with_repository(file_path)
                
        except Exception as e:
            self._logger.error(f"Error detecting and loading DICOM series: {e}")
            return None
    
    def _load_single_file_with_repository(self, file_path: str) -> Optional[MedicalImage]:
        try:
            import asyncio

            image_repository = self._services.image_repository
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                dicom_path = Path(file_path)
                medical_image = loop.run_until_complete(
                    image_repository._load_dicom_as_image(dicom_path)
                )
                return medical_image
            finally:
                loop.close()
                
        except Exception as e:
            self._logger.error(f"Error loading single file with repository: {e}")
            return None
    
    def _load_folder_with_repository(self, folder_path: str) -> Optional[MedicalImage]:
        try:
            import asyncio

            folder = Path(folder_path)
            dicom_files = []
            
            for file_path in folder.iterdir():
                if (file_path.is_file() and 
                    self._validate_medical_image_format(str(file_path))):
                    dicom_files.append(str(file_path))
            
            if not dicom_files:
                return None

            image_repository = self._services.image_repository
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if dicom_files and len(dicom_files) > 1:
                    medical_image = self._load_dicom_series_as_volume_simple(dicom_files)
                elif dicom_files:
                    dicom_path = Path(dicom_files[0])
                    medical_image = loop.run_until_complete(
                        image_repository._load_dicom_as_image(dicom_path)
                    )
                else:
                    medical_image = None
                return medical_image
            finally:
                loop.close()
                
        except Exception as e:
            self._logger.error(f"Error loading folder with repository: {e}")
            return None
    
    def _validate_medical_image_original(self, image: MedicalImage) -> bool:
        try:
            if not image or image.image_data is None:
                return False
            
            if image.image_data.size == 0:
                return False
            
            if len(image.image_data.shape) < 2:
                return False

            if len(image.image_data.shape) == 2:
                if any(dim < 32 for dim in image.image_data.shape):
                    return False
            elif len(image.image_data.shape) >= 3:
                if len(image.image_data.shape) == 3:
                    dims = list(image.image_data.shape)
                    dims.sort(reverse=True)
                    largest_two = dims[:2]
                    if any(dim < 32 for dim in largest_two):
                        return False
                else:
                    max_dims = sorted(image.image_data.shape, reverse=True)[:2]
                    if any(dim < 32 for dim in max_dims):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _generate_workflow_id(self, workflow_type: str) -> str:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{workflow_type}_{timestamp}"
    
    def _load_dicom_series_as_volume_simple(self, dicom_files: List[str]) -> Optional[MedicalImage]:
        try:
            import SimpleITK as sitk
            import numpy as np

            if len(dicom_files) == 1:
                return self._load_single_file_with_repository(dicom_files[0])

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)

            sitk_image = reader.Execute()
            volume_array = sitk.GetArrayFromImage(sitk_image)

            first_file_medical_image = self._load_single_file_with_repository(dicom_files[0])
            
            if first_file_medical_image:
                first_file_medical_image._replace_image_data(volume_array)
                first_file_medical_image._dicom_metadata['is_series'] = True
                first_file_medical_image._dicom_metadata['series_file_count'] = len(dicom_files)
                slice_counts = first_file_medical_image.get_total_slices_all_planes()
                return first_file_medical_image
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error loading series as 3D volume: {e}")
            if dicom_files:
                return self._load_single_file_with_repository(dicom_files[0])
            return None