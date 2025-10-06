import logging
from typing import Optional
from pathlib import Path

from .loading_interfaces import (
    MedicalImageLoaderInterface,
    LoadingProgressCallback,
    LoadingProgress,
    LoadingResult,
    LoadingStatus
)
from .memory_manager import MemoryManagerInterface
from .medical_format_registry import get_medical_format_registry


class DicomLoader(MedicalImageLoaderInterface):
    def __init__(self, memory_manager: MemoryManagerInterface):
        self._memory_manager = memory_manager
        self._format_registry = get_medical_format_registry()
        self._logger = logging.getLogger(__name__)

    def load_by_series_uid(
        self,
        series_uid: str,
        progress_callback: Optional[LoadingProgressCallback] = None
    ) -> LoadingResult:
        try:
            self._memory_manager.log_memory_status("Before DICOM loading")
            self._memory_manager.force_cleanup()
            medical_image = self._format_registry.load_medical_image_by_series_uid(series_uid)

            if medical_image is None:
                error_msg = f"Failed to load series: {series_uid}"
                result = LoadingResult(LoadingStatus.FAILED, error_message=error_msg)
                if progress_callback:
                    progress_callback.on_loading_completed(result)
                return result

            self._memory_manager.log_memory_status("After DICOM loading")
            result = LoadingResult(LoadingStatus.COMPLETED, data=medical_image)

            if progress_callback:
                progress_callback.on_loading_completed(result)

            return result

        except Exception as e:
            error_msg = f"Loading error: {str(e)}"
            self._logger.error(f"Error in DICOM loader: {e}")

            result = LoadingResult(LoadingStatus.FAILED, error_message=error_msg)
            if progress_callback:
                progress_callback.on_loading_completed(result)

            return result

    def load_from_path(
        self,
        file_path: str,
        progress_callback: Optional[LoadingProgressCallback] = None
    ) -> LoadingResult:
        try:
            self._memory_manager.log_memory_status("Before file loading")
            self._memory_manager.force_cleanup()
            medical_image = self._format_registry.load_medical_image(Path(file_path))

            if medical_image is None:
                error_msg = f"Failed to load file: {file_path}"
                result = LoadingResult(LoadingStatus.FAILED, error_message=error_msg)
                if progress_callback:
                    progress_callback.on_loading_completed(result)
                return result

            self._memory_manager.log_memory_status("After file loading")
            result = LoadingResult(LoadingStatus.COMPLETED, data=medical_image)
            
            if progress_callback:
                progress_callback.on_loading_completed(result)

            return result

        except Exception as e:
            error_msg = f"Loading error: {str(e)}"
            self._logger.error(f"Error in file loader: {e}")

            result = LoadingResult(LoadingStatus.FAILED, error_message=error_msg)
            if progress_callback:
                progress_callback.on_loading_completed(result)

            return result


class NullProgressCallback(LoadingProgressCallback):
    def on_progress_updated(self, progress: LoadingProgress) -> None:
        pass

    def on_loading_completed(self, result: LoadingResult) -> None:
        pass