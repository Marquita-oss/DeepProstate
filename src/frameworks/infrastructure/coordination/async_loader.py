import threading
import uuid
import logging
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from .loading_interfaces import (
    AsyncLoaderInterface,
    LoadingProgressCallback,
    MedicalImageLoaderInterface,
    SeriesLoadRequest,
    FileLoadRequest,
    LoadingResult,
    LoadingStatus
)


class AsyncMedicalImageLoader(AsyncLoaderInterface):
    def __init__(
        self,
        image_loader: MedicalImageLoaderInterface,
        max_concurrent_loads: int = 2
    ):
        self._image_loader = image_loader
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_loads, thread_name_prefix="DicomLoader")
        self._active_operations: Dict[str, Future] = {}
        self._operation_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def start_async_load(
        self,
        loader_request,
        progress_callback: LoadingProgressCallback
    ) -> str:
        operation_id = str(uuid.uuid4())

        try:
            if isinstance(loader_request, SeriesLoadRequest):
                load_func = lambda: self._image_loader.load_by_series_uid(
                    loader_request.series_uid, progress_callback
                )
                self._logger.info(f"Starting async series load: {loader_request.series_uid[:8]}...")
            elif isinstance(loader_request, FileLoadRequest):
                load_func = lambda: self._image_loader.load_from_path(
                    loader_request.file_path, progress_callback
                )
                self._logger.info(f"Starting async file load: {loader_request.file_path}")
            else:
                raise ValueError(f"Unsupported loader request type: {type(loader_request)}")

            future = self._executor.submit(self._execute_with_error_handling, load_func, progress_callback)

            with self._operation_lock:
                self._active_operations[operation_id] = future

            future.add_done_callback(lambda f: self._cleanup_operation(operation_id))

            self._logger.debug(f"Async load operation started: {operation_id}")
            return operation_id

        except Exception as e:
            self._logger.error(f"Error starting async load: {e}")
            error_result = LoadingResult(LoadingStatus.FAILED, error_message=str(e))
            progress_callback.on_loading_completed(error_result)
            return operation_id

    def cancel_load(self, operation_id: str) -> bool:
        try:
            with self._operation_lock:
                future = self._active_operations.get(operation_id)

            if future and not future.done():
                cancelled = future.cancel()
                if cancelled:
                    self._logger.info(f"Cancelled loading operation: {operation_id}")
                else:
                    self._logger.info(f"Could not cancel operation (already running): {operation_id}")
                return cancelled
            else:
                self._logger.debug(f"Operation not found or already completed: {operation_id}")
                return False

        except Exception as e:
            self._logger.error(f"Error cancelling operation {operation_id}: {e}")
            return False

    def is_loading(self, operation_id: str) -> bool:
        try:
            with self._operation_lock:
                future = self._active_operations.get(operation_id)

            return future is not None and not future.done()

        except Exception as e:
            self._logger.error(f"Error checking operation status {operation_id}: {e}")
            return False

    def get_active_operation_count(self) -> int:
        with self._operation_lock:
            return len([f for f in self._active_operations.values() if not f.done()])

    def shutdown(self, wait: bool = True) -> None:
        try:
            self._logger.info("Shutting down AsyncMedicalImageLoader...")

            # Cancel all pending operations
            with self._operation_lock:
                for operation_id, future in self._active_operations.items():
                    if not future.done():
                        future.cancel()
                        self._logger.debug(f"Cancelled operation during shutdown: {operation_id}")

            # Shutdown thread pool
            self._executor.shutdown(wait=wait)
            self._logger.info("AsyncMedicalImageLoader shutdown complete")

        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")

    def _execute_with_error_handling(
        self,
        load_func,
        progress_callback: LoadingProgressCallback
    ) -> LoadingResult:
        try:
            return load_func()
        except Exception as e:
            self._logger.error(f"Error in async loading execution: {e}")
            error_result = LoadingResult(LoadingStatus.FAILED, error_message=str(e))
            try:
                progress_callback.on_loading_completed(error_result)
            except Exception as callback_error:
                self._logger.error(f"Error in progress callback: {callback_error}")
            return error_result

    def _cleanup_operation(self, operation_id: str) -> None:
        try:
            with self._operation_lock:
                if operation_id in self._active_operations:
                    del self._active_operations[operation_id]
                    self._logger.debug(f"Cleaned up operation: {operation_id}")
        except Exception as e:
            self._logger.error(f"Error cleaning up operation {operation_id}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)