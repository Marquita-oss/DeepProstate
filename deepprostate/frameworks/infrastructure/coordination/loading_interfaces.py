from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class LoadingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoadingProgress:
    current_step: int
    total_steps: int
    message: str
    percentage: float

    @classmethod
    def create(cls, step: int, total: int, message: str) -> 'LoadingProgress':
        percentage = (step / total * 100) if total > 0 else 0.0
        return cls(step, total, message, percentage)


@dataclass
class LoadingResult:
    status: LoadingStatus
    data: Optional[Any] = None
    error_message: Optional[str] = None


class LoadingProgressCallback(ABC):
    @abstractmethod
    def on_progress_updated(self, progress: LoadingProgress) -> None:
        pass

    @abstractmethod
    def on_loading_completed(self, result: LoadingResult) -> None:
        pass


class MedicalImageLoaderInterface(ABC):

    @abstractmethod
    def load_by_series_uid(
        self,
        series_uid: str,
        progress_callback: Optional[LoadingProgressCallback] = None
    ) -> LoadingResult:
        pass

    @abstractmethod
    def load_from_path(
        self,
        file_path: str,
        progress_callback: Optional[LoadingProgressCallback] = None
    ) -> LoadingResult:
        pass


class AsyncLoaderInterface(ABC):
    @abstractmethod
    def start_async_load(
        self,
        loader_request: Any,
        progress_callback: LoadingProgressCallback
    ) -> str:
        pass

    @abstractmethod
    def cancel_load(self, operation_id: str) -> bool:
        pass

    @abstractmethod
    def is_loading(self, operation_id: str) -> bool:
        pass


@dataclass
class SeriesLoadRequest:
    series_uid: str
    estimated_size_mb: float = 512.0
    priority: int = 1


@dataclass
class FileLoadRequest:
    file_path: str
    estimated_size_mb: float = 100.0
    priority: int = 1


class ThreadSafeProgressCallback(LoadingProgressCallback):
    def __init__(
        self,
        progress_func: Callable[[LoadingProgress], None],
        completion_func: Callable[[LoadingResult], None]
    ):
        self._progress_func = progress_func
        self._completion_func = completion_func

    def on_progress_updated(self, progress: LoadingProgress) -> None:
        try:
            self._progress_func(progress)
        except Exception as e:
            # Log error but don't crash the loading operation
            import logging
            logging.getLogger(__name__).error(f"Error in progress callback: {e}")

    def on_loading_completed(self, result: LoadingResult) -> None:
        try:
            self._completion_func(result)
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.getLogger(__name__).error(f"Error in completion callback: {e}")