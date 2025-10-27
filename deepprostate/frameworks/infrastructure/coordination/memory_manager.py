import gc
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryInfo:
    def __init__(self, current_mb: float, available_mb: float, total_mb: float):
        self.current_mb = current_mb
        self.available_mb = available_mb
        self.total_mb = total_mb
        self.usage_percentage = (current_mb / total_mb) * 100 if total_mb > 0 else 0.0


class MemoryManagerInterface(ABC):
    @abstractmethod
    def get_memory_info(self) -> MemoryInfo:
        pass

    @abstractmethod
    def check_availability(self, estimated_size_mb: float) -> bool:
        pass

    @abstractmethod
    def force_cleanup(self) -> float:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class MemoryManager(MemoryManagerInterface):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._available = PSUTIL_AVAILABLE

        if not self._available:
            self._logger.warning("psutil not available - memory monitoring disabled")

    def get_memory_info(self) -> MemoryInfo:
        if not self._available:
            return MemoryInfo(0.0, 1024.0, 1024.0)

        try:
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / 1024 / 1024
            total_mb = system_memory.total / 1024 / 1024
            process = psutil.Process()
            current_mb = process.memory_info().rss / 1024 / 1024

            return MemoryInfo(current_mb, available_mb, total_mb)

        except Exception as e:
            self._logger.debug(f"Error getting memory info: {e}")
            return MemoryInfo(0.0, 1024.0, 1024.0)  # Safe defaults

    def check_availability(self, estimated_size_mb: float, buffer_percentage: float = 20.0) -> bool:
        if not self._available:
            return True 
        try:
            memory_info = self.get_memory_info()
            needed_with_buffer = estimated_size_mb * (1 + buffer_percentage / 100)

            memory_sufficient = memory_info.available_mb > needed_with_buffer

            self._logger.debug(
                f"Memory check - Available: {memory_info.available_mb:.1f} MB, "
                f"Current: {memory_info.current_mb:.1f} MB, "
                f"Needed: {estimated_size_mb:.1f} MB "
                f"({'✓' if memory_sufficient else '⚠️'})"
            )

            if not memory_sufficient:
                self._logger.warning(
                    f"Low memory: Need {needed_with_buffer:.1f} MB, "
                    f"only {memory_info.available_mb:.1f} MB available"
                )

            return memory_sufficient

        except Exception as e:
            self._logger.debug(f"Memory check failed: {e}")
            return True 

    def force_cleanup(self) -> float:
        try:
            initial_info = self.get_memory_info()

            # PERFORMANCE FIX: Reduced from 3 to 1 gc.collect() call
            # Previous: 3× gc.collect() = 300-1500ms blocking time
            # Current: 1× gc.collect() = 100-500ms blocking time
            gc.collect()

            final_info = self.get_memory_info()
            freed_mb = initial_info.current_mb - final_info.current_mb

            if freed_mb > 0:
                self._logger.debug(f"Memory cleanup freed {freed_mb:.1f} MB")

            return max(0.0, freed_mb)

        except Exception as e:
            self._logger.debug(f"Memory cleanup warning: {e}")
            return 0.0

    def is_available(self) -> bool:
        return self._available

    def log_memory_status(self, operation: str = "Operation") -> None:
        try:
            memory_info = self.get_memory_info()
            self._logger.info(
                f"{operation} - Memory status: "
                f"Current: {memory_info.current_mb:.1f} MB, "
                f"Available: {memory_info.available_mb:.1f} MB, "
                f"Usage: {memory_info.usage_percentage:.1f}%"
            )
        except Exception as e:
            self._logger.debug(f"Error logging memory status: {e}")


class NullMemoryManager(MemoryManagerInterface):
    def get_memory_info(self) -> MemoryInfo:
        return MemoryInfo(0.0, 1024.0, 1024.0)

    def check_availability(self, estimated_size_mb: float) -> bool:
        return True

    def force_cleanup(self) -> float:
        gc.collect()
        return 0.0

    def is_available(self) -> bool:
        return False


def create_memory_manager() -> MemoryManagerInterface:
    if PSUTIL_AVAILABLE:
        return MemoryManager()
    else:
        return NullMemoryManager()