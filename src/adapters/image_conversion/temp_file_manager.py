"""
adapters/image_conversion/temp_file_manager.py

Manages the lifecycle of temporary files created during image conversion.
This service handles registration, tracking, and cleanup of temporary files
in a thread-safe manner to prevent resource leaks.

Part of the Adapters layer - Infrastructure concern for file system management.
"""

import logging
from pathlib import Path
from typing import List, Set
from threading import Lock
import weakref


class TempFileManager:
    """
    Thread-safe manager for temporary file lifecycle.

    Responsibilities:
    - Register temporary files for tracking
    - Clean up individual files
    - Clean up all registered files
    - Handle cleanup errors gracefully
    - Provide thread-safe operations

    This class uses a singleton pattern to ensure all components
    share the same temp file registry.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern to ensure single registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize temp file tracking structures."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._temp_files: Set[Path] = set()
        self._file_lock = Lock()
        self._logger = logging.getLogger(__name__)

        weakref.finalize(self, self._cleanup_on_shutdown)

    def register_temp_file(self, path: Path) -> None:
        """
        Register a temporary file for tracking.

        Args:
            path: Path to the temporary file

        Raises:
            TypeError: If path is not a Path object
            ValueError: If path is invalid or empty
        """
        if not isinstance(path, Path):
            raise TypeError(f"Expected Path object, got {type(path)}")

        if not path or not str(path).strip():
            raise ValueError("Path cannot be empty")

        with self._file_lock:
            self._temp_files.add(path)
            self._logger.debug(f"Registered temp file: {path}")

    def unregister_temp_file(self, path: Path) -> None:
        """
        Remove a file from tracking without deleting it.

        Args:
            path: Path to unregister
        """
        with self._file_lock:
            self._temp_files.discard(path)
            self._logger.debug(f"Unregistered temp file: {path}")

    def cleanup_file(self, path: Path, remove_from_registry: bool = True) -> bool:
        """
        Delete a specific temporary file.

        Args:
            path: Path to the file to delete
            remove_from_registry: Whether to remove from tracking

        Returns:
            True if file was successfully deleted, False otherwise
        """
        try:
            if path.exists():
                path.unlink()
                self._logger.debug(f"Deleted temp file: {path}")
            else:
                self._logger.debug(f"Temp file already removed: {path}")

            if remove_from_registry:
                self.unregister_temp_file(path)

            return True

        except Exception as e:
            self._logger.warning(
                f"Failed to delete temp file {path}: {e}",
                exc_info=False
            )
            return False

    def cleanup_all(self) -> int:
        """
        Clean up all registered temporary files.

        Returns:
            Number of files successfully deleted
        """
        deleted_count = 0

        with self._file_lock:
            files_to_delete = list(self._temp_files)
            self._temp_files.clear()

        for temp_file in files_to_delete:
            if self.cleanup_file(temp_file, remove_from_registry=False):
                deleted_count += 1

        if deleted_count > 0:
            self._logger.info(
                f"Cleaned up {deleted_count}/{len(files_to_delete)} temp files"
            )

        return deleted_count

    def cleanup_directory(self, directory: Path, remove_if_empty: bool = True) -> bool:
        """
        Clean up a temporary directory and optionally remove it if empty.

        Args:
            directory: Path to the directory
            remove_if_empty: Whether to remove the directory if it's empty

        Returns:
            True if directory was removed, False otherwise
        """
        try:
            if not directory.exists():
                return False

            if not directory.is_dir():
                self._logger.warning(f"Path is not a directory: {directory}")
                return False

            if remove_if_empty and not any(directory.iterdir()):
                directory.rmdir()
                self._logger.debug(f"Removed empty temp directory: {directory}")
                return True

            return False

        except Exception as e:
            self._logger.warning(
                f"Failed to remove temp directory {directory}: {e}",
                exc_info=False
            )
            return False

    def get_registered_files(self) -> List[Path]:
        """
        Get list of currently registered temporary files.

        Returns:
            List of registered file paths
        """
        with self._file_lock:
            return list(self._temp_files)

    def get_registered_count(self) -> int:
        """
        Get count of currently registered temporary files.

        Returns:
            Number of registered files
        """
        with self._file_lock:
            return len(self._temp_files)

    def clear_registry(self) -> None:
        """
        Clear the registry without deleting files.

        Use this when files have been deleted externally
        or when you want to stop tracking without cleanup.
        """
        with self._file_lock:
            count = len(self._temp_files)
            self._temp_files.clear()
            self._logger.debug(f"Cleared registry of {count} temp files")

    def _cleanup_on_shutdown(self):
        """Internal cleanup called on manager destruction."""
        try:
            file_count = self.get_registered_count()
            if file_count > 0:
                self._logger.info(
                    f"TempFileManager shutdown: cleaning up {file_count} files"
                )
                self.cleanup_all()
        except Exception as e:
            self._logger.error(f"Error during shutdown cleanup: {e}", exc_info=False)


_global_manager = None


def get_temp_file_manager() -> TempFileManager:
    """
    Get the global TempFileManager instance.

    Returns:
        Singleton TempFileManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = TempFileManager()
    return _global_manager
