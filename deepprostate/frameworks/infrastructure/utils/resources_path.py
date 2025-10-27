"""
Cross-platform Resource Path Management Module

This module provides centralized path management for all application resources
to ensure compatibility across Windows, Linux, and macOS.

All paths use pathlib.Path to automatically handle platform-specific separators.
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResourcePathManager:
    """
    Centralized manager for all application resource paths.
    Automatically handles platform-specific path separators.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the resource path manager.

        Args:
            project_root: Root directory of the project. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect: this file is in src/frameworks/infrastructure/utils/
            # Resources are now in src/resources/
            # So src directory is 4 levels up from this file
            self._src_root = Path(__file__).parent.parent.parent.parent
            self._project_root = self._src_root.parent
        else:
            self._project_root = Path(project_root)
            self._src_root = self._project_root / "src"

        logger.info(f"Resource path manager initialized with root: {self._project_root}")

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    # ==================== Resources Directory ====================

    @property
    def resources_dir(self) -> Path:
        """Get the resources directory path."""
        return self._src_root / "resources"

    @property
    def images_dir(self) -> Path:
        """Get the images resource directory path."""
        return self.resources_dir / "image"

    @property
    def icons_dir(self) -> Path:
        """Get the icons resource directory path."""
        return self.resources_dir / "icons"

    # ==================== Specific Resource Files ====================

    @property
    def logo_dp2_png(self) -> Path:
        """Get the main logo (dp2.png) path."""
        return self.images_dir / "dp2.png"

    @property
    def logo_svg(self) -> Path:
        """Get the SVG logo (logo2.svg) path."""
        return self.images_dir / "logo2.svg"

    def get_icon_path(self, icon_name: str) -> Path:
        """
        Get the path for a specific icon.

        Args:
            icon_name: Name of the icon file (e.g., 'ai_analysis.svg')

        Returns:
            Path to the icon file
        """
        return self.icons_dir / icon_name

    # ==================== Data Directories ====================

    @property
    def data_dir(self) -> Path:
        """Get the main data directory path."""
        return self._project_root / "data"

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.data_dir / "logs"

    @property
    def images_data_dir(self) -> Path:
        """Get the images data directory path."""
        return self.data_dir / "images"

    @property
    def temp_dir(self) -> Path:
        """Get the temporary files directory path."""
        return self.data_dir / "temp"

    @property
    def exports_dir(self) -> Path:
        """Get the exports directory path."""
        return self.data_dir / "exports"

    @property
    def dicom_storage_dir(self) -> Path:
        """Get the DICOM storage directory path."""
        return self._project_root / "medical_data" / "dicom_storage"

    # ==================== Specific Data Files ====================

    @property
    def main_log_file(self) -> Path:
        """Get the main application log file path."""
        return self.logs_dir / "deepprostate.log"

    # ==================== Utility Methods ====================

    def ensure_directories_exist(self) -> None:
        """
        Create all necessary directories if they don't exist.
        This should be called during application initialization.

        Only creates data/logs - other directories created on-demand when needed.
        """
        directories = [
            self.logs_dir,  # Only logs directory is created at startup
            # self.images_data_dir,  # Not used - removed
            # self.temp_dir,  # Using /tmp/deep_prostate instead
            # self.exports_dir,  # Created on-demand when exporting
            # self.dicom_storage_dir  # Created on-demand when saving DICOM
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")

    def verify_resources_exist(self) -> dict:
        """
        Verify that essential resource files exist.

        Returns:
            Dictionary with resource names as keys and boolean existence status as values
        """
        resources = {
            "logo_dp2_png": self.logo_dp2_png.exists(),
            "logo_svg": self.logo_svg.exists(),
            "resources_dir": self.resources_dir.exists(),
            "icons_dir": self.icons_dir.exists(),
            "images_dir": self.images_dir.exists()
        }

        for resource_name, exists in resources.items():
            if not exists:
                logger.warning(f"Resource not found: {resource_name} at {getattr(self, resource_name, 'unknown')}")

        return resources

    def get_relative_path_str(self, absolute_path: Path) -> str:
        """
        Convert an absolute path to a relative path string from project root.

        Args:
            absolute_path: The absolute path to convert

        Returns:
            Relative path as string
        """
        try:
            relative = absolute_path.relative_to(self._project_root)
            return str(relative)
        except ValueError:
            logger.warning(f"Path {absolute_path} is not relative to project root")
            return str(absolute_path)


# Global singleton instance for easy access throughout the application
_global_path_manager: Optional[ResourcePathManager] = None


def get_resource_path_manager() -> ResourcePathManager:
    """
    Get the global ResourcePathManager singleton instance.

    Returns:
        The global ResourcePathManager instance
    """
    global _global_path_manager
    if _global_path_manager is None:
        _global_path_manager = ResourcePathManager()
    return _global_path_manager


def initialize_path_manager(project_root: Optional[Path] = None) -> ResourcePathManager:
    """
    Initialize the global ResourcePathManager with a specific project root.

    Args:
        project_root: Root directory of the project. If None, auto-detects.

    Returns:
        The initialized ResourcePathManager instance
    """
    global _global_path_manager
    _global_path_manager = ResourcePathManager(project_root)
    return _global_path_manager


# Convenience functions for common paths
def get_logo_path() -> Path:
    """Get the main logo path (dp2.png)."""
    return get_resource_path_manager().logo_dp2_png


def get_logo_svg_path() -> Path:
    """Get the SVG logo path (logo2.svg)."""
    return get_resource_path_manager().logo_svg


def get_icon_path(icon_name: str) -> Path:
    """Get an icon path by name."""
    return get_resource_path_manager().get_icon_path(icon_name)


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_resource_path_manager().data_dir


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return get_resource_path_manager().logs_dir


def get_main_log_file() -> Path:
    """Get the main log file path."""
    return get_resource_path_manager().main_log_file


def get_dicom_storage_dir() -> Path:
    """Get the DICOM storage directory path."""
    return get_resource_path_manager().dicom_storage_dir


def ensure_data_directories() -> None:
    """Ensure all data directories exist."""
    get_resource_path_manager().ensure_directories_exist()


def verify_resources() -> dict:
    """Verify that essential resources exist."""
    return get_resource_path_manager().verify_resources_exist()
