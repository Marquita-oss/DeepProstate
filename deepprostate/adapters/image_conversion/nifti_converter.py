"""
adapters/image_conversion/nifti_converter.py

Adapter for converting MedicalImage entities to NIfTI format.

This belongs in the Adapters layer because it bridges domain entities
(MedicalImage) with external format (NIfTI via nibabel library).

Key responsibilities:
- Convert MedicalImage domain entity to NIfTI format
- Handle proper affine matrix creation from spacing
- Set appropriate NIfTI header metadata
- Manage temporary file creation and tracking
- Provide robust error handling for conversion failures
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from deepprostate.core.domain.entities.medical_image import MedicalImage
from .temp_file_manager import TempFileManager


class NIfTIConversionError(Exception):
    """Raised when NIfTI conversion fails."""
    pass


class NIfTIConverter:
    """
    Adapter for converting MedicalImage entities to NIfTI format.

    This adapter provides a clean separation between domain entities
    and external file formats, allowing domain entities to remain
    pure and framework-independent.

    Usage:
        converter = NIfTIConverter()
        nifti_path = converter.convert_to_nifti(
            medical_image=my_image,
            temp_dir=Path("/tmp/analysis")
        )
    """

    def __init__(
        self,
        temp_file_manager: Optional[TempFileManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the NIfTI converter.

        Args:
            temp_file_manager: Optional manager for tracking temp files.
                              If None, creates a new instance.
            logger: Optional logger instance. If None, creates module logger.
        """
        self._temp_file_manager = temp_file_manager or TempFileManager()
        self._logger = logger or logging.getLogger(__name__)

    def convert_to_nifti(
        self,
        medical_image: MedicalImage,
        temp_dir: Path,
        filename: Optional[str] = None,
        skip_if_exists: bool = True
    ) -> Path:
        """
        Convert a MedicalImage entity to NIfTI format (.nii.gz).

        Simplified version using SimpleITK directly - just 3 core lines.

        Args:
            medical_image: Domain entity containing image data and metadata
            temp_dir: Directory for temporary NIfTI file
            filename: Optional custom filename. If None, generates from series UID
            skip_if_exists: If True, skip conversion if file already exists

        Returns:
            Path to the created NIfTI file

        Raises:
            NIfTIConversionError: If conversion fails
            ValueError: If input parameters are invalid
            ImportError: If SimpleITK is not available
        """
        self._validate_inputs(medical_image, temp_dir)

        try:
            import SimpleITK as sitk
        except ImportError as e:
            raise ImportError(
                "SimpleITK is required for NIfTI conversion. "
                "Install it with: pip install SimpleITK"
            ) from e

        try:
            # Don't create temp_dir - it's passed from orchestrator which already created it
            # temp_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                filename = self._generate_filename(medical_image)

            temp_path = temp_dir / filename

            if skip_if_exists and self._is_valid_existing_file(temp_path):
                self._logger.info(f"Using existing NIfTI file: {temp_path}")
                self._temp_file_manager.register_temp_file(temp_path)
                return temp_path

            # ===== SIMPLIFIED CONVERSION: 3 LINES =====
            sitk_image = sitk.GetImageFromArray(medical_image.image_data)
            sitk_image.SetSpacing([medical_image.spacing.x, medical_image.spacing.y, medical_image.spacing.z])
            sitk.WriteImage(sitk_image, str(temp_path), useCompression=True)
            # ==========================================

            self._verify_created_file(temp_path)
            self._temp_file_manager.register_temp_file(temp_path)

            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            self._logger.info(
                f"Created NIfTI file: {temp_path.name} ({file_size_mb:.2f} MB)"
            )

            return temp_path

        except NIfTIConversionError:
            raise

        except Exception as e:
            error_msg = f"Failed to convert MedicalImage to NIfTI: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            raise NIfTIConversionError(error_msg) from e

    def _validate_inputs(
        self,
        medical_image: MedicalImage,
        temp_dir: Path
    ) -> None:
        """
        Validate input parameters.

        Args:
            medical_image: Image to validate
            temp_dir: Directory path to validate

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(medical_image, MedicalImage):
            raise ValueError(
                f"Expected MedicalImage, got {type(medical_image)}"
            )

        if not isinstance(temp_dir, Path):
            raise ValueError(
                f"temp_dir must be a Path object, got {type(temp_dir)}"
            )

        if medical_image.image_data.size == 0:
            raise ValueError("MedicalImage has empty image data")

    def _generate_filename(self, medical_image: MedicalImage) -> str:
        """
        Generate a unique filename based on series UID.

        Args:
            medical_image: Image to generate filename for

        Returns:
            Generated filename with .nii.gz extension
        """
        series_uid = medical_image.series_instance_uid
        uid_prefix = series_uid[:8] if len(series_uid) >= 8 else series_uid

        safe_prefix = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in uid_prefix
        )

        filename = f"temp_{safe_prefix}.nii.gz"
        return filename

    def _is_valid_existing_file(self, path: Path) -> bool:
        """
        Check if an existing NIfTI file is valid.

        Args:
            path: Path to check

        Returns:
            True if file exists and appears valid
        """
        if not path.exists():
            return False

        file_size = path.stat().st_size
        if file_size < 100:
            self._logger.warning(
                f"Existing file too small, will recreate: {path}"
            )
            return False

        return True

    # NOTE: Removed _prepare_image_data, _create_affine_matrix, and _set_header_metadata
    # SimpleITK handles data types and metadata automatically.

    def _verify_created_file(self, path: Path) -> None:
        """
        Verify that NIfTI file was created successfully.

        Args:
            path: Path to verify

        Raises:
            NIfTIConversionError: If file verification fails
        """
        if not path.exists():
            raise NIfTIConversionError(
                f"NIfTI file was not created: {path}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise NIfTIConversionError(
                f"Created NIfTI file is empty: {path}"
            )

        self._logger.debug(f"Verified NIfTI file: {path} ({file_size} bytes)")

    def cleanup_temp_files(self) -> int:
        """
        Clean up all temporary files created by this converter.

        Returns:
            Number of files successfully deleted
        """
        deleted_count = self._temp_file_manager.cleanup_all()
        self._logger.info(f"Cleaned up {deleted_count} temporary NIfTI files")
        return deleted_count

    def cleanup_temp_directory(
        self,
        temp_dir: Path,
        remove_if_empty: bool = True
    ) -> bool:
        """
        Clean up a temporary directory.

        Args:
            temp_dir: Directory to clean up
            remove_if_empty: Whether to remove directory if empty

        Returns:
            True if directory was removed, False otherwise
        """
        return self._temp_file_manager.cleanup_directory(
            temp_dir,
            remove_if_empty=remove_if_empty
        )
