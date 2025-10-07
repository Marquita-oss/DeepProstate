import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

from deepprostate.core.domain.entities.medical_image import MedicalImage, ImageSpacing


class AIPreprocessingService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._logger.info("AIPreprocessingService initialized - NO preprocessing, data structure only")

    async def prepare_for_nnunet(self, image: MedicalImage) -> Dict[str, Any]:
        try:
            self._logger.info(f"Preparing (NOT preprocessing) image {image.image_id} for nnUNet")

            original_data = image.pixel_data.copy()

            if not isinstance(original_data, np.ndarray):
                raise ValueError("Image data must be numpy array")

            if original_data.ndim != 3:
                raise ValueError(f"Expected 3D data, got {original_data.ndim}D")

            metadata = {
                "image_id": image.image_id,
                "original_shape": list(original_data.shape),
                "data_type": str(original_data.dtype),
                "spacing": [
                    image.spacing.row_spacing,
                    image.spacing.column_spacing,
                    image.spacing.slice_spacing
                ] if image.spacing else None,
                "preprocessing_applied": "NONE - nnUNet handles all preprocessing",
                "data_modified": False,
                "values_preserved": "original_unchanged"
            }

            result = {
                "original_data": original_data,  
                "properties": metadata, 
                "image_id": image.image_id
            }

            self._logger.info(
                f"Data preparation completed for {image.image_id}: "
                f"Shape {original_data.shape}, dtype {original_data.dtype} - NO modifications applied"
            )

            return result

        except Exception as e:
            self._logger.error(f"Data preparation failed for {image.image_id}: {e}")
            raise AIPreprocessingError(f"Data preparation failed: {e}")

    async def prepare_multi_sequence_for_nnunet(
        self,
        sequences: Dict[str, MedicalImage]
    ) -> Dict[str, Any]:
        try:
            self._logger.info(f"Preparing multi-sequence data: {list(sequences.keys())}")

            prepared_sequences = {}
            sequence_metadata = {}

            for seq_name, image in sequences.items():
                prepared = await self.prepare_for_nnunet(image)
                prepared_sequences[seq_name] = prepared["original_data"]
                sequence_metadata[seq_name] = prepared["properties"]

            result = {
                "sequences_data": prepared_sequences, 
                "sequences_metadata": sequence_metadata,
                "multi_sequence_info": {
                    "total_sequences": len(sequences),
                    "sequence_names": list(sequences.keys()),
                    "spatial_alignment_note": "Alignment handled by nnUNet preprocessing",
                    "interpolation_note": "nnUNet handles all spatial transformations"
                }
            }

            self._logger.info(f"Multi-sequence preparation completed: {len(sequences)} sequences")
            return result

        except Exception as e:
            self._logger.error(f"Multi-sequence preparation failed: {e}")
            raise AIPreprocessingError(f"Multi-sequence preparation failed: {e}")


class AIPreprocessingError(Exception):
    pass