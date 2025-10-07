"""
application/services/segmentation_services.py

Segmentation services for medical imaging.
Production mode with nnUNetv2.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime
import asyncio
from pathlib import Path
import os
import logging

from deepprostate.core.domain.entities.medical_image import MedicalImage, ImageSpacing
from deepprostate.core.domain.entities.segmentation import (
    MedicalSegmentation, AnatomicalRegion, SegmentationType,
    SegmentationMetrics, IntensityStatistics, ConfidenceLevel
)
from deepprostate.core.domain.repositories.repositories import (
    MedicalImageRepository, SegmentationRepository,
    SegmentationNotFoundError, RepositoryError
)

class SegmentationEditingService:    
    def __init__(self, segmentation_repository: SegmentationRepository):
        self._segmentation_repository = segmentation_repository
    
    async def apply_brush_edit(
        self,
        segmentation: MedicalSegmentation,
        brush_coordinates: List[Tuple[int, int, int]],
        brush_radius: int,
        edit_mode: str,
        editor_id: str
    ) -> MedicalSegmentation:
        if segmentation.is_locked:
            raise SegmentationEditingError(
                f"Segmentation {segmentation.segmentation_id} is locked for editing"
            )

        edited_mask = segmentation.mask_data.copy()

        for coord in brush_coordinates:
            brush_mask = self._create_spherical_brush(
                edited_mask.shape, coord, brush_radius
            )

            if edit_mode == "add":
                edited_mask = np.logical_or(edited_mask, brush_mask)
            elif edit_mode == "remove":
                edited_mask = np.logical_and(edited_mask, ~brush_mask)
            elif edit_mode == "replace":
                edited_mask = np.logical_and(edited_mask, ~brush_mask)
                edited_mask = np.logical_or(edited_mask, brush_mask)

        new_segmentation = MedicalSegmentation(
            segmentation_id=f"edited_{segmentation.segmentation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            parent_image_id=segmentation.parent_image_id,
            anatomical_region=segmentation.anatomical_region,
            segmentation_type=SegmentationType.MANUAL_EDIT,
            mask_data=edited_mask.astype(np.uint8),
            confidence_level=ConfidenceLevel.HIGH,  
            created_at=datetime.now(),
            metadata={
                **segmentation.metadata,
                "edited_from": segmentation.segmentation_id,
                "editor_id": editor_id,
                "edit_mode": edit_mode,
                "brush_radius": brush_radius,
                "edits_applied": len(brush_coordinates)
            }
        )
        
        await self._segmentation_repository.save_segmentation(new_segmentation)
        
        return new_segmentation
    
    async def merge_segmentations(
        self,
        segmentations: List[MedicalSegmentation],
        merge_strategy: str,  
        merger_id: str
    ) -> MedicalSegmentation:
        if not segmentations:
            raise SegmentationEditingError("No segmentations to combine")

        parent_image_id = segmentations[0].parent_image_id
        if not all(seg.parent_image_id == parent_image_id for seg in segmentations):
            raise SegmentationEditingError("All segmentations must be from the same image")

        base_mask = segmentations[0].mask_data
        combined_mask = base_mask.copy()
        
        for segmentation in segmentations[1:]:
            if merge_strategy == "union":
                combined_mask = np.logical_or(combined_mask, segmentation.mask_data)
            elif merge_strategy == "intersection":
                combined_mask = np.logical_and(combined_mask, segmentation.mask_data)
            elif merge_strategy == "dominant":
                combined_mask = np.where(segmentation.mask_data > 0, segmentation.mask_data, combined_mask)

        merged_segmentation = MedicalSegmentation(
            segmentation_id=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            parent_image_id=parent_image_id,
            anatomical_region=segmentations[0].anatomical_region,  # Use first region
            segmentation_type=SegmentationType.MANUAL_EDIT,
            mask_data=combined_mask.astype(np.uint8),
            confidence_level=ConfidenceLevel.HIGH,
            created_at=datetime.now(),
            metadata={
                "merged_from": [seg.segmentation_id for seg in segmentations],
                "merge_strategy": merge_strategy,
                "merger_id": merger_id,
                "source_count": len(segmentations)
            }
        )
        
        await self._segmentation_repository.save_segmentation(merged_segmentation)
        
        return merged_segmentation
    
    def _create_spherical_brush(
        self,
        volume_shape: Tuple[int, int, int],
        center: Tuple[int, int, int],
        radius: int
    ) -> np.ndarray:
        brush_mask = np.zeros(volume_shape, dtype=bool)

        z_coords, y_coords, x_coords = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]

        distance_squared = (
            (z_coords - center[0])**2 +
            (y_coords - center[1])**2 +
            (x_coords - center[2])**2
        )

        brush_mask = distance_squared <= radius**2

        return brush_mask


class AISegmentationError(Exception):
    pass


class ImageValidationError(Exception):
    pass


class SegmentationEditingError(Exception):
    pass