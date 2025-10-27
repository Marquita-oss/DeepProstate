import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import uuid

from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.core.domain.entities.segmentation import (
    MedicalSegmentation, AnatomicalRegion, SegmentationType,
    SegmentationMetrics, IntensityStatistics, ConfidenceLevel
)
from deepprostate.core.domain.services.ai_model_service import AIModelService


class SegmentationConversionService:
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
        self._logger = logging.getLogger(__name__)
        
        self._logger.info("SegmentationConversionService initialized with Clean Architecture")
    
    async def create_segmentation_from_prediction(
        self,
        mask_data: np.ndarray,
        confidence_map: np.ndarray,
        anatomical_region: AnatomicalRegion,
        parent_image: MedicalImage,
        preprocessing_metadata: Optional[Dict[str, Any]] = None
    ) -> MedicalSegmentation:
        try:
            confidence_threshold = self._model_service.get_confidence_threshold(anatomical_region)
            thresholded_mask = self._apply_confidence_threshold(
                mask_data, confidence_map, confidence_threshold
            )
            
            metrics = self._calculate_segmentation_metrics(
                thresholded_mask, parent_image, confidence_map
            )
            
            confidence_level = self._determine_confidence_level(confidence_map, confidence_threshold)

            # Calculate average confidence score for the segmentation
            mean_confidence = float(np.mean(confidence_map[thresholded_mask > 0])) if np.sum(thresholded_mask) > 0 else 0.0

            segmentation = MedicalSegmentation(
                mask_data=thresholded_mask,
                anatomical_region=anatomical_region,
                segmentation_type=SegmentationType.AUTOMATIC_AI,
                creation_date=datetime.now(),
                creator_id="ai_model",
                confidence_score=mean_confidence,
                parent_image_uid=parent_image.image_id,
                description=f"AI-generated {anatomical_region.value}"
            )

            self._logger.info(
                f"Created segmentation {segmentation.segmentation_id} for {anatomical_region} "
                f"with confidence {confidence_level}"
            )
            
            return segmentation
            
        except Exception as e:
            self._logger.error(f"Failed to create segmentation for {anatomical_region}: {e}")
            raise SegmentationConversionError(f"Conversion failed: {e}")
    
    async def create_zone_segmentations(
        self, 
        zone_predictions: Dict[str, np.ndarray], 
        parent_image: MedicalImage,
        preprocessing_metadata: Optional[Dict[str, Any]] = None
    ) -> List[MedicalSegmentation]:
        segmentations = []
        
        zone_mapping = {
            "peripheral_zone": AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE,
            "transition_zone": AnatomicalRegion.PROSTATE_TRANSITION_ZONE
        }
        
        for zone_key, mask_data in zone_predictions.items():
            if zone_key in zone_mapping:
                anatomical_region = zone_mapping[zone_key]
                
                confidence_map = mask_data.copy()
                
                segmentation = await self.create_segmentation_from_prediction(
                    mask_data=mask_data,
                    confidence_map=confidence_map,
                    anatomical_region=anatomical_region,
                    parent_image=parent_image,
                    preprocessing_metadata=preprocessing_metadata
                )
                
                segmentations.append(segmentation)
        
        self._logger.info(f"Created {len(segmentations)} zone segmentations")
        return segmentations
    
    async def create_lesion_segmentations(
        self, 
        lesion_predictions: Dict[str, np.ndarray], 
        parent_image: MedicalImage,
        preprocessing_metadata: Optional[Dict[str, Any]] = None
    ) -> List[MedicalSegmentation]:
        segmentations = []
        
        if "suspicious_lesions" in lesion_predictions:
            lesion_mask = lesion_predictions["suspicious_lesions"]
            
            individual_lesions = self._separate_individual_lesions(lesion_mask)
            
            for i, lesion_mask in enumerate(individual_lesions):
                confidence_threshold = 0.7
                confidence_map = lesion_mask.copy()

                segmentation = await self.create_segmentation_from_prediction(
                    mask_data=lesion_mask,
                    confidence_map=confidence_map,
                    anatomical_region=AnatomicalRegion.SUSPICIOUS_LESION,
                    parent_image=parent_image,
                    preprocessing_metadata=preprocessing_metadata
                )

                # Note: Metadata like lesion_index, total_lesions_detected, pi_rads_estimated
                # would need to be stored via a separate metadata service if needed

                segmentations.append(segmentation)
        
        self._logger.info(f"Created {len(segmentations)} lesion segmentations")
        return segmentations
    
    def _apply_confidence_threshold(
        self, 
        mask_data: np.ndarray, 
        confidence_map: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        high_confidence_mask = confidence_map >= threshold
        thresholded_mask = np.logical_and(mask_data > 0.5, high_confidence_mask)
        
        return thresholded_mask.astype(np.uint8)
    
    def _calculate_segmentation_metrics(
        self, 
        mask_data: np.ndarray, 
        parent_image: MedicalImage,
        confidence_map: np.ndarray
    ) -> SegmentationMetrics:
        
        total_voxels = int(np.sum(mask_data))
        total_volume_voxels = int(np.prod(mask_data.shape))
        volume_percentage = float(total_voxels / total_volume_voxels * 100)
        
        physical_volume_mm3 = None
        if parent_image.spacing:
            voxel_volume = (
                parent_image.spacing.x *
                parent_image.spacing.y *
                parent_image.spacing.z
            )
            physical_volume_mm3 = float(total_voxels * voxel_volume)
        
        intensity_stats = self._calculate_intensity_statistics(
            parent_image.image_data, mask_data
        )
        
        return SegmentationMetrics(
            volume_voxels=total_voxels,
            volume_percentage=volume_percentage,
            physical_volume_mm3=physical_volume_mm3,
            intensity_statistics=intensity_stats,
            confidence_metrics={
                "mean_confidence": float(np.mean(confidence_map[mask_data > 0])) if total_voxels > 0 else 0.0,
                "min_confidence": float(np.min(confidence_map[mask_data > 0])) if total_voxels > 0 else 0.0,
                "max_confidence": float(np.max(confidence_map[mask_data > 0])) if total_voxels > 0 else 0.0
            }
        )
    
    def _calculate_intensity_statistics(
        self,
        image_data: np.ndarray,
        mask_data: np.ndarray
    ) -> IntensityStatistics:

        if np.sum(mask_data) == 0:
            return IntensityStatistics(
                mean_intensity=0.0,
                std_intensity=0.0,
                min_intensity=0.0,
                max_intensity=0.0,
                median_intensity=0.0,
                percentile_25=0.0,
                percentile_75=0.0,
                entropy=0.0,
                uniformity=0.0
            )

        region_intensities = image_data[mask_data > 0]

        # Calculate entropy
        if len(region_intensities) > 0:
            hist, _ = np.histogram(region_intensities, bins=256, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
            uniformity = np.sum(hist ** 2)
        else:
            entropy = 0.0
            uniformity = 0.0

        return IntensityStatistics(
            mean_intensity=float(np.mean(region_intensities)),
            std_intensity=float(np.std(region_intensities)),
            min_intensity=float(np.min(region_intensities)),
            max_intensity=float(np.max(region_intensities)),
            median_intensity=float(np.median(region_intensities)),
            percentile_25=float(np.percentile(region_intensities, 25)),
            percentile_75=float(np.percentile(region_intensities, 75)),
            entropy=entropy,
            uniformity=uniformity
        )
    
    def _determine_confidence_level(
        self, 
        confidence_map: np.ndarray, 
        threshold: float
    ) -> ConfidenceLevel:        
        if confidence_map.size == 0:
            return ConfidenceLevel.LOW
        
        mean_confidence = float(np.mean(confidence_map))
        
        if mean_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif mean_confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif mean_confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_confidence_stats(self, confidence_map: np.ndarray) -> Dict[str, float]:        
        if confidence_map.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": float(np.mean(confidence_map)),
            "std": float(np.std(confidence_map)),
            "min": float(np.min(confidence_map)),
            "max": float(np.max(confidence_map)),
            "median": float(np.median(confidence_map)),
            "percentile_90": float(np.percentile(confidence_map, 90))
        }
    
    def _separate_individual_lesions(self, lesion_mask: np.ndarray) -> List[np.ndarray]:
        try:
            from scipy import ndimage
            
            labeled_array, num_features = ndimage.label(lesion_mask > 0.5)
            
            individual_lesions = []
            for i in range(1, num_features + 1):
                lesion_mask_individual = (labeled_array == i).astype(np.uint8)
                
                if np.sum(lesion_mask_individual) >= 5:
                    individual_lesions.append(lesion_mask_individual)
            
            return individual_lesions
            
        except ImportError:
            self._logger.warning("scipy not available, cannot separate individual lesions")
            return [lesion_mask] if np.sum(lesion_mask) > 0 else []
    
    def _estimate_pi_rads(self, lesion_mask: np.ndarray, confidence_map: np.ndarray) -> int:        
        volume = np.sum(lesion_mask)
        mean_confidence = np.mean(confidence_map[lesion_mask > 0]) if volume > 0 else 0
        
        if volume < 10 or mean_confidence < 0.6:
            return 2 
        elif volume < 50 and mean_confidence < 0.8:
            return 3  
        elif volume < 100 and mean_confidence >= 0.8:
            return 4 
        else:
            return 5 
    
    def postprocess_segmentation(
        self, 
        segmentation: MedicalSegmentation,
        postprocessing_options: Optional[Dict[str, Any]] = None
    ) -> MedicalSegmentation:
        if postprocessing_options is None:
            return segmentation
        
        processed_mask = segmentation.mask_data.copy()
        
        if postprocessing_options.get("remove_small_components", False):
            processed_mask = self._remove_small_components(
                processed_mask, 
                min_size=postprocessing_options.get("min_component_size", 10)
            )
        
        if postprocessing_options.get("fill_holes", False):
            processed_mask = self._fill_holes(processed_mask)

        # Note: MedicalSegmentation mask_data is read-only, and metadata is not directly accessible
        # This postprocessing would need to return a new MedicalSegmentation instance
        # For now, we return the original segmentation unchanged
        self._logger.warning("postprocess_segmentation not fully implemented - returning original segmentation")

        return segmentation
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        try:
            from scipy import ndimage
            
            labeled_array, num_features = ndimage.label(mask)
            
            component_sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
            
            large_components = component_sizes >= min_size
            large_components_mask = large_components[labeled_array]
            
            return (mask * large_components_mask).astype(np.uint8)
            
        except ImportError:
            self._logger.warning("scipy not available, skipping small component removal")
            return mask
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        try:
            from scipy import ndimage
            return ndimage.binary_fill_holes(mask).astype(np.uint8)
        except ImportError:
            self._logger.warning("scipy not available, skipping hole filling")
            return mask


class SegmentationConversionError(Exception):
    pass