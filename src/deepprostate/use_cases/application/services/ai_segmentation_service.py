import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.core.domain.entities.segmentation import MedicalSegmentation, AnatomicalRegion
from deepprostate.core.domain.repositories.repositories import SegmentationRepository
from deepprostate.core.domain.services.ai_model_service import AIModelService
from deepprostate.core.domain.services.image_validation_service import ImageValidationService, ImageValidationError
from deepprostate.core.domain.services.ai_preprocessing_service import AIPreprocessingService, AIPreprocessingError
from deepprostate.core.domain.services.ai_inference_service import AIInferenceService, AIInferenceError
from deepprostate.core.domain.services.segmentation_conversion_service import (
    SegmentationConversionService, SegmentationConversionError
)


class AISegmentationService:
    def __init__(
        self,
        segmentation_repository: SegmentationRepository,
        model_config: Dict[str, Any],
        dynamic_config_service=None
    ):
        self._segmentation_repository = segmentation_repository
        self._dynamic_config_service = dynamic_config_service
        self._logger = logging.getLogger(__name__)

        self._model_service = AIModelService(model_config, dynamic_config_service)
        self._validation_service = ImageValidationService()
        self._preprocessing_service = AIPreprocessingService()
        self._inference_service = AIInferenceService(self._model_service)
        self._conversion_service = SegmentationConversionService(self._model_service)

        self._logger.info("AISegmentationService initialized")
    
    async def predict_prostate_segmentation(
        self,
        image: MedicalImage,
        include_zones: bool = True,
        detect_lesions: bool = True
    ) -> List[MedicalSegmentation]:
        try:
            self._logger.info(f"Starting AI segmentation workflow for {image.image_id}")
            
            await self._validation_service.validate_for_prostate_segmentation(image)
            self._logger.debug("✓ Image validation completed")
            
            prepared_data = await self._preprocessing_service.prepare_for_nnunet(image)
            self._logger.debug("✓ Data preparation completed - original data preserved")
            
            segmentations = []

            prostate_segmentation = await self._predict_prostate_whole(
                image, prepared_data
            )
            segmentations.append(prostate_segmentation)

            if include_zones:
                zone_segmentations = await self._predict_prostate_zones(
                    image, prepared_data
                )
                segmentations.extend(zone_segmentations)

            if detect_lesions:
                lesion_segmentations = await self._predict_lesions(
                    image, prepared_data
                )
                segmentations.extend(lesion_segmentations)

            await self._save_segmentations(segmentations)
            
            self._logger.info(
                f"AI segmentation workflow completed for {image.image_id}. "
                f"Generated {len(segmentations)} segmentations"
            )
            
            return segmentations
            
        except (ImageValidationError, AIPreprocessingError, AIInferenceError,
                SegmentationConversionError) as e:
            self._logger.error(f"AI segmentation workflow failed for {image.image_id}: {e}")
            raise AISegmentationError(f"Workflow failed: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error in AI segmentation workflow: {e}")
            raise AISegmentationError(f"Unexpected error: {e}")
    
    async def _predict_prostate_whole(
        self,
        image: MedicalImage,
        prepared_data: Dict[str, Any]
    ) -> MedicalSegmentation:
        predictions = await self._inference_service.run_inference(
            prepared_data, "prostate_whole"
        )
        segmentation = await self._conversion_service.create_segmentation_from_prediction(
            mask_data=predictions["prostate_mask"],
            confidence_map=predictions["confidence_map"],
            anatomical_region=AnatomicalRegion.PROSTATE_WHOLE,
            parent_image=image,
            preprocessing_metadata=prepared_data["properties"]
        )
        
        self._logger.debug("✓ Prostate whole segmentation created")
        return segmentation
    
    async def _predict_prostate_zones(
        self,
        image: MedicalImage,
        prepared_data: Dict[str, Any]
    ) -> List[MedicalSegmentation]:
        zone_predictions = await self._inference_service.run_inference(
            prepared_data, "prostate_zones"
        )
        zone_segmentations = await self._conversion_service.create_zone_segmentations(
            zone_predictions, image, prepared_data["properties"]
        )

        self._logger.debug(f"✓ {len(zone_segmentations)} zone segmentations created")
        return zone_segmentations
    
    async def _predict_lesions(
        self,
        image: MedicalImage,
        prepared_data: Dict[str, Any]
    ) -> List[MedicalSegmentation]:
        lesion_predictions = await self._inference_service.run_inference(
            prepared_data, "lesion_detection"
        )
        lesion_segmentations = await self._conversion_service.create_lesion_segmentations(
            lesion_predictions, image, prepared_data["properties"]
        )

        self._logger.debug(f"✓ {len(lesion_segmentations)} lesion segmentations created")
        return lesion_segmentations
    
    async def _save_segmentations(self, segmentations: List[MedicalSegmentation]) -> None:        
        save_tasks = [
            self._segmentation_repository.save_segmentation(seg)
            for seg in segmentations
        ]
        
        await asyncio.gather(*save_tasks)
        self._logger.debug(f"✓ {len(segmentations)} segmentations saved")
    
    async def refine_segmentation_with_ai(
        self,
        existing_segmentation: MedicalSegmentation,
        refinement_hints: Dict[str, Any]
    ) -> MedicalSegmentation:
        try:
            self._logger.info(f"Refining segmentation {existing_segmentation.segmentation_id}")            
            refined_mask = await self._apply_ai_refinement(
                existing_segmentation.mask_data, refinement_hints
            )

            refined_segmentation = MedicalSegmentation(
                segmentation_id=f"refined_{existing_segmentation.segmentation_id}",
                parent_image_id=existing_segmentation.parent_image_id,
                anatomical_region=existing_segmentation.anatomical_region,
                segmentation_type=existing_segmentation.segmentation_type,
                mask_data=refined_mask,
                confidence_level=existing_segmentation.confidence_level,
                created_at=datetime.now(),
                metadata={
                    **existing_segmentation.metadata,
                    "refined_from": existing_segmentation.segmentation_id,
                    "refinement_hints": refinement_hints,
                    "refinement_applied": True
                }
            )

            await self._segmentation_repository.save_segmentation(refined_segmentation)
            self._logger.info(f"Segmentation refinement completed")
            return refined_segmentation
            
        except Exception as e:
            self._logger.error(f"Segmentation refinement failed: {e}")
            raise AISegmentationError(f"Refinement failed: {e}")
    
    async def _apply_ai_refinement(
        self,
        mask_data,
        refinement_hints: Dict[str, Any]
    ):
        refined_mask = mask_data.copy()

        if refinement_hints.get("smooth", False):
            from scipy import ndimage
            refined_mask = ndimage.binary_closing(refined_mask)

        if refinement_hints.get("fill_holes", False):
            from scipy import ndimage
            refined_mask = ndimage.binary_fill_holes(refined_mask)

        return refined_mask
    
    async def validate_ai_predictions(
        self,
        segmentations: List[MedicalSegmentation]
    ) -> Dict[str, Any]:
        try:
            validation_report = {
                "total_segmentations": len(segmentations),
                "validation_timestamp": datetime.now().isoformat(),
                "segmentations": [],
                "overall_status": "passed",
                "warnings": [],
                "errors": []
            }
            
            for segmentation in segmentations:
                seg_validation = await self._validate_single_segmentation(segmentation)
                validation_report["segmentations"].append(seg_validation)
                
                if seg_validation["status"] == "failed":
                    validation_report["overall_status"] = "failed"
                    validation_report["errors"].extend(seg_validation.get("errors", []))
                elif seg_validation["status"] == "warning":
                    validation_report["warnings"].extend(seg_validation.get("warnings", []))
            
            self._logger.info(
                f"AI predictions validation completed: {validation_report['overall_status']}"
            )
            
            return validation_report
            
        except Exception as e:
            self._logger.error(f"AI predictions validation failed: {e}")
            raise AISegmentationError(f"Validation failed: {e}")
    
    async def _validate_single_segmentation(
        self,
        segmentation: MedicalSegmentation
    ) -> Dict[str, Any]:        
        validation = {
            "segmentation_id": segmentation.segmentation_id,
            "anatomical_region": segmentation.anatomical_region.value,
            "status": "passed",
            "warnings": [],
            "errors": []
        }        
        if segmentation.metrics and segmentation.metrics.volume_voxels:
            volume = segmentation.metrics.volume_voxels
            volume_limits = {
                AnatomicalRegion.PROSTATE_WHOLE: (1000, 100000),
                AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: (500, 50000),
                AnatomicalRegion.PROSTATE_TRANSITION_ZONE: (200, 30000),
                AnatomicalRegion.SUSPICIOUS_LESION: (5, 5000)
            }
            
            if segmentation.anatomical_region in volume_limits:
                min_vol, max_vol = volume_limits[segmentation.anatomical_region]
                
                if volume < min_vol:
                    validation["warnings"].append(f"Volume too small: {volume} < {min_vol}")
                    validation["status"] = "warning"
                elif volume > max_vol:
                    validation["errors"].append(f"Volume too large: {volume} > {max_vol}")
                    validation["status"] = "failed"
        
        if segmentation.confidence_level.value == "low":
            validation["warnings"].append("Low confidence segmentation")
            if validation["status"] == "passed":
                validation["status"] = "warning"
        
        return validation
    
    def get_model_status(self) -> Dict[str, Any]:
        return {
            "loaded_models": self._model_service.get_loaded_models(),
            "model_availability": {
                "prostate_whole": self._model_service.validate_model_availability("prostate_whole"),
                "prostate_zones": self._model_service.validate_model_availability("prostate_zones"),
                "lesion_detection": self._model_service.validate_model_availability("lesion_detection")
            },
            "inference_ready": self._inference_service.validate_inference_requirements("prostate_whole")
        }
    
    def configure_confidence_thresholds(self, thresholds: Dict[AnatomicalRegion, float]) -> None:
        for region, threshold in thresholds.items():
            self._model_service.set_confidence_threshold(region, threshold)
        
        self._logger.info(f"Confidence thresholds updated for {len(thresholds)} regions")
    
    def clear_model_cache(self) -> None:
        self._model_service.clear_cache()
        self._logger.debug("Model cache cleared")


class AISegmentationError(Exception):
    pass