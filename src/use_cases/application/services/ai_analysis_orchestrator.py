import asyncio
import uuid
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from src.core.domain.utils.medical_shape_handler import MedicalShapeHandler

from src.core.domain.entities.medical_image import MedicalImage
from src.core.domain.entities.ai_analysis import (
    AIAnalysisType, AIAnalysisRequest, AIAnalysisResult, 
    OverlayVisualizationData, AIModelStatus
)
from src.core.domain.entities.segmentation import MedicalSegmentation, AnatomicalRegion

from src.use_cases.application.services.ai_segmentation_service import AISegmentationService
from src.frameworks.infrastructure.coordination.medical_format_registry import MedicalFormatRegistry


class AIAnalysisOrchestrator:
    def __init__(
        self,
        segmentation_service: AISegmentationService,
        format_registry: MedicalFormatRegistry,
        temp_storage_path: Path
    ):
        self._segmentation_service = segmentation_service
        self._format_registry = format_registry
        self._temp_storage = temp_storage_path
        self._temp_storage.mkdir(parents=True, exist_ok=True)
        
        self._logger = logging.getLogger(__name__)
        
        self._overlay_colors = {
            AnatomicalRegion.PROSTATE_WHOLE: (0.8, 0.6, 0.4, 0.4),           
            AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: (0.4, 0.8, 0.4, 0.4), 
            AnatomicalRegion.PROSTATE_TRANSITION_ZONE: (0.4, 0.4, 0.8, 0.4), 
            AnatomicalRegion.SUSPICIOUS_LESION: (1.0, 0.8, 0.0, 0.5),        
            AnatomicalRegion.CONFIRMED_CANCER: (1.0, 0.2, 0.2, 0.5),        
        }
        
        self._logger.info("AI Analysis Orchestrator initialized")
    
    async def run_ai_analysis(self, request: AIAnalysisRequest) -> AIAnalysisResult:
        start_time = datetime.now()
        analysis_id = str(uuid.uuid4())[:8]
        
        self._logger.info(f"Starting AI analysis {analysis_id}: {request.analysis_type.value}")
        
        result = AIAnalysisResult(
            segmentations=[],
            overlay_data=[],
            analysis_type=request.analysis_type,
            processing_metadata={
                "analysis_id": analysis_id,
                "request_timestamp": request.request_timestamp,
                "orchestrator_version": "1.0.0"
            },
            status=AIModelStatus.PENDING,
            temp_files_created=[]
        )
        
        try:
            result.status = AIModelStatus.PREPROCESSING
            self._logger.info(f"Analysis {analysis_id}: Validating requirements")
            
            is_valid, validation_errors = request.validate_requirements()
            if not is_valid:
                raise AIAnalysisError(f"Request validation failed: {'; '.join(validation_errors)}")
            
            self._logger.info(f"Analysis {analysis_id}: Converting to nnUNet format")
            
            original_images, temp_nifti_files = await self._convert_to_nifti_for_inference(
                request, analysis_id
            )
            result.temp_files_created.extend(temp_nifti_files.values())
            result.original_image_uid = original_images["primary"].series_instance_uid
            
            result.status = AIModelStatus.RUNNING_INFERENCE
            self._logger.info(f"Analysis {analysis_id}: Running nnUNet inference")
            
            ai_predictions = await self._run_inference_on_nifti_files(
                temp_nifti_files, request.analysis_type, analysis_id
            )
            
            result.status = AIModelStatus.POSTPROCESSING
            self._logger.info(f"Analysis {analysis_id}: Converting results to original format")
            
            self._logger.info(f"Analysis {analysis_id}: Creating domain segmentations")
            segmentations = await self._create_domain_segmentations(
                ai_predictions, original_images["primary"], request.analysis_type
            )
            result.segmentations = segmentations
            self._logger.info(f"Analysis {analysis_id}: Created {len(segmentations)} segmentations")
            
            self._logger.debug(f"Analysis {analysis_id}: Created {len(segmentations)} segmentations with regions: {[seg.anatomical_region.value for seg in segmentations]}")
            
            self._logger.debug(f"Analysis {analysis_id}: Preparing overlay visualization")
            
            additional_sequences = {}
            for seq_name, seq_path in request.additional_sequences.items():
                if seq_name.upper() != "T2W":
                    seq_image = original_images.get(seq_name.lower())
                    if seq_image:
                        additional_sequences[seq_name.upper()] = seq_image
            
            if not additional_sequences:
                additional_sequences = None
            else:
                self._logger.debug(f"Creating overlays for {len(additional_sequences)} additional sequences")
            
            if "visualization_overlays" in ai_predictions and ai_predictions["visualization_overlays"]:
                overlay_data = ai_predictions["visualization_overlays"]
                self._logger.debug(f"Using universal overlays from segmentation service: {len(overlay_data)}")
            else:
                overlay_data = await self._prepare_overlay_visualization(
                    segmentations, 
                    original_images["primary"], 
                    request.overlay_opacity,
                    additional_sequences
                )
            result.overlay_data = overlay_data
            self._logger.info(f"Analysis {analysis_id}: Created {len(overlay_data)} overlays")
            
            if not overlay_data:
                self._logger.error(f"Analysis {analysis_id}: NO SE CREARON OVERLAYS!")
            else:
                for i, overlay in enumerate(overlay_data):
                    non_zero = np.sum(overlay.mask_array > 0)
                    self._logger.debug(f"  Overlay {i}: {overlay.anatomical_region.value}, non-zero pixels: {non_zero}, color: {overlay.color_rgba}")
            
            result.status = AIModelStatus.COMPLETED
            result.completed_timestamp = datetime.now()
            result.processing_time_seconds = (result.completed_timestamp - start_time).total_seconds()
            
            result.processing_metadata.update({
                "model_predictions": ai_predictions.get("metadata", {}),
                "segmentations_created": len(segmentations),
                "overlays_created": len(overlay_data),
                "temp_files_count": len(result.temp_files_created)
            })
            
            self._logger.info(
                f"Analysis {analysis_id} completed successfully in {result.processing_time_seconds:.1f}s. "
                f"Created {len(segmentations)} segmentations."
            )
            
            return result
            
        except Exception as e:
            result.status = AIModelStatus.FAILED
            result.error_message = str(e)
            result.completed_timestamp = datetime.now()
            result.processing_time_seconds = (result.completed_timestamp - start_time).total_seconds()            
            self._logger.error(f"Analysis {analysis_id} failed after {result.processing_time_seconds:.1f}s: {e}")
            
            return result
        
        finally:
            if not request.save_intermediate_files:
                result.cleanup_temp_files()
    
    async def _convert_to_nifti_for_inference(
        self, 
        request: AIAnalysisRequest,
        analysis_id: str
    ) -> Tuple[Dict[str, MedicalImage], Dict[str, Path]]:
        original_images = {}
        temp_nifti_files = {}
        
        primary_image = await self._load_medical_image(request.primary_image_path)
        original_images["primary"] = primary_image
        
        if request.analysis_type == AIAnalysisType.PROSTATE_GLAND:
            t2w_nifti_path = await self._convert_single_image_to_nifti(
                primary_image, f"{analysis_id}_0000"
            )
            temp_nifti_files["t2w"] = t2w_nifti_path
            
        elif request.analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            t2w_nifti_path = await self._convert_single_image_to_nifti(
                primary_image, f"{analysis_id}_0000"
            )
            temp_nifti_files["t2w"] = t2w_nifti_path
            
        elif request.analysis_type == AIAnalysisType.CSPCA_DETECTION:
            t2w_nifti_path = await self._convert_single_image_to_nifti(
                primary_image, f"{analysis_id}_0000"
            )
            temp_nifti_files["t2w"] = t2w_nifti_path
            original_images["t2w"] = primary_image
            
            for seq_name, seq_path in request.additional_sequences.items():
                if seq_name.upper() in ["ADC", "HBV"]:
                    seq_image = await self._load_medical_image(seq_path)
                    original_images[seq_name.lower()] = seq_image
                    
                    channel_map = {"adc": "0001", "hbv": "0002"}
                    channel_num = channel_map.get(seq_name.lower(), "0003")
                    nifti_path = await self._convert_single_image_to_nifti(
                        seq_image, f"{analysis_id}_{channel_num}"
                    )
                    temp_nifti_files[seq_name.lower()] = nifti_path
        
        return original_images, temp_nifti_files
    
    async def _load_medical_image(self, file_path: Path) -> MedicalImage:
        if not self._format_registry.can_load_file(file_path):
            raise AIAnalysisError(f"Unsupported file format: {file_path}")
        
        medical_image = self._format_registry.load_medical_image(file_path)
        if medical_image is None:
            raise AIAnalysisError(f"Failed to load medical image: {file_path}")
            
        return medical_image
    
    async def _convert_single_image_to_nifti(
        self, 
        medical_image: MedicalImage,
        temp_filename_base: str
    ) -> Path:
        try:
            import SimpleITK as sitk
        except ImportError:
            raise AIAnalysisError("SimpleITK not available for format conversion")
        
        temp_file = self._temp_storage / f"{temp_filename_base}.nii.gz"
        
        image_data = medical_image.image_data
        MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=3)
        
        sitk_image = sitk.GetImageFromArray(image_data)
        
        spacing = medical_image.spacing
        sitk_image.SetSpacing([spacing.x, spacing.y, spacing.z])
        
        if hasattr(medical_image, 'origin') and medical_image.origin:
            sitk_image.SetOrigin(medical_image.origin)
        else:
            sitk_image.SetOrigin([0.0, 0.0, 0.0])
            
        if hasattr(medical_image, 'direction') and medical_image.direction:
            sitk_image.SetDirection(medical_image.direction)
        else:
            sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        sitk.WriteImage(sitk_image, str(temp_file), useCompression=True)
        
        return temp_file
    
    async def _run_inference_on_nifti_files(
        self,
        temp_nifti_files: Dict[str, Path],
        analysis_type: AIAnalysisType,
        analysis_id: str
    ) -> Dict[str, Any]:
        primary_file = temp_nifti_files.get("t2w")
        if not primary_file:
            raise AIAnalysisError("No T2W sequence available for inference")
        
        try:
            import SimpleITK as sitk
            sitk_image = sitk.ReadImage(str(primary_file))
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            if len(image_array.shape) == 3:
                MedicalShapeHandler.validate_medical_shape(image_array, expected_dims=3)
                
        except ImportError:
            raise AIAnalysisError("SimpleITK not available for reading converted files")
        
        preprocessed_data = {
            "image_data": image_array,
            "metadata": {
                "analysis_id": analysis_id,
                "analysis_type": analysis_type.value,
                "nifti_files": {k: str(v) for k, v in temp_nifti_files.items()}
            },
            "spacing": [1.0, 1.0, 1.0] 
        }
        
        if analysis_type == AIAnalysisType.PROSTATE_GLAND:
            return await self._segmentation_service._inference_service.run_inference(
                preprocessed_data, "prostate_whole"
            )
        elif analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            return await self._segmentation_service._inference_service.run_inference(
                preprocessed_data, "prostate_zones"
            )
        elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
            return await self._segmentation_service._inference_service.run_inference(
                preprocessed_data, "lesion_detection"
            )
        else:
            raise AIAnalysisError(f"Unsupported analysis type: {analysis_type}")
    
    async def _create_domain_segmentations(
        self,
        ai_predictions: Dict[str, Any],
        original_image: MedicalImage,
        analysis_type: AIAnalysisType
    ) -> List[MedicalSegmentation]:
        segmentations = []
        
        if analysis_type == AIAnalysisType.PROSTATE_GLAND:
            mask_data = ai_predictions.get("unified_mask", ai_predictions.get("prostate_mask"))
            prostate_segmentation = await self._segmentation_service._conversion_service.create_segmentation_from_prediction(
                mask_data=mask_data,
                confidence_map=ai_predictions["confidence_map"],
                anatomical_region=AnatomicalRegion.PROSTATE_WHOLE,
                parent_image=original_image,
                preprocessing_metadata=ai_predictions.get("metadata", {})
            )
            segmentations.append(prostate_segmentation)

        elif analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            zone_segmentations = await self._segmentation_service._conversion_service.create_zone_segmentations(
                ai_predictions, original_image, ai_predictions.get("metadata", {})
            )
            segmentations.extend(zone_segmentations)

        elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
            lesion_segmentations = await self._segmentation_service._conversion_service.create_lesion_segmentations(
                ai_predictions, original_image, ai_predictions.get("metadata", {})
            )
            segmentations.extend(lesion_segmentations)
        
        return segmentations
    
    async def _prepare_overlay_visualization(
        self,
        segmentations: List[MedicalSegmentation],
        original_image: MedicalImage,
        opacity: float = 0.4,
        additional_sequences: Optional[Dict[str, MedicalImage]] = None
    ) -> List[OverlayVisualizationData]:
        overlay_data = []
        
        for segmentation in segmentations:
            color_rgba = self._overlay_colors.get(
                segmentation.anatomical_region,
                (0.5, 0.5, 0.5, opacity) 
            )
            
            volume_mm3 = 0.0
            if original_image.spacing:
                voxel_volume = original_image.spacing.get_voxel_volume()
                voxel_count = np.sum(segmentation.mask_data > 0)
                volume_mm3 = voxel_count * voxel_volume
            
            shape_info = MedicalShapeHandler.format_shape_info(segmentation.mask_data)
            mask_stats = {
                'shape_info': shape_info,
                'min': np.min(segmentation.mask_data),
                'max': np.max(segmentation.mask_data),
                'non_zero_count': np.sum(segmentation.mask_data > 0)
            }
            
            primary_overlay = OverlayVisualizationData(
                mask_array=segmentation.mask_data,
                color_rgba=color_rgba,
                anatomical_region=segmentation.anatomical_region,
                opacity=opacity,
                confidence_score=getattr(segmentation, 'confidence_score', 0.0),
                volume_mm3=volume_mm3,
                target_sequence="T2W",  
                original_dimensions=segmentation.mask_data.shape
            )
            
            overlay_data.append(primary_overlay)
            
            if additional_sequences:
                sequence_overlays = await self._create_multi_sequence_overlays(
                    primary_overlay, additional_sequences, original_image
                )
                overlay_data.extend(sequence_overlays)
        
        self._logger.info(f"Prepared {len(overlay_data)} total overlays for visualization")
        return overlay_data
    
    async def _create_multi_sequence_overlays(
        self,
        primary_overlay: OverlayVisualizationData,
        additional_sequences: Dict[str, MedicalImage],
        reference_image: MedicalImage
    ) -> List[OverlayVisualizationData]:
        sequence_overlays = []
        try:            
            for seq_name, seq_image in additional_sequences.items():
                sequence_overlay = primary_overlay.create_sequence_specific_overlay(
                    target_sequence=seq_name,
                    target_dimensions=seq_image.image_data.shape,
                    interpolation_transform=self._calculate_interpolation_transform(seq_image, reference_image)
                )
                
                sequence_overlays.append(sequence_overlay)
            
            
        except Exception as e:
            self._logger.error(f"Failed to create multi-sequence overlays: {e}")
            for seq_name, seq_image in additional_sequences.items():
                try:
                    basic_overlay = OverlayVisualizationData(
                        mask_array=primary_overlay.mask_array,
                        color_rgba=primary_overlay.color_rgba,
                        anatomical_region=primary_overlay.anatomical_region,
                        opacity=primary_overlay.opacity,
                        name=f"{primary_overlay.name} ({seq_name})",
                        confidence_score=primary_overlay.confidence_score,
                        target_sequence=seq_name
                    )
                    sequence_overlays.append(basic_overlay)
                except Exception as fallback_error:
                    self._logger.error(f"Evin fallback failed for {seq_name}: {fallback_error}")
        
        return sequence_overlays
    
    def _calculate_interpolation_transform(
        self, 
        source_image: MedicalImage, 
        reference_image: MedicalImage
    ) -> Dict[str, Any]:
        source_dims = source_image.dimensions
        ref_dims = reference_image.dimensions
        
        if len(source_dims) == len(ref_dims):
            scale_factors = [ref_dims[i] / source_dims[i] for i in range(len(source_dims))]
        else:
            scale_factors = [1.0, 1.0, 1.0]
        
        transform_info = {
            'method': 'interpolation_scaling',
            'scale_factors': scale_factors,
            'source_dimensions': source_dims,
            'target_dimensions': ref_dims,
            'source_spacing': source_image.spacing.to_tuple() if source_image.spacing else (1.0, 1.0, 1.0),
            'target_spacing': reference_image.spacing.to_tuple() if reference_image.spacing else (1.0, 1.0, 1.0)
        }
        
        return transform_info


class AIAnalysisError(Exception):
    pass