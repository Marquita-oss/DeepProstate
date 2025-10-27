import asyncio
import uuid
import logging
import numpy as np
import tempfile
import atexit
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from deepprostate.core.domain.utils.medical_shape_handler import MedicalShapeHandler

from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.core.domain.entities.ai_analysis import (
    AIAnalysisType, AIAnalysisRequest, AIAnalysisResult, 
    OverlayVisualizationData, AIModelStatus
)
from deepprostate.core.domain.entities.segmentation import MedicalSegmentation, AnatomicalRegion

from deepprostate.use_cases.application.services.ai_segmentation_service import AISegmentationService
from deepprostate.frameworks.infrastructure.coordination.medical_format_registry import MedicalFormatRegistry


class AIAnalysisOrchestrator:
    def __init__(
        self,
        segmentation_service: AISegmentationService,
        format_registry: MedicalFormatRegistry,
        temp_storage_path: Path = None
    ):
        self._segmentation_service = segmentation_service
        self._format_registry = format_registry

        # Use system temp directory with auto-cleanup
        if temp_storage_path is None:
            temp_base = Path(tempfile.gettempdir()) / "deep_prostate"
            self._temp_storage = temp_base / str(uuid.uuid4())[:8]
        else:
            self._temp_storage = temp_storage_path

        self._temp_storage.mkdir(parents=True, exist_ok=True)

        # Register cleanup on exit
        atexit.register(self._cleanup_temp_storage)

        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Temp storage: {self._temp_storage}")
        
        self._overlay_colors = {
            AnatomicalRegion.PROSTATE_WHOLE: (0.8, 0.6, 0.4, 0.7),           # Marrón 70% opacidad
            AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: (0.4, 0.8, 0.4, 0.7), # Verde 70% opacidad
            AnatomicalRegion.PROSTATE_TRANSITION_ZONE: (0.4, 0.4, 0.8, 0.7), # Azul 70% opacidad
            AnatomicalRegion.SUSPICIOUS_LESION: (1.0, 0.8, 0.0, 0.7),        # Amarillo 70% opacidad
            AnatomicalRegion.CONFIRMED_CANCER: (1.0, 0.2, 0.2, 0.7),         # Rojo 70% opacidad
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

            # DEBUG: Log what the inference service returned
            self._logger.info(f"Analysis {analysis_id}: Inference returned keys: {list(ai_predictions.keys())}")
            for key, value in ai_predictions.items():
                if isinstance(value, np.ndarray):
                    self._logger.info(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    self._logger.info(f"  - {key}: {type(value)}")
            
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

        # Check if primary image is already a NIfTI file
        primary_path = Path(request.primary_image_path)
        is_nifti = primary_path.suffix in ['.nii', '.gz'] or str(primary_path).endswith('.nii.gz')

        print(f"\n🔍 PRIMARY IMAGE PATH CHECK")
        print(f"   Path: {primary_path}")
        print(f"   Suffix: {primary_path.suffix}")
        print(f"   Is NIfTI: {is_nifti}")

        if is_nifti and primary_path.exists():
            # Use existing NIfTI file directly to avoid double conversion and axis issues
            print(f"   ✅ Using existing NIfTI file directly (no reconversion)")
            temp_nifti_files["t2w"] = primary_path

            # Still need to load as MedicalImage for metadata and original reference
            primary_image = await self._load_medical_image(request.primary_image_path)
            original_images["primary"] = primary_image
        else:
            # Load and convert to NIfTI
            print(f"   🔄 Loading and converting to NIfTI")
            primary_image = await self._load_medical_image(request.primary_image_path)
            original_images["primary"] = primary_image

            t2w_nifti_path = await self._convert_single_image_to_nifti(
                primary_image, f"{analysis_id}_0000"
            )
            temp_nifti_files["t2w"] = t2w_nifti_path

        if request.analysis_type == AIAnalysisType.CSPCA_DETECTION:
            original_images["t2w"] = original_images["primary"]

            # Convert T2W first (reference image)
            t2w_nifti = temp_nifti_files["t2w"]

            for seq_name, seq_path in request.additional_sequences.items():
                if seq_name.upper() in ["ADC", "HBV"]:
                    seq_image = await self._load_medical_image(seq_path)
                    original_images[seq_name.lower()] = seq_image

                    # Convert to NIfTI
                    channel_map = {"adc": "0001", "hbv": "0002"}
                    channel_num = channel_map.get(seq_name.lower(), "0003")
                    seq_nifti_path = await self._convert_single_image_to_nifti(
                        seq_image, f"{analysis_id}_{channel_num}_original"
                    )

                    # REGISTER to T2W space (CRITICAL for csPCa)
                    registered_path = await self._register_to_reference(
                        moving_path=seq_nifti_path,
                        fixed_path=t2w_nifti,
                        output_suffix=f"{analysis_id}_{channel_num}",
                        sequence_name=seq_name.upper()
                    )

                    temp_nifti_files[seq_name.lower()] = registered_path
        
        return original_images, temp_nifti_files
    
    async def _register_to_reference(
        self,
        moving_path: Path,
        fixed_path: Path,
        output_suffix: str,
        sequence_name: str
    ) -> Path:
        """
        Register moving image (ADC/HBV) to fixed image (T2W) space.

        CRITICAL for csPCa detection: ADC and HBV must be in T2W space for nnUNet.

        This method:
        1. Resamples moving image to fixed image grid
        2. Uses LINEAR interpolation (preserves physical values)
        3. Does NOT normalize values (ADC in mm²/s, HBV in signal intensity)

        Args:
            moving_path: Path to image to be registered (ADC or HBV)
            fixed_path: Path to reference image (T2W)
            output_suffix: Suffix for output filename
            sequence_name: Name of sequence for logging (ADC or HBV)

        Returns:
            Path to registered image
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise AIAnalysisError("SimpleITK required for image registration")

        print(f"\n🔧 REGISTERING {sequence_name} TO T2W SPACE")
        print(f"   Fixed (reference): {fixed_path.name}")
        print(f"   Moving (to register): {moving_path.name}")

        try:
            # Read images
            fixed_image = sitk.ReadImage(str(fixed_path))
            moving_image = sitk.ReadImage(str(moving_path))

            print(f"   Fixed size: {fixed_image.GetSize()}, spacing: {fixed_image.GetSpacing()}")
            print(f"   Moving size: {moving_image.GetSize()}, spacing: {moving_image.GetSpacing()}")

            # Simple resampling to fixed grid (identity transform)
            # This assumes images are already roughly aligned (same patient, same session)
            # For more complex registration, would need to compute transform first

            registered_image = sitk.Resample(
                moving_image,                     # Image to resample
                fixed_image,                      # Reference image (defines output grid)
                sitk.Transform(),                 # Identity transform (no rotation/translation)
                sitk.sitkLinear,                  # LINEAR interpolation (preserves values better)
                0.0,                               # Default pixel value for out-of-bounds
                moving_image.GetPixelID()         # Keep original pixel type
            )

            print(f"   Registered size: {registered_image.GetSize()}")
            print(f"   Registered spacing: {registered_image.GetSpacing()}")

            # Save registered image
            output_path = self._temp_storage / f"{output_suffix}.nii.gz"
            sitk.WriteImage(registered_image, str(output_path), useCompression=True)

            print(f"   ✅ Saved registered {sequence_name} to: {output_path.name}\n")

            self._logger.info(
                f"Registered {sequence_name}: {moving_image.GetSize()} → {registered_image.GetSize()}"
            )

            return output_path

        except Exception as e:
            self._logger.error(f"Registration failed for {sequence_name}: {e}", exc_info=True)
            raise AIAnalysisError(f"Failed to register {sequence_name} to T2W space: {e}")

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

        print("\n" + "="*60)
        print("📥 LOADING ORIGINAL IMAGE")
        print("="*60)
        print(f"Image shape: {image_data.shape}")
        print(f"Image dtype: {image_data.dtype}")
        print(f"Spacing: ({medical_image.spacing.x}, {medical_image.spacing.y}, {medical_image.spacing.z})")
        print("="*60 + "\n")

        self._logger.info(f"Converting MedicalImage to NIfTI - Original shape: {image_data.shape}, dtype: {image_data.dtype}")

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
        """
        SIMPLIFIED VERSION: Use predict_from_files() directly - NO file organization needed!

        Steps:
        1. Prepare list of input files
        2. Call predictor.predict_from_files()
        3. Read result
        4. Done!
        """
        primary_file = temp_nifti_files.get("t2w")
        if not primary_file:
            raise AIAnalysisError("No T2W sequence available for inference")

        # Map analysis type to model task
        task_mapping = {
            AIAnalysisType.PROSTATE_GLAND: "prostate_whole",
            AIAnalysisType.ZONES_TZ_PZ: "prostate_zones",
            AIAnalysisType.CSPCA_DETECTION: "lesion_detection"
        }

        model_task = task_mapping.get(analysis_type)
        if not model_task:
            raise AIAnalysisError(f"Unsupported analysis type: {analysis_type}")

        # Prepare output directory
        output_dir = self._temp_storage / f"{analysis_id}_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import SimpleITK as sitk

            # Prepare input files list based on analysis type
            input_files = []

            if analysis_type in [AIAnalysisType.PROSTATE_GLAND, AIAnalysisType.ZONES_TZ_PZ]:
                # Single channel: only T2W
                input_files = [temp_nifti_files["t2w"]]
                print(f"\n📋 INPUT FILES (Single-channel)")
                print(f"   - T2W: {temp_nifti_files['t2w'].name}")

            elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
                # Multi-channel: T2W + ADC + HBV
                if "adc" not in temp_nifti_files or "hbv" not in temp_nifti_files:
                    raise AIAnalysisError("csPCa detection requires T2W, ADC, and HBV sequences")

                input_files = [
                    temp_nifti_files["t2w"],
                    temp_nifti_files["adc"],
                    temp_nifti_files["hbv"]
                ]
                print(f"\n📋 INPUT FILES (Multi-channel)")
                print(f"   - Channel 0 (T2W): {temp_nifti_files['t2w'].name}")
                print(f"   - Channel 1 (ADC): {temp_nifti_files['adc'].name}")
                print(f"   - Channel 2 (HBV): {temp_nifti_files['hbv'].name}")

            # Get AIModelService
            model_service = self._segmentation_service._model_service

            # Run prediction using the SIMPLE method
            self._logger.info(f"Running nnUNet prediction for {model_task}")
            success = model_service.run_nnunet_prediction(
                model_task=model_task,
                input_files=input_files,
                output_dir=output_dir,
                use_gpu=True
            )

            if not success:
                raise AIAnalysisError("nnUNet prediction failed")

            # Find and read prediction result
            # predict_from_files() saves with same name as input (without _0000)
            # For input: temp_abc123_0000.nii.gz -> output: temp_abc123.nii.gz
            input_base_name = temp_nifti_files["t2w"].name.replace("_0000.nii.gz", ".nii.gz")
            prediction_file = output_dir / input_base_name

            if not prediction_file.exists():
                # Try alternative naming
                prediction_files = list(output_dir.glob("*.nii.gz"))
                if prediction_files:
                    prediction_file = prediction_files[0]
                else:
                    raise AIAnalysisError(f"Prediction output not found in {output_dir}")

            print(f"\n📖 READING PREDICTION RESULT")
            print(f"   File: {prediction_file.name}")

            sitk_prediction = sitk.ReadImage(str(prediction_file))
            prediction_array = sitk.GetArrayFromImage(sitk_prediction)

            print(f"   Shape: {prediction_array.shape}")
            print(f"   Unique labels: {np.unique(prediction_array)}\n")

            self._logger.info(f"Prediction completed: {prediction_array.shape}, unique labels: {np.unique(prediction_array)}")

            # Return in expected format
            return {
                "segmentation": prediction_array,
                "metadata": {
                    "analysis_id": analysis_id,
                    "analysis_type": analysis_type.value,
                    "model_task": model_task,
                    "prediction_file": str(prediction_file)
                }
            }

        except ImportError as e:
            raise AIAnalysisError(f"Required library not available: {e}")
        except Exception as e:
            self._logger.error(f"Inference failed: {e}", exc_info=True)
            raise AIAnalysisError(f"Inference execution failed: {e}")
    
    def _normalize_inference_predictions(
        self,
        ai_predictions: Dict[str, Any],
        analysis_type: AIAnalysisType
    ) -> Dict[str, Any]:
        """
        SIMPLIFIED: Normalize and postprocess predictions from nnUNet.

        Since we now use predict_from_files() directly, we always get:
        {"segmentation": ndarray, "metadata": dict}

        This method:
        1. Validates and normalizes dimensions
        2. APPLIES POSTPROCESSING: Maps labels to anatomical structures
        3. Creates confidence map
        """
        self._logger.debug(f"Normalizing predictions for {analysis_type.value}")

        normalized = ai_predictions.copy()

        # Get segmentation array (we always use "segmentation" key now)
        if "segmentation" not in ai_predictions:
            raise AIAnalysisError(
                f"Expected 'segmentation' key in predictions. "
                f"Got keys: {list(ai_predictions.keys())}"
            )

        segmentation = ai_predictions["segmentation"]

        if not isinstance(segmentation, np.ndarray):
            raise AIAnalysisError(
                f"Segmentation must be numpy array, got {type(segmentation)}"
            )

        # Log segmentation info
        self._logger.info(f"Processing segmentation:")
        self._logger.info(f"  Shape: {segmentation.shape}")
        self._logger.info(f"  Dtype: {segmentation.dtype}")
        self._logger.info(f"  Value range: [{np.min(segmentation)}, {np.max(segmentation)}]")
        self._logger.info(f"  Unique labels: {np.unique(segmentation)}")

        print("\n" + "="*60)
        print("🔍 POSTPROCESSING NNUNET PREDICTION")
        print("="*60)
        print(f"Shape: {segmentation.shape}")
        print(f"Dtype: {segmentation.dtype}")
        print(f"Value range: [{np.min(segmentation)}, {np.max(segmentation)}]")
        print(f"Unique labels: {np.unique(segmentation)}")
        print(f"Non-zero voxels: {np.sum(segmentation > 0)}")
        print("="*60 + "\n")

        # Remove extra dimensions if present
        # nnUNet may return (1, Z, Y, X) or (C, Z, Y, X), we need (Z, Y, X)
        original_shape = segmentation.shape
        while segmentation.ndim > 3:
            if segmentation.shape[0] == 1:
                print(f"⚠️  Removing singleton dimension: {segmentation.shape} -> {segmentation.shape[1:]}")
                segmentation = segmentation[0]
            else:
                print(f"⚠️  Taking first channel from {segmentation.shape}")
                segmentation = segmentation[0]

        # Validate 3D
        if segmentation.ndim != 3:
            raise AIAnalysisError(f"Segmentation must be 3D, got shape {segmentation.shape}")

        if original_shape != segmentation.shape:
            print(f"✅ Shape normalized: {original_shape} -> {segmentation.shape}")

        # ========== POSTPROCESSING: Map labels to anatomical structures ==========
        print(f"\n📊 APPLYING POSTPROCESSING FOR {analysis_type.value.upper()}")

        if analysis_type == AIAnalysisType.PROSTATE_GLAND:
            # Map entire segmentation to unified mask
            normalized["unified_mask"] = segmentation
            normalized["prostate_mask"] = segmentation
            print(f"   ✓ Mapped all labels to prostate gland")
            self._logger.info("Postprocessing: Mapped to unified_mask for PROSTATE_GLAND")

        elif analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            # Map labels: 0=background, 1=PZ, 2=TZ
            pz_mask = (segmentation == 1).astype(np.float32)
            tz_mask = (segmentation == 2).astype(np.float32)
            normalized["peripheral_zone"] = pz_mask
            normalized["transition_zone"] = tz_mask
            print(f"   ✓ Label 1 → Peripheral Zone ({np.sum(pz_mask > 0)} voxels)")
            print(f"   ✓ Label 2 → Transition Zone ({np.sum(tz_mask > 0)} voxels)")
            self._logger.info(f"Postprocessing: Mapped zones - PZ: {np.sum(pz_mask > 0)}, TZ: {np.sum(tz_mask > 0)} voxels")

        elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
            # Any non-zero label is a lesion
            lesion_mask = (segmentation > 0).astype(np.float32)
            normalized["suspicious_lesions"] = lesion_mask
            print(f"   ✓ Labels > 0 → Suspicious Lesions ({np.sum(lesion_mask > 0)} voxels)")
            self._logger.info(f"Postprocessing: Mapped lesions - {np.sum(lesion_mask > 0)} voxels")

        # Generate confidence_map
        # Since we set save_probabilities=False, we create a synthetic confidence map
        if "probabilities" in ai_predictions:
            # If probabilities are available, use them
            probabilities = ai_predictions["probabilities"]
            if probabilities.ndim == 4:  # (classes, z, y, x)
                confidence_map = np.max(probabilities, axis=0)
            else:
                confidence_map = probabilities
            normalized["confidence_map"] = confidence_map
            print(f"   ✓ Created confidence map from probabilities")
        else:
            # Create synthetic confidence map (0.95 where mask exists)
            confidence_map = np.where(segmentation > 0, 0.95, 0.0).astype(np.float32)
            normalized["confidence_map"] = confidence_map
            print(f"   ✓ Created synthetic confidence map (0.95 for masked regions)")

        print(f"\n✅ Postprocessing complete. Output keys: {list(normalized.keys())}\n")
        self._logger.info(f"Postprocessing complete. Keys: {list(normalized.keys())}")
        return normalized

    async def _create_domain_segmentations(
        self,
        ai_predictions: Dict[str, Any],
        original_image: MedicalImage,
        analysis_type: AIAnalysisType
    ) -> List[MedicalSegmentation]:
        segmentations = []

        # Normalize predictions to expected format
        normalized_predictions = self._normalize_inference_predictions(ai_predictions, analysis_type)

        if analysis_type == AIAnalysisType.PROSTATE_GLAND:
            mask_data = normalized_predictions.get("unified_mask", normalized_predictions.get("prostate_mask"))
            if mask_data is None:
                raise AIAnalysisError("No mask data found in predictions")

            confidence_map = normalized_predictions.get("confidence_map")
            if confidence_map is None:
                raise AIAnalysisError("No confidence_map found in predictions")

            self._logger.info(f"Creating segmentation:")
            self._logger.info(f"  Mask shape: {mask_data.shape}")
            self._logger.info(f"  Original image shape: {original_image.image_data.shape}")
            self._logger.info(f"  Confidence map shape: {confidence_map.shape}")

            print("\n" + "="*60)
            print("🔍 VALIDATING DIMENSIONS")
            print("="*60)
            print(f"Mask shape:          {mask_data.shape}")
            print(f"Original image shape: {original_image.image_data.shape}")
            print(f"Confidence map shape: {confidence_map.shape}")

            # Validate dimensions match
            if mask_data.shape != original_image.image_data.shape:
                print("❌ DIMENSION MISMATCH DETECTED!")
                print(f"   Mask: {mask_data.shape}")
                print(f"   Image: {original_image.image_data.shape}")

                self._logger.error(f"DIMENSION MISMATCH!")
                self._logger.error(f"  Mask: {mask_data.shape}")
                self._logger.error(f"  Image: {original_image.image_data.shape}")

                # Try to fix if only axes are transposed
                if sorted(mask_data.shape) == sorted(original_image.image_data.shape):
                    print("⚠️  Shapes have same dimensions but different order")
                    print("   This might indicate an axis transposition issue")
                    self._logger.warning("Shapes have same dimensions but different order - this might indicate axis transposition issue")

                raise AIAnalysisError(
                    f"Mask dimensions {mask_data.shape} do not match image dimensions {original_image.image_data.shape}"
                )

            print("✅ Dimensions match!")
            print("="*60 + "\n")

            prostate_segmentation = await self._segmentation_service._conversion_service.create_segmentation_from_prediction(
                mask_data=mask_data,
                confidence_map=confidence_map,
                anatomical_region=AnatomicalRegion.PROSTATE_WHOLE,
                parent_image=original_image,
                preprocessing_metadata=normalized_predictions.get("metadata", {})
            )
            segmentations.append(prostate_segmentation)

            self._logger.info(f"Segmentation created with mask shape: {prostate_segmentation.mask_data.shape}")

        elif analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            zone_segmentations = await self._segmentation_service._conversion_service.create_zone_segmentations(
                normalized_predictions, original_image, normalized_predictions.get("metadata", {})
            )
            segmentations.extend(zone_segmentations)

        elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
            lesion_segmentations = await self._segmentation_service._conversion_service.create_lesion_segmentations(
                normalized_predictions, original_image, normalized_predictions.get("metadata", {})
            )
            segmentations.extend(lesion_segmentations)

        return segmentations
    
    async def _prepare_overlay_visualization(
        self,
        segmentations: List[MedicalSegmentation],
        original_image: MedicalImage,
        opacity: float = 0.7,  # Opacidad por defecto aumentada a 70%
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

    def _cleanup_temp_storage(self):
        """Clean up temporary storage directory on exit."""
        try:
            if self._temp_storage and self._temp_storage.exists():
                shutil.rmtree(self._temp_storage)
                self._logger.info(f"Cleaned up temp storage: {self._temp_storage}")
        except Exception as e:
            self._logger.warning(f"Failed to cleanup temp storage: {e}")


class AIAnalysisError(Exception):
    pass