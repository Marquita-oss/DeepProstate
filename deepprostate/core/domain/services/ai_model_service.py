import logging
from typing import Dict, Any, Optional
from pathlib import Path

from deepprostate.core.domain.entities.segmentation import AnatomicalRegion
from deepprostate.core.domain.services.dynamic_model_config_service import DynamicModelConfigService


class AIModelService:
    def __init__(self, model_config: Dict[str, Any], dynamic_config_service: Optional[DynamicModelConfigService] = None):
        self._model_config = model_config
        self._dynamic_config = dynamic_config_service
        self._logger = logging.getLogger(__name__)

        self._loaded_models: Dict[str, Any] = {}

        self._confidence_thresholds = {
            AnatomicalRegion.PROSTATE_WHOLE: 0.85,
            AnatomicalRegion.SUSPICIOUS_LESION: 0.70,
            AnatomicalRegion.CONFIRMED_CANCER: 0.80,
            AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: 0.75,
            AnatomicalRegion.PROSTATE_TRANSITION_ZONE: 0.75
        }

        if self._dynamic_config:
            self._dynamic_config.add_config_listener(self._on_model_config_changed)

        self._logger.info("AIModelService initialized with Clean Architecture")
    
    def get_confidence_threshold(self, region: AnatomicalRegion) -> float:
        return self._confidence_thresholds.get(region, 0.75)
    
    def set_confidence_threshold(self, region: AnatomicalRegion, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._confidence_thresholds[region] = threshold
        self._logger.info(f"Confidence threshold for {region} set to {threshold}")
    
    def is_model_loaded(self, model_task: str) -> bool:
        return model_task in self._loaded_models
    
    def load_model(self, model_task: str) -> Optional[Any]:
        """
        DEPRECATED: This method is no longer used. Use run_nnunet_prediction() instead.

        The new simplified approach uses nnUNet CLI directly via subprocess,
        avoiding the complexity of loading models into memory.
        """
        self._logger.warning(
            "load_model() is deprecated. Use run_nnunet_prediction() instead."
        )
        return None
    
    def unload_model(self, model_task: str) -> bool:
        if model_task in self._loaded_models:
            del self._loaded_models[model_task]
            self._logger.debug(f"Model {model_task} unloaded from memory")
            return True
        return False
    
    def clear_cache(self) -> None:
        count = len(self._loaded_models)
        self._loaded_models.clear()
        self._logger.debug(f"Model cache cleared ({count} models unloaded)")
    
    def get_loaded_models(self) -> Dict[str, Any]:
        return self._loaded_models.copy()
    
    def get_model_config(self, model_task: str) -> Dict[str, Any]:
        return self._model_config.get(model_task, {})
    
    def _get_model_path(self, model_task: str) -> Path:
        if self._dynamic_config:
            model_config = self._dynamic_config.get_model_config(model_task)
            if model_config and model_config.get("model_path"):
                return Path(model_config["model_path"])

        base_path = Path(self._model_config.get("base_path", "./models"))

        task_paths = {
            "prostate_whole": "Dataset998_PICAI_Prostate/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
            "prostate_zones": "Dataset600_PICAI_PZ_TZ_T2W/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
            "lesion_detection": "Dataset500_PICAI_csPCa/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
        }

        task_path = task_paths.get(model_task, f"Dataset_Unknown_{model_task}")
        return base_path / task_path
    
    def validate_model_availability(self, model_task: str) -> bool:
        model_path = self._get_model_path(model_task)
        return model_path.exists()

    def run_nnunet_prediction(
        self,
        model_task: str,
        input_files: list,
        output_dir: Path,
        use_gpu: bool = True
    ) -> bool:
        """
        Run nnUNet prediction using predict_from_files() - SIMPLE AND DIRECT.

        Args:
            model_task: Task identifier (prostate_whole, prostate_zones, lesion_detection)
            input_files: List of input .nii.gz file paths
                         For single-channel: [t2w_path]
                         For multi-channel: [t2w_path, adc_path, hbv_path]
            output_dir: Directory for prediction output
            use_gpu: Whether to use GPU (default: True)

        Returns:
            True if prediction succeeded, False otherwise
        """
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch
        except ImportError as e:
            self._logger.error(f"Required package not installed: {e}")
            self._logger.error("Install with: pip install nnunetv2 torch")
            return False

        # Get model parameters
        model_params = self._get_model_parameters(model_task)
        if not model_params:
            self._logger.error(f"Unknown model task: {model_task}")
            return False

        # Check GPU availability
        device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        perform_on_gpu = (device == 'cuda')

        if use_gpu and not torch.cuda.is_available():
            self._logger.warning("GPU requested but not available, using CPU")
            device = 'cpu'
            perform_on_gpu = False

        # Prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 RUNNING NNUNET PREDICTION (SIMPLE METHOD)")
        print(f"   Task: {model_task}")
        print(f"   Input files: {[str(f) for f in input_files]}")
        print(f"   Output: {output_dir}")
        print(f"   Model path: {model_params['model_path']}")
        print(f"   Checkpoint: {model_params['checkpoint']}")
        print(f"   Device: {device}")

        try:
            # Initialize predictor
            self._logger.info("Initializing nnUNet predictor...")
            predictor = nnUNetPredictor(
                tile_step_size=model_params.get('tile_step_size', 0.5),
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=perform_on_gpu,
                device=torch.device(device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )

            # Initialize from trained model
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=str(model_params['model_path']),
                use_folds=model_params.get('folds'),
                checkpoint_name=model_params['checkpoint']
            )

            self._logger.info("Running prediction...")
            print(f"   🧠 Model loaded, running prediction...")

            # Determine if we need to save probabilities
            # REQUIRED for lesion_detection because:
            # 1. Uses ensemble (5 folds) - needs .npz files for averaging
            # 2. Has variable spacing - needs resample back to original size
            # NOT required for whole gland/zones (single fold, consistent spacing)
            needs_probabilities = (model_task == "lesion_detection")

            if needs_probabilities:
                print(f"   📊 save_probabilities=True (required for ensemble + resample back)")
            else:
                print(f"   📊 save_probabilities=False (single fold, no ensemble needed)")

            # Execute prediction
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[str(f) for f in input_files]],
                output_folder_or_list_of_truncated_output_files=str(output_dir),
                save_probabilities=needs_probabilities,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )

            self._logger.info("nnUNet prediction completed successfully")
            print(f"   ✅ Prediction completed successfully\n")
            return True

        except Exception as e:
            self._logger.error(f"nnUNet prediction failed: {e}", exc_info=True)
            print(f"   ❌ Prediction failed: {e}\n")
            return False

    def _get_model_parameters(self, model_task: str) -> Optional[Dict[str, Any]]:
        """
        Get model parameters for each task.

        Returns dictionary with: model_path, checkpoint, folds, tile_step_size
        """
        base_path = Path(self._model_config.get("base_path", "./models"))

        params = {
            "prostate_whole": {
                "model_path": base_path / "Dataset998_PICAI_Prostate/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
                "checkpoint": "checkpoint_best.pth",
                "folds": (0,),
                "tile_step_size": 0.5
            },
            "prostate_zones": {
                "model_path": base_path / "Dataset600_PICAI_PZ_TZ_T2W/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
                "checkpoint": "checkpoint_best.pth",
                "folds": (0,),
                "tile_step_size": 0.5
            },
            "lesion_detection": {
                "model_path": base_path / "Dataset500_PICAI_csPCa/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
                "checkpoint": "checkpoint_best.pth",
                "folds": (0, 1, 2, 3, 4),  # Ensemble of 5 folds
                "tile_step_size": 0.3
            }
        }

        return params.get(model_task)

    def _get_nnunet_config_for_task(self, model_task: str) -> Dict[str, Any]:
        """
        DEPRECATED: Configuration is now handled by _get_nnunet_cli_params().

        This method is kept for backward compatibility but is no longer used
        in the simplified subprocess-based approach.
        """
        self._logger.warning(
            "_get_nnunet_config_for_task() is deprecated. "
            "Use _get_nnunet_cli_params() instead."
        )
        return {}

    def update_dynamic_config(self, dynamic_config_service: DynamicModelConfigService) -> None:
        if self._dynamic_config:
            try:
                self._dynamic_config.remove_config_listener(self._on_model_config_changed)
            except Exception as e:
                self._logger.debug(f"Could not remove old config listener: {e}")

        self._dynamic_config = dynamic_config_service
        if self._dynamic_config:
            self._dynamic_config.add_config_listener(self._on_model_config_changed)

        self._logger.info("Dynamic configuration service updated")

    def invalidate_model_cache(self, model_task: Optional[str] = None) -> None:
        if model_task:
            if model_task in self._loaded_models:
                del self._loaded_models[model_task]
                self._logger.info(f"Invalidated cache for model: {model_task}")
        else:
            cleared_count = len(self._loaded_models)
            self._loaded_models.clear()
            self._logger.info(f"Invalidated cache for all models ({cleared_count} models)")

    def get_system_health(self) -> Dict[str, Any]:
        health = {
            "service_status": "active",
            "loaded_models_count": len(self._loaded_models),
            "loaded_models": list(self._loaded_models.keys()),
            "model_availability": {},
            "dynamic_config_available": self._dynamic_config is not None
        }

        if self._dynamic_config:
            available_models = self._dynamic_config.get_available_models()
            for model_task in ["prostate_whole", "prostate_zones", "lesion_detection"]:
                health["model_availability"][model_task] = model_task in available_models
        else:
            for model_task in ["prostate_whole", "prostate_zones", "lesion_detection"]:
                health["model_availability"][model_task] = self.validate_model_availability(model_task)

        return health

    def _on_model_config_changed(self, new_status: Dict[str, Any]) -> None:
        self._logger.info("Model configuration changed, invalidating cache")

        self.invalidate_model_cache()

        if new_status.get("base_path_configured"):
            available_count = new_status.get("available_count", 0)
            total_count = new_status.get("total_models", 0)
            self._logger.info(f"Dynamic config updated: {available_count}/{total_count} models available")
        else:
            self._logger.warning("Dynamic config updated: No base path configured")