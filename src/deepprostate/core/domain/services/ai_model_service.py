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
        if self.is_model_loaded(model_task):
            self._logger.debug(f"Model {model_task} already loaded")
            return self._loaded_models[model_task]
        
        try:
            model_path = self._get_model_path(model_task)
            if not model_path.exists():
                self._logger.error(f"Model path not found: {model_path}")
                return None

            model = self._load_nnunet_model(model_path, model_task)
            if model is not None:
                self._loaded_models[model_task] = model
                self._logger.info(f"Model {model_task} loaded successfully from {model_path}")
                return model
            else:
                self._logger.error(f"Failed to load nnUNet model from {model_path}")
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to load model {model_task}: {e}")
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

    def _load_nnunet_model(self, model_path: Path, model_task: str) -> Optional[Any]:
        try:
            try:
                from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            except ImportError:
                self._logger.error("nnUNetv2 not installed. Install with: pip install nnunetv2")
                return None

            model_config = self._get_nnunet_config_for_task(model_task)

            predictor = nnUNetPredictor(
                tile_step_size=model_config.get("tile_step_size", 0.5),
                use_gaussian=model_config.get("use_gaussian", True),
                use_mirroring=model_config.get("use_mirroring", True),
                perform_everything_on_device=model_config.get("use_gpu", True), 
                device=model_config.get("device", "cuda"),
                verbose=model_config.get("verbose", False),
                verbose_preprocessing=model_config.get("verbose_preprocessing", False),
                allow_tqdm=model_config.get("allow_tqdm", True)
            )

            use_folds_config = model_config.get("use_folds", None)
            if use_folds_config is not None:
                use_folds_config = tuple(use_folds_config) 

            predictor.initialize_from_trained_model_folder(
                str(model_path.parent),
                use_folds=use_folds_config,
                checkpoint_name=model_config.get("checkpoint_name", "checkpoint_best.pth") 
            )

            self._logger.info(f"nnUNet model {model_task} loaded successfully")
            return predictor

        except Exception as e:
            self._logger.error(f"Failed to load nnUNet model {model_task}: {e}")
            return None

    def _get_nnunet_config_for_task(self, model_task: str) -> Dict[str, Any]:
        base_config = {
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "use_mirroring": True,
            "use_gpu": True,
            "device": "cuda",
            "verbose": False,
            "verbose_preprocessing": False,
            "allow_tqdm": True,
            "use_folds": None, 
            "checkpoint_name": "checkpoint_best.pth" 
        }

        task_configs = {
            "prostate_whole": {
                "tile_step_size": 0.5,
                "use_mirroring": True,
                "use_folds": [0] 
            },
            "prostate_zones": {
                "tile_step_size": 0.5,
                "use_mirroring": True,
                "use_folds": [0] 
            },
            "lesion_detection": {
                "tile_step_size": 0.3,  
                "use_mirroring": True,
                "use_folds": [0, 1, 2, 3, 4]  
            }
        }

        config = base_config.copy()
        config.update(task_configs.get(model_task, {}))

        if self._dynamic_config:
            metadata = self._dynamic_config.get_model_metadata(model_task)
            if metadata:
                config["use_folds"] = metadata.available_folds if metadata.num_folds > 0 else None

                self._logger.info(
                    f"Configured {model_task} with {len(metadata.available_folds)} fold(s): "
                    f"{metadata.available_folds}"
                )

        return config

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