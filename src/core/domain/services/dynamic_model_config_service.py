import logging
import json
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path
from datetime import datetime

from src.core.domain.services.model_metadata_service import ModelMetadataService, ModelMetadata


class DynamicModelConfigService:
    def __init__(self, metadata_service: Optional[ModelMetadataService] = None):
        self._logger = logging.getLogger(__name__)

        self._base_model_path: Optional[Path] = None
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._model_metadata: Dict[str, ModelMetadata] = {}
        self._config_listeners: List[Callable[[Dict[str, Any]], None]] = []

        self._metadata_service = metadata_service or ModelMetadataService()

        self._expected_model_structure = {
            "prostate_whole": "Dataset998_PICAI_Prostate/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
            "prostate_zones": "Dataset600_PICAI_PZ_TZ_T2W/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
            "lesion_detection": "Dataset500_PICAI_csPCa/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
        }

        self._logger.info("DynamicModelConfigService initialized with ModelMetadataService")

    def set_base_model_path(self, model_directory: Path) -> bool:
        try:
            model_path = Path(model_directory).resolve()

            if not model_path.exists():
                self._logger.error(f"Model directory does not exist: {model_path}")
                return False

            if not model_path.is_dir():
                self._logger.error(f"Model path is not a directory: {model_path}")
                return False

            validation_result = self._validate_model_structure(model_path)
            if not validation_result["valid"]:
                self._logger.error(f"Invalid model structure: {validation_result['errors']}")
                return False

            self._base_model_path = model_path
            self._update_model_configs()

            self._notify_config_change()

            self._logger.info(f"Base model path set successfully: {model_path}")
            self._logger.info(f"Available models: {list(self._model_configs.keys())}")

            return True

        except Exception as e:
            self._logger.error(f"Failed to set base model path: {e}")
            return False

    def get_base_model_path(self) -> Optional[Path]:
        """Get the current base model path."""
        return self._base_model_path

    def get_model_config(self, model_task: str) -> Optional[Dict[str, Any]]:
        return self._model_configs.get(model_task)

    def get_available_models(self) -> List[str]:
        return list(self._model_configs.keys())

    def is_model_available(self, model_task: str) -> bool:
        config = self._model_configs.get(model_task)
        if not config:
            return False

        model_path = Path(config["model_path"])
        return model_path.exists() and model_path.is_dir()

    def get_model_status(self) -> Dict[str, Any]:
        status = {
            "base_path_configured": self._base_model_path is not None,
            "base_path": str(self._base_model_path) if self._base_model_path else None,
            "models_available": {},
            "total_models": len(self._expected_model_structure),
            "available_count": 0,
            "last_updated": datetime.now().isoformat()
        }

        for task in self._expected_model_structure.keys():
            is_available = self.is_model_available(task)
            status["models_available"][task] = is_available
            if is_available:
                status["available_count"] += 1

        return status

    def add_config_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        if listener not in self._config_listeners:
            self._config_listeners.append(listener)

    def remove_config_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        if listener in self._config_listeners:
            self._config_listeners.remove(listener)

    def get_model_metadata(self, model_task: str) -> Optional[ModelMetadata]:
        return self._model_metadata.get(model_task)

    def get_all_metadata(self) -> Dict[str, ModelMetadata]:
        return self._model_metadata.copy()

    def clear_configuration(self) -> None:
        self._base_model_path = None
        self._model_configs.clear()
        self._model_metadata.clear()
        self._metadata_service.clear_cache()
        self._notify_config_change()
        self._logger.info("Model configuration cleared")

    def export_configuration(self) -> Dict[str, Any]:
        return {
            "base_model_path": str(self._base_model_path) if self._base_model_path else None,
            "model_configs": self._model_configs,
            "export_timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }

    def import_configuration(self, config: Dict[str, Any]) -> bool:
        try:
            if "base_model_path" in config and config["base_model_path"]:
                return self.set_base_model_path(Path(config["base_model_path"]))
            return True
        except Exception as e:
            self._logger.error(f"Failed to import configuration: {e}")
            return False

    def _validate_model_structure(self, base_path: Path) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "found_models": []
        }

        try:
            for task, relative_path in self._expected_model_structure.items():
                model_path = base_path / relative_path

                if model_path.exists() and model_path.is_dir():
                    result["found_models"].append(task)
                    self._logger.debug(f"Found model for {task}: {model_path}")
                else:
                    error_msg = f"Missing model for {task}: {model_path}"
                    result["errors"].append(error_msg)
                    self._logger.warning(error_msg)

            if not result["found_models"]:
                result["valid"] = False
                result["errors"].append("No valid models found in directory")
            elif len(result["found_models"]) < len(self._expected_model_structure):
                result["warnings"].append(f"Only {len(result['found_models'])} of {len(self._expected_model_structure)} models found")

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {e}")

        return result

    def _update_model_configs(self) -> None:
        self._model_configs.clear()
        self._model_metadata.clear()

        if not self._base_model_path:
            return

        for task, relative_path in self._expected_model_structure.items():
            model_path = self._base_model_path / relative_path

            if model_path.exists() and model_path.is_dir():
                metadata = self._metadata_service.load_model_metadata(model_path)

                if metadata:
                    self._model_metadata[task] = metadata
                    self._model_configs[task] = {
                        "task": task,
                        "model_path": str(model_path),
                        "relative_path": relative_path,
                        "available": True,
                        "last_checked": datetime.now().isoformat(),
                        "dataset_id": metadata.dataset_id,
                        "num_input_channels": metadata.num_input_channels,
                        "input_channels": metadata.get_input_channel_list(),
                        "num_classes": metadata.num_classes,
                        "labels": metadata.get_label_names(),
                        "available_folds": metadata.available_folds,
                        "num_folds": metadata.num_folds,
                        "requires_ensemble": metadata.requires_ensemble(),
                        "plans_name": metadata.plans_name
                    }

                    self._logger.info(
                        f"Configured {task}: {metadata.dataset_id} with {metadata.num_folds} fold(s), "
                        f"ensemble={'yes' if metadata.requires_ensemble() else 'no'}"
                    )
                else:
                    self._logger.info(f"Could not load metadata for {task}, using basic config")
                    self._model_configs[task] = {
                        "task": task,
                        "model_path": str(model_path),
                        "relative_path": relative_path,
                        "available": True,
                        "last_checked": datetime.now().isoformat(),
                        "metadata_loaded": False
                    }

    def _notify_config_change(self) -> None:
        current_status = self.get_model_status()

        for listener in self._config_listeners:
            try:
                listener(current_status)
            except Exception as e:
                self._logger.error(f"Error notifying config listener: {e}")


class DynamicModelConfigError(Exception):
    pass