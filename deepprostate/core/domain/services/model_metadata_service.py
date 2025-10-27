import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelMetadata:
    dataset_id: str  
    dataset_name: str 
    channel_names: Dict[str, str] 
    num_input_channels: int  
    labels: Dict[str, int]  
    num_classes: int 
    description: Optional[str] = None
    reference: Optional[str] = None

    plans_name: str = ""  
    image_reader_writer: str = "SimpleITKIO"
    original_median_spacing: Optional[List[float]] = None
    original_median_shape: Optional[List[int]] = None
    batch_size: Optional[int] = None
    patch_size: Optional[List[int]] = None
    normalization_schemes: Optional[List[str]] = None

    available_folds: List[int] = None 
    num_folds: int = 0 
    model_path: Optional[Path] = None

    loaded_at: str = ""

    def __post_init__(self):
        if self.available_folds is None:
            object.__setattr__(self, 'available_folds', [])
        if not self.loaded_at:
            object.__setattr__(self, 'loaded_at', datetime.now().isoformat())

    def requires_ensemble(self) -> bool:
        return self.num_folds > 1

    def get_input_channel_list(self) -> List[str]:
        return [self.channel_names[str(i)] for i in range(self.num_input_channels)]

    def get_label_names(self) -> List[str]:
        return list(self.labels.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "channel_names": self.channel_names,
            "num_input_channels": self.num_input_channels,
            "labels": self.labels,
            "num_classes": self.num_classes,
            "description": self.description,
            "reference": self.reference,
            "plans_name": self.plans_name,
            "image_reader_writer": self.image_reader_writer,
            "original_median_spacing": self.original_median_spacing,
            "original_median_shape": self.original_median_shape,
            "batch_size": self.batch_size,
            "patch_size": self.patch_size,
            "normalization_schemes": self.normalization_schemes,
            "available_folds": self.available_folds,
            "num_folds": self.num_folds,
            "model_path": str(self.model_path) if self.model_path else None,
            "loaded_at": self.loaded_at,
            "requires_ensemble": self.requires_ensemble()
        }


class ModelMetadataService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._metadata_cache: Dict[str, ModelMetadata] = {}

    def load_model_metadata(self, model_path: Path) -> Optional[ModelMetadata]:
        try:
            model_path = Path(model_path).resolve()

            cache_key = str(model_path)
            if cache_key in self._metadata_cache:
                self._logger.debug(f"Returning cached metadata for {model_path.name}")
                return self._metadata_cache[cache_key]

            if not model_path.exists():
                self._logger.error(f"Model path does not exist: {model_path}")
                return None

            dataset_json_path = model_path / "dataset.json"
            if not dataset_json_path.exists():
                self._logger.error(f"dataset.json not found in {model_path}")
                return None

            dataset_data = self._load_json(dataset_json_path)
            if not dataset_data:
                return None

            plans_json_path = model_path / "plans.json"
            plans_data = {}
            if plans_json_path.exists():
                plans_data = self._load_json(plans_json_path) or {}
            else:
                self._logger.warning(f"plans.json not found in {model_path}, using defaults")

            available_folds = self._detect_folds(model_path)

            metadata = self._parse_metadata(dataset_data, plans_data, available_folds, model_path)

            self._metadata_cache[cache_key] = metadata

            self._logger.info(
                f"Loaded metadata for {metadata.dataset_id}: "
                f"{metadata.num_input_channels} channels, "
                f"{metadata.num_classes} classes, "
                f"{metadata.num_folds} fold(s)"
            )

            return metadata

        except Exception as e:
            self._logger.error(f"Failed to load model metadata from {model_path}: {e}")
            return None

    def _load_json(self, json_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._logger.error(f"Failed to load JSON from {json_path}: {e}")
            return None

    def _detect_folds(self, model_path: Path) -> List[int]:
        folds = []
        try:
            for item in model_path.iterdir():
                if item.is_dir() and item.name.startswith("fold_"):
                    try:
                        fold_num = int(item.name.split("_")[1])
                        folds.append(fold_num)
                    except (ValueError, IndexError):
                        self._logger.warning(f"Invalid fold directory name: {item.name}")

            folds.sort()
            self._logger.debug(f"Detected folds in {model_path.name}: {folds}")

        except Exception as e:
            self._logger.error(f"Failed to detect folds in {model_path}: {e}")

        return folds

    def _parse_metadata(
        self,
        dataset_data: Dict[str, Any],
        plans_data: Dict[str, Any],
        available_folds: List[int],
        model_path: Path
    ) -> ModelMetadata:
        channel_names = dataset_data.get("channel_names", {})
        num_input_channels = len(channel_names)

        labels = dataset_data.get("labels", {})
        num_classes = len(labels)

        dataset_id = dataset_data.get("name", model_path.parent.parent.name)
        if not dataset_id.startswith("Dataset"):
            dataset_id = model_path.parent.name

        plans_name = plans_data.get("plans_name", "")
        image_reader_writer = plans_data.get("image_reader_writer", "SimpleITKIO")
        original_median_spacing = plans_data.get("original_median_spacing_after_transp")
        original_median_shape = plans_data.get("original_median_shape_after_transp")

        batch_size = None
        patch_size = None
        normalization_schemes = None

        configurations = plans_data.get("configurations", {})
        fullres_config = configurations.get("3d_fullres", {})
        if fullres_config:
            batch_size = fullres_config.get("batch_size")
            patch_size = fullres_config.get("patch_size")
            normalization_schemes = fullres_config.get("normalization_schemes")

        return ModelMetadata(
            dataset_id=dataset_id,
            dataset_name=dataset_data.get("name", dataset_id),
            channel_names=channel_names,
            num_input_channels=num_input_channels,
            labels=labels,
            num_classes=num_classes,
            description=dataset_data.get("description"),
            reference=dataset_data.get("reference"),
            plans_name=plans_name,
            image_reader_writer=image_reader_writer,
            original_median_spacing=original_median_spacing,
            original_median_shape=original_median_shape,
            batch_size=batch_size,
            patch_size=patch_size,
            normalization_schemes=normalization_schemes,
            available_folds=available_folds,
            num_folds=len(available_folds),
            model_path=model_path
        )

    def clear_cache(self) -> None:
        count = len(self._metadata_cache)
        self._metadata_cache.clear()
        self._logger.debug(f"Metadata cache cleared ({count} entries)")

    def get_cached_metadata(self, model_path: Path) -> Optional[ModelMetadata]:
        return self._metadata_cache.get(str(Path(model_path).resolve()))

    def validate_metadata(self, metadata: ModelMetadata) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        if not metadata.dataset_id:
            result["errors"].append("Missing dataset_id")
            result["valid"] = False

        if metadata.num_input_channels == 0:
            result["errors"].append("No input channels defined")
            result["valid"] = False

        if metadata.num_classes == 0:
            result["errors"].append("No output classes defined")
            result["valid"] = False

        if metadata.num_folds == 0:
            result["errors"].append("No fold directories found")
            result["valid"] = False

        if not metadata.plans_name:
            result["warnings"].append("No plans_name found in plans.json")

        if not metadata.normalization_schemes:
            result["warnings"].append("No normalization schemes specified")

        if metadata.normalization_schemes:
            if len(metadata.normalization_schemes) != metadata.num_input_channels:
                result["warnings"].append(
                    f"Normalization schemes count ({len(metadata.normalization_schemes)}) "
                    f"does not match input channels ({metadata.num_input_channels})"
                )

        return result
