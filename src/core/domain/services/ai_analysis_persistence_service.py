import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.core.domain.entities.ai_analysis import AIAnalysisResult, AIAnalysisType
from src.core.domain.entities.medical_image import MedicalImage


class AIAnalysisPersistenceService:
    def __init__(self, storage_path: Path = None):
        self._logger = logging.getLogger(__name__)

        self._storage_path = storage_path or Path("./ai_analysis_data")
        self._storage_path.mkdir(exist_ok=True)

        self._history_file = self._storage_path / "prediction_history.json"
        self._results_cache_file = self._storage_path / "results_cache.json"

        self._prediction_history: List[Dict[str, Any]] = []
        self._current_session_results: Dict[str, AIAnalysisResult] = {}

        self._load_prediction_history()

        self._logger.info("AI Analysis Persistence Service initialized")

    def add_prediction_to_history(self, analysis_result: AIAnalysisResult) -> None:
        try:
            history_record = {
                "timestamp": datetime.now().isoformat(),
                "analysis_id": analysis_result.analysis_id,
                "analysis_type": analysis_result.analysis_type.value,
                "patient_id": analysis_result.patient_context.get("patient_id", "Unknown"),
                "series_description": analysis_result.patient_context.get("series_description", ""),
                "segmentation_count": len(analysis_result.segmentations),
                "overlay_count": len(analysis_result.overlay_data),
                "processing_time_ms": analysis_result.processing_metadata.get("processing_time_ms", 0),
                "confidence_scores": self._extract_confidence_scores(analysis_result),
                "volume_measurements": self._extract_volume_measurements(analysis_result),
                "model_info": analysis_result.processing_metadata.get("model_info", {}),
                "session_id": analysis_result.processing_metadata.get("session_id", "")
            }

            self._prediction_history.append(history_record)

            self._current_session_results[analysis_result.analysis_id] = analysis_result

            self._save_prediction_history()

            self._logger.debug(f"Added prediction to history: {analysis_result.analysis_id}")

        except Exception as e:
            self._logger.error(f"Failed to add prediction to history: {e}")

    def get_prediction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sorted_history = sorted(
            self._prediction_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )

        if limit:
            return sorted_history[:limit]

        return sorted_history

    def get_prediction_by_id(self, analysis_id: str) -> Optional[AIAnalysisResult]:
        return self._current_session_results.get(analysis_id)

    def get_history_summary(self) -> Dict[str, Any]:
        if not self._prediction_history:
            return {
                "total_predictions": 0,
                "analysis_types": {},
                "date_range": None,
                "avg_processing_time_ms": 0
            }

        analysis_types = {}
        processing_times = []

        for record in self._prediction_history:
            analysis_type = record["analysis_type"]
            analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1

            if record["processing_time_ms"]:
                processing_times.append(record["processing_time_ms"])

        timestamps = [record["timestamp"] for record in self._prediction_history]
        date_range = {
            "earliest": min(timestamps),
            "latest": max(timestamps)
        } if timestamps else None

        return {
            "total_predictions": len(self._prediction_history),
            "analysis_types": analysis_types,
            "date_range": date_range,
            "avg_processing_time_ms": sum(processing_times) / len(processing_times) if processing_times else 0
        }

    def filter_history(self,
                      analysis_type: Optional[AIAnalysisType] = None,
                      patient_id: Optional[str] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None) -> List[Dict[str, Any]]:
        filtered_history = self._prediction_history.copy()

        if analysis_type:
            filtered_history = [
                record for record in filtered_history
                if record["analysis_type"] == analysis_type.value
            ]

        if patient_id:
            filtered_history = [
                record for record in filtered_history
                if patient_id.lower() in record["patient_id"].lower()
            ]

        if date_from:
            filtered_history = [
                record for record in filtered_history
                if datetime.fromisoformat(record["timestamp"]) >= date_from
            ]

        if date_to:
            filtered_history = [
                record for record in filtered_history
                if datetime.fromisoformat(record["timestamp"]) <= date_to
            ]

        return sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)

    def clear_history(self) -> bool:
        try:
            self._prediction_history.clear()
            self._current_session_results.clear()

            if self._history_file.exists():
                self._history_file.unlink()

            if self._results_cache_file.exists():
                self._results_cache_file.unlink()

            self._logger.info("Prediction history cleared")
            return True

        except Exception as e:
            self._logger.error(f"Failed to clear history: {e}")
            return False

    def export_history(self, export_path: Path, format: str = "json") -> bool:
        try:
            if format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump({
                        "export_metadata": {
                            "export_date": datetime.now().isoformat(),
                            "total_records": len(self._prediction_history),
                            "version": "v17_clean_architecture"
                        },
                        "prediction_history": self._prediction_history
                    }, f, indent=2)

            elif format.lower() == "csv":
                import csv
                with open(export_path, 'w', newline='') as f:
                    if self._prediction_history:
                        writer = csv.DictWriter(f, fieldnames=self._prediction_history[0].keys())
                        writer.writeheader()
                        writer.writerows(self._prediction_history)

            self._logger.info(f"History exported to {export_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to export history: {e}")
            return False

    def _load_prediction_history(self) -> None:
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    self._prediction_history = data
                elif isinstance(data, dict) and "prediction_history" in data:
                    self._prediction_history = data["prediction_history"]

                self._logger.debug(f"Loaded {len(self._prediction_history)} history records")
            else:
                self._logger.debug("No existing history file found")

        except Exception as e:
            self._logger.error(f"Failed to load prediction history: {e}")
            self._prediction_history = []

    def _save_prediction_history(self) -> None:
        try:
            data = {
                "metadata": {
                    "version": "v17_clean_architecture",
                    "last_updated": datetime.now().isoformat(),
                    "total_records": len(self._prediction_history)
                },
                "prediction_history": self._prediction_history
            }

            with open(self._history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self._logger.error(f"Failed to save prediction history: {e}")

    def _extract_confidence_scores(self, analysis_result: AIAnalysisResult) -> Dict[str, float]:
        confidence_scores = {}

        for segmentation in analysis_result.segmentations:
            if hasattr(segmentation, 'confidence_level') and segmentation.confidence_level:
                confidence_map = {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.5
                }
                confidence_scores[segmentation.anatomical_region.value] = confidence_map.get(
                    segmentation.confidence_level.value.lower(), 0.5
                )

        return confidence_scores

    def _extract_volume_measurements(self, analysis_result: AIAnalysisResult) -> Dict[str, float]:
        volume_measurements = {}

        for overlay in analysis_result.overlay_data:
            if hasattr(overlay, 'volume_mm3') and overlay.volume_mm3:
                volume_measurements[overlay.anatomical_region.value] = overlay.volume_mm3

        return volume_measurements