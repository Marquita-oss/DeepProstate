import logging
import asyncio
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from src.core.domain.services.ai_model_service import AIModelService


class AIInferenceService:
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
        self._logger = logging.getLogger(__name__)

        self._inference_timeout = 300 
        self._batch_size = 1 

        self._logger.info("AIInferenceService initialized for production mode")

    async def run_inference(
        self,
        prepared_data: Dict[str, Any],
        model_task: str
    ) -> Dict[str, np.ndarray]:
        try:
            self._logger.info(f"Starting production inference for model task: {model_task}")

            if not self._model_service.validate_model_availability(model_task):
                raise AIInferenceError(f"Model {model_task} not available")

            model = self._model_service.load_model(model_task)
            if model is None:
                raise AIInferenceError(f"Failed to load model {model_task}")

            predictions = await self._execute_real_inference(
                prepared_data, model_task, model
            )

            self._logger.info(f"Production inference completed for {model_task}")
            return predictions

        except Exception as e:
            self._logger.error(f"Inference failed for {model_task}: {e}")
            raise AIInferenceError(f"Inference failed: {e}")

    async def _execute_real_inference(
        self,
        prepared_data: Dict[str, Any],
        model_task: str,
        model: Any
    ) -> Dict[str, np.ndarray]:
        try:
            self._logger.info(f"Executing real nnUNet inference for {model_task} with original data")

            input_data = prepared_data["original_data"]
            properties = prepared_data.get("properties", {})

            prediction_task = asyncio.create_task(
                self._run_nnunet_prediction(model, input_data, properties)
            )

            try:
                predictions = await asyncio.wait_for(
                    prediction_task,
                    timeout=self._inference_timeout
                )
            except asyncio.TimeoutError:
                prediction_task.cancel()
                raise AIInferenceError(
                    f"Inference timeout after {self._inference_timeout} seconds"
                )

            # Validate resultados
            if not self._validate_predictions(predictions, model_task):
                raise AIInferenceError("Invalid prediction results")

            self._logger.info(f"Real inference completed successfully for {model_task}")
            return predictions

        except Exception as e:
            self._logger.error(f"Real inference failed: {e}")
            raise AIInferenceError(f"Real inference failed: {e}")

    async def _run_nnunet_prediction(
        self,
        model: Any,
        input_data: np.ndarray,
        properties: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        try:
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self._predict_with_nnunet,
                model,
                input_data,
                properties
            )

            return predictions

        except Exception as e:
            self._logger.error(f"nnUNet prediction failed: {e}")
            raise

    def _predict_with_nnunet(
        self,
        model: Any,
        input_data: np.ndarray,
        properties: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        try:
            try:
                from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            except ImportError:
                raise AIInferenceError(
                    "nnUNetv2 not installed. Install with: pip install nnunetv2"
                )

            if hasattr(model, 'predict_single_npy_array'):
                prediction = model.predict_single_npy_array(
                    input_data, properties, None, None, False
                )
            elif hasattr(model, 'predict'):
                prediction = model.predict(input_data)
            else:
                raise AIInferenceError("Model does not have a compatible predict method")

            if isinstance(prediction, np.ndarray):
                result = {"segmentation": prediction}
            elif isinstance(prediction, tuple) and len(prediction) >= 1:
                result = {"segmentation": prediction[0]}
                if len(prediction) > 1:
                    result["probabilities"] = prediction[1]
            else:
                raise AIInferenceError(f"Unexpected prediction format: {type(prediction)}")

            self._logger.info("nnUNet prediction completed successfully")
            return result

        except Exception as e:
            self._logger.error(f"nnUNet prediction error: {e}")
            raise AIInferenceError(f"nnUNet prediction failed: {e}")

    def _validate_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        model_task: str
    ) -> bool:
        try:
            if not isinstance(predictions, dict):
                self._logger.error("Predictions must be a dictionary")
                return False

            if "segmentation" not in predictions:
                self._logger.error("Missing 'segmentation' in predictions")
                return False

            segmentation = predictions["segmentation"]
            if not isinstance(segmentation, np.ndarray):
                self._logger.error("Segmentation must be a numpy array")
                return False

            if segmentation.size == 0:
                self._logger.error("Empty segmentation array")
                return False

            if model_task == "prostate_whole":
                unique_labels = np.unique(segmentation)
                if not all(label in [0, 1] for label in unique_labels):
                    self._logger.warning(f"Unexpected labels in prostate segmentation: {unique_labels}")

            elif model_task == "prostate_zones":
                unique_labels = np.unique(segmentation)
                if not all(label in [0, 1, 2] for label in unique_labels):
                    self._logger.warning(f"Unexpected labels in zones segmentation: {unique_labels}")

            elif model_task == "lesion_detection":
                unique_labels = np.unique(segmentation)
                if len(unique_labels) < 1:
                    self._logger.warning("No labels found in lesion detection")

            self._logger.info(f"Predictions validated successfully for {model_task}")
            return True

        except Exception as e:
            self._logger.error(f"Prediction validation failed: {e}")
            return False


class AIInferenceError(Exception):
    pass


def create_ai_inference_service(model_service: AIModelService) -> AIInferenceService:
    return AIInferenceService(model_service)