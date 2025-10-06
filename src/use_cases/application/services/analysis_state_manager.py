from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

from ....core.domain.entities.analysis_state import AnalysisState, AnalysisStatus
from ....core.domain.entities.ai_analysis import AIAnalysisType, AIAnalysisResult, AISequenceRequirement
from ....core.domain.entities.medical_image import MedicalImage
from ....core.domain.services.analysis_state_machine import AnalysisStateMachine, TransitionError


class AnalysisStateManager:
    def __init__(
        self,
        state_machine: Optional[AnalysisStateMachine] = None,
        validation_service: Optional[any] = None
    ):
        self._state_machine = state_machine or AnalysisStateMachine()
        self._validation_service = validation_service
        self._state_observers: List[callable] = []

        self._state_machine.add_observer(self._on_state_changed)

    def get_current_state(self) -> AnalysisState:
        return self._state_machine.current_state

    def get_current_status(self) -> AnalysisStatus:
        return self._state_machine.current_state.status

    def update_models_status(self, available: bool, error: Optional[str] = None) -> AnalysisState:
        current_state = self._state_machine.current_state

        if available:
            new_state = AnalysisState(
                status=AnalysisStatus.MODELS_LOADED,
                models_available=True,
                models_loading_error=None,
                selected_analysis_type=current_state.selected_analysis_type,
                current_medical_image=current_state.current_medical_image,
                case_explicitly_selected=current_state.case_explicitly_selected,
                available_sequences=current_state.available_sequences,
                missing_sequences=current_state.missing_sequences,
                sequences_validated=current_state.sequences_validated
            )

            return self._state_machine.transition_to(AnalysisStatus.MODELS_LOADED, new_state)

        else:
            new_state = AnalysisState(
                status=AnalysisStatus.FAILED,
                models_available=False,
                models_loading_error=error or "Failed to load AI models",
                error_message=error or "Failed to load AI models"
            )

            return self._state_machine.transition_to(AnalysisStatus.FAILED, new_state)

    def select_analysis_type(self, analysis_type: AIAnalysisType) -> AnalysisState:
        if analysis_type is None:
            raise ValueError("Analysis type cannot be None")

        current_state = self._state_machine.current_state

        if not current_state.models_available:
            raise TransitionError("Cannot select analysis type: models not loaded")

        new_state = AnalysisState(
            status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
            models_available=current_state.models_available,
            selected_analysis_type=analysis_type,
            current_medical_image=None,
            case_explicitly_selected=False,
            available_sequences={},
            missing_sequences=[],
            sequences_validated=False
        )

        return self._state_machine.transition_to(AnalysisStatus.ANALYSIS_TYPE_SELECTED, new_state)

    def select_case(self, medical_image: MedicalImage, explicit: bool = True) -> AnalysisState:
        if medical_image is None:
            raise ValueError("Medical image cannot be None")

        current_state = self._state_machine.current_state

        if current_state.selected_analysis_type is None:
            raise TransitionError("Cannot select case: analysis type not selected")

        new_state = AnalysisState(
            status=AnalysisStatus.CASE_SELECTED,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=medical_image,
            case_explicitly_selected=explicit,
            available_sequences={},
            missing_sequences=[],
            sequences_validated=False
        )

        return self._state_machine.transition_to(AnalysisStatus.CASE_SELECTED, new_state)

    def validate_sequences(
        self,
        available_sequences: Dict[str, Path],
        analysis_type: Optional[AIAnalysisType] = None
    ) -> AnalysisState:
        current_state = self._state_machine.current_state

        if current_state.current_medical_image is None:
            raise TransitionError("Cannot validate sequences: no case selected")

        target_analysis_type = analysis_type or current_state.selected_analysis_type
        if target_analysis_type is None:
            raise TransitionError("Cannot validate sequences: no analysis type selected")

        requirements = AISequenceRequirement.get_requirements_for_analysis(target_analysis_type)
        missing_sequences = []
        for req in requirements:
            if req.is_required and req.sequence_name not in available_sequences:
                missing_sequences.append(req.sequence_name)

        sequences_valid = len(missing_sequences) == 0
        if sequences_valid:
            new_state = AnalysisState(
                status=AnalysisStatus.SEQUENCES_READY,
                models_available=current_state.models_available,
                selected_analysis_type=current_state.selected_analysis_type,
                current_medical_image=current_state.current_medical_image,
                case_explicitly_selected=current_state.case_explicitly_selected,
                available_sequences=available_sequences,
                missing_sequences=[],
                sequences_validated=True
            )

            return self._state_machine.transition_to(AnalysisStatus.SEQUENCES_READY, new_state)

        else:
            new_state = AnalysisState(
                status=AnalysisStatus.CASE_SELECTED,
                models_available=current_state.models_available,
                selected_analysis_type=current_state.selected_analysis_type,
                current_medical_image=current_state.current_medical_image,
                case_explicitly_selected=current_state.case_explicitly_selected,
                available_sequences=available_sequences,
                missing_sequences=missing_sequences,
                sequences_validated=True,
                validation_errors=[f"Missing required sequences: {', '.join(missing_sequences)}"]
            )
            self._state_machine._current_state = new_state
            return new_state

    def mark_ready_to_run(self) -> AnalysisState:
        current_state = self._state_machine.current_state

        if not current_state.can_run_analysis():
            errors = current_state.get_validation_errors()
            error_msg = "Cannot mark as ready: " + "; ".join(errors)
            raise TransitionError(error_msg)

        new_state = AnalysisState(
            status=AnalysisStatus.READY_TO_RUN,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=current_state.current_medical_image,
            case_explicitly_selected=current_state.case_explicitly_selected,
            available_sequences=current_state.available_sequences,
            missing_sequences=current_state.missing_sequences,
            sequences_validated=current_state.sequences_validated,
            worker_thread_active=False
        )

        return self._state_machine.transition_to(AnalysisStatus.READY_TO_RUN, new_state)

    def start_analysis(self) -> AnalysisState:
        current_state = self._state_machine.current_state

        if not current_state.can_run_analysis():
            errors = current_state.get_validation_errors()
            error_msg = "Cannot start analysis: " + "; ".join(errors)
            raise TransitionError(error_msg)

        new_state = AnalysisState(
            status=AnalysisStatus.RUNNING,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=current_state.current_medical_image,
            case_explicitly_selected=current_state.case_explicitly_selected,
            available_sequences=current_state.available_sequences,
            missing_sequences=current_state.missing_sequences,
            sequences_validated=current_state.sequences_validated,
            worker_thread_active=True,
            progress_percentage=0.0,
            current_operation="Starting analysis..."
        )

        return self._state_machine.transition_to(AnalysisStatus.RUNNING, new_state)

    def update_progress(self, percentage: float, operation: Optional[str] = None) -> AnalysisState:
        current_state = self._state_machine.current_state

        if not current_state.is_running():
            raise TransitionError("Cannot update progress: analysis not running")

        from dataclasses import replace
        new_state = replace(
            current_state,
            progress_percentage=max(0.0, min(100.0, percentage)),
            current_operation=operation or current_state.current_operation,
            last_updated=datetime.now()
        )

        self._state_machine._current_state = new_state
        return new_state

    def complete_analysis(self, result: AIAnalysisResult) -> AnalysisState:
        if result is None:
            raise ValueError("Analysis result cannot be None")

        current_state = self._state_machine.current_state

        if not current_state.is_running():
            raise TransitionError("Cannot complete: analysis not running")

        new_state = AnalysisState(
            status=AnalysisStatus.COMPLETED,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=current_state.current_medical_image,
            case_explicitly_selected=current_state.case_explicitly_selected,
            available_sequences=current_state.available_sequences,
            missing_sequences=current_state.missing_sequences,
            sequences_validated=current_state.sequences_validated,
            worker_thread_active=False,
            progress_percentage=100.0,
            current_operation="Completed",
            current_result=result
        )

        return self._state_machine.transition_to(AnalysisStatus.COMPLETED, new_state)

    def fail_analysis(self, error: str) -> AnalysisState:
        if not error:
            error = "Analysis failed with unknown error"

        current_state = self._state_machine.current_state

        new_state = AnalysisState(
            status=AnalysisStatus.FAILED,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=current_state.current_medical_image,
            case_explicitly_selected=current_state.case_explicitly_selected,
            available_sequences=current_state.available_sequences,
            missing_sequences=current_state.missing_sequences,
            sequences_validated=current_state.sequences_validated,
            worker_thread_active=False,
            error_message=error
        )

        return self._state_machine.transition_to(AnalysisStatus.FAILED, new_state)

    def cancel_analysis(self) -> AnalysisState:
        current_state = self._state_machine.current_state

        if not current_state.is_running():
            raise TransitionError("Cannot cancel: analysis not running")

        new_state = AnalysisState(
            status=AnalysisStatus.CANCELLED,
            models_available=current_state.models_available,
            selected_analysis_type=current_state.selected_analysis_type,
            current_medical_image=current_state.current_medical_image,
            case_explicitly_selected=current_state.case_explicitly_selected,
            available_sequences=current_state.available_sequences,
            missing_sequences=current_state.missing_sequences,
            sequences_validated=current_state.sequences_validated,
            worker_thread_active=False
        )

        return self._state_machine.transition_to(AnalysisStatus.CANCELLED, new_state)

    def reset(self) -> AnalysisState:
        self._state_machine.reset()
        return self._state_machine.current_state

    def can_run_analysis(self) -> bool:
        return self._state_machine.current_state.can_run_analysis()

    def get_validation_errors(self) -> List[str]:
        return self._state_machine.current_state.get_validation_errors()

    def get_status_message(self) -> str:
        state = self._state_machine.current_state

        base_message = state.get_progress_message()

        if not state.is_running() and not state.is_terminal():
            errors = state.get_validation_errors()
            if errors:
                return f"{base_message} - Issues: {'; '.join(errors[:2])}"

        return base_message

    def get_detailed_status(self) -> Dict[str, any]:
        state = self._state_machine.current_state

        return {
            "status": state.status.value,
            "status_message": self.get_status_message(),
            "models_available": state.models_available,
            "analysis_type": state.selected_analysis_type.value if state.selected_analysis_type else None,
            "case_selected": state.current_medical_image is not None,
            "case_explicitly_selected": state.case_explicitly_selected,
            "available_sequences": list(state.available_sequences.keys()),
            "missing_sequences": state.missing_sequences,
            "sequences_validated": state.sequences_validated,
            "can_run": state.can_run_analysis(),
            "is_running": state.is_running(),
            "is_terminal": state.is_terminal(),
            "progress": state.progress_percentage,
            "current_operation": state.current_operation,
            "has_result": state.has_result(),
            "worker_active": state.worker_thread_active,
            "error": state.error_message,
            "validation_errors": state.get_validation_errors(),
            "allowed_transitions": [s.value for s in self._state_machine.get_allowed_transitions()],
            "state_summary": state.get_status_summary()
        }

    def add_state_observer(self, observer: callable) -> None:
        if observer not in self._state_observers:
            self._state_observers.append(observer)

    def remove_state_observer(self, observer: callable) -> None:
        if observer in self._state_observers:
            self._state_observers.remove(observer)

    def _on_state_changed(self, old_state: AnalysisState, new_state: AnalysisState) -> None:
        for observer in self._state_observers:
            try:
                observer(old_state, new_state)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in state manager observer: {e}", exc_info=True)

    def __str__(self) -> str:
        state = self._state_machine.current_state
        return (
            f"AnalysisStateManager(status={state.status.value}, "
            f"can_run={state.can_run_analysis()}, "
            f"validation_errors={len(state.get_validation_errors())})"
        )
