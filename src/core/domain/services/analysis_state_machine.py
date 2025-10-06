from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum

from ..entities.analysis_state import AnalysisState, AnalysisStatus


class TransitionError(Exception):
    pass


@dataclass
class StateTransitionRule:
    from_status: AnalysisStatus
    to_status: AnalysisStatus
    description: str
    validator: Optional[Callable[[AnalysisState], bool]] = None
    error_message: Optional[str] = None

    def can_transition(self, state: AnalysisState) -> bool:
        if state.status != self.from_status:
            return False

        if self.validator is not None:
            return self.validator(state)

        return True

    def get_error_message(self, state: AnalysisState) -> str:
        if self.error_message:
            return self.error_message

        if state.status != self.from_status:
            return f"Cannot transition from {state.status.value} to {self.to_status.value}"

        if self.validator is not None:
            return f"Validation failed for transition to {self.to_status.value}"

        return f"Invalid transition to {self.to_status.value}"


class AnalysisStateMachine:
    def __init__(self, initial_state: Optional[AnalysisState] = None):
        if initial_state is None:
            initial_state = AnalysisState(status=AnalysisStatus.MODELS_NOT_LOADED)

        self._current_state = initial_state
        self._transition_rules = self._build_transition_rules()
        self._observers: List[Callable[[AnalysisState, AnalysisState], None]] = []
        self._transition_history: List[tuple[AnalysisStatus, AnalysisStatus, str]] = []

    def _build_transition_rules(self) -> Dict[AnalysisStatus, List[StateTransitionRule]]:
        rules = {
            AnalysisStatus.MODELS_NOT_LOADED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.MODELS_NOT_LOADED,
                    to_status=AnalysisStatus.MODELS_LOADED,
                    description="Models successfully loaded",
                    validator=lambda s: s.models_available,
                    error_message="Models must be available before transitioning to MODELS_LOADED"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.MODELS_NOT_LOADED,
                    to_status=AnalysisStatus.FAILED,
                    description="Failed to load models"
                )
            ],

            AnalysisStatus.MODELS_LOADED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.MODELS_LOADED,
                    to_status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
                    description="Analysis type selected",
                    validator=lambda s: s.selected_analysis_type is not None,
                    error_message="Analysis type must be selected"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.MODELS_LOADED,
                    to_status=AnalysisStatus.FAILED,
                    description="Error during analysis type selection"
                )
            ],

            AnalysisStatus.ANALYSIS_TYPE_SELECTED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Medical case selected",
                    validator=lambda s: s.current_medical_image is not None and s.case_explicitly_selected,
                    error_message="Medical case must be explicitly selected"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
                    to_status=AnalysisStatus.MODELS_LOADED,
                    description="Change analysis type"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
                    to_status=AnalysisStatus.FAILED,
                    description="Error during case selection"
                )
            ],

            AnalysisStatus.CASE_SELECTED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.CASE_SELECTED,
                    to_status=AnalysisStatus.SEQUENCES_READY,
                    description="Sequences validated successfully",
                    validator=lambda s: s.sequences_validated and not s.missing_sequences,
                    error_message="All required sequences must be available"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.CASE_SELECTED,
                    to_status=AnalysisStatus.ANALYSIS_TYPE_SELECTED,
                    description="Change selected case"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.CASE_SELECTED,
                    to_status=AnalysisStatus.FAILED,
                    description="Sequence validation failed"
                )
            ],

            AnalysisStatus.SEQUENCES_READY: [
                StateTransitionRule(
                    from_status=AnalysisStatus.SEQUENCES_READY,
                    to_status=AnalysisStatus.READY_TO_RUN,
                    description="All prerequisites met",
                    validator=lambda s: s.can_run_analysis(),
                    error_message="Cannot run analysis: validation checks failed"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.SEQUENCES_READY,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Re-validate sequences"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.SEQUENCES_READY,
                    to_status=AnalysisStatus.FAILED,
                    description="Validation error"
                )
            ],

            AnalysisStatus.READY_TO_RUN: [
                StateTransitionRule(
                    from_status=AnalysisStatus.READY_TO_RUN,
                    to_status=AnalysisStatus.RUNNING,
                    description="Analysis started",
                    validator=lambda s: not s.worker_thread_active,
                    error_message="Cannot start: worker thread already active"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.READY_TO_RUN,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Change configuration before running"
                )
            ],

            AnalysisStatus.RUNNING: [
                StateTransitionRule(
                    from_status=AnalysisStatus.RUNNING,
                    to_status=AnalysisStatus.COMPLETED,
                    description="Analysis completed successfully",
                    validator=lambda s: s.current_result is not None,
                    error_message="Cannot complete: no result available"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.RUNNING,
                    to_status=AnalysisStatus.FAILED,
                    description="Analysis failed during execution"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.RUNNING,
                    to_status=AnalysisStatus.CANCELLED,
                    description="Analysis cancelled by user"
                )
            ],

            AnalysisStatus.COMPLETED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.COMPLETED,
                    to_status=AnalysisStatus.READY_TO_RUN,
                    description="Re-run analysis on same case",
                    validator=lambda s: s.can_run_analysis(),
                    error_message="Cannot re-run: prerequisites not met"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.COMPLETED,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Start new analysis"
                )
            ],

            AnalysisStatus.FAILED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.FAILED,
                    to_status=AnalysisStatus.READY_TO_RUN,
                    description="Retry analysis after failure",
                    validator=lambda s: s.can_run_analysis(),
                    error_message="Cannot retry: prerequisites not met"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.FAILED,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Reconfigure and retry"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.FAILED,
                    to_status=AnalysisStatus.MODELS_LOADED,
                    description="Restart from beginning"
                )
            ],

            AnalysisStatus.CANCELLED: [
                StateTransitionRule(
                    from_status=AnalysisStatus.CANCELLED,
                    to_status=AnalysisStatus.READY_TO_RUN,
                    description="Restart cancelled analysis",
                    validator=lambda s: s.can_run_analysis(),
                    error_message="Cannot restart: prerequisites not met"
                ),
                StateTransitionRule(
                    from_status=AnalysisStatus.CANCELLED,
                    to_status=AnalysisStatus.CASE_SELECTED,
                    description="Reconfigure after cancellation"
                )
            ]
        }

        return rules

    @property
    def current_state(self) -> AnalysisState:
        return self._current_state

    def transition_to(self, new_status: AnalysisStatus, updated_state: Optional[AnalysisState] = None) -> AnalysisState:
        current_status = self._current_state.status
        rules = self._transition_rules.get(current_status, [])

        matching_rule = None
        for rule in rules:
            if rule.to_status == new_status:
                matching_rule = rule
                break

        if matching_rule is None:
            raise TransitionError(
                f"No transition rule from {current_status.value} to {new_status.value}"
            )

        if updated_state is None:
            new_state = self._current_state.with_updated_status(new_status)
        else:
            if updated_state.status != new_status:
                raise ValueError(
                    f"Updated state has status {updated_state.status.value}, "
                    f"but transition target is {new_status.value}"
                )
            new_state = updated_state

        if matching_rule.validator is not None:
            if not matching_rule.validator(new_state):
                error_msg = matching_rule.get_error_message(new_state)
                raise TransitionError(error_msg)

        old_state = self._current_state
        self._current_state = new_state
        self._transition_history.append((current_status, new_status, matching_rule.description))

        self._notify_observers(old_state, new_state)

        return new_state

    def can_transition_to(self, new_status: AnalysisStatus) -> bool:
        current_status = self._current_state.status
        rules = self._transition_rules.get(current_status, [])

        for rule in rules:
            if rule.to_status == new_status:
                return rule.can_transition(self._current_state)

        return False

    def get_allowed_transitions(self) -> List[AnalysisStatus]:
        current_status = self._current_state.status
        rules = self._transition_rules.get(current_status, [])

        allowed = []
        for rule in rules:
            if rule.can_transition(self._current_state):
                allowed.append(rule.to_status)

        return allowed

    def get_transition_errors(self, target_status: AnalysisStatus) -> Optional[str]:
        current_status = self._current_state.status
        rules = self._transition_rules.get(current_status, [])

        for rule in rules:
            if rule.to_status == target_status:
                if not rule.can_transition(self._current_state):
                    return rule.get_error_message(self._current_state)
                return None

        return f"No transition rule from {current_status.value} to {target_status.value}"

    def add_observer(self, observer: Callable[[AnalysisState, AnalysisState], None]) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[AnalysisState, AnalysisState], None]) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, old_state: AnalysisState, new_state: AnalysisState) -> None:
        for observer in self._observers:
            try:
                observer(old_state, new_state)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in state machine observer: {e}", exc_info=True)

    def get_transition_history(self) -> List[tuple[AnalysisStatus, AnalysisStatus, str]]:
        return self._transition_history.copy()

    def reset(self) -> None:
        old_state = self._current_state
        self._current_state = AnalysisState(status=AnalysisStatus.MODELS_NOT_LOADED)
        self._transition_history.clear()
        self._notify_observers(old_state, self._current_state)

    def get_state_summary(self) -> Dict[str, any]:
        return {
            "current_status": self._current_state.status.value,
            "allowed_transitions": [s.value for s in self.get_allowed_transitions()],
            "transition_count": len(self._transition_history),
            "observer_count": len(self._observers),
            "can_run_analysis": self._current_state.can_run_analysis(),
            "validation_errors": self._current_state.get_validation_errors()
        }

    def __str__(self) -> str:
        allowed = ", ".join(s.value for s in self.get_allowed_transitions())
        return f"AnalysisStateMachine(status={self._current_state.status.value}, allowed=[{allowed}])"
