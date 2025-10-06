from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from enum import Enum
from datetime import datetime

from .ai_analysis import AIAnalysisType, AIAnalysisResult
from .medical_image import MedicalImage


class AnalysisStatus(Enum):
    MODELS_NOT_LOADED = "models_not_loaded"
    MODELS_LOADED = "models_loaded"
    ANALYSIS_TYPE_SELECTED = "analysis_type_selected"
    CASE_SELECTED = "case_selected"
    SEQUENCES_READY = "sequences_ready"
    READY_TO_RUN = "ready_to_run"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        return self in (
            AnalysisStatus.COMPLETED,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED
        )

    def can_run_analysis(self) -> bool:
        return self in (
            AnalysisStatus.READY_TO_RUN,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED
        )

    def is_active(self) -> bool:
        return self == AnalysisStatus.RUNNING


@dataclass(frozen=True)
class AnalysisState:
    status: AnalysisStatus
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    models_available: bool = False
    models_loading_error: Optional[str] = None

    selected_analysis_type: Optional[AIAnalysisType] = None

    current_medical_image: Optional[MedicalImage] = None
    case_explicitly_selected: bool = False

    available_sequences: Dict[str, Path] = field(default_factory=dict)
    missing_sequences: List[str] = field(default_factory=list)
    sequences_validated: bool = False

    worker_thread_active: bool = False
    progress_percentage: float = 0.0
    current_operation: Optional[str] = None

    current_result: Optional[AIAnalysisResult] = None

    error_message: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

    def can_run_analysis(self) -> bool:
        if not self.status.can_run_analysis():
            return False

        if not self.models_available:
            return False

        if self.selected_analysis_type is None:
            return False

        if self.current_medical_image is None:
            return False

        if not self.case_explicitly_selected:
            return False

        if not self.sequences_validated:
            return False

        if self.missing_sequences:
            return False

        if self.worker_thread_active:
            return False

        return True

    def get_validation_errors(self) -> List[str]:
        errors = []

        if not self.status.can_run_analysis():
            if self.status.is_active():
                errors.append("Analysis is already running")
            elif self.status == AnalysisStatus.MODELS_NOT_LOADED:
                errors.append("AI models are not loaded")
            elif self.status == AnalysisStatus.COMPLETED:
                errors.append("Analysis has already been completed")
            else:
                errors.append(f"Cannot start analysis from status: {self.status.value}")

        if not self.models_available:
            if self.models_loading_error:
                errors.append(f"Models not available: {self.models_loading_error}")
            else:
                errors.append("AI models are not loaded")

        if self.selected_analysis_type is None:
            errors.append("No analysis type selected")

        if self.current_medical_image is None:
            errors.append("No medical image case selected")
        elif not self.case_explicitly_selected:
            errors.append("Case must be explicitly selected by user")

        if not self.sequences_validated:
            errors.append("Image sequences have not been validated")

        if self.missing_sequences:
            missing_str = ", ".join(self.missing_sequences)
            errors.append(f"Missing required sequences: {missing_str}")

        if not self.available_sequences:
            errors.append("No image sequences available")

        if self.worker_thread_active:
            errors.append("Another analysis operation is in progress")

        errors.extend(self.validation_errors)

        return errors

    def is_ready_to_run(self) -> bool:
        return self.status == AnalysisStatus.READY_TO_RUN and self.can_run_analysis()

    def is_terminal(self) -> bool:
        return self.status.is_terminal()

    def is_running(self) -> bool:
        return self.status.is_active()

    def has_result(self) -> bool:
        return self.current_result is not None

    def get_status_summary(self) -> Dict[str, any]:
        return {
            "status": self.status.value,
            "models_available": self.models_available,
            "analysis_type": self.selected_analysis_type.value if self.selected_analysis_type else None,
            "case_selected": self.current_medical_image is not None,
            "case_explicitly_selected": self.case_explicitly_selected,
            "sequences_count": len(self.available_sequences),
            "missing_sequences_count": len(self.missing_sequences),
            "can_run": self.can_run_analysis(),
            "is_running": self.is_running(),
            "has_result": self.has_result(),
            "progress": self.progress_percentage,
            "worker_active": self.worker_thread_active,
            "error": self.error_message,
            "validation_errors_count": len(self.get_validation_errors())
        }

    def get_progress_message(self) -> str:
        if self.status == AnalysisStatus.RUNNING:
            if self.current_operation:
                return f"{self.current_operation} ({self.progress_percentage:.0f}%)"
            else:
                return f"Running analysis... ({self.progress_percentage:.0f}%)"

        elif self.status == AnalysisStatus.COMPLETED:
            return "Analysis completed successfully"

        elif self.status == AnalysisStatus.FAILED:
            return f"Analysis failed: {self.error_message or 'Unknown error'}"

        elif self.status == AnalysisStatus.CANCELLED:
            return "Analysis was cancelled"

        elif self.status == AnalysisStatus.READY_TO_RUN:
            return "Ready to run analysis"

        elif self.status == AnalysisStatus.SEQUENCES_READY:
            return "Sequences validated and ready"

        elif self.status == AnalysisStatus.CASE_SELECTED:
            return "Case selected, validating sequences..."

        elif self.status == AnalysisStatus.ANALYSIS_TYPE_SELECTED:
            return "Analysis type selected, please select a case"

        elif self.status == AnalysisStatus.MODELS_LOADED:
            return "Models loaded, please select analysis type"

        elif self.status == AnalysisStatus.MODELS_NOT_LOADED:
            return "Loading AI models..."

        else:
            return f"Status: {self.status.value}"

    def with_updated_status(self, new_status: AnalysisStatus) -> 'AnalysisState':
        from dataclasses import replace
        return replace(self, status=new_status, last_updated=datetime.now())

    def with_error(self, error_message: str) -> 'AnalysisState':
        from dataclasses import replace
        return replace(
            self,
            status=AnalysisStatus.FAILED,
            error_message=error_message,
            last_updated=datetime.now()
        )

    def with_result(self, result: AIAnalysisResult) -> 'AnalysisState':
        from dataclasses import replace
        return replace(
            self,
            status=AnalysisStatus.COMPLETED,
            current_result=result,
            progress_percentage=100.0,
            worker_thread_active=False,
            last_updated=datetime.now()
        )

    def __str__(self) -> str:
        return (
            f"AnalysisState(status={self.status.value}, "
            f"models={self.models_available}, "
            f"type={self.selected_analysis_type.value if self.selected_analysis_type else None}, "
            f"case={'Yes' if self.current_medical_image else 'No'}, "
            f"can_run={self.can_run_analysis()})"
        )
