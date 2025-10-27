from typing import List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status_message: str = ""
    suggested_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.is_valid and not self.errors:
            raise ValueError("ValidationResult marked invalid must have at least one error")

        if self.is_valid and self.errors:
            raise ValueError("ValidationResult marked valid cannot have errors")

    @classmethod
    def success(
        cls,
        message: str = "Validation passed",
        warnings: List[str] = None
    ) -> "ValidationResult":
        return cls(
            is_valid=True,
            errors=[],
            warnings=warnings or [],
            status_message=message,
            suggested_actions=[]
        )

    @classmethod
    def failure(
        cls,
        errors: List[str],
        suggestions: List[str],
        status_message: str = ""
    ) -> "ValidationResult":
        if not errors:
            raise ValueError("Failure result must have at least one error")

        if not status_message:
            if len(errors) == 1:
                status_message = errors[0]
            else:
                status_message = f"{len(errors)} validation errors found"

        return cls(
            is_valid=False,
            errors=errors,
            warnings=[],
            status_message=status_message,
            suggested_actions=suggestions
        )

    @classmethod
    def warning(
        cls,
        warnings: List[str],
        status_message: str = "Validation passed with warnings"
    ) -> "ValidationResult":
        return cls(
            is_valid=True,
            errors=[],
            warnings=warnings,
            status_message=status_message,
            suggested_actions=[]
        )

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def get_all_messages(self) -> List[str]:
        return self.errors + self.warnings

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "status_message": self.status_message,
            "suggested_actions": self.suggested_actions
        }

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}): {self.status_message}"

    def __bool__(self) -> bool:
        return self.is_valid
