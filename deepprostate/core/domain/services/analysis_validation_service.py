import logging
from typing import Optional, Dict, List
from pathlib import Path

from ..entities.ai_analysis import AIAnalysisType, AISequenceRequirement
from ..entities.medical_image import MedicalImage
from ..value_objects.validation_result import ValidationResult
from .sequence_detection_service import SequenceDetectionService

logger = logging.getLogger(__name__)


class AnalysisValidationService:
    def __init__(self, sequence_detection_service: SequenceDetectionService):
        self._sequence_detector = sequence_detection_service

    def validate_analysis_readiness(
        self,
        analysis_type: AIAnalysisType,
        current_image: Optional[MedicalImage],
        available_sequences: Dict[str, MedicalImage],
        models_available: bool,
        case_explicitly_selected: bool
    ) -> ValidationResult:
        if not models_available:
            return ValidationResult.failure(
                errors=["AI models not loaded"],
                suggestions=["Load AI models before running analysis"],
                status_message="AI models not loaded - Please load models first"
            )

        if not analysis_type:
            return ValidationResult.failure(
                errors=["No analysis type selected"],
                suggestions=["Select Prostate Gland, Zones, or csPCa Detection"],
                status_message="Please select an analysis type"
            )

        if not case_explicitly_selected or not current_image:
            return ValidationResult.failure(
                errors=["No case selected"],
                suggestions=["Select a case from Input Sequences panel"],
                status_message="Select a case from the list above"
            )

        missing_sequences = self.get_missing_sequences(
            analysis_type=analysis_type,
            current_image=current_image,
            available_sequences=available_sequences
        )

        if missing_sequences:
            return self._create_missing_sequences_result(
                missing_sequences=missing_sequences,
                analysis_type=analysis_type
            )

        if analysis_type == AIAnalysisType.CSPCA_DETECTION:
            compatibility_errors = self._validate_multi_sequence_compatibility(
                primary=current_image,
                available_sequences=available_sequences
            )
            if compatibility_errors:
                return ValidationResult.failure(
                    errors=compatibility_errors,
                    suggestions=["Ensure all sequences are from the same study"],
                    status_message="Sequence compatibility issues detected"
                )

        return ValidationResult.success(
            message="Ready for analysis",
            warnings=self._generate_warnings(analysis_type, available_sequences)
        )

    def get_required_sequences(
        self,
        analysis_type: AIAnalysisType
    ) -> List[AISequenceRequirement]:
        return AISequenceRequirement.get_requirements_for_analysis(analysis_type)

    def get_missing_sequences(
        self,
        analysis_type: AIAnalysisType,
        current_image: MedicalImage,
        available_sequences: Dict[str, MedicalImage]
    ) -> List[str]:
        if analysis_type in [AIAnalysisType.PROSTATE_GLAND, AIAnalysisType.ZONES_TZ_PZ]:
            return []

        if analysis_type == AIAnalysisType.CSPCA_DETECTION:
            required_sequences = ["ADC", "HBV"]
            missing = []

            for seq_name in required_sequences:
                if seq_name not in available_sequences or available_sequences[seq_name] is None:
                    missing.append(seq_name)

            return missing

        logger.warning(f"Unknown analysis type: {analysis_type}")
        return []

    def validate_sequence_compatibility(
        self,
        primary: MedicalImage,
        secondary: MedicalImage
    ) -> bool:
        if not primary or not secondary:
            return False

        primary_study = self._get_study_uid(primary)
        secondary_study = self._get_study_uid(secondary)

        if primary_study and secondary_study:
            if primary_study != secondary_study:
                logger.warning(
                    f"StudyInstanceUID mismatch: {primary_study} != {secondary_study}"
                )
                return False

        primary_patient = self._get_patient_id(primary)
        secondary_patient = self._get_patient_id(secondary)

        if primary_patient and secondary_patient:
            if primary_patient != secondary_patient:
                logger.warning(
                    f"PatientID mismatch: {primary_patient} != {secondary_patient}"
                )
                return False

        return True

    def _create_missing_sequences_result(
        self,
        missing_sequences: List[str],
        analysis_type: AIAnalysisType
    ) -> ValidationResult:
        missing_str = ", ".join(missing_sequences)

        suggestions = [
            f"Load {seq} sequence from the same study" for seq in missing_sequences
        ]
        suggestions.append("Ensure all sequences are from the same imaging study")

        return ValidationResult.failure(
            errors=[f"Missing required sequences: {missing_str}"],
            suggestions=suggestions,
            status_message=f"Missing sequences: {missing_str}"
        )

    def _validate_multi_sequence_compatibility(
        self,
        primary: MedicalImage,
        available_sequences: Dict[str, MedicalImage]
    ) -> List[str]:
        errors = []

        for seq_name, seq_image in available_sequences.items():
            if seq_image is None:
                continue

            if not self.validate_sequence_compatibility(primary, seq_image):
                errors.append(
                    f"{seq_name} sequence is not from the same study as primary image"
                )

        return errors

    def _generate_warnings(
        self,
        analysis_type: AIAnalysisType,
        available_sequences: Dict[str, MedicalImage]
    ) -> List[str]:
        warnings = []
        return warnings

    def _get_study_uid(self, medical_image: MedicalImage) -> Optional[str]:
        if hasattr(medical_image, 'study_instance_uid'):
            return medical_image.study_instance_uid

        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            return medical_image.dicom_metadata.get('StudyInstanceUID')

        return None

    def _get_patient_id(self, medical_image: MedicalImage) -> Optional[str]:
        if hasattr(medical_image, 'patient_id'):
            return medical_image.patient_id

        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            return medical_image.dicom_metadata.get('PatientID')

        return None
