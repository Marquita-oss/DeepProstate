import re
import logging
from typing import Optional, List, Dict, Set
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceDetectionResult:
    sequence_type: Optional[str]
    confidence: float
    matched_patterns: List[str]
    metadata_source: str


class SequenceDetectionService:
    T2W_PATTERNS = [
        re.compile(r'[_\s]T2W?\b', re.IGNORECASE),  # _T2W or _T2 with word boundary after
        re.compile(r'\bT2W?\b', re.IGNORECASE),  # T2W or T2 with word boundaries
        re.compile(r'T2[_\s]?WEIGHTED', re.IGNORECASE),
        re.compile(r'T2[_\s]?(TSE|HASTE|SPACE|FRFSE)', re.IGNORECASE),
        re.compile(r'\bAXIAL[_\s]T2\b', re.IGNORECASE),
        re.compile(r'\bSAGITTAL[_\s]T2\b', re.IGNORECASE),
    ]

    ADC_PATTERNS = [
        re.compile(r'[_\s]ADC\b', re.IGNORECASE),  # _ADC with word boundary after
        re.compile(r'\bADC\b', re.IGNORECASE),
        re.compile(r'APPARENT[_\s]DIFFUSION', re.IGNORECASE),
        re.compile(r'DWI[_\s]?ADC', re.IGNORECASE),
        re.compile(r'ADC[_\s]MAP', re.IGNORECASE),
    ]

    HBV_PATTERNS = [
        re.compile(r'[_\s]HBV\b', re.IGNORECASE),  # _HBV with word boundary after
        re.compile(r'\bHBV\b', re.IGNORECASE),
        re.compile(r'HIGH[_\s]B[_\s]?VALUE', re.IGNORECASE),
        re.compile(r'\bB[_\s]?(\d{3,4})\b', re.IGNORECASE),  # B800, B1000, etc.
        re.compile(r'DWI[_\s]?HIGH', re.IGNORECASE),
    ]

    EXCLUSION_PATTERNS = [
        re.compile(r'\bLOCALIZER\b', re.IGNORECASE),
        re.compile(r'\bSCOUT\b', re.IGNORECASE),
        re.compile(r'\bCALIBRATION\b', re.IGNORECASE),
        re.compile(r'\bPHASE\b', re.IGNORECASE),
        re.compile(r'\bSURVEY\b', re.IGNORECASE),
        re.compile(r'\bCOLOR\b', re.IGNORECASE),
        re.compile(r'\bFUSION\b', re.IGNORECASE),
    ]

    HBV_B_VALUE_THRESHOLD = 800

    @classmethod
    def detect_sequence_type(cls, medical_image) -> SequenceDetectionResult:
        metadata_text = cls._extract_metadata_text(medical_image)

        if cls._is_excluded_sequence(metadata_text):
            return SequenceDetectionResult(
                sequence_type=None,
                confidence=1.0,
                matched_patterns=["EXCLUSION"],
                metadata_source="exclusion_check"
            )

        adc_result = cls._detect_adc(metadata_text)
        if adc_result.sequence_type:
            return adc_result

        hbv_result = cls._detect_hbv(metadata_text, medical_image)
        if hbv_result.sequence_type:
            return hbv_result

        t2w_result = cls._detect_t2w(metadata_text)
        if t2w_result.sequence_type:
            return t2w_result

        return SequenceDetectionResult(
            sequence_type=None,
            confidence=0.0,
            matched_patterns=[],
            metadata_source="no_match"
        )

    @classmethod
    def detect_related_sequences(
        cls,
        primary_image,
        all_loaded_images: Dict
    ) -> Dict[str, Optional[object]]:
        detected_sequences = {
            "ADC": None,
            "HBV": None
        }

        if not primary_image:
            return detected_sequences

        primary_study_uid = cls._get_study_uid(primary_image)
        primary_patient_id = cls._get_patient_id(primary_image)

        if not primary_study_uid:
            logger.warning("Primary image has no StudyInstanceUID - cannot detect related sequences")
            return detected_sequences

        for series_uid, medical_image in all_loaded_images.items():
            if series_uid == cls._get_series_uid(primary_image):
                continue

            if cls._get_study_uid(medical_image) != primary_study_uid:
                continue

            if cls._get_patient_id(medical_image) != primary_patient_id:
                continue

            result = cls.detect_sequence_type(medical_image)

            if result.sequence_type == "ADC" and not detected_sequences["ADC"]:
                detected_sequences["ADC"] = medical_image
                logger.info(f"Detected related ADC sequence: {series_uid}")

            elif result.sequence_type == "HBV" and not detected_sequences["HBV"]:
                detected_sequences["HBV"] = medical_image
                logger.info(f"Detected related HBV sequence: {series_uid}")

        return detected_sequences

    @classmethod
    def filter_t2w_sequences(cls, loaded_images: Dict) -> Dict:
        t2w_images = {}
        logger.debug(f"Filtering T2W sequences from {len(loaded_images)} total images")

        for series_uid, medical_image in loaded_images.items():
            result = cls.detect_sequence_type(medical_image)

            if result.sequence_type == "T2W":
                t2w_images[series_uid] = medical_image
                logger.debug(f"Added T2W sequence: {series_uid[:8]} - {getattr(medical_image, 'series_description', 'N/A')}")

        logger.info(f"Filtered {len(t2w_images)} T2W sequences from {len(loaded_images)} total cases")
        return t2w_images

    @classmethod
    def _extract_metadata_text(cls, medical_image) -> str:
        texts = []

        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            texts.append(medical_image.dicom_metadata.get('SeriesDescription', ''))
            texts.append(medical_image.dicom_metadata.get('ProtocolName', ''))
            texts.append(medical_image.dicom_metadata.get('SequenceName', ''))
            texts.append(medical_image.dicom_metadata.get('SeriesNumber', ''))

        if hasattr(medical_image, 'series_description'):
            texts.append(medical_image.series_description or '')

        if hasattr(medical_image, 'protocol_name'):
            texts.append(medical_image.protocol_name or '')

        if hasattr(medical_image, 'series_instance_uid'):
            texts.append(medical_image.series_instance_uid or '')

        if hasattr(medical_image, 'file_path'):
            texts.append(str(medical_image.file_path))

        result_text = ' '.join(str(t) for t in texts if t)
        return result_text

    @classmethod
    def _is_excluded_sequence(cls, text: str) -> bool:
        return any(pattern.search(text) for pattern in cls.EXCLUSION_PATTERNS)

    @classmethod
    def _detect_t2w(cls, text: str) -> SequenceDetectionResult:
        matched_patterns = []

        for pattern in cls.T2W_PATTERNS:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            confidence = min(0.7 + 0.1 * len(matched_patterns), 1.0)
            return SequenceDetectionResult(
                sequence_type="T2W",
                confidence=confidence,
                matched_patterns=matched_patterns,
                metadata_source="dicom_metadata"
            )

        return SequenceDetectionResult(None, 0.0, [], "")

    @classmethod
    def _detect_adc(cls, text: str) -> SequenceDetectionResult:
        matched_patterns = []

        for pattern in cls.ADC_PATTERNS:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            # ADC is very specific, high confidence
            confidence = 0.95
            return SequenceDetectionResult(
                sequence_type="ADC",
                confidence=confidence,
                matched_patterns=matched_patterns,
                metadata_source="dicom_metadata"
            )

        return SequenceDetectionResult(None, 0.0, [], "")

    @classmethod
    def _detect_hbv(cls, text: str, medical_image) -> SequenceDetectionResult:
        matched_patterns = []

        for pattern in cls.HBV_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_patterns.append(pattern.pattern)

        if not matched_patterns:
            return SequenceDetectionResult(None, 0.0, [], "")

        b_value = cls._extract_b_value(text)

        if b_value and b_value >= cls.HBV_B_VALUE_THRESHOLD:
            confidence = 0.9
        elif b_value and b_value < cls.HBV_B_VALUE_THRESHOLD:
            return SequenceDetectionResult(None, 0.0, [], "b_value_too_low")
        else:
            confidence = 0.7

        return SequenceDetectionResult(
            sequence_type="HBV",
            confidence=confidence,
            matched_patterns=matched_patterns,
            metadata_source="dicom_metadata_with_b_value"
        )

    @classmethod
    def _extract_b_value(cls, text: str) -> Optional[int]:
        match = re.search(r'\bB[_\s]?(\d{3,4})\b', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def _get_study_uid(cls, medical_image) -> Optional[str]:
        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            return medical_image.dicom_metadata.get('StudyInstanceUID')
        if hasattr(medical_image, 'study_instance_uid'):
            return medical_image.study_instance_uid
        return None

    @classmethod
    def _get_patient_id(cls, medical_image) -> Optional[str]:
        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            return medical_image.dicom_metadata.get('PatientID')
        if hasattr(medical_image, 'patient_id'):
            return medical_image.patient_id
        return None

    @classmethod
    def _get_series_uid(cls, medical_image) -> Optional[str]:
        if hasattr(medical_image, 'series_instance_uid'):
            return medical_image.series_instance_uid
        if hasattr(medical_image, 'dicom_metadata') and medical_image.dicom_metadata:
            return medical_image.dicom_metadata.get('SeriesInstanceUID')
        return None