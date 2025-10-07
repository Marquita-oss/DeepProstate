import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

from .segmentation import MedicalSegmentation, AnatomicalRegion


class AIAnalysisType(Enum):
    PROSTATE_GLAND = "prostate_gland"
    ZONES_TZ_PZ = "zones_tz_pz"
    CSPCA_DETECTION = "cspca_detection"


class AIModelStatus(Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    RUNNING_INFERENCE = "running_inference"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OverlayVisualizationData:
    mask_array: np.ndarray                  
    color_rgba: Tuple[float, float, float, float]  
    anatomical_region: AnatomicalRegion      
    opacity: float = 0.4                  
    name: str = ""                        
    confidence_score: float = 0.0         
    voxel_count: int = 0                   
    volume_mm3: float = 0.0                 
    target_sequence: str = "T2W"            
    original_dimensions: Optional[Tuple[int, ...]] = None 
    spatial_transform_info: Optional[Dict[str, Any]] = None  
    
    def __post_init__(self):
        if self.mask_array is not None:
            self.voxel_count = int(np.sum(self.mask_array > 0))
        
        if not self.name:
            self.name = self.anatomical_region.value.replace('_', ' ').title()



@dataclass
class AISequenceRequirement:
    sequence_name: str 
    is_required: bool 
    description: str
    typical_file_extensions: List[str]
    
    @classmethod
    def get_requirements_for_analysis(cls, analysis_type: AIAnalysisType) -> List['AISequenceRequirement']:
        if analysis_type == AIAnalysisType.PROSTATE_GLAND:
            return [
                cls("T2W", True, "T2-weighted axial sequence", [".dcm", ".nii.gz", ".mha"])
            ]
        elif analysis_type == AIAnalysisType.ZONES_TZ_PZ:
            return [
                cls("T2W", True, "T2-weighted axial sequence for zonal anatomy", [".dcm", ".nii.gz", ".mha"])
            ]
        elif analysis_type == AIAnalysisType.CSPCA_DETECTION:
            return [
                cls("T2W", True, "T2-weighted sequence", [".dcm", ".nii.gz", ".mha"]),
                cls("ADC", True, "Apparent Diffusion Coefficient map", [".dcm", ".nii.gz", ".mha"]),
                cls("HBV", True, "High b-value DWI sequence", [".dcm", ".nii.gz", ".mha"])
            ]
        else:
            return []


@dataclass
class AIAnalysisResult:
    segmentations: List[MedicalSegmentation]
    overlay_data: List[OverlayVisualizationData]
    analysis_type: AIAnalysisType
    
    processing_metadata: Dict[str, Any]
    processing_time_seconds: float = 0.0
    model_confidence_overall: float = 0.0
    
    temp_files_created: List[Path] = None 
    original_image_uid: str = ""
    
    status: AIModelStatus = AIModelStatus.COMPLETED
    error_message: str = ""
    created_timestamp: datetime = None
    completed_timestamp: datetime = None
    
    def __post_init__(self):
        if self.temp_files_created is None:
            self.temp_files_created = []
            
        if self.created_timestamp is None:
            self.created_timestamp = datetime.now()
            
        if self.status == AIModelStatus.COMPLETED and self.completed_timestamp is None:
            self.completed_timestamp = datetime.now()
        
        if self.segmentations and self.model_confidence_overall == 0.0:
            confidences = [seg.confidence_score for seg in self.segmentations if hasattr(seg, 'confidence_score')]
            if confidences:
                self.model_confidence_overall = sum(confidences) / len(confidences)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type.value,
            "segmentations_count": len(self.segmentations),
            "overlays_count": len(self.overlay_data),
            "overall_confidence": f"{self.model_confidence_overall:.2f}",
            "processing_time": f"{self.processing_time_seconds:.1f}s",
            "status": self.status.value,
            "structures_detected": [seg.anatomical_region.value for seg in self.segmentations],
            "total_volume_mm3": sum([overlay.volume_mm3 for overlay in self.overlay_data])
        }
    
    def has_clinically_significant_findings(self) -> bool:
        lesion_findings = [
            seg for seg in self.segmentations 
            if seg.anatomical_region == AnatomicalRegion.SUSPICIOUS_LESION
            and getattr(seg, 'confidence_score', 0.0) > 0.7
        ]
        
        return len(lesion_findings) > 0
    
    def cleanup_temp_files(self) -> None:
        for temp_file in self.temp_files_created:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        self.temp_files_created.clear()


@dataclass
class AIAnalysisRequest:
    analysis_type: AIAnalysisType
    primary_image_path: Path
    additional_sequences: Dict[str, Path] = None
    
    apply_refinement: bool = True
    confidence_threshold: float = 0.5
    save_intermediate_files: bool = False
    
    overlay_opacity: float = 0.4 
    include_volume_calculations: bool = True 
    
    requested_by: str = "system"
    request_timestamp: datetime = None
    
    def __post_init__(self):
        if self.additional_sequences is None:
            self.additional_sequences = {}
            
        if self.request_timestamp is None:
            self.request_timestamp = datetime.now()
    
    def validate_requirements(self) -> Tuple[bool, List[str]]:
        errors = []
        
        if not self.primary_image_path.exists():
            errors.append(f"Primary image file not found: {self.primary_image_path}")
        
        for seq_name, seq_path in self.additional_sequences.items():
            if not seq_path.exists():
                errors.append(f"Additional sequence file not found: {seq_path}")
        
        available_sequences = ["T2W"]
        available_sequences.extend(self.additional_sequences.keys())
        
        return len(errors) == 0, errors
    
    def get_all_input_files(self) -> Dict[str, Path]:
        files = {"primary": self.primary_image_path}
        files.update(self.additional_sequences)
        return files