import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

from src.core.domain.entities.medical_image import MedicalImage
from src.frameworks.infrastructure.constants.medical_sequences import (
    get_sequence_description, SEQUENCE_INDICATORS, is_prostate_study_complete,
    get_missing_prostate_sequences, PROSTATE_REQUIRED_SEQUENCES
)


@dataclass
class SequenceAnalysisResult:
    detected_sequences: Dict[str, List[str]]  
    is_complete_study: bool
    completeness_percentage: float
    recommended_analysis: List[str]  
    sequence_quality_scores: Dict[str, float]  


@dataclass
class SeriesEnhancement:
    original_description: str
    enhanced_description: str
    detected_sequence_type: Optional[str]
    quality_indicators: List[str]
    suitability_score: float 
    recommendations: List[str]


class SequenceAnalysisService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        self._sequence_patterns = {
            'T2W': {
                'primary': ['T2', 'T2W', 'T2_TSE', 'TURBO SPIN ECHO', 'T2 WEIGHTED'],
                'secondary': ['TSE', 'FIESTA', 'SSFSE', 'HASTE'],
                'exclude': ['T2*', 'T2STAR', 'T2_FLAIR']
            },
            'T1W': {
                'primary': ['T1', 'T1W', 'T1_WEIGHTED', 'T1 WEIGHTED'],
                'secondary': ['FLASH', 'SPOILED', 'T1_FFE'],
                'exclude': ['T1_POST', 'T1_GAD', 'T1_CONTRAST']
            },
            'DWI': {
                'primary': ['DWI', 'DIFFUSION WEIGHTED', 'EP2D_DIFF', 'DIFFUSION'],
                'secondary': ['DW_', 'DIFFUSION_', 'EPI_DWI'],
                'exclude': ['ADC', 'CALCULATED', 'DERIVED']
            },
            'ADC': {
                'primary': ['ADC', 'APPARENT_DIFFUSION', 'DIFFUSION_COEFFICIENT', 'ADC_MAP'],
                'secondary': ['APPARENT DIFFUSION', 'ADC MAP', 'DWI_ADC'],
                'exclude': ['TRACE', 'EXPONENTIAL']
            },
            'HBV': {
                'primary': ['HBV', 'HIGH_B_VALUE', 'B1000', 'B800', 'B2000', 'B1400'],
                'secondary': ['DWI_B', 'DIFFUSION_B', 'HIGH B VALUE'],
                'exclude': ['B0', 'B50', 'B100']
            },
            'DCE': {
                'primary': ['DCE', 'DYNAMIC', 'CONTRAST_ENHANCED', 'PERFUSION'],
                'secondary': ['POST_CONTRAST', 'ENHANCEMENT', 'KTRANS'],
                'exclude': ['DELAYED', 'EQUILIBRIUM']
            }
        }
        
        self._quality_indicators = {
            'T2W': {
                'good': ['AXIAL', 'HIGH_RESOLUTION', 'THIN_SLICE', '≤3MM'],
                'acceptable': ['SAGITTAL', 'CORONAL', '≤5MM'],
                'poor': ['THICK_SLICE', '>5MM', 'LOW_RESOLUTION']
            },
            'DWI': {
                'good': ['MULTIPLE_B_VALUES', 'B0_B1000', 'ADC_INCLUDED'],
                'acceptable': ['SINGLE_B_VALUE', 'B1000'],
                'poor': ['LOW_B_VALUE', 'B<800', 'NO_ADC']
            }
        }
        
        self._analysis_recommendations = {
            frozenset(['T2W']): ['Prostate Gland Segmentation', 'Zonal Anatomy'],
            frozenset(['T2W', 'DWI']): ['Prostate Gland Segmentation', 'Zonal Anatomy', 'Lesion Detection'],
            frozenset(['T2W', 'ADC']): ['Prostate Gland Segmentation', 'Zonal Anatomy', 'Advanced Lesion Detection'],
            frozenset(['T2W', 'ADC', 'HBV']): ['Complete csPCa Detection', 'PI-RADS Assessment'],
            frozenset(['T2W', 'DCE']): ['Prostate Gland Segmentation', 'Enhancement Analysis'],
        }
    
    def analyze_study_sequences(self, study_info: Dict[str, Any]) -> SequenceAnalysisResult:
        series_data = study_info.get('series', {})
        detected_sequences = {}
        quality_scores = {}
        
        for series_uid, series_info in series_data.items():
            sequence_type = self.detect_sequence_type(series_info)
            
            if sequence_type:
                if sequence_type not in detected_sequences:
                    detected_sequences[sequence_type] = []
                detected_sequences[sequence_type].append(series_uid)                
                quality_scores[series_uid] = self._calculate_series_quality_score(
                    series_info, sequence_type
                )
        
        available_sequence_types = set(detected_sequences.keys())
        missing_sequences = []
        
        for required_seq in PROSTATE_REQUIRED_SEQUENCES:
            if required_seq not in available_sequence_types:
                missing_sequences.append(required_seq)
        
        total_required = len(PROSTATE_REQUIRED_SEQUENCES)
        available_required = len([seq for seq in available_sequence_types if seq in PROSTATE_REQUIRED_SEQUENCES])
        completeness_percentage = (available_required / total_required * 100) if total_required > 0 else 0
        is_complete_study = is_prostate_study_complete(available_sequence_types)       
        recommended_analysis = self._get_analysis_recommendations(available_sequence_types)
        
        result = SequenceAnalysisResult(
            detected_sequences=detected_sequences,
            missing_sequences=missing_sequences,
            is_complete_study=is_complete_study,
            completeness_percentage=completeness_percentage,
            recommended_analysis=recommended_analysis,
            sequence_quality_scores=quality_scores
        )
        
        self._logger.info(f"Study analysis completed: {len(detected_sequences)} sequence types, "
                         f"{completeness_percentage:.1f}% complete")
        
        return result
    
    def detect_sequence_type(self, series_info: Dict[str, Any]) -> Optional[str]:
        series_description = series_info.get('series_description', '').upper()
        protocol_name = series_info.get('protocol_name', '').upper()
        search_text = f"{series_description} {protocol_name}".strip()
        
        if not search_text:
            return None
        
        best_match = None
        highest_score = 0.0
        
        for sequence_type, patterns in self._sequence_patterns.items():
            score = self._calculate_sequence_match_score(search_text, patterns)
            
            if score > highest_score and score > 0.6:
                highest_score = score
                best_match = sequence_type
        
        if best_match:
            self._logger.debug(f"Detected {best_match} with score {highest_score:.2f}: {search_text[:50]}")
        
        return best_match
    
    def _calculate_sequence_match_score(self, search_text: str, patterns: Dict[str, List[str]]) -> float:
        score = 0.0
        
        for pattern in patterns.get('primary', []):
            if pattern in search_text:
                score += 1.0
        
        for pattern in patterns.get('secondary', []):
            if pattern in search_text:
                score += 0.5
        
        for pattern in patterns.get('exclude', []):
            if pattern in search_text:
                score -= 0.8
        
        max_possible = len(patterns.get('primary', [])) + len(patterns.get('secondary', [])) * 0.5
        if max_possible > 0:
            score = max(0.0, score / max_possible)
        
        return score
    
    def enhance_series_display(self, series_info: Dict[str, Any], 
                              study_info: Optional[Dict[str, Any]] = None) -> SeriesEnhancement:
        original_desc = series_info.get('series_description', 'Unknown Series')
        detected_type = self.detect_sequence_type(series_info)
        enhanced_desc = self._create_enhanced_description(series_info, detected_type)
        quality_indicators = self._get_quality_indicators(series_info, detected_type)
        suitability_score = self._calculate_series_suitability(series_info, detected_type)        
        recommendations = self._get_series_recommendations(series_info, detected_type)
        
        enhancement = SeriesEnhancement(
            original_description=original_desc,
            enhanced_description=enhanced_desc,
            detected_sequence_type=detected_type,
            quality_indicators=quality_indicators,
            suitability_score=suitability_score,
            recommendations=recommendations
        )
        
        return enhancement
    
    def _create_enhanced_description(self, series_info: Dict[str, Any], 
                                   detected_type: Optional[str]) -> str:
        original = series_info.get('series_description', 'Unknown Series')
        
        if not detected_type:
            return original
        
        readable_name = {
            'T2W': 'T2-Weighted',
            'T1W': 'T1-Weighted', 
            'DWI': 'Diffusion Weighted',
            'ADC': 'ADC Map',
            'HBV': 'High B-value DWI',
            'DCE': 'Dynamic Contrast Enhanced'
        }.get(detected_type, detected_type)
        
        orientation = self._detect_orientation(series_info)
        if orientation:
            readable_name += f' ({orientation})'
        
        tech_details = self._extract_technical_details(series_info)
        if tech_details:
            readable_name += f' - {tech_details}'
        
        return readable_name
    
    def _detect_orientation(self, series_info: Dict[str, Any]) -> Optional[str]:
        description = series_info.get('series_description', '').upper()
        
        if 'AXIAL' in description or 'AX' in description:
            return 'Axial'
        elif 'SAGITTAL' in description or 'SAG' in description:
            return 'Sagittal'
        elif 'CORONAL' in description or 'COR' in description:
            return 'Coronal'
        
        return None
    
    def _extract_technical_details(self, series_info: Dict[str, Any]) -> str:
        details = []
        
        thickness = series_info.get('slice_thickness', '')
        if thickness:
            try:
                thick_float = float(thickness)
                details.append(f"{thick_float:.1f}mm")
            except (ValueError, TypeError):
                pass
        
        image_count = series_info.get('images_count', 0)
        if image_count:
            details.append(f"{image_count} images")
        
        return ', '.join(details)
    
    def _calculate_series_quality_score(self, series_info: Dict[str, Any], 
                                       sequence_type: str) -> float:
        if sequence_type not in self._quality_indicators:
            return 0.5 
        
        indicators = self._quality_indicators[sequence_type]
        description = series_info.get('series_description', '').upper()
        
        score = 0.5 
        
        for indicator in indicators.get('good', []):
            if indicator in description:
                score += 0.2
        
        for indicator in indicators.get('acceptable', []):
            if indicator in description:
                score += 0.1
        
        for indicator in indicators.get('poor', []):
            if indicator in description:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _get_quality_indicators(self, series_info: Dict[str, Any], 
                              detected_type: Optional[str]) -> List[str]:
        indicators = []
        
        if not detected_type:
            return indicators
        
        thickness = series_info.get('slice_thickness', '')
        if thickness:
            try:
                thick_float = float(thickness)
                if thick_float <= 3.0:
                    indicators.append('High Resolution')
                elif thick_float <= 5.0:
                    indicators.append('Standard Resolution')
                else:
                    indicators.append('Low Resolution')
            except (ValueError, TypeError):
                pass
        
        image_count = series_info.get('images_count', 0)
        if image_count:
            if image_count >= 20:
                indicators.append('Good Coverage')
            elif image_count >= 10:
                indicators.append('Adequate Coverage')
            else:
                indicators.append('Limited Coverage')
        
        return indicators
    
    def _calculate_series_suitability(self, series_info: Dict[str, Any], 
                                    detected_type: Optional[str]) -> float:
        if not detected_type:
            return 0.3
        
        base_score = 0.7
        
        image_count = series_info.get('images_count', 0)
        if image_count >= 20:
            base_score += 0.2
        elif image_count < 10:
            base_score -= 0.2
        
        thickness = series_info.get('slice_thickness', '')
        if thickness:
            try:
                thick_float = float(thickness)
                if thick_float <= 3.0:
                    base_score += 0.1
                elif thick_float > 5.0:
                    base_score -= 0.1
            except (ValueError, TypeError):
                pass
        
        return max(0.0, min(1.0, base_score))
    
    def _get_series_recommendations(self, series_info: Dict[str, Any], 
                                  detected_type: Optional[str]) -> List[str]:
        recommendations = []
        
        if not detected_type:
            recommendations.append("Sequence type unclear - check acquisition parameters")
            return recommendations
        
        if detected_type == 'T2W':
            recommendations.append("Suitable for prostate gland segmentation")
            recommendations.append("Good for zonal anatomy assessment")
        elif detected_type == 'ADC':
            recommendations.append("Useful for lesion detection")
            recommendations.append("Quantitative analysis capable")
        elif detected_type == 'HBV':
            recommendations.append("Essential for csPCa detection")
            recommendations.append("Combine with T2W and ADC for optimal results")
        
        quality_score = self._calculate_series_quality_score(series_info, detected_type)
        if quality_score < 0.5:
            recommendations.append("Consider re-acquisition for better quality")
        elif quality_score > 0.8:
            recommendations.append("Excellent quality for AI analysis")
        
        return recommendations
    
    def _get_analysis_recommendations(self, available_sequences: Set[str]) -> List[str]:
        recommendations = []
        
        for seq_combo, analyses in self._analysis_recommendations.items():
            if seq_combo.issubset(available_sequences):
                recommendations.extend(analyses)
        
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def validate_prostate_study_completeness(self, study_info: Dict[str, Any]) -> Dict[str, Any]:
        analysis_result = self.analyze_study_sequences(study_info)
        
        return {
            'is_complete': analysis_result.is_complete_study,
            'completeness_percentage': analysis_result.completeness_percentage,
            'detected_sequences': list(analysis_result.detected_sequences.keys()),
            'missing_sequences': analysis_result.missing_sequences,
            'recommended_analysis': analysis_result.recommended_analysis,
            'total_series': len(study_info.get('series', {})),
            'analyzable_series': len(analysis_result.sequence_quality_scores)
        }
    
    def get_sequence_type_statistics(self, patient_data: Dict[str, Any]) -> Dict[str, int]:
        sequence_counts = {}
        
        for patient_info in patient_data.get('patients', {}).values():
            for study_info in patient_info.get('studies', {}).values():
                for series_info in study_info.get('series', {}).values():
                    sequence_type = self.detect_sequence_type(series_info)
                    if sequence_type:
                        sequence_counts[sequence_type] = sequence_counts.get(sequence_type, 0) + 1
        
        return sequence_counts