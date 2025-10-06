import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FilenameStudyGroup:
    study_id: str
    patient_id: Optional[str]
    files: List[str]
    sequences: Dict[str, str]
    confidence: float


class FilenameStudyGrouper:    
    def __init__(self):
        self.logger = logging.getLogger(__name__)        
        self.study_patterns = [
            {
                'name': 'double_id_pattern',
                'regex': r'^(\d+)_(\d+)_([a-zA-Z0-9]+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [1, 2], 
                'sequence_group': 3,
                'confidence': 0.9
            },
            
            {
                'name': 'patient_study_pattern',
                'regex': r'^patient(\d+)_study(\d+)_([a-zA-Z0-9]+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [1, 2],
                'sequence_group': 3,
                'confidence': 0.9
            },
            
            {
                'name': 'case_id_pattern', 
                'regex': r'^case_(\d+)_(\d+)_([a-zA-Z0-9]+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [1, 2],
                'sequence_group': 3,
                'confidence': 0.85
            },
            
            {
                'name': 'single_id_pattern',
                'regex': r'^(\d+)_([a-zA-Z0-9]+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [1],
                'sequence_group': 2,
                'confidence': 0.8
            },
            
            {
                'name': 'study_prefix_pattern',
                'regex': r'^study(\d+)_([a-zA-Z0-9]+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [1],
                'sequence_group': 2, 
                'confidence': 0.8
            },
            
            {
                'name': 'suffix_id_pattern',
                'regex': r'^([a-zA-Z0-9]+)_(\d+)_(\d+)\.(mha|nii|nii\.gz|dcm)$',
                'study_id_groups': [2, 3],
                'sequence_group': 1,
                'confidence': 0.75
            }
        ]
        
        self.sequence_mappings = {
            't2w': 'T2W', 't2': 'T2W', 't2_w': 'T2W',
            'adc': 'ADC', 'apparent': 'ADC', 'app_diff': 'ADC',
            'dwi': 'DWI', 'diffusion': 'DWI', 'diff': 'DWI', 
            'hbv': 'HBV', 'high_b': 'HBV', 'b1000': 'HBV', 'b1400': 'HBV', 'b2000': 'HBV'
        }
        
        self._compiled_patterns = []
        for pattern in self.study_patterns:
            try:
                compiled = re.compile(pattern['regex'], re.IGNORECASE)
                self._compiled_patterns.append({
                    **pattern,
                    'compiled_regex': compiled
                })
            except re.error as e:
                self.logger.warning(f"Invalid pattern '{pattern['name']}': {e}")
    
    def group_files_by_study(self, file_paths: List[str]) -> List[FilenameStudyGroup]:
        self.logger.info(f"Grouping {len(file_paths)} files by study ID from filename")
        
        file_analyses = []
        for file_path in file_paths:
            analysis = self._analyze_filename(file_path)
            if analysis:
                file_analyses.append(analysis)
        
        if not file_analyses:
            self.logger.warning("No files matched any study ID pattern")
            return []
        
        study_groups = defaultdict(list)
        for analysis in file_analyses:
            study_id = analysis['study_id']
            study_groups[study_id].append(analysis)
        
        result_groups = []
        for study_id, analyses in study_groups.items():
            group = self._create_study_group(study_id, analyses)
            if group:
                result_groups.append(group)
        
        result_groups.sort(key=lambda g: g.confidence, reverse=True)
        
        self.logger.info(f"Found {len(result_groups)} study groups")
        for group in result_groups:
            self.logger.info(f"  - Study {group.study_id}: {len(group.sequences)} sequences ({', '.join(group.sequences.keys())})")
        
        return result_groups
    
    def _analyze_filename(self, file_path: str) -> Optional[Dict]:
        filename = Path(file_path).name
        
        for pattern in self._compiled_patterns:
            match = pattern['compiled_regex'].match(filename)
            if match:
                groups = match.groups()
                
                # Extraer study_id
                study_id_parts = []
                for group_idx in pattern['study_id_groups']:
                    if group_idx <= len(groups):
                        study_id_parts.append(groups[group_idx - 1])
                
                study_id = '_'.join(study_id_parts)
                
                # Extraer secuencia
                sequence_group_idx = pattern['sequence_group']
                if sequence_group_idx <= len(groups):
                    raw_sequence = groups[sequence_group_idx - 1].lower()
                    normalized_sequence = self.sequence_mappings.get(raw_sequence, raw_sequence.upper())
                else:
                    normalized_sequence = 'UNKNOWN'
                
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'study_id': study_id,
                    'sequence_type': normalized_sequence,
                    'raw_sequence': groups[sequence_group_idx - 1] if sequence_group_idx <= len(groups) else None,
                    'pattern_name': pattern['name'],
                    'confidence': pattern['confidence'],
                    'match_groups': groups
                }
        
        self.logger.debug(f"No pattern matched for: {filename}")
        return None
    
    def _create_study_group(self, study_id: str, analyses: List[Dict]) -> Optional[FilenameStudyGroup]:
        if not analyses:
            return None
        
        avg_confidence = sum(a['confidence'] for a in analyses) / len(analyses)
        
        files = [a['file_path'] for a in analyses]
        sequences = {a['sequence_type']: a['file_path'] for a in analyses}
        
        sequence_counts = defaultdict(int)
        for analysis in analyses:
            sequence_counts[analysis['sequence_type']] += 1
        
        duplicate_penalty = 0.0
        for seq_type, count in sequence_counts.items():
            if count > 1:
                duplicate_penalty += 0.1 * (count - 1)
                self.logger.warning(f"Study {study_id} has {count} files for sequence {seq_type}")
        
        final_confidence = max(0.1, avg_confidence - duplicate_penalty)
        
        patient_id = None
        if '_' in study_id:
            patient_id = study_id.split('_')[0]
        
        return FilenameStudyGroup(
            study_id=study_id,
            patient_id=patient_id,
            files=files,
            sequences=sequences,
            confidence=final_confidence
        )
    
    def validate_multi_sequence_study(
        self, 
        study_group: FilenameStudyGroup, 
        required_sequences: List[str]
    ) -> Dict[str, any]:
        available = set(study_group.sequences.keys())
        required = set(required_sequences)
        
        missing = required - available
        extra = available - required
        
        is_complete = len(missing) == 0
        completeness_ratio = len(required & available) / len(required) if required else 1.0
        
        return {
            'study_id': study_group.study_id,
            'is_complete': is_complete,
            'completeness_ratio': completeness_ratio,
            'available_sequences': list(available),
            'required_sequences': list(required),
            'missing_sequences': list(missing),
            'extra_sequences': list(extra),
            'confidence': study_group.confidence,
            'suitable_for_analysis': is_complete and study_group.confidence > 0.7
        }
    
    def find_multi_sequence_studies(
        self, 
        file_paths: List[str], 
        required_sequences: List[str] = None
    ) -> List[Tuple[FilenameStudyGroup, Dict]]:
        if required_sequences is None:
            required_sequences = ['T2W', 'ADC', 'HBV'] 
        
        study_groups = self.group_files_by_study(file_paths)
        
        complete_studies = []
        for group in study_groups:
            validation = self.validate_multi_sequence_study(group, required_sequences)
            if validation['suitable_for_analysis']:
                complete_studies.append((group, validation))
        
        self.logger.debug(f"Found {len(complete_studies)} complete multi-sequence studies")
        return complete_studies
    
    def get_study_summary(self, study_groups: List[FilenameStudyGroup]) -> Dict[str, any]:
        if not study_groups:
            return {'total_studies': 0}
        
        total_files = sum(len(group.files) for group in study_groups)
        sequence_distribution = defaultdict(int)
        
        for group in study_groups:
            for seq_type in group.sequences.keys():
                sequence_distribution[seq_type] += 1
        
        avg_confidence = sum(group.confidence for group in study_groups) / len(study_groups)
        
        return {
            'total_studies': len(study_groups),
            'total_files': total_files,
            'avg_files_per_study': total_files / len(study_groups),
            'avg_confidence': avg_confidence,
            'sequence_distribution': dict(sequence_distribution),
            'studies_by_confidence': {
                'high (>0.8)': len([g for g in study_groups if g.confidence > 0.8]),
                'medium (0.6-0.8)': len([g for g in study_groups if 0.6 <= g.confidence <= 0.8]),
                'low (<0.6)': len([g for g in study_groups if g.confidence < 0.6])
            }
        }
    
    def add_custom_pattern(
        self, 
        name: str, 
        regex: str, 
        study_id_groups: List[int], 
        sequence_group: int, 
        confidence: float = 0.8
    ) -> bool:
        try:
            compiled = re.compile(regex, re.IGNORECASE)
            
            pattern = {
                'name': name,
                'regex': regex,
                'study_id_groups': study_id_groups,
                'sequence_group': sequence_group,
                'confidence': confidence,
                'compiled_regex': compiled
            }
            
            self._compiled_patterns.append(pattern)
            self.logger.info(f"Added custom pattern: {name}")
            return True
            
        except re.error as e:
            self.logger.error(f"Invalid custom pattern '{name}': {e}")
            return False