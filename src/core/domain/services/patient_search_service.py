import logging
import re
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.domain.entities.medical_image import MedicalImage


@dataclass
class SearchCriteria:
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_date_from: Optional[str] = None
    study_date_to: Optional[str] = None
    modality: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    free_text: Optional[str] = None
    case_insensitive: bool = True


@dataclass
class FilterResult:
    total_patients: int
    filtered_patients: int
    total_studies: int
    filtered_studies: int
    total_series: int
    filtered_series: int
    filter_applied: bool
    execution_time_ms: int


class PatientSearchService:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        self._cache_search_results = True
        self._search_result_cache: Dict[str, Any] = {}
        self._max_cache_entries = 100
        
        self._total_searches = 0
        self._cache_hits = 0
        
    def apply_search_filters(self, patient_data: Dict[str, Any], 
                           criteria: SearchCriteria) -> Tuple[Dict[str, Any], FilterResult]:
        start_time = datetime.now()
        
        cache_key = self._generate_cache_key(patient_data, criteria)
        if self._cache_search_results and cache_key in self._search_result_cache:
            self._cache_hits += 1
            cached_result = self._search_result_cache[cache_key]
            self._logger.debug(f"Search cache HIT: {cache_key[:16]}")
            return cached_result['filtered_data'], cached_result['filter_result']
        
        self._total_searches += 1
        original_stats = self._count_data_elements(patient_data)
        filtered_data = self._apply_filters_internal(patient_data, criteria)
        filtered_stats = self._count_data_elements(filtered_data)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        filter_result = FilterResult(
            total_patients=original_stats['patients'],
            filtered_patients=filtered_stats['patients'],
            total_studies=original_stats['studies'],
            filtered_studies=filtered_stats['studies'],
            total_series=original_stats['series'],
            filtered_series=filtered_stats['series'],
            filter_applied=self._has_active_filters(criteria),
            execution_time_ms=int(execution_time)
        )
        
        if self._cache_search_results:
            self._cache_result(cache_key, filtered_data, filter_result)
        
        self._logger.debug(f"Search completed in {execution_time:.1f}ms: "
                          f"{filtered_stats['patients']}/{original_stats['patients']} patients")
        
        return filtered_data, filter_result
    
    def _apply_filters_internal(self, patient_data: Dict[str, Any], 
                               criteria: SearchCriteria) -> Dict[str, Any]:
        filtered_data = {"patients": {}}
        
        for patient_id, patient_info in patient_data.get('patients', {}).items():
            if not self._matches_patient_criteria(patient_info, criteria):
                continue
            
            filtered_patient = patient_info.copy()
            filtered_patient['studies'] = {}
            
            for study_uid, study_info in patient_info.get('studies', {}).items():
                if not self._matches_study_criteria(study_info, criteria):
                    continue
                
                filtered_study = study_info.copy()
                filtered_study['series'] = {}
                
                for series_uid, series_info in study_info.get('series', {}).items():
                    if self._matches_series_criteria(series_info, criteria):
                        filtered_study['series'][series_uid] = series_info
                
                if filtered_study['series']:
                    filtered_patient['studies'][study_uid] = filtered_study
            
            if filtered_patient['studies']:
                filtered_data['patients'][patient_id] = filtered_patient
        
        return filtered_data
    
    def _matches_patient_criteria(self, patient_info: Dict[str, Any], 
                                 criteria: SearchCriteria) -> bool:
        if criteria.patient_id:
            patient_id = patient_info.get('patient_id', '')
            if not self._text_matches(patient_id, criteria.patient_id, criteria.case_insensitive):
                return False
        
        if criteria.patient_name:
            patient_name = patient_info.get('patient_name', '')
            if not self._text_matches(patient_name, criteria.patient_name, criteria.case_insensitive):
                return False
        
        if criteria.free_text:
            searchable_fields = [
                patient_info.get('patient_id', ''),
                patient_info.get('patient_name', ''),
                patient_info.get('sex', ''),
                patient_info.get('birth_date', '')
            ]
            
            if not any(self._text_matches(field, criteria.free_text, criteria.case_insensitive) 
                      for field in searchable_fields):
                return False
        
        return True
    
    def _matches_study_criteria(self, study_info: Dict[str, Any], 
                               criteria: SearchCriteria) -> bool:
        
        if criteria.study_date_from or criteria.study_date_to:
            study_date = study_info.get('study_date', '')
            if study_date:
                if criteria.study_date_from and study_date < criteria.study_date_from:
                    return False
                if criteria.study_date_to and study_date > criteria.study_date_to:
                    return False
            elif criteria.study_date_from or criteria.study_date_to:
                return False
        
        if criteria.modality:
            modality = study_info.get('modality', '')
            if not self._text_matches(modality, criteria.modality, criteria.case_insensitive):
                return False
        
        if criteria.study_description:
            study_desc = study_info.get('study_description', '')
            if not self._text_matches(study_desc, criteria.study_description, criteria.case_insensitive):
                return False
        
        if criteria.free_text:
            searchable_fields = [
                study_info.get('study_uid', ''),
                study_info.get('study_description', ''),
                study_info.get('modality', ''),
                study_info.get('study_date', '')
            ]
            
            if not any(self._text_matches(field, criteria.free_text, criteria.case_insensitive) 
                      for field in searchable_fields):
                has_matching_series = False
                for series_info in study_info.get('series', {}).values():
                    if self._matches_series_free_text(series_info, criteria.free_text, criteria.case_insensitive):
                        has_matching_series = True
                        break
                
                if not has_matching_series:
                    return False
        
        return True
    
    def _matches_series_criteria(self, series_info: Dict[str, Any], 
                                criteria: SearchCriteria) -> bool:
        if criteria.series_description:
            series_desc = series_info.get('series_description', '')
            if not self._text_matches(series_desc, criteria.series_description, criteria.case_insensitive):
                return False
        
        if criteria.free_text:
            if not self._matches_series_free_text(series_info, criteria.free_text, criteria.case_insensitive):
                return False
        
        return True
    
    def _matches_series_free_text(self, series_info: Dict[str, Any], free_text: str, 
                                 case_insensitive: bool) -> bool:
        searchable_fields = [
            series_info.get('series_uid', ''),
            series_info.get('series_description', ''),
            series_info.get('modality', ''),
            series_info.get('series_number', ''),
            str(series_info.get('images_count', '')),
            series_info.get('slice_thickness', '')
        ]
        
        return any(self._text_matches(field, free_text, case_insensitive) 
                  for field in searchable_fields)
    
    def _text_matches(self, text: str, pattern: str, case_insensitive: bool) -> bool:
        if not text or not pattern:
            return not pattern  
        
        if case_insensitive:
            text = text.lower()
            pattern = pattern.lower()
        
        if '*' in pattern or '?' in pattern:
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            return bool(re.search(regex_pattern, text))
        else:
            return pattern in text
    
    def _has_active_filters(self, criteria: SearchCriteria) -> bool:
        return any([
            criteria.patient_id,
            criteria.patient_name,
            criteria.study_date_from,
            criteria.study_date_to,
            criteria.modality,
            criteria.study_description,
            criteria.series_description,
            criteria.free_text
        ])
    
    def _count_data_elements(self, patient_data: Dict[str, Any]) -> Dict[str, int]:
        patients = patient_data.get('patients', {})
        patient_count = len(patients)
        study_count = 0
        series_count = 0
        
        for patient_info in patients.values():
            studies = patient_info.get('studies', {})
            study_count += len(studies)
            
            for study_info in studies.values():
                series = study_info.get('series', {})
                series_count += len(series)
        
        return {
            'patients': patient_count,
            'studies': study_count,
            'series': series_count
        }
    
    def _generate_cache_key(self, patient_data: Dict[str, Any], 
                           criteria: SearchCriteria) -> str:
        data_hash = hash(str(sorted(patient_data.get('patients', {}).keys())))
        criteria_str = (f"{criteria.patient_id}|{criteria.patient_name}|"
                       f"{criteria.study_date_from}|{criteria.study_date_to}|"
                       f"{criteria.modality}|{criteria.study_description}|"
                       f"{criteria.series_description}|{criteria.free_text}|"
                       f"{criteria.case_insensitive}")
        criteria_hash = hash(criteria_str)
        
        return f"{data_hash}_{criteria_hash}"
    
    def _cache_result(self, cache_key: str, filtered_data: Dict[str, Any],
                     filter_result: FilterResult) -> None:
        if len(self._search_result_cache) >= self._max_cache_entries:
            oldest_key = next(iter(self._search_result_cache))
            del self._search_result_cache[oldest_key]
        
        self._search_result_cache[cache_key] = {
            'filtered_data': filtered_data,
            'filter_result': filter_result,
            'timestamp': datetime.now()
        }
    
    def create_search_criteria(self, **kwargs) -> SearchCriteria:
        return SearchCriteria(**kwargs)
    
    def get_available_modalities(self, patient_data: Dict[str, Any]) -> Set[str]:
        modalities = set()
        
        for patient_info in patient_data.get('patients', {}).values():
            for study_info in patient_info.get('studies', {}).values():
                modality = study_info.get('modality', '').strip()
                if modality:
                    modalities.add(modality)
                
                # Also check series-level modalities
                for series_info in study_info.get('series', {}).values():
                    series_modality = series_info.get('modality', '').strip()
                    if series_modality:
                        modalities.add(series_modality)
        
        return modalities
    
    def get_date_range(self, patient_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        dates = []
        
        for patient_info in patient_data.get('patients', {}).values():
            for study_info in patient_info.get('studies', {}).values():
                study_date = study_info.get('study_date', '').strip()
                if study_date:
                    dates.append(study_date)
        
        if not dates:
            return None, None
        
        dates.sort()
        return dates[0], dates[-1]
    
    def clear_search_cache(self) -> None:
        cleared_count = len(self._search_result_cache)
        self._search_result_cache.clear()
        
        if cleared_count > 0:
            self._logger.info(f"Cleared {cleared_count} cached search results")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        cache_hit_rate = (self._cache_hits / self._total_searches * 100) if self._total_searches > 0 else 0
        
        return {
            'total_searches': self._total_searches,
            'cache_hits': self._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'cached_results': len(self._search_result_cache),
            'max_cache_entries': self._max_cache_entries
        }
    
    def configure_cache(self, enable_cache: bool = True, max_entries: int = 100) -> None:
        self._cache_search_results = enable_cache
        self._max_cache_entries = max_entries
        
        if not enable_cache:
            self.clear_search_cache()
        
        self._logger.debug(f"Search cache configured: enabled={enable_cache}, max_entries={max_entries}")
    
    def suggest_search_terms(self, patient_data: Dict[str, Any], 
                           partial_term: str, max_suggestions: int = 10) -> List[str]:
        suggestions = set()
        partial_lower = partial_term.lower()
        
        for patient_info in patient_data.get('patients', {}).values():
            for field in ['patient_id', 'patient_name']:
                value = patient_info.get(field, '').strip()
                if value and partial_lower in value.lower():
                    suggestions.add(value)
            
            for study_info in patient_info.get('studies', {}).values():
                for field in ['study_description', 'modality']:
                    value = study_info.get(field, '').strip()
                    if value and partial_lower in value.lower():
                        suggestions.add(value)
                
                for series_info in study_info.get('series', {}).values():
                    for field in ['series_description', 'modality']:
                        value = series_info.get(field, '').strip()
                        if value and partial_lower in value.lower():
                            suggestions.add(value)
        
        sorted_suggestions = sorted(suggestions)[:max_suggestions]
        return sorted_suggestions