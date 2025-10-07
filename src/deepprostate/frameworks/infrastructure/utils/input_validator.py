import re
import html
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path


class ValidationError(Exception):
    pass


class InputValidator:    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        self._sql_injection_patterns = [
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b|\bUPDATE\b)",
            r"(\-\-|\#|/\*|\*/)",
            r"(\bOR\b|\bAND\b).*?(\=|\<|\>)",
            r"(\bEXEC\b|\bEXECUTE\b)"
        ]
        
        self._script_injection_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"expression\s*\("
        ]
        
        self._dicom_tag_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9]*$')
    
    def validate_metadata_dict(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            raise ValidationError(f"Metadata must be dictionary, got {type(metadata)}")
        
        if len(metadata) > 100:
            raise ValidationError("Metadata dictionary too large (max 100 keys)")
        
        sanitized_metadata = {}
        
        for key, value in metadata.items():
            sanitized_key = self.validate_metadata_key(key)            
            sanitized_value = self.validate_metadata_value(value)
            sanitized_metadata[sanitized_key] = sanitized_value
        
        self._logger.debug(f"Validated metadata with {len(sanitized_metadata)} fields")
        return sanitized_metadata
    
    def validate_metadata_key(self, key: str) -> str:
        if not isinstance(key, str):
            raise ValidationError(f"Metadata key must be string, got {type(key)}")
        
        if not key.strip():
            raise ValidationError("Metadata key cannot be empty")
        
        if len(key) > 100:
            raise ValidationError("Metadata key too long (max 100 characters)")
        
        if self._contains_sql_injection(key):
            raise ValidationError("Metadata key contains potentially dangerous SQL patterns")
        
        if self._contains_script_injection(key):
            raise ValidationError("Metadata key contains potentially dangerous script patterns")
        
        sanitized_key = key.strip()
        
        if not (self._dicom_tag_pattern.match(sanitized_key) or sanitized_key.startswith('Custom')):
            sanitized_key = re.sub(r'[^A-Za-z0-9_]', '_', sanitized_key)
        
        return sanitized_key
    
    def validate_metadata_value(self, value: Any) -> Any:
        if value is None:
            return None
        
        if isinstance(value, str):
            return self.sanitize_string(value)
        elif isinstance(value, (int, float)):
            return self.validate_numeric_value(value)
        elif isinstance(value, bool):
            return value
        elif isinstance(value, list):
            if len(value) > 50: 
                raise ValidationError("Metadata list too large (max 50 items)")
            return [self.validate_metadata_value(item) for item in value]
        elif isinstance(value, dict):
            if len(value) > 20: 
                raise ValidationError("Nested metadata dictionary too large (max 20 keys)")
            return {k: self.validate_metadata_value(v) for k, v in value.items()}
        else:
            return self.sanitize_string(str(value))
    
    def sanitize_string(self, text: str, max_length: int = 1000) -> str:
        if not isinstance(text, str):
            text = str(text)
        
        if len(text) > max_length:
            raise ValidationError(f"String too long (max {max_length} characters)")
        
        if self._contains_sql_injection(text):
            raise ValidationError("String contains potentially dangerous SQL patterns")
        
        if self._contains_script_injection(text):
            raise ValidationError("String contains potentially dangerous script patterns")
        
        sanitized = html.escape(text)
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized.strip()
    
    def validate_numeric_value(self, value: Union[int, float]) -> Union[int, float]:
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Expected numeric value, got {type(value)}")
        
        if isinstance(value, float):
            if not (float('-inf') < value < float('inf')):
                raise ValidationError("Invalid float value (NaN or infinity)")
        
        if abs(value) > 1e10:
            raise ValidationError("Numeric value too large for medical data")
        
        return value
    
    def validate_file_path(self, file_path: str, must_exist: bool = True) -> str:
        if not isinstance(file_path, str):
            raise ValidationError(f"File path must be string, got {type(file_path)}")
        
        if not file_path.strip():
            raise ValidationError("File path cannot be empty")
        
        if len(file_path) > 500:
            raise ValidationError("File path too long (max 500 characters)")
        
        try:
            path_obj = Path(file_path).resolve()
            
            if '..' in str(path_obj) or str(path_obj) != str(Path(file_path).resolve()):
                raise ValidationError("Path traversal attempt detected")
            
            if self._is_system_path(str(path_obj)):
                raise ValidationError("Access to system paths not allowed")
            
            if must_exist and not path_obj.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            return str(path_obj)
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid file path: {e}")
    
    def validate_series_uid(self, series_uid: str) -> str:
        if not isinstance(series_uid, str):
            raise ValidationError(f"Series UID must be string, got {type(series_uid)}")
        
        if series_uid == "UNKNOWN":
            return series_uid
        
        uid_pattern = re.compile(r'^[0-9]+(\.[0-9]+)*$')
        if not uid_pattern.match(series_uid):
            raise ValidationError("Invalid DICOM Series UID format")
        
        if len(series_uid) > 64:  
            raise ValidationError("Series UID too long (max 64 characters)")
        
        return series_uid
    
    def _contains_sql_injection(self, text: str) -> bool:
        text_upper = text.upper()
        for pattern in self._sql_injection_patterns:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        return False
    
    def _contains_script_injection(self, text: str) -> bool:
        for pattern in self._script_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_system_path(self, path: str) -> bool:
        path_lower = path.lower()
        
        system_paths = [
            '/etc/', '/sys/', '/proc/', '/dev/', '/boot/', '/usr/bin/', '/sbin/',
            '/var/log/', '/tmp/', '/root/'
        ]
        
        windows_paths = [
            'system32', 'windows/system32', 'program files', 'programdata'
        ]
        
        forbidden_paths = system_paths + windows_paths
        
        return any(forbiddin in path_lower for forbiddin in forbidden_paths)
    
    def validate_confidence_score(self, confidence: float) -> float:
        if not isinstance(confidence, (int, float)):
            raise ValidationError(f"Confidence must be numeric, got {type(confidence)}")
        
        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        return float(confidence)