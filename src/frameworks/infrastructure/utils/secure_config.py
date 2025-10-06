import os
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SecureConfigPaths:
    storage_path: str
    model_path: str
    logs_path: str
    temp_path: str
    config_path: str
    
    def __post_init__(self):
        self._validate_paths()
    
    def _validate_paths(self):
        paths_to_validate = [
            ('storage_path', self.storage_path),
            ('logs_path', self.logs_path), 
            ('temp_path', self.temp_path),
            ('config_path', self.config_path)
        ]
        
        for path_name, path_value in paths_to_validate:
            try:
                path_obj = Path(path_value).resolve()
                path_obj.mkdir(parents=True, exist_ok=True)                
                setattr(self, path_name, str(path_obj))
                
            except Exception as e:
                raise ValueError(f"Invalid {path_name}: {path_value} - {e}")
        
        if self.model_path and not Path(self.model_path).exists():
            logging.warning(f"Model path does not exist: {self.model_path}")


class SecureConfigManager:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._config: Optional[Dict[str, Any]] = None
        self._paths: Optional[SecureConfigPaths] = None
    
    def load_configuration(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        try:
            config = self._get_secure_defaults()
            
            if config_file:
                file_config = self._load_config_file(config_file)
                if file_config:
                    config.update(file_config)
            
            env_config = self._load_environment_config()
            config.update(env_config)
            config = self._validate_and_secure_config(config)
            
            self._config = config
            self._logger.info("Configuration loaded securely")
            
            return config
            
        except Exception as e:
            self._logger.error(f"Error loading configuration: {e}")
            return self._get_secure_defaults()
    
    def _get_secure_defaults(self) -> Dict[str, Any]:
        base_dir = os.getenv('MEDICAL_APP_BASE_DIR', os.getcwd())
        
        return {
            'storage_path': os.path.join(base_dir, 'medical_data'),
            'ai_config': {
                'model_path': os.getenv('MEDICAL_MODEL_PATH', None),  # Configured dynamically by user
                'confidence_threshold': float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.7')),
                'preprocessing_params': {
                    'normalize': True,
                    'resample': True,
                    'target_spacing': [1.0, 1.0, 3.0]
                }
            },
            'visualization_config': {
                'default_window_width': int(os.getenv('DEFAULT_WINDOW_WIDTH', '400')),
                'default_window_level': int(os.getenv('DEFAULT_WINDOW_LEVEL', '40')),
                'enable_gpu_rendering': os.getenv('ENABLE_GPU_RENDERING', 'true').lower() == 'true',
                'max_texture_memory_mb': int(os.getenv('MAX_TEXTURE_MEMORY_MB', '512'))
            },
            'logging_config': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'medical_audit': os.getenv('MEDICAL_AUDIT', 'true').lower() == 'true',
                'hipaa_compliant': os.getenv('HIPAA_COMPLIANT', 'true').lower() == 'true',
                'log_file': os.getenv('LOG_FILE', os.path.join(base_dir, 'logs', 'medical_imaging.log'))
            },
            'security_config': {
                'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '1024')),  # 1GB default
                'allowed_extensions': ['.dcm', '.dicom', '.ima', '.IMA'],
                'enable_path_validation': True,
                'restrict_system_paths': True
            }
        }
    
    def _load_config_file(self, config_file: str) -> Optional[Dict[str, Any]]:
        try:
            config_path = Path(config_file).resolve()
            
            if self._is_system_path(str(config_path)):
                self._logger.error(f"Access to system path denied: {config_file}")
                return None
            
            if not config_path.exists() or not config_path.is_file():
                self._logger.warning(f"Configuration file not found: {config_file}")
                return None
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self._logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error loading configuration file {config_file}: {e}")
            return None
    
    def _load_environment_config(self) -> Dict[str, Any]:
        env_config = {}
        
        if os.getenv('MEDICAL_STORAGE_PATH'):
            env_config['storage_path'] = os.getenv('MEDICAL_STORAGE_PATH')
        
        ai_config = {}
        if os.getenv('MEDICAL_MODEL_PATH'):
            ai_config['model_path'] = os.getenv('MEDICAL_MODEL_PATH')
        if os.getenv('AI_CONFIDENCE_THRESHOLD'):
            ai_config['confidence_threshold'] = float(os.getenv('AI_CONFIDENCE_THRESHOLD'))
        
        if ai_config:
            env_config['ai_config'] = ai_config
        
        logging_config = {}
        if os.getenv('LOG_LEVEL'):
            logging_config['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            logging_config['log_file'] = os.getenv('LOG_FILE')
        
        if logging_config:
            env_config['logging_config'] = logging_config
        
        return env_config
    
    def _validate_and_secure_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if 'storage_path' in config:
                config['storage_path'] = self._validate_path(config['storage_path'], 'storage')
            
            if 'ai_config' in config and 'model_path' in config['ai_config']:
                model_path = config['ai_config']['model_path']
                if model_path:  
                    config['ai_config']['model_path'] = self._validate_path(model_path, 'model', create_if_missing=False)
            
            if 'logging_config' in config and 'log_file' in config['logging_config']:
                log_file = config['logging_config']['log_file']
                log_dir = str(Path(log_file).parent)
                validated_log_dir = self._validate_path(log_dir, 'logs')
                config['logging_config']['log_file'] = os.path.join(validated_log_dir, Path(log_file).name)
            
            if 'ai_config' in config:
                ai_config = config['ai_config']
                if 'confidence_threshold' in ai_config:
                    threshold = ai_config['confidence_threshold']
                    if not (0.0 <= threshold <= 1.0):
                        self._logger.warning(f"Invalid confidence threshold {threshold}, using default 0.7")
                        ai_config['confidence_threshold'] = 0.7
            
            if 'visualization_config' in config:
                vis_config = config['visualization_config']
                if 'max_texture_memory_mb' in vis_config:
                    memory_mb = vis_config['max_texture_memory_mb']
                    if not (64 <= memory_mb <= 8192): 
                        self._logger.warning(f"Invalid texture memory {memory_mb}MB, using default 512MB")
                        vis_config['max_texture_memory_mb'] = 512
            
            return config
            
        except Exception as e:
            self._logger.error(f"Error validating configuration: {e}")
            return self._get_secure_defaults()
    
    def _validate_path(self, path: str, path_type: str, create_if_missing: bool = True) -> str:
        try:
            path_obj = Path(path).resolve()
            
            if self._is_system_path(str(path_obj)):
                raise ValueError(f"Access to system path denied: {path}")
            
            if create_if_missing:
                path_obj.mkdir(parents=True, exist_ok=True)
            
            return str(path_obj)
            
        except Exception as e:
            self._logger.error(f"Invalid {path_type} path '{path}': {e}")
            raise ValueError(f"Invalid {path_type} path: {path}")
    
    def _is_system_path(self, path: str) -> bool:
        path_lower = path.lower()
        system_paths = [
            '/etc/', '/sys/', '/proc/', '/dev/', '/boot/', '/usr/bin/', '/sbin/',
            '/var/log/', '/tmp/', '/root/'
        ]
        
        windows_paths = [
            'system32', 'windows/system32', 'program files', 'programdata',
            'users/public', 'windows/temp'
        ]
        
        forbidden_paths = system_paths + windows_paths
        
        return any(forbiddin in path_lower for forbiddin in forbidden_paths)
    
    def get_paths(self) -> SecureConfigPaths:
        if not self._config:
            self._config = self.load_configuration()
        
        if not self._paths:
            base_dir = Path(self._config['storage_path']).parent
            
            self._paths = SecureConfigPaths(
                storage_path=self._config['storage_path'],
                model_path=self._config['ai_config']['model_path'],
                logs_path=str(base_dir / 'logs'),
                temp_path=str(base_dir / 'temp'),
                config_path=str(base_dir / 'config')
            )
        
        return self._paths
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        if not self._config:
            self._config = self.load_configuration()
        
        try:
            value = self._config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default