import os
import logging
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Union
import mimetypes
import pydicom
from deepprostate.core.domain.exceptions import (
    StorageError, DataValidationError, SecurityError, 
    create_error_context, handle_exception_with_context
)


class FileSystemValidator:
    _logger: logging.Logger
    _dicom_extensions: set
    _image_extensions: set
    _allowed_paths: List[str]
    _max_file_size_mb: float
    _max_path_length: int
    
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        
        # Supported file extensions
        self._dicom_extensions = {'.dcm', '.dicom', '.ima', '.IMA', '.DCM', '.DICOM'}
        self._image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self._video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'}
        self._document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.rtf'}
        
        # Security: Define allowed directories for medical data
        self._allowed_base_paths = set()
        
        # Maximum file sizes (in bytes)
        self._max_file_sizes = {
            'dicom': 2 * 1024 * 1024 * 1024,  # 2GB for large DICOM volumes
            'image': 100 * 1024 * 1024,       # 100MB for images
            'video': 1024 * 1024 * 1024,      # 1GB for videos
            'document': 50 * 1024 * 1024,     # 50MB for documents
            'default': 500 * 1024 * 1024      # 500MB default
        }
    
    def configure_allowed_paths(self, allowed_paths: List[str]) -> None:
        self._allowed_base_paths = {Path(p).resolve() for p in allowed_paths}
        self._logger.info(f"Configured {len(self._allowed_base_paths)} allowed base paths")
    
    def validate_file_path(
        self, 
        file_path: Union[str, Path], 
        check_existence: bool = True,
        check_permissions: bool = True,
        check_security: bool = True
    ) -> Tuple[bool, str]:
        try:
            path_obj = Path(file_path).resolve()
            
            if check_security and self._allowed_base_paths:
                if not any(str(path_obj).startswith(str(allowed)) 
                          for allowed in self._allowed_base_paths):
                    return False, f"Path outsiof allowed directories: {path_obj}"
            
            if check_security:
                path_str = str(path_obj)
                if '..' in path_str or path_str.startswith('~'):
                    return False, f"Potentially unsafe path: {path_obj}"
            
            if check_existence and not path_obj.exists():
                return False, f"File does not exist: {path_obj}"
            
            if check_existence and path_obj.exists() and not path_obj.is_file():
                return False, f"Path is not a file: {path_obj}"
            
            if check_permissions and check_existence and path_obj.exists():
                if not os.access(path_obj, os.R_OK):
                    return False, f"File is not readable: {path_obj}"
            
            if check_existence and path_obj.exists():
                file_size = path_obj.stat().st_size
                file_type = self._get_file_type(path_obj)
                max_size = self._max_file_sizes.get(file_type, self._max_file_sizes['default'])
                
                if file_size > max_size:
                    return False, f"File too large ({file_size} bytes, max {max_size}): {path_obj}"
            
            return True, ""
            
        except Exception as e:
            self._logger.error(f"Error validating file path {file_path}: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_directory_path(
        self, 
        directory_path: Union[str, Path],
        check_existence: bool = True,
        check_permissions: bool = True,
        check_security: bool = True,
        require_non_empty: bool = False
    ) -> Tuple[bool, str]:
        try:
            dir_obj = Path(directory_path).resolve()
            
            if check_security and self._allowed_base_paths:
                if not any(str(dir_obj).startswith(str(allowed)) 
                          for allowed in self._allowed_base_paths):
                    return False, f"Directory outsiof allowed paths: {dir_obj}"
            
            if check_security:
                dir_str = str(dir_obj)
                if '..' in dir_str or dir_str.startswith('~'):
                    return False, f"Potentially unsafe directory path: {dir_obj}"
            
            if check_existence and not dir_obj.exists():
                return False, f"Directory does not exist: {dir_obj}"
            
            if check_existence and dir_obj.exists() and not dir_obj.is_dir():
                return False, f"Path is not a directory: {dir_obj}"
            
            if check_permissions and check_existence and dir_obj.exists():
                if not os.access(dir_obj, os.R_OK):
                    return False, f"Directory is not readable: {dir_obj}"
                if not os.access(dir_obj, os.X_OK):
                    return False, f"Directory is not executable: {dir_obj}"
            
            if require_non_empty and check_existence and dir_obj.exists():
                if not any(dir_obj.iterdir()):
                    return False, f"Directory is empty: {dir_obj}"
            
            return True, ""
            
        except Exception as e:
            self._logger.error(f"Error validating directory path {directory_path}: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_dicom_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        try:
            is_valid, error = self.validate_file_path(file_path)
            if not is_valid:
                return False, error
            
            path_obj = Path(file_path)
            
            if path_obj.suffix.lower() not in self._dicom_extensions:
                self._logger.debug(f"File extension not in DICOM extensions: {path_obj.suffix}")
            
            try:
                ds = pydicom.dcmread(str(path_obj), stop_before_pixels=True)                
                required_elements = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID']
                missing_elements = []
                
                for element in required_elements:
                    if not hasattr(ds, element) or getattr(ds, element) is None:
                        missing_elements.append(element)
                
                if missing_elements:
                    return False, f"Missing required DICOM elements: {missing_elements}"
                
                if hasattr(ds, 'Modality'):
                    modality = str(ds.Modality).strip()
                    if not modality:
                        return False, "DICOM file has empty modality"
                
                return True, ""
                
            except Exception as dicom_error:
                return False, f"Invalid DICOM format: {str(dicom_error)}"
                
        except Exception as e:
            self._logger.error(f"Error validating DICOM file {file_path}: {e}")
            return False, f"DICOM validation error: {str(e)}"
    
    def validate_dicom_directory(
        self, 
        directory_path: Union[str, Path],
        min_dicom_files: int = 1,
        validate_all_files: bool = False
    ) -> Tuple[bool, str, List[str]]:
        try:
            is_valid, error = self.validate_directory_path(directory_path, require_non_empty=True)
            if not is_valid:
                return False, error, []
            
            dir_obj = Path(directory_path)
            dicom_files = []
            invalid_files = []
            
            for file_path in dir_obj.rglob('*'):
                if file_path.is_file():
                    if (file_path.suffix.lower() in self._dicom_extensions or 
                        file_path.suffix == ''): 
                        
                        if validate_all_files:
                            is_dicom_valid, dicom_error = self.validate_dicom_file(file_path)
                            if is_dicom_valid:
                                dicom_files.append(str(file_path))
                            else:
                                invalid_files.append(f"{file_path}: {dicom_error}")
                        else:
                            try:
                                pydicom.dcmread(str(file_path), stop_before_pixels=True)
                                dicom_files.append(str(file_path))
                            except:
                                invalid_files.append(str(file_path))
            
            if len(dicom_files) < min_dicom_files:
                return False, f"Directory contains only {len(dicom_files)} DICOM files, minimum {min_dicom_files} required", dicom_files
            
            if invalid_files and validate_all_files:
                self._logger.warning(f"Found {len(invalid_files)} invalid DICOM files in {directory_path}")
            
            return True, "", dicom_files
            
        except Exception as e:
            self._logger.error(f"Error validating DICOM directory {directory_path}: {e}")
            return False, f"Directory validation error: {str(e)}", []
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, any]:
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return {'exists': False, 'error': 'File does not exist'}
            
            stat_info = path_obj.stat()
            
            file_info = {
                'exists': True,
                'path': str(path_obj.resolve()),
                'name': path_obj.name,
                'stem': path_obj.stem,
                'suffix': path_obj.suffix,
                'size_bytes': stat_info.st_size,
                'size_mb': round(stat_info.st_size / (1024 * 1024), 2),
                'created_time': stat_info.st_ctime,
                'modified_time': stat_info.st_mtime,
                'accessed_time': stat_info.st_atime,
                'is_readable': os.access(path_obj, os.R_OK),
                'is_writable': os.access(path_obj, os.W_OK),
                'file_type': self._get_file_type(path_obj),
                'mime_type': mimetypes.guess_type(str(path_obj))[0]
            }
            
            if file_info['file_type'] == 'dicom':
                is_valid_dicom, dicom_error = self.validate_dicom_file(path_obj)
                file_info['is_valid_dicom'] = is_valid_dicom
                if not is_valid_dicom:
                    file_info['dicom_error'] = dicom_error
            
            return file_info
            
        except Exception as e:
            self._logger.error(f"Error getting file info for {file_path}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def scan_directory_for_medical_files(
        self, 
        directory_path: Union[str, Path],
        include_subdirectories: bool = True
    ) -> Dict[str, List[str]]:
        try:
            dir_obj = Path(directory_path)
            
            if not dir_obj.exists() or not dir_obj.is_dir():
                return {'error': [f"Invalid directory: {directory_path}"]}
            
            file_categories = {
                'dicom_files': [],
                'image_files': [],
                'video_files': [],
                'document_files': [],
                'other_files': [],
                'invalid_files': []
            }
            
            scan_method = dir_obj.rglob if include_subdirectories else dir_obj.glob
            
            for file_path in scan_method('*'):
                if file_path.is_file():
                    try:
                        is_valid, error = self.validate_file_path(file_path, check_security=False)
                        if not is_valid:
                            file_categories['invalid_files'].append(f"{file_path}: {error}")
                            continue
                        
                        file_type = self._get_file_type(file_path)
                        
                        if file_type == 'dicom':
                            file_categories['dicom_files'].append(str(file_path))
                        elif file_type == 'image':
                            file_categories['image_files'].append(str(file_path))
                        elif file_type == 'video':
                            file_categories['video_files'].append(str(file_path))
                        elif file_type == 'document':
                            file_categories['document_files'].append(str(file_path))
                        else:
                            file_categories['other_files'].append(str(file_path))
                            
                    except Exception as e:
                        file_categories['invalid_files'].append(f"{file_path}: {str(e)}")
            
            return file_categories
            
        except Exception as e:
            self._logger.error(f"Error scanning directory {directory_path}: {e}")
            return {'error': [f"Scan error: {str(e)}"]}
    
    def _get_file_type(self, file_path: Path) -> str:
        suffix_lower = file_path.suffix.lower()
        
        if suffix_lower in self._dicom_extensions:
            return 'dicom'
        elif suffix_lower in self._image_extensions:
            return 'image'
        elif suffix_lower in self._video_extensions:
            return 'video'
        elif suffix_lower in self._document_extensions:
            return 'document'
        else:
            if suffix_lower == '':
                try:
                    pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    return 'dicom'
                except:
                    pass
            
            return 'other'
    
    def clean_path_string(self, path_string: str) -> str:
        try:
            cleaned = path_string.strip()            
            cleaned = ''.join(char for char in cleaned if ord(char) >= 32)            
            cleaned = cleaned.replace('\\', '/')

            while '//' in cleaned:
                cleaned = cleaned.replace('//', '/')
            
            return cleaned
            
        except Exception as e:
            self._logger.error(f"Error cleaning path string: {e}")
            return ""

filesystem_validator = FileSystemValidator()