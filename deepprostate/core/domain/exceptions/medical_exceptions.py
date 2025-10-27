from typing import Optional, Dict, Any

class MedicalImagingError(Exception):
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_coof = error_code
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None
        }


class DicomProcessingError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        dicom_tag: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if file_path:
            self.details['file_path'] = file_path
        if dicom_tag:
            self.details['dicom_tag'] = dicom_tag


class ImageLoadingError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        image_path: Optional[str] = None,
        series_uid: Optional[str] = None,
        memory_usage: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if image_path:
            self.details['image_path'] = image_path
        if series_uid:
            self.details['series_uid'] = series_uid
        if memory_usage:
            self.details['memory_usage_mb'] = memory_usage


class MaskFileDetectedError(MedicalImagingError):
    def __init__(
        self, 
        message: str,
        file_path: Optional[str] = None,
        detected_patterns: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, error_code="MASK_FILE_DETECTED", **kwargs)
        if file_path:
            self.details['file_path'] = file_path
        if detected_patterns:
            self.details['detected_patterns'] = detected_patterns
        
        self.is_controlled_rejection = True


class AIAnalysisError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        analysis_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if model_name:
            self.details['model_name'] = model_name
        if analysis_type:
            self.details['analysis_type'] = analysis_type
        if confidence_threshold:
            self.details['confidence_threshold'] = confidence_threshold


class SegmentationError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        segmentation_id: Optional[str] = None,
        region_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if segmentation_id:
            self.details['segmentation_id'] = segmentation_id
        if region_type:
            self.details['region_type'] = region_type
        if operation:
            self.details['operation'] = operation


class DataValidationError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        validation_type: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if validation_type:
            self.details['validation_type'] = validation_type
        if expected_value is not None:
            self.details['expected_value'] = expected_value
        if actual_value is not None:
            self.details['actual_value'] = actual_value


class WorkflowExecutionError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        workflow_id: Optional[str] = None,
        workflow_step: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if workflow_id:
            self.details['workflow_id'] = workflow_id
        if workflow_step:
            self.details['workflow_step'] = workflow_step


class StorageError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        storage_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if storage_path:
            self.details['storage_path'] = storage_path
        if operation:
            self.details['operation'] = operation


class ConfigurationError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if config_key:
            self.details['config_key'] = config_key
        if config_file:
            self.details['config_file'] = config_file


class SecurityError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        security_context: Optional[str] = None,
        attempted_operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if security_context:
            self.details['security_context'] = security_context
        if attempted_operation:
            self.details['attempted_operation'] = attempted_operation


class MemoryError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        requested_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if requested_memory:
            self.details['requested_memory_mb'] = requested_memory
        if available_memory:
            self.details['available_memory_mb'] = available_memory
        if operation:
            self.details['operation'] = operation


class UIError(MedicalImagingError):    
    def __init__(
        self, 
        message: str, 
        component_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if component_name:
            self.details['component_name'] = component_name
        if operation:
            self.details['operation'] = operation

def create_error_context(
    operation: str,
    component: Optional[str] = None,
    **additional_context
) -> Dict[str, Any]:
    context = {
        'operation': operation,
        'timestamp': None
    }

    if component:
        context['component'] = component

    context.update(additional_context)
    return context


def handle_exception_with_context(
    exception: Exception,
    context: Dict[str, Any],
    logger = None
) -> MedicalImagingError:
    if isinstance(exception, FileNotFoundError):
        return ImageLoadingError(
            f"Medical image file not found: {str(exception)}",
            error_code="IMAGE_NOT_FOUND",
            details=context,
            cause=exception
        )
    elif isinstance(exception, MemoryError):
        return MemoryError(
            f"Insufficient memory for medical imaging operation: {str(exception)}",
            error_code="INSUFFICIENT_MEMORY",
            details=context,
            cause=exception
        )
    elif isinstance(exception, PermissionError):
        return SecurityError(
            f"Permission denied for medical imaging operation: {str(exception)}",
            error_code="ACCESS_DENIED",
            details=context,
            cause=exception
        )
    elif isinstance(exception, ValueError):
        return DataValidationError(
            f"Invalid medical data value: {str(exception)}",
            error_code="INVALID_DATA_VALUE",
            details=context,
            cause=exception
        )
    else:
        return MedicalImagingError(
            f"Unexpected error in medical imaging operation: {str(exception)}",
            error_code="UNKNOWN_ERROR",
            details=context,
            cause=exception
        )