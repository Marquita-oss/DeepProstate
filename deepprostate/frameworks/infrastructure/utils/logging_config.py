import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import sys


class MedicalLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        medical_timestamp = datetime.fromtimestamp(record.created).isoformat()

        log_data = {
            'timestamp': medical_timestamp,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'medical_context'):
            log_data['medical_context'] = record.medical_context

        if hasattr(record, 'patient_id'):
            log_data['patient_context'] = f"PATIENT_{hash(record.patient_id) % 10000:04d}"

        if hasattr(record, 'workflow_id'):
            log_data['workflow_id'] = record.workflow_id

        return json.dumps(log_data, ensure_ascii=False)


class HIPAACompliantFilter(logging.Filter):    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            'ssn', 'social_security', 'birth_date',
            'address', 'phone', 'email_personal'
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()

        for pattern in self.sensitive_patterns:
            if pattern in message:
                record.msg = record.msg.replace(pattern, '[REDACTED_FOR_HIPAA]')
        
        return True


def setup_medical_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    console_level: Union[str, int] = None,
    medical_audit: bool = True,
    hipaa_compliant: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    # Disable root logger to prevent any console output
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.CRITICAL + 1)  # Disable all root logging

    logger = logging.getLogger('medical_workstation')

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        numeric_level = log_level
    logger.setLevel(numeric_level)

    if medical_audit:
        formatter = MedicalLogFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    if console_output:
        if console_level is not None:
            if isinstance(console_level, str):
                console_numeric_level = getattr(logging, console_level.upper(), logging.ERROR)
            else:
                console_numeric_level = console_level
        else:
            console_numeric_level = numeric_level

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_numeric_level)
        console_handler.setFormatter(formatter)

        if hipaa_compliant:
            console_handler.addFilter(HIPAACompliantFilter())

        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )

        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

        if hipaa_compliant:
            file_handler.addFilter(HIPAACompliantFilter())

        logger.addHandler(file_handler)

    if medical_audit and log_file:
        audit_file = str(log_path.parent / f"{log_path.stem}_audit.jsonl")
        
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=max_file_size,
            backupCount=backup_count * 2, 
            encoding='utf-8'
        )
        
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(MedicalLogFormatter())
        
        if hipaa_compliant:
            audit_handler.addFilter(HIPAACompliantFilter())
        
        logger.addHandler(audit_handler)
    
    logger.info("Sistema of logging médico configurado correctamente", extra={
        'medical_context': 'system_initialization',
        'log_level': numeric_level,
        'medical_audit': medical_audit,
        'hipaa_compliant': hipaa_compliant,
        'log_file': log_file
    })
    
    return logger


def create_medical_log_entry(
    logger: logging.Logger,
    level: str,
    message: str,
    medical_context: Optional[str] = None,
    patient_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    extra_data = {}
    
    if medical_context:
        extra_data['medical_context'] = medical_context
    
    if patient_id:
        extra_data['patient_id'] = patient_id
    
    if workflow_id:
        extra_data['workflow_id'] = workflow_id
    
    if additional_data:
        extra_data.update(additional_data)
    
    log_method = getattr(logger, level.lower(), logger.info)
    
    log_method(message, extra=extra_data)


def setup_development_logging() -> logging.Logger:
    return setup_medical_logging(
        log_level=logging.DEBUG,
        log_file="./logs/development.log",
        console_output=True,
        medical_audit=False,
        hipaa_compliant=False
    )


def get_medical_logger(name: str = 'medical_workstation') -> logging.Logger:
  
    return logging.getLogger(name)
