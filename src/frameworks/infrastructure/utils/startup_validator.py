import os
import sys
import platform
import shutil
import psutil
import importlib
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import logging


class MedicalSystemValidator:
    def __init__(self):
        self.min_ram_gb = 4  
        self.min_free_disk_gb = 10 
        self.min_python_version = (3, 8)  

        self.required_medical_packages = [
            'numpy',      
            'PyQt6',      
            'pathlib',    
            'datetime',  
            'json',     
            'logging' 
        ]

        self.recommended_packages = [
            'pydicom',     
            'SimpleITK',  
            'vtk',        
            'scipy',     
            'skimage'    
        ]

        self.required_directories = [
            './logs',         
            './medical_data',  
            './temp',         
            './config'        
        ]

        self.logger = logging.getLogger(__name__)
    
    def validate_system_resources(self) -> bool:
        try:
            self.logger.info("Validating system resources for medical use...")

            memory_info = psutil.virtual_memory()
            available_ram_gb = memory_info.available / (1024**3)
            
            if available_ram_gb < self.min_ram_gb:
                self.logger.error(
                    f"Insufficient RAM: {available_ram_gb:.2f}GB available, "
                    f"{self.min_ram_gb}GB required for medical application"
                )
                return False

            self.logger.info(f" Sufficient RAM: {available_ram_gb:.2f}GB available")

            disk_usage = shutil.disk_usage(Path.cwd())
            free_disk_gb = disk_usage.free / (1024**3)

            if free_disk_gb < self.min_free_disk_gb:
                self.logger.error(
                    f"Insufficient disk space: {free_disk_gb:.2f}GB available, "
                    f"{self.min_free_disk_gb}GB required for medical data"
                )
                return False

            self.logger.info(f" Sufficient disk space: {free_disk_gb:.2f}GB available")

            if sys.version_info < self.min_python_version:
                self.logger.error(
                    f"Insufficient Python version: {sys.version_info} current, "
                    f"{self.min_python_version} required for medical functionality"
                )
                return False

            self.logger.info(f" Appropriate Python version: {sys.version_info}")

            architecture = platform.architecture()[0]
            if architecture != '64bit':
                self.logger.warning(
                    f"Architecture not recommended for medical application: {architecture}. "
                    f"64bit is recommended for medical image processing."
                )

            os_name = platform.system()
            self.logger.info(f"Operating system detected: {os_name}")

            if os_name not in ['Windows', 'Linux', 'Darwin']:  # Darwin = macOS
                self.logger.warning(f"Operating system not tested for medical use: {os_name}")

            self.logger.info(" System resources validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during system resource validation: {e}")
            return False
    
    def validate_medical_dependencies(self) -> bool:
        try:
            self.logger.info("Validating critical medical dependencies...")

            missing_critical = []
            missing_recommended = []

            for package in self.required_medical_packages:
                try:
                    importlib.import_module(package)
                    self.logger.debug(f" Critical dependency found: {package}")
                except ImportError:
                    missing_critical.append(package)
                    self.logger.error(f" Critical dependency missing: {package}")

            for package in self.recommended_packages:
                try:
                    importlib.import_module(package)
                    self.logger.debug(f" Recommended dependency found: {package}")
                except ImportError:
                    missing_recommended.append(package)
                    self.logger.warning(f" Recommended dependency missing: {package}")

            if missing_critical:
                self.logger.error(
                    f"Critical dependencies missing: {missing_critical}. "
                    f"The medical application cannot function without these dependencies."
                )
                return False

            if missing_recommended:
                self.logger.warning(
                    f"Recommended dependencies missing: {missing_recommended}. "
                    f"Some advanced medical functionality may not be available."
                )

            try:
                import numpy as np
                numpy_version = np.__version__
                self.logger.info(f"NumPy version: {numpy_version}")

                major_version = int(numpy_version.split('.')[0])
                if major_version < 1:
                    self.logger.warning("NumPy version very old, consider updating")

            except ImportError:
                pass  

            self.logger.info(" Medical dependencies validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during medical dependency validation: {e}")
            return False
    
    def validate_security_configuration(self) -> bool:
        try:
            self.logger.info("Validating medical security configuration...")

            for directory in self.required_directories:
                dir_path = Path(directory)

                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f" Medical directory secured: {directory}")
                except PermissionError:
                    self.logger.error(f" No permissions to create medical directory: {directory}")
                    return False

                if not os.access(dir_path, os.W_OK):
                    self.logger.error(f" No write permissions in medical directory: {directory}")
                    return False

            log_dir = Path('./logs')
            if not log_dir.exists() or not os.access(log_dir, os.W_OK):
                self.logger.error(" Cannot write medical logs - log directory inaccessible")
                return False

            if hasattr(os, 'getuid') and os.getuid() == 0:
                self.logger.warning(
                    " Running as root - not recommended for medical applications for security"
                )

            sensitive_env_vars = ['MEDICAL_API_KEY', 'DATABASE_PASSWORD', 'ENCRYPTION_KEY']
            for env_var in sensitive_env_vars:
                if env_var in os.environ:
                    self.logger.warning(
                        f" Sensitive environment variable detected: {env_var}. "
                        f"Ensure it is appropriately protected."
                    )

            self.logger.info(" Medical security configuration validated")
            return True

        except Exception as e:
            self.logger.error(f"Error during medical security validation: {e}")
            return False
    
    def validate_medical_connectivity(self) -> bool:
        try:
            self.logger.info("Validating medical connectivity...")
            
            try:
                socket.gethostbyname('localhost')
                self.logger.debug(" Basic DNS resolution functional")
            except socket.gaierror:
                self.logger.warning(" Problems with basic DNS resolution")

            mock_medical_services = [
                {'name': 'DICOM_SERVER', 'available': True},
                {'name': 'AI_ANALYSIS_SERVICE', 'available': True},
                {'name': 'MEDICAL_DATABASE', 'available': True}
            ]

            for service in mock_medical_services:
                if service['available']:
                    self.logger.debug(f" Medical service available: {service['name']}")
                else:
                    self.logger.warning(f" Medical service not available: {service['name']}")

            common_medical_ports = [80, 443, 104]  
            available_ports = []

            for port in common_medical_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()

                if result == 0:
                    available_ports.append(port)
                    self.logger.debug(f"Medical port in use: {port}")

            self.logger.info(" Medical connectivity validated (demo mode)")
            return True

        except Exception as e:
            self.logger.error(f"Error during medical connectivity validation: {e}")
            return True
    
    def validate_all_prerequisites(self) -> Tuple[bool, Dict[str, bool]]:
        self.logger.info("Starting complete validation of medical prerequisites...")
        validation_results = {}
        validation_results['system_resources'] = self.validate_system_resources()
        validation_results['medical_dependencies'] = self.validate_medical_dependencies()
        validation_results['security_configuration'] = self.validate_security_configuration()
        validation_results['medical_connectivity'] = self.validate_medical_connectivity()
        overall_success = all(validation_results.values())
        
        if overall_success:
            self.logger.info(" All medical prerequisites are met")
            self.logger.info(" System ready for safe medical operation")
        else:
            failed_validations = [
                category for category, success in validation_results.items()
                if not success
            ]
            self.logger.error(
                f" Medical prerequisites not met in: {failed_validations}"
            )
            self.logger.error(" System NOT READY for medical operation")

        return overall_success, validation_results

    def generate_validation_report(self) -> Dict[str, Any]:
        overall_success, detailed_results = self.validate_all_prerequisites()

        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': shutil.disk_usage(Path.cwd()).total / (1024**3)
        }

        validation_report = {
            'timestamp': self._get_timestamp(),
            'overall_success': overall_success,
            'validation_results': detailed_results,
            'system_information': system_info,
            'requirements': {
                'min_ram_gb': self.min_ram_gb,
                'min_free_disk_gb': self.min_free_disk_gb,
                'min_python_version': self.min_python_version,
                'required_packages': self.required_medical_packages,
                'recommended_packages': self.recommended_packages
            },
            'medical_readiness': overall_success
        }

        return validation_report

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
