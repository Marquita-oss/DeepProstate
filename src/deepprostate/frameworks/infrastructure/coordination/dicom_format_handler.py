import logging
import pydicom
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np

from .medical_format_handler import (
    MedicalFormatHandler, FormatCapabilities, LoadedImageData
)
from deepprostate.core.domain.entities.medical_image import ImageSpacing, ImageModalityType
from deepprostate.core.domain.entities.enhanced_medical_image import EnhancedMedicalImage


class DICOMFormatHandler(MedicalFormatHandler):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def get_format_name(self) -> str:
        return "DICOM"
    
    def get_capabilities(self) -> FormatCapabilities:
        return FormatCapabilities(
            can_read=True,
            can_write=False, 
            supports_metadata=True, 
            supports_series=True,   
            supports_3d=True,       
            typical_extensions=['.dcm', '.dicom', '.ima', '.IMA']
        )
    
    def can_handle_file(self, file_path: Path) -> bool:
        extension = file_path.suffix.lower()
        
        if extension in ['.dcm', '.dicom', '.ima']:
            return True
        if extension.upper() == '.IMA':
            return True
        
        if not extension:
            return self._quick_dicom_check(file_path)
        
        return False
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            if not hasattr(ds, 'SOPClassUID'):
                return False, "Missing required DICOM SOP Class UID"
            
            if not hasattr(ds, 'StudyInstanceUID'):
                return False, "Missing required Study Instance UID"
            
            if not hasattr(ds, 'SeriesInstanceUID'):
                return False, "Missing required Series Instance UID"
            
            return True, None
            
        except Exception as e:
            return False, f"DICOM validation error: {e}"
    
    def load_image(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            try:
                import SimpleITK as sitk
                
                sitk_image = sitk.ReadImage(str(file_path))
                image_array = sitk.GetArrayFromImage(sitk_image)
                
                spacing_sitk = sitk_image.GetSpacing()
                spacing = ImageSpacing(
                    x=float(spacing_sitk[0]),
                    y=float(spacing_sitk[1]),
                    z=float(spacing_sitk[2]) if len(spacing_sitk) > 2 else 1.0
                )
                
                self._logger.debug(f"SimpleITK spacing extracted: x={spacing.x}, y={spacing.y}, z={spacing.z}")                
                ds = pydicom.dcmread(str(file_path))                
                modality = self._extract_dicom_modality(ds)
                metadata = self._extract_dicom_metadata(ds)
                patient_id = self._extract_patient_id(ds)
                study_uid = getattr(ds, 'StudyInstanceUID', '')
                series_uid = getattr(ds, 'SeriesInstanceUID', '')
                acquisition_date = self._extract_acquisition_date(ds)
                
                return LoadedImageData(
                    image_array=image_array,
                    spacing=spacing,
                    modality=modality,
                    metadata=metadata,
                    patient_id=patient_id,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    acquisition_date=acquisition_date
                )
                
            except ImportError:
                self._logger.info("SimpleITK not available, falling back to pydicom-only approach")
                return self._load_image_pydicom_fallback(file_path)
            except Exception as sitk_error:
                self._logger.info(f"SimpleITK failed, falling back to pydicom: {sitk_error}")
                return self._load_image_pydicom_fallback(file_path)
            
        except Exception as e:
            self._logger.error(f"Failed to load DICOM file {file_path}: {e}")
            return None
    
    def _load_image_pydicom_fallback(self, file_path: Path) -> Optional[LoadedImageData]:
        try:
            ds = pydicom.dcmread(str(file_path))            
            image_data = self._extract_dicom_image_data(ds)
            
            if image_data is None:
                return None
            
            spacing = self._extract_dicom_spacing(ds)            
            modality = self._extract_dicom_modality(ds)            
            metadata = self._extract_dicom_metadata(ds)            
            patient_id = self._extract_patient_id(ds)
            study_uid = getattr(ds, 'StudyInstanceUID', '')
            series_uid = getattr(ds, 'SeriesInstanceUID', '')
            acquisition_date = self._extract_acquisition_date(ds)
            
            return LoadedImageData(
                image_array=image_data,
                spacing=spacing,
                modality=modality,
                metadata=metadata,
                patient_id=patient_id,
                study_uid=study_uid,
                series_uid=series_uid,
                acquisition_date=acquisition_date
            )
            
        except Exception as e:
            self._logger.error(f"Failed to load DICOM file with pydicom fallback {file_path}: {e}")
            return None
    
    def _quick_dicom_check(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                f.seek(128)  
                prefix = f.read(4)
                return prefix == b'DICM'
        except:
            return False
    
    def _extract_dicom_image_data(self, ds: pydicom.Dataset) -> Optional[np.ndarray]:
        try:
            if not hasattr(ds, 'pixel_array'):
                self._logger.error("DICOM file has no pixel data")
                return None
            
            pixel_array = ds.pixel_array            
            image_data = pixel_array.astype(np.float32)
            
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                image_data = image_data * slope + intercept
            
            return image_data
            
        except Exception as e:
            self._logger.error(f"Failed to extract DICOM image data: {e}")
            return None
    
    def _extract_dicom_spacing(self, ds: pydicom.Dataset) -> ImageSpacing:
        try:
            if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
                pixel_spacing = ds.PixelSpacing
                spacing_row = float(pixel_spacing[0])  
                spacing_col = float(pixel_spacing[1])  
                
                if abs(spacing_row - spacing_col) / max(spacing_row, spacing_col) > 0.5:
                    self._logger.info(f"DICOM spacing values differ significantly: row={spacing_row}, col={spacing_col}. Using uniform spacing.")
                    uniform_spacing = min(spacing_row, spacing_col) 
                    spacing_x = spacing_y = uniform_spacing
                else:
                    spacing_x = spacing_col
                    spacing_y = spacing_row
            else:
                spacing_x = spacing_y = 1.0
            
            if hasattr(ds, 'SliceThickness') and ds.SliceThickness:
                spacing_z = float(ds.SliceThickness)
            elif hasattr(ds, 'SpacingBetweenSlices') and ds.SpacingBetweenSlices:
                spacing_z = float(ds.SpacingBetweenSlices)
            else:
                spacing_z = max(spacing_x, spacing_y)
            
            self._logger.debug(f"DICOM spacing extracted: x={spacing_x}, y={spacing_y}, z={spacing_z}")
            
            return ImageSpacing(x=spacing_x, y=spacing_y, z=spacing_z)
            
        except Exception as e:
            self._logger.error(f"Could not extract DICOM spacing: {e}")
            return ImageSpacing(x=1.0, y=1.0, z=1.0)
    
    def _extract_dicom_modality(self, ds: pydicom.Dataset) -> ImageModalityType:
        try:
            modality_str = getattr(ds, 'Modality', '').upper()
            
            modality_map = {
                'CT': ImageModalityType.CT,
                'MR': ImageModalityType.MRI,
                'US': ImageModalityType.ULTRASOUND,
                'CR': ImageModalityType.XRAY,
                'DX': ImageModalityType.XRAY,
                'RF': ImageModalityType.XRAY,
                'PT': ImageModalityType.PET
            }
            
            return modality_map.get(modality_str, ImageModalityType.CT)
            
        except Exception as e:
            self._logger.error(f"Could not extract DICOM modality: {e}")
            return ImageModalityType.CT
    
    def _extract_dicom_metadata(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        metadata = {
            'format': 'DICOM',
            'sop_class_uid': getattr(ds, 'SOPClassUID', ''),
            'sop_instance_uid': getattr(ds, 'SOPInstanceUID', ''),
            'modality': getattr(ds, 'Modality', ''),
            'manufacturer': getattr(ds, 'Manufacturer', ''),
            'institution_name': getattr(ds, 'InstitutionName', ''),
            'series_description': getattr(ds, 'SeriesDescription', ''),
            'protocol_name': getattr(ds, 'ProtocolName', ''),
        }
        
        if hasattr(ds, 'KVP'):
            metadata['kvp'] = str(ds.KVP)
        if hasattr(ds, 'XRayTubeCurrent'):
            metadata['tube_current'] = str(ds.XRayTubeCurrent)
        if hasattr(ds, 'ExposureTime'):
            metadata['exposure_time'] = str(ds.ExposureTime)
        
        if hasattr(ds, 'RepetitionTime'):
            metadata['tr'] = str(ds.RepetitionTime)
        if hasattr(ds, 'EchoTime'):
            metadata['te'] = str(ds.EchoTime)
        if hasattr(ds, 'FlipAngle'):
            metadata['flip_angle'] = str(ds.FlipAngle)
        
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            metadata['window_center'] = str(ds.WindowCenter)
            metadata['window_width'] = str(ds.WindowWidth)
        
        return metadata
    
    def _extract_patient_id(self, ds: pydicom.Dataset) -> str:
        patient_id = getattr(ds, 'PatientID', '')
        
        if not patient_id or patient_id.strip() == '':
            patient_name = getattr(ds, 'PatientName', '')
            if patient_name:
                patient_id = str(patient_name).replace('^', '_')
            else:
                study_uid = getattr(ds, 'StudyInstanceUID', '')
                if study_uid:
                    patient_id = f"PATIENT_{study_uid.split('.')[-1]}"
                else:
                    patient_id = "UNKNOWN_PATIENT"
        
        return str(patient_id)
    
    def _extract_acquisition_date(self, ds: pydicom.Dataset) -> datetime:
        try:
            if hasattr(ds, 'StudyDate') and ds.StudyDate:
                date_str = ds.StudyDate
                if hasattr(ds, 'StudyTime') and ds.StudyTime:
                    time_str = ds.StudyTime
                    datetime_str = f"{date_str}{time_str}"
                    return datetime.strptime(datetime_str[:14], '%Y%m%d%H%M%S')
                else:
                    return datetime.strptime(date_str, '%Y%m%d')
            
            elif hasattr(ds, 'SeriesDate') and ds.SeriesDate:
                date_str = ds.SeriesDate
                return datetime.strptime(date_str, '%Y%m%d')
            
            elif hasattr(ds, 'AcquisitionDate') and ds.AcquisitionDate:
                date_str = ds.AcquisitionDate
                return datetime.strptime(date_str, '%Y%m%d')
            
        except Exception as e:
            self._logger.error(f"Could not parse DICOM date: {e}")

        return datetime.now()
    
    def create_enhanced_medical_image(self, file_path: Path) -> Optional[EnhancedMedicalImage]:
        try:
            loaded_data = self.load_image(file_path)
            if not loaded_data:
                return None
            
            enhanced_image = EnhancedMedicalImage(
                image_data=loaded_data.image_array,
                spacing=loaded_data.spacing,
                modality=loaded_data.modality,
                patient_id=loaded_data.patient_id,
                study_instance_uid=loaded_data.study_uid,
                series_instance_uid=loaded_data.series_uid,
                acquisition_date=loaded_data.acquisition_date or datetime.now(),
                dicom_metadata=loaded_data.metadata,
                auto_analyze_dimensions=True 
            )
            
            self._logger.info(f"Created enhanced DICOM image with structure: {enhanced_image.structure}")
            return enhanced_image
            
        except Exception as e:
            self._logger.error(f"Failed to create enhanced DICOM image: {e}")
            return None