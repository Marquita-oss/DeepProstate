import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np
from deepprostate.core.domain.utils.medical_shape_handler import MedicalShapeHandler
import pydicom
from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
from deepprostate.core.domain.entities.medical_image import (
    MedicalImage, ImageSpacing, ImageModalityType, WindowLevel
)
from deepprostate.core.domain.repositories.repositories import (
    MedicalImageRepository, RepositoryError, ImageNotFoundError,
    DuplicateEntityError, InvalidQueryError
)
from deepprostate.frameworks.infrastructure.utils.dicom_metadata_extractor import dicom_extractor


class DICOMImageRepository(MedicalImageRepository):    
    def __init__(self, storage_path: str):
        self._storage_path = Path(storage_path)
        self._images_path = self._storage_path / "images"
        self._metadata_path = self._storage_path / "metadata"
        self._index_file = self._storage_path / "index.json"

        self._images_path.mkdir(parents=True, exist_ok=True)
        self._metadata_path.mkdir(parents=True, exist_ok=True)

        self._index = self._load_or_create_index()
        self._supported_extensions = {'.dcm', '.dicom', '.ima'}
    
    async def save_image(self, image: MedicalImage) -> bool:
        try:
            if await self.exists_image(image.series_instance_uid):
                raise DuplicateEntityError(
                    f"La image with series UID {image.series_instance_uid} ya existe"
                )

            patient_dir = self._images_path / self._sanitize_filename(image.patient_id)
            study_dir = patient_dir / self._sanitize_filename(image.study_instance_uid)
            study_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{self._sanitize_filename(image.series_instance_uid)}.dcm"
            dicom_file_path = study_dir / filename

            dicom_dataset = await self._create_dicom_from_image(image)
            dicom_dataset.save_as(str(dicom_file_path), write_like_original=False)

            metadata_file = self._metadata_path / f"{image.series_instance_uid}.json"
            await self._save_extended_metadata(image, metadata_file)

            self._update_index(image, str(dicom_file_path))
            await self._save_index()
            
            return True
            
        except Exception as e:
            raise RepositoryError(f"Error guardando image DICOM: {e}") from e
    
    async def find_by_series_uid(self, series_uid: str) -> Optional[MedicalImage]:
        try:
            image_info = self._index.get("series", {}).get(series_uid)
            if not image_info:
                return None

            dicom_path = Path(image_info["file_path"])
            if not dicom_path.exists():
                await self._remove_from_index(series_uid)
                return None

            image = await self._load_dicom_as_image(dicom_path)
            
            return image
            
        except Exception as e:
            raise RepositoryError(f"Error buscando image by series UID: {e}") from e
    
    async def find_by_study_uid(self, study_uid: str) -> List[MedicalImage]:
        try:
            study_info = self._index.get("studies", {}).get(study_uid)
            if not study_info:
                return []

            images = []
            series_uids = study_info.get("series_list", [])

            load_tasks = [self.find_by_series_uid(uid) for uid in series_uids]
            loaded_images = await asyncio.gather(*load_tasks, return_exceptions=True)

            for img in loaded_images:
                if isinstance(img, MedicalImage):
                    images.append(img)
                elif isinstance(img, Exception):
                    logging.debug(f"Warning: Error cargando image of the estudio {study_uid}: {img}")
            
            return images
            
        except Exception as e:
            raise RepositoryError(f"Error buscando imágenes by study UID: {e}") from e
    
    async def find_by_patient_id(self, patient_id: str) -> List[MedicalImage]:
        try:
            patient_info = self._index.get("patients", {}).get(patient_id)
            if not patient_info:
                return []

            images = []
            study_uids = patient_info.get("study_list", [])

            for study_uid in study_uids:
                study_images = await self.find_by_study_uid(study_uid)
                images.extend(study_images)
            
            return images
            
        except Exception as e:
            raise RepositoryError(f"Error buscando imágenes by patient ID: {e}") from e
    
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[MedicalImage]:
        try:
            if start_date > end_date:
                raise InvalidQueryError("La fecha of inicio debe ser anterior a la fecha of fin")

            matching_images = []

            for series_uid, series_info in self._index.get("series", {}).items():
                acquisition_date_str = series_info.get("acquisition_date")
                if not acquisition_date_str:
                    continue
                
                try:
                    acquisition_date = datetime.fromisoformat(acquisition_date_str)
                    if start_date <= acquisition_date <= end_date:
                        image = await self.find_by_series_uid(series_uid)
                        if image:
                            matching_images.append(image)
                except ValueError:
                    continue
            
            return matching_images
            
        except Exception as e:
            raise RepositoryError(f"Error buscando imágenes by rango of fechas: {e}") from e
    
    async def delete_image(self, series_uid: str) -> bool:
        try:
            image_info = self._index.get("series", {}).get(series_uid)
            if not image_info:
                return False

            dicom_path = Path(image_info["file_path"])
            if dicom_path.exists():
                dicom_path.unlink()

            metadata_file = self._metadata_path / f"{series_uid}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            await self._remove_from_index(series_uid)
            await self._save_index()
            
            return True
            
        except Exception as e:
            raise RepositoryError(f"Error deleting imagen: {e}") from e
    
    async def update_image_metadata(
        self,
        series_uid: str,
        metadata: Dict[str, Any]
    ) -> bool:
        try:
            from deepprostate.frameworks.infrastructure.utils.input_validator import InputValidator, ValidationError
            
            validator = InputValidator()

            try:
                validated_series_uid = validator.validate_series_uid(series_uid)
            except ValidationError as ve:
                raise RepositoryError(f"Invalid series UID: {ve}")

            try:
                sanitized_metadata = validator.validate_metadata_dict(metadata)
            except ValidationError as ve:
                raise RepositoryError(f"Invalid metadata: {ve}")

            if not await self.exists_image(validated_series_uid):
                raise ImageNotFoundError(f"Imagin with series UID {validated_series_uid} no encontrada")

            metadata_file = self._metadata_path / f"{validated_series_uid}.json"
            existing_metadata = {}
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)

            try:
                existing_metadata = validator.validate_metadata_dict(existing_metadata)
            except ValidationError:
                self._logger.warning(f"Existing metadata for {validated_series_uid} failed validation, using sanitized version")
                existing_metadata = {}

            existing_metadata.update(sanitized_metadata)
            existing_metadata["last_modified"] = datetime.now().isoformat()

            final_metadata = validator.validate_metadata_dict(existing_metadata)

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(final_metadata, f, indent=2, ensure_ascii=False)

            critical_fields = ["patient_id", "study_instance_uid", "acquisition_date"]
            if any(key in sanitized_metadata for key in critical_fields):
                image = await self.find_by_series_uid(validated_series_uid)
                if image:
                    self._update_index(image, self._index["series"][validated_series_uid]["file_path"])
                    await self._save_index()
            
            self._logger.info(f"Successfully updated metadata for series {validated_series_uid}")
            return True
            
        except Exception as e:
            if isinstance(e, (RepositoryError, ImageNotFoundError)):
                raise
            raise RepositoryError(f"Error updating metadatos: {e}") from e
    
    async def exists_image(self, series_uid: str) -> bool:
        image_info = self._index.get("series", {}).get(series_uid)
        if not image_info:
            return False

        dicom_path = Path(image_info["file_path"])
        return dicom_path.exists()
    
    async def _load_dicom_as_image(self, dicom_path: Path) -> MedicalImage:
        try:
            if dicom_path.is_dir():
                try:
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
                    
                    if not dicom_names:
                        raise RepositoryError(f"No se encontraron archivos DICOM válidos in {dicom_path}")

                    reader.SetFileNames(dicom_names)
                    sitk_image = reader.Execute()
                    
                except Exception as series_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error cargando serie DICOM completa: {series_error}")
                    logger.info("Intentando cargar el primer archivo of la serie...")

                    dicom_files = list(dicom_path.glob("*.dcm")) + list(dicom_path.glob("*.dicom"))
                    if not dicom_files:
                        dicom_files = [f for f in dicom_path.iterdir() if f.is_file()]

                    if not dicom_files:
                        raise RepositoryError(f"No se encontraron archivos in the directorio {dicom_path}")

                    sitk_image = sitk.ReadImage(str(dicom_files[0]))
            else:
                try:
                    sitk_image = sitk.ReadImage(str(dicom_path))
                except Exception as file_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"SimpleITK falló, intentando with pydicom: {file_error}")
                    return await self._load_dicom_with_pydicom_fallback(dicom_path)

            image_array = sitk.GetArrayFromImage(sitk_image)

            spacing_sitk = sitk_image.GetSpacing()
            spacing = ImageSpacing(
                x=float(spacing_sitk[0]),
                y=float(spacing_sitk[1]),
                z=float(spacing_sitk[2]) if len(spacing_sitk) > 2 else 1.0
            )

            if dicom_path.is_file():
                ds = pydicom.dcmread(str(dicom_path))
            else:
                first_file = next(dicom_path.glob("*.dcm"))
                ds = pydicom.dcmread(str(first_file))

            modality = self._parse_modality(ds.get("Modality", "CT"))

            patient_id_raw = ds.get("PatientID", "")
            if not patient_id_raw or str(patient_id_raw).strip() == "":
                import logging
                logger = logging.getLogger(__name__)
                patient_id = f"ANONYMOUS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"DICOM anonimizado detectado. Usando ID generado: {patient_id}")
            else:
                patient_id = str(patient_id_raw)

            study_uid_raw = ds.get("StudyInstanceUID", "")
            if not study_uid_raw or str(study_uid_raw).strip() == "":
                study_uid = f"STUDY_{datetime.now().timestamp()}"
            else:
                study_uid = str(study_uid_raw)
                
            series_uid_raw = ds.get("SeriesInstanceUID", "")
            if not series_uid_raw or str(series_uid_raw).strip() == "":
                series_uid = f"SERIES_{datetime.now().timestamp()}"
            else:
                series_uid = str(series_uid_raw)

            acquisition_date = self._parse_dicom_date(
                ds.get("AcquisitionDate"), 
                ds.get("AcquisitionTime")
            )

            dicom_metadata = self._extract_dicom_metadata(ds)

            medical_image = MedicalImage(
                image_data=image_array,
                spacing=spacing,
                modality=modality,
                patient_id=patient_id,
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                acquisition_date=acquisition_date,
                dicom_metadata=dicom_metadata
            )
            
            return medical_image
            
        except Exception as e:
            raise RepositoryError(f"Error cargando archivo DICOM {dicom_path}: {e}") from e
    
    async def _create_dicom_from_image(self, image: MedicalImage) -> FileDataset:
        try:
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = FileDataset(
                "temp", {}, 
                file_meta=file_meta,
                preamble=b"\0" * 128
            )

            ds.PatientName = image.patient_id
            ds.PatientID = image.patient_id

            ds.StudyInstanceUID = image.study_instance_uid
            ds.StudyDate = image.acquisition_date.strftime("%Y%m%d")
            ds.StudyTime = image.acquisition_date.strftime("%H%M%S")

            ds.SeriesInstanceUID = image.series_instance_uid
            ds.SeriesDate = image.acquisition_date.strftime("%Y%m%d")
            ds.SeriesTime = image.acquisition_date.strftime("%H%M%S")
            ds.Modality = image.modality.value

            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

            image_data = image.image_data
            
            if len(image_data.shape) == 3:
                MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=3)
                depth, height, width = MedicalShapeHandler.get_spatial_dimensions(image_data)
                ds.NumberOfFrames = depth
                ds.PixelData = image_data.astype(np.uint16).tobytes()
                ds.Rows, ds.Columns = height, width
            else:
                MedicalShapeHandler.validate_medical_shape(image_data, expected_dims=2)
                height, width = image_data.shape
                ds.PixelData = image_data.astype(np.uint16).tobytes()
                ds.Rows, ds.Columns = height, width

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            ds.PixelSpacing = [image.spacing.y, image.spacing.x]
            if len(image_data.shape) == 3:
                ds.SliceThickness = image.spacing.z

            wl = image.current_window_level
            ds.WindowCenter = int(wl.level)
            ds.WindowWidth = int(wl.window)

            for key, value in image._dicom_metadata.items():
                if hasattr(ds, key) and value is not None:
                    setattr(ds, key, value)
            
            return ds
            
        except Exception as e:
            raise RepositoryError(f"Error creando DICOM from imagen: {e}") from e
    
    def _parse_modality(self, modality_str: str) -> ImageModalityType:
        modality_map = {
            "CT": ImageModalityType.CT,
            "MR": ImageModalityType.MRI,
            "US": ImageModalityType.ULTRASOUND,
            "XA": ImageModalityType.XRAY,
            "CR": ImageModalityType.XRAY,
            "PT": ImageModalityType.PET
        }
        
        return modality_map.get(modality_str.upper(), ImageModalityType.CT)
    
    def _parse_dicom_date(self, date_str: str, time_str: str = None) -> datetime:
        try:
            if not date_str:
                return datetime.now()
            
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                hour = minute = second = 0
                if time_str and len(time_str) >= 6:
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:6])
                
                return datetime(year, month, day, hour, minute, second)
            
            return datetime.now()
            
        except (ValueError, IndexError):
            return datetime.now()
    
    def _extract_dicom_metadata(self, ds: Dataset) -> Dict[str, Any]:
        return dicom_extractor.extract_standard_metadata(ds)
    
    def _load_or_create_index(self) -> Dict[str, Any]:
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "series": {},   
            "studies": {},   
            "patients": {} 
        }
    
    def _update_index(self, image: MedicalImage, file_path: str) -> None:
        self._index["series"][image.series_instance_uid] = {
            "file_path": file_path,
            "patient_id": image.patient_id,
            "study_instance_uid": image.study_instance_uid,
            "modality": image.modality.value,
            "acquisition_date": image.acquisition_date.isoformat(),
            "dimensions": list(image.dimensions),
            "spacing": {
                "x": image.spacing.x,
                "y": image.spacing.y,
                "z": image.spacing.z
            },
            "added_date": datetime.now().isoformat()
        }
        
        if image.study_instance_uid not in self._index["studies"]:
            self._index["studies"][image.study_instance_uid] = {
                "patient_id": image.patient_id,
                "study_date": image.acquisition_date.isoformat(),
                "series_list": [],
                "series_count": 0
            }
        
        study_info = self._index["studies"][image.study_instance_uid]
        if image.series_instance_uid not in study_info["series_list"]:
            study_info["series_list"].append(image.series_instance_uid)
            study_info["series_count"] = len(study_info["series_list"])
        
        if image.patient_id not in self._index["patients"]:
            self._index["patients"][image.patient_id] = {
                "study_list": [],
                "study_count": 0,
                "first_study_date": image.acquisition_date.isoformat()
            }
        
        patient_info = self._index["patients"][image.patient_id]
        if image.study_instance_uid not in patient_info["study_list"]:
            patient_info["study_list"].append(image.study_instance_uid)
            patient_info["study_count"] = len(patient_info["study_list"])
        
        self._index["last_updated"] = datetime.now().isoformat()
    
    async def _remove_from_index(self, series_uid: str) -> None:
        series_info = self._index["series"].pop(series_uid, None)
        if not series_info:
            return
        
        study_uid = series_info["study_instance_uid"]
        patient_id = series_info["patient_id"]
        
        if study_uid in self._index["studies"]:
            study_info = self._index["studies"][study_uid]
            if series_uid in study_info["series_list"]:
                study_info["series_list"].remove(series_uid)
                study_info["series_count"] = len(study_info["series_list"])
            
            if not study_info["series_list"]:
                self._index["studies"].pop(study_uid, None)
        
        if patient_id in self._index["patients"]:
            patient_info = self._index["patients"][patient_id]
            remaining_studies = [
                uid for uid in patient_info["study_list"] 
                if uid in self._index["studies"]
            ]
            
            patient_info["study_list"] = remaining_studies
            patient_info["study_count"] = len(remaining_studies)
            
            if not remaining_studies:
                self._index["patients"].pop(patient_id, None)
        
        self._index["last_updated"] = datetime.now().isoformat()
    
    async def _save_index(self) -> None:
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise RepositoryError(f"Error guardando índice: {e}") from e
    
    async def _save_extended_metadata(self, image: MedicalImage, metadata_file: Path) -> None:
        try:
            extended_metadata = {
                "series_instance_uid": image.series_instance_uid,
                "domain_metadata": {
                    "intensity_statistics": image.get_intensity_statistics(),
                    "physical_dimensions": image.get_physical_dimensions(),
                    "current_window_level": {
                        "window": image.current_window_level.window,
                        "level": image.current_window_level.level
                    }
                },
                "dicom_metadata": image._dicom_metadata,
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(extended_metadata, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            raise RepositoryError(f"Error guardando metadatos extendidos: {e}") from e
    
    def _sanitize_filename(self, filename: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        return filename[:200]
    
    async def _load_dicom_with_pydicom_fallback(self, dicom_path: Path) -> MedicalImage:
        try:
            ds = pydicom.dcmread(str(dicom_path), force=True)

            if hasattr(ds, 'pixel_array'):
                image_array = ds.pixel_array
            else:
                raise RepositoryError(f"No se pudo extraer pixel data of the archivo DICOM {dicom_path}")

            if len(image_array.shape) == 1:
                size = image_array.shape[0]
                sqrt_size = int(size ** 0.5)
                if sqrt_size * sqrt_size == size and sqrt_size >= 8:
                    image_array = image_array.reshape(sqrt_size, sqrt_size)
                else:
                    image_array = image_array.reshape(1, -1)

            pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(ds, 'SliceThickness', 1.0)
            
            spacing = ImageSpacing(
                x=float(pixel_spacing[0]) if len(pixel_spacing) > 0 else 1.0,
                y=float(pixel_spacing[1]) if len(pixel_spacing) > 1 else 1.0,
                z=float(slice_thickness) if slice_thickness else 1.0
            )

            modality = self._parse_modality(getattr(ds, 'Modality', 'CT'))
            
            patient_id_raw = getattr(ds, 'PatientID', None)
            if not patient_id_raw or str(patient_id_raw).strip() == "":
                patient_id = f"ANONYMOUS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"PatientID no disponible. Usando ID anonimizado: {patient_id}")
            else:
                patient_id = str(patient_id_raw)

            study_uid_raw = getattr(ds, 'StudyInstanceUID', None)
            study_uid = str(study_uid_raw) if study_uid_raw else f"STUDY_{datetime.now().timestamp()}"

            series_uid_raw = getattr(ds, 'SeriesInstanceUID', None)
            series_uid = str(series_uid_raw) if series_uid_raw else f"SERIES_{datetime.now().timestamp()}"

            acquisition_date = self._parse_dicom_date(
                getattr(ds, 'AcquisitionDate', None),
                getattr(ds, 'AcquisitionTime', None)
            )

            dicom_metadata = self._extract_dicom_metadata(ds)
            dicom_metadata['fallback_loader'] = 'pydicom_only'

            medical_image = MedicalImage(
                image_data=image_array,
                spacing=spacing,
                modality=modality,
                patient_id=patient_id,
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                acquisition_date=acquisition_date,
                dicom_metadata=dicom_metadata
            )
            
            return medical_image
            
        except Exception as e:
            raise RepositoryError(f"Error cargando DICOM with fallback pydicom {dicom_path}: {e}") from e