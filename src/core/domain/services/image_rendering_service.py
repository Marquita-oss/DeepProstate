import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import QPointF

from src.core.domain.entities.medical_image import MedicalImage


class ImageRenderingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0, 0)
        
        self._current_series_uid = None
        self._sequence_window_levels = {}
        self._default_window = 400
        self._default_level = 40
        
        self.logger.debug("ImageRenderingService initialized with per-sequence W/L support")
    
    def set_window_level(self, window: float, level: float, series_uid: str = None) -> None:
        window = max(1, window) 
        
        if series_uid is None:
            series_uid = self._current_series_uid
        
        if series_uid:
            self._sequence_window_levels[series_uid] = {
                'window': window,
                'level': level
            }
            self.logger.debug(f"Window/Level set for {series_uid}: {window}/{level}")
        else:
            self._default_window = window
            self._default_level = level
            self.logger.debug(f"Default Window/Level set to: {window}/{level}")
    
    def get_window_level(self, series_uid: str = None) -> Tuple[float, float]:
        if series_uid is None:
            series_uid = self._current_series_uid
        
        if series_uid and series_uid in self._sequence_window_levels:
            wl = self._sequence_window_levels[series_uid]
            return wl['window'], wl['level']
        else:
            return self._default_window, self._default_level
    
    def set_current_series(self, series_uid: str, image_data: np.ndarray = None) -> None:
        if self._current_series_uid == series_uid:
            return
        
        self._current_series_uid = series_uid
        
        if series_uid not in self._sequence_window_levels and image_data is not None:
            optimal_window, optimal_level = self._calculate_optimal_window_level_by_modality(
                image_data, series_uid
            )
            self.set_window_level(optimal_window, optimal_level, series_uid)
            self.logger.info(f"Auto-calculated W/L for {series_uid}: {optimal_window:.1f}/{optimal_level:.1f}")
    
    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._zoom_factor = max(0.1, min(10.0, zoom_factor))
        self.logger.debug(f"Zoom factor set to: {self._zoom_factor}")
    
    def get_zoom_factor(self) -> float:
        return self._zoom_factor
    
    def set_pan_offset(self, offset: QPointF) -> None:
        self._pan_offset = offset
    
    def get_pan_offset(self) -> QPointF:
        return self._pan_offset
    
    def apply_windowing(self, image_data: np.ndarray, series_uid: str = None) -> np.ndarray:
        if image_data is None or image_data.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)
        
        window, level = self.get_window_level(series_uid)
        
        lower_bound = level - window / 2
        upper_bound = level + window / 2
        
        windowed = np.clip(image_data, lower_bound, upper_bound)
        
        if upper_bound > lower_bound:
            windowed = (windowed - lower_bound) / (upper_bound - lower_bound) * 255
        else:
            windowed = np.zeros_like(windowed)
        
        return windowed.astype(np.uint8)
    
    def convert_to_qpixmap(
        self,
        image_data: np.ndarray,
        apply_windowing: bool = True,
        series_uid: str = None,
        spacing: Optional[Tuple[float, float]] = None,
        plane: str = "axial"
    ) -> Optional[QPixmap]:
        try:
            if image_data is None or image_data.size == 0:
                return None

            if len(image_data.shape) > 2:
                image_data = image_data[:, :, 0] if image_data.shape[2] == 1 else image_data

            if apply_windowing:
                processed_data = self.apply_windowing(image_data, series_uid)
            else:
                min_val, max_val = image_data.min(), image_data.max()
                if max_val > min_val:
                    processed_data = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    processed_data = np.zeros_like(image_data, dtype=np.uint8)
            
            height, width = processed_data.shape
            bytes_per_line = width
            
            qimage = QImage(
                processed_data.data.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
            
            pixmap = QPixmap.fromImage(qimage)
            
            pixmap = self._apply_aspect_ratio_correction(pixmap, spacing, plane, series_uid)
            
            if abs(self._zoom_factor - 1.0) > 0.01:
                scaled_size = pixmap.size() * self._zoom_factor
                from PyQt6.QtCore import Qt
                pixmap = pixmap.scaled(
                    scaled_size,
                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                    transformMode=Qt.TransformationMode.SmoothTransformation
                )
            
            return pixmap
            
        except Exception as e:
            self.logger.error(f"Error converting to QPixmap: {e}")
            return None
    
    def zoom_in(self, factor: float = 1.2) -> None:
        new_zoom = self._zoom_factor * factor
        self.set_zoom_factor(new_zoom)
    
    def zoom_out(self, factor: float = 1.2) -> None:
        new_zoom = self._zoom_factor / factor
        self.set_zoom_factor(new_zoom)
    
    def reset_zoom(self) -> None:
        self.set_zoom_factor(1.0)
        self.set_pan_offset(QPointF(0, 0))
    
    def adjust_pan(self, delta: QPointF) -> None:
        new_offset = self._pan_offset + delta
        self.set_pan_offset(new_offset)
    
    def get_rendering_state(self) -> Dict[str, Any]:
        current_window, current_level = self.get_window_level()
        return {
            'zoom_factor': self._zoom_factor,
            'pan_offset': {'x': self._pan_offset.x(), 'y': self._pan_offset.y()},
            'window': current_window,
            'level': current_level,
            'current_series': self._current_series_uid
        }
    
    def set_rendering_state(self, state: Dict[str, Any]) -> None:
        if 'zoom_factor' in state:
            self.set_zoom_factor(state['zoom_factor'])
        
        if 'pan_offset' in state:
            offset_data = state['pan_offset']
            self.set_pan_offset(QPointF(offset_data['x'], offset_data['y']))
        
        if 'window' in state and 'level' in state:
            series_uid = state.get('current_series', self._current_series_uid)
            self.set_window_level(state['window'], state['level'], series_uid)
        
        self.logger.debug("Rendering state restored")
    
    def _apply_aspect_ratio_correction(
        self, 
        pixmap: QPixmap, 
        spacing: Optional[Tuple[float, float]], 
        plane: str,
        series_uid: Optional[str] = None
    ) -> QPixmap:
        original_spacing = self._get_original_image_spacing(series_uid)
        if original_spacing is None or len(original_spacing) != 2:
            return pixmap
        
        spacing_x, spacing_y = original_spacing
        
        if spacing_x <= 0 or spacing_y <= 0:
            return pixmap
        
        raw_aspect_ratio = spacing_y / spacing_x 
        
        if abs(raw_aspect_ratio - 1.0) < 0.02:
            return pixmap
        
        from src.core.domain.utils.image_orientation_detector import detect_image_orientation
        
        dicom_metadata = None
        if series_uid and hasattr(self, '_series_data') and series_uid in self._series_data:
            image_data = self._series_data[series_uid]
            if hasattr(image_data, '_dicom_metadata'):
                dicom_metadata = image_data._dicom_metadata
        
        original_orientation = detect_image_orientation(dicom_metadata, series_uid)
        
        if original_orientation != plane:
            if original_orientation in ["sagittal", "coronal"] or plane in ["sagittal", "coronal"]:
                max_correction_factor = 6.0  
                correction_strength = 1.0 
            else:
                max_correction_factor = 5.0 
                correction_strength = 0.9 
        else:
            if plane in ["sagittal", "coronal"]:
                max_correction_factor = 6.0 
                correction_strength = 1.0 
            else:
                max_correction_factor = 4.0
                correction_strength = 0.8
        
        if raw_aspect_ratio > max_correction_factor:
            excess_factor = raw_aspect_ratio / max_correction_factor
            aspect_ratio = max_correction_factor * (1.0 + 0.3 * np.log(excess_factor))
            self.logger.debug(f"Progressive correction applied for {plane}: {raw_aspect_ratio:.2f} -> {aspect_ratio:.2f}")
        elif raw_aspect_ratio < (1.0 / max_correction_factor):
            min_threshold = (1.0 / max_correction_factor)
            
            if original_orientation != plane and (original_orientation in ["sagittal", "coronal"] or plane in ["sagittal", "coronal"]):
                if raw_aspect_ratio < 0.3:  
                    target_ratio = 0.8 
                    correction_power = 1.0
                else:
                    target_ratio = 0.6
                    correction_power = 0.8
            else:
                correction_power = 0.3
                target_ratio = min_threshold
                
            if raw_aspect_ratio < 0.3:
                aspect_ratio = max(0.6, min(0.9, raw_aspect_ratio * 6.0))
            else:
                excess_factor = target_ratio / raw_aspect_ratio
                aspect_ratio = target_ratio / (1.0 + correction_power * np.log(excess_factor))
            self.logger.debug(f"Progressive correction applied for {plane}: {raw_aspect_ratio:.2f} -> {aspect_ratio:.2f}")
        else:
            aspect_ratio = raw_aspect_ratio
        
        if correction_strength < 1.0:
            aspect_ratio = 1.0 + (aspect_ratio - 1.0) * correction_strength
        
        current_size = pixmap.size()
        
        if aspect_ratio > 1.0:
            new_width = int(current_size.width() * aspect_ratio)
            new_height = current_size.height()
        else:
            new_width = current_size.width()
            new_height = int(current_size.height() / aspect_ratio)
        
        from PyQt6.QtCore import QSize, Qt
        corrected_pixmap = pixmap.scaled(
            QSize(new_width, new_height),
            aspectRatioMode=Qt.AspectRatioMode.IgnoreAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        
        self.logger.debug(
            f"Aspect ratio corrected for {plane}: "
            f"spacing({spacing_x:.2f},{spacing_y:.2f}) -> "
            f"ratio({aspect_ratio:.2f}) -> "
            f"size({current_size.width()}x{current_size.height()}) -> "
            f"({new_width}x{new_height})"
        )
        
        
        return corrected_pixmap
    
    def _get_original_image_spacing(self, series_uid: Optional[str]) -> Optional[Tuple[float, float]]:
        if not series_uid or not hasattr(self, '_series_data') or series_uid not in self._series_data:
            return None
            
        try:
            image_data = self._series_data[series_uid]
            if hasattr(image_data, 'spacing'):
                original_spacing = (image_data.spacing.x, image_data.spacing.y)
                return original_spacing
        except Exception as e:
            self.logger.debug(f"Could not get original spacing for {series_uid}: {e}")
        
        return None
    
    def _calculate_optimal_window_level_by_modality(
        self, 
        image_data: np.ndarray,
        series_uid: str
    ) -> Tuple[float, float]:
        try:
            if image_data is None or image_data.size == 0:
                return self._default_window, self._default_level
            
            modality = self._detect_modality_from_series_uid(series_uid)
            
            filtered_data = self._filter_outliers_for_wl_calculation(image_data)
            
            min_val = float(np.min(filtered_data))
            max_val = float(np.max(filtered_data))
            mean_val = float(np.mean(filtered_data))
            std_val = float(np.std(filtered_data))
            
            p0_5 = float(np.percentile(filtered_data, 0.5))
            p99_5 = float(np.percentile(filtered_data, 99.5))
            p2 = float(np.percentile(filtered_data, 2))
            p98 = float(np.percentile(filtered_data, 98))
            p15 = float(np.percentile(filtered_data, 15))
            p85 = float(np.percentile(filtered_data, 85))
            p25 = float(np.percentile(filtered_data, 25))
            p75 = float(np.percentile(filtered_data, 75))
            
            if modality == 'T2W':
                optimal_window = max(400, p99_5 - p0_5)
                optimal_level = (p75 + p25) / 2
                
            elif modality == 'ADC':
                optimal_window = min(4000, max(2000, p98 - p2))
                optimal_level = max(p15, min(p85, mean_val))
                
            elif modality == 'HBV' or modality == 'DWI':
                optimal_window = max(500, p98 - p2)
                optimal_level = max(p15, min(p85, p75))
                
            else:
                optimal_window = max(300, p99_5 - p0_5)
                optimal_level = (p25 + p75) / 2
            
            optimal_window = max(10, min(optimal_window, max_val - min_val))
            optimal_level = max(min_val, min(optimal_level, max_val))
            
            level_min = optimal_level - optimal_window / 2
            level_max = optimal_level + optimal_window / 2
            
            if level_max > max_val:
                optimal_level = max_val - optimal_window / 2
            elif level_min < min_val:
                optimal_level = min_val + optimal_window / 2
            
            self.logger.info(
                f"Global W/L for {modality}: range({min_val:.1f}-{max_val:.1f}) "
                f"-> W/L({optimal_window:.1f}/{optimal_level:.1f}) "
                f"[covers {level_min:.1f} to {level_max:.1f}]"
            )
            
            return optimal_window, optimal_level
            
        except Exception as e:
            self.logger.error(f"Error calculating global W/L: {e}")
            return self._default_window, self._default_level
    
    def _filter_outliers_for_wl_calculation(self, image_data: np.ndarray) -> np.ndarray:
        try:
            p0_1 = np.percentile(image_data, 0.1)
            p99_9 = np.percentile(image_data, 99.9)
            
            filtered_data = image_data[(image_data >= p0_1) & (image_data <= p99_9)]
            
            if filtered_data.size < image_data.size * 0.8:
                self.logger.warning("Outlier filtering removed too much data, using original")
                return image_data
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error filtering outliers: {e}")
            return image_data
    
    def _detect_modality_from_series_uid(self, series_uid: str) -> str:
        if not series_uid:
            return 'UNKNOWN'
        
        series_upper = series_uid.upper()
        
        if 'T2W' in series_upper or 'T2-W' in series_upper:
            return 'T2W'
        elif 'ADC' in series_upper:
            return 'ADC'
        elif 'HBV' in series_upper or 'HIGH_B' in series_upper or 'HIGHB' in series_upper:
            return 'HBV'
        elif 'DWI' in series_upper or 'DIFFUSION' in series_upper:
            return 'DWI'
        else:
            return 'UNKNOWN'