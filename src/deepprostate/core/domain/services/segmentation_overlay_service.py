import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from PyQt6.QtGui import QColor, QPixmap, QImage, QPainter, QBrush
from PyQt6.QtCore import Qt

from deepprostate.core.domain.entities.segmentation import MedicalSegmentation, AnatomicalRegion


class SegmentationOverlayService:
    def __init__(self, view_manager=None):
        self.logger = logging.getLogger(__name__)
        
        self._view_manager = view_manager
        
        self._segmentation_overlays: Dict[str, np.ndarray] = {}
        self._segmentation_colors: Dict[str, QColor] = {}
        self._segmentation_visibility: Dict[str, bool] = {}
        self._global_opacity = 0.4
        
        self._current_slice_indices: Dict[str, int] = {
            'axial': 0,
            'sagittal': 0,
            'coronal': 0
        }
        self._current_orientation: str = 'axial' 
        self._mask_is_3d: Dict[str, bool] = {}
        
        self._default_colors = {
            'prostate': QColor(255, 0, 0, 150),      # Rojo semi-transparente
            'lesion': QColor(255, 255, 0, 150),      # Amarillo semi-transparente  
            'urethra': QColor(0, 255, 0, 150),       # Verof semi-transparente
            'capsule': QColor(0, 0, 255, 150),       # Azul semi-transparente
            'roi': QColor(255, 0, 255, 150),         # Magenta semi-transparente
            'measurement': QColor(255, 165, 0, 150), # Naranja semi-transparente
            'manual': QColor(128, 128, 128, 150)     # Gris semi-transparente
        }
        
        self.logger.info("SegmentationOverlayService initialized")

    def get_overlay_mask_data(self, overlay_id: str) -> Optional[np.ndarray]:
        if overlay_id in self._segmentation_overlays:
            return self._segmentation_overlays[overlay_id]
        else:
            self.logger.debug(f"Overlay {overlay_id} not found")
            return None
    
    def add_segmentation_overlay(
        self, 
        segmentation_id: str, 
        mask_data: np.ndarray, 
        color: Optional[QColor] = None,
        region_type: str = 'unknown'
    ) -> None:
        try:
            if mask_data is None or mask_data.size == 0:
                self.logger.error(f"Empty mask data for segmentation {segmentation_id}")
                return
            
            if mask_data.dtype != bool:
                mask_data = mask_data > 0
            
            if color is None:
                color = self._get_default_color_for_type(region_type)
            
            self._segmentation_overlays[segmentation_id] = mask_data.copy()
            self._segmentation_colors[segmentation_id] = color
            self._segmentation_visibility[segmentation_id] = True  
            
            self._mask_is_3d[segmentation_id] = len(mask_data.shape) > 2
            
            
        except Exception as e:
            self.logger.error(f"Error adding segmentation overlay {segmentation_id}: {e}")
    
    def remove_segmentation_overlay(self, segmentation_id: str) -> bool:
        try:
            if segmentation_id in self._segmentation_overlays:
                del self._segmentation_overlays[segmentation_id]
                del self._segmentation_colors[segmentation_id]
                del self._segmentation_visibility[segmentation_id]

                if segmentation_id in self._mask_is_3d:
                    del self._mask_is_3d[segmentation_id]
                
                return True
            else:
                self.logger.debug(f"Segmentation overlay not found: {segmentation_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing segmentation overlay {segmentation_id}: {e}")
            return False
    
    def clear_all_overlays(self) -> None:
        self._segmentation_overlays.clear()
        self._segmentation_colors.clear()
        self._segmentation_visibility.clear()
        self._mask_is_3d.clear()
        self.logger.info("All segmentation overlays cleared")
    
    def set_overlay_visibility(self, segmentation_id: str, visible: bool) -> None:
        if segmentation_id in self._segmentation_visibility:
            self._segmentation_visibility[segmentation_id] = visible
    
    def set_overlay_color(self, segmentation_id: str, color: QColor) -> None:
        if segmentation_id in self._segmentation_colors:
            self._segmentation_colors[segmentation_id] = color

    def get_segmentation_mask(self, segmentation_id: str) -> Optional['np.ndarray']:
        return self._segmentation_overlays.get(segmentation_id)

    def get_overlay_color(self, segmentation_id: str) -> Optional[QColor]:
        return self._segmentation_colors.get(segmentation_id)
    
    def set_global_opacity(self, opacity: float) -> None:
        self._global_opacity = max(0.0, min(1.0, opacity))
    
    def get_global_opacity(self) -> float:
        return self._global_opacity
    
    def set_current_slice_index(self, slice_index: int, orientation: str = 'axial') -> None:
        if orientation in self._current_slice_indices:
            if self._current_slice_indices[orientation] != slice_index:
                self._current_slice_indices[orientation] = slice_index
    
    def set_current_orientation(self, orientation: str) -> None:
        if orientation in self._current_slice_indices:
            self._current_orientation = orientation
    
    def get_current_slice_index(self, orientation: str = None) -> int:
        if orientation is None:
            orientation = self._current_orientation
        return self._current_slice_indices.get(orientation, 0)
    
    def get_slice_range(self, orientation: str) -> tuple[int, int]:
        default_ranges = {
            'axial': (0, 18),      
            'sagittal': (0, 83),  
            'coronal': (0, 127)
        }
        return default_ranges.get(orientation, (0, 0))
    
    def has_overlays(self) -> bool:
        return bool(self._segmentation_overlays)
    
    def compose_overlays_on_image(
        self, 
        base_pixmap: QPixmap, 
        orientation: str = None
    ) -> QPixmap:
        try:
            if not base_pixmap or base_pixmap.isNull():
                self.logger.error("Base pixmap is null or invalid")
                return base_pixmap
            
            result_pixmap = base_pixmap.copy()
            
            visible_overlays = [
                (seg_id, mask) for seg_id, mask in self._segmentation_overlays.items()
                if self._segmentation_visibility.get(seg_id, False)
            ]
            
            if not visible_overlays:
                return result_pixmap
            
            painter = QPainter(result_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            for segmentation_id, mask_data in visible_overlays:
                color = self._segmentation_colors.get(segmentation_id, QColor(255, 0, 0, 100))
                self._apply_single_overlay(painter, mask_data, color, result_pixmap.size(), orientation)
            
            painter.end()
            
            return result_pixmap
            
        except Exception as e:
            self.logger.error(f"Error composing overlays: {e}")
            return base_pixmap
    
    def _apply_single_overlay(
        self, 
        painter: QPainter, 
        mask_data: np.ndarray, 
        color: QColor, 
        canvas_size,
        orientation: str = None
    ) -> None:
        try:
            adjusted_color = QColor(color)
            adjusted_color.setAlphaF(adjusted_color.alphaF() * self._global_opacity)
            
            overlay_image = self._mask_to_qimage(mask_data, adjusted_color, orientation)
            if overlay_image.isNull():
                return
            
            if overlay_image.size().width() != canvas_size.width() or overlay_image.size().height() != canvas_size.height():
                overlay_image = overlay_image.scaled(
                    canvas_size,
                    Qt.AspectRatioMode.IgnoreAspectRatio,  
                    Qt.TransformationMode.SmoothTransformation
                )
            
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.drawImage(0, 0, overlay_image)
            
        except Exception as e:
            self.logger.error(f"Error applying single overlay: {e}")
    
    def _mask_to_qimage(self, mask_data: np.ndarray, color: QColor, orientation: str = None) -> QImage:
        try:
            if mask_data.size == 0:
                return QImage()
            
            if len(mask_data.shape) > 2:
                if orientation is None:
                    orientation = self._current_orientation
                
                mask_data = self._interpolate_mask_to_current_image_if_needed(mask_data)
                if orientation == 'axial':
                    slice_idx = self.get_current_slice_index('axial')
                    slice_idx = min(slice_idx, mask_data.shape[0] - 1)
                    slice_idx = max(0, slice_idx)
                    mask_data = mask_data[slice_idx, :, :]
                    
                elif orientation == 'sagittal':
                    slice_idx = self.get_current_slice_index('sagittal')
                    slice_idx = min(slice_idx, mask_data.shape[2] - 1)
                    slice_idx = max(0, slice_idx)
                    mask_data = mask_data[:, :, slice_idx]
                    
                elif orientation == 'coronal':
                    slice_idx = self.get_current_slice_index('coronal')
                    slice_idx = min(slice_idx, mask_data.shape[1] - 1)
                    slice_idx = max(0, slice_idx)
                    mask_data = mask_data[:, slice_idx, :]
                    
                else:
                    slice_idx = self.get_current_slice_index('axial')
                    slice_idx = min(slice_idx, mask_data.shape[0] - 1)
                    slice_idx = max(0, slice_idx)
                    mask_data = mask_data[slice_idx, :, :]
            
            if len(mask_data.shape) != 2:
                self.logger.error(f"Cannot convert mask to 2D: shape {mask_data.shape}")
                return QImage()
            
            height, width = mask_data.shape
            rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
            mask_indices = mask_data > 0
            rgba_data[mask_indices] = [color.red(), color.green(), color.blue(), color.alpha()]
            
            bytes_per_line = width * 4
            qimage = QImage(
                rgba_data.data.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGBA8888
            )
            
            return qimage
            
        except Exception as e:
            self.logger.error(f"Error converting mask to QImage: {e}")
            return QImage()
    
    def _interpolate_mask_to_current_image_if_needed(self, mask_data: np.ndarray) -> np.ndarray:
        if not self._view_manager:
            return mask_data
            
        current_image = self._view_manager.get_current_image()
        if not current_image or not hasattr(current_image, 'image_data'):
            return mask_data
            
        target_shape = current_image.image_data.shape
        if len(target_shape) != 3:
            return mask_data
        
        if mask_data.shape == target_shape:
            return mask_data
        
        try:
            from scipy import ndimage
            
            scale_factors = [
                target_shape[i] / mask_data.shape[i] 
                for i in range(3)
            ]
            
            interpolated_mask = ndimage.zoom(
                mask_data.astype(np.float32), 
                scale_factors, 
                order=0, 
                prefilter=False
            )

            interpolated_mask = (interpolated_mask > 0.5).astype(mask_data.dtype)
            
            return interpolated_mask
            
        except ImportError:
            self.logger.debug("scipy not available for mask interpolation, using original mask")
            return mask_data
        except Exception as e:
            self.logger.error(f"Error interpolating mask: {e}")
            return mask_data
    
    def _get_default_color_for_type(self, region_type: str) -> QColor:
        normalized_type = region_type.lower().strip()
        
        for key, color in self._default_colors.items():
            if key in normalized_type:
                return color
        
        return self._default_colors.get('manual', QColor(128, 128, 128, 150))
    
    def get_overlay_info(self, segmentation_id: str) -> Optional[Dict[str, Any]]:
        if segmentation_id not in self._segmentation_overlays:
            return None
        
        mask_data = self._segmentation_overlays[segmentation_id]
        color = self._segmentation_colors[segmentation_id]
        visible = self._segmentation_visibility[segmentation_id]
        
        return {
            'id': segmentation_id,
            'shape': mask_data.shape,
            'pixel_count': np.sum(mask_data),
            'color': {
                'red': color.red(),
                'green': color.green(),
                'blue': color.blue(),
                'alpha': color.alpha()
            },
            'visible': visible,
            'data_type': str(mask_data.dtype)
        }
    
    def get_all_overlay_ids(self) -> List[str]:
        return list(self._segmentation_overlays.keys())
    
    def get_visible_overlay_ids(self) -> List[str]:
        return [
            seg_id for seg_id, visible in self._segmentation_visibility.items()
            if visible
        ]
    
    def add_medical_segmentation(self, segmentation: MedicalSegmentation) -> None:
        try:
            seg_id = segmentation.segmentation_id
            
            region_type = 'unknown'
            if segmentation.anatomical_regions:
                region_type = segmentation.anatomical_regions[0].region_type.value
            
            mask_data = segmentation.get_mask_data()
            
            self.add_segmentation_overlay(
                segmentation_id=seg_id,
                mask_data=mask_data,
                region_type=region_type
            )
            
            self.logger.info(f"Added medical segmentation: {seg_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding medical segmentation: {e}")
    
    def get_overlay_statistics(self) -> Dict[str, Any]:
        total_overlays = len(self._segmentation_overlays)
        visible_overlays = len(self.get_visible_overlay_ids())
        
        total_pixels = sum(
            np.sum(mask) for mask in self._segmentation_overlays.values()
        )
        
        return {
            'total_overlays': total_overlays,
            'visible_overlays': visible_overlays,
            'hidden_overlays': total_overlays - visible_overlays,
            'total_segmented_pixels': int(total_pixels),
            'global_opacity': self._global_opacity,
            'available_colors': len(self._default_colors)
        }
    
    def add_overlay(self, overlay_data: np.ndarray, overlay_id: str, color: Tuple[int, int, int]) -> None:
        qcolor = QColor(color[0], color[1], color[2], 150)  # Semi-transparent
        
        self.add_segmentation_overlay(
            segmentation_id=overlay_id,
            mask_data=overlay_data,
            color=qcolor,
            region_type='mask'
        )
    
    def remove_overlay(self, overlay_id: str) -> bool:
        return self.remove_segmentation_overlay(overlay_id)
    
    def get_mask_data(self, overlay_id: str) -> Optional[np.ndarray]:
        if overlay_id not in self._segmentation_overlays:
            return None
        return self._segmentation_overlays[overlay_id]
    
    def update_mask_data(self, overlay_id: str, new_mask_data: np.ndarray) -> bool:
        if overlay_id not in self._segmentation_overlays:
            self.logger.error(f"Cannot update mask: overlay {overlay_id} not found")
            return False
        
        original_shape = self._segmentation_overlays[overlay_id].shape
        if new_mask_data.shape != original_shape:
            self.logger.error(f"Shape mismatch: expected {original_shape}, got {new_mask_data.shape}")
            return False
        
        self._segmentation_overlays[overlay_id] = new_mask_data.copy()
        return True
    
    def modify_mask_at_position(self, overlay_id: str, position: tuple, brush_size: int, 
                               operation: str = "add", slice_index: Optional[int] = None) -> bool:
        if overlay_id not in self._segmentation_overlays:
            self.logger.error(f"Cannot modify mask: overlay {overlay_id} not found")
            return False
        
        mask_data = self._segmentation_overlays[overlay_id]
        x, y = position
        
        if len(mask_data.shape) == 3: 
            if slice_index is None:
                slice_index = self._current_slice_indices.get(self._current_orientation, 0)
            
            if slice_index >= mask_data.shape[0]:
                self.logger.error(f"Slice index {slice_index} out of bounds for mask shape {mask_data.shape}")
                return False
            
            target_slice = mask_data[slice_index]
            height, width = target_slice.shape
        else: 
            target_slice = mask_data
            height, width = mask_data.shape
        
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        brush_radius = brush_size // 2
        for dy in range(-brush_radius, brush_radius + 1):
            for dx in range(-brush_radius, brush_radius + 1):
                if dx*dx + dy*dy <= brush_radius*brush_radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        if operation == "add":
                            target_slice[py, px] = 1
                        elif operation == "remove":
                            target_slice[py, px] = 0
        
        if len(mask_data.shape) == 3:
            mask_data[slice_index] = target_slice
        
        return True
    
    def flood_fill_mask(self, overlay_id: str, position: tuple, tolerance: int = 10, 
                       operation: str = "add", slice_index: Optional[int] = None) -> bool:
        if overlay_id not in self._segmentation_overlays:
            self.logger.error(f"Cannot flood fill mask: overlay {overlay_id} not found")
            return False
        
        mask_data = self._segmentation_overlays[overlay_id]
        x, y = position
        
        if len(mask_data.shape) == 3:
            if slice_index is None:
                slice_index = self._current_slice_indices.get(self._current_orientation, 0)
            
            if slice_index >= mask_data.shape[0]:
                self.logger.error(f"Slice index {slice_index} out of bounds for mask shape {mask_data.shape}")
                return False
            
            target_slice = mask_data[slice_index].copy()
            height, width = target_slice.shape
        else: 
            target_slice = mask_data.copy()
            height, width = mask_data.shape
        
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        original_value = target_slice[y, x]
        target_value = 1 if operation == "add" else 0
        
        if original_value == target_value:
            return True
        
        stack = [(x, y)]
        visited = set()
        
        while stack:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue
            
            if target_slice[cy, cx] != original_value:
                continue
            
            visited.add((cx, cy))
            target_slice[cy, cx] = target_value
            
            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        
        if len(mask_data.shape) == 3:
            mask_data[slice_index] = target_slice
        else:
            self._segmentation_overlays[overlay_id] = target_slice
        
        return True
    
    def save_segmentation_changes(self, segmentation_name: str) -> bool:
        try:
            target_overlay_id = None
            for overlay_id in self._segmentation_overlays.keys():
                if segmentation_name in overlay_id:
                    target_overlay_id = overlay_id
                    break
            
            if not target_overlay_id:
                self.logger.error(f"No overlay found for segmentation: {segmentation_name}")
                return False
            
            mask_data = self._segmentation_overlays.get(target_overlay_id)
            if mask_data is None:
                self.logger.error(f"No mask data found for overlay: {target_overlay_id}")
                return False
            
            return self.save_segmentation_to_file(target_overlay_id, None)
            
        except Exception as e:
            self.logger.error(f"Error saving segmentation changes: {e}")
            return False

    def save_segmentation_to_file(self, overlay_id: str, file_path: Optional[str] = None) -> bool:
        try:
            mask_data = self._segmentation_overlays.get(overlay_id)
            if mask_data is None:
                self.logger.error(f"No mask data found for overlay: {overlay_id}")
                return False

            if file_path is None:
                file_path = f"{overlay_id}.nii.gz"

            import nibabel as nib
            import numpy as np

            if mask_data.dtype == bool:
                save_data = mask_data.astype(np.uint8)
            else:
                save_data = mask_data.copy()

            affine = np.eye(4)
            nii_img = nib.Nifti1Image(save_data, affine)

            nib.save(nii_img, file_path)

            self.logger.info(f"Successfully saved segmentation {overlay_id} to {file_path}")
            return True

        except ImportError:
            self.logger.error("nibabel library not available. Cannot save NIfTI files.")
            self.logger.error("Install with: pip install nibabel")
            return False
        except Exception as e:
            self.logger.error(f"Error saving segmentation to file: {e}")
            return False
    
    def restore_segmentation_data(self, segmentation_name: str, original_data: Dict[str, Any]) -> bool:
        try:
            target_overlay_id = None
            for overlay_id in self._segmentation_overlays.keys():
                if segmentation_name in overlay_id:
                    target_overlay_id = overlay_id
                    break
            
            if not target_overlay_id:
                self.logger.error(f"No overlay found for segmentation: {segmentation_name}")
                return False
            
            if 'mask_data' in original_data:
                self._segmentation_overlays[target_overlay_id] = original_data['mask_data'].copy()
                return True
            else:
                self.logger.error(f"No mask_data found in original_data for {segmentation_name}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error restoring segmentation data: {e}")
            return False
    
    def apply_segmentation_state(self, segmentation_name: str, state_data: Dict[str, Any]) -> bool:
        try:
            target_overlay_id = None
            for overlay_id in self._segmentation_overlays.keys():
                if segmentation_name in overlay_id:
                    target_overlay_id = overlay_id
                    break
            
            if not target_overlay_id:
                self.logger.error(f"No overlay found for segmentation: {segmentation_name}")
                return False
            
            if 'mask_data' in state_data:
                self._segmentation_overlays[target_overlay_id] = state_data['mask_data'].copy()
                return True
            else:
                self.logger.error(f"No mask_data found in state_data for {segmentation_name}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error applying segmentation state: {e}")
            return False
    
    def get_segmentation_state(self, segmentation_name: str) -> Optional[Dict[str, Any]]:
        try:
            target_overlay_id = None
            for overlay_id in self._segmentation_overlays.keys():
                if segmentation_name in overlay_id:
                    target_overlay_id = overlay_id
                    break
            
            if not target_overlay_id:
                self.logger.debug(f"No overlay found for segmentation: {segmentation_name}")
                return None
            
            mask_data = self._segmentation_overlays.get(target_overlay_id)
            if mask_data is None:
                self.logger.error(f"No mask data found for overlay: {target_overlay_id}")
                return None
            
            return {
                'mask_data': mask_data.copy(),
                'overlay_id': target_overlay_id,
                'timestamp': self._get_current_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting segmentation state: {e}")
            return None
    
    def _get_current_timestamp(self) -> str:
        import datetime
        return datetime.datetime.now().isoformat()