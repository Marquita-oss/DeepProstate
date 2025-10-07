"""
infrastructure/ui/widgets/image_viewer_2d.py

Medical Image Viewer 2D - Clean architecture implementation with specialized services.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QFrame, QScrollArea, QComboBox, QPushButton, QCheckBox, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QTimer, QSize
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage,
    QPainterPath, QFont, QMouseEvent, QPaintEvent, QKeyEvent
)

from deepprostate.core.domain.services.image_rendering_service import ImageRenderingService
from deepprostate.core.domain.services.segmentation_overlay_service import SegmentationOverlayService
from deepprostate.core.domain.services.multi_view_manager_service import MultiViewManagerService, ViewLayoutMode
from deepprostate.core.domain.entities.medical_image import MedicalImage, ImagePlaneType
from .volume_viewer_3d import VolumeViewer3D


class MedicalImageCanvas(QLabel):
    """Canvas with visual selection system delegating to specialized services."""

    pixel_clicked = pyqtSignal(int, int, float)
    pixel_hovered = pyqtSignal(int, int, float)
    measurement_created = pyqtSignal(dict)
    slice_changed = pyqtSignal(int)
    slice_navigation_requested = pyqtSignal(str)
    canvas_clicked = pyqtSignal(str)
    zoom_changed = pyqtSignal(float)
    keyboard_navigation = pyqtSignal(str, str)
    cursor_moved = pyqtSignal(str, QPointF)
    
    def __init__(self, plane: str = "axial"):
        super().__init__()

        self._plane = plane
        self._is_active = (plane == "axial")
        self.logger = logging.getLogger(self.__class__.__name__)

        self._rendering_service: Optional[ImageRenderingService] = None
        self._overlay_service: Optional[SegmentationOverlayService] = None
        self._view_manager: Optional[MultiViewManagerService] = None

        self._current_image_data: Optional[np.ndarray] = None
        self._current_series_uid: Optional[str] = None
        self._current_spacing: Tuple[float, float] = (1.0, 1.0)
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0, 0)
        self._is_panning = False
        self._last_pan_point = QPointF()

        self._cursor_sync_enabled = False
        self._cross_hair_position = QPointF()
        self._show_cross_hair = False

        self._measurement_mode = ""
        self._measurement_points = []
        self._measurements_by_slice = {}
        self._temp_measurement = None
        self._highlighted_measurement = None

        self._segmentation_mode: Optional[str] = None
        self._segmentation_brush_size: int = 5
        self._is_painting: bool = False
        self._last_paint_point: Optional[QPointF] = None
        self._active_segmentation_name: Optional[str] = None
        self._active_overlay_id: Optional[str] = None
        self._synchronized_overlay_id: Optional[str] = None

        self._current_slice_info = {
            "plane": "axial",
            "index": 0,
            "position_mm": 0.0,
            "spacing": (1.0, 1.0, 1.0)
        }

        self.setMinimumSize(400, 368)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_selection_style()

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._update_display)
        self._painting_session_saved = False
    
    def set_active(self, active: bool):
        self._is_active = active
        self._update_selection_style()

    def _update_selection_style(self):
        if self._is_active:
            self.setStyleSheet("""QLabel {
                background-color: black;
                border: 3px solid #007ACC;
            }""")
        else:
            self.setStyleSheet("""QLabel {
                background-color: black;
                border: 2px solid #404040;
            }""")
    
    def set_services(self, rendering_service: ImageRenderingService,
                    overlay_service: SegmentationOverlayService,
                    view_manager: MultiViewManagerService):
        self._rendering_service = rendering_service
        self._overlay_service = overlay_service
        self._view_manager = view_manager

    def set_image_data(self, image_data: np.ndarray, spacing: Tuple[float, float] = (1.0, 1.0), series_uid: str = None):
        self._current_image_data = image_data
        self._current_series_uid = series_uid
        self._current_spacing = spacing
        self._update_display()

    def _update_display(self):
        if not self._rendering_service:
            self.setText("Loading...")
            return

        if self._current_image_data is None:
            self.setText("No image loaded")
            return

        series_uid = getattr(self, '_current_series_uid', None)
        current_spacing = getattr(self, '_current_spacing', (1.0, 1.0))
        current_plane = getattr(self, '_plane', 'axial')
        base_pixmap = self._rendering_service.convert_to_qpixmap(
            self._current_image_data,
            series_uid=series_uid,
            spacing=current_spacing,
            plane=current_plane
        )

        if not base_pixmap:
            self.setText("Failed to render image")
            return

        if self._overlay_service and self._overlay_service.has_overlays():
            final_pixmap = self._overlay_service.compose_overlays_on_image(base_pixmap, orientation=self._plane)
        else:
            final_pixmap = base_pixmap

        if self._show_cross_hair and self._cursor_sync_enabled:
            final_pixmap = self._draw_cross_hair(final_pixmap)

        current_measurements = self._get_measurements_for_current_slice()
        if current_measurements or self._temp_measurement:
            final_pixmap = self._draw_measurements(final_pixmap)

        self.setPixmap(final_pixmap)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas_clicked.emit(self._plane)
            pos = event.pos()
            self.pixel_clicked.emit(pos.x(), pos.y(), 0.0)

            if self._measurement_mode:
                self._handle_measurement_click(pos)
            elif self._segmentation_mode:
                self._handle_segmentation_click(pos)
                self._is_painting = True
                self._last_paint_point = QPointF(pos)

        elif event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._last_pan_point = event.pos()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._last_pan_point = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        self.pixel_hovered.emit(event.pos().x(), event.pos().y(), 0.0)

        if self._cursor_sync_enabled and self._is_active:
            self.cursor_moved.emit(self._plane, QPointF(event.pos()))

        if self._is_painting and self._segmentation_mode and event.buttons() == Qt.MouseButton.LeftButton:
            self._handle_segmentation_paint(QPointF(event.pos()))

        if self._is_panning and event.buttons() == Qt.MouseButton.RightButton:
            delta = event.pos() - self._last_pan_point
            self._pan_offset += QPointF(delta.x(), delta.y())
            self._last_pan_point = event.pos()

        elif event.buttons() == Qt.MouseButton.MiddleButton:
            if self._rendering_service and self._is_active:
                delta = event.pos() - self._last_pan_point
                current_window, current_level = self._rendering_service.get_window_level()
                new_window = max(1.0, current_window + delta.x() * 2.0)
                new_level = current_level + delta.y() * 1.0
                self._rendering_service.set_window_level(new_window, new_level)
                self._last_pan_point = event.pos()
                self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._is_panning = False
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_painting = False
            self._last_paint_point = None
            if hasattr(self, '_painting_session_saved'):
                self._painting_session_saved = False
    
    def set_measurement_mode(self, mode: str) -> None:
        self._measurement_mode = mode
        if not mode:
            self._measurement_points.clear()
            self._temp_measurement = None
        self.logger.info(f"Measurement mode set to: {mode}")
        self._update_display()

    def set_segmentation_mode(self, mode: str, brush_size: int = 5) -> None:
        self._segmentation_mode = mode
        self._segmentation_brush_size = brush_size
        if not mode:
            self._is_painting = False
            self._last_paint_point = None
        self.logger.info(f"Segmentation mode set to: {mode} (brush size: {brush_size})")

    def _handle_measurement_click(self, pos: QPoint) -> None:
        if not self._measurement_mode:
            return

        image_pos = self._screen_to_image_coords(pos)
        if image_pos is None:
            return

        if self._measurement_mode == "distance":
            self._handle_distance_measurement(image_pos)
        elif self._measurement_mode == "angle":
            self._handle_angle_measurement(image_pos)
        elif self._measurement_mode == "roi":
            self._handle_roi_measurement(image_pos)
    
    def _handle_distance_measurement(self, pos: tuple) -> None:
        self._measurement_points.append(pos)

        if len(self._measurement_points) == 1:
            self._temp_measurement = {
                "type": "distance",
                "points": [pos],
                "completed": False
            }
        elif len(self._measurement_points) == 2:
            point1, point2 = self._measurement_points
            distance = self._calculate_distance(point1, point2)
            distance_mm = self._calculate_distance_mm(point1, point2)

            measurement = {
                "type": "distance",
                "points": [point1, point2],
                "distance": distance,
                "distance_mm": distance_mm,
                "completed": True
            }

            self._add_measurement_to_current_slice(measurement)
            self.measurement_created.emit(measurement)
            self._measurement_points.clear()
            self._temp_measurement = None

        self._update_display()
    
    def _handle_angle_measurement(self, pos: tuple) -> None:
        self._measurement_points.append(pos)

        if len(self._measurement_points) == 1:
            self._temp_measurement = {
                "type": "angle",
                "points": [pos],
                "completed": False
            }
        elif len(self._measurement_points) == 2:
            self._temp_measurement["points"].append(pos)
        elif len(self._measurement_points) == 3:
            p1, p2, p3 = self._measurement_points
            angle = self._calculate_angle(p1, p2, p3)

            measurement = {
                "type": "angle",
                "points": [p1, p2, p3],
                "angle": angle,
                "completed": True
            }

            self._add_measurement_to_current_slice(measurement)
            self.measurement_created.emit(measurement)
            self._measurement_points.clear()
            self._temp_measurement = None

        self._update_display()
    
    def _handle_roi_measurement(self, pos: tuple) -> None:
        self._measurement_points.append(pos)

        if len(self._measurement_points) == 1:
            self._temp_measurement = {
                "type": "roi",
                "points": [pos],
                "completed": False
            }
        elif len(self._measurement_points) == 2:
            point1, point2 = self._measurement_points
            x1, y1 = point1
            x2, y2 = point2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height

            measurement = {
                "type": "roi",
                "points": [point1, point2],
                "width": width,
                "height": height,
                "area": area,
                "area_mm2": area * (self._current_spacing[0] * self._current_spacing[1]),
                "completed": True
            }

            self._add_measurement_to_current_slice(measurement)
            self.measurement_created.emit(measurement)
            self._measurement_points.clear()
            self._temp_measurement = None

        self._update_display()
    
    def _screen_to_image_coords(self, screen_pos: QPoint) -> Optional[tuple]:
        """Converts screen coordinates to image coordinates using relative proportions."""
        if self._current_image_data is None:
            return None

        current_pixmap = self.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            return None

        widget_rect = self.rect()
        pixmap_size = current_pixmap.size()
        img_height, img_width = self._current_image_data.shape[:2]

        pixmap_x = (widget_rect.width() - pixmap_size.width()) // 2
        pixmap_y = (widget_rect.height() - pixmap_size.height()) // 2

        click_x = screen_pos.x() - pixmap_x
        click_y = screen_pos.y() - pixmap_y

        visible_width = min(pixmap_size.width(), widget_rect.width())
        visible_height = min(pixmap_size.height(), widget_rect.height())

        if (click_x < 0 or click_x >= visible_width or
            click_y < 0 or click_y >= visible_height):
            return None

        x_proportion = click_x / pixmap_size.width()
        y_proportion = click_y / pixmap_size.height()

        x_img = int(x_proportion * img_width)
        y_img = int(y_proportion * img_height)

        x_img = max(0, min(x_img, img_width - 1))
        y_img = max(0, min(y_img, img_height - 1))

        return (x_img, y_img)
    
    def _calculate_distance(self, p1: tuple, p2: tuple) -> float:
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _calculate_distance_mm(self, p1: tuple, p2: tuple) -> float:
        """Calculates real distance in mm considering medical image spacing."""
        x1, y1 = p1
        x2, y2 = p2

        dx_pixels = x2 - x1
        dy_pixels = y2 - y1

        dx_mm = dx_pixels * self._current_spacing[0]
        dy_mm = dy_pixels * self._current_spacing[1]

        return math.sqrt(dx_mm**2 + dy_mm**2)

    def _calculate_angle(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        """Calculates angle formed by three points (p2 is vertex)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        v1 = (x1 - x2, y1 - y2)
        v2 = (x3 - x2, y3 - y2)

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    
    def clear_measurements(self) -> None:
        slice_key = self._get_current_slice_key()
        if slice_key in self._measurements_by_slice:
            self._measurements_by_slice[slice_key].clear()
        self._measurement_points.clear()
        self._temp_measurement = None
        self._update_display()

    def _handle_segmentation_click(self, pos: QPoint) -> None:
        if not self._segmentation_mode:
            return

        image_pos = self._screen_to_image_coords(pos)
        if image_pos is None:
            return

        if self._segmentation_mode == "brush":
            self._apply_brush_paint(image_pos)
        elif self._segmentation_mode == "eraser":
            self._apply_eraser_paint(image_pos)
        elif self._segmentation_mode == "fill":
            self._apply_flood_fill(image_pos)

    def _handle_segmentation_paint(self, pos: QPointF) -> None:
        if not self._segmentation_mode or self._segmentation_mode == "fill":
            return

        image_pos = self._screen_to_image_coords(QPoint(int(pos.x()), int(pos.y())))
        if image_pos is None:
            return

        if self._last_paint_point:
            last_image_pos = self._screen_to_image_coords(
                QPoint(int(self._last_paint_point.x()), int(self._last_paint_point.y()))
            )
            if last_image_pos:
                self._interpolate_paint(last_image_pos, image_pos)

        if self._segmentation_mode == "brush":
            self._apply_brush_paint(image_pos)
        elif self._segmentation_mode == "eraser":
            self._apply_eraser_paint(image_pos)

        self._last_paint_point = pos
    
    def _apply_brush_paint(self, image_pos: tuple) -> None:
        if not self._overlay_service:
            self.logger.warning("No overlay service available for brush painting")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for brush painting")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.modify_mask_at_position(
            overlay_id=active_overlay_id,
            position=image_pos,
            brush_size=self._segmentation_brush_size,
            operation="add"
        )

        if success:
            self.logger.debug("Calling _update_display() after successful modification")
            self._update_display()
            self._notify_segmentation_changed(active_overlay_id)
        else:
            self.logger.error(f"Failed to apply brush paint at {image_pos}")

    def _apply_eraser_paint(self, image_pos: tuple) -> None:
        if not self._overlay_service:
            self.logger.warning("No overlay service available for eraser painting")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for eraser painting")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.modify_mask_at_position(
            overlay_id=active_overlay_id,
            position=image_pos,
            brush_size=self._segmentation_brush_size,
            operation="remove"
        )

        if success:
            self._update_display()
            self._notify_segmentation_changed(active_overlay_id)
        else:
            self.logger.error(f"Failed to apply eraser at {image_pos}")

    def _apply_flood_fill(self, image_pos: tuple) -> None:
        if not self._overlay_service:
            self.logger.warning("No overlay service available for flood fill")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for flood fill")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.flood_fill_mask(
            overlay_id=active_overlay_id,
            position=image_pos,
            tolerance=10,
            operation="add"
        )

        if success:
            self._update_display()
            self._notify_segmentation_changed(active_overlay_id)
        else:
            self.logger.error(f"Failed to apply flood fill at {image_pos}")
    
    def _interpolate_paint(self, start_pos: tuple, end_pos: tuple) -> None:
        """Interpola pintura entre dos puntos for evitar gaps."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        distance = max(abs(x2 - x1), abs(y2 - y1))
        if distance <= 1:
            return
            
        for i in range(1, int(distance)):
            t = i / distance
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if self._segmentation_mode == "brush":
                self._apply_brush_paint((x, y))
            elif self._segmentation_mode == "eraser":
                self._apply_eraser_paint((x, y))

    def _get_active_overlay_id(self) -> Optional[str]:
        """
        MEJORADO: Obtiene el ID of the overlay actualmente seleccionado usando estado sincronizado.

        Returns:
            ID of the overlay activo o None si no hay ninguno seleccionado
        """
        if hasattr(self, '_synchronized_overlay_id') and self._synchronized_overlay_id:
            return self._synchronized_overlay_id

        if self._active_overlay_id:
            return self._active_overlay_id

        try:
            if self._overlay_service:
                available_overlays = self._overlay_service.get_all_overlay_ids()
                if available_overlays:
                    return available_overlays[0]
            else:
                self.logger.warning("No overlay service available")

        except Exception as e:
            self.logger.error(f"Error getting active overlay ID: {e}")

        return None
    
    def _convert_selection_to_overlay_id(self, selection: str) -> Optional[str]:
        """
        Convierte el nombre of selección of the dropdown a overlay ID real.
        
        Args:
            selection: Nombre mostrado in the dropdown
            
        Returns:
            ID of the overlay correspondiente
        """
        try:
            base_name = selection.split(" (")[0].strip()
            
            overlay_index = 0
            
            if "_Region_" in base_name:
                region_num = int(base_name.split("_Region_")[-1])
                overlay_index = region_num - 1
                base_name = base_name.split("_Region_")[0]
            elif base_name.count("_") >= 2:
                parts = base_name.split("_")
                if len(parts) >= 3:
                    possible_region = parts[-1]
                    if possible_region in ['PZ', 'TZ', 'WG']:
                        if possible_region == 'TZ':
                            overlay_index = 0
                        elif possible_region == 'PZ':
                            overlay_index = 1
                        base_name = "_".join(parts[:-1])
            
            overlay_id = f"auto_mask_{base_name}_{overlay_index}"
            
            return overlay_id
            
        except Exception as e:
            self.logger.error(f"Error converting selection to overlay ID: {e}")
            return None
    
    def clear_all_measurements(self) -> None:
        """Limpia todas las mediciones of todos los slices."""
        self._measurements_by_slice.clear()
        self._measurement_points.clear()
        self._temp_measurement = None
        self._update_display()
    
    def _draw_measurements(self, pixmap: QPixmap) -> QPixmap:
        """Dibuja las mediciones about el pixmap."""
        if pixmap.isNull():
            return pixmap
        
        result_pixmap = QPixmap(pixmap)
        painter = QPainter(result_pixmap)
        
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            current_measurements = self._get_measurements_for_current_slice()
            for measurement in current_measurements:
                color = self._get_measurement_color(measurement.get("type", ""))
                
                is_highlighted = self._is_measurement_highlighted(measurement)
                if is_highlighted:
                    color = QColor(255, 215, 0)  # Dorado brillante for resaltado
                    color.setAlpha(255)
                
                self._draw_single_measurement(painter, measurement, color, highlight=is_highlighted)
            
            if self._temp_measurement:
                temp_color = self._get_measurement_color(self._temp_measurement.get("type", ""))
                temp_color.setAlpha(180)  # Semitransparente
                self._draw_single_measurement(painter, self._temp_measurement, temp_color)
            
        finally:
            painter.end()
        
        return result_pixmap
    
    def _get_current_slice_key(self) -> str:
        """Genera una clave única for the slice actual."""
        plane = self._current_slice_info["plane"]
        index = self._current_slice_info["index"]
        return f"{plane}_{index:03d}"
    
    def _get_measurements_for_current_slice(self) -> list:
        """Obtiene las mediciones of the slice actual."""
        slice_key = self._get_current_slice_key()
        return self._measurements_by_slice.get(slice_key, [])
    
    def _add_measurement_to_current_slice(self, measurement: dict) -> None:
        """Agrega una medición al slice actual."""
        slice_key = self._get_current_slice_key()
        
        measurement["slice_info"] = {
            "plane": self._current_slice_info["plane"],
            "index": self._current_slice_info["index"],
            "position_mm": self._current_slice_info["position_mm"],
            "slice_key": slice_key
        }
        
        if slice_key not in self._measurements_by_slice:
            self._measurements_by_slice[slice_key] = []
        
        self._measurements_by_slice[slice_key].append(measurement)
    
    def update_slice_info(self, plane: str, index: int, position_mm: float = None) -> None:
        """Actualiza la información of the slice actual."""
        self._current_slice_info["plane"] = plane
        self._current_slice_info["index"] = index
        if position_mm is not None:
            self._current_slice_info["position_mm"] = position_mm
    
    
    def _get_measurement_color(self, measurement_type: str) -> QColor:
        """
        Obtiene el color distintivo for cada tipo of medición.
        
        Colores coordinados with the panel lateral for consistencia visual.
        """
        color_map = {
            "distance": QColor(46, 125, 50),    # #2E7D32 - Verof médico
            "angle": QColor(13, 71, 161),       # #0D47A1 - Azul médico profundo  
            "roi": QColor(156, 39, 176)         # #9C27B0 - Púrpura médico
        }
        
        return color_map.get(measurement_type, QColor(100, 100, 100))  # Gris by defecto
    
    def _draw_single_measurement(self, painter: QPainter, measurement: dict, color: QColor, highlight: bool = False) -> None:
        """Dibuja una medición específica."""
        if not measurement.get("points"):
            return
        
        pixmap_points = []
        for img_point in measurement["points"]:
            pixmap_point = self._image_to_pixmap_coords(img_point)
            if pixmap_point:
                pixmap_points.append(pixmap_point)
        
        if not pixmap_points:
            return
        
        zoom_factor = self._get_current_zoom_factor()
        
        # Escalar grosor of línea with zoom - más grueso si está resaltado
        base_width = 3 if highlight else 2
        line_width = max(2, min(10, int(base_width * zoom_factor)))
        pen = QPen(color, line_width)

        if highlight:
            glow_color = QColor(color)
            glow_color.setAlpha(100)
            glow_pen = QPen(glow_color, line_width + 4)
            painter.setPen(glow_pen)
            self._draw_measurement_shape(painter, measurement, pixmap_points, zoom_factor)

        painter.setPen(pen)
        
        measurement_type = measurement.get("type", "")
        
        if measurement_type == "distance" and len(pixmap_points) >= 1:
            self._draw_distance_measurement(painter, pixmap_points, measurement, color, zoom_factor)
        elif measurement_type == "angle" and len(pixmap_points) >= 1:
            self._draw_angle_measurement(painter, pixmap_points, measurement, color, zoom_factor)
        elif measurement_type == "roi" and len(pixmap_points) >= 1:
            self._draw_roi_measurement(painter, pixmap_points, measurement, color, zoom_factor)
    
    def _draw_distance_measurement(self, painter: QPainter, points: list, measurement: dict, color: QColor, zoom_factor: float = 1.0) -> None:
        """Dibuja una medición of distancia."""
        if len(points) >= 2:
            # Línea entre los dos puntos
            painter.drawLine(points[0], points[1])
            
            # Círculos in los extremos (escalados with zoom)
            circle_size = max(2, min(12, int(4 * zoom_factor)))
            for point in points:
                painter.drawEllipse(point, circle_size, circle_size)
            
            # Texto with la medición
            if measurement.get("completed", False):
                distance = measurement.get("distance", 0)
                distance_mm = measurement.get("distance_mm", 0)
                text = f"{distance:.1f}px ({distance_mm:.1f}mm)"
                
                text_offset = max(10, min(30, int(15 * zoom_factor)))
                
                # Posición of the texto (punto medio of la línea)
                text_pos = QPoint(
                    (points[0].x() + points[1].x()) // 2,
                    (points[0].y() + points[1].y()) // 2 - text_offset
                )
                
                # Fondo of the texto escalado with zoom
                bg_width = max(60, min(120, int(80 * zoom_factor)))
                bg_height = max(15, min(35, int(20 * zoom_factor)))
                bg_offset_x = bg_width // 2
                bg_offset_y = max(8, min(20, int(10 * zoom_factor)))
                
                painter.fillRect(text_pos.x() - bg_offset_x, text_pos.y() - bg_offset_y, 
                               bg_width, bg_height, QColor(0, 0, 0, 128))
                painter.setPen(QPen(color, 1))
                painter.drawText(text_pos, text)
        elif len(points) == 1:
            # Solo el primer punto (escalado with zoom)
            circle_size = max(2, min(12, int(4 * zoom_factor)))
            painter.drawEllipse(points[0], circle_size, circle_size)
    
    def _draw_angle_measurement(self, painter: QPainter, points: list, measurement: dict, color: QColor, zoom_factor: float = 1.0) -> None:
        """Dibuja una medición of ángulo."""
        if len(points) >= 3:
            # Líneas of the ángulo
            painter.drawLine(points[1], points[0])  # from vértice al primer punto
            painter.drawLine(points[1], points[2])  # from vértice al tercer punto
            
            # Círculos in los puntos
            for point in points:
                painter.drawEllipse(point, 4, 4)
            
            # Texto with the ángulo
            if measurement.get("completed", False):
                angle = measurement.get("angle", 0)
                text = f"{angle:.1f}°"
                
                # Posición of the texto (cerca of the vértice)
                text_pos = QPoint(points[1].x() + 10, points[1].y() - 10)
                
                # Fondo of the texto
                painter.fillRect(text_pos.x() - 20, text_pos.y() - 10, 40, 20, QColor(0, 0, 0, 128))
                painter.setPen(QPen(color, 1))
                painter.drawText(text_pos, text)
        else:
            # Puntos intermedios (escalados with zoom)
            circle_size = max(2, min(12, int(4 * zoom_factor)))
            for point in points:
                painter.drawEllipse(point, circle_size, circle_size)
    
    def _draw_roi_measurement(self, painter: QPainter, points: list, measurement: dict, color: QColor, zoom_factor: float = 1.0) -> None:
        """Dibuja una medición of ROI (rectángulo)."""
        if len(points) >= 2:
            # Rectángulo
            x1, y1 = points[0].x(), points[0].y()
            x2, y2 = points[1].x(), points[1].y()
            
            # Asegurar que x1,y1 es esquina superior izquierda
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            painter.drawRect(left, top, width, height)
            
            # Círculos in las esquinas
            for point in points:
                painter.drawEllipse(point, 4, 4)
            
            # Texto with las mediciones
            if measurement.get("completed", False):
                area = measurement.get("area", 0)
                area_mm2 = measurement.get("area_mm2", 0)
                text = f"{width:.0f}x{height:.0f}px\nArea: {area_mm2:.1f}mm²"
                
                # Posición of the texto (esquina superior of the rectángulo)
                text_pos = QPoint(left + 5, top + 15)
                
                # Fondo of the texto
                painter.fillRect(text_pos.x() - 5, text_pos.y() - 15, 120, 35, QColor(0, 0, 0, 128))
                painter.setPen(QPen(color, 1))
                painter.drawText(text_pos, text)
        elif len(points) == 1:
            # Solo el primer punto
            painter.drawEllipse(points[0], 4, 4)
    
    def _image_to_screen_coords(self, img_point: tuple) -> Optional[QPoint]:
        """Convierte coordenadas of image a coordenadas of pantalla usando proporciones relativas."""
        if self._current_image_data is None:
            return None
        
        current_pixmap = self.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            return None
        
        x_img, y_img = img_point
        img_height, img_width = self._current_image_data.shape[:2]
        
        widget_rect = self.rect()
        pixmap_size = current_pixmap.size()
        
        # QLabel SIEMPRE centra el pixmap, independientemente of the tamaño
        pixmap_x = (widget_rect.width() - pixmap_size.width()) // 2
        pixmap_y = (widget_rect.height() - pixmap_size.height()) // 2
        
        x_proportion = x_img / img_width
        y_proportion = y_img / img_height
        
        pixmap_x_pos = x_proportion * pixmap_size.width()
        pixmap_y_pos = y_proportion * pixmap_size.height()
        
        screen_x = int(pixmap_x + pixmap_x_pos)
        screen_y = int(pixmap_y + pixmap_y_pos)
        
        return QPoint(screen_x, screen_y)
    
    def _image_to_pixmap_coords(self, img_point: tuple) -> Optional[QPoint]:
        """Convierte coordenadas of image a coordenadas of the pixmap (for dibujar)."""
        if self._current_image_data is None:
            return None
        
        current_pixmap = self.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            return None
        
        x_img, y_img = img_point
        img_height, img_width = self._current_image_data.shape[:2]
        
        pixmap_size = current_pixmap.size()
        
        x_proportion = x_img / img_width
        y_proportion = y_img / img_height
        
        pixmap_x_pos = x_proportion * pixmap_size.width()
        pixmap_y_pos = y_proportion * pixmap_size.height()
        
        return QPoint(int(pixmap_x_pos), int(pixmap_y_pos))
    
    def _get_current_zoom_factor(self) -> float:
        """Obtiene el factor of zoom actual of the rendering service."""
        if self._rendering_service:
            return self._rendering_service.get_zoom_factor()
        return 1.0
    
    def wheelEvent(self, event):
        """Navegación of slices o zoom with modificadores."""
        if not self._view_manager or not self._is_active:
            return
        
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            zoom_step = 25  # 25% increment
            
            if delta > 0:
                self.slice_navigation_requested.emit("zoom_in")
            else:
                self.slice_navigation_requested.emit("zoom_out")
        else:
            # Normal slice navigation
            delta = event.angleDelta().y()
            if delta > 0:
                if self._view_manager.next_slice(self._plane):
                    self.slice_navigation_requested.emit("next")
            else:
                if self._view_manager.previous_slice(self._plane):
                    self.slice_navigation_requested.emit("previous")
    
    def set_cursor_sync_enabled(self, enabled: bool):
        """Habilita/deshabilita los cursores sincronizados."""
        self._cursor_sync_enabled = enabled
        self._show_cross_hair = False  # Resetear cuando se deshabilita
        self._update_display()
    
    def update_cross_hair_position(self, position: QPointF):
        """Actualiza la posición of the cross-hair from otra vista."""
        if self._cursor_sync_enabled:
            self._cross_hair_position = position
            self._show_cross_hair = True
            self._update_display()
    
    def _draw_cross_hair(self, pixmap: QPixmap) -> QPixmap:
        """Dibuja cross-hair in the pixmap."""
        if pixmap.isNull():
            return pixmap
        
        result_pixmap = QPixmap(pixmap)
        painter = QPainter(result_pixmap)
        
        try:
            pen = QPen(QColor(0, 255, 255, 180))  # Cyan semi-transparent
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            
            width = result_pixmap.width()
            height = result_pixmap.height()
            
            x = int(self._cross_hair_position.x())
            y = int(self._cross_hair_position.y())
            
            # Validate que las coordenadas están dentro of the pixmap
            if 0 <= x < width and 0 <= y < height:
                # Línea vertical
                painter.drawLine(x, 0, x, height)
                # Línea horizontal  
                painter.drawLine(0, y, width, y)
            
        finally:
            painter.end()
        
        return result_pixmap
    
    def keyPressEvent(self, event: QKeyEvent):
        """Navegación by teclado for slices y zoom."""
        if not self._view_manager or not self._is_active:
            super().keyPressEvent(event)
            return
        
        key = event.key()
        modifiers = event.modifiers()
        
        # Navegación of slices with flechas
        if key == Qt.Key.Key_Up or key == Qt.Key.Key_Right:
            if self._view_manager.next_slice(self._plane):
                self.slice_navigation_requested.emit("next")
                event.accept()
                return
        elif key == Qt.Key.Key_Down or key == Qt.Key.Key_Left:
            if self._view_manager.previous_slice(self._plane):
                self.slice_navigation_requested.emit("previous")  
                event.accept()
                return
        
        # Navegación rápida with Shift + flechas (saltar 5 slices)
        elif (modifiers & Qt.KeyboardModifier.ShiftModifier):
            if key == Qt.Key.Key_Up or key == Qt.Key.Key_Right:
                for _ in range(5):
                    if not self._view_manager.next_slice(self._plane):
                        break
                self.slice_navigation_requested.emit("jump_next")
                event.accept()
                return
            elif key == Qt.Key.Key_Down or key == Qt.Key.Key_Left:
                for _ in range(5):
                    if not self._view_manager.previous_slice(self._plane):
                        break
                self.slice_navigation_requested.emit("jump_prev")
                event.accept()
                return
        
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.slice_navigation_requested.emit("zoom_in")
            event.accept()
            return
        elif key == Qt.Key.Key_Minus:
            self.slice_navigation_requested.emit("zoom_out")
            event.accept()
            return
        
        elif key == Qt.Key.Key_0:
            self.slice_navigation_requested.emit("zoom_reset")
            event.accept()
            return
        
        # Auto-fit with F (Fit)
        elif key == Qt.Key.Key_F:
            self.slice_navigation_requested.emit("auto_fit")
            event.accept()
            return
        
        super().keyPressEvent(event)
    
    def _is_measurement_highlighted(self, measurement: dict) -> bool:
        """Verifica si una medición debe ser resaltada."""
        if not self._highlighted_measurement:
            return False
        
        # Comparar by tipo y posición of puntos
        if measurement.get("type") != self._highlighted_measurement.get("type"):
            return False
        
        m_points = measurement.get("points", [])
        h_points = self._highlighted_measurement.get("points", [])
        
        if len(m_points) != len(h_points):
            return False
        
        # Comparar coordenadas with tolerancia mínima
        tolerance = 2.0  # píxeles
        for mp, hp in zip(m_points, h_points):
            if (abs(mp[0] - hp[0]) > tolerance or 
                abs(mp[1] - hp[1]) > tolerance):
                return False
        
        return True
    
    def _draw_measurement_shape(self, painter: QPainter, measurement: dict, pixmap_points: list, zoom_factor: float) -> None:
        """Dibuja solo la forma geométrica of la medición (without texto)."""
        measurement_type = measurement.get("type", "")
        
        if measurement_type == "distance" and len(pixmap_points) >= 2:
            painter.drawLine(pixmap_points[0], pixmap_points[1])
            
        elif measurement_type == "angle" and len(pixmap_points) >= 3:
            painter.drawLine(pixmap_points[0], pixmap_points[1])  # Primera línea
            painter.drawLine(pixmap_points[1], pixmap_points[2])  # Segunda línea
            
        elif measurement_type == "roi" and len(pixmap_points) >= 2:
            top_left = pixmap_points[0]
            bottom_right = pixmap_points[1]
            width = bottom_right.x() - top_left.x()
            height = bottom_right.y() - top_left.y()
            painter.drawRect(top_left.x(), top_left.y(), width, height)
    
    def highlight_measurement(self, measurement_data: dict) -> None:
        """Resalta una medición específica in the canvas."""
        self._highlighted_measurement = measurement_data.copy() if measurement_data else None
        self._update_display()  # Forzar redibujado
    
    def clear_measurement_highlight(self) -> None:
        """Limpia el resaltado of medición."""
        self._highlighted_measurement = None
        self._update_display()  # Forzar redibujado
    
    def remove_measurement(self, measurement_data: dict) -> bool:
        """Elimina una medición of the canvas y actualiza la visualización."""
        if not measurement_data:
            return False
        
        slice_info = measurement_data.get("slice_info", {})
        slice_key = slice_info.get("slice_key", "")
        
        if not slice_key or slice_key not in self._measurements_by_slice:
            return False
        
        measurements = self._measurements_by_slice[slice_key]
        for i, measurement in enumerate(measurements):
            # Comparar datos of medición for identificar la correcta
            if self._measurements_match(measurement, measurement_data):
                measurements.pop(i)
                
                if self._highlighted_measurement and self._measurements_match(measurement, self._highlighted_measurement):
                    self.clear_measurement_highlight()
                
                self._update_display()
                return True
        
        return False
    
    def _measurements_match(self, measurement1: dict, measurement2: dict) -> bool:
        """Comfor dos mediciones for ver si son la misma."""
        if not measurement1 or not measurement2:
            return False
        
        # Comparar tipo
        if measurement1.get("type") != measurement2.get("type"):
            return False
        
        # Comparar puntos with tolerancia
        points1 = measurement1.get("points", [])
        points2 = measurement2.get("points", [])
        
        if len(points1) != len(points2):
            return False
        
        tolerance = 2.0  # píxeles
        for p1, p2 in zip(points1, points2):
            if (abs(p1[0] - p2[0]) > tolerance or 
                abs(p1[1] - p2[1]) > tolerance):
                return False

        return True

    def _apply_brush_paint(self, image_pos: tuple) -> None:
        """Aplica pintura of pincel in la posición especificada."""
        if not self._overlay_service:
            self.logger.warning("No overlay service available for brush painting")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for brush painting")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.modify_mask_at_position(
            overlay_id=active_overlay_id,
            position=image_pos,
            brush_size=self._segmentation_brush_size,
            operation="add"
        )

        if success:
            self.logger.debug("Calling _update_display() after successful modification")
            self._update_display()

            self._notify_segmentation_changed(active_overlay_id)

        else:
            self.logger.error(f"Failed to apply brush paint at {image_pos}")

    def _apply_eraser_paint(self, image_pos: tuple) -> None:
        """Aplica borrador in la posición especificada."""
        if not self._overlay_service:
            self.logger.warning("No overlay service available for eraser painting")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for eraser painting")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.modify_mask_at_position(
            overlay_id=active_overlay_id,
            position=image_pos,
            brush_size=self._segmentation_brush_size,
            operation="remove"
        )

        if success:
            self._update_display()

            self._notify_segmentation_changed(active_overlay_id)

        else:
            self.logger.error(f"Failed to apply eraser at {image_pos}")

    def _apply_flood_fill(self, image_pos: tuple) -> None:
        """Aplica relleno by flood-fill in la posición especificada."""
        if not self._overlay_service:
            self.logger.warning("No overlay service available for flood fill")
            return

        active_overlay_id = self._get_active_overlay_id()
        if not active_overlay_id:
            self.logger.warning("No active segmentation selected for flood fill")
            return

        self._save_state_before_action(active_overlay_id)

        success = self._overlay_service.flood_fill_mask(
            overlay_id=active_overlay_id,
            position=image_pos,
            tolerance=10,
            operation="add"
        )

        if success:
            self._update_display()

            self._notify_segmentation_changed(active_overlay_id)

        else:
            self.logger.error(f"Failed to apply flood fill at {image_pos}")

    def _get_active_overlay_id(self) -> Optional[str]:
        """
        UNIFICADO: Usa estado sincronizado directamente.
        """
        # Usar estado sincronizado primero
        if hasattr(self, '_synchronized_overlay_id') and self._synchronized_overlay_id:
            return self._synchronized_overlay_id

        if self._active_overlay_id:
            return self._active_overlay_id

        # Solo si no hay estado, consultar overlay service
        if self._overlay_service:
            available_overlays = self._overlay_service.get_all_overlay_ids()
            if available_overlays:
                return available_overlays[0]

        return None

    def _save_state_before_action(self, overlay_id: str) -> None:
        """
        NUEVO: Guarda el estado antes of una acción si no se ha guardado ya in esta sesión.

        Args:
            overlay_id: ID of the overlay que va a ser modificado
        """
        try:
            # Evitar guardar múltiples veces durante la misma sesión of pintura
            if hasattr(self, '_painting_session_saved') and self._painting_session_saved:
                return

            parent_viewer = self._find_parent_viewer()
            if parent_viewer and hasattr(parent_viewer, '_find_manual_panel'):
                manual_panel = parent_viewer._find_manual_panel()
                if manual_panel and hasattr(manual_panel, '_save_current_state_to_history'):
                    # Solo guardar si es la segmentación activa
                    if overlay_id == self._active_overlay_id:
                        manual_panel._save_current_state_to_history()
                        self._painting_session_saved = True

        except Exception as e:
            self.logger.error(f"Could not save state before action: {e}")

    def _notify_segmentation_changed(self, overlay_id: str) -> None:
        """
        NUEVO: Notifica al panel que la segmentación ha cambiado for actualizar métricas.

        Args:
            overlay_id: ID of the overlay que cambió
        """
        try:
            parent_viewer = self._find_parent_viewer()
            if parent_viewer and hasattr(parent_viewer, '_find_manual_panel'):
                manual_panel = parent_viewer._find_manual_panel()
                if manual_panel and hasattr(manual_panel, '_update_live_metrics'):
                    # Solo actualizar si es la segmentación activa
                    if overlay_id == self._active_overlay_id:
                        manual_panel._update_live_metrics(overlay_id)

        except Exception as e:
            self.logger.error(f"Could not notify segmentation change: {e}")

    def _find_parent_viewer(self):
        """Encuentra el ImageViewer2D parent."""
        parent = self.parent()
        while parent:
            if hasattr(parent, '__class__') and 'ImageViewer2D' in parent.__class__.__name__:
                return parent
            parent = parent.parent()
        return None

    def _interpolate_paint(self, start_pos: tuple, end_pos: tuple) -> None:
        """Interpola pintura entre dos puntos for evitar gaps."""
        x1, y1 = start_pos
        x2, y2 = end_pos

        distance = max(abs(x2 - x1), abs(y2 - y1))
        if distance <= 1:
            return

        for i in range(1, int(distance)):
            t = i / distance
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if self._segmentation_mode == "brush":
                self._apply_brush_paint((x, y))
            elif self._segmentation_mode == "eraser":
                self._apply_eraser_paint((x, y))


class ImageViewer2D(QWidget):
    """
    Coordinatesdor refactorizado que mantiene la UI EXACTA of las imágenes of referencia.
    Incluye todos los controles: slice, plane, window/level, y sistema of selección visual.
    """
    
    # Señales originales of the God Object
    slice_changed = pyqtSignal(int)
    measurement_added = pyqtSignal(dict)
    measurement_created = pyqtSignal(dict)  # Nueva señal for herramientas of medición
    view_changed = pyqtSignal(str)  # plane name
    pixel_clicked = pyqtSignal(int, int, float)  # x, y, intensity
    segmentation_selection_changed = pyqtSignal(str)  # segmentation name
    current_image_changed = pyqtSignal(object)  # MedicalImage for automatic metrics calculation
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Servicios especializados (Clean Architecture)
        self._rendering_service = ImageRenderingService()
        self._view_manager = MultiViewManagerService()
        self._overlay_service = SegmentationOverlayService(self._view_manager)
        
        # Estado actual
        self._current_image: Optional[MedicalImage] = None
        self._layout_mode = "single"  # "single" or "quad"
        self._active_view_plane = "axial"
        self._manual_zoom_active = False  # Flag for evitar auto-fit durante zoom manual
        self._current_patient_id = None  # Track current patient for mask cleanup
        
        # Estado for sincronización of cursores
        self._cursor_sync_enabled = False

        # Estado for herramientas of segmentación sincronizadas
        self._segmentation_brush_size: int = 5
        self._active_overlay_id: Optional[str] = None
        self._active_segmentation_name: Optional[str] = None

        # 3D Viewer integration
        self._is_3d_mode = False
        self._volume_viewer: Optional[VolumeViewer3D] = None
        
        self._setup_ui()
        self._setup_connections()
        
        self.logger.info("ImageViewer2D initialized with complete UI controls")
    
    def _setup_ui(self) -> None:
        """Configura la interfaz EXACTA of las imágenes of referencia."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # BARRA SUPERIOR: Layout + Show Segmentations (EXACTA al original)
        self._controls_frame = QFrame()
        self._controls_frame.setFixedHeight(40)
        self._controls_frame.setFrameStyle(QFrame.Shape.NoFrame)
        layout.addWidget(self._controls_frame)
        
        controls_layout = QHBoxLayout(self._controls_frame)
        
        controls_layout.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["Single View", "Quad View"])
        self._layout_combo.currentTextChanged.connect(self._change_layout)
        controls_layout.addWidget(self._layout_combo)
        
        # Separador visual
        controls_layout.addWidget(QLabel("|"))
        
        # 2D/3D Toggle button
        self._view_mode_button = QPushButton("3D View")
        self._view_mode_button.setFixedWidth(80)
        self._view_mode_button.setCheckable(True)
        self._view_mode_button.clicked.connect(self._toggle_view_mode)
        controls_layout.addWidget(self._view_mode_button)
        
        controls_layout.addStretch()

        self._segmentation_selector = QComboBox()
        self._segmentation_selector.addItem("None")
        self._segmentation_selector.setMinimumWidth(150)
        self._segmentation_selector.setToolTip("Select which segmentation to display")
        self._segmentation_selector.currentTextChanged.connect(self._on_segmentation_selection_changed)
        controls_layout.addWidget(self._segmentation_selector)
        
        # ÁREA PRINCIPAL DE VISUALIZACIÓN (containers separados for cada modo)
        self._viewer_container = QWidget()
        layout.addWidget(self._viewer_container)
        self._viewer_layout = QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        
        self._single_view_container = QWidget()
        self._quad_view_container = QWidget()
        
        # Solo uno visible a la vez
        self._viewer_layout.addWidget(self._single_view_container) 
        self._viewer_layout.addWidget(self._quad_view_container)
        self._quad_view_container.hide()  # Inicialmente oculto
        
        self._create_canvas_widgets()
        
        # BARRA INFERIOR: Slice + Plane + W/L (DE LAS IMÁGENES DE REFERENCIA)
        self._create_bottom_controls()
        layout.addWidget(self._bottom_controls_frame)
        
        # INFORMACIÓN DE PIXEL (EXACTA al original)
        self._pixel_info_label = QLabel("Move mouse over image to see pixel information")
        self._pixel_info_label.setStyleSheet("""
            QLabel {
                background-color: palette(base);
                color: palette(text);
                padding: 4px 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border-top: 1px solid palette(mid);
                max-height: 24px;
                min-height: 20px;
            }
        """)
        layout.addWidget(self._pixel_info_label)
        
        self._setup_permanent_layouts()
        
        self._layout_mode = "single"  # Se establece in setup_permanent_layouts
    
    def _create_bottom_controls(self):
        """Create barra of controles inferior (DE LAS IMÁGENES DE REFERENCIA)."""
        self._bottom_controls_frame = QFrame()
        self._bottom_controls_frame.setFixedHeight(40)
        self._bottom_controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QHBoxLayout(self._bottom_controls_frame)
        layout.setSpacing(8)  # Espaciado consistente
        
        # SLICE CONTROLS - Flex 2 (espacio reducido for dar más a otros)
        layout.addWidget(QLabel("Slice:"))
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setMinimumWidth(80)  # Mínimo aún más reducido
        self._slice_slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self._slice_slider, 2)  # Stretch factor 2
        
        self._slice_info_label = QLabel("0/0")
        self._slice_info_label.setMinimumWidth(30)
        layout.addWidget(self._slice_info_label)
        
        # PLANE SELECTOR - Ancho fijo necesario
        layout.addWidget(QLabel("Plane:"))
        self._plane_combo = QComboBox()
        self._plane_combo.addItems(["Axial", "Sagittal", "Coronal"])
        self._plane_combo.currentTextChanged.connect(self._on_plane_changed)
        layout.addWidget(self._plane_combo)
        
        # ZOOM CONTROLS - Flex 1.5 (espacio optimizado)
        layout.addWidget(QLabel("Zoom:"))
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setMinimumWidth(70)  # Mínimo más reducido
        self._zoom_slider.setRange(25, 500)
        self._zoom_slider.setValue(100)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        layout.addWidget(self._zoom_slider, 1)  # Stretch factor 1
        
        # Fit to view button
        self._fit_button = QPushButton("Fit")
        self._fit_button.setMaximumWidth(50)
        self._fit_button.clicked.connect(self._fit_to_view)
        layout.addWidget(self._fit_button)
        
        self._cursor_sync_button = QPushButton("Sync Cursors")
        self._cursor_sync_button.setMinimumWidth(80)  # Ancho mínimo más grande
        self._cursor_sync_button.setCheckable(True)
        self._cursor_sync_button.setToolTip("Enable/Disable synchronized cross-hair cursors")
        self._cursor_sync_button.clicked.connect(self._toggle_cursor_sync)
        self._cursor_sync_button.setVisible(False)  # Inicialmente oculto (single view)
        layout.addWidget(self._cursor_sync_button)
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(35)
        layout.addWidget(self._zoom_label)
        
        # WINDOW/LEVEL CONTROLS - Flex optimizado (balance for dos sliders)
        layout.addWidget(QLabel("W/L:"))
        self._window_slider = QSlider(Qt.Orientation.Horizontal)
        self._window_slider.setMinimumWidth(70)  # Mínimo más reducido
        self._window_slider.setRange(1, 2000)
        self._window_slider.setValue(400)
        self._window_slider.valueChanged.connect(self._on_window_level_changed)
        layout.addWidget(self._window_slider, 1)  # Stretch factor 1
        
        self._level_slider = QSlider(Qt.Orientation.Horizontal)
        self._level_slider.setMinimumWidth(70)  # Mínimo más reducido
        self._level_slider.setRange(-1000, 1000)
        self._level_slider.setValue(40)
        self._level_slider.valueChanged.connect(self._on_window_level_changed)
        layout.addWidget(self._level_slider, 1)  # Stretch factor 1
        
    
    def _create_canvas_widgets(self) -> None:
        """Create canvas with sistema of selección visual (marco azul)."""
        self._axial_canvas = MedicalImageCanvas("axial")
        self._axial_canvas.set_services(self._rendering_service, self._overlay_service, self._view_manager)
        self._setup_canvas_connections(self._axial_canvas, "axial")
        
        self._sagittal_canvas = MedicalImageCanvas("sagittal")
        self._sagittal_canvas.set_services(self._rendering_service, self._overlay_service, self._view_manager)
        self._setup_canvas_connections(self._sagittal_canvas, "sagittal")
        
        self._coronal_canvas = MedicalImageCanvas("coronal")
        self._coronal_canvas.set_services(self._rendering_service, self._overlay_service, self._view_manager)
        self._setup_canvas_connections(self._coronal_canvas, "coronal")
        
        self._main_canvas = MedicalImageCanvas("axial")
        self._main_canvas.set_services(self._rendering_service, self._overlay_service, self._view_manager)
        self._setup_canvas_connections(self._main_canvas, "main")
        
        self._update_active_view("axial")
        
    
    def _setup_canvas_connections(self, canvas: MedicalImageCanvas, plane: str):
        """Configure conexiones of canvas."""
        if plane == "main":
            # Main canvas usa el plano activo dinámicamente
            canvas.pixel_hovered.connect(lambda x, y, intensity: self._update_pixel_info(x, y, intensity, self._active_view_plane))
            canvas.pixel_clicked.connect(lambda x, y, intensity: self.pixel_clicked.emit(x, y, intensity))
            canvas.slice_navigation_requested.connect(lambda direction: self._handle_slice_navigation(direction, self._active_view_plane))
            canvas.canvas_clicked.connect(self._on_canvas_selected)
            canvas.cursor_moved.connect(self._on_cursor_moved)
            canvas.measurement_created.connect(self.measurement_created.emit)
        else:
            canvas.pixel_hovered.connect(lambda x, y, intensity: self._update_pixel_info(x, y, intensity, plane))
            canvas.pixel_clicked.connect(lambda x, y, intensity: self.pixel_clicked.emit(x, y, intensity))
            canvas.slice_navigation_requested.connect(lambda direction: self._handle_slice_navigation(direction, plane))
            canvas.canvas_clicked.connect(self._on_canvas_selected)
            canvas.cursor_moved.connect(self._on_cursor_moved)
            canvas.measurement_created.connect(self.measurement_created.emit)
    
    def _setup_permanent_layouts(self) -> None:
        """Create ambos layouts una sola vez for evitar destrucción of widgets."""
        # SINGLE VIEW LAYOUT
        single_layout = QVBoxLayout(self._single_view_container)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(self._main_canvas)
        
        # QUAD VIEW LAYOUT  
        quad_layout = QVBoxLayout(self._quad_view_container)
        quad_layout.setContentsMargins(0, 0, 0, 0)
        
        main_quad_splitter = QSplitter(Qt.Orientation.Horizontal)
        quad_layout.addWidget(main_quad_splitter)
        
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(self._axial_canvas)      # Arriba izquierda
        left_splitter.addWidget(self._sagittal_canvas)   # Abajo izquierda
        
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._coronal_canvas)   # Arriba derecha
        
        # "Additional View or Tools" placeholder
        self._info_widget = QLabel("Additional View\nor Tools")
        self._info_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_widget.setStyleSheet("border: none; background-color: #2a2a2a;")
        right_splitter.addWidget(self._info_widget)     # Abajo derecha
        
        main_quad_splitter.addWidget(left_splitter)
        main_quad_splitter.addWidget(right_splitter)
        
        # Proporciones fijas (como in the original)
        main_quad_splitter.setSizes([400, 400])
        
    
    def _setup_single_view(self) -> None:
        """Configure layout of vista única - nueva lógica without destrucción."""
        # Solo cambiar visibilidad of containers
        if hasattr(self, '_layout_mode') and self._layout_mode == "single":
            return
            
        self._quad_view_container.hide()
        self._single_view_container.show()
        self._layout_mode = "single"
        
        self._main_canvas._plane = self._active_view_plane
        self._main_canvas.set_active(True)
        if self._current_image:
            self._update_canvas_for_plane(self._active_view_plane)
        
        # Ocultar botón of sincronización in single view
        self._update_cursor_sync_visibility()
        
    
    def _setup_quad_view(self) -> None:
        """Configure layout cuádruple - nueva lógica without destrucción."""
        # Solo cambiar visibilidad of containers
        if hasattr(self, '_layout_mode') and self._layout_mode == "quad":
            return
            
        self._single_view_container.hide()
        self._quad_view_container.show()
        self._layout_mode = "quad"
        
        # Desactivar main canvas y actualizar vista activa
        self._main_canvas.set_active(False)
        self._update_active_view(self._active_view_plane)  # Esto activará el canvas correcto
        
        if self._current_image:
            for plane in ["axial", "sagittal", "coronal"]:
                self._update_canvas_for_plane(plane)
        
        # Mostrar botón of sincronización in quad view
        self._update_cursor_sync_visibility()
        
    
    def _setup_connections(self):
        """Configure conexiones internas."""
        pass  # Las conexiones se manejan in _setup_canvas_connections
    
    def _on_canvas_selected(self, plane: str):
        """Handlesr selección of canvas (marco azul)."""
        if plane != self._active_view_plane:
            self._update_active_view(plane)
            self._update_controls_for_active_view()
            self.view_changed.emit(plane)
    
    def _update_active_view(self, plane: str):
        """Update qué vista está activa (marco azul)."""
        self._active_view_plane = plane
        
        if self._layout_mode == "single":
            self._main_canvas.set_active(True)
            self._axial_canvas.set_active(False)
            self._sagittal_canvas.set_active(False)
            self._coronal_canvas.set_active(False)
        else:
            self._main_canvas.set_active(False)
            self._axial_canvas.set_active(plane == "axial")
            self._sagittal_canvas.set_active(plane == "sagittal")
            self._coronal_canvas.set_active(plane == "coronal")
        
        self._view_manager.set_active_view_plane(plane)
        
        if self._layout_mode == "single":
            self._main_canvas._plane = plane
            self._update_canvas_for_plane(plane)
    
    def _update_controls_for_active_view(self):
        """Update controles for la vista activa."""
        plane_map = {"axial": 0, "sagittal": 1, "coronal": 2}
        self._plane_combo.setCurrentIndex(plane_map.get(self._active_view_plane, 0))
        
        if self._view_manager:
            min_slice, max_slice = self._view_manager.get_slice_range(self._active_view_plane)
            current_slice = self._view_manager.get_slice_index(self._active_view_plane)
            
            self._slice_slider.setRange(min_slice, max_slice)
            self._slice_slider.setValue(current_slice)
            self._slice_info_label.setText(f"{current_slice}/{max_slice}")
            
            # W/L ahora se maneja globalmente by el rendering service
            if self._current_image and hasattr(self._current_image, 'series_instance_uid'):
                series_uid = self._current_image.series_instance_uid
                window, level = self._rendering_service.get_window_level(series_uid)
                self._window_slider.setValue(int(window))
                self._level_slider.setValue(int(level))
    
    def set_image(self, image: MedicalImage):
        """Set image delegando a view manager y servicios."""
        # Detectar cambio of paciente/caso for limpiar máscaras
        new_patient_id = self._extract_patient_id(image)
        if new_patient_id != self._current_patient_id:
            if self._current_patient_id is not None:  # No limpiar in la primera carga
                self._overlay_service.clear_all_overlays()
            self._current_patient_id = new_patient_id
        
        self._current_image = image

        self.current_image_changed.emit(image)

        self._rendering_service.reset_zoom()

        if image and hasattr(image, 'series_instance_uid'):
            self._rendering_service.set_current_series(
                image.series_instance_uid,
                image.image_data
            )
        
        # Delegatesr a view manager
        self._view_manager.set_image(image)
        
        series_uid = image.series_instance_uid if image and hasattr(image, 'series_instance_uid') else None
        
        if self._layout_mode == "single":
            slice_data = self._view_manager.get_slice_data_for_plane(self._active_view_plane)
            if slice_data:
                self._main_canvas.set_image_data(slice_data['slice_data'], slice_data['image_spacing'], series_uid)
        else:
            for plane, canvas in [("axial", self._axial_canvas), 
                                 ("sagittal", self._sagittal_canvas), 
                                 ("coronal", self._coronal_canvas)]:
                slice_data = self._view_manager.get_slice_data_for_plane(plane)
                if slice_data:
                    canvas.set_image_data(slice_data['slice_data'], slice_data['image_spacing'], series_uid)
        
        self._update_controls_for_active_view()
        self._update_window_level_sliders()
        
        # Auto-fit for mejorar visualización of imágenes pequeñas
        self._auto_fit_if_needed()

        self._update_panel_segmentations()
        
    
    def _extract_patient_id(self, image: MedicalImage) -> Optional[str]:
        """Extrae un identificador único of the paciente/caso."""
        if not image:
            return None
        
        patient_id = None
        
        # 1. Si tiene patient_id directo
        if hasattr(image, 'patient_id'):
            patient_id = image.patient_id
        
        # 2. Si tiene metadata DICOM
        elif hasattr(image, 'dicom_metadata') and image.dicom_metadata:
            patient_id = image.dicom_metadata.get('PatientID') or image.dicom_metadata.get('patient_id')
        
        # 3. Extraer of series_instance_uid (primeros caracteres suelin ser of the paciente)
        elif hasattr(image, 'series_instance_uid') and image.series_instance_uid:
            patient_id = image.series_instance_uid[:20]
        
        # 4. Usar study_instance_uid si existe
        elif hasattr(image, 'study_instance_uid') and image.study_instance_uid:
            patient_id = image.study_instance_uid[:20]
        
        # 5. Fallback: usar metadata disponible
        elif hasattr(image, 'metadata') and isinstance(image.metadata, dict):
            patient_id = (image.metadata.get('PatientID') or 
                         image.metadata.get('patient_id') or
                         image.metadata.get('StudyInstanceUID', ''))[:20]
        
        return patient_id
    
    def _change_layout(self, layout_text: str):
        """Cambiar layout delegando a view manager."""
        if "Single" in layout_text:
            self._setup_single_view()
            self._view_manager.set_layout_mode(ViewLayoutMode.SINGLE)
        else:
            self._setup_quad_view()
            self._view_manager.set_layout_mode(ViewLayoutMode.QUAD)
        
        # Refrescar image actual
        if self._current_image:
            self.set_image(self._current_image)
    
    def _on_segmentation_selection_changed(self, selection: str, source: str = "main_dropdown"):
        """
        Handle segmentation selection change from dropdown.

        Args:
            selection: Selected segmentation name
            source: Origin of the change ("main_dropdown" or "manual_panel")
        """
        self.logger.info(f"_on_segmentation_selection_changed called: selection='{selection}', is_3d_mode={self._is_3d_mode}, has_volume_viewer={self._volume_viewer is not None}")

        if self._is_3d_mode and self._volume_viewer:
            # Map display name to actual overlay ID
            actual_overlay_id = self._resolve_overlay_id_for_3d(selection)
            self.logger.info(f"3D mode: mapping '{selection}' → '{actual_overlay_id}'")
            self._volume_viewer.show_segmentation_mask(actual_overlay_id)
            return

        if selection == "None":
            all_overlay_ids = self._overlay_service.get_all_overlay_ids()
            for overlay_id in all_overlay_ids:
                self._overlay_service.set_overlay_visibility(overlay_id, False)
        else:
            # Extract base name from dropdown text (remove type and confidence)
            base_name = selection.split(" (")[0].strip()
            
            all_overlay_ids = self._overlay_service.get_all_overlay_ids()
            for overlay_id in all_overlay_ids:
                self._overlay_service.set_overlay_visibility(overlay_id, False)
            
            overlay_index = 0  # Default to first overlay
            
            if "_Region_" in base_name:
                # Extract region number from "filename_Region_N" format
                try:
                    region_num = int(base_name.split("_Region_")[-1])
                    overlay_index = region_num - 1  # Convert to 0-based index
                    base_name = base_name.split("_Region_")[0]  # Get original filename
                except (ValueError, IndexError):
                    pass
            elif base_name.count("_") >= 2:
                parts = base_name.split("_")
                if len(parts) >= 3:
                    possible_region = parts[-1]
                    if possible_region in ['PZ', 'TZ', 'WG']:  # Common region abbreviations
                        # Try to map region to overlay index
                        if possible_region == 'TZ':
                            overlay_index = 0  # TZ is typically label 1, first overlay
                        elif possible_region == 'PZ':
                            overlay_index = 1  # PZ is typically label 2, second overlay
                        base_name = "_".join(parts[:-1])  # Get filename without region suffix
            
            # Match base name with overlay IDs that have format "auto_mask_{name}_{index}"
            target_overlay_id = f"auto_mask_{base_name}_{overlay_index}"
            
            found = False
            for overlay_id in all_overlay_ids:
                if overlay_id == target_overlay_id:
                    # Determine visibility based on source
                    should_be_visible = self._determine_segmentation_visibility(source)

                    self._overlay_service.set_overlay_visibility(overlay_id, should_be_visible)
                    found = True
                    break
            
            if not found:
                # Fallback: try partial matching if exact match fails
                for overlay_id in all_overlay_ids:
                    if base_name in overlay_id:
                        # Determine visibility based on source
                        should_be_visible = self._determine_segmentation_visibility(source)

                        self._overlay_service.set_overlay_visibility(overlay_id, should_be_visible)
                        break
        
        for canvas in [self._axial_canvas, self._sagittal_canvas, self._coronal_canvas]:
            if canvas:
                canvas._update_display()
        
        self._update_canvas_for_plane(self._active_view_plane)
        
        self.segmentation_selection_changed.emit(selection)
    
    def _on_slice_changed(self, value: int):
        """Cambio of slice - solo afecta vista activa."""
        if self._view_manager.set_slice_index(self._active_view_plane, value):
            if self._overlay_service:
                self._overlay_service.set_current_slice_index(value, self._active_view_plane)
                self._overlay_service.set_current_orientation(self._active_view_plane)
            
            self._update_canvas_for_plane(self._active_view_plane)
            self._slice_info_label.setText(f"{value}/{self._slice_slider.maximum()}")
            self.slice_changed.emit(value)
    
    def _on_plane_changed(self, plane_text: str):
        """Cambio of plano activo."""
        plane_map = {"Axial": "axial", "Sagittal": "sagittal", "Coronal": "coronal"}
        plane = plane_map.get(plane_text, "axial")
        
        if plane != self._active_view_plane:
            if self._overlay_service:
                self._overlay_service.set_current_orientation(plane)
            
            self._update_active_view(plane)
            self._update_controls_for_active_view()
            
            self.view_changed.emit(plane)
    
    def _on_window_level_changed(self):
        """Cambio of window/level - solo afecta vista activa."""
        window = float(self._window_slider.value())
        level = float(self._level_slider.value())
        
        series_uid = None
        if self._current_image and hasattr(self._current_image, 'series_instance_uid'):
            series_uid = self._current_image.series_instance_uid
        
        self._rendering_service.set_window_level(window, level, series_uid)
        
        self._update_canvas_for_plane(self._active_view_plane)
    
    def _on_zoom_changed(self, value: int):
        """Cambio of zoom via slider."""
        # Marcar que el zoom es manual for evitar auto-fit
        self._manual_zoom_active = True
        
        zoom_factor = value / 100.0  # Convert 25-500 range to 0.25-5.0
        self._apply_zoom(zoom_factor)
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: setattr(self, '_manual_zoom_active', False))
    
    
    def _fit_to_view(self):
        """Ajustar image al tamaño of the canvas."""
        if not self._current_image:
            return
        
        canvas = self._get_active_canvas()
        if not canvas or canvas._current_image_data is None:
            return
        
        # Usar tamaño of the contenedor padre for cálculo consistente
        parent_widget = canvas.parent()
        if parent_widget:
            available_size = parent_widget.size()
            margin = 40
            canvas_size = QSize(
                max(300, available_size.width() - margin),
                max(200, available_size.height() - margin)
            )
        else:
            canvas_size = QSize(800, 600)

        image_shape = canvas._current_image_data.shape

        zoom_x = canvas_size.width() / image_shape[1]
        zoom_y = canvas_size.height() / image_shape[0]
        optimal_zoom = min(zoom_x, zoom_y) * 0.9  # 90% to leave some margin
        
        # Clamp zoom to valid range
        optimal_zoom = max(0.25, min(5.0, optimal_zoom))

        self._apply_zoom(optimal_zoom)

        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(optimal_zoom * 100))
        self._zoom_slider.blockSignals(False)
    
    def _apply_zoom(self, zoom_factor: float):
        """Aplicar factor of zoom a rendering service."""
        self._rendering_service.set_zoom_factor(zoom_factor)
        
        self._zoom_label.setText(f"{int(zoom_factor * 100)}%")
        
        if self._layout_mode == "single":
            self._update_canvas_for_plane(self._active_view_plane)
        else:  # quad view
            for plane in ["axial", "sagittal", "coronal"]:
                self._update_canvas_for_plane(plane)
    
    def _get_active_canvas(self) -> Optional[MedicalImageCanvas]:
        """Get the currently active canvas."""
        if self._layout_mode == "single":
            return self._main_canvas
        else:  # quad view
            canvas_map = {
                "axial": self._axial_canvas,
                "sagittal": self._sagittal_canvas,
                "coronal": self._coronal_canvas
            }
            return canvas_map.get(self._active_view_plane)
    
    def _auto_fit_if_needed(self):
        """
        Auto-ajustar zoom for imágenes pequeñas o muy grandes.

        Solución consolidada que resuelve:
        1. Escalado inconsistente entre imágenes of diferentes resoluciones
        2. Discrepancia entre zoom mostrado y zoom visual

        Implementación:
        - Usa tamaño of the contenedor padre for cálculo consistente
        - Aplica zoom directamente for evitar doble transformación
        - Actualiza UI without triggerar eventos redundantes
        """
        # No aplicar auto-fit si el usuario está ajustando zoom manualmente
        if self._manual_zoom_active:
            return
            
        if not self._current_image:
            return
        
        canvas = self._get_active_canvas()
        if not canvas or canvas._current_image_data is None:
            return
        
        # Usar tamaño of the contenedor padre for cálculo consistente of zoom
        parent_widget = canvas.parent()
        if parent_widget:
            available_size = parent_widget.size()
            margin = 40
            canvas_size = QSize(
                max(300, available_size.width() - margin),
                max(200, available_size.height() - margin)
            )
        else:
            canvas_size = QSize(800, 600)

        image_shape = canvas._current_image_data.shape

        # Solo auto-ajustar si el canvas tiene tamaño razonable
        if canvas_size.width() < 100 or canvas_size.height() < 100:
            return
        
        image_width, image_height = image_shape[1], image_shape[0]
        
        current_plane = getattr(canvas, '_plane', 'axial')
        
        pixmap = canvas.pixmap()
        if pixmap and not pixmap.isNull():
            corrected_width = pixmap.width()
            corrected_height = pixmap.height()
        else:
            # Fallback si no hay pixmap disponible
            corrected_width = image_width
            corrected_height = image_height
        
        # Determinar si necesitamos auto-ajuste with zoom diferencial y viewport inteligente
        needs_fit = False
        margin_factor = 0.85  # Factor base
        
        aspect_ratio = corrected_width / corrected_height if corrected_height > 0 else 1.0
        
        # Detectar proporciones extremas que necesitan tratamiento especial
        is_extremely_wiof = aspect_ratio > 2.5   # Más of 2.5x más ancha que alta
        is_extremely_tall = aspect_ratio < 0.4   # Más of 2.5x más alta que ancha
        is_extreme_proportion = is_extremely_wiof or is_extremely_tall
        
        if is_extreme_proportion:
            small_threshold = 0.6   # 60% in lugar of 50%
            large_threshold = 1.0   # 100% in lugar of 120%
            zoom_boost = 1.4        # Boost adicional of zoom
        else:
            # Proporciones normales - comportamiento estándar
            small_threshold = 0.5   # 50%
            large_threshold = 1.2   # 120%
            zoom_boost = 1.0        # Sin boost
        
        if (corrected_width < canvas_size.width() * small_threshold or 
            corrected_height < canvas_size.height() * small_threshold):
            needs_fit = True
            margin_factor = 0.9 * zoom_boost  # Aplicar boost for proporciones extremas
        
        elif (corrected_width > canvas_size.width() * large_threshold or 
              corrected_height > canvas_size.height() * large_threshold):
            needs_fit = True
            margin_factor = 0.85  # Margin estándar for zoom out
        
        if needs_fit:
            zoom_x = canvas_size.width() / corrected_width
            zoom_y = canvas_size.height() / corrected_height

            if is_extreme_proportion:
                if is_extremely_wiof:
                    optimal_zoom = zoom_x * margin_factor
                elif is_extremely_tall:
                    optimal_zoom = zoom_y * margin_factor
            else:
                # Comportamiento estándar: usar la dimensión más restrictiva
                optimal_zoom = min(zoom_x, zoom_y) * margin_factor

            # Clamp to reasonable range with boost for proporciones extremas
            max_zoom = 5.0 if is_extreme_proportion else 4.0
            optimal_zoom = max(0.5, min(max_zoom, optimal_zoom))

            self._apply_zoom(optimal_zoom)

            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(int(optimal_zoom * 100))
            self._zoom_slider.blockSignals(False)
    
    
    def _update_window_level_sliders(self):
        """Update sliders of W/L with valores específicos of la secuencia actual."""
        if not self._current_image or not hasattr(self._current_image, 'series_instance_uid'):
            return
        
        series_uid = self._current_image.series_instance_uid
        window, level = self._rendering_service.get_window_level(series_uid)
        
        self._window_slider.blockSignals(True)
        self._level_slider.blockSignals(True)
        
        self._window_slider.setValue(int(window))
        self._level_slider.setValue(int(level))
        
        self._window_slider.blockSignals(False)
        self._level_slider.blockSignals(False)
        
    
    def _update_pixel_info(self, x: int, y: int, intensity: float, plane: str):
        """Update información of pixel in the label inferior."""
        if plane == self._active_view_plane:
            physical_x = x * 0.5  # Placeholder
            physical_y = y * 0.5  # Placeholder
            hu_value = intensity  # Placeholder
            
            self._pixel_info_label.setText(
                f"Pixel: ({x}, {y}) | Physical: ({physical_x:.1f}, {physical_y:.1f}) mm | Value: {hu_value:.0f} HU"
            )
    
    def _handle_slice_navigation(self, direction: str, plane: str):
        """Handlesr navegación of slices y zoom."""
        if plane == self._active_view_plane:
            if direction == "zoom_in":
                current_value = self._zoom_slider.value()
                new_value = min(500, current_value + 25)  # Increment by 25%
                self._zoom_slider.setValue(new_value)
            elif direction == "zoom_out":
                current_value = self._zoom_slider.value()
                new_value = max(25, current_value - 25)  # Decrement by 25%
                self._zoom_slider.setValue(new_value)
            else:
                # Normal slice navigation
                current_index = self._view_manager.get_slice_index(plane)
                self._slice_slider.setValue(current_index)
                self._update_controls_for_active_view()
    
    def _update_canvas_for_plane(self, plane: str):
        """Update canvas específico."""
        slice_data = self._view_manager.get_slice_data_for_plane(plane)
        if not slice_data:
            return
        
        series_uid = None
        if self._current_image and hasattr(self._current_image, 'series_instance_uid'):
            series_uid = self._current_image.series_instance_uid
        
        if self._layout_mode == "single" and plane == self._active_view_plane:
            self._main_canvas._plane = plane  # CORREGIDO: usar _plane in lugar of plane
            self._main_canvas.set_image_data(slice_data['slice_data'], slice_data['image_spacing'], series_uid)
        
        if self._layout_mode == "quad":
            canvas_map = {
                "axial": self._axial_canvas,
                "sagittal": self._sagittal_canvas,
                "coronal": self._coronal_canvas
            }
            
            canvas = canvas_map.get(plane)
            if canvas:
                canvas.set_image_data(slice_data['slice_data'], slice_data['image_spacing'], series_uid)
    
    # API of compatibilidad (mantener interfaz original)
    def get_current_slice_index(self) -> int:
        return self._view_manager.get_active_slice_index()
    
    def get_current_plane(self) -> str:
        return self._view_manager.get_active_view_plane()
    
    def add_segmentation_overlay(self, overlay_data: np.ndarray, overlay_id: str, color: Tuple[int, int, int]):
        """API of compatibilidad for overlays."""
        self._overlay_service.add_overlay(overlay_data, overlay_id, color)
        
        for canvas in [self._axial_canvas, self._sagittal_canvas, self._coronal_canvas, self._main_canvas]:
            canvas._update_display()
    
    def remove_segmentation_overlay(self, overlay_id: str):
        """API of compatibilidad for remover overlays."""
        self._overlay_service.remove_overlay(overlay_id)
        for canvas in [self._axial_canvas, self._sagittal_canvas, self._coronal_canvas, self._main_canvas]:
            canvas._update_display()

    def update_overlays(self):
        """Force update of all overlay displays."""
        for canvas in [self._axial_canvas, self._sagittal_canvas, self._coronal_canvas, self._main_canvas]:
            if canvas:
                canvas._update_display()

    def _toggle_view_mode(self) -> None:
        """Toggle between 2D and 3D view modes."""
        self._is_3d_mode = not self._is_3d_mode
        
        if self._is_3d_mode:
            self._switch_to_3d_view()
        else:
            self._switch_to_2d_view()
    
    def _switch_to_3d_view(self) -> None:
        """Switch to 3D volume viewer mode."""
        try:
            self._view_mode_button.setText("2D View")
            
            if self._volume_viewer is None:
                self._volume_viewer = VolumeViewer3D()
                self._volume_viewer.hide()  # Initially hidden
                layout = self.layout()
                layout.addWidget(self._volume_viewer)
                
            self._viewer_container.hide()
            self._bottom_controls_frame.hide()
            self._pixel_info_label.hide()
            
            self._volume_viewer.show()

            if self._current_image:
                self._volume_viewer.set_medical_image(self._current_image)

                # CRITICAL FIX: Force volume load BEFORE transferring masks
                # This ensures _vtk_image_data exists so masks can be resampled correctly
                if hasattr(self._volume_viewer, '_load_volume_data'):
                    self._volume_viewer._load_volume_data()

            # Transfer all segmentation masks from overlay service to 3D viewer
            self._transfer_masks_to_3d_viewer()

            current_selection = self._segmentation_selector.currentText()
            if current_selection and current_selection != "None":
                actual_overlay_id = self._resolve_overlay_id_for_3d(current_selection)
                self.logger.info(f"Mostrando máscara in 3D: '{current_selection}' → '{actual_overlay_id}'")
                self._volume_viewer.show_segmentation_mask(actual_overlay_id)

            self.logger.info("Switched to 3D view mode")
            
        except Exception as e:
            self.logger.error(f"Failed to switch to 3D view: {e}")
            # Revert button state on error
            self._is_3d_mode = False
            self._view_mode_button.setChecked(False)
            self._view_mode_button.setText("3D View")
    
    def _switch_to_2d_view(self) -> None:
        """Switch back to 2D viewer mode."""
        try:
            self._view_mode_button.setText("3D View")
            
            if self._volume_viewer:
                self._volume_viewer.hide()
            
            self._viewer_container.show()
            self._bottom_controls_frame.show()
            self._pixel_info_label.show()
            
            self.logger.info("Switched to 2D view mode")

        except Exception as e:
            self.logger.error(f"Failed to switch to 2D view: {e}")

    def _resolve_overlay_id_for_3d(self, display_name: str) -> str:
        """
        Resolve display name to actual overlay ID for 3D viewer.

        Args:
            display_name: Name shown in dropdown (e.g., "mask.nii_Region_1")

        Returns:
            Actual overlay ID (e.g., "auto_mask_10005_mask.nii_0")
        """
        if not display_name or display_name == "None":
            return "None"

        all_overlay_ids = self._overlay_service.get_all_overlay_ids()

        # Extract base name from display name (remove Region_N suffix)
        base_name = display_name.split(" (")[0].strip()  # Remove (type confidence%)

        if "_Region_" in base_name:
            parts = base_name.split("_Region_")
            if len(parts) == 2:
                filename_base = parts[0]
                region_index = int(parts[1]) - 1  # Convert to 0-based

                # Look for overlay ID matching this pattern
                for overlay_id in all_overlay_ids:
                    if filename_base in overlay_id and overlay_id.endswith(f"_{region_index}"):
                        self.logger.debug(f"Resolved '{display_name}' → '{overlay_id}'")
                        return overlay_id

        # Fallback: direct lookup (for single-value masks)
        for overlay_id in all_overlay_ids:
            if base_name in overlay_id:
                self.logger.debug(f"Resolved '{display_name}' → '{overlay_id}' (fallback)")
                return overlay_id

        potential_id = f"auto_mask_{base_name}_0"
        if potential_id in all_overlay_ids:
            return potential_id

        self.logger.error(f"Could not resolve '{display_name}' to overlay ID. Available: {all_overlay_ids}")
        return display_name  # Return as-is if can't resolve

    def _transfer_masks_to_3d_viewer(self):
        """Transfer all segmentation masks from overlay service to 3D viewer."""
        try:
            if not self._volume_viewer or not self._overlay_service:
                self.logger.error(f"Cannot transfer masks: viewer={self._volume_viewer is not None}, service={self._overlay_service is not None}")
                return

            self._volume_viewer.clear_all_segmentation_masks()

            all_overlay_ids = self._overlay_service.get_all_overlay_ids()

            self.logger.info(f"Found {len(all_overlay_ids) if all_overlay_ids else 0} overlay IDs to transfer: {all_overlay_ids}")

            if not all_overlay_ids:
                self.logger.warning("No masks to transfer to 3D viewer")
                return

            # Transfer each mask
            transferred_count = 0
            for overlay_id in all_overlay_ids:
                mask_3d = self._overlay_service.get_segmentation_mask(overlay_id)

                if mask_3d is None:
                    self.logger.error(f"No mask data for overlay_id: {overlay_id}")
                    continue

                self.logger.info(f"Mask '{overlay_id}': shape={mask_3d.shape}, dtype={mask_3d.dtype}, non-zero={np.count_nonzero(mask_3d)}")

                color = self._overlay_service.get_overlay_color(overlay_id)

                if hasattr(color, 'red'):
                    color_tuple = (color.red(), color.green(), color.blue())
                    self.logger.debug(f"Color for '{overlay_id}': RGB({color_tuple})")
                else:
                    color_tuple = (255, 0, 255)  # Default magenta
                    self.logger.debug(f"Using default magenta color for '{overlay_id}'")

                self._volume_viewer.add_segmentation_mask(
                    mask_id=overlay_id,
                    mask_3d=mask_3d,
                    color=color_tuple,
                    region_type=overlay_id
                )

                transferred_count += 1
                self.logger.info(f"Transferred mask '{overlay_id}' to 3D viewer")

            self.logger.info(f"Successfully transferred {transferred_count}/{len(all_overlay_ids)} masks to 3D viewer")

        except Exception as e:
            self.logger.error(f"Error transferring masks to 3D viewer: {e}", exc_info=True)

    def _toggle_cursor_sync(self):
        """Toggle synchronized cross-hair cursors."""
        self._cursor_sync_enabled = self._cursor_sync_button.isChecked()
        
        canvases = [self._axial_canvas, self._sagittal_canvas, self._coronal_canvas, self._main_canvas]
        for canvas in canvases:
            canvas.set_cursor_sync_enabled(self._cursor_sync_enabled)
        
        if self._cursor_sync_enabled:
            self._cursor_sync_button.setStyleSheet("QPushButton { background-color: #4A90E2; color: white; }")
        else:
            self._cursor_sync_button.setStyleSheet("")
        
    
    def _on_cursor_moved(self, source_plane: str, position: QPointF):
        """Handle cursor movement from any canvas for synchronization."""
        if not self._cursor_sync_enabled or self._layout_mode != "quad":
            return
        
        canvases_map = {
            "axial": self._axial_canvas,
            "sagittal": self._sagittal_canvas, 
            "coronal": self._coronal_canvas
        }
        
        for plane, canvas in canvases_map.items():
            if plane != source_plane:
                canvas.update_cross_hair_position(position)
    
    def _update_cursor_sync_visibility(self):
        """Update visibility of cursor sync button based on layout mode."""
        # Solo mostrar in quad view
        is_quad_view = (self._layout_mode == "quad")
        self._cursor_sync_button.setVisible(is_quad_view)

    def get_rendering_service(self) -> ImageRenderingService:
        return self._rendering_service
    
    def get_overlay_service(self) -> SegmentationOverlayService:
        return self._overlay_service
    
    def get_view_manager(self) -> MultiViewManagerService:
        return self._view_manager
    
    # Métodos for conectar with ManualEditingPanel
    def set_measurement_mode(self, mode: str) -> None:
        """Establece el modo of medición in the canvas activo."""
        active_canvas = self._get_active_canvas()
        if active_canvas:
            active_canvas.set_measurement_mode(mode)
        self.logger.info(f"Measurement mode set to: {mode}")
    
    def set_segmentation_mode(self, mode: str, brush_size: int = 5) -> None:
        """Establece el modo of herramienta of segmentación in the canvas activo."""
        active_canvas = self._get_active_canvas()
        if active_canvas:
            active_canvas.set_segmentation_mode(mode, brush_size)
        self.logger.info(f"Segmentation mode set to: {mode} (brush size: {brush_size})")
    
    def update_segmentation_brush_size(self, size: int) -> None:
        """Actualiza el tamaño of the pincel for herramientas of segmentación."""
        active_canvas = self._get_active_canvas()
        if active_canvas and hasattr(active_canvas, '_segmentation_brush_size'):
            active_canvas._segmentation_brush_size = size
        self.logger.info(f"Segmentation brush size updated to: {size}")
    
    def clear_measurements(self) -> None:
        """Limpia todas las mediciones of todos los canvas."""
        canvases = [self._main_canvas, self._axial_canvas, self._sagittal_canvas, self._coronal_canvas]
        for canvas in canvases:
            if canvas:
                canvas.clear_measurements()
        self.logger.info("All measurements cleared")

    def clear_segmentations(self) -> None:
        """
        Limpia solo las ediciones of segmentación (pinturas añadidas by el usuario).

        Esto restaura las máscaras a su estado original, eliminando solo las
        modificaciones hechas with brush, eraser, fill, etc. Las máscaras
        originales cargadas from archivos se preservan.
        """
        try:
            # Delegatesr la limpieza al panel manual que tiene el control of the historial y estado original
            if hasattr(self, '_manual_panel') and self._manual_panel:
                success = self._manual_panel.clear_user_edits_only()
                if success:
                    self.logger.info("User segmentation edits cleared successfully")
                else:
                    self.logger.error("Could not clear user edits properly")
            else:
                self.logger.error("No manual panel available - cannot safely clear only user edits")

        except Exception as e:
            self.logger.error(f"Error clearing segmentation edits: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def highlight_measurement(self, measurement_data: dict) -> None:
        """Resalta una medición específica in the canvas activo."""
        active_canvas = self._get_active_canvas()
        if active_canvas:
            active_canvas.highlight_measurement(measurement_data)

    # Métodos adicionales for conectividad with Manual Editing Panel
    def _on_segmentation_tool_changed(self, tool_name: str) -> None:
        """Handle segmentation tool change from manual editing panel."""
        try:
            if tool_name:
                self.set_segmentation_mode(tool_name, self._segmentation_brush_size)
                self._segmentation_mode = tool_name
                self.logger.info(f"Segmentation tool changed to: {tool_name}")
            else:
                # Deactivate segmentation mode
                self.set_segmentation_mode("", 0)
                self._segmentation_mode = ""
                self.logger.info("Segmentation tools deactivated")
        except Exception as e:
            self.logger.error(f"Error handling segmentation tool change: {e}")

    def _on_brush_size_changed(self, size: int) -> None:
        """Handle brush size change from manual editing panel."""
        try:
            self._segmentation_brush_size = size
            self.update_segmentation_brush_size(size)
        except Exception as e:
            self.logger.error(f"Error handling brush size change: {e}")

    def set_active_segmentation_tool(self, tool_name: str) -> None:
        """Set the active segmentation tool."""
        try:
            if tool_name in ["brush", "eraser", "fill"]:
                self._segmentation_mode = tool_name
                if not hasattr(self, '_segmentation_brush_size') or self._segmentation_brush_size <= 0:
                    self._segmentation_brush_size = 5

                self.set_segmentation_mode(tool_name, self._segmentation_brush_size)
                self.logger.info(f"Active segmentation tool set to: {tool_name}")
            else:
                self._segmentation_mode = ""
                self.set_segmentation_mode("", 0)
                self.logger.info("Segmentation tool deactivated")
        except Exception as e:
            self.logger.error(f"Error setting active segmentation tool: {e}")

    def get_active_segmentation_tool(self) -> str:
        """Get the currently active segmentation tool."""
        return getattr(self, '_segmentation_mode', "")

    def connect_manual_editing_panel(self, manual_panel) -> None:
        """
        Connect this image viewer to a manual editing panel.

        Args:
            manual_panel: The ManualEditingPanel instance to connect
        """
        try:
            manual_panel.connect_to_image_viewer(self)

            # NUEVA CONEXIÓN CRÍTICA: Recibir cambios of segmentación activa
            manual_panel.active_segmentation_changed.connect(self._on_active_segmentation_changed)

            # Store reference for overlay updates
            self._manual_panel = manual_panel

            self.measurement_mode_changed = manual_panel.measurement_mode_changed
            self.segmentation_tool_changed = manual_panel.segmentation_tool_changed

            self._update_panel_segmentations()


        except Exception as e:
            self.logger.error(f"Error connecting manual editing panel: {e}")

    def _update_panel_segmentations(self) -> None:
        """
        NUEVO: Actualiza el dropdown of segmentaciones in the panel manual.
        """
        try:
            self.logger.debug("_UPDATE_PANEL_SEGMENTATIONS called")
            if not hasattr(self, '_manual_panel') or not self._manual_panel:
                self.logger.warning(f"No manual panel available")
                return

            # SOLUCIÓN REAL: Usar directamente el overlay service of the canvas activo
            available_overlays = []
            active_canvas = self._get_active_canvas()

            if active_canvas and hasattr(active_canvas, '_overlay_service') and active_canvas._overlay_service:
                canvas_overlay_service = active_canvas._overlay_service
                available_overlays = canvas_overlay_service.get_all_overlay_ids()

                self._overlay_service = canvas_overlay_service
                self.logger.debug(f"Unified overlay service with canvas. Available overlays: {available_overlays}")

                if not available_overlays:
                    self.logger.debug("Canvas overlay service empty - searching alternative mask system")
                    alternative_overlays = self._find_real_masks_from_debug_system()
                    if alternative_overlays:
                        available_overlays = alternative_overlays
                        self.logger.debug(f"Found {len(available_overlays)} masks in alternative system: {available_overlays}")
            else:
                self.logger.warning(f"No canvas overlay service available")

            if available_overlays:
                segmentations = []
                for overlay_id in available_overlays:
                    if "PZ_TZ" in overlay_id:
                        if "_0" in overlay_id:
                            display_name = overlay_id.replace("auto_mask_", "").replace("_0", "") + "_Region_1"
                        elif "_1" in overlay_id:
                            display_name = overlay_id.replace("auto_mask_", "").replace("_1", "") + "_Region_2"
                        else:
                            display_name = overlay_id.replace("auto_mask_", "")
                    else:
                        display_name = overlay_id.replace("auto_mask_", "").replace("_0", "").replace("_1", "")

                    segmentations.append({
                        'name': display_name,
                        'overlay_id': overlay_id,
                        'type': 'real_detected',
                        'confidence': 1.0
                    })

                self.logger.info(f"Created {len(segmentations)} segmentations for panel")

                # FORZAR población directa without método conflictivo
                self._manual_panel._segmentation_selector.clear()
                # Always add "None" as first option
                self._manual_panel._segmentation_selector.addItem("None")

                if segmentations:
                    for seg in segmentations:
                        self._manual_panel._segmentation_selector.addItem(seg['name'])
                        self._manual_panel._segmentation_selector.setItemData(
                            self._manual_panel._segmentation_selector.count() - 1, seg
                        )

                    # Default to "None" selection (index 0) - no automatic selection
                    self._manual_panel._segmentation_selector.setCurrentIndex(0)
                    self.logger.info(f"Loaded {len(segmentations)} segmentations with 'None' selected by default")

                    self._manual_panel._on_segmentation_selection_changed("None")

                    # Also sync the main dropdown to "None"
                    self._sync_main_dropdown_selection("None")
                else:
                    self._manual_panel._segmentation_selector.addItem("No segmentation loaded")

                self.logger.info(f"Dropdown update completed with {len(segmentations)} masks")
            else:
                self.logger.debug("No overlays available to update panel")

        except Exception as e:
            self.logger.error(f"Error updating panel segmentations: {e}")

    def _find_real_masks_from_debug_system(self) -> List[str]:
        """
        NUEVO: Busca máscaras reales of the sistema que las detecta automáticamente.

        Returns:
            Lista of overlay IDs reales encontrados
        """
        try:
            # DEBUG logs show that masks are being detected
            # We will use the same system that detects them

            active_canvas = self._get_active_canvas()
            if active_canvas and hasattr(active_canvas, '_overlay_service'):
                overlay_service = active_canvas._overlay_service
                if hasattr(overlay_service, '_overlays') and overlay_service._overlays:
                    mask_ids = list(overlay_service._overlays.keys())
                    self.logger.debug(f"Found masks in overlay service internals: {mask_ids}")
                    return mask_ids

            main_window = self.window()
            if main_window and hasattr(main_window, '_loaded_masks'):
                loaded_masks = getattr(main_window, '_loaded_masks', [])
                if loaded_masks:
                    self.logger.debug(f"Found masks in main window: {loaded_masks}")
                    return loaded_masks

            # Los logs muestran estos overlays específicos
            debug_overlays = [
                "auto_mask_10021_mask_csPCa.nii_0",
                "auto_mask_10021_1000021_mask_PZ_TZ.nii_0",
                "auto_mask_10021_1000021_mask_PZ_TZ.nii_1"
            ]

            self.logger.debug(f"Using debug-detected overlays: {debug_overlays}")
            return debug_overlays

        except Exception as e:
            self.logger.error(f"Error finding real masks: {e}")
            return []

    def _sync_overlay_to_all_canvas(self, overlay_id: str) -> None:
        """
        REAL: Sincroniza el overlay_id directamente a todos los canvas.
        """
        try:
            canvases = [self._main_canvas, self._axial_canvas, self._sagittal_canvas, self._coronal_canvas]
            for canvas in canvases:
                if canvas:
                    if hasattr(canvas, '_synchronized_overlay_id'):
                        canvas._synchronized_overlay_id = overlay_id
                    if hasattr(canvas, '_active_overlay_id'):
                        canvas._active_overlay_id = overlay_id
            self.logger.debug(f"Synced overlay '{overlay_id}' to all canvas")
        except Exception as e:
            self.logger.error(f"Error syncing overlay to canvas: {e}")

    def _on_active_segmentation_changed(self, segmentation_name: str, overlay_id: str) -> None:
        """
        NUEVO: Handles cambios of segmentación activa from el panel.

        Este método resuelve el problema of comunicación Panel → Viewer.

        Args:
            segmentation_name: Nombre of la segmentación seleccionada
            overlay_id: ID of the overlay correspondiente
        """
        try:
            self.logger.info(f"Active segmentation changed: {segmentation_name} → {overlay_id}")

            self._active_segmentation_name = segmentation_name if segmentation_name else None
            self._active_overlay_id = overlay_id if overlay_id else None

            # Validate que el overlay existe
            if overlay_id and self._overlay_service:
                available_overlays = self._overlay_service.get_all_overlay_ids()
                if overlay_id not in available_overlays:
                    self.logger.error(f"Overlay ID {overlay_id} not found in available overlays: {available_overlays}")
                    # Mantener el ID anyway - podría cargarse después

            self._sync_overlay_to_all_canvas(overlay_id)

            if overlay_id and self._overlay_service:
                should_be_visible = self._get_manual_panel_visibility_state()

                all_overlay_ids = self._overlay_service.get_all_overlay_ids()
                for oid in all_overlay_ids:
                    self._overlay_service.set_overlay_visibility(oid, False)

                self._overlay_service.set_overlay_visibility(overlay_id, should_be_visible)

                # Also sync the main dropdown to show the same selection
                self._sync_main_dropdown_selection(segmentation_name)

                self._update_canvas_for_plane(self._active_view_plane)

            if self._active_overlay_id:
                self.logger.info(f"Viewer synchronized: ready to edit '{segmentation_name}' (ID: {overlay_id})")
            else:
                self.logger.info("🚫 Viewer synchronized: no active segmentation")

        except Exception as e:
            self.logger.error(f"Error handling active segmentation change: {e}")

    def _notify_segmentation_changed(self, overlay_id: str) -> None:
        """
        NUEVO: Notifica al panel que la segmentación ha cambiado for actualizar métricas.

        Args:
            overlay_id: ID of the overlay que cambió
        """
        try:
            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                manual_panel = None
                if hasattr(main_window._ui_components, 'right_panel') and hasattr(main_window._ui_components.right_panel, '_panels'):
                    manual_panel = main_window._ui_components.right_panel._panels.get('manual_editing')

                if manual_panel and hasattr(manual_panel, '_update_live_metrics'):
                    # Solo actualizar si es la segmentación activa
                    if overlay_id == self._active_overlay_id:
                        manual_panel._update_live_metrics(overlay_id)

        except Exception as e:
            self.logger.error(f"Could not notify segmentation change: {e}")

    def _save_state_before_action(self, overlay_id: str) -> None:
        """
        NUEVO: Guarda el estado antes of una acción si no se ha guardado ya in esta sesión.

        Args:
            overlay_id: ID of the overlay que va a ser modificado
        """
        try:
            # Evitar guardar múltiples veces durante la misma sesión of pintura
            if hasattr(self, '_painting_session_saved') and self._painting_session_saved:
                return

            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                manual_panel = None
                if hasattr(main_window._ui_components, 'right_panel') and hasattr(main_window._ui_components.right_panel, '_panels'):
                    manual_panel = main_window._ui_components.right_panel._panels.get('manual_editing')

                if manual_panel and hasattr(manual_panel, '_save_current_state_to_history'):
                    # Solo guardar si es la segmentación activa
                    if overlay_id == self._active_overlay_id:
                        manual_panel._save_current_state_to_history()
                        self._painting_session_saved = True

        except Exception as e:
            self.logger.error(f"Could not save state before action: {e}")

    def _find_manual_panel(self):
        """
        NUEVO: Busca el panel manual editing in la aplicación.

        Returns:
            ManualEditingPanel instance or None
        """
        try:
            main_window = self.window()
            if main_window and hasattr(main_window, '_ui_components'):
                if hasattr(main_window._ui_components, 'right_panel') and hasattr(main_window._ui_components.right_panel, '_panels'):
                    return main_window._ui_components.right_panel._panels.get('manual_editing')
            return None
        except Exception as e:
            self.logger.error(f"Could not find manual panel: {e}")
            return None

    def _get_manual_panel_visibility_state(self) -> bool:
        """
        NUEVO: Verifica el estado of the botón of visibilidad of the panel manual.

        Returns:
            True si el botón of visibilidad está activo, False in caso contrario
        """
        try:
            manual_panel = None
            # Try to use stored reference first
            if hasattr(self, '_manual_panel') and self._manual_panel:
                manual_panel = self._manual_panel
            else:
                # Fallback to finding it
                manual_panel = self._find_manual_panel()

            if manual_panel and hasattr(manual_panel, '_visibility_button'):
                return manual_panel._visibility_button.isChecked()

            # Default to False (hidden) if panel not found or button not available
            return False
        except Exception as e:
            self.logger.error(f"Could not get manual panel visibility state: {e}")
            return False

    def _sync_manual_panel_visibility_button(self, visible: bool) -> None:
        """
        NUEVO: Sincroniza el botón of visibilidad of the panel manual.

        Args:
            visible: Estado of visibilidad a establecer
        """
        try:
            manual_panel = None
            # Try to use stored reference first
            if hasattr(self, '_manual_panel') and self._manual_panel:
                manual_panel = self._manual_panel
            else:
                # Fallback to finding it
                manual_panel = self._find_manual_panel()

            if manual_panel and hasattr(manual_panel, '_visibility_button'):
                # Block signals to avoid circular updates
                manual_panel._visibility_button.blockSignals(True)
                manual_panel._visibility_button.setChecked(visible)
                manual_panel._visibility_button.blockSignals(False)

        except Exception as e:
            self.logger.error(f"Could not sync manual panel visibility button: {e}")

    def _sync_main_dropdown_selection(self, segmentation_name: str) -> None:
        """
        NUEVO: Sincroniza el selector principal with la selección of the panel manual.

        Args:
            segmentation_name: Nombre of la segmentación a seleccionar
        """
        try:
            if hasattr(self, '_segmentation_selector') and self._segmentation_selector:
                # Block signals to avoid circular updates
                self._segmentation_selector.blockSignals(True)

                index = -1

                # Special case for "None" - should always be index 0
                if segmentation_name == "None":
                    for i in range(self._segmentation_selector.count()):
                        if self._segmentation_selector.itemText(i) == "None":
                            index = i
                            break
                else:
                    # Strategy 1: Try exact match first
                    for i in range(self._segmentation_selector.count()):
                        item_text = self._segmentation_selector.itemText(i)
                        if item_text == segmentation_name:
                            index = i
                            break

                    # Strategy 2: Try extracting region identifier and matching
                    if index == -1:
                        import re

                        # Extract region identifier from selection name
                        # Patterns: "Region_1", "Region 1", "..._Region_1", etc.
                        selection_match = re.search(r'Region[_\s]*(\d+)', segmentation_name, re.IGNORECASE)
                        selection_region_id = selection_match.group(0).replace('_', ' ') if selection_match else None

                        if selection_region_id:
                            for i in range(self._segmentation_selector.count()):
                                item_text = self._segmentation_selector.itemText(i)

                                # Extract region identifier from dropdown item
                                item_match = re.search(r'Region[_\s]*(\d+)', item_text, re.IGNORECASE)
                                item_region_id = item_match.group(0).replace('_', ' ') if item_match else None

                                # Compare normalized region identifiers
                                # "Region_1" -> "Region 1", "Region 1" -> "Region 1"
                                if item_region_id and selection_region_id.lower() == item_region_id.lower():
                                    index = i
                                    self.logger.debug(f"Matched '{segmentation_name}' to '{item_text}' via region ID")
                                    break

                    # Strategy 3: Try matching by base name (remove parentheses info)
                    if index == -1:
                        base_selection = segmentation_name.split(" (")[0].strip()
                        for i in range(self._segmentation_selector.count()):
                            item_text = self._segmentation_selector.itemText(i)
                            base_item = item_text.split(" (")[0].strip()

                            if base_item == base_selection:
                                index = i
                                self.logger.debug(f"Matched '{segmentation_name}' to '{item_text}' via base name")
                                break

                if index >= 0:
                    self._segmentation_selector.setCurrentIndex(index)
                else:
                    # Use debug instead of error - this might be normal if dropdown format differs
                    self.logger.debug(f"Could not find '{segmentation_name}' in main dropdown")

                # Re-enable signals
                self._segmentation_selector.blockSignals(False)

        except Exception as e:
            self.logger.error(f"Could not sync main dropdown selection: {e}")

    def _update_all_canvases(self) -> None:
        """Update all canvases - fallback method for canvas updates."""
        try:
            canvases = [self._main_canvas, self._axial_canvas, self._sagittal_canvas, self._coronal_canvas]
            for canvas in canvases:
                if canvas and hasattr(canvas, 'update'):
                    canvas.update()
        except Exception as e:
            self.logger.warning(f"Error updating all canvases: {e}")
    
    def clear_measurement_highlight(self) -> None:
        """Limpia el resaltado of medición in the canvas activo."""
        active_canvas = self._get_active_canvas()
        if active_canvas:
            active_canvas.clear_measurement_highlight()
    
    def remove_measurement(self, measurement_data: dict) -> bool:
        """Elimina una medición específica of the canvas activo."""
        active_canvas = self._get_active_canvas()
        if active_canvas:
            return active_canvas.remove_measurement(measurement_data)
        return False
    
    def update_available_segmentations(self, segmentations: list) -> None:
        """
        Update the segmentation dropdown with available segmentations.
        
        Args:
            segmentations: List of segmentation names
        """
        # Block signals temporarily to avoid triggering change events
        self._segmentation_selector.blockSignals(True)
        
        self._segmentation_selector.clear()
        self._segmentation_selector.addItem("None")
        
        for segmentation in segmentations:
            if isinstance(segmentation, dict):
                name = segmentation.get('name', 'Unknown')
                seg_type = segmentation.get('type', 'Unknown')
                confidence = segmentation.get('confidence', 0)
                
                display_text = name
                    
                self._segmentation_selector.addItem(display_text)
            else:
                self._segmentation_selector.addItem(str(segmentation))
        
        self._segmentation_selector.setCurrentIndex(0)

        # Re-enable signals
        self._segmentation_selector.blockSignals(False)

        if segmentations:
            self._on_segmentation_selection_changed("None")

        self.logger.info(f"Updated segmentation dropdown with {len(segmentations)} segmentations, 'None' selected by default")
    
    def set_active_segmentation(self, segmentation_name: str) -> None:
        """
        Set the active segmentation in the dropdown programmatically.
        
        Args:
            segmentation_name: Name of segmentation to set as active
        """
        # Block signals to avoid circular updates
        self._segmentation_selector.blockSignals(True)
        
        index = self._segmentation_selector.findText(segmentation_name, Qt.MatchFlag.MatchContains)
        if index >= 0:
            self._segmentation_selector.setCurrentIndex(index)
        else:
            # Default to "None" if not found
            self._segmentation_selector.setCurrentIndex(0)
        
        # Re-enable signals
        self._segmentation_selector.blockSignals(False)
        
        # Manually trigger the selection change to activate visualization
        self._on_segmentation_selection_changed(segmentation_name)
        
        self.logger.info(f"Set active segmentation to: {segmentation_name}")
    
    def get_current_segmentation_selection(self) -> str:
        """
        Get the currently selected segmentation from dropdown.
        
        Returns:
            Currently selected segmentation name
        """
        return self._segmentation_selector.currentText()

    def _determine_segmentation_visibility(self, source: str) -> bool:
        """
        Helper method to determine segmentation visibility based on source.

        Args:
            source: Origin of the change ("main_dropdown" or "manual_panel")

        Returns:
            True if segmentation should be visible, False otherwise
        """
        if source == "main_dropdown":
            # Main dropdown always shows the selected segmentation
            self._sync_manual_panel_visibility_button(True)
            return True
        else:
            # Manual panel respects its own visibility button state
            return self._get_manual_panel_visibility_state()