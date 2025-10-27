"""
Quantitative analysis panel for medical images.
Metrics, statistics, histograms, and ROI analysis.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QTableWidget, QTableWidgetItem, QProgressBar,
    QComboBox, QFrame, QTextEdit, QSpinBox, QCheckBox,
    QGridLayout, QSplitter, QTabWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class QuantitativeAnalysisPanel(QWidget):
    """
    Quantitative analysis panel for medical images.
    Auto-calculates metrics, exports results, ROI analysis for segmentations.
    """

    export_results_requested = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self._current_image_data: Optional[np.ndarray] = None
        self._analysis_results: Dict[str, Any] = {}

        self._setup_ui()
        self._setup_connections()

        self.logger.info("Quantitative Analysis Panel initialized")
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title_label = QLabel("Quantitative Analysis")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
        """)

        self._metrics_tab = self._create_metrics_tab()
        self._tab_widget.addTab(self._metrics_tab, "📊 Image Metrics")

        self._roi_tab = self._create_roi_analysis_tab()
        self._tab_widget.addTab(self._roi_tab, "🎯 ROI Analysis")

        self._histogram_tab = self._create_histogram_tab()
        self._tab_widget.addTab(self._histogram_tab, "📈 Histograms")

        layout.addWidget(self._tab_widget)
    
    def _create_metrics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        results_group = self._create_results_area()
        layout.addWidget(results_group, 1)

        auto_status = QLabel("🔄 Analysis runs automatically when image is loaded")
        auto_status.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 8px;
                background-color: #f0f8ff;
                border-radius: 4px;
                text-align: center;
            }
        """)
        auto_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(auto_status, 0)

        actions_layout = self._create_action_buttons()
        layout.addLayout(actions_layout)

        return widget
    
    def _create_roi_analysis_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        mask_info_group = QGroupBox("Segmentation Mask Information")
        mask_info_layout = QVBoxLayout(mask_info_group)

        self._selected_mask_label = QLabel("No mask selected")
        self._selected_mask_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 8px;
                background-color: #f0f8ff;
                border-radius: 4px;
                text-align: center;
            }
        """)
        self._selected_mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_info_layout.addWidget(self._selected_mask_label)

        layout.addWidget(mask_info_group, 0)

        slice_analysis_group = QGroupBox("Slice-by-Slice Analysis")
        slice_layout = QVBoxLayout(slice_analysis_group)

        self._slice_table = QTableWidget()
        self._slice_table.setColumnCount(5)
        self._slice_table.setHorizontalHeaderLabels(["Slice#", "Area (px²)", "Mean Intensity", "Std Dev", "SNR (dB)"])
        self._slice_table.setMinimumHeight(200)
        self._slice_table.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )

        self._slice_table.setAlternatingRowColors(True)
        self._slice_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._slice_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._slice_table.verticalHeader().setVisible(False)

        header = self._slice_table.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(4):
            header.setSectionResizeMode(i, header.ResizeMode.ResizeToContents)

        slice_layout.addWidget(self._slice_table, 1)

        analysis_controls_layout = QHBoxLayout()
        self._refresh_analysis_button = QPushButton("🔄 Refresh Analysis")
        self._refresh_analysis_button.setToolTip("Recalculate ROI analysis for current mask")
        self._export_roi_button = QPushButton("💾 Export ROI Data")
        self._export_roi_button.setToolTip("Export slice analysis to CSV")

        analysis_controls_layout.addWidget(self._refresh_analysis_button)
        analysis_controls_layout.addWidget(self._export_roi_button)
        analysis_controls_layout.addStretch()

        slice_layout.addLayout(analysis_controls_layout, 0)

        layout.addWidget(slice_analysis_group, 1)

        self._current_mask_data = None
        self._current_slice_index = 0
        self._current_orientation = "axial"

        return widget
    
    def _create_histogram_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        histogram_controls = QGroupBox("Histogram Controls")
        histogram_controls.setMaximumHeight(200)
        controls_layout = QGridLayout(histogram_controls)
        
        controls_layout.addWidget(QLabel("Bins:"), 0, 0)
        self._histogram_bins_spinbox = QSpinBox()
        self._histogram_bins_spinbox.setRange(10, 500)
        self._histogram_bins_spinbox.setValue(256)
        controls_layout.addWidget(self._histogram_bins_spinbox, 0, 1)
        
        self._show_cumulative_checkbox = QCheckBox("Show Cumulative")
        controls_layout.addWidget(self._show_cumulative_checkbox, 1, 0, 1, 2)
        
        self._generate_histogram_button = QPushButton("📈 Generate Histogram")
        controls_layout.addWidget(self._generate_histogram_button, 2, 0, 1, 2)

        layout.addWidget(histogram_controls, 0)

        self._histogram_canvas = self._create_histogram_canvas()
        layout.addWidget(self._histogram_canvas, 1)
        
        return widget
    
    def _create_histogram_canvas(self) -> FigureCanvas:
        self._histogram_figure = Figure(dpi=100)
        self._histogram_figure.patch.set_facecolor('white')

        self._histogram_axis = self._histogram_figure.add_subplot(111)
        self._histogram_figure.subplots_adjust(
            left=0.1, right=0.95,
            top=0.9, bottom=0.15,
            hspace=0.3, wspace=0.3
        )

        canvas = FigureCanvas(self._histogram_figure)
        canvas.setMinimumSize(400, 250)
        canvas.setMaximumHeight(400)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        
        return canvas
    
    def _create_results_area(self) -> QGroupBox:
        group = QGroupBox("Analysis Results")
        layout = QVBoxLayout(group)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(3)
        self._results_table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self._results_table.setMinimumHeight(300)
        self._results_table.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._results_table, 4)

        self._results_text = QTextEdit()
        self._results_text.setMinimumHeight(40)
        self._results_text.setPlaceholderText("Detailed analysis results will appear here...")
        self._results_text.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._results_text, 1)
        
        return group
    
    def _create_action_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        
        self._export_csv_button = QPushButton("💾 Export CSV")
        self._export_csv_button.setToolTip("Export results to CSV file")
        layout.addWidget(self._export_csv_button)
        
        self._export_pdf_button = QPushButton("📄 Export PDF")
        self._export_pdf_button.setToolTip("Export results to PDF report")
        layout.addWidget(self._export_pdf_button)
        
        self._clear_results_button = QPushButton("🗑️ Clear Results")
        self._clear_results_button.setToolTip("Clear all analysis results")
        layout.addWidget(self._clear_results_button)
        
        return layout
    
    def _setup_connections(self) -> None:
        self._refresh_analysis_button.clicked.connect(self._refresh_roi_analysis)
        self._export_roi_button.clicked.connect(self._export_roi_analysis)
        self._generate_histogram_button.clicked.connect(self._generate_histogram)
        self._export_csv_button.clicked.connect(lambda: self._export_results("csv"))
        self._export_pdf_button.clicked.connect(lambda: self._export_results("pdf"))
        self._clear_results_button.clicked.connect(self._clear_results)

    def connect_to_mask_selector(self, image_viewer) -> None:
        try:
            self._image_viewer = image_viewer

            if hasattr(image_viewer, '_segmentation_selector'):
                image_viewer._segmentation_selector.currentTextChanged.connect(self._on_mask_selection_changed)
                self.logger.info("Connected to external segmentation selector")

            self.logger.info("ROI Analysis connected to mask selector")

        except Exception as e:
            self.logger.error(f"Error connecting to external mask selector: {e}")
    
    def _on_mask_selection_changed(self, mask_name: str) -> None:
        self.logger.info(f"Mask selection changed: {mask_name}")

        if mask_name == "No segmentation loaded" or mask_name == "None" or not mask_name:
            self._selected_mask_label.setText("No mask selected")
            self._slice_table.setRowCount(0)
            self._current_mask_data = None
            return

        self._selected_mask_label.setText(f"Selected: {mask_name}")

        self._update_image_data_from_viewer()

        if hasattr(self, '_image_viewer') and self._image_viewer:
            mask_data = self._get_mask_data_from_viewer(mask_name)
            if mask_data:
                self._current_mask_data = mask_data
                self._refresh_roi_analysis()
            else:
                self.logger.error(f"Could not get mask data for {mask_name}")

    def _show_no_mask_message(self) -> None:
        self._slice_table.setRowCount(0)

    def _show_mask_info(self, mask_name: str) -> None:
        try:
            self._slice_table.setRowCount(0)
            self.logger.info(f"Starting ROI analysis for mask: {mask_name}")

            self._refresh_roi_analysis()

        except Exception as e:
            self.logger.error(f"Error showing mask info: {e}")

    def _get_mask_data_from_viewer(self, mask_name: str) -> Optional[Dict]:
        try:
            if hasattr(self._image_viewer, '_overlay_service') and self._image_viewer._overlay_service:
                overlay_service = self._image_viewer._overlay_service

                real_overlay_id = self._convert_dropdown_name_to_overlay_id(mask_name)
                if real_overlay_id:
                    self.logger.debug(f"Converted dropdown name '{mask_name}' to overlay ID '{real_overlay_id}'")

                    if hasattr(overlay_service, 'get_overlay_mask_data'):
                        mask_array = overlay_service.get_overlay_mask_data(real_overlay_id)
                        if mask_array is not None:
                            self.logger.info(f"Found mask data for {mask_name}: shape {mask_array.shape}")
                            return {
                                'mask_array': mask_array,
                                'name': mask_name,
                                'overlay_id': real_overlay_id
                            }

                if hasattr(overlay_service, 'get_all_overlay_ids'):
                    all_overlay_ids = overlay_service.get_all_overlay_ids()
                    self.logger.debug(f"Available overlay IDs: {all_overlay_ids}")

                    for overlay_id in all_overlay_ids:
                        if mask_name in overlay_id or overlay_id in mask_name:
                            if hasattr(overlay_service, 'get_overlay_mask_data'):
                                mask_array = overlay_service.get_overlay_mask_data(overlay_id)
                                if mask_array is not None:
                                    self.logger.info(f"Found mask data for {mask_name}: shape {mask_array.shape}")
                                    return {
                                        'mask_array': mask_array,
                                        'name': mask_name,
                                        'overlay_id': overlay_id
                                    }

            self.logger.warning(f"No overlay data found for mask: {mask_name}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting mask data from viewer: {e}")
            return None

    def _convert_dropdown_name_to_overlay_id(self, selection: str) -> Optional[str]:
        try:
            base_name = selection.split(" (")[0].strip()

            if base_name == "None" or base_name == "No segmentation loaded" or not base_name:
                return None

            self.logger.debug(f"Converting dropdown name: '{selection}' → base_name: '{base_name}'")

            overlay_index = 0

            if "_Region_" in base_name:
                try:
                    region_part = base_name.split("_Region_")[-1]
                    region_num = int(region_part)
                    overlay_index = region_num - 1
                    base_name = base_name.split("_Region_")[0]
                    self.logger.debug(f"Extracted region: {region_num} → index: {overlay_index}, base: '{base_name}'")
                except (ValueError, IndexError):
                    pass
            elif base_name.count("_") >= 2:
                parts = base_name.split("_")
                if len(parts) >= 3:
                    possible_region = parts[-1]
                    if possible_region in ['PZ', 'TZ', 'WG']:
                        if possible_region == 'TZ':
                            overlay_index = 0
                        elif possible_region == 'PZ':
                            overlay_index = 1
                        elif possible_region == 'WG':
                            overlay_index = 2
                        base_name = "_".join(parts[:-1])
                        self.logger.debug(f"Region '{possible_region}' → index: {overlay_index}, base: '{base_name}'")

            overlay_id = f"auto_mask_{base_name}_{overlay_index}"

            self.logger.info(f"CONVERTED: '{selection}' → '{overlay_id}'")
            return overlay_id

        except Exception as e:
            self.logger.error(f"Error converting selection to overlay ID: {e}")
            return None

    def _refresh_roi_analysis(self) -> None:
        if self._current_mask_data is None or self._current_image_data is None:
            self.logger.warning("No mask or image data available for ROI analysis")
            return

        try:
            self.logger.info("Performing ROI analysis on selected mask")
            self._calculate_mask_roi_analysis()

        except Exception as e:
            self.logger.error(f"Error during ROI analysis: {e}")

    def _export_roi_analysis(self) -> None:
        if self._slice_table.rowCount() == 0:
            self.logger.warning("No ROI analysis data to export")
            return

        try:
            from PyQt6.QtWidgets import QFileDialog
            from datetime import datetime

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export ROI Analysis",
                f"roi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )

            if file_path:
                self._export_roi_to_csv(file_path)
                self.logger.info(f"ROI analysis exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting ROI analysis: {e}")

    def _calculate_mask_roi_analysis(self) -> None:
        if self._current_mask_data is None or self._current_image_data is None:
            return

        try:
            mask_array = self._current_mask_data.get('mask_array')
            if mask_array is None:
                self.logger.error("No mask array in segmentation data")
                return

            self._slice_table.setRowCount(0)

            image_data = self._get_compatible_image_data(mask_array)
            if image_data is None:
                self.logger.error(f"No compatible image data for mask shape {mask_array.shape}")
                return

            current_orientation = self._get_current_orientation()

            num_slices = self._get_num_slices_for_orientation(mask_array, current_orientation)

            for slice_idx in range(num_slices):
                slice_mask, slice_image = self._extract_slice_for_orientation(
                    mask_array, image_data, slice_idx, current_orientation
                )

                if slice_mask is None or slice_image is None:
                    continue

                metrics = self._calculate_slice_metrics(slice_image, slice_mask, slice_idx)

                if metrics is not None:
                    self._add_slice_to_table(slice_idx, metrics)

            self._highlight_current_slice()

            self.logger.info(f"ROI analysis completed for {num_slices} slices in {current_orientation} orientation")

        except Exception as e:
            self.logger.error(f"Error calculating mask ROI analysis: {e}")

    def _get_compatible_image_data(self, mask_array: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self._current_image_data is not None and mask_array.shape == self._current_image_data.shape:
                return self._current_image_data

            if hasattr(self, '_image_viewer') and self._image_viewer:
                if hasattr(self._image_viewer, 'get_view_manager'):
                    view_manager = self._image_viewer.get_view_manager()
                    if view_manager:
                        current_image = view_manager.get_current_image()
                        if current_image and hasattr(current_image, 'image_data'):
                            full_image_data = current_image.image_data
                            if full_image_data.shape == mask_array.shape:
                                self.logger.info(f"Using 3D image data: shape {full_image_data.shape}")
                                return full_image_data

            if len(mask_array.shape) == 3 and self._current_image_data is not None:
                if len(self._current_image_data.shape) == 2:
                    if self._current_image_data.shape == mask_array.shape[1:]:
                        expanded_image = np.repeat(self._current_image_data[np.newaxis, :, :],
                                                 mask_array.shape[0], axis=0)
                        self.logger.info(f"Expanded 2D to 3D: shape {expanded_image.shape}")
                        return expanded_image

            self.logger.error(f"No compatible image for mask shape {mask_array.shape}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting compatible image data: {e}")
            return None

    def _get_current_orientation(self) -> str:
        try:
            if hasattr(self, '_image_viewer') and self._image_viewer:
                if hasattr(self._image_viewer, 'get_view_manager'):
                    view_manager = self._image_viewer.get_view_manager()
                    if view_manager and hasattr(view_manager, 'get_active_view_plane'):
                        orientation = view_manager.get_active_view_plane()
                        self.logger.debug(f"Orientation: {orientation}")
                        return orientation
            return "axial"
        except Exception as e:
            self.logger.error(f"Error getting current orientation: {e}")
            return "axial"

    def _get_num_slices_for_orientation(self, data_array: np.ndarray, orientation: str) -> int:
        if len(data_array.shape) < 3:
            return 1

        if orientation == "axial":
            return data_array.shape[0]
        elif orientation == "sagittal":
            return data_array.shape[2]
        elif orientation == "coronal":
            return data_array.shape[1]
        else:
            return data_array.shape[0]

    def _extract_slice_for_orientation(self, mask_array: np.ndarray, image_array: np.ndarray,
                                     slice_idx: int, orientation: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            if len(mask_array.shape) < 3 or len(image_array.shape) < 3:
                return mask_array, image_array

            if orientation == "axial":
                slice_mask = mask_array[slice_idx, :, :]
                slice_image = image_array[slice_idx, :, :]
            elif orientation == "sagittal":
                slice_mask = mask_array[:, :, slice_idx]
                slice_image = image_array[:, :, slice_idx]
            elif orientation == "coronal":
                slice_mask = mask_array[:, slice_idx, :]
                slice_image = image_array[:, slice_idx, :]
            else:
                slice_mask = mask_array[slice_idx, :, :]
                slice_image = image_array[slice_idx, :, :]

            return slice_mask, slice_image

        except IndexError:
            return None, None
        except Exception as e:
            self.logger.error(f"Error extracting slice for {orientation}: {e}")
            return None, None

    def _calculate_slice_metrics(self, slice_image: np.ndarray, slice_mask: np.ndarray, slice_idx: int) -> Optional[Dict[str, float]]:
        try:
            roi_pixels = slice_image[slice_mask > 0]

            if len(roi_pixels) == 0:
                return {
                    'area': 0.0,
                    'mean_intensity': 0.0,
                    'std_dev': 0.0,
                    'snr': 0.0
                }

            area = float(np.sum(slice_mask > 0))
            mean_intensity = float(np.mean(roi_pixels))
            std_dev = float(np.std(roi_pixels))

            snr = 0.0
            if std_dev > 0:
                snr = 20 * np.log10(mean_intensity / std_dev)

            return {
                'area': area,
                'mean_intensity': mean_intensity,
                'std_dev': std_dev,
                'snr': snr
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics for slice {slice_idx}: {e}")
            return None

    def _add_slice_to_table(self, slice_idx: int, metrics: Dict[str, float]) -> None:
        try:
            row = self._slice_table.rowCount()
            self._slice_table.insertRow(row)

            self._slice_table.setItem(row, 0, QTableWidgetItem(str(slice_idx)))
            self._slice_table.setItem(row, 1, QTableWidgetItem(f"{metrics['area']:.0f}"))
            self._slice_table.setItem(row, 2, QTableWidgetItem(f"{metrics['mean_intensity']:.2f}"))
            self._slice_table.setItem(row, 3, QTableWidgetItem(f"{metrics['std_dev']:.2f}"))

            snr_value = metrics['snr']
            if np.isfinite(snr_value):
                self._slice_table.setItem(row, 4, QTableWidgetItem(f"{snr_value:.1f}"))
            else:
                self._slice_table.setItem(row, 4, QTableWidgetItem("∞"))

        except Exception as e:
            self.logger.error(f"Error adding slice {slice_idx} to table: {e}")

    def _highlight_current_slice(self) -> None:
        try:
            for row in range(self._slice_table.rowCount()):
                slice_item = self._slice_table.item(row, 0)
                if slice_item and slice_item.text().isdigit():
                    if int(slice_item.text()) == self._current_slice_index:
                        self._slice_table.selectRow(row)
                        self._slice_table.scrollToItem(slice_item)
                        break

        except Exception:
            pass

    def _export_roi_to_csv(self, file_path: str) -> None:
        try:
            import csv
            from datetime import datetime

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['# ROI Analysis Export'])
                writer.writerow(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# Selected Mask:', self._selected_mask_label.text()])
                writer.writerow(['# Orientation:', self._current_orientation])
                writer.writerow([])

                writer.writerow(['Slice#', 'Area (px²)', 'Mean Intensity', 'Std Dev', 'SNR (dB)'])

                for row in range(self._slice_table.rowCount()):
                    row_data = []
                    for col in range(self._slice_table.columnCount()):
                        item = self._slice_table.item(row, col)
                        row_data.append(item.text() if item else '')
                    writer.writerow(row_data)

                self.logger.info(f"ROI analysis data exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting ROI data to CSV: {e}")
            raise

    def set_current_slice(self, slice_index: int) -> None:
        self._current_slice_index = slice_index
        self._highlight_current_slice()

    def set_current_orientation(self, orientation: str) -> None:
        self.logger.info(f"Orientation: {orientation}")
        self._current_orientation = orientation

        self._update_image_data_from_viewer()

        if self._current_mask_data:
            self.logger.info(f"Refreshing ROI for {orientation}")
            self._refresh_roi_analysis()

    def connect_to_image_viewer(self, image_viewer) -> None:
        try:
            if not hasattr(self, '_image_viewer'):
                self._image_viewer = image_viewer

            if hasattr(image_viewer, 'slice_changed'):
                image_viewer.slice_changed.connect(self.set_current_slice)
                self.logger.info("Connected to slice changes")

            if hasattr(image_viewer, 'view_changed'):
                image_viewer.view_changed.connect(self.set_current_orientation)
                self.logger.info("Connected to view changes")

            self._update_image_data_from_viewer()

        except Exception as e:
            self.logger.error(f"Error connecting to image viewer: {e}")

    def _update_image_data_from_viewer(self) -> None:
        try:
            if hasattr(self, '_image_viewer') and self._image_viewer:
                if hasattr(self._image_viewer, '_get_active_canvas'):
                    active_canvas = self._image_viewer._get_active_canvas()
                    if active_canvas and hasattr(active_canvas, '_current_image_data'):
                        if active_canvas._current_image_data is not None:
                            self._current_image_data = active_canvas._current_image_data
                            self.logger.info(f"Updated from canvas: shape {self._current_image_data.shape}")
                            return

                if hasattr(self._image_viewer, 'get_view_manager'):
                    view_manager = self._image_viewer.get_view_manager()
                    if view_manager:
                        current_image = view_manager.get_current_image()
                        if current_image and hasattr(current_image, 'image_data'):
                            if hasattr(self._image_viewer, 'get_current_plane'):
                                current_plane = self._image_viewer.get_current_plane()
                                if hasattr(view_manager, 'get_slice_data_for_plane'):
                                    slice_data = view_manager.get_slice_data_for_plane(current_plane)

                                    if slice_data and 'slice_data' in slice_data:
                                        self._current_image_data = slice_data['slice_data']
                                        self.logger.info(f"Updated from view manager: shape {self._current_image_data.shape}")
                                        return

                            self._current_image_data = current_image.image_data
                            self.logger.info(f"Updated with 3D data: shape {self._current_image_data.shape}")
                            return

                self.logger.warning("No image data from viewer")

        except Exception as e:
            self.logger.error(f"Error updating image data from viewer: {e}")

    def _generate_histogram(self) -> None:
        if self._current_image_data is not None:
            bins = self._histogram_bins_spinbox.value()
            show_cumulative = self._show_cumulative_checkbox.isChecked()

            self._histogram_figure.clear()

            self._histogram_axis = self._histogram_figure.add_subplot(111)

            counts, bin_edges = np.histogram(self._current_image_data.flatten(), bins=bins)

            self._histogram_axis.hist(
                self._current_image_data.flatten(),
                bins=bins,
                alpha=0.7,
                color='blue',
                label='Histogram'
            )

            if show_cumulative:
                cumulative = np.cumsum(counts)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax2 = self._histogram_axis.twinx()
                ax2.plot(bin_centers, cumulative, 'r-', label='Cumulative', linewidth=2)
                ax2.set_ylabel('Cumulative Count', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            
            self._histogram_axis.set_xlabel('Intensity')
            self._histogram_axis.set_ylabel('Count')
            self._histogram_axis.set_title('Image Intensity Histogram')
            self._histogram_axis.grid(True, alpha=0.3)

            self._histogram_figure.tight_layout()
            self._histogram_canvas.draw()

            self.logger.debug("Histogram generated")
    
    def _export_results(self, format_type: str) -> None:
        if not self._analysis_results:
            self.logger.warning("No analysis results to export")
            return
        
        try:
            from PyQt6.QtWidgets import QFileDialog
            import csv
            import os
            from datetime import datetime

            if format_type == "csv":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Export Analysis Results", 
                    f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "CSV Files (*.csv)"
                )
                if file_path:
                    self._export_to_csv(file_path)
            elif format_type == "pdf":
                file_path, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Export Analysis Report", 
                    f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "PDF Files (*.pdf)"
                )
                if file_path:
                    self._export_to_pdf(file_path)
                    
        except Exception as e:
            self.logger.error(f"Error during export: {e}")
        
        self.export_results_requested.emit(format_type)
        self.logger.debug(f"Export requested: {format_type}")
    
    def _export_to_csv(self, file_path: str) -> None:
        import csv
        from datetime import datetime

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['# Quantitative Analysis Export'])
                writer.writerow(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# Image Shape:', str(self._current_image_data.shape) if self._current_image_data is not None else 'N/A'])
                writer.writerow([])

                writer.writerow(['Category', 'Metric', 'Value', 'Unit', 'Description'])

                basic_metrics = ['Mean Intensity', 'Standard Deviation', 'Min Intensity', 'Max Intensity', 'Median Intensity']
                texture_metrics = ['Entropy', 'Homogeneity', 'Local Contrast', 'Energy']
                quality_metrics = ['SNR']
                
                for metric_name, data in self._analysis_results.items():
                    if metric_name in basic_metrics:
                        category = 'Basic Statistics'
                    elif metric_name in texture_metrics:
                        category = 'Texture Analysis'
                    elif metric_name in quality_metrics:
                        category = 'Image Quality'
                    else:
                        category = 'Other'
                    
                    description = self._get_metric_description(metric_name)
                    writer.writerow([category, metric_name, data['value'], data['unit'], description])
                
                self.logger.info(f"Analysis results exported to CSV: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def _get_metric_description(self, metric_name: str) -> str:
        descriptions = {
            'Mean Intensity': 'Average pixel intensity value',
            'Standard Deviation': 'Measure of intensity variability',
            'Min Intensity': 'Minimum pixel intensity value',
            'Max Intensity': 'Maximum pixel intensity value',
            'Median Intensity': 'Middle value of intensity distribution',
            'Entropy': 'Shannon entropy for texture complexity',
            'SNR': 'Signal-to-Noise Ratio in decibels',
            'Homogeneity': 'Local uniformity measure',
            'Local Contrast': 'Local intensity variation',
            'Energy': 'Uniformity of intensity distribution'
        }
        return descriptions.get(metric_name, 'Advanced image metric')
    
    def _export_to_pdf(self, file_path: str) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from datetime import datetime
            
            with PdfPages(file_path) as pdf:
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Advanced Quantitative Analysis Report', fontsize=16, fontweight='bold')

                gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

                basic_metrics = ['Mean Intensity', 'Standard Deviation', 'Min Intensity', 'Max Intensity', 'Median Intensity']
                texture_metrics = ['Entropy', 'Homogeneity', 'Local Contrast', 'Energy']
                quality_metrics = ['SNR']

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.axis('tight')
                ax1.axis('off')
                ax1.set_title('Basic Statistics', fontweight='bold', fontsize=12)
                
                basic_data = [[name, f"{self._analysis_results[name]['value']:.4f}", self._analysis_results[name]['unit']] 
                             for name in basic_metrics if name in self._analysis_results]
                
                if basic_data:
                    table1 = ax1.table(cellText=basic_data, 
                                      colLabels=['Metric', 'Value', 'Unit'],
                                      cellLoc='center', loc='center')
                    table1.auto_set_font_size(False)
                    table1.set_fontsize(8)
                    table1.scale(1.2, 1.2)

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.axis('tight')
                ax2.axis('off')
                ax2.set_title('Texture Analysis', fontweight='bold', fontsize=12)
                
                texture_data = [[name, f"{self._analysis_results[name]['value']:.4f}", self._analysis_results[name]['unit']] 
                               for name in texture_metrics if name in self._analysis_results]
                
                if texture_data:
                    table2 = ax2.table(cellText=texture_data, 
                                      colLabels=['Metric', 'Value', 'Unit'],
                                      cellLoc='center', loc='center')
                    table2.auto_set_font_size(False)
                    table2.set_fontsize(8)
                    table2.scale(1.2, 1.2)

                ax3 = fig.add_subplot(gs[1, 0])
                ax3.axis('tight')
                ax3.axis('off')
                ax3.set_title('Image Quality', fontweight='bold', fontsize=12)
                
                quality_data = [[name, f"{self._analysis_results[name]['value']:.4f}", self._analysis_results[name]['unit']] 
                               for name in quality_metrics if name in self._analysis_results]
                
                if quality_data:
                    table3 = ax3.table(cellText=quality_data, 
                                      colLabels=['Metric', 'Value', 'Unit'],
                                      cellLoc='center', loc='center')
                    table3.auto_set_font_size(False)
                    table3.set_fontsize(8)
                    table3.scale(1.2, 1.2)

                ax4 = fig.add_subplot(gs[1, 1])
                if self._current_image_data is not None:
                    ax4.hist(self._current_image_data.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                    ax4.set_title('Intensity Distribution', fontweight='bold', fontsize=12)
                    ax4.set_xlabel('Intensity')
                    ax4.set_ylabel('Count')
                    ax4.grid(True, alpha=0.3)

                ax5 = fig.add_subplot(gs[2, :])
                ax5.axis('off')
                ax5.set_title('Report Summary', fontweight='bold', fontsize=12)

                total_metrics = len(self._analysis_results)
                basic_count = len([m for m in basic_metrics if m in self._analysis_results])
                texture_count = len([m for m in texture_metrics if m in self._analysis_results])
                quality_count = len([m for m in quality_metrics if m in self._analysis_results])
                
                report_info = f"""Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: Advanced Quantitative Medical Image Analysis
Image Shape: {self._current_image_data.shape if self._current_image_data is not None else 'N/A'}

Metrics Summary:
  • Basic Statistics: {basic_count} metrics (intensity statistics)
  • Texture Analysis: {texture_count} metrics (complexity, homogeneity, contrast)
  • Image Quality: {quality_count} metrics (signal-to-noise ratio)
  • Total Computed: {total_metrics} metrics

This comprehensive report includes advanced medical imaging metrics for quantitative analysis.
Generated by the Quantitative Analysis panel with clean architecture implementation."""
                
                ax5.text(0.05, 0.95, report_info, transform=ax5.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Advanced analysis report exported to PDF: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting to PDF: {e}")
            raise
    
    def _clear_results(self) -> None:
        self._results_table.setRowCount(0)
        self._results_text.clear()
        self._analysis_results.clear()

        self._histogram_figure.clear()
        self._histogram_axis = self._histogram_figure.add_subplot(111)
        self._histogram_axis.set_title('No histogram generated')
        self._histogram_axis.text(0.5, 0.5, 'Load an image to generate histogram',
                                  ha='center', va='center', transform=self._histogram_axis.transAxes)
        self._histogram_canvas.draw()

        self.logger.debug("Results cleared")
    
    def set_image_data(self, image_data: np.ndarray) -> None:
        self._current_image_data = image_data
        self.logger.debug(f"Image data: shape {image_data.shape}")

        self._results_table.setRowCount(0)
        self._results_text.clear()
        self._analysis_results.clear()

        self._calculate_basic_metrics()

        self.logger.info("Metrics calculated")
    
    def add_analysis_result(self, metric_name: str, value: float, unit: str = "") -> None:
        row_count = self._results_table.rowCount()
        self._results_table.insertRow(row_count)

        self._results_table.setItem(row_count, 0, QTableWidgetItem(metric_name))
        self._results_table.setItem(row_count, 1, QTableWidgetItem(f"{value:.4f}"))
        self._results_table.setItem(row_count, 2, QTableWidgetItem(unit))

        self._analysis_results[metric_name] = {'value': value, 'unit': unit}

        self.logger.debug(f"Added: {metric_name} = {value} {unit}")
    
    def get_analysis_results(self) -> Dict[str, Any]:
        return self._analysis_results.copy()

    def _calculate_basic_metrics(self) -> None:
        if self._current_image_data is None:
            return
        
        try:
            data_flat = self._current_image_data.flatten()

            mean_intensity = float(np.mean(data_flat))
            self.add_analysis_result("Mean Intensity", mean_intensity, "intensity units")

            std_intensity = float(np.std(data_flat))
            self.add_analysis_result("Standard Deviation", std_intensity, "intensity units")

            min_intensity = float(np.min(data_flat))
            max_intensity = float(np.max(data_flat))
            self.add_analysis_result("Minimum Intensity", min_intensity, "intensity units")
            self.add_analysis_result("Maximum Intensity", max_intensity, "intensity units")

            intensity_range = max_intensity - min_intensity
            self.add_analysis_result("Intensity Range", intensity_range, "intensity units")

            contrast = 0.0
            if mean_intensity > 0:
                contrast = std_intensity / mean_intensity
                self.add_analysis_result("Contrast Ratio", contrast, "ratio")
            else:
                self.add_analysis_result("Contrast Ratio", 0.0, "ratio")

            shape = self._current_image_data.shape
            if len(shape) == 3:
                depth, height, width = shape
                self.add_analysis_result("Image Depth", float(depth), "slices")
                self.add_analysis_result("Image Height", float(height), "pixels")
                self.add_analysis_result("Image Width", float(width), "pixels")
                total_voxels = depth * height * width
                self.add_analysis_result("Total Voxels", float(total_voxels), "voxels")
            elif len(shape) == 2:
                height, width = shape
                self.add_analysis_result("Image Height", float(height), "pixels")
                self.add_analysis_result("Image Width", float(width), "pixels")
                total_pixels = height * width
                self.add_analysis_result("Total Pixels", float(total_pixels), "pixels")

            self._calculate_advanced_metrics(data_flat, mean_intensity, std_intensity)

            entropy_value = self._analysis_results.get("Entropy", {}).get("value", 0)
            snr_value = self._analysis_results.get("SNR", {}).get("value", 0)
            
            summary = f"""Medical Image Analysis Summary:
            
Shape: {shape}
Mean Intensity: {mean_intensity:.2f}
Standard Deviation: {std_intensity:.2f}
Intensity Range: [{min_intensity:.2f}, {max_intensity:.2f}]
Contrast Ratio: {contrast:.4f}
Entropy: {entropy_value:.4f}
SNR: {snr_value:.2f} dB

Analysis completed automatically when image was loaded.
Switch to the Histograms tab to see the intensity distribution."""

            self._results_text.setText(summary)

            self.logger.info("Metrics calculated")
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            self._results_text.setText(f"Error calculating metrics: {str(e)}")
    
    def _calculate_advanced_metrics(self, data_flat: np.ndarray, mean_intensity: float, std_intensity: float) -> None:
        try:
            entropy_value = self._calculate_shannon_entropy(data_flat)
            self.add_analysis_result("Entropy", entropy_value, "bits")

            snr_value = self._calculate_snr(mean_intensity, std_intensity)
            self.add_analysis_result("SNR", snr_value, "dB")

            contrast_enhancement = self._calculate_contrast_enhancement_index(data_flat, mean_intensity, std_intensity)
            self.add_analysis_result("Contrast Enhancement Index", contrast_enhancement, "ratio")

            if mean_intensity > 0:
                cv = (std_intensity / mean_intensity) * 100
                self.add_analysis_result("Coefficient of Variation", cv, "%")

            texture_metrics = self._calculate_texture_metrics(data_flat)
            for metric_name, value in texture_metrics.items():
                self.add_analysis_result(metric_name, value, "ratio")

            self.logger.debug("Advanced metrics calculated")
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
    
    def _calculate_shannon_entropy(self, data_flat: np.ndarray) -> float:
        try:
            hist, _ = np.histogram(data_flat, bins=256, density=True)

            hist = hist[hist > 0]

            entropy = -np.sum(hist * np.log2(hist))
            
            return float(entropy)
            
        except Exception as e:
            self.logger.error(f"Error calculating Shannon entropy: {e}")
            return 0.0
    
    def _calculate_snr(self, mean_intensity: float, std_intensity: float) -> float:
        try:
            if std_intensity > 0:
                snr_db = 20 * np.log10(mean_intensity / std_intensity)
                return float(snr_db)
            else:
                return float('inf')
                
        except Exception as e:
            self.logger.error(f"Error calculating SNR: {e}")
            return 0.0
    
    def _calculate_contrast_enhancement_index(self, data_flat: np.ndarray, mean_intensity: float, std_intensity: float) -> float:
        try:
            q25, q75 = np.percentile(data_flat, [25, 75])
            interquartile_range = q75 - q25

            if std_intensity > 0:
                contrast_index = interquartile_range / std_intensity
                return float(contrast_index)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating contrast enhancement index: {e}")
            return 0.0
    

    def _calculate_texture_metrics(self, data_flat: np.ndarray) -> Dict[str, float]:
        try:
            metrics = {}

            if len(data_flat) > 100:
                sqrt_len = int(np.sqrt(len(data_flat)))
                if sqrt_len > 10:
                    reshaped = data_flat[:sqrt_len*sqrt_len].reshape(sqrt_len, sqrt_len)

                    homogeneity = self._calculate_homogeneity(reshaped)
                    metrics["Homogeneity"] = homogeneity

                    local_contrast = self._calculate_local_contrast(reshaped)
                    metrics["Local Contrast"] = local_contrast

                    energy = self._calculate_energy(reshaped)
                    metrics["Energy"] = energy
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating texture metrics: {e}")
            return {}
    
    def _calculate_homogeneity(self, image_2d: np.ndarray) -> float:
        try:
            diff_horizontal = np.diff(image_2d, axis=1)
            diff_vertical = np.diff(image_2d, axis=0)

            mean_diff = np.mean(np.abs(diff_horizontal)) + np.mean(np.abs(diff_vertical))

            if mean_diff > 0:
                homogeneity = 1.0 / (1.0 + mean_diff)
            else:
                homogeneity = 1.0
                
            return float(homogeneity)
            
        except Exception as e:
            self.logger.error(f"Error calculating homogeneity: {e}")
            return 0.0
    
    def _calculate_local_contrast(self, image_2d: np.ndarray) -> float:
        try:
            grad_x = np.gradient(image_2d, axis=1)
            grad_y = np.gradient(image_2d, axis=0)

            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            local_contrast = np.mean(gradient_magnitude)

            return float(local_contrast)
            
        except Exception as e:
            self.logger.error(f"Error calculating local contrast: {e}")
            return 0.0
    
    def _calculate_energy(self, image_2d: np.ndarray) -> float:
        try:
            normalized = image_2d / (np.max(image_2d) + 1e-8)

            energy = np.sum(normalized**2) / normalized.size
            
            return float(energy)
            
        except Exception as e:
            self.logger.error(f"Error calculating energy: {e}")
            return 0.0