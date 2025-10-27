"""
infrastructure/ui/widgets/volume_viewer_3d.py

3D Volume Viewer widget with VTK medical rendering.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, 
    QPushButton, QSlider, QRadioButton, QCheckBox, QGroupBox,
    QSizePolicy, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

from deepprostate.core.domain.entities.medical_image import MedicalImage


class VolumeViewer3D(QWidget):
    """
    3D Volume Viewer for medical image visualization.
    Supports surface rendering, ray casting, and real-time parameter adjustment.
    """

    volume_loaded = pyqtSignal(str)
    rendering_mode_changed = pyqtSignal(str)
    parameter_changed = pyqtSignal(str, object)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._logger = logging.getLogger(self.__class__.__name__)

        self._current_image: Optional[MedicalImage] = None
        self._current_volume = None
        self._current_actor = None
        self._iso_value = 50

        self._segmentation_masks: Dict[str, np.ndarray] = {}
        self._segmentation_colors: Dict[str, Tuple[float, float, float]] = {}
        self._segmentation_actors: Dict[str, Any] = {}
        self._active_segmentation_id: Optional[str] = None

        self._vtk_widget: Optional[QVTKRenderWindowInteractor] = None
        self._renderer: Optional[vtk.vtkRenderer] = None
        self._render_window: Optional[vtk.vtkRenderWindow] = None
        self._render_window_interactor = None

        self._setup_ui()
        self._setup_vtk_components()
        self._connect_signals()
        self._apply_medical_theme()
    
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        self._create_left_control_panel(main_layout)
        self._create_viewport_area(main_layout)
    
    def _create_left_control_panel(self, layout: QHBoxLayout):
        controls_frame = QFrame()
        controls_frame.setMinimumWidth(300)
        controls_frame.setMaximumWidth(350)
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #272c36;
                border: none;
                border-radius: 10px;
            }
        """)

        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        controls_layout.setSpacing(15)

        self._create_main_controls_section(controls_layout)
        self._create_visualization_controls_section(controls_layout)

        controls_layout.addStretch()

        layout.addWidget(controls_frame)
    
    def _create_viewport_area(self, layout: QHBoxLayout):
        self._viewport_widget = QWidget()
        self._viewport_widget.setMinimumSize(400, 300)
        self._viewport_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2d31;
                border: none;
                border-radius: 5px;
            }
        """)

        layout.addWidget(self._viewport_widget, 1)
    
    def _create_main_controls_section(self, layout: QVBoxLayout):
        title_layout = QHBoxLayout()

        title_label = QLabel("Volume Rendering")
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: rgb(255, 255, 255); margin-bottom: 10px;")
        title_layout.addWidget(title_label)

        self._color_scheme_combo = QComboBox()
        self._color_scheme_combo.setMinimumWidth(140)
        self._color_scheme_combo.setMaximumHeight(30)
        self._color_scheme_combo.addItems([
            "Standard Medical",
            "Bone/Tissue",
            "Angiography",
            "Rainbow Medical",
            "Grayscale Enhanced",
            "Hot Iron",
            "PET/CT",
            "Prostate Specific"
        ])
        self._color_scheme_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d30;
                color: white;
                border: 1px solid #3d3d40;
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: white;
                selection-background-color: #405cf5;
            }
        """)
        title_layout.addWidget(self._color_scheme_combo)

        layout.addLayout(title_layout)

        self._load_button = QPushButton("Load Volume Data")
        self._load_button.setMinimumHeight(40)
        self._load_button.setEnabled(False)
        self._setup_button_style(self._load_button, "#405cf5")
        layout.addWidget(self._load_button)

        iso_group = QGroupBox("ISO Surface Value")
        iso_group.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); font-weight: bold; }")
        iso_layout = QVBoxLayout(iso_group)

        self._iso_slider = QSlider(Qt.Orientation.Horizontal)
        self._iso_slider.setMinimum(1)
        self._iso_slider.setMaximum(2000)
        self._iso_slider.setValue(50)
        self._iso_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._iso_slider.setTickInterval(200)
        iso_layout.addWidget(self._iso_slider)

        self._iso_label = QLabel("ISO Value: 50")
        self._iso_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._setup_label_style(self._iso_label)
        iso_layout.addWidget(self._iso_label)

        layout.addWidget(iso_group)

        mode_group = QGroupBox("Rendering Mode")
        mode_group.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); font-weight: bold; }")
        mode_layout = QVBoxLayout(mode_group)

        self._surface_radio = QRadioButton("Surface Rendering")
        self._raycast_radio = QRadioButton("Ray Cast Rendering")
        self._raycast_radio.setChecked(True)

        self._setup_radio_style(self._surface_radio)
        self._setup_radio_style(self._raycast_radio)

        mode_layout.addWidget(self._surface_radio)
        mode_layout.addWidget(self._raycast_radio)
        layout.addWidget(mode_group)

        options_group = QGroupBox("Options")
        options_group.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); font-weight: bold; }")
        options_layout = QVBoxLayout(options_group)

        self._realtime_check = QCheckBox("Real Time Updates")
        self._realtime_check.setChecked(True)
        self._setup_checkbox_style(self._realtime_check)
        options_layout.addWidget(self._realtime_check)

        self._render_button = QPushButton("Render")
        self._render_button.setMinimumHeight(35)
        self._render_button.setEnabled(False)
        self._setup_button_style(self._render_button, "#28a745")
        options_layout.addWidget(self._render_button)

        layout.addWidget(options_group)

    def _create_visualization_controls_section(self, layout: QVBoxLayout):
        viz_group = QGroupBox("Visualization Controls")
        viz_group.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); font-weight: bold; }")
        viz_layout = QVBoxLayout(viz_group)

        # Mask opacity control
        mask_opacity_layout = QVBoxLayout()
        mask_opacity_label = QLabel("Mask Opacity")
        mask_opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._setup_label_style(mask_opacity_label)
        mask_opacity_layout.addWidget(mask_opacity_label)

        self._mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_opacity_slider.setMinimum(0)
        self._mask_opacity_slider.setMaximum(100)
        self._mask_opacity_slider.setValue(60)  # 0.6 default
        self._mask_opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._mask_opacity_slider.setTickInterval(10)
        mask_opacity_layout.addWidget(self._mask_opacity_slider)

        self._mask_opacity_label = QLabel("60%")
        self._mask_opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._setup_label_style(self._mask_opacity_label)
        mask_opacity_layout.addWidget(self._mask_opacity_label)

        viz_layout.addLayout(mask_opacity_layout)

        # Volume opacity control
        volume_opacity_layout = QVBoxLayout()
        volume_opacity_label = QLabel("Volume Opacity")
        volume_opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._setup_label_style(volume_opacity_label)
        volume_opacity_layout.addWidget(volume_opacity_label)

        self._volume_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_opacity_slider.setMinimum(0)
        self._volume_opacity_slider.setMaximum(100)
        self._volume_opacity_slider.setValue(80)  # 0.8 default
        self._volume_opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._volume_opacity_slider.setTickInterval(10)
        volume_opacity_layout.addWidget(self._volume_opacity_slider)

        self._volume_opacity_label = QLabel("80%")
        self._volume_opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._setup_label_style(self._volume_opacity_label)
        volume_opacity_layout.addWidget(self._volume_opacity_label)

        viz_layout.addLayout(volume_opacity_layout)

        # Toggle volume visibility
        self._show_volume_check = QCheckBox("Show Volume")
        self._show_volume_check.setChecked(True)
        self._setup_checkbox_style(self._show_volume_check)
        viz_layout.addWidget(self._show_volume_check)

        # Reset camera button
        reset_camera_button = QPushButton("Reset Camera")
        reset_camera_button.setMinimumHeight(35)
        self._setup_button_style(reset_camera_button, "#6c757d")
        viz_layout.addWidget(reset_camera_button)

        layout.addWidget(viz_group)

        self._reset_camera_button = reset_camera_button

    def _setup_vtk_components(self):
        if not VTK_AVAILABLE:
            self._logger.error("VTK not available - 3D rendering disabled")
            return


        try:
            self._vtk_widget = QVTKRenderWindowInteractor()

            if not self._viewport_widget.layout():
                layout = QVBoxLayout(self._viewport_widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
            else:
                layout = self._viewport_widget.layout()

            layout.addWidget(self._vtk_widget)

            self._vtk_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            self._renderer = vtk.vtkRenderer()
            self._renderer.SetBackground(0.1, 0.1, 0.1)
            self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

            self._render_window_interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
            self._render_window = self._vtk_widget.GetRenderWindow()

            self._render_window.SetMultiSamples(4)

            self._logger.info("VTK components initialized successfully")

        except Exception as e:
            self._logger.error(f"Error setting up VTK components: {e}")

    def _connect_signals(self):
        self._iso_slider.valueChanged.connect(self._on_iso_value_changed)
        self._load_button.clicked.connect(self._on_load_volume)
        self._render_button.clicked.connect(self._on_render_volume)
        self._color_scheme_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        self._surface_radio.toggled.connect(self._on_rendering_mode_changed)
        self._raycast_radio.toggled.connect(self._on_rendering_mode_changed)
        self._mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        self._volume_opacity_slider.valueChanged.connect(self._on_volume_opacity_changed)
        self._show_volume_check.toggled.connect(self._on_show_volume_toggled)
        self._reset_camera_button.clicked.connect(self._on_reset_camera)
    
    def set_medical_image(self, image: MedicalImage):
        """
        Set the medical image for 3D visualization.

        Args:
            image: Medical image to visualize
        """
        self._current_image = image
        self._load_button.setEnabled(True)

        # Auto-load if enabled OR if volume already loaded (to update spacing)
        if hasattr(self, '_auto_load') and self._auto_load:
            self._load_volume_data()
        elif hasattr(self, '_vtk_image_data') and self._vtk_image_data:
            # Volume already loaded, reload to update spacing for new sequence
            self._load_volume_data()

    def add_segmentation_mask(self, mask_id: str, mask_3d: np.ndarray, color: Tuple[int, int, int], region_type: str = "unknown"):
        """
        Add a 3D segmentation mask for visualization.

        Args:
            mask_id: Unique identifier for the mask
            mask_3d: 3D numpy array (boolean or int) with segmentation data
            color: RGB color tuple (0-255)
            region_type: Anatomical region type for reference
        """
        try:
            if mask_3d is None or mask_3d.size == 0:
                self._logger.warning(f"Empty mask data for {mask_id}")
                return

            # Convert to boolean if needed
            if mask_3d.dtype != bool:
                mask_3d = mask_3d > 0

            # CHECK: If mask dimensions don't match base image, resample it
            if hasattr(self, '_vtk_image_data') and self._vtk_image_data:
                base_dims = self._vtk_image_data.GetDimensions()
                base_dims_numpy = (base_dims[2], base_dims[1], base_dims[0])  # VTK to numpy order
                mask_dims = mask_3d.shape

                if mask_dims != base_dims_numpy:
                    mask_3d = self._resample_mask_to_image(mask_3d, base_dims_numpy)

            # Store mask and color (convert RGB 0-255 to 0-1 for VTK)
            self._segmentation_masks[mask_id] = mask_3d.copy()
            self._segmentation_colors[mask_id] = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            self._logger.info(f"Added segmentation mask '{mask_id}' with shape {mask_3d.shape}")

        except Exception as e:
            self._logger.error(f"Error adding segmentation mask {mask_id}: {e}")

    def clear_all_segmentation_masks(self):
        """Clear all segmentation masks and actors."""
        try:
            # Remove all actors from renderer
            for actor in self._segmentation_actors.values():
                if self._renderer:
                    self._renderer.RemoveActor(actor)

            # Clear storage
            self._segmentation_masks.clear()
            self._segmentation_colors.clear()
            self._segmentation_actors.clear()
            self._active_segmentation_id = None

            # Refresh view
            if self._render_window:
                self._render_window.Render()

            self._logger.debug("Cleared all segmentation masks")

        except Exception as e:
            self._logger.error(f"Error clearing segmentation masks: {e}")

    def _recreate_mask_actors(self):
        """
        Recreate all mask actors with updated spacing from current _vtk_image_data.

        This is called after _vtk_image_data is updated (e.g., when sequence changes)
        to ensure masks use the correct spacing and origin.

        IMPORTANT: If mask dimensions don't match base image, resample the mask.
        """
        try:
            if not self._segmentation_masks:
                return  # No masks to recreate

            # Store which mask was active before recreation
            active_mask = self._active_segmentation_id

            # Get base image dimensions for comparison
            base_dims = None
            if hasattr(self, '_vtk_image_data') and self._vtk_image_data:
                base_dims = self._vtk_image_data.GetDimensions()

            # Remove old actors from renderer (but keep mask data)
            for mask_id, actor in self._segmentation_actors.items():
                if self._renderer:
                    self._renderer.RemoveActor(actor)

            # Clear actor registry (but keep mask data and colors)
            self._segmentation_actors.clear()

            # Recreate actors with updated spacing AND resampling if needed
            for mask_id, mask_data in self._segmentation_masks.items():
                color = self._segmentation_colors.get(mask_id, (1.0, 0.0, 0.0))

                # Check if mask needs resampling (different dimensions than base image)
                mask_dims_numpy = mask_data.shape  # (depth, height, width)

                if base_dims is not None:
                    # VTK dimensions are (width, height, depth)
                    base_dims_numpy = (base_dims[2], base_dims[1], base_dims[0])

                    if mask_dims_numpy != base_dims_numpy:
                        # Resample mask to match base image dimensions
                        mask_data = self._resample_mask_to_image(mask_data, base_dims_numpy)

                        # CRITICAL: Update stored mask with resampled version
                        self._segmentation_masks[mask_id] = mask_data

                # Create VTK image from mask using CURRENT spacing/origin
                vtk_mask_image = self._numpy_to_vtk_image(mask_data)

                # Create surface using marching cubes
                marching_cubes = vtk.vtkMarchingCubes()
                marching_cubes.SetInputData(vtk_mask_image)
                marching_cubes.SetValue(0, 0.5)
                marching_cubes.Update()

                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(marching_cubes.GetOutputPort())
                mapper.ScalarVisibilityOff()

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetOpacity(0.6)

                # Store new actor
                self._segmentation_actors[mask_id] = actor

            self._logger.info(f"Recreated {len(self._segmentation_actors)} mask actors with updated spacing")

            # Restore active mask if there was one
            if active_mask and active_mask in self._segmentation_actors:
                self.show_segmentation_mask(active_mask)

        except Exception as e:
            self._logger.error(f"Error recreating mask actors: {e}")
            import traceback
            traceback.print_exc()

    def _resample_mask_to_image(self, mask_array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Resample a mask to match the target image dimensions using nearest-neighbor interpolation.

        Args:
            mask_array: Original mask array with shape (depth, height, width)
            target_shape: Target shape (depth, height, width)

        Returns:
            Resampled mask array with target shape
        """
        try:
            from scipy.ndimage import zoom

            # Calculate zoom factors for each dimension
            zoom_factors = (
                target_shape[0] / mask_array.shape[0],  # depth
                target_shape[1] / mask_array.shape[1],  # height
                target_shape[2] / mask_array.shape[2]   # width
            )

            # Use nearest-neighbor interpolation (order=0) to preserve binary mask values
            resampled_mask = zoom(mask_array.astype(float), zoom_factors, order=0, mode='nearest')

            # Convert back to boolean
            resampled_mask = resampled_mask > 0.5

            return resampled_mask

        except Exception as e:
            self._logger.error(f"Error resampling mask: {e}")
            import traceback
            traceback.print_exc()
            return mask_array  # Return original if resampling fails

    def show_segmentation_mask(self, mask_id: str):
        """
        Show a specific segmentation mask in 3D.

        Args:
            mask_id: ID of the mask to show (or "None" to hide all)
        """
        try:
            self._logger.info(f"show_segmentation_mask called with: '{mask_id}'")
            self._logger.info(f"   Available masks: {list(self._segmentation_masks.keys())}")
            self._logger.info(f"   VTK_AVAILABLE: {VTK_AVAILABLE}, renderer: {self._renderer is not None}")

            if not VTK_AVAILABLE or not self._renderer:
                self._logger.error("VTK not available or no renderer")
                return

            # Hide all masks first
            for actor in self._segmentation_actors.values():
                actor.SetVisibility(False)

            # If "None" or empty, just hide all
            if not mask_id or mask_id == "None":
                self._active_segmentation_id = None
                self._logger.info("Hiding all masks (None selected)")
                if self._render_window:
                    self._render_window.Render()
                return

            # Show the requested mask
            if mask_id in self._segmentation_masks:
                self._logger.info(f"Found mask '{mask_id}' in storage")
                # Create actor if it doesn't exist
                if mask_id not in self._segmentation_actors:
                    self._logger.info(f"Creating new actor for '{mask_id}'")
                    actor = self._create_mask_actor(mask_id)
                    if actor:
                        self._segmentation_actors[mask_id] = actor
                        self._renderer.AddActor(actor)
                        self._logger.info(f"Actor created and added to renderer")
                    else:
                        self._logger.error(f"Failed to create actor for '{mask_id}'")
                        return

                # Make visible
                if mask_id in self._segmentation_actors:
                    self._segmentation_actors[mask_id].SetVisibility(True)
                    self._active_segmentation_id = mask_id

                    # Refresh view
                    if self._render_window:
                        self._render_window.Render()

                    self._logger.info(f"Showing segmentation mask: {mask_id}")
                else:
                    self._logger.error(f"Mask '{mask_id}' not found after creation attempt")
            else:
                self._logger.warning(f"Mask '{mask_id}' not found in storage")

        except Exception as e:
            self._logger.error(f"Error showing segmentation mask {mask_id}: {e}", exc_info=True)

    def _create_mask_actor(self, mask_id: str):
        """
        Create a VTK actor for a segmentation mask using Marching Cubes.

        Args:
            mask_id: ID of the mask to create actor for

        Returns:
            vtkActor or None if failed
        """
        try:
            self._logger.info(f"_create_mask_actor for '{mask_id}'")

            if not VTK_AVAILABLE:
                self._logger.error("VTK not available")
                return None

            mask_data = self._segmentation_masks.get(mask_id)
            color = self._segmentation_colors.get(mask_id, (1.0, 0.0, 1.0))

            self._logger.info(f"   Mask data: {mask_data.shape if mask_data is not None else 'None'}")
            self._logger.info(f"   Color: {color}")

            if mask_data is None:
                self._logger.error(f"No mask data found for {mask_id}")
                return None

            # Convert numpy array to VTK image data
            self._logger.info(f"   Converting numpy to VTK image...")
            vtk_mask = self._numpy_to_vtk_image(mask_data)

            if not vtk_mask:
                self._logger.error(f"Failed to convert to VTK image")
                return None

            self._logger.info(f"   VTK image created: {vtk_mask.GetDimensions()}")

            # Apply Marching Cubes to extract surface where mask > 0
            self._logger.info(f"   Applying Marching Cubes...")
            marching_cubes = vtk.vtkMarchingCubes()
            marching_cubes.SetInputData(vtk_mask)
            marching_cubes.SetValue(0, 0.5)  # Iso-value at 0.5 (for boolean masks)
            marching_cubes.ComputeNormalsOn()
            marching_cubes.Update()

            num_points = marching_cubes.GetOutput().GetNumberOfPoints()
            self._logger.info(f"   Marching Cubes result: {num_points} points")

            # Check if surface was generated
            if num_points == 0:
                self._logger.warning(f"No surface generated for mask {mask_id} (all zeros or too sparse)")
                return None

            # Smooth the surface for better visualization
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(marching_cubes.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.SetRelaxationFactor(0.1)
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOn()
            smoother.Update()

            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()

            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Set color and properties
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetOpacity(0.6)  # Semi-transparent
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(20)

            self._logger.info(f"Created 3D actor for mask '{mask_id}' with {marching_cubes.GetOutput().GetNumberOfPoints()} points")

            return actor

        except Exception as e:
            self._logger.error(f"Error creating mask actor for {mask_id}: {e}")
            return None

    def _numpy_to_vtk_image(self, numpy_array: np.ndarray):
        """
        Convert numpy 3D array to vtkImageData.
        CRITICAL: Uses the SAME spacing and origin as the base volume image to avoid misalignment.

        Args:
            numpy_array: 3D numpy array

        Returns:
            vtkImageData or None if failed
        """
        try:
            if not VTK_AVAILABLE:
                return None

            from vtk.util import numpy_support

            # Convert boolean to uint8
            if numpy_array.dtype == bool:
                numpy_array = numpy_array.astype(np.uint8)

            # Create VTK image data - SAME WAY AS BASE IMAGE
            vtk_image = vtk.vtkImageData()

            # Set dimensions: depth, height, width -> width, height, depth (VTK order)
            if len(numpy_array.shape) == 3:
                depth, height, width = numpy_array.shape
                vtk_image.SetDimensions(width, height, depth)
            else:
                height, width = numpy_array.shape
                vtk_image.SetDimensions(width, height, 1)

            # Convert numpy array to VTK array - NO TRANSPOSE, just flattin (same as base image)
            flat_data = numpy_array.flatten()
            vtk_array = numpy_support.numpy_to_vtk(
                num_array=flat_data,
                deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR
            )

            vtk_image.GetPointData().SetScalars(vtk_array)

            # CRITICAL FIX: Use the EXACT same spacing and origin as the base volume
            # This prevents the "repeated grid" artifact
            if hasattr(self, '_vtk_image_data') and self._vtk_image_data:
                # Copy spacing from base volume image
                base_spacing = self._vtk_image_data.GetSpacing()
                vtk_image.SetSpacing(base_spacing[0], base_spacing[1], base_spacing[2])

                # Copy origin from base volume image
                base_origin = self._vtk_image_data.GetOrigin()
                vtk_image.SetOrigin(base_origin[0], base_origin[1], base_origin[2])

                self._logger.info(f"Mask VTK: spacing={base_spacing}, origin={base_origin}")
            else:
                # Fallback: use current image spacing and origin
                if self._current_image and hasattr(self._current_image, 'spacing'):
                    spacing = self._current_image.spacing
                    if hasattr(spacing, 'x') and hasattr(spacing, 'y') and hasattr(spacing, 'z'):
                        vtk_image.SetSpacing(spacing.x, spacing.y, spacing.z)
                    else:
                        vtk_image.SetSpacing(1.0, 1.0, 1.0)
                else:
                    vtk_image.SetSpacing(1.0, 1.0, 1.0)

                # Set origin
                if self._current_image and hasattr(self._current_image, 'origin') and self._current_image.origin:
                    vtk_image.SetOrigin(self._current_image.origin[0], self._current_image.origin[1], self._current_image.origin[2])
                else:
                    vtk_image.SetOrigin(0.0, 0.0, 0.0)

            return vtk_image

        except Exception as e:
            self._logger.error(f"Error converting numpy to VTK image: {e}")
            return None

    def _load_volume_data(self):
        """Load volume data into VTK pipeline."""
        if not self._current_image or not VTK_AVAILABLE:
            return
        
        try:
            # Clear previous rendering
            self._clear_scene()
            
            # Create VTK image data from medical image
            vtk_image_data = self._create_vtk_image_data(self._current_image)
            
            # Get scalar range for slider adjustment
            scalar_range = vtk_image_data.GetScalarRange()
            if scalar_range[1] > 0:
                self._iso_slider.setMaximum(int(scalar_range[1]))
                
                self._iso_slider.setValue(int(scalar_range[1] * 0.3))  # Start at 30% of max
                self._iso_value = self._iso_slider.value()
                self._iso_label.setText(f"ISO Value: {self._iso_value}")
            
            # Store the image data for rendering
            self._vtk_image_data = vtk_image_data

            # FIX: Recreate mask actors with updated spacing (for multi-sequence support)
            # This ensures masks scale properly when switching between T2W/ADC/DWI
            self._recreate_mask_actors()

            # Reset camera and render
            self._reset_camera_with_proper_bounds()
            self._render_volume()
            
            # Enable render button
            self._render_button.setEnabled(True)
            
            # Emit signal
            self.volume_loaded.emit(self._current_image.series_instance_uid)
            
            self._logger.debug("Volume data loaded successfully")
            
        except Exception as e:
            self._logger.error(f"Error loading volume data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load volume data:\n{str(e)}")
    
    def _create_vtk_image_data(self, image: MedicalImage) -> vtk.vtkImageData:
        """Convert medical image to VTK image data with format standardization."""
        import numpy as np
        from vtk.util import numpy_support
        
        image_data = image.image_data.copy()  # Make a copy to avoid modifying original
        
        
        # STANDARDIZATION: Normalize intensity ranges by modality
        if hasattr(image, 'modality'):
            if image.modality.value == 'MRI':
                # Normalize MRI to consistent range 0-1000 for standardized visualization
                original_range = (float(np.min(image_data)), float(np.max(image_data)))
                
                # Clip outliers and normalize
                percentile_99 = np.percentile(image_data, 99)
                image_data = np.clip(image_data, 0, percentile_99)
                
                if np.max(image_data) > 0:
                    image_data = (image_data / np.max(image_data) * 1000.0).astype(np.float32)
                
                self._logger.info(f"MRI intensity normalization: {original_range} -> (0, {np.max(image_data):.1f})")
                
            elif image.modality.value == 'CT':
                # CT typically has standard HU range, minimal normalization needed
                image_data = np.clip(image_data, -1000, 3000).astype(np.float32)
                self._logger.info(f"CT intensity clipping applied: range ({np.min(image_data):.1f}, {np.max(image_data):.1f})")
        
        # Create VTK image data
        vtk_image = vtk.vtkImageData()
        
        if len(image_data.shape) == 3:
            depth, height, width = image_data.shape
            vtk_image.SetDimensions(width, height, depth)
        else:
            height, width = image_data.shape
            vtk_image.SetDimensions(width, height, 1)
        
        # Set spacing with NIfTI stretch correction
        spacing = image.spacing
        corrected_spacing_x = spacing.x
        corrected_spacing_y = spacing.y
        corrected_spacing_z = spacing.z
        
        # NIFTI STRETCH CORRECTION: Auto-detect and fix common spacing issues (siempre activo)
        if hasattr(image, 'dicom_metadata') and image.dicom_metadata:
            image_format = image.dicom_metadata.get('format', '')
            
            if image_format == 'NIfTI':
                # Common NIfTI spacing issues and corrections
                max_spacing = max(spacing.x, spacing.y, spacing.z)
                min_spacing = min(spacing.x, spacing.y, spacing.z)
                
                # Check for excessive anisotropy (stretching indicator)
                anisotropy_ratio = max_spacing / min_spacing if min_spacing > 0 else 1.0
                
                if anisotropy_ratio > 3.0:  # High anisotropy detected
                    self._logger.warning(f"High anisotropy detected in NIfTI: {anisotropy_ratio:.2f}")
                    self._logger.warning(f"Original spacing: X={spacing.x:.3f}, Y={spacing.y:.3f}, Z={spacing.z:.3f}")
                    
                    # Common correction: Z-spacing oftin corrupted in MHA->NIfTI conversion
                    if spacing.z > spacing.x * 2 or spacing.z > spacing.y * 2:
                        # Z-spacing too large, likely corrupted - use average of X,Y
                        corrected_spacing_z = (spacing.x + spacing.y) / 2.0
                        self._logger.info(f"Correcting Z-spacing: {spacing.z:.3f} -> {corrected_spacing_z:.3f}")
                        
                    elif spacing.z < spacing.x / 2 or spacing.z < spacing.y / 2:
                        # Z-spacing too small - use reasonable medical default
                        corrected_spacing_z = max(spacing.x, spacing.y) * 0.8
                        self._logger.info(f"Correcting small Z-spacing: {spacing.z:.3f} -> {corrected_spacing_z:.3f}")
                        
                    # Additional checks for X,Y spacing
                    if spacing.x > spacing.y * 3:
                        corrected_spacing_x = spacing.y
                        self._logger.info(f"Correcting X-spacing: {spacing.x:.3f} -> {corrected_spacing_x:.3f}")
                    elif spacing.y > spacing.x * 3:
                        corrected_spacing_y = spacing.x  
                        self._logger.info(f"Correcting Y-spacing: {spacing.y:.3f} -> {corrected_spacing_y:.3f}")
                
                # Log final corrected spacing
                if (corrected_spacing_x != spacing.x or corrected_spacing_y != spacing.y or corrected_spacing_z != spacing.z):
                    self._logger.info(f"Final corrected spacing: X={corrected_spacing_x:.3f}, Y={corrected_spacing_y:.3f}, Z={corrected_spacing_z:.3f}")
        
        vtk_image.SetSpacing(corrected_spacing_x, corrected_spacing_y, corrected_spacing_z)
        
        # Set origin using real medical image origin if available
        if hasattr(image, 'origin') and image.origin:
            vtk_image.SetOrigin(image.origin[0], image.origin[1], image.origin[2])
            self._logger.info(f"Using medical origin: ({image.origin[0]:.2f}, {image.origin[1]:.2f}, {image.origin[2]:.2f})")
        else:
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
            self._logger.info("Using default origin: (0.0, 0.0, 0.0)")
        
        # Convert numpy data to VTK
        flat_data = image_data.flatten()
        vtk_array = numpy_support.numpy_to_vtk(flat_data)
        vtk_image.GetPointData().SetScalars(vtk_array)
        
        # Log final data range for debugging
        final_range = vtk_image.GetScalarRange()
        self._logger.debug(f"Final VTK data range: ({final_range[0]:.1f}, {final_range[1]:.1f})")
        
        return vtk_image
    
    def _render_volume(self):
        """Render the volume based on selected mode."""
        if not hasattr(self, '_vtk_image_data') or not VTK_AVAILABLE:
            return
        
        try:
            if self._surface_radio.isChecked():
                self._render_iso_surface()
            else:
                self._render_ray_casting()
                
        except Exception as e:
            self._logger.error(f"Error rendering volume: {e}")
            QMessageBox.critical(self, "Rendering Error", f"Failed to render volume:\n{str(e)}")
    
    def _render_iso_surface(self):
        """Render iso-surface using marching cubes."""
        # Store current camera position
        camera = self._renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        
        # Create marching cubes filter
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(self._vtk_image_data)
        marching_cubes.SetValue(0, self._iso_value)
        marching_cubes.ComputeNormalsOn()
        marching_cubes.ComputeGradientsOn()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marching_cubes.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set surface properties for medical visualization
        actor.GetProperty().SetColor(1.0, 0.9, 0.8)  # Bone-like color
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(30)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetAmbient(0.2)
        
        # Clear scene and add actor
        self._clear_scene()
        self._renderer.AddActor(actor)
        self._current_actor = actor
        
        # Restore camera position or reset if first time
        if position == (0, 0, 0):
            self._reset_camera_with_proper_bounds()
        else:
            camera.SetPosition(position)
            camera.SetFocalPoint(focal_point)

        self._render_window.Render()

    def _render_ray_casting(self):
        """Render using GPU-accelerated ray casting."""
        # Store current camera position
        camera = self._renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        
        # Create GPU volume ray cast mapper
        ray_casting_mapper = vtk.vtkGPUVolumeRayCastMapper()
        ray_casting_mapper.SetInputData(self._vtk_image_data)
        
        # Optimize for quality
        ray_casting_mapper.SetSampleDistance(0.5)
        ray_casting_mapper.SetAutoAdjustSampleDistances(True)
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        
        # Set up transfer functions
        volume_property.SetColor(self._create_color_transfer_function())
        volume_property.SetScalarOpacity(self._create_scalar_opacity_function())
        volume_property.SetGradientOpacity(self._create_gradient_opacity_function())
        
        # Configure rendering properties
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        
        # Apply lighting values (valores by defecto optimizados)
        ambient_val = 0.2  # Default ambient
        diffuse_val = 0.7  # Default diffuse
        spec_val = 0.3     # Default specular

        volume_property.SetAmbient(ambient_val)
        volume_property.SetDiffuse(diffuse_val)
        volume_property.SetSpecular(spec_val)
        volume_property.SetSpecularPower(50)
        
        # Create volume
        volume = vtk.vtkVolume()
        volume.SetMapper(ray_casting_mapper)
        volume.SetProperty(volume_property)
        
        # Clear scene and add volume
        self._clear_scene()
        self._renderer.AddVolume(volume)
        self._current_volume = volume
        
        # Restore camera position or reset if first time
        if position == (0, 0, 0):
            self._reset_camera_with_proper_bounds()
        else:
            camera.SetPosition(position)
            camera.SetFocalPoint(focal_point)

        self._render_window.Render()

    def _create_color_transfer_function(self):
        """Create medical color transfer function based on selected scheme."""
        volume_color = vtk.vtkColorTransferFunction()
        
        # Get selected color scheme
        selected_scheme = self._color_scheme_combo.currentText()
        
        # Get data range for normalization
        data_range = self._vtk_image_data.GetScalarRange()
        min_val, max_val = data_range
        
        # Use standardized range for MRI (0-1000) if we normalized it
        if hasattr(self._current_image, 'modality') and self._current_image.modality.value == 'MRI':
            min_val, max_val = 0, 1000
        
        # Medical color schemes optimized for radiological diagnosis
        if selected_scheme == "Standard Medical":
            # Standard medical grayscale-to-blue scheme
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.2, 0.2, 0.2, 0.6)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.4, 0.4, 0.6, 0.9)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.7, 0.8, 0.9, 1.0)
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
            
        elif selected_scheme == "Bone/Tissue":
            # Bone and soft tissue optimized colors
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)        # Background: Black
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.1, 0.2, 0.0, 0.0)  # Soft tissue: Dark red
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.3, 0.8, 0.4, 0.2)  # Tissue: Orange
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.6, 1.0, 0.8, 0.6)  # Dense tissue: Cream
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 0.9)                          # Bone: White
            
        elif selected_scheme == "Angiography":
            # Blood vessel visualization
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.2, 0.8, 0.0, 0.0)  # Red for vessels
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.5, 1.0, 0.5, 0.0)  # Orange
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.8, 1.0, 1.0, 0.0)  # Yellow
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
            
        elif selected_scheme == "Rainbow Medical":
            # Full spectrum for detailed tissue differentiation
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)        # Black
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.15, 0.0, 0.0, 1.0) # Blue
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.3, 0.0, 1.0, 1.0)  # Cyan
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.5, 0.0, 1.0, 0.0)  # Green
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.7, 1.0, 1.0, 0.0)  # Yellow
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.85, 1.0, 0.0, 0.0) # Red
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)                          # White
            
        elif selected_scheme == "Grayscale Enhanced":
            # Enhanced grayscale with better contrast
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.1, 0.1, 0.1, 0.1)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.3, 0.4, 0.4, 0.4)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.6, 0.7, 0.7, 0.7)
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
            
        elif selected_scheme == "Hot Iron":
            # Hot iron scale - classic medical visualization
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)        # Black
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.2, 0.5, 0.0, 0.0)  # Dark red
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.4, 1.0, 0.0, 0.0)  # Red
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.6, 1.0, 0.5, 0.0)  # Orange
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.8, 1.0, 1.0, 0.0)  # Yellow
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)                          # White
            
        elif selected_scheme == "PET/CT":
            # PET/CT fusion colors
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.3, 0.0, 0.3, 0.7)  # Blue tissue
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.5, 0.0, 0.7, 0.3)  # Green
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.7, 0.7, 0.7, 0.0)  # Yellow hot spots
            volume_color.AddRGBPoint(max_val, 1.0, 0.0, 0.0)                          # Red hot spots
            
        elif selected_scheme == "Prostate Specific":
            # Optimized for prostate imaging
            volume_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)        # Background
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.2, 0.1, 0.1, 0.4)  # Dark tissue
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.4, 0.3, 0.5, 0.8)  # Prostate tissue
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.6, 0.6, 0.8, 1.0)  # Gland
            volume_color.AddRGBPoint(min_val + (max_val-min_val)*0.8, 0.9, 0.9, 1.0)  # Central zone
            volume_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)                          # High intensity
        
        return volume_color
    
    def _create_scalar_opacity_function(self):
        """Create standardized scalar opacity function based on modality."""
        volume_scalar_opacity = vtk.vtkPiecewiseFunction()
        
        # Use fixed ranges based on modality instead of dynamic ranges
        if hasattr(self._current_image, 'modality'):
            if self._current_image.modality.value == 'MRI':
                # Standardized MRI opacity mapping (0-1000 range)
                # REDUCED VALUES: Ahora el slider tiene rango útil of 0-100% in lugar of 0-1%
                volume_scalar_opacity.AddPoint(0, 0.0)      # Transparent background
                volume_scalar_opacity.AddPoint(100, 0.05)   # Muy transparente (era 0.15)
                volume_scalar_opacity.AddPoint(300, 0.15)   # Transparente (era 0.4)
                volume_scalar_opacity.AddPoint(600, 0.30)   # Semi-transparente (era 0.7)
                volume_scalar_opacity.AddPoint(1000, 0.50)  # Máximo 50% opacidad (era 1.0)
                
            elif self._current_image.modality.value == 'CT':
                # Standardized CT opacity mapping (HU range)
                volume_scalar_opacity.AddPoint(0, 0.0)
                volume_scalar_opacity.AddPoint(100, 0.1)
                volume_scalar_opacity.AddPoint(500, 0.3)
                volume_scalar_opacity.AddPoint(1000, 0.8)
                volume_scalar_opacity.AddPoint(3000, 1.0)
                
            else:
                # Fallback to dynamic range for unknown modalities
                data_range = self._vtk_image_data.GetScalarRange()
                min_val, max_val = data_range
                range_span = max_val - min_val
                
                volume_scalar_opacity.AddPoint(min_val, 0.0)
                volume_scalar_opacity.AddPoint(min_val + range_span * 0.1, 0.0)
                volume_scalar_opacity.AddPoint(min_val + range_span * 0.3, 0.2)
                volume_scalar_opacity.AddPoint(min_val + range_span * 0.5, 0.4)
                volume_scalar_opacity.AddPoint(min_val + range_span * 0.8, 0.8)
                volume_scalar_opacity.AddPoint(max_val, 0.9)
        else:
            # Fallback to dynamic range if no modality info
            data_range = self._vtk_image_data.GetScalarRange()
            min_val, max_val = data_range
            range_span = max_val - min_val
            
            volume_scalar_opacity.AddPoint(min_val, 0.0)
            volume_scalar_opacity.AddPoint(min_val + range_span * 0.1, 0.0)
            volume_scalar_opacity.AddPoint(min_val + range_span * 0.3, 0.2)
            volume_scalar_opacity.AddPoint(min_val + range_span * 0.5, 0.4)
            volume_scalar_opacity.AddPoint(min_val + range_span * 0.8, 0.8)
            volume_scalar_opacity.AddPoint(max_val, 0.9)
        
        return volume_scalar_opacity
    
    def _create_gradient_opacity_function(self):
        """Create gradient opacity function for edge enhancement."""
        volume_gradient_opacity = vtk.vtkPiecewiseFunction()
        
        # Enhance edges and boundaries
        volume_gradient_opacity.AddPoint(0, 0.0)
        volume_gradient_opacity.AddPoint(10, 0.1)
        volume_gradient_opacity.AddPoint(50, 0.4)
        volume_gradient_opacity.AddPoint(100, 0.8)
        volume_gradient_opacity.AddPoint(200, 1.0)
        
        return volume_gradient_opacity
    
    def _reset_camera_with_proper_bounds(self):
        """Reset camera to show full volume properly."""
        if not hasattr(self, '_vtk_image_data'):
            return
        
        # Get data bounds
        bounds = self._vtk_image_data.GetBounds()
        
        # Calculate center and size
        center = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0
        ]
        
        # Calculate maximum dimension for proper distance
        max_dim = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        
        # Set up camera for optimal viewing
        camera = self._renderer.GetActiveCamera()
        
        # Position camera at appropriate distance
        distance = max_dim * 2.5
        camera.SetPosition(center[0], center[1], center[2] + distance)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, -1, 0)
        
        # Reset camera clipping range
        self._renderer.ResetCameraClippingRange()
    
    def _clear_scene(self):
        """Clear volume/actor from scene, preserving segmentation masks."""
        if self._renderer:
            # Remove only volume and surface actor, NOT segmentation masks
            if self._current_volume:
                self._renderer.RemoveVolume(self._current_volume)
            if self._current_actor:
                self._renderer.RemoveActor(self._current_actor)

            self._current_volume = None
            self._current_actor = None

            # DON'T use RemoveAllViewProps() - it deletes mask actors too
    
    # Event handlers
    def _on_load_volume(self):
        """Handle load volume button click."""
        self._load_volume_data()
    
    def _on_render_volume(self):
        """Handle render button click."""
        self._render_volume()
    
    def _on_iso_value_changed(self):
        """Handle ISO value slider change."""
        self._iso_value = self._iso_slider.value()
        self._iso_label.setText(f"ISO Value: {self._iso_value}")

        if self._surface_radio.isChecked() and self._realtime_check.isChecked():
            self._render_iso_surface()

        self.parameter_changed.emit("iso_value", self._iso_value)
    
    def _on_rendering_mode_changed(self):
        """Handle rendering mode change."""
        if self._surface_radio.isChecked():
            mode = "surface"
            # Show ISO controls, hide lighting
            self._iso_slider.show()
            self._iso_label.show()
        else:
            mode = "raycast"
            # Show all controls
            self._iso_slider.hide()
            self._iso_label.hide()

        if self._realtime_check.isChecked() and hasattr(self, '_vtk_image_data'):
            self._render_volume()

        self.rendering_mode_changed.emit(mode)
    
    def _on_color_scheme_changed(self):
        """Handle color scheme change for medical visualization."""
        if hasattr(self, '_vtk_image_data') and self._vtk_image_data:
            # Store active mask before re-rendering
            active_mask = self._active_segmentation_id

            # Update the color scheme and re-render
            self._render_volume()

            # Restore active mask after re-render
            if active_mask and active_mask in self._segmentation_actors:
                self.show_segmentation_mask(active_mask)

            self._logger.info(f"Color scheme changed to: {self._color_scheme_combo.currentText()}")
    
    # Métodos of aspect ratio y lighting eliminados (auto-corrección siempre activa, lighting usa defaults)

    def _on_mask_opacity_changed(self):
        """Handle mask opacity slider change."""
        opacity = self._mask_opacity_slider.value() / 100.0
        self._mask_opacity_label.setText(f"{self._mask_opacity_slider.value()}%")

        # Update opacity for all active segmentation actors
        for mask_id, actor in self._segmentation_actors.items():
            if actor:
                actor.GetProperty().SetOpacity(opacity)

        # Refresh view
        if self._render_window:
            self._render_window.Render()

        self.parameter_changed.emit("mask_opacity", opacity)

    def _on_volume_opacity_changed(self):
        """Handle volume opacity slider change."""
        opacity = self._volume_opacity_slider.value() / 100.0
        self._volume_opacity_label.setText(f"{self._volume_opacity_slider.value()}%")

        # Update volume opacity if in raycast mode
        if self._current_volume and self._raycast_radio.isChecked():
            # Get volume property
            volume_property = self._current_volume.GetProperty()

            # Create NEW opacity function from scratch (don't modify existing)
            new_opacity_func = vtk.vtkPiecewiseFunction()

            # Get original opacity points based on modality (VALORES REDUCIDOS)
            if hasattr(self, '_current_image') and hasattr(self._current_image, 'modality'):
                if self._current_image.modality.value == 'MRI':
                    # REDUCIDO: Máximo 50% opacidad base for rango útil 0-100%
                    points = [(0, 0.0), (100, 0.05), (300, 0.15), (600, 0.30), (1000, 0.50)]
                elif self._current_image.modality.value == 'CT':
                    points = [(0, 0.0), (100, 0.1), (500, 0.3), (1000, 0.8), (3000, 1.0)]
                else:
                    # Fallback: generic points
                    data_range = self._vtk_image_data.GetScalarRange() if hasattr(self, '_vtk_image_data') else (0, 1000)
                    min_val, max_val = data_range
                    range_span = max_val - min_val
                    points = [
                        (min_val, 0.0),
                        (min_val + range_span * 0.1, 0.0),
                        (min_val + range_span * 0.3, 0.2),
                        (min_val + range_span * 0.5, 0.4),
                        (min_val + range_span * 0.8, 0.8),
                        (max_val, 0.9)
                    ]
            else:
                # Fallback if no modality info (valores reducidos)
                points = [(0, 0.0), (100, 0.05), (300, 0.15), (600, 0.30), (1000, 0.50)]

            # Apply opacity multiplier to ORIGINAL values
            for value, original_opacity in points:
                new_opacity_func.AddPoint(value, original_opacity * opacity)

            # Replace opacity function (don't modify, replace)
            volume_property.SetScalarOpacity(new_opacity_func)

            # Refresh view
            if self._render_window:
                self._render_window.Render()

        self.parameter_changed.emit("volume_opacity", opacity)

    def _on_show_volume_toggled(self):
        """Handle show/hide volume checkbox."""
        show = self._show_volume_check.isChecked()

        if self._current_volume:
            self._current_volume.SetVisibility(show)

        if self._current_actor:
            self._current_actor.SetVisibility(show)

        # Refresh view
        if self._render_window:
            self._render_window.Render()

        self.parameter_changed.emit("volume_visible", show)

    def _on_reset_camera(self):
        """Handle reset camera button click."""
        if hasattr(self, '_vtk_image_data'):
            self._reset_camera_with_proper_bounds()

            # Refresh view
            if self._render_window:
                self._render_window.Render()

            self._logger.info("Camera reset to optimal view")

    # Styling methods
    def _setup_button_style(self, button: QPushButton, color: str):
        """Setup button styling."""
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border-radius: 8px;
                border: none;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
                color: #adb5bd;
            }}
        """)
    
    def _setup_label_style(self, label: QLabel):
        """Setup label styling."""
        font = QFont("Segoe UI", 10)
        label.setFont(font)
        label.setStyleSheet("color: rgb(255, 255, 255);")
    
    def _setup_radio_style(self, radio: QRadioButton):
        """Setup radio button styling."""
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        radio.setFont(font)
        radio.setStyleSheet("color: rgb(255, 255, 255);")
    
    def _setup_checkbox_style(self, checkbox: QCheckBox):
        """Setup checkbox styling."""
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        checkbox.setFont(font)
        checkbox.setStyleSheet("color: rgb(255, 255, 255);")
    
    def _lighten_color(self, color: str) -> str:
        """Lightin a hex color."""
        if color == "#405cf5":
            return "#5a6cf7"
        elif color == "#28a745":
            return "#34ce57"
        return color
    
    def _darken_color(self, color: str) -> str:
        """Darkin a hex color."""
        if color == "#405cf5":
            return "#2d47d9"
        elif color == "#28a745":
            return "#1e7e34"
        return color
    
    def _apply_medical_theme(self):
        """Apply medical dark theme to the widget."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1b1d23;
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: none;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #405cf5;
                border: none;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #fff;
                border: none;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #405cf5;
                border: none;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::groove:vertical {
                border: none;
                background: white;
                width: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:vertical {
                background: #405cf5;
                border: none;
                width: 10px;
                border-radius: 4px;
            }
            QSlider::add-page:vertical {
                background: #fff;
                border: none;
                width: 10px;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: #405cf5;
                border: none;
                height: 18px;
                margin: 0 -2px;
                border-radius: 9px;
            }
        """)

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        if self._vtk_widget and self._render_window:
            # Update VTK render window size
            size = self._viewport_widget.size()
            self._render_window.SetSize(size.width(), size.height())
            
            # Re-render if we have data
            if hasattr(self, '_vtk_image_data'):
                self._render_window.Render()
    
    def showEvent(self, event):
        """Handle show events."""
        super().showEvent(event)
        if self._vtk_widget and VTK_AVAILABLE:
            try:
                self._vtk_widget.Initialize()
                self._vtk_widget.Start()
            except Exception as e:
                self._logger.warning(f"VTK initialization warning: {e}")
    
    def __del__(self):
        """
        MEMORY MANAGEMENT: Ensure Qt and VTK resources are cleaned up.
        """
        self.cleanup()
    
    def cleanup(self):
        """
        CRITICAL: Explicit cleanup of Qt and VTK resources to prevent memory leaks.
        
        QVTKRenderWindowInteractor and VTK objects can create circular references
        with Qt parent-child relationships. Explicit cleanup prevents memory leaks.
        """
        try:
            self._logger.info("Cleaning up VolumeViewer3D resources...")
            
            # Clean up VTK components first
            if hasattr(self, '_current_actor') and self._current_actor:
                if self._renderer:
                    self._renderer.RemoveActor(self._current_actor)
                self._current_actor = None
            
            if hasattr(self, '_current_volume') and self._current_volume:
                self._current_volume = None
            
            if hasattr(self, '_renderer') and self._renderer:
                self._renderer.RemoveAllViewProps()
                self._renderer = None
            
            # Clean up QVTKRenderWindowInteractor
            if hasattr(self, '_vtk_widget') and self._vtk_widget:
                try:
                    # Stop the interactor
                    if hasattr(self._vtk_widget, 'Finalize'):
                        self._vtk_widget.Finalize()
                    
                    # Remove from layout
                    if self._vtk_widget.parent():
                        layout = self._vtk_widget.parent().layout()
                        if layout:
                            layout.removeWidget(self._vtk_widget)
                    
                    # Delete the widget
                    self._vtk_widget.setParent(None)
                    self._vtk_widget.deleteLater()
                    self._vtk_widget = None
                except:
                    pass
            
            if hasattr(self, '_render_window') and self._render_window:
                try:
                    self._render_window.Finalize()
                    self._render_window = None
                except:
                    pass
            
            # Clear references
            self._current_image = None
            
            self._logger.info("VolumeViewer3D cleanup completed")
            
        except Exception as e:
            # Log error but don't raise during cleanup
            self._logger.error(f"Error during VolumeViewer3D cleanup: {e}")
    
    def closeEvent(self, event):
        """
        Handle widget close event with proper cleanup.
        """
        self.cleanup()
        super().closeEvent(event)