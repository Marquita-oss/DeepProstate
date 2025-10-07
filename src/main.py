#!/usr/bin/env python3
"""
DeepProstate - Clean Architecture Implementation

Main entry point for DeepProstate medical imaging application.
Clean architecture with complete separation of concerns.
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox, QDockWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont

import qdarktheme
from frameworks.infrastructure.ui.themes.medical_theme import create_medical_radiology_theme

from frameworks.infrastructure.di.medical_service_container import (
    MedicalServiceContainer, create_medical_service_container
)
from frameworks.infrastructure.coordination.workflow_orchestrator import WorkflowOrchestrator

from frameworks.infrastructure.ui.main_window import MedicalMainWindow

from frameworks.infrastructure.ui.components.medical_splash_screen import SplashScreenManager

from frameworks.infrastructure.utils.logging_config import setup_medical_logging
from frameworks.infrastructure.utils.startup_validator import MedicalSystemValidator


class DeepProstateApplication:
    """
    DeepProstate v20 - Medical imaging application with Clean Architecture.
    """

    def __init__(self):
        self._qt_application: Optional[QApplication] = None
        self._main_window: Optional[MedicalMainWindow] = None
        self._service_container: Optional[MedicalServiceContainer] = None
        self._workflow_coordinator: Optional[WorkflowOrchestrator] = None
        self._splash_manager: Optional[SplashScreenManager] = None
        self._initialization_successful = False
        self._startup_timestamp = None
        self._logger: Optional[logging.Logger] = None
    
    def run(self) -> int:
        """Execute the medical application."""
        try:
            self._qt_application = QApplication(sys.argv)
            self._setup_qt_application_properties()
            self._splash_manager = SplashScreenManager(
                app_name="DeepProstate v20 - Clean Architecture",
                app_version="20.0.0",
                logo_path="./resources/image/dp2.png"
            )
            self._splash_manager.show_splash(self._qt_application)

            self._splash_manager.update_progress("Setting up medical environment...")
            self._logger = self._setup_medical_environment()

            self._splash_manager.update_progress("Validating medical prerequisites...")
            if not self._validate_medical_prerequisites():
                self._logger.error("Medical prerequisites not met")
                return 1

            self._splash_manager.update_progress("Initializing medical services...")
            if not self._initialize_architectural_components():
                self._logger.error("Component initialization failed")
                return 1

            self._splash_manager.update_progress("Integrating complete system...")
            if not self._integrate_and_validate_system():
                self._logger.error("System integration failed")
                return 1

            self._splash_manager.update_progress("Preparing medical workstation...")
            return self._launch_medical_workstation()

        except Exception as e:
            error_message = f"Critical error in medical application: {e}"
            if self._logger:
                self._logger.critical(error_message)
                self._logger.critical(traceback.format_exc())
            else:
                logging.error(f" {error_message}")
                logging.error(traceback.format_exc())

            return 1

        finally:
            self._cleanup_medical_application()
    
    def _setup_medical_environment(self) -> logging.Logger:
        """Setup medical execution environment with logging and directories."""
        logging.info("Setting up medical environment...")

        logger = setup_medical_logging(
            log_level=logging.INFO,
            console_level=logging.INFO,
            medical_audit=True,
            hipaa_compliant=True,
            log_file="./data/logs/deepprostate_v20.log"
        )

        required_dirs = [
            "./data/logs",
            "./data/images",
            "./data/temp",
            "./data/exports"
        ]

        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        import os
        os.environ['DEEPPROSTATE_VERSION'] = '2.0'
        os.environ['ARCHITECTURE_TYPE'] = 'CLEAN'
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
        os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

        return logger
    
    def _validate_medical_prerequisites(self) -> bool:
        """Validate medical prerequisites for safe operation."""
        self._logger.info("Validating medical prerequisites...")

        try:
            validator = MedicalSystemValidator()

            if not validator.validate_system_resources():
                self._logger.error("Insufficient system resources for medical application")
                return False

            if not validator.validate_medical_dependencies():
                self._logger.error("Missing or incorrect medical dependencies")
                return False

            if not validator.validate_security_configuration():
                self._logger.error("Insufficient medical security configuration")
                return False

            if not validator.validate_medical_connectivity():
                self._logger.warning("Limited medical connectivity - continuing in local mode")

            self._logger.info("Medical prerequisites validated successfully")
            return True

        except Exception as e:
            self._logger.error(f"Error during prerequisite validation: {e}")
            return False
    
    def _initialize_architectural_components(self) -> bool:
        """Initialize architectural components."""
        try:
            self._splash_manager.update_progress("Creating medical services...")
            self._service_container = create_medical_service_container()

            self._splash_manager.update_progress("Creating workflow orchestrator...")
            self._workflow_coordinator = WorkflowOrchestrator(self._service_container)

            self._splash_manager.update_progress("Preparing UI components...")
            return True

        except Exception as e:
            self._logger.error(f"Error initializing components: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
    
    def _integrate_and_validate_system(self) -> bool:
        """Integrate and validate system components."""
        try:
            self._splash_manager.update_progress("Creating main window...")
            self._main_window = MedicalMainWindow(
                service_container=self._service_container,
                workflow_coordinator=self._workflow_coordinator
            )

            if hasattr(self._main_window._ui_components.right_panel, 'image_information_panel'):
                self._main_window.image_information_panel = self._main_window._ui_components.right_panel.image_information_panel
                self._main_window.setup_image_information_connections()
                self._setup_manual_editing_connections()

            self._workflow_coordinator.set_parent_window(self._main_window)

            self._splash_manager.update_progress("Validating integration...")

            assert self._workflow_coordinator._services is not None, "Coordinator without services"
            assert self._main_window._ui_components.central_viewer is not None, "Central viewer not created"

            if self._main_window._ui_components.segmentation_panel is None:
                self._logger.warning("Segmentation panel disabled pending service refactoring")

            assert self._main_window._services is not None, "Main window without services"
            assert self._main_window._coordinator is not None, "Main window without coordinator"
            assert self._main_window._ui_components is not None, "Main window without UI components"

            self._setup_inter_component_communication()

            return True

        except Exception as e:
            self._logger.error(f"Error in system integration: {e}")
            self._logger.error(traceback.format_exc())
            return False
    
    def _setup_inter_component_communication(self) -> None:
        """Setup high-level component communication."""
        self._main_window.application_closing.connect(
            self._on_application_closing_requested
        )

        self._main_window.patient_context_changed.connect(
            self._on_patient_context_changed
        )

        self._workflow_coordinator.workflow_error.connect(
            self._on_critical_workflow_error
        )
        
    def _launch_medical_workstation(self) -> int:
        """Launch medical workstation."""
        try:
            if not self._qt_application:
                raise RuntimeError("QApplication should have been created in initialization")

            self._splash_manager.complete_initialization()
            self._splash_manager.splash_closed.connect(self._show_main_window)

            self._initialization_successful = True
            from datetime import datetime
            self._startup_timestamp = datetime.now()

            self._logger.info("Medical Workstation ready for medical operation")
            return self._qt_application.exec()

        except Exception as e:
            self._logger.error(f"Error during launch: {e}")
            return 1

    def _show_main_window(self) -> None:
        self._main_window.show()
        self._logger.info("Main window shown after splash screen")

    def _setup_qt_application_properties(self) -> None:
        self._qt_application.setApplicationName("DeepProstate v20 - Clean Architecture")
        self._qt_application.setApplicationVersion("20.0.0")
        self._qt_application.setOrganizationName("Medical Imaging Lab")
        self._qt_application.setApplicationDisplayName("DeepProstate v20")

        medical_stylesheet = create_medical_radiology_theme()
        self._qt_application.setStyleSheet(medical_stylesheet)
    
    def _on_application_closing_requested(self) -> None:
        pass

    def _on_patient_context_changed(self, patient_id: str) -> None:
        pass

    def _on_critical_workflow_error(self, workflow_id: str, error_message: str) -> None:
        self._logger.critical(f"Critical error in workflow {workflow_id}: {error_message}")
    def _cleanup_medical_application(self) -> None:
        """Clean shutdown of medical application."""
        if self._logger:
            self._logger.info("Starting medical application cleanup...")

        try:
            if self._service_container:
                self._service_container.shutdown()
                if self._logger:
                    self._logger.info("Medical services closed")

            if self._logger and self._initialization_successful:
                self._logger.info("Medical application closed correctly")
                self._logger.info("Medical session completed - Audit finalized")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Error during cleanup: {e}")
            else:
                logging.error(f"Error during cleanup: {e}")
    def _setup_manual_editing_connections(self) -> None:
        """Setup manual editing panel connections."""
        try:
            image_viewer = self._main_window._ui_components.central_viewer
            manual_panel = None

            if hasattr(self._main_window._ui_components.right_panel, '_panels'):
                manual_panel = self._main_window._ui_components.right_panel._panels.get('manual_editing')

            if not image_viewer or not manual_panel:
                self._logger.warning("Could not connect manual editing panel - components not found")
                return

            manual_panel.measurement_mode_changed.connect(image_viewer.set_measurement_mode)
            manual_panel.clear_measurements_requested.connect(image_viewer.clear_measurements)
            manual_panel.clear_segmentations_requested.connect(image_viewer.clear_segmentations)
            manual_panel.measurement_selected.connect(image_viewer.highlight_measurement)
            manual_panel.measurement_delete_requested.connect(self._on_measurement_delete_requested)

            manual_panel.segmentation_tool_changed.connect(image_viewer.set_segmentation_mode)
            manual_panel.brush_size_changed.connect(image_viewer.update_segmentation_brush_size)

            if hasattr(image_viewer, '_overlay_service'):
                manual_panel.set_overlay_service_reference(image_viewer._overlay_service)

            manual_panel.connect_to_image_viewer(image_viewer)
            image_viewer.connect_manual_editing_panel(manual_panel)

            manual_panel.active_segmentation_changed.connect(image_viewer._on_active_segmentation_changed)

            image_viewer.measurement_created.connect(self._on_measurement_created)

            image_viewer.slice_changed.connect(lambda index: self._on_slice_changed(manual_panel, image_viewer, index))
            image_viewer.view_changed.connect(lambda plane: self._on_view_changed(manual_panel, image_viewer, plane))

            self._update_panel_from_viewer_state(manual_panel, image_viewer)

        except Exception as e:
            self._logger.error(f"Error configuring manual editing connections: {e}")
    
    def _on_measurement_created(self, measurement_data: dict) -> None:
        """
        Handle creation of new measurements from image viewer.

        Args:
            measurement_data: Created measurement data
        """
        measurement_type = measurement_data.get('type', 'unknown')

        try:
            manual_panel = self._get_manual_panel()
            image_viewer = self._get_image_viewer()

            if manual_panel and hasattr(manual_panel, '_measurement_tools'):
                self._route_measurement_to_panel(manual_panel, measurement_data)
                if image_viewer and manual_panel:
                    self._update_panel_from_viewer_state(manual_panel, image_viewer)
            else:
                self._logger.warning(f"Panel not found or invalid. Panel: {manual_panel}")
                if manual_panel:
                    self._logger.warning(f"Panel type: {type(manual_panel).__name__}")
                    self._logger.warning(f"Has _measurement_tools: {hasattr(manual_panel, '_measurement_tools')}")
                

        except Exception as e:
            self._logger.error(f"Error sending measurement to panel: {e}")
            import traceback
            self._logger.error(traceback.format_exc())

    def _route_measurement_to_panel(self, manual_panel, measurement_data: dict) -> None:
        """Route measurement to corresponding panel tool."""
        measurement_type = measurement_data.get('type')

        if not hasattr(manual_panel, '_measurement_tools'):
            self._logger.error("Panel does not have _measurement_tools")
            return
            
        available_tools = list(manual_panel._measurement_tools.keys())
        
        if measurement_type == "distance":
            tool = manual_panel._measurement_tools.get("distance")
            if tool and hasattr(tool, 'add_measurement'):
                points = measurement_data.get('points', [])
                if len(points) >= 2:
                    distance_mm = measurement_data.get('distance_mm', 0)
                    distance_px = measurement_data.get('distance', 0)
                    distance = distance_px if distance_px > 0 else distance_mm
                    tool.add_measurement(points[0], points[1], distance)
                else:
                    self._logger.warning(f"Insufficient points for distance: {len(points)}")
            else:
                self._logger.error(f"Distance tool not valid or missing add_measurement method")
                    
        elif measurement_type == "angle":
            tool = manual_panel._measurement_tools.get("angle")
            if tool and hasattr(tool, 'add_measurement'):
                points = measurement_data.get('points', [])
                if len(points) >= 3:
                    angle = measurement_data.get('angle', 0)
                    tool.add_measurement(points[0], points[1], points[2], angle)
                else:
                    self._logger.warning(f"Insufficient points for angle: {len(points)}")
            else:
                self._logger.error(f"Angle tool not valid or missing add_measurement method")
                    
        elif measurement_type == "roi":
            tool = manual_panel._measurement_tools.get("roi")
            if tool and hasattr(tool, 'add_roi'):
                area = measurement_data.get('area', 0)
                coordinates = {"points": measurement_data.get('points', [])}
                tool.add_roi("rectangle", coordinates, area)
    
    def _on_measurement_delete_requested(self, measurement_data: dict) -> None:
        """
        Handle measurement deletion request from panel.

        Args:
            measurement_data: Measurement data to delete
        """
        try:
            image_viewer = self._get_image_viewer()
            manual_panel = self._get_manual_panel()

            if not image_viewer or not manual_panel:
                self._logger.error("Could not get necessary components to delete measurement")
                return

            success = image_viewer.remove_measurement(measurement_data)

            if success:
                self._update_panel_from_viewer_state(manual_panel, image_viewer)
            else:
                self._logger.warning("Could not remove measurement from canvas")
                
        except Exception as e:
            self._logger.error(f"Error deleting measurement: {e}")
            import traceback

    def _on_slice_changed(self, manual_panel, image_viewer, slice_index: int) -> None:
        """Update measurement panel when slice changes."""
        try:
            self._update_panel_from_viewer_state(manual_panel, image_viewer, slice_index=slice_index)
        except Exception as e:
            self._logger.error(f"Error updating panel on slice change: {e}")
            import traceback

    def _on_view_changed(self, manual_panel, image_viewer, plane: str) -> None:
        """Update measurement panel when view changes."""
        try:
            current_slice = 0
            if hasattr(image_viewer, '_slice_slider'):
                current_slice = image_viewer._slice_slider.value()

            self._update_panel_from_viewer_state(manual_panel, image_viewer, plane=plane, slice_index=current_slice)
        except Exception as e:
            self._logger.error(f"Error updating panel on view change: {e}")
            import traceback

    def _update_panel_from_viewer_state(self, manual_panel, image_viewer, plane: str = None, slice_index: int = None) -> None:
        """Update panel based on current viewer state."""
        active_plane = plane or getattr(image_viewer, '_active_view_plane', 'axial')
        current_slice = slice_index
        
        if current_slice is None and hasattr(image_viewer, '_slice_slider'):
            current_slice = image_viewer._slice_slider.value()
        elif current_slice is None:
            current_slice = 0

        total_slices = 1
        if hasattr(image_viewer, '_slice_slider'):
            total_slices = image_viewer._slice_slider.maximum()

        position_mm = 0.0
        if hasattr(image_viewer, '_current_slice_info'):
            position_mm = image_viewer._current_slice_info.get("position_mm", 0.0)

        # Update slice info in main canvas for measurements
        main_canvas = getattr(image_viewer, '_main_canvas', None)
        if main_canvas and hasattr(main_canvas, 'update_slice_info'):
            main_canvas.update_slice_info(active_plane, current_slice, position_mm)
            if hasattr(main_canvas, '_update_display'):
                main_canvas._update_display()
        
        manual_panel.update_slice_info(
            plane=active_plane,
            index=current_slice,
            total_slices=total_slices,
            position_mm=position_mm
        )
        
        canvas_measurements = {}
        if main_canvas and hasattr(main_canvas, '_measurements_by_slice'):
            canvas_measurements = main_canvas._measurements_by_slice
        
        manual_panel.update_measurements_for_slice(canvas_measurements)

    def _get_manual_panel(self):
        """
        Helper method to get the manual editing panel with consistent error handling.

        Returns:
            ManualEditingPanel instance or None if not found
        """
        try:
            if hasattr(self._main_window, '_ui_components') and \
               hasattr(self._main_window._ui_components, 'right_panel'):
                return self._main_window._ui_components.right_panel._panels.get('manual_editing')
        except AttributeError:
            pass
        return None

    def _get_image_viewer(self):
        """
        Helper method to get the image viewer with consistent error handling.

        Returns:
            ImageViewer instance or None if not found
        """
        try:
            if hasattr(self._main_window, '_ui_components') and \
               hasattr(self._main_window._ui_components, 'central_viewer'):
                return self._main_window._ui_components.central_viewer
        except AttributeError:
            pass
        return None
        
def main():
    medical_app = DeepProstateApplication()
    exit_code = medical_app.run()

    return exit_code

if __name__ == "__main__":
    sys.exit(main())