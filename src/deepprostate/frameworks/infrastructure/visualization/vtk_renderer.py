import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import vtk
from vtk.util import numpy_support
import asyncio
import logging
from PyQt6.QtCore import QMutex, QMutexLocker

from deepprostate.core.domain.entities.medical_image import MedicalImage, ImageSpacing
from deepprostate.core.domain.entities.segmentation import MedicalSegmentation, AnatomicalRegion


class MedicalVTKRenderer:
    def __init__(self, width: int = 800, height: int = 600):
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self._vtk_mutex = QMutex()
        
        self._renderer = vtk.vtkRenderer()
        self._render_window = vtk.vtkRenderWindow()
        self._render_window_interactor = vtk.vtkRenderWindowInteractor()
        
        with QMutexLocker(self._vtk_mutex):
            self._render_window.AddRenderer(self._renderer)
            self._render_window.SetSize(width, height)
            self._render_window_interactor.SetRenderWindow(self._render_window)
        
        self._vtk_objects = {
            'actors': [],
            'volumes': [],
            'mappers': [],
            'filters': [],
            'sources': []
        }
        self._is_destroyed = False
        
        self._setup_medical_interaction_style()
        
        self._setup_medical_lighting()
        
        with QMutexLocker(self._vtk_mutex):
            self._renderer.SetBackground(0.1, 0.1, 0.1) 
        
        self._active_volumes: Dict[str, vtk.vtkVolume] = {}
        self._active_segmentations: Dict[str, vtk.vtkActor] = {}
        self._active_measurements: Dict[str, List[vtk.vtkActor]] = {}
        
        self._modality_presets = self._create_modality_presets()
        
        self._event_callbacks: Dict[str, List[Callable]] = {
            "volume_loaded": [],
            "segmentation_added": [],
            "measurement_created": [],
            "view_changed": []
        }
    
    async def render_volume(
        self,
        image: MedicalImage,
        rendering_mode: str = "mip",  # "mip", "composite", "isosurface"
        opacity_curve: Optional[List[Tuple[float, float]]] = None
    ) -> str:
        try:
            volume_id = f"volume_{image.series_instance_uid}"
            
            vtk_image_data = await self._create_vtk_image_data(image)
            
            volume_mapper = self._create_volume_mapper(
                image.modality, rendering_mode
            )
            volume_mapper.SetInputData(vtk_image_data)
            
            volume_property = self._create_volume_property(
                image, rendering_mode, opacity_curve
            )
            
            with QMutexLocker(self._vtk_mutex):
                volume = vtk.vtkVolume()
                volume.SetMapper(volume_mapper)
                volume.SetProperty(volume_property)
                
                transform = self._create_physical_transform(image.spacing)
                volume.SetUserTransform(transform)
                
                self._renderer.AddVolume(volume)
                self._active_volumes[volume_id] = volume
                
                # Ajustar cámara automáticamente
                self._renderer.ResetCamera()
                
                self._register_vtk_object(volume, 'volumes')
                self._register_vtk_object(volume_mapper, 'mappers')
            
            await self._notify_callbacks("volume_loaded", {
                "volume_id": volume_id,
                "image": image,
                "rendering_mode": rendering_mode
            })
            
            return volume_id
            
        except Exception as e:
            raise VTKRenderingError(f"Error renderizando volumen: {e}") from e
    
    async def add_segmentation_surface(
        self,
        segmentation: MedicalSegmentation,
        image_spacing: ImageSpacing,
        surface_color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 0.6,
        smoothing_iterations: int = 15
    ) -> str:
        try:
            surface_id = f"surface_{segmentation.segmentation_id}"
            
            vtk_mask_data = await self._create_vtk_mask_data(
                segmentation, image_spacing
            )
            
            surface_filter = vtk.vtkMarchingCubes()
            surface_filter.SetInputData(vtk_mask_data)
            surface_filter.SetValue(0, 0.5)  # Isovalor for superficie
            surface_filter.Update()
            
            if smoothing_iterations > 0:
                smoother = vtk.vtkSmoothPolyDataFilter()
                smoother.SetInputConnection(surface_filter.GetOutputPort())
                smoother.SetNumberOfIterations(smoothing_iterations)
                smoother.SetRelaxationFactor(0.1)
                smoother.FeatureEdgeSmoothingOff()
                smoother.BoundarySmoothingOn()
                smoother.Update()
                polydata = smoother.GetOutput()
            else:
                polydata = surface_filter.GetOutput()
            
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ConsistencyOn()
            normals.SplittingOff()
            normals.Update()
            
            with QMutexLocker(self._vtk_mutex):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())
                mapper.ScalarVisibilityOff()
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                property = actor.GetProperty()
                
                if surface_color is None:
                    surface_color = self._get_anatomical_color(segmentation.anatomical_region)
                
                property.SetColor(surface_color)
                property.SetOpacity(opacity)
                property.SetSpecular(0.6)
                property.SetSpecularPower(30)
                
                self._renderer.AddActor(actor)
                self._active_segmentations[surface_id] = actor
                
                self._register_vtk_object(actor, 'actors')
                self._register_vtk_object(mapper, 'mappers')
                self._register_vtk_object(surface_filter, 'filters')
                self._register_vtk_object(normals, 'filters')
            
            await self._notify_callbacks("segmentation_added", {
                "surface_id": surface_id,
                "segmentation": segmentation,
                "color": surface_color,
                "opacity": opacity
            })
            
            return surface_id
            
        except Exception as e:
            raise VTKRenderingError(f"Error añadiendo superficie of segmentación: {e}") from e
    
    async def create_3d_measurement(
        self,
        points: List[Tuple[float, float, float]],
        measurement_type: str = "distance",  # "distance", "angle", "volume"
        color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        font_size: int = 14
    ) -> str:
        try:
            measurement_id = f"measurement_{len(self._active_measurements)}"
            measurement_actors = []
            
            if measurement_type == "distance" and len(points) >= 2:
                line_actor = self._create_line_actor(points[0], points[1], color)
                measurement_actors.append(line_actor)
                
                p1, p2 = np.array(points[0]), np.array(points[1])
                distance = np.linalg.norm(p2 - p1)
                
                midpoint = (p1 + p2) / 2
                text_actor = self._create_text_actor(
                    f"{distance:.2f} mm", midpoint, color, font_size
                )
                measurement_actors.append(text_actor)
            
            elif measurement_type == "angle" and len(points) >= 3:
                line1_actor = self._create_line_actor(points[1], points[0], color)
                line2_actor = self._create_line_actor(points[1], points[2], color)
                measurement_actors.extend([line1_actor, line2_actor])
                
                v1 = np.array(points[0]) - np.array(points[1])
                v2 = np.array(points[2]) - np.array(points[1])
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle_degrees = np.degrees(angle)
                
                text_actor = self._create_text_actor(
                    f"{angle_degrees:.1f}°", points[1], color, font_size
                )
                measurement_actors.append(text_actor)
            
            with QMutexLocker(self._vtk_mutex):
                for actor in measurement_actors:
                    self._renderer.AddActor(actor)
                    self._register_vtk_object(actor, 'actors')
                
                self._active_measurements[measurement_id] = measurement_actors
            
            await self._notify_callbacks("measurement_created", {
                "measurement_id": measurement_id,
                "type": measurement_type,
                "points": points,
                "color": color
            })
            
            return measurement_id
            
        except Exception as e:
            raise VTKRenderingError(f"Error creando medición 3D: {e}") from e
    
    async def set_camera_view(
        self,
        view_type: str = "axial",  # "axial", "sagittal", "coronal", "oblique"
        custom_position: Optional[Tuple[float, float, float]] = None,
        custom_focal_point: Optional[Tuple[float, float, float]] = None
    ) -> None:
        with QMutexLocker(self._vtk_mutex):
            camera = self._renderer.GetActiveCamera()
            
            bounds = self._renderer.ComputeVisiblePropBounds()
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
            
            max_dimension = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4]
            )
            camera_distance = max_dimension * 2.5
        
            if custom_position and custom_focal_point:
                camera.SetPosition(custom_position)
                camera.SetFocalPoint(custom_focal_point)
            else:
                if view_type == "axial":
                    camera.SetPosition(center[0], center[1], center[2] + camera_distance)
                    camera.SetViewUp(0, 1, 0)
                elif view_type == "sagittal":
                    camera.SetPosition(center[0] + camera_distance, center[1], center[2])
                    camera.SetViewUp(0, 0, 1)
                elif view_type == "coronal":
                    camera.SetPosition(center[0], center[1] + camera_distance, center[2])
                    camera.SetViewUp(0, 0, 1)
                else:  # oblique
                    camera.SetPosition(
                        center[0] + camera_distance * 0.7,
                        center[1] + camera_distance * 0.7,
                        center[2] + camera_distance * 0.7
                    )
                    camera.SetViewUp(0, 0, 1)
                
                camera.SetFocalPoint(center)
            
            self._renderer.ResetCameraClippingRange()
            
            camera_position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
        
        await self._notify_callbacks("view_changed", {
            "view_type": view_type,
            "camera_position": camera_position,
            "focal_point": focal_point
        })
    
    def remove_volume(self, volume_id: str) -> bool:
        with QMutexLocker(self._vtk_mutex):
            if volume_id in self._active_volumes:
                volume = self._active_volumes[volume_id]
                self._renderer.RemoveVolume(volume)
                del self._active_volumes[volume_id]
                return True
        return False
    
    def remove_segmentation(self, surface_id: str) -> bool:
        with QMutexLocker(self._vtk_mutex):
            if surface_id in self._active_segmentations:
                actor = self._active_segmentations[surface_id]
                self._renderer.RemoveActor(actor)
                del self._active_segmentations[surface_id]
                return True
        return False
    
    def remove_measurement(self, measurement_id: str) -> bool:
        with QMutexLocker(self._vtk_mutex):
            if measurement_id in self._active_measurements:
                actors = self._active_measurements[measurement_id]
                for actor in actors:
                    self._renderer.RemoveActor(actor)
                del self._active_measurements[measurement_id]
                return True
        return False
    
    def clear_all(self) -> None:
        with QMutexLocker(self._vtk_mutex):
            self._renderer.RemoveAllViewProps()
            self._active_volumes.clear()
            self._active_segmentations.clear()
            self._active_measurements.clear()
    
    def render(self) -> None:
        with QMutexLocker(self._vtk_mutex):
            self._render_window.Render()
    
    def start_interaction(self) -> None:
        with QMutexLocker(self._vtk_mutex):
            self._render_window_interactor.Start()
    
    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self._event_callbacks:
            self._event_callbacks[event_type].append(callback)
    
    async def _create_vtk_image_data(self, image: MedicalImage) -> vtk.vtkImageData:
        image_data = image.image_data.copy()  
        if hasattr(image, 'dicom_metadata') and image.dicom_metadata:
            image_format = image.dicom_metadata.get('format', '')
            self._logger.debug(f"VTK: Processing {image_format} format")
            self._logger.info("VTK: Direction matrices are equivalent - no orientation correction needed")
        
        if hasattr(image, 'modality'):
            if image.modality.value == 'MRI':
                original_range = (float(np.min(image_data)), float(np.max(image_data)))
                percentile_99 = np.percentile(image_data, 99)
                image_data = np.clip(image_data, 0, percentile_99)
                
                if np.max(image_data) > 0:
                    image_data = (image_data / np.max(image_data) * 1000.0).astype(np.float32)
                
                self._logger.info(f"VTK MRI intensity normalization: {original_range} -> (0, {np.max(image_data):.1f})")
                
            elif image.modality.value == 'CT':
                image_data = np.clip(image_data, -1000, 3000).astype(np.float32)
                self._logger.info(f"VTK CT intensity clipping applied: range ({np.min(image_data):.1f}, {np.max(image_data):.1f})")
        
        vtk_image = vtk.vtkImageData()
        
        if len(image_data.shape) == 3:
            depth, height, width = image_data.shape
            vtk_image.SetDimensions(width, height, depth)
        else:
            height, width = image_data.shape
            vtk_image.SetDimensions(width, height, 1)
        
        spacing = image.spacing
        corrected_spacing_x = spacing.x
        corrected_spacing_y = spacing.y
        corrected_spacing_z = spacing.z
        
        if hasattr(image, 'dicom_metadata') and image.dicom_metadata:
            image_format = image.dicom_metadata.get('format', '')
            
            if image_format == 'NIfTI':
                max_spacing = max(spacing.x, spacing.y, spacing.z)
                min_spacing = min(spacing.x, spacing.y, spacing.z)
                
                anisotropy_ratio = max_spacing / min_spacing if min_spacing > 0 else 1.0
                
                if anisotropy_ratio > 3.0:
                    self._logger.warning(f"VTK: High anisotropy detected in NIfTI: {anisotropy_ratio:.2f}")
                    self._logger.warning(f"VTK: Original spacing: X={spacing.x:.3f}, Y={spacing.y:.3f}, Z={spacing.z:.3f}")
                    
                    if spacing.z > spacing.x * 2 or spacing.z > spacing.y * 2:
                        corrected_spacing_z = (spacing.x + spacing.y) / 2.0
                        self._logger.info(f"VTK: Correcting Z-spacing: {spacing.z:.3f} -> {corrected_spacing_z:.3f}")
                        
                    elif spacing.z < spacing.x / 2 or spacing.z < spacing.y / 2:
                        corrected_spacing_z = max(spacing.x, spacing.y) * 0.8
                        self._logger.info(f"VTK: Correcting small Z-spacing: {spacing.z:.3f} -> {corrected_spacing_z:.3f}")
                        
                    if spacing.x > spacing.y * 3:
                        corrected_spacing_x = spacing.y
                        self._logger.info(f"VTK: Correcting X-spacing: {spacing.x:.3f} -> {corrected_spacing_x:.3f}")
                    elif spacing.y > spacing.x * 3:
                        corrected_spacing_y = spacing.x  
                        self._logger.info(f"VTK: Correcting Y-spacing: {spacing.y:.3f} -> {corrected_spacing_y:.3f}")
                
                if (corrected_spacing_x != spacing.x or corrected_spacing_y != spacing.y or corrected_spacing_z != spacing.z):
                    self._logger.info(f"VTK: Final corrected spacing: X={corrected_spacing_x:.3f}, Y={corrected_spacing_y:.3f}, Z={corrected_spacing_z:.3f}")
        
        vtk_image.SetSpacing(corrected_spacing_x, corrected_spacing_y, corrected_spacing_z)
        
        if hasattr(image, 'origin') and image.origin:
            vtk_image.SetOrigin(image.origin[0], image.origin[1], image.origin[2])
            self._logger.info(f"VTK: Using medical origin: ({image.origin[0]:.2f}, {image.origin[1]:.2f}, {image.origin[2]:.2f})")
        else:
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
            self._logger.info("VTK: Using default origin: (0.0, 0.0, 0.0)")
        
        flat_data = image_data.flatten()
        vtk_array = numpy_support.numpy_to_vtk(flat_data)
        vtk_image.GetPointData().SetScalars(vtk_array)
        
        final_range = vtk_image.GetScalarRange()
        self._logger.debug(f"VTK renderer final data range: ({final_range[0]:.1f}, {final_range[1]:.1f})")
        
        return vtk_image
    
    async def _create_vtk_mask_data(
        self,
        segmentation: MedicalSegmentation,
        spacing: ImageSpacing
    ) -> vtk.vtkImageData:
        mask_data = segmentation.mask_data.astype(np.uint8)
        
        vtk_mask = vtk.vtkImageData()
        
        if len(mask_data.shape) == 3:
            depth, height, width = mask_data.shape
            vtk_mask.SetDimensions(width, height, depth)
        else:
            height, width = mask_data.shape
            vtk_mask.SetDimensions(width, height, 1)
        
        vtk_mask.SetSpacing(spacing.x, spacing.y, spacing.z)
        vtk_mask.SetOrigin(0.0, 0.0, 0.0)
        
        flat_mask = mask_data.flatten()
        vtk_array = numpy_support.numpy_to_vtk(flat_mask)
        vtk_mask.GetPointData().SetScalars(vtk_array)
        
        return vtk_mask
    
    def _create_volume_mapper(
        self,
        modality: 'ImageModalityType',
        rendering_mode: str
    ) -> vtk.vtkVolumeMapper:
        if rendering_mode == "mip":
            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetBlendModeToMaximumIntensity()
        elif rendering_mode == "isosurface":
            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetBlendModeToIsoSurface()
        else:  # composite
            mapper = vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetBlendModeToComposite()
        
        if modality.value == "CT":
            mapper.SetSampleDistance(0.5)
        elif modality.value == "MRI":
            mapper.SetSampleDistance(1.0)
        
        return mapper
    
    def _create_volume_property(
        self,
        image: MedicalImage,
        rendering_mode: str,
        opacity_curve: Optional[List[Tuple[float, float]]]
    ) -> vtk.vtkVolumeProperty:
        property = vtk.vtkVolumeProperty()
        
        color_func = vtk.vtkColorTransferFunction()
        
        opacity_func = vtk.vtkPiecewiseFunction()
        
        modality_preset = self._modality_presets.get(image.modality.value, {})
        
        if opacity_curve:
            for value, opacity in opacity_curve:
                opacity_func.AddPoint(value, opacity)
        else:
            opacity_points = modality_preset.get("opacity_points", [(0, 0), (255, 1)])
            for value, opacity in opacity_points:
                opacity_func.AddPoint(value, opacity)
        
        color_points = modality_preset.get("color_points", [
            (0, 0, 0, 0), (255, 1, 1, 1)
        ])
        for value, r, g, b in color_points:
            color_func.AddRGBPoint(value, r, g, b)
        
        property.SetColor(color_func)
        property.SetScalarOpacity(opacity_func)
        
        property.SetInterpolationTypeToLinear()
        property.ShadeOn()
        property.SetAmbient(0.4)
        property.SetDiffuse(0.6)
        property.SetSpecular(0.2)
        
        return property
    
    def _create_physical_transform(self, spacing: ImageSpacing) -> vtk.vtkTransform:
        transform = vtk.vtkTransform()
        return transform
    
    def _get_anatomical_color(
        self,
        region: AnatomicalRegion
    ) -> Tuple[float, float, float]:
        color_map = {
            AnatomicalRegion.PROSTATE_WHOLE: (0.8, 0.6, 0.4),  
            AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE: (0.4, 0.8, 0.4),
            AnatomicalRegion.PROSTATE_TRANSITION_ZONE: (0.4, 0.4, 0.8),
            AnatomicalRegion.SUSPICIOUS_LESION: (1.0, 0.8, 0.0),  
            AnatomicalRegion.CONFIRMED_CANCER: (1.0, 0.2, 0.2), 
            AnatomicalRegion.BENIGN_HYPERPLASIA: (0.6, 0.8, 0.6), 
            AnatomicalRegion.URETHRA: (0.8, 0.4, 0.8),  
            AnatomicalRegion.SEMINAL_VESICLES: (0.8, 0.8, 0.4)  
        }
        
        return color_map.get(region, (0.7, 0.7, 0.7))  
    
    def _create_line_actor(
        self,
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        color: Tuple[float, float, float]
    ) -> vtk.vtkActor:
        line = vtk.vtkLineSource()
        line.SetPoint1(point1)
        line.SetPoint2(point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(2)
        
        return actor
    
    def _create_text_actor(
        self,
        text: str,
        position: Tuple[float, float, float],
        color: Tuple[float, float, float],
        font_size: int
    ) -> vtk.vtkTextActor3D:
        text_actor = vtk.vtkTextActor3D()
        text_actor.SetInput(text)
        text_actor.SetPosition(position)
        
        text_property = text_actor.GetTextProperty()
        text_property.SetFontSize(font_size)
        text_property.SetColor(color)
        text_property.SetFontFamilyToArial()
        text_property.BoldOn()
        
        return text_actor
    
    def _create_modality_presets(self) -> Dict[str, Dict]:
        return {
            "CT": {
                "opacity_points": [(0, 0), (100, 0.1), (500, 0.3), (1000, 0.8), (3000, 1.0)],
                "color_points": [
                    (0, 0, 0, 0),
                    (100, 0.5, 0.3, 0.3),
                    (500, 0.8, 0.8, 0.6),
                    (1000, 1.0, 1.0, 0.9),
                    (3000, 1.0, 1.0, 1.0)
                ]
            },
            "MRI": {
                "opacity_points": [(0, 0), (100, 0.15), (300, 0.4), (600, 0.7), (1000, 1.0)],
                "color_points": [
                    (0, 0, 0, 0),          
                    (100, 0.2, 0.2, 0.6),  
                    (300, 0.4, 0.6, 0.9),  
                    (600, 0.8, 0.9, 1.0),  
                    (1000, 1.0, 1.0, 1.0) 
                ]
            }
        }
    
    def _setup_medical_interaction_style(self) -> None:
        with QMutexLocker(self._vtk_mutex):
            style = vtk.vtkInteractorStyleTrackballCamera()
            self._render_window_interactor.SetInteractorStyle(style)
    
    def _setup_medical_lighting(self) -> None:
        with QMutexLocker(self._vtk_mutex):
            light1 = vtk.vtkLight()
            light1.SetPosition(1, 1, 1)
            light1.SetIntensity(0.8)
            light1.SetColor(1, 1, 1)
            self._renderer.AddLight(light1)
            
            light2 = vtk.vtkLight()
            light2.SetPosition(-1, -1, 1)
            light2.SetIntensity(0.4)
            light2.SetColor(1, 1, 1)
            self._renderer.AddLight(light2)
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        if event_type in self._event_callbacks:
            for callback in self._event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self._logger.error(f"Error in callback {event_type}: {e}")
    
    def __del__(self):
        self.cleanup()
    
    def cleanup(self) -> None:
        if self._is_destroyed:
            return
        
        try:
            self._logger.info("Cleaning up VTK renderer resources...")
            
            self._renderer.RemoveAllViewProps()
            
            for category, objects in self._vtk_objects.items():
                for obj in objects:
                    try:
                        if hasattr(obj, 'UnRegister'):
                            obj.UnRegister(None)
                    except:
                        pass  # Ignore errors during cleanup
                objects.clear()
            
            with QMutexLocker(self._vtk_mutex):
                if hasattr(self, '_render_window_interactor') and self._render_window_interactor:
                    try:
                        self._render_window_interactor.SetRenderWindow(None)
                        self._render_window_interactor = None
                    except:
                        pass
                
                if hasattr(self, '_render_window') and self._render_window:
                    try:
                        self._render_window.RemoveRenderer(self._renderer)
                        self._render_window.Finalize()
                        self._render_window = None
                    except:
                        pass
                
                if hasattr(self, '_renderer') and self._renderer:
                    try:
                        self._renderer.RemoveAllViewProps()
                        self._renderer = None
                    except:
                        pass
            
            self._is_destroyed = True
            self._logger.info("VTK renderer cleanup completed")
            
        except Exception as e:
            self._logger.error(f"Error during VTK cleanup: {e}")
    
    def _register_vtk_object(self, obj: Any, category: str) -> None:
        if category in self._vtk_objects:
            self._vtk_objects[category].append(obj)


class VTKRenderingError(Exception):
    pass