import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid


class SegmentationType(Enum):
    MANUAL = "manual"           
    AUTOMATIC = "automatic"     
    SEMI_AUTOMATIC = "semi_automatic"  
    IMPORTED = "imported" 


class AnatomicalRegion(Enum):
    PROSTATE_WHOLE = "prostate_whole"
    PROSTATE_PERIPHERAL_ZONE = "prostate_peripheral_zone"
    PROSTATE_TRANSITION_ZONE = "prostate_transition_zone"
    PROSTATE_CENTRAL_ZONE = "prostate_central_zone"
    SUSPICIOUS_LESION = "suspicious_lesion"
    CONFIRMED_CANCER = "confirmed_cancer"
    BENIGN_HYPERPLASIA = "benign_hyperplasia"
    URETHRA = "urethra"
    SEMINAL_VESICLES = "seminal_vesicles"


class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MODERATE = "moderate"      # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # > 0.9


@dataclass
class SegmentationMetrics:
    volume_mm3: float 
    surface_area_mm2: float
    max_diameter_mm: float 
    sphericity: float
    compactness: float
    voxel_count: int
    
    def get_equivalent_sphere_diameter(self) -> float:
        if self.volume_mm3 <= 0:
            return 0.0
        return 2.0 * ((3.0 * self.volume_mm3) / (4.0 * np.pi)) ** (1.0/3.0)


@dataclass
class IntensityStatistics:
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    median_intensity: float
    percentile_25: float
    percentile_75: float
    entropy: float              # Entropía of Shannon
    uniformity: float           # Uniformidad of intensidades
    
    def get_intensity_range(self) -> float:
        return self.max_intensity - self.min_intensity


class MedicalSegmentation:
    def __init__(
        self,
        mask_data: np.ndarray,
        anatomical_region: AnatomicalRegion,
        segmentation_type: SegmentationType,
        creation_date: datetime,
        creator_id: str,
        confidence_score: Optional[float] = None,
        parent_image_uid: Optional[str] = None,
        description: Optional[str] = None
    ):
        self._validate_mask_data(mask_data)
        self._validate_confidence_score(confidence_score)
        
        self._segmentation_id = str(uuid.uuid4())
        self._mask_data = mask_data.astype(bool)  
        self._anatomical_region = anatomical_region
        self._segmentation_type = segmentation_type
        self._creation_date = creation_date
        self._creator_id = creator_id
        self._confidence_score = confidence_score
        self._parent_image_uid = parent_image_uid
        self._description = description or f"{anatomical_region.value} segmentation"
        
        self._is_locked = False
        self._modification_history: List[Dict] = []
        
        self._cached_metrics: Optional[SegmentationMetrics] = None
        self._cached_intensity_stats: Optional[IntensityStatistics] = None
        self._cache_invalidated = True
    
    @property
    def segmentation_id(self) -> str:
        return self._segmentation_id
    
    @property
    def mask_data(self) -> np.ndarray:
        return self._mask_data.copy()
    
    @property
    def dimensions(self) -> Tuple[int, ...]:
        return self._mask_data.shape
    
    @property
    def anatomical_region(self) -> AnatomicalRegion:
        return self._anatomical_region
    
    @property
    def segmentation_type(self) -> SegmentationType:
        return self._segmentation_type
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        if self._confidence_score is None:
            return None
        
        if self._confidence_score < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self._confidence_score < 0.5:
            return ConfidenceLevel.LOW
        elif self._confidence_score < 0.7:
            return ConfidenceLevel.MODERATE
        elif self._confidence_score < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @property
    def is_locked(self) -> bool:
        return self._is_locked
    
    @property
    def voxel_count(self) -> int:
        return int(np.sum(self._mask_data))
    
    @property
    def is_empty(self) -> bool:
        return self.voxel_count == 0
    
    def get_bounding_box(self) -> Tuple[Tuple[int, int], ...]:
        if self.is_empty:
            return tuple((0, 0) for _ in self._mask_data.shape)
        
        indices = np.where(self._mask_data)
        bounding_box = []
        
        for i in range(len(indices)):
            min_idx = int(np.min(indices[i]))
            max_idx = int(np.max(indices[i]))
            bounding_box.append((min_idx, max_idx))
        
        return tuple(bounding_box)
    
    def get_centroid(self) -> Tuple[float, ...]:
        if self.is_empty:
            return tuple(0.0 for _ in self._mask_data.shape)
        
        indices = np.where(self._mask_data)
        centroid = []
        
        for i in range(len(indices)):
            centroid.append(float(np.mean(indices[i])))
        
        return tuple(centroid)
    
    def calculate_metrics(self, spacing: 'ImageSpacing') -> SegmentationMetrics:
        if self._cached_metrics is not None and not self._cache_invalidated:
            return self._cached_metrics
        
        if self.is_empty:
            self._cached_metrics = SegmentationMetrics(
                volume_mm3=0.0, surface_area_mm2=0.0, max_diameter_mm=0.0,
                sphericity=0.0, compactness=0.0, voxel_count=0
            )
            return self._cached_metrics
        
        voxel_volume = spacing.get_voxel_volume()
        volume_mm3 = self.voxel_count * voxel_volume
        
        max_diameter_mm = self._calculate_max_diameter(spacing)
        
        surface_area_mm2 = self._calculate_surface_area(spacing)
        
        sphericity = self._calculate_sphericity(volume_mm3, surface_area_mm2)
        compactness = self._calculate_compactness(volume_mm3, surface_area_mm2)
        
        self._cached_metrics = SegmentationMetrics(
            volume_mm3=volume_mm3,
            surface_area_mm2=surface_area_mm2,
            max_diameter_mm=max_diameter_mm,
            sphericity=sphericity,
            compactness=compactness,
            voxel_count=self.voxel_count
        )
        
        self._cache_invalidated = False
        return self._cached_metrics
    
    def calculate_intensity_statistics(self, image_data: np.ndarray) -> IntensityStatistics:
        if image_data.shape != self._mask_data.shape:
            raise ValueError("Las dimensiones of image y máscara debin coincidir")
        
        if self.is_empty:
            return IntensityStatistics(
                mean_intensity=0.0, std_intensity=0.0, min_intensity=0.0,
                max_intensity=0.0, median_intensity=0.0, percentile_25=0.0,
                percentile_75=0.0, entropy=0.0, uniformity=0.0
            )
        
        masked_values = image_data[self._mask_data]
        
        mean_intensity = float(np.mean(masked_values))
        std_intensity = float(np.std(masked_values))
        min_intensity = float(np.min(masked_values))
        max_intensity = float(np.max(masked_values))
        median_intensity = float(np.median(masked_values))
        percentile_25 = float(np.percentile(masked_values, 25))
        percentile_75 = float(np.percentile(masked_values, 75))
        
        entropy = self._calculate_entropy(masked_values)
        uniformity = self._calculate_uniformity(masked_values)
        
        self._cached_intensity_stats = IntensityStatistics(
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            median_intensity=median_intensity,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            entropy=entropy,
            uniformity=uniformity
        )
        
        return self._cached_intensity_stats
    
    def union_with(self, other: 'MedicalSegmentation') -> 'MedicalSegmentation':
        if self._mask_data.shape != other._mask_data.shape:
            raise ValueError("Las segmentaciones debin tener las mismas dimensiones")
        
        union_mask = np.logical_or(self._mask_data, other._mask_data)
        
        return MedicalSegmentation(
            mask_data=union_mask,
            anatomical_region=self._anatomical_region,
            segmentation_type=SegmentationType.MANUAL,
            creation_date=datetime.now(),
            creator_id="system_union",
            parent_image_uid=self._parent_image_uid,
            description=f"Union of {self._description} and {other._description}"
        )
    
    def intersection_with(self, other: 'MedicalSegmentation') -> 'MedicalSegmentation':
        if self._mask_data.shape != other._mask_data.shape:
            raise ValueError("Las segmentaciones debin tener las mismas dimensiones")
        
        intersection_mask = np.logical_and(self._mask_data, other._mask_data)
        
        return MedicalSegmentation(
            mask_data=intersection_mask,
            anatomical_region=self._anatomical_region,
            segmentation_type=SegmentationType.MANUAL,
            creation_date=datetime.now(),
            creator_id="system_intersection",
            parent_image_uid=self._parent_image_uid,
            description=f"Intersection of {self._description} and {other._description}"
        )
    
    def apply_morphological_operation(self, operation: str, iterations: int = 1) -> 'MedicalSegmentation':
        from scipy import ndimage
        
        if operation == 'erode':
            processed_mask = ndimage.binary_erosion(self._mask_data, iterations=iterations)
        elif operation == 'dilate':
            processed_mask = ndimage.binary_dilation(self._mask_data, iterations=iterations)
        elif operation == 'open':
            processed_mask = ndimage.binary_opening(self._mask_data, iterations=iterations)
        elif operation == 'close':
            processed_mask = ndimage.binary_closing(self._mask_data, iterations=iterations)
        else:
            raise ValueError(f"Operación morfológica '{operation}' no reconocida")
        
        return MedicalSegmentation(
            mask_data=processed_mask,
            anatomical_region=self._anatomical_region,
            segmentation_type=SegmentationType.MANUAL,
            creation_date=datetime.now(),
            creator_id="system_morphology",
            parent_image_uid=self._parent_image_uid,
            description=f"{operation.capitalize()} of {self._description}"
        )
    
    def lock(self) -> None:
        self._is_locked = True
    
    def unlock(self) -> None:
        self._is_locked = False
    
    def _validate_mask_data(self, mask_data: np.ndarray) -> None:
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("La máscara debe ser un numpy array")
        
        if mask_data.size == 0:
            raise ValueError("La máscara no pueof estar vacía")
        
        if not (2 <= len(mask_data.shape) <= 3):
            raise ValueError("La máscara debe ser 2D o 3D")
    
    def _validate_confidence_score(self, confidence_score: Optional[float]) -> None:
        if confidence_score is not None:
            if not (0.0 <= confidence_score <= 1.0):
                raise ValueError("El score of confianza debe estar entre 0.0 y 1.0")
    
    def _calculate_max_diameter(self, spacing: 'ImageSpacing') -> float:
        if self.is_empty:
            return 0.0
        
        indices = np.where(self._mask_data)
        if len(indices[0]) < 2:
            return 0.0
        
        max_distance = 0.0
        coords = list(zip(*indices))
        
        if len(coords) > 1000:
            step = len(coords) // 1000
            coords = coords[::step]
        
        for i, coord1 in enumerate(coords):
            for coord2 in coords[i+1:]:
                distance = 0.0
                if len(coord1) == 3:  # 3D
                    dx = (coord1[2] - coord2[2]) * spacing.x
                    dy = (coord1[1] - coord2[1]) * spacing.y
                    dz = (coord1[0] - coord2[0]) * spacing.z
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                else:  # 2D
                    dx = (coord1[1] - coord2[1]) * spacing.x
                    dy = (coord1[0] - coord2[0]) * spacing.y
                    distance = np.sqrt(dx*dx + dy*dy)
                
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _calculate_surface_area(self, spacing: 'ImageSpacing') -> float:
        if self.is_empty:
            return 0.0
        
        if len(self._mask_data.shape) == 3:
            grad_x = np.gradient(self._mask_data.astype(float), axis=2)
            grad_y = np.gradient(self._mask_data.astype(float), axis=1)
            grad_z = np.gradient(self._mask_data.astype(float), axis=0)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            surface_voxels = np.sum(gradient_magnitude > 0)
            avg_voxel_face_area = (spacing.x * spacing.y + 
                                 spacing.y * spacing.z + 
                                 spacing.x * spacing.z) / 3.0
            return surface_voxels * avg_voxel_face_area
        
        else:
            grad_x = np.gradient(self._mask_data.astype(float), axis=1)
            grad_y = np.gradient(self._mask_data.astype(float), axis=0)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            perimeter_pixels = np.sum(gradient_magnitude > 0)
            avg_pixel_length = (spacing.x + spacing.y) / 2.0
            return perimeter_pixels * avg_pixel_length
    
    def _calculate_sphericity(self, volume: float, surface_area: float) -> float:
        if volume <= 0 or surface_area <= 0:
            return 0.0
        
        if len(self._mask_data.shape) == 3:
            # Esfericidad 3D: π^(1/3) * (6V)^(2/3) / A
            sphere_surface_area = np.pi**(1.0/3.0) * (6.0 * volume)**(2.0/3.0)
            return sphere_surface_area / surface_area
        else:
            # Circularidad 2D: 4πA / P²
            return (4.0 * np.pi * volume) / (surface_area ** 2)
    
    def _calculate_compactness(self, volume: float, surface_area: float) -> float:
        if volume <= 0 or surface_area <= 0:
            return 0.0
        
        if len(self._mask_data.shape) == 3:
            # Compacidad 3D: V / (A^(3/2))
            return volume / (surface_area ** 1.5)
        else:
            # Compacidad 2D: A / P²
            return volume / (surface_area ** 2)
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        
        hist, _ = np.histogram(values, bins=256, density=True)
        hist = hist[hist > 0] 
        
        if len(hist) == 0:
            return 0.0
        
        # Calcular entropía: -Σ(p * log2(p))
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_uniformity(self, values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        
        hist, _ = np.histogram(values, bins=256, density=True)
        
        # Uniformidad: Σ(p²)
        return np.sum(hist ** 2)