import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass

from src.core.domain.entities.segmentation import AnatomicalRegion
from src.core.domain.entities.ai_analysis import OverlayVisualizationData


@dataclass
class LabelColorMapping:
    label_value: int
    anatomical_region: AnatomicalRegion
    color_rgba: Tuple[float, float, float, float]
    opacity: float
    name: str


class UniversalMaskVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._color_mappings = self._initialize_medical_color_mappings()
    
    def _initialize_medical_color_mappings(self) -> Dict[str, Dict[int, LabelColorMapping]]:
        return {
            'whole_gland': {
                1: LabelColorMapping(
                    label_value=1,
                    anatomical_region=AnatomicalRegion.PROSTATE_WHOLE,
                    color_rgba=(0.2, 0.8, 0.2, 0.6), 
                    opacity=0.6,
                    name="Prostate Whole Gland"
                )
            },
            
            'zones': {
                1: LabelColorMapping(
                    label_value=1,
                    anatomical_region=AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE,
                    color_rgba=(0.3, 0.5, 0.9, 0.6), 
                    opacity=0.6,
                    name="Peripheral Zone (PZ)"
                ),
                2: LabelColorMapping(
                    label_value=2,
                    anatomical_region=AnatomicalRegion.PROSTATE_TRANSITION_ZONE,
                    color_rgba=(0.9, 0.3, 0.3, 0.6), 
                    opacity=0.6,
                    name="Transition Zone (TZ)"
                )
            },
            
            'lesions': {
                1: LabelColorMapping(
                    label_value=1,
                    anatomical_region=AnatomicalRegion.SUSPICIOUS_LESION,
                    color_rgba=(1.0, 0.6, 0.0, 0.7), 
                    opacity=0.7,
                    name="csPCa Graof 1"
                ),
                2: LabelColorMapping(
                    label_value=2,
                    anatomical_region=AnatomicalRegion.SUSPICIOUS_LESION,
                    color_rgba=(1.0, 0.4, 0.4, 0.7), 
                    opacity=0.7,
                    name="csPCa Graof 2"
                ),
                3: LabelColorMapping(
                    label_value=3,
                    anatomical_region=AnatomicalRegion.CONFIRMED_CANCER,
                    color_rgba=(0.9, 0.1, 0.1, 0.8),
                    opacity=0.8,
                    name="csPCa Graof 3"
                ),
                4: LabelColorMapping(
                    label_value=4,
                    anatomical_region=AnatomicalRegion.CONFIRMED_CANCER,
                    color_rgba=(0.7, 0.0, 0.3, 0.8), 
                    opacity=0.8,
                    name="csPCa Graof 4"
                ),
                5: LabelColorMapping(
                    label_value=5,
                    anatomical_region=AnatomicalRegion.CONFIRMED_CANCER,
                    color_rgba=(0.5, 0.0, 0.5, 0.9),  
                    opacity=0.9,
                    name="csPCa Graof 5"
                )
            },
            
            'generic': {
                1: LabelColorMapping(1, AnatomicalRegion.PROSTATE_WHOLE, (0.5, 0.5, 0.5, 0.6), 0.6, "Label 1"),
                2: LabelColorMapping(2, AnatomicalRegion.PROSTATE_PERIPHERAL_ZONE, (0.7, 0.3, 0.7, 0.6), 0.6, "Label 2"),
                3: LabelColorMapping(3, AnatomicalRegion.PROSTATE_TRANSITION_ZONE, (0.3, 0.7, 0.7, 0.6), 0.6, "Label 3"),
                4: LabelColorMapping(4, AnatomicalRegion.SUSPICIOUS_LESION, (0.9, 0.9, 0.3, 0.6), 0.6, "Label 4")
            }
        }
    
    def create_visualization_overlays(
        self,
        mask_array: np.ndarray,
        mask_type: Optional[str] = None
    ) -> List[OverlayVisualizationData]:
        self.logger.debug("Creating visualization overlays")
        self.logger.debug(f"Mask array shape: {mask_array.shape}, type: {mask_type}")
        
        unique_labels = np.unique(mask_array)
        unique_labels = unique_labels[unique_labels > 0]  # Excluir fondo (0)
        
        self.logger.debug(f"Unique labels found: {unique_labels}")
        
        if len(unique_labels) == 0:
            self.logger.warning("No se encontraron labels in la máscara")
            return []
        
        self.logger.info(f"Processing máscara with labels: {unique_labels}")
        
        if mask_type is None:
            mask_type = self._auto_detect_mask_type(unique_labels, mask_array)
        
        color_mapping = self._get_color_mapping(mask_type, unique_labels)
        
        overlays = []
        for label_value in unique_labels:
            try:
                binary_mask = (mask_array == label_value).astype(np.uint8)
                
                if np.sum(binary_mask) == 0:
                    continue
                
                label_info = color_mapping.get(label_value)
                if not label_info:
                    self.logger.warning(f"No se encontró color for label {label_value}, usando genérico")
                    label_info = self._create_generic_label_info(label_value)
                
                overlay = OverlayVisualizationData(
                    mask_array=binary_mask,
                    anatomical_region=label_info.anatomical_region,
                    color_rgba=label_info.color_rgba,
                    opacity=label_info.opacity,
                    target_sequence="ALL"  
                )
                
                overlays.append(overlay)
                
                self.logger.debug(f"Creado overlay for {label_info.name}: {np.sum(binary_mask)} voxels")
                
            except Exception as e:
                self.logger.error(f"Error creando overlay for label {label_value}: {e}")
                continue
        
        self.logger.info(f"Generados {len(overlays)} overlays of visualización")
        return overlays
    
    def _auto_detect_mask_type(self, unique_labels: np.ndarray, mask_array: np.ndarray) -> str:
        num_labels = len(unique_labels)
        max_label = np.max(unique_labels)
        
        label_sizes = {}
        for label in unique_labels:
            label_sizes[label] = np.sum(mask_array == label)
        
        self.logger.debug(f"Auto-detección: {num_labels} labels, max={max_label}, sizes={label_sizes}")
        
        if num_labels == 1:
            if np.sum(mask_array > 0) / mask_array.size > 0.1: 
                return 'whole_gland'
            else:
                return 'lesions'
                
        elif num_labels == 2:
            if max_label > 2:
                self.logger.info(f"Detected csPCa pattern: non-consecutive values {unique_labels}")
                return 'lesions'  
            elif max_label <= 2:
                if all(size > 1000 for size in label_sizes.values()):
                    return 'zones'
                else:
                    return 'lesions'
                    
        else:
            if max_label >= 3:
                self.logger.info(f"Detected multi-grade csPCa pattern: {unique_labels}")
                return 'lesions'  
            else:
                return 'lesions' if num_labels <= 4 else 'generic'
    
    def _get_color_mapping(self, mask_type: str, unique_labels: np.ndarray) -> Dict[int, LabelColorMapping]:
        if mask_type in self._color_mappings:
            base_mapping = self._color_mappings[mask_type]
        else:
            base_mapping = self._color_mappings['generic']
        
        final_mapping = {}
        for label in unique_labels:
            if label in base_mapping:
                final_mapping[label] = base_mapping[label]
            else:
                final_mapping[label] = self._create_generic_label_info(label)
        
        return final_mapping
    
    def _create_generic_label_info(self, label_value: int) -> LabelColorMapping:
        np.random.seed(label_value * 42)  # Seed consistente
        color = np.random.uniform(0.3, 0.9, 3)  # RGB
        
        return LabelColorMapping(
            label_value=label_value,
            anatomical_region=AnatomicalRegion.PROSTATE_WHOLE,  # Fallback
            color_rgba=(*color, 0.6),
            opacity=0.6,
            name=f"Label {label_value}"
        )
    
    def get_supported_mask_types(self) -> List[str]:
        return list(self._color_mappings.keys())
    
    def get_color_info_for_type(self, mask_type: str) -> Dict[int, LabelColorMapping]:
        return self._color_mappings.get(mask_type, self._color_mappings['generic'])