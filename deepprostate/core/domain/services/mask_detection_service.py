import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from deepprostate.core.domain.entities.medical_image import MedicalImage


@dataclass
class MaskDetectionConfig:
    naming_patterns_enabled: bool = True
    content_analysis_enabled: bool = True
    directory_scan_enabled: bool = True
    dimension_matching_enabled: bool = True
    max_unique_values: int = 10
    min_background_ratio: float = 0.7
    mask_subdirs: List[str] = None
    
    def __post_init__(self):
        if self.mask_subdirs is None:
            self.mask_subdirs = ['masks', 'segmentations', 'labels', 'annotations', 'seg']


class MaskDetectionService:
    
    def __init__(self, config: Optional[MaskDetectionConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or MaskDetectionConfig()
        
        self.naming_patterns = [
            '{stem}_seg{ext}', '{stem}_mask{ext}', '{stem}_segmentation{ext}',
            '{stem}_label{ext}', '{stem}_annotation{ext}',
            'seg_{stem}{ext}', 'mask_{stem}{ext}', 'segmentation_{stem}{ext}',
            'label_{stem}{ext}', 'annotation_{stem}{ext}'
        ]
        
        self.contextual_patterns = [
            lambda image_stem, files: self._find_contextual_masks_cross_format(image_stem, files),
            lambda image_stem, files: self._find_shared_prefix_masks(image_stem[0], files, image_stem[1]),
            lambda image_stem, files: self._find_sequence_related_masks(image_stem[0], files, image_stem[1])
        ]
        
        self.supported_extensions = ['.nii', '.nii.gz', '.mha', '.mhd', '.dcm']

        self.cross_format_mapping = {
            '.dcm': ['.nii.gz', '.nii', '.mha', '.mhd'],        
            '.mha': ['.nii.gz', '.nii', '.mhd', '.dcm'],       
            '.mhd': ['.nii.gz', '.nii', '.mha', '.dcm'],      
            '.nii': ['.nii.gz', '.mha', '.mhd', '.dcm'],      
            '.nii.gz': ['.nii', '.mha', '.mhd', '.dcm']         
        }
        
        self.logger.info("MaskDetectionService initialized with multi-strategy detection")
    
    def find_associated_masks(self, image_path: Path) -> List[Path]:
        if not image_path.exists():
            return []
            
        found_masks = []
        
        try:
            if self.config.naming_patterns_enabled:
                pattern_masks = self._find_by_naming_patterns(image_path)
                found_masks.extend(pattern_masks)
                
                if pattern_masks:
                    self.logger.debug(f"Found {len(pattern_masks)} masks by naming patterns for {image_path.name}")
            
            if not found_masks and (self.config.content_analysis_enabled or self.config.directory_scan_enabled):
                self.logger.debug(f"No pattern matches for {image_path.name}, analyzing content...")
                
                potential_masks = self._scan_for_potential_masks(image_path)
                
                for potential_mask in potential_masks:
                    if self._is_valid_mask_file(potential_mask, image_path):
                        found_masks.append(potential_mask)
            
            found_masks = list(dict.fromkeys(found_masks))
            
            if found_masks:
                self.logger.info(f"Total found {len(found_masks)} associated masks for {image_path.name}")
            
            return found_masks
            
        except Exception as e:
            self.logger.error(f"Error detecting masks for {image_path}: {e}")
            return []
    
    def _find_by_naming_patterns(self, image_path: Path) -> List[Path]:
        found_masks = []
        stem = image_path.stem
        if image_path.suffix == '.gz' and image_path.stem.endswith('.nii'):
            stem = image_path.stem[:-4] 
            ext = '.nii.gz'
        else:
            ext = image_path.suffix

        parent = image_path.parent

        for pattern in self.naming_patterns:
            mask_name = pattern.format(stem=stem, ext=ext)
            mask_path = parent / mask_name

            if mask_path.exists() and mask_path.suffix in self.supported_extensions:
                found_masks.append(mask_path)
                self.logger.info(f"Found same-format mask: {mask_path.name}")

        possible_mask_extensions = self.cross_format_mapping.get(ext, [])
        for pattern in self.naming_patterns:
            for mask_ext in possible_mask_extensions:
                mask_name = pattern.format(stem=stem, ext=mask_ext)
                mask_path = parent / mask_name

                if mask_path.exists() and mask_path not in found_masks:
                    found_masks.append(mask_path)
                    self.logger.info(f"Found cross-format mask: {image_path.name} ({ext}) → {mask_path.name} ({mask_ext})")

        if not found_masks:
            for file in parent.iterdir():
                if file.is_file() and file != image_path:
                    has_mask = 'mask' in file.name.lower()

                    file_ext = file.suffix
                    if file.name.lower().endswith('.nii.gz'):
                        file_ext = '.nii.gz'

                    same_ext = file_ext == ext

                    if has_mask and same_ext:
                        found_masks.append(file)
                        self.logger.info(f"Found contextual same-format mask: {file.name} for {image_path.name}")

            if not found_masks:
                possible_mask_extensions = self.cross_format_mapping.get(ext, [])
                for file in parent.iterdir():
                    if file.is_file() and file != image_path:
                        has_mask = 'mask' in file.name.lower()

                        file_ext = file.suffix
                        if file.name.lower().endswith('.nii.gz'):
                            file_ext = '.nii.gz'

                        cross_format_match = file_ext in possible_mask_extensions

                        if has_mask and cross_format_match:
                            found_masks.append(file)
                            self.logger.info(f"Found contextual cross-format mask: {image_path.name} ({ext}) → {file.name} ({file_ext})")
        
        return found_masks
    
    def _find_shared_prefix_masks(self, image_stem: str, files: List[Path], target_ext: str) -> List[Path]:
        matches = []

        possible_mask_extensions = self.cross_format_mapping.get(target_ext, [])
        all_valid_extensions = [target_ext] + possible_mask_extensions

        if '_' in image_stem:
            parts = image_stem.split('_')
            for i in range(len(parts)):
                prefix = '_'.join(parts[:i+1]) + '_'
                for file in files:
                    file_ext = file.suffix
                    if file.name.lower().endswith('.nii.gz'):
                        file_ext = '.nii.gz'

                    if (file.name.lower().startswith(prefix.lower()) and
                        'mask' in file.name.lower() and
                        file_ext in all_valid_extensions):
                        matches.append(file)
                        if file_ext != target_ext:
                            self.logger.info(f"Found cross-format prefix mask: {prefix}... ({target_ext}) → {file.name} ({file_ext})")

        return matches
    
    def _find_sequence_related_masks(self, image_stem: str, files: List[Path], target_ext: str) -> List[Path]:
        matches = []

        possible_mask_extensions = self.cross_format_mapping.get(target_ext, [])
        all_valid_extensions = [target_ext] + possible_mask_extensions

        sequence_types = ['t2w', 'adc', 'dwi', 'flair', 't1', 't1c', 'hbv']

        if image_stem.lower() in sequence_types:
            self.logger.debug(f"Detected medical sequence '{image_stem}', looking for any masks in directory (cross-format enabled)...")
            for file in files:
                file_ext = file.suffix
                if file.name.lower().endswith('.nii.gz'):
                    file_ext = '.nii.gz'

                if ('mask' in file.name.lower() and
                    file_ext in all_valid_extensions and
                    not any(seq in file.name.lower() for seq in sequence_types if seq != image_stem.lower())):
                    matches.append(file)
                    if file_ext != target_ext:
                        self.logger.info(f"Found cross-format sequence mask: {image_stem} ({target_ext}) → {file.name} ({file_ext})")
                    else:
                        self.logger.debug(f"Found same-format sequence mask: {image_stem} → {file.name}")

        return matches

    def _find_contextual_masks_cross_format(self, image_stem: tuple, files: List[Path]) -> List[Path]:
        stem, ext = image_stem

        possible_mask_extensions = self.cross_format_mapping.get(ext, [])
        all_valid_extensions = [ext] + possible_mask_extensions

        matches = []
        for file in files:
            if 'mask' in file.name.lower():
                file_ext = file.suffix
                if file.name.lower().endswith('.nii.gz'):
                    file_ext = '.nii.gz'

                if file_ext in all_valid_extensions:
                    matches.append(file)
                    if file_ext != ext:
                        self.logger.info(f"Found contextual cross-format mask: {stem} ({ext}) → {file.name} ({file_ext})")

        return matches

    def _scan_for_potential_masks(self, image_path: Path) -> List[Path]:
        potential_masks = []
        parent_dir = image_path.parent
        
        if self.config.directory_scan_enabled:
            for subdir_name in self.config.mask_subdirs:
                mask_dir = parent_dir / subdir_name
                if mask_dir.exists() and mask_dir.is_dir():
                    for ext in self.supported_extensions:
                        potential_masks.extend(mask_dir.glob(f"*{ext}"))
                        if ext == '.nii': 
                            potential_masks.extend(mask_dir.glob(f"*.nii.gz"))
        
        for ext in self.supported_extensions:
            same_dir_files = parent_dir.glob(f"*{ext}")
            for file_path in same_dir_files:
                if file_path != image_path: 
                    potential_masks.append(file_path)
            
            if ext == '.nii':
                same_dir_files = parent_dir.glob("*.nii.gz")
                for file_path in same_dir_files:
                    if file_path != image_path:
                        potential_masks.append(file_path)
        
        return list(set(potential_masks)) 
    
    def _is_valid_mask_file(self, file_path: Path, reference_image_path: Path) -> bool:
        try:
            if not file_path.exists() or file_path.suffix not in self.supported_extensions:
                return False
            
            if self.config.content_analysis_enabled:
                if not self._analyze_mask_content(file_path):
                    return False
            
            if self.config.dimension_matching_enabled:
                if not self._dimensions_match(reference_image_path, file_path):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating mask file {file_path}: {e}")
            return False
    
    def _analyze_mask_content(self, file_path: Path) -> bool:
        try:
            file_size = file_path.stat().st_size
            
            if file_size < 1024:  
                return False
            
            filename_lower = file_path.name.lower()
            
            mask_indicators = ['seg', 'mask', 'label', 'annotation', 'contour']
            if any(indicator in filename_lower for indicator in mask_indicators):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error analyzing content of {file_path}: {e}")
            return False
    
    def _dimensions_match(self, image_path: Path, mask_path: Path) -> bool:
        try:
            image_size = image_path.stat().st_size
            mask_size = mask_path.stat().st_size

            if mask_size > image_size:
                return False
            
            if mask_size < image_size * 0.01:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error comparing dimensions of {image_path} and {mask_path}: {e}")
            return False
    
    def get_detection_strategy_info(self) -> Dict[str, Any]:
        return {
            'naming_patterns_enabled': self.config.naming_patterns_enabled,
            'content_analysis_enabled': self.config.content_analysis_enabled,
            'directory_scan_enabled': self.config.directory_scan_enabled,
            'dimension_matching_enabled': self.config.dimension_matching_enabled,
            'supported_extensions': self.supported_extensions,
            'naming_patterns_count': len(self.naming_patterns),
            'mask_subdirs': self.config.mask_subdirs
        }