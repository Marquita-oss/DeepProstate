import logging
import psutil
import threading
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from PyQt6.QtCore import QMutex, QMutexLocker

from deepprostate.core.domain.entities.medical_image import MedicalImage


class MedicalImageCacheService:
    
    def __init__(self, max_cached_images: int = 50, max_memory_mb: int = 2048):
        self._logger = logging.getLogger(__name__)
        
        self._max_cached_images = max_cached_images
        self._max_memory_mb = max_memory_mb
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache_mutex = QMutex()
        self._loaded_images: OrderedDict[str, MedicalImage] = OrderedDict()
        self._image_access_times: Dict[str, datetime] = {}
        self._image_memory_usage: Dict[str, int] = {}
        
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_evictions = 0
        
        self._logger.debug(f"Initialized cache with limits: {max_cached_images} images, {max_memory_mb}MB")
    
    def get_cached_image(self, series_uid: str) -> Optional[MedicalImage]:
        with QMutexLocker(self._cache_mutex):
            if series_uid in self._loaded_images:
                medical_image = self._loaded_images.pop(series_uid)
                self._loaded_images[series_uid] = medical_image
                self._image_access_times[series_uid] = datetime.now()
                
                self._cache_hits += 1
                self._logger.debug(f"Cache HIT for series: {series_uid[:8]}")
                return medical_image
            else:
                self._cache_misses += 1
                self._logger.debug(f"Cache MISS for series: {series_uid[:8]}")
                return None
    
    def cache_image(self, series_uid: str, medical_image: MedicalImage) -> bool:
        if not medical_image:
            return False
        
        with QMutexLocker(self._cache_mutex):
            try:
                image_memory = self._estimate_image_memory(medical_image)
                
                if series_uid in self._loaded_images:
                    self._remove_from_cache_internal(series_uid)
                
                self._ensure_cache_limits(additional_memory=image_memory)
                
                self._loaded_images[series_uid] = medical_image
                self._image_access_times[series_uid] = datetime.now()
                self._image_memory_usage[series_uid] = image_memory
                
                self._logger.debug(f"Cached image: {series_uid[:8]} ({image_memory / 1024 / 1024:.1f}MB)")
                return True
                
            except Exception as e:
                self._logger.error(f"Error caching image {series_uid[:8]}: {e}")
                return False
    
    def remove_from_cache(self, series_uid: str) -> bool:
        with QMutexLocker(self._cache_mutex):
            return self._remove_from_cache_internal(series_uid)
    
    def _remove_from_cache_internal(self, series_uid: str) -> bool:
        if series_uid in self._loaded_images:
            del self._loaded_images[series_uid]
            self._image_access_times.pop(series_uid, None)
            memory_freed = self._image_memory_usage.pop(series_uid, 0)
            
            self._logger.debug(f"Removed from cache: {series_uid[:8]} (freed {memory_freed / 1024 / 1024:.1f}MB)")
            return True
        return False
    
    def _ensure_cache_limits(self, additional_memory: int = 0) -> None:
        current_memory = sum(self._image_memory_usage.values()) + additional_memory
        
        while len(self._loaded_images) >= self._max_cached_images:
            self._evict_oldest_image()
        
        while current_memory > self._max_memory_bytes and self._loaded_images:
            self._evict_oldest_image()
            current_memory = sum(self._image_memory_usage.values()) + additional_memory
    
    def _evict_oldest_image(self) -> None:
        if not self._loaded_images:
            return
        
        oldest_series_uid = next(iter(self._loaded_images))
        memory_freed = self._image_memory_usage.get(oldest_series_uid, 0)
        
        self._remove_from_cache_internal(oldest_series_uid)
        self._total_evictions += 1
        
        self._logger.debug(f"Evicted oldest image: {oldest_series_uid[:8]} (freed {memory_freed / 1024 / 1024:.1f}MB)")
    
    def _estimate_image_memory(self, medical_image: MedicalImage) -> int:
        try:
            if hasattr(medical_image, 'pixel_array') and medical_image.pixel_array is not None:
                return medical_image.pixel_array.nbytes
            else:
                return 50 * 1024 * 1024
        except Exception:
            return 50 * 1024 * 1024
    
    def should_cleanup_memory(self) -> bool:
        try:
            memory_info = psutil.virtual_memory()
            system_memory_percent = memory_info.percent
            
            cache_memory_mb = sum(self._image_memory_usage.values()) / 1024 / 1024
            cache_memory_percent = (cache_memory_mb / self._max_memory_mb) * 100
            
            if system_memory_percent > 85.0:
                self._logger.debug(f"System memory high: {system_memory_percent:.1f}%")
                return True
            
            if cache_memory_percent > 90.0:
                self._logger.debug(f"Cache memory high: {cache_memory_percent:.1f}%")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Error checking memory usage: {e}")
            return False
    
    def cleanup_old_images(self, max_age_minutes: int = 30) -> int:
        with QMutexLocker(self._cache_mutex):
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            to_remove = []
            
            for series_uid, access_time in self._image_access_times.items():
                if access_time < cutoff_time:
                    to_remove.append(series_uid)
            
            cleaned_count = 0
            for series_uid in to_remove:
                if self._remove_from_cache_internal(series_uid):
                    cleaned_count += 1
            
            if cleaned_count > 0:
                self._logger.info(f"Cleaned up {cleaned_count} old images (age > {max_age_minutes}min)")
            
            return cleaned_count
    
    def force_cleanup(self, target_memory_percent: float = 50.0) -> int:
        with QMutexLocker(self._cache_mutex):
            target_memory_bytes = int(self._max_memory_bytes * (target_memory_percent / 100.0))
            current_memory = sum(self._image_memory_usage.values())
            
            evicted_count = 0
            while current_memory > target_memory_bytes and self._loaded_images:
                self._evict_oldest_image()
                evicted_count += 1
                current_memory = sum(self._image_memory_usage.values())
            
            if evicted_count > 0:
                self._logger.info(f"Force cleanup evicted {evicted_count} images")
            
            return evicted_count
    
    def clear_cache(self) -> int:
        with QMutexLocker(self._cache_mutex):
            cleared_count = len(self._loaded_images)
            
            self._loaded_images.clear()
            self._image_access_times.clear()
            self._image_memory_usage.clear()
            
            if cleared_count > 0:
                self._logger.info(f"Cleared {cleared_count} cached images")
            
            return cleared_count
    
    def get_cache_statistics(self) -> Dict[str, any]:
        with QMutexLocker(self._cache_mutex):
            total_memory_bytes = sum(self._image_memory_usage.values())
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cached_images_count': len(self._loaded_images),
                'max_cached_images': self._max_cached_images,
                'total_memory_mb': total_memory_bytes / 1024 / 1024,
                'max_memory_mb': self._max_memory_mb,
                'memory_usage_percent': (total_memory_bytes / self._max_memory_bytes * 100) if self._max_memory_bytes > 0 else 0,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'hit_rate_percent': hit_rate,
                'total_evictions': self._total_evictions
            }
    
    def get_cached_series_list(self) -> List[str]:
        with QMutexLocker(self._cache_mutex):
            return list(self._loaded_images.keys())
    
    def is_series_cached(self, series_uid: str) -> bool:
        with QMutexLocker(self._cache_mutex):
            return series_uid in self._loaded_images
    
    def get_cache_status_summary(self) -> str:
        stats = self.get_cache_statistics()
        
        return (f"Cache: {stats['cached_images_count']}/{stats['max_cached_images']} images, "
                f"{stats['total_memory_mb']:.1f}/{stats['max_memory_mb']}MB "
                f"({stats['memory_usage_percent']:.1f}%), "
                f"Hit rate: {stats['hit_rate_percent']:.1f}%")
    
    def configure_limits(self, max_cached_images: Optional[int] = None, 
                        max_memory_mb: Optional[int] = None) -> None:
        with QMutexLocker(self._cache_mutex):
            if max_cached_images is not None:
                self._max_cached_images = max_cached_images
                self._logger.info(f"Updated max cached images: {max_cached_images}")
            
            if max_memory_mb is not None:
                self._max_memory_mb = max_memory_mb
                self._max_memory_bytes = max_memory_mb * 1024 * 1024
                self._logger.debug(f"Updated max memory: {max_memory_mb}MB")
            
            # Enforce new limits
            self._ensure_cache_limits()
    
    def get_memory_pressure_level(self) -> str:
        try:
            stats = self.get_cache_statistics()
            memory_percent = stats['memory_usage_percent']
            
            if memory_percent >= 95:
                return 'CRITICAL'
            elif memory_percent >= 80:
                return 'HIGH'
            elif memory_percent >= 60:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception:
            return 'UNKNOWN'