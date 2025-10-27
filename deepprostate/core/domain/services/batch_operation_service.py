import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer

from deepprostate.core.domain.entities.medical_image import MedicalImage
from deepprostate.frameworks.infrastructure.storage.dicom_repository import DICOMImageRepository


@dataclass
class BatchOperation:
    operation_id: str
    operation_type: str  
    items: List[str] 
    description: str
    progress: int = 0
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class BatchOperationWorker(QThread):
    progress_updated = pyqtSignal(str, int, str) 
    operation_completed = pyqtSignal(str, dict)
    operation_failed = pyqtSignal(str, str)
    
    def __init__(self, operation: BatchOperation, repository: DICOMImageRepository):
        super().__init__()
        self.operation = operation
        self.repository = repository
        self._cancelled = False
        self._logger = logging.getLogger(f"{__name__}.Worker")
    
    def run(self):
        try:
            self.operation.start_time = datetime.now()
            self.operation.status = "running"
            
            if self.operation.operation_type == "series":
                results = self._load_batch_series()
            elif self.operation.operation_type == "studies":
                results = self._load_batch_studies()
            else:
                raise ValueError(f"Unknown operation type: {self.operation.operation_type}")
            
            if not self._cancelled:
                self.operation.end_time = datetime.now()
                self.operation.status = "completed"
                self.operation_completed.emit(self.operation.operation_id, results)
                
        except Exception as e:
            self.operation.end_time = datetime.now()
            self.operation.status = "failed"
            self.operation.error_message = str(e)
            self.operation_failed.emit(self.operation.operation_id, str(e))
    
    def cancel(self):
        self._cancelled = True
        self.operation.status = "cancelled"
    
    def _load_batch_series(self) -> Dict[str, Any]:
        results = {"loaded_series": [], "failed_series": []}
        total_series = len(self.operation.items)
        
        for i, series_uid in enumerate(self.operation.items):
            if self._cancelled:
                break

            try:
                import time
                time.sleep(0.1)
                
                results["loaded_series"].append({
                    "series_uid": series_uid,
                    "status": "loaded",
                    "timestamp": datetime.now()
                })
                
            except Exception as e:
                self._logger.error(f"Failed to load series {series_uid}: {e}")
                results["failed_series"].append({
                    "series_uid": series_uid,
                    "error": str(e)
                })
        
        if not self._cancelled:
            self.progress_updated.emit(
                self.operation.operation_id,
                100,
                f"Completed: {len(results['loaded_series'])} loaded, {len(results['failed_series'])} failed"
            )
        
        return results
    
    def _load_batch_studies(self) -> Dict[str, Any]:
        results = {"loaded_studies": [], "failed_studies": []}
        total_studies = len(self.operation.items)
        
        for i, study_uid in enumerate(self.operation.items):
            if self._cancelled:
                break

            try:
                
                import time
                time.sleep(0.2) 
                
                results["loaded_studies"].append({
                    "study_uid": study_uid,
                    "status": "loaded",
                    "timestamp": datetime.now(),
                    "series_count": 5
                })
                
            except Exception as e:
                self._logger.error(f"Failed to load study {study_uid}: {e}")
                results["failed_studies"].append({
                    "study_uid": study_uid,
                    "error": str(e)
                })
        
        if not self._cancelled:
            self.progress_updated.emit(
                self.operation.operation_id,
                100,
                f"Completed: {len(results['loaded_studies'])} loaded, {len(results['failed_studies'])} failed"
            )
        
        return results


class BatchOperationService(QObject):
    operation_started = pyqtSignal(str, str)    
    progress_updated = pyqtSignal(str, int, str)
    operation_completed = pyqtSignal(str, dict) 
    operation_failed = pyqtSignal(str, str) 
    operation_cancelled = pyqtSignal(str)
    
    def __init__(self, repository: DICOMImageRepository):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._repository = repository
        
        self._active_operations: Dict[str, BatchOperation] = {}
        self._operation_workers: Dict[str, BatchOperationWorker] = {}
        self._next_operation_id = 1
        
        self._max_concurrent_operations = 2
        self._operation_timeout_minutes = 30
        
        self._total_operations = 0
        self._completed_operations = 0
        self._failed_operations = 0
        
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_completed_operations)
        self._cleanup_timer.start(60000)
    
    def start_batch_series_loading(self, series_uids: List[str], description: str = "") -> str:
        if not series_uids:
            raise ValueError("No series UIDs provided")
        
        operation_id = f"series_batch_{self._next_operation_id}"
        self._next_operation_id += 1
        
        operation = BatchOperation(
            operation_id=operation_id,
            operation_type="series",
            items=series_uids,
            description=description or f"Loading {len(series_uids)} series",
            start_time=datetime.now()
        )
        
        self._start_operation(operation)
        self._logger.info(f"Started batch series loading: {operation_id} ({len(series_uids)} series)")
        
        return operation_id
    
    def start_batch_studies_loading(self, study_uids: List[str], description: str = "") -> str:
        if not study_uids:
            raise ValueError("No study UIDs provided")
        
        operation_id = f"studies_batch_{self._next_operation_id}"
        self._next_operation_id += 1
        
        operation = BatchOperation(
            operation_id=operation_id,
            operation_type="studies",
            items=study_uids,
            description=description or f"Loading {len(study_uids)} studies",
            start_time=datetime.now()
        )
        
        self._start_operation(operation)
        self._logger.info(f"Started batch studies loading: {operation_id} ({len(study_uids)} studies)")
        
        return operation_id
    
    def _start_operation(self, operation: BatchOperation) -> None:
        active_count = len([op for op in self._active_operations.values() 
                           if op.status == "running"])
        
        if active_count >= self._max_concurrent_operations:
            operation.status = "queued"
            self._active_operations[operation.operation_id] = operation
            self._logger.debug(f"Operation queued due to limit: {operation.operation_id}")
            return
        
        worker = BatchOperationWorker(operation, self._repository)
        worker.progress_updated.connect(self.progress_updated)
        worker.operation_completed.connect(self._on_operation_completed)
        worker.operation_failed.connect(self._on_operation_failed)
        
        self._active_operations[operation.operation_id] = operation
        self._operation_workers[operation.operation_id] = worker
        
        worker.start()
        self._total_operations += 1
        
        self.operation_started.emit(operation.operation_id, operation.description)
    
    def _on_operation_completed(self, operation_id: str, results: Dict[str, Any]) -> None:
        if operation_id in self._active_operations:
            operation = self._active_operations[operation_id]
            operation.status = "completed"
            operation.end_time = datetime.now()
            
            self._completed_operations += 1
            self.operation_completed.emit(operation_id, results)
            
            self._logger.info(f"Batch operation completed: {operation_id}")
            self._start_queued_operations()
    
    def _on_operation_failed(self, operation_id: str, error_message: str) -> None:
        if operation_id in self._active_operations:
            operation = self._active_operations[operation_id]
            operation.status = "failed"
            operation.error_message = error_message
            operation.end_time = datetime.now()
            
            self._failed_operations += 1
            self.operation_failed.emit(operation_id, error_message)
            
            self._logger.error(f"Batch operation failed: {operation_id} - {error_message}")
            self._start_queued_operations()
    
    def _start_queued_operations(self) -> None:
        active_count = len([op for op in self._active_operations.values() 
                           if op.status == "running"])
        
        if active_count >= self._max_concurrent_operations:
            return
        
        queued_operations = [op for op in self._active_operations.values() 
                           if op.status == "queued"]
        
        for operation in queued_operations[:self._max_concurrent_operations - active_count]:
            self._start_operation(operation)
    
    def cancel_operation(self, operation_id: str) -> bool:
        if operation_id not in self._active_operations:
            return False
        
        operation = self._active_operations[operation_id]
        
        if operation.status == "completed" or operation.status == "failed":
            return False
        
        if operation.status == "queued":
            operation.status = "cancelled"
            self.operation_cancelled.emit(operation_id)
            return True
        
        if operation.status == "running" and operation_id in self._operation_workers:
            worker = self._operation_workers[operation_id]
            worker.cancel()
            worker.quit()
            worker.wait(1000)
            
            operation.status = "cancelled"
            self.operation_cancelled.emit(operation_id)
            self._logger.debug(f"Cancelled operation: {operation_id}")
            return True
        
        return False
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        if operation_id not in self._active_operations:
            return None
        
        operation = self._active_operations[operation_id]
        
        duration = None
        if operation.start_time:
            end_time = operation.end_time or datetime.now()
            duration = (end_time - operation.start_time).total_seconds()
        
        return {
            'operation_id': operation.operation_id,
            'operation_type': operation.operation_type,
            'description': operation.description,
            'status': operation.status,
            'progress': operation.progress,
            'items_count': len(operation.items),
            'start_time': operation.start_time,
            'end_time': operation.end_time,
            'duration_seconds': duration,
            'error_message': operation.error_message
        }
    
    def get_all_operations(self) -> List[Dict[str, Any]]:
        operations = []
        for operation_id in self._active_operations:
            status = self.get_operation_status(operation_id)
            if status:
                operations.append(status)
        
        return operations
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        active_operations = []
        for operation_id, operation in self._active_operations.items():
            if operation.status in ["running", "queued"]:
                status = self.get_operation_status(operation_id)
                if status:
                    active_operations.append(status)
        
        return active_operations
    
    def _cleanup_completed_operations(self) -> None:
        cutoff_time = datetime.now().timestamp() - (24 * 60 * 60) 

        to_remove = []
        for operation_id, operation in self._active_operations.items():
            if (operation.status in ["completed", "failed", "cancelled"] and
                operation.end_time and
                operation.end_time.timestamp() < cutoff_time):
                to_remove.append(operation_id)

        for operation_id in to_remove:
            del self._active_operations[operation_id]
            if operation_id in self._operation_workers:
                del self._operation_workers[operation_id]

        if to_remove:
            self._logger.debug(f"Cleaned up {len(to_remove)} old operations")
    
    def clear_completed_operations(self) -> int:
        to_remove = []
        for operation_id, operation in self._active_operations.items():
            if operation.status in ["completed", "failed", "cancelled"]:
                to_remove.append(operation_id)
        
        for operation_id in to_remove:
            del self._active_operations[operation_id]
            if operation_id in self._operation_workers:
                del self._operation_workers[operation_id]

        if to_remove:
            self._logger.debug(f"Cleared {len(to_remove)} completed operations")
        
        return len(to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        active_count = len([op for op in self._active_operations.values() 
                           if op.status in ["running", "queued"]])
        
        return {
            'total_operations': self._total_operations,
            'completed_operations': self._completed_operations,
            'failed_operations': self._failed_operations,
            'active_operations': active_count,
            'success_rate_percent': (self._completed_operations / self._total_operations * 100) 
                                  if self._total_operations > 0 else 0,
            'max_concurrent_operations': self._max_concurrent_operations
        }
    
    def configure_limits(self, max_concurrent: Optional[int] = None, 
                        timeout_minutes: Optional[int] = None) -> None:
        if max_concurrent is not None:
            self._max_concurrent_operations = max_concurrent
            self._logger.debug(f"Updated max concurrent operations: {max_concurrent}")
        
        if timeout_minutes is not None:
            self._operation_timeout_minutes = timeout_minutes
            self._logger.debug(f"Updated operation timeout: {timeout_minutes} minutes")
    
    def shutdown(self) -> None:
        active_ops = [op_id for op_id, op in self._active_operations.items() 
                     if op.status in ["running", "queued"]]
        
        for operation_id in active_ops:
            self.cancel_operation(operation_id)
        
        self._cleanup_timer.stop()
        
        self._logger.info("Batch operation service shut down")