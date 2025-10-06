import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal, QTimer


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowProgress:
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    current_step: int = 0
    total_steps: int = 1
    progress_percentage: float = 0.0
    current_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker(QObject):
    workflow_started = pyqtSignal(str, str)  
    workflow_progress_updated = pyqtSignal(str, int, str)  
    workflow_completed = pyqtSignal(str, bool) 
    workflow_step_completed = pyqtSignal(str, int, str) 

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        
        import threading
        self._lock = threading.Lock()
        
        self._workflows: Dict[str, WorkflowProgress] = {}
        self._workflow_counter = 0
        
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_estimates)
        self._update_timer.start(1000)  
        
        self._logger.info("ProgressTracker initialized")
    
    def create_workflow(self, workflow_name: str, total_steps: int = 1,
                       workflow_id: Optional[str] = None) -> str:
        with self._lock:
            if workflow_id is None:
                self._workflow_counter += 1
                workflow_id = f"workflow_{self._workflow_counter}_{uuid.uuid4().hex[:8]}"
            
            workflow_progress = WorkflowProgress(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                status=WorkflowStatus.PENDING,
                total_steps=max(1, total_steps),
                start_time=datetime.now()
            )
            
            self._workflows[workflow_id] = workflow_progress
            self._logger.info(f"Created workflow: {workflow_id} - {workflow_name}")
            
            # Emit signal
            self.workflow_started.emit(workflow_id, workflow_name)
            
            return workflow_id
    
    def start_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            if workflow_id not in self._workflows:
                self._logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow = self._workflows[workflow_id]
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            
            self._logger.info(f"Started workflow: {workflow_id}")
            return True
    
    def update_workflow_progress(self, workflow_id: str, 
                               current_step: Optional[int] = None,
                               progress_percentage: Optional[float] = None,
                               message: str = "") -> bool:
        with self._lock:
            if workflow_id not in self._workflows:
                self._logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow = self._workflows[workflow_id]
            
            if current_step is not None:
                workflow.current_step = min(current_step, workflow.total_steps)
            
            if progress_percentage is not None:
                workflow.progress_percentage = max(0, min(100, progress_percentage))
            else:
                workflow.progress_percentage = (workflow.current_step / workflow.total_steps) * 100
            
            workflow.current_message = message
            
            if workflow.status == WorkflowStatus.RUNNING and workflow.start_time:
                elapsed = datetime.now() - workflow.start_time
                if workflow.progress_percentage > 5:  
                    total_estimated = elapsed * (100 / workflow.progress_percentage)
                    workflow.estimated_completion = workflow.start_time + total_estimated
            
            self._logger.debug(f"Updated workflow {workflow_id}: {workflow.progress_percentage:.1f}% - {message}")
            
            self.workflow_progress_updated.emit(
                workflow_id, 
                int(workflow.progress_percentage), 
                message
            )
            
            return True
    
    def complete_workflow_step(self, workflow_id: str, step_name: str = "") -> bool:
        with self._lock:
            if workflow_id not in self._workflows:
                self._logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow = self._workflows[workflow_id]
            workflow.current_step += 1
            workflow.progress_percentage = (workflow.current_step / workflow.total_steps) * 100
            
            step_message = f"Completed step {workflow.current_step}/{workflow.total_steps}"
            if step_name:
                step_message += f": {step_name}"
            
            workflow.current_message = step_message
            self._logger.info(f"Completed step in workflow {workflow_id}: {step_message}")
            
            self.workflow_step_completed.emit(workflow_id, workflow.current_step, step_name)
            self.workflow_progress_updated.emit(
                workflow_id,
                int(workflow.progress_percentage),
                step_message
            )
            
            if workflow.current_step >= workflow.total_steps:
                return self.complete_workflow(workflow_id, success=True)
            
            return True
    
    def complete_workflow(self, workflow_id: str, success: bool = True, 
                         error_message: Optional[str] = None) -> bool:
        with self._lock:
            if workflow_id not in self._workflows:
                self._logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow = self._workflows[workflow_id]
            
            workflow.end_time = datetime.now()
            workflow.error_message = error_message
            
            if success:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.progress_percentage = 100.0
                workflow.current_message = "Completed successfully"
                self._logger.info(f"Workflow completed successfully: {workflow_id}")
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.current_message = error_message or "Failed"
                self._logger.error(f"Workflow failed: {workflow_id} - {error_message}")
            
            self.workflow_completed.emit(workflow_id, success)
            
            return True
    
    def cancel_workflow(self, workflow_id: str, reason: str = "") -> bool:
        with self._lock:
            if workflow_id not in self._workflows:
                self._logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow = self._workflows[workflow_id]
            
            if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                self._logger.info(f"Cannot cancel workflow in status {workflow.status}: {workflow_id}")
                return False
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = datetime.now()
            workflow.current_message = f"Cancelled: {reason}" if reason else "Cancelled"
            workflow.error_message = reason
            
            self._logger.info(f"Workflow cancelled: {workflow_id} - {reason}")
            self.workflow_completed.emit(workflow_id, False)
            
            return True
    
    def get_workflow_progress(self, workflow_id: str) -> Optional[WorkflowProgress]:
        with self._lock:
            return self._workflows.get(workflow_id)
    
    def get_active_workflows(self) -> List[WorkflowProgress]:
        with self._lock:
            return [
                workflow for workflow in self._workflows.values()
                if workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
            ]
    
    def get_all_workflows(self) -> List[WorkflowProgress]:
        with self._lock:
            return list(self._workflows.values())
    
    def remove_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            if workflow_id in self._workflows:
                del self._workflows[workflow_id]
                self._logger.debug(f"Removed workflow: {workflow_id}")
                return True
            return False
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._lock:
            workflows_to_remove = []
            
            for workflow_id, workflow in self._workflows.items():
                if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                    workflow.end_time and workflow.end_time < cutoff_time):
                    workflows_to_remove.append(workflow_id)
            
            for workflow_id in workflows_to_remove:
                del self._workflows[workflow_id]
                removed_count += 1
            
            if removed_count > 0:
                self._logger.info(f"Cleaned up {removed_count} old workflows")
        
        return removed_count
    
    def _update_estimates(self) -> None:
        try:
            with self._lock:
                for workflow in self._workflows.values():
                    if (workflow.status == WorkflowStatus.RUNNING and 
                        workflow.start_time and 
                        workflow.progress_percentage > 5):
                        
                        elapsed = datetime.now() - workflow.start_time
                        total_estimated = elapsed * (100 / workflow.progress_percentage)
                        workflow.estimated_completion = workflow.start_time + total_estimated
                        
        except Exception as e:
            self._logger.error(f"Error updating workflow estimates: {e}")
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = {
                'total_workflows': len(self._workflows),
                'active_workflows': 0,
                'completed_workflows': 0,
                'failed_workflows': 0,
                'cancelled_workflows': 0,
                'average_completion_time': None,
                'success_rate': 0.0
            }
            
            completion_times = []
            
            for workflow in self._workflows.values():
                if workflow.status == WorkflowStatus.RUNNING or workflow.status == WorkflowStatus.PENDING:
                    stats['active_workflows'] += 1
                elif workflow.status == WorkflowStatus.COMPLETED:
                    stats['completed_workflows'] += 1
                    if workflow.start_time and workflow.end_time:
                        completion_time = (workflow.end_time - workflow.start_time).total_seconds()
                        completion_times.append(completion_time)
                elif workflow.status == WorkflowStatus.FAILED:
                    stats['failed_workflows'] += 1
                elif workflow.status == WorkflowStatus.CANCELLED:
                    stats['cancelled_workflows'] += 1
            
            if completion_times:
                stats['average_completion_time'] = sum(completion_times) / len(completion_times)
            
            total_finished = stats['completed_workflows'] + stats['failed_workflows']
            if total_finished > 0:
                stats['success_rate'] = stats['completed_workflows'] / total_finished
            
            return stats