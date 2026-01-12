"""Task management system for long-running operations across codebases."""

from __future__ import annotations
import uuid
import threading
import json
from enum import Enum
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from _mcp.logger import app_logger as logger
from _mcp.context import get_current_data, set_current_data


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    task_id: str
    codebase_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    future: Optional[Future] = None


class TaskQueue:
    """Thread-safe task queue manager."""
    
    def __init__(self, max_workers: int = 5):
        """Initialize task queue with thread pool.
        
        Args:
            max_workers: Maximum number of concurrent tasks
        """
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_handlers: Dict[str, Callable] = {}
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"✅ Task queue initialized with {max_workers} workers")
    
    def register_handler(self, task_type: str, handler: Callable, description: Optional[str] = None, params_schema: Optional[Dict[str, Any]] = None):
        """Register a handler function for a task type.
        
        Args:
            task_type: Type of task
            handler: Function that executes the task
            description: Optional description of what the task does
            params_schema: Optional JSON schema for task parameters
        """
        self.task_handlers[task_type] = handler
        self.task_metadata[task_type] = {
            "description": description or f"Task type: {task_type}",
            "params_schema": params_schema or {}
        }
        logger.info(f"📝 Registered handler for task type: {task_type}")
    
    def get_available_task_types(self) -> Dict[str, Any]:
        """Get all available task types with their metadata."""
        return self.task_metadata
    
    def _enforce_codebase_limits(self, codebase_id: str):
        """Enforce per-codebase task limits: 1 in-progress, 2 pending, 3 completed."""
        with self.lock:
            codebase_tasks = [t for t in self.tasks.values() if t.codebase_id == codebase_id]
            
            running_tasks = [t for t in codebase_tasks if t.status == TaskStatus.RUNNING]
            pending_tasks = [t for t in codebase_tasks if t.status == TaskStatus.PENDING]
            completed_tasks = [t for t in codebase_tasks 
                             if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
            
            if len(running_tasks) > 1:
                for task in sorted(running_tasks, key=lambda t: t.started_at or t.created_at)[1:]:
                    if task.future:
                        task.future.cancel()
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    logger.warning(f"🛑 Cancelled excess running task {task.task_id} for codebase {codebase_id}")
            
            if len(pending_tasks) > 2:
                for task in sorted(pending_tasks, key=lambda t: t.created_at)[2:]:
                    if task.future:
                        task.future.cancel()
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    logger.info(f"🛑 Cancelled excess pending task {task.task_id} for codebase {codebase_id}")
            
            if len(completed_tasks) > 3:
                sorted_completed = sorted(
                    completed_tasks,
                    key=lambda t: t.completed_at or t.created_at,
                    reverse=True
                )
                for task in sorted_completed[3:]:
                    del self.tasks[task.task_id]
                    logger.info(f"🗑️  Removed old completed task {task.task_id} for codebase {codebase_id}")
    
    def submit_task(
        self,
        codebase_id: str,
        task_type: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task to the queue."""
        self._enforce_codebase_limits(codebase_id)
        
        task_id = str(uuid.uuid4())
        normalized_task_type = task_type.upper()
        
        if normalized_task_type in self.task_handlers:
            handler = self.task_handlers[normalized_task_type]
        elif task_type in self.task_handlers:
            handler = self.task_handlers[task_type]
        else:
            task = Task(
                task_id=task_id,
                codebase_id=codebase_id,
                task_type=normalized_task_type,
                status=TaskStatus.FAILED,
                error=f"Unknown task type: {task_type}",
                completed_at=datetime.now(),
                metadata=metadata or {}
            )
            with self.lock:
                self.tasks[task_id] = task
            logger.error(f"❌ Unknown task type: {task_type}")
            return task_id
        
        with self.lock:
            codebase_tasks = [t for t in self.tasks.values() if t.codebase_id == codebase_id]
            codebase_pending = [t for t in codebase_tasks if t.status == TaskStatus.PENDING]
            
            if len(codebase_pending) >= 2:
                task = Task(
                    task_id=task_id,
                    codebase_id=codebase_id,
                    task_type=normalized_task_type,
                    status=TaskStatus.FAILED,
                    error="Too many pending tasks. Maximum 2 pending tasks allowed per codebase.",
                    completed_at=datetime.now(),
                    metadata=metadata or {}
                )
                self.tasks[task_id] = task
                logger.warning(f"⚠️  Rejected task submission for codebase {codebase_id}: too many pending tasks ({len(codebase_pending)}/2)")
                return task_id
        
        task = Task(
            task_id=task_id,
            codebase_id=codebase_id,
            task_type=normalized_task_type,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        future = self.executor.submit(self._execute_task, task_id, handler, params)
        
        with self.lock:
            task.future = future
        
        logger.info(f"📤 Submitted task {task_id} (type: {task_type}, codebase: {codebase_id})")
        return task_id
    
    def _execute_task(self, task_id: str, handler: Callable, params: Dict[str, Any]):
        """Execute a task (internal method)."""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"❌ Task {task_id} not found")
                return
            
            if task.status == TaskStatus.CANCELLED:
                logger.info(f"⏭️  Task {task_id} was cancelled before execution")
                return
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            codebase_id = task.codebase_id
        
        self._enforce_codebase_limits(codebase_id)
        
        thread_name = threading.current_thread().name
        logger.info(f"▶️  Starting task {task_id} in thread {thread_name}")
        
        try:
            context_data = {
                "codebase_id": task.codebase_id,
                "request_id": task.metadata.get("request_id"),
                "task_id": task_id
            }
            set_current_data(context_data)
            
            result = handler(**params)
            
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    logger.warning(f"⚠️  Task {task_id} was removed during execution")
                    return
                
                if task.status == TaskStatus.CANCELLED:
                    logger.info(f"⏭️  Task {task_id} was cancelled during execution, ignoring result")
                    return
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                task.progress = 1.0
                task.future = None
                codebase_id = task.codebase_id
            
            self._enforce_codebase_limits(codebase_id)
            logger.info(f"✅ Task {task_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Task {task_id} failed: {error_msg}")
            
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return
                
                if task.status == TaskStatus.CANCELLED:
                    logger.info(f"⏭️  Task {task_id} was cancelled, ignoring error")
                    return
                
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error_msg
                task.future = None
                codebase_id = task.codebase_id
            
            self._enforce_codebase_limits(codebase_id)
        finally:
            try:
                set_current_data({})
            except Exception:
                pass
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_progress(self, task_id: str, progress: float):
        """Update task progress (0.0 to 1.0)."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.progress = max(0.0, min(1.0, progress))
                logger.debug(f"📊 Task {task_id} progress: {task.progress * 100:.1f}%")
    
    def get_tasks_by_codebase(self, codebase_id: str) -> list[Task]:
        """Get all tasks for a codebase."""
        with self.lock:
            return [task for task in self.tasks.values() if task.codebase_id == codebase_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            if task.future:
                cancelled = task.future.cancel()
                if cancelled:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    task.future = None
                    logger.info(f"🛑 Task {task_id} cancelled (before execution)")
                    return True
                else:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    task.error = "Task was cancelled while running"
                    logger.info(f"🛑 Task {task_id} marked as cancelled (was running)")
                    return True
            else:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"🛑 Task {task_id} cancelled (pending)")
                return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        with self.lock:
            total = len(self.tasks)
            by_status = {}
            for status in TaskStatus:
                by_status[status.value] = sum(
                    1 for t in self.tasks.values() if t.status == status
                )
            
            return {
                "total_tasks": total,
                "by_status": by_status,
                "active_workers": self.executor._max_workers,
                "active_tasks": by_status.get(TaskStatus.RUNNING.value, 0)
            }
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the task queue executor gracefully."""
        logger.info("🛑 Shutting down task queue executor...")
        if wait:
            with self.lock:
                for task in list(self.tasks.values()):
                    if task.status == TaskStatus.PENDING and task.future:
                        task.future.cancel()
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = datetime.now()
                        logger.info(f"🛑 Cancelled pending task {task.task_id} during shutdown")
        
        self.executor.shutdown(wait=wait)
        logger.info("✅ Task queue executor shut down")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        import os
        max_workers = int(os.getenv("TASK_QUEUE_WORKERS", "5"))
        _task_queue = TaskQueue(max_workers=max_workers)
    return _task_queue


def submit_task(
    codebase_id: str,
    task_type: str,
    params: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Submit a task to the queue."""
    queue = get_task_queue()
    task_id = queue.submit_task(codebase_id, task_type, params, metadata)
    
    result = get_task_results(codebase_id)
    result["newly_created_task_id"] = task_id
    
    return result


def get_task_status(task_id: str, codebase_id: str) -> Dict[str, Any]:
    """Get task status for a specific codebase.

    This function requires an explicit `codebase_id` and will not fall
    back to request context.
    """
    current_codebase_id = codebase_id

    if not current_codebase_id:
        return {
            "success": False,
            "error": "No codebase_id provided"
        }
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    if current_codebase_id and task.codebase_id != current_codebase_id:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    result = {
        "success": True,
        "task_id": task.task_id,
        "codebase_id": task.codebase_id,
        "task_type": task.task_type,
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "progress": task.progress,
        "metadata": task.metadata
    }
    
    if task.result:
        result["result"] = task.result
    
    if task.error:
        result["error"] = task.error
    
    return result


def cancel_task(task_id: str, codebase_id: str) -> Dict[str, Any]:
    """Cancel a task for a given `codebase_id`.

    `codebase_id` is required; there is no fallback to request context.
    """
    current_codebase_id = codebase_id

    if not current_codebase_id:
        return {
            "success": False,
            "error": "No codebase_id provided"
        }
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    if current_codebase_id and task.codebase_id != current_codebase_id:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    cancelled = queue.cancel_task(task_id)
    
    if cancelled:
        return get_task_results(current_codebase_id)
    else:
        return {
            "success": False,
            "error": f"Task {task_id} cannot be cancelled (status: {task.status.value})"
        }


def get_queue_status() -> Dict[str, Any]:
    """Get queue status."""
    queue = get_task_queue()
    return {
        "success": True,
        **queue.get_status()
    }


def get_task_results(codebase_id: str) -> Dict[str, Any]:
    """Get all tasks for the given codebase, grouped by status.

    `codebase_id` is required; there is no fallback to request context.
    """
    current_codebase_id = codebase_id

    if not current_codebase_id:
        return {
            "success": False,
            "error": "No codebase_id provided"
        }

    queue = get_task_queue()
    codebase_tasks = queue.get_tasks_by_codebase(current_codebase_id)
    
    pending = []
    running = []
    completed = []
    
    for task in codebase_tasks:
        try:
            task_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status.value,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "progress": task.progress,
            }
            
            if task.error:
                task_data["error"] = task.error
            
            if task.status == TaskStatus.PENDING:
                pending.append(task_data)
            elif task.status == TaskStatus.RUNNING:
                running.append(task_data)
            elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.result:
                    try:
                        json.dumps(task.result)
                        task_data["result"] = task.result
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Task {task.task_id} result is not JSON-serializable: {e}")
                        task_data["result"] = {"error": "Result data is not serializable"}
                completed.append(task_data)
        except Exception as e:
            logger.error(f"Error serializing task {task.task_id}: {e}", exc_info=True)
            continue
    
    completed = sorted(
        completed,
        key=lambda t: t.get("completed_at") or t.get("created_at") or "",
        reverse=True
    )[:3]
    
    pending = sorted(pending, key=lambda t: t.get("created_at") or "")
    running = sorted(running, key=lambda t: t.get("started_at") or t.get("created_at") or "")
    
    return {
        "success": True,
        "codebase_id": current_codebase_id,
        "pending": pending,
        "running": running,
        "completed": completed,
        "summary": {
            "pending_count": len(pending),
            "running_count": len(running),
            "completed_count": len(completed)
        }
    }


def get_available_task_types() -> Dict[str, Any]:
    """Get available task types with their parameter schemas."""
    queue = get_task_queue()
    task_types = queue.get_available_task_types()
    
    return {
        "success": True,
        "task_types": task_types,
        "message": f"Found {len(task_types)} available task type(s)"
    }
