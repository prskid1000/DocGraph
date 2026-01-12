"""Task management system for long-running operations across tenants."""

from __future__ import annotations
import uuid
import threading
import json
from enum import Enum
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from logger import app_logger as logger
from context import get_current_data, set_current_data


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
    tenant: str
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
        self.lock = threading.RLock()  # Reentrant lock to allow nested lock acquisition
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
        """Get all available task types with their metadata.
        
        Returns:
            Dict mapping task types to their metadata
        """
        return self.task_metadata
    
    def _enforce_tenant_limits(self, tenant: str):
        """Enforce per-tenant task limits: 1 in-progress, 2 pending, 3 completed.
        
        Args:
            tenant: Tenant identifier
        """
        with self.lock:
            tenant_tasks = [t for t in self.tasks.values() if t.tenant == tenant]
            
            # Count tasks by status
            running_tasks = [t for t in tenant_tasks if t.status == TaskStatus.RUNNING]
            pending_tasks = [t for t in tenant_tasks if t.status == TaskStatus.PENDING]
            # Include COMPLETED, FAILED, and CANCELLED in completed count (max 3 total)
            completed_tasks = [t for t in tenant_tasks 
                             if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
            
            # Enforce: max 1 running task
            if len(running_tasks) > 1:
                # Cancel oldest running tasks (keep the most recent one)
                for task in sorted(running_tasks, key=lambda t: t.started_at or t.created_at)[1:]:
                    if task.future:
                        task.future.cancel()
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    logger.warning(f"🛑 Cancelled excess running task {task.task_id} for tenant {tenant}")
            
            # Enforce: max 2 pending tasks
            if len(pending_tasks) > 2:
                # Cancel oldest pending tasks (keep the 2 most recent)
                for task in sorted(pending_tasks, key=lambda t: t.created_at)[2:]:
                    if task.future:
                        task.future.cancel()
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    logger.info(f"🛑 Cancelled excess pending task {task.task_id} for tenant {tenant}")
            
            # Enforce: keep only 3 most recent completed/failed/cancelled tasks
            if len(completed_tasks) > 3:
                # Sort by completed_at (most recent first), remove oldest
                sorted_completed = sorted(
                    completed_tasks,
                    key=lambda t: t.completed_at or t.created_at,
                    reverse=True
                )
                for task in sorted_completed[3:]:
                    del self.tasks[task.task_id]
                    logger.info(f"🗑️  Removed old completed task {task.task_id} for tenant {tenant}")
    
    def submit_task(
        self,
        tenant: str,
        task_type: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task to the queue.
        
        Enforces per-tenant limits: 1 in-progress, 2 pending, 3 completed.
        
        Args:
            tenant: Tenant identifier
            task_type: Type of task
            params: Task parameters
            metadata: Optional metadata
            
        Returns:
            Task ID
        """
        # Enforce limits before submitting
        self._enforce_tenant_limits(tenant)
        
        task_id = str(uuid.uuid4())
        
        # Normalize task_type to uppercase for consistency
        normalized_task_type = task_type.upper()
        
        # Find handler - try uppercase first, then original case
        if normalized_task_type in self.task_handlers:
            handler = self.task_handlers[normalized_task_type]
        elif task_type in self.task_handlers:
            handler = self.task_handlers[task_type]
        else:
            # Create task with failed status
            task = Task(
                task_id=task_id,
                tenant=tenant,
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
        
        # Check tenant limits before submitting
        # Limits: max 1 running, max 2 pending, max 3 completed
        with self.lock:
            tenant_tasks = [t for t in self.tasks.values() if t.tenant == tenant]
            tenant_pending = [t for t in tenant_tasks if t.status == TaskStatus.PENDING]
            tenant_running = [t for t in tenant_tasks if t.status == TaskStatus.RUNNING]
            
            # Check if we can accept more pending tasks (max 2)
            if len(tenant_pending) >= 2:
                # Create task with failed status - too many pending
                task = Task(
                    task_id=task_id,
                    tenant=tenant,
                    task_type=normalized_task_type,
                    status=TaskStatus.FAILED,
                    error="Too many pending tasks. Maximum 2 pending tasks allowed per tenant.",
                    completed_at=datetime.now(),
                    metadata=metadata or {}
                )
                self.tasks[task_id] = task
                logger.warning(f"⚠️  Rejected task submission for tenant {tenant}: too many pending tasks ({len(tenant_pending)}/2)")
                return task_id
            
            # Note: We allow pending tasks even if there's 1 running task
            # The running task limit (max 1) is enforced by _enforce_tenant_limits
            # which will cancel excess running tasks if needed
        
        # Create task with normalized uppercase task_type
        task = Task(
            task_id=task_id,
            tenant=tenant,
            task_type=normalized_task_type,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        # Submit task to thread pool - this will execute in a separate thread
        future = self.executor.submit(self._execute_task, task_id, handler, params)
        
        with self.lock:
            task.future = future
        
        main_thread = threading.current_thread().name
        logger.info(f"📤 Submitted task {task_id} (type: {task_type}, tenant: {tenant}) from thread {main_thread}")
        return task_id
    
    def _execute_task(self, task_id: str, handler: Callable, params: Dict[str, Any]):
        """Execute a task (internal method).
        
        Args:
            task_id: Task ID
            handler: Handler function
            params: Task parameters
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"❌ Task {task_id} not found")
                return
            
            # Check if task was cancelled before starting
            if task.status == TaskStatus.CANCELLED:
                logger.info(f"⏭️  Task {task_id} was cancelled before execution")
                return
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            tenant = task.tenant
        
        # Enforce running task limit (max 1) when task starts
        # (called outside lock to avoid deadlock)
        self._enforce_tenant_limits(tenant)
        
        # Log thread information to verify threads are being used
        thread_name = threading.current_thread().name
        thread_id = threading.get_ident()
        logger.info(f"▶️  Starting task {task_id} in thread {thread_name} (ID: {thread_id})")
        
        try:
            # Set context in worker thread from task metadata
            # This allows API calls within the task to access tenant and access_token
            context_data = {
                "tenant": task.tenant,
                "access_token": task.metadata.get("access_token"),
                "username": task.metadata.get("username"),
                "task_id": task_id  # Include task_id so handlers can update progress
            }
            set_current_data(context_data)
            logger.debug(f"🔑 Context set for task {task_id} - Tenant: {task.tenant} (Thread: {thread_name})")
            
            # Execute the handler
            result = handler(**params)
            
            # Check if task was cancelled during execution
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    logger.warning(f"⚠️  Task {task_id} was removed during execution")
                    return
                
                # If task was cancelled, don't overwrite the cancelled status
                if task.status == TaskStatus.CANCELLED:
                    logger.info(f"⏭️  Task {task_id} was cancelled during execution, ignoring result")
                    return
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                task.progress = 1.0
                task.future = None  # Clear future reference
                tenant = task.tenant
            
            # Enforce limits after completion (clean up old completed tasks)
            # Safe to call here even though we're in a lock - RLock allows reentrant acquisition
            self._enforce_tenant_limits(tenant)
            
            logger.info(f"✅ Task {task_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Task {task_id} failed: {error_msg}")
            
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return
                
                # If task was cancelled, don't overwrite the cancelled status
                if task.status == TaskStatus.CANCELLED:
                    logger.info(f"⏭️  Task {task_id} was cancelled, ignoring error")
                    return
                
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error_msg
                task.future = None  # Clear future reference
                tenant = task.tenant
            
            # Enforce limits after failure (clean up old completed tasks)
            self._enforce_tenant_limits(tenant)
        finally:
            # Clean up context data from worker thread
            try:
                set_current_data({})
            except Exception:
                pass
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task object or None
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_progress(self, task_id: str, progress: float):
        """Update task progress (0.0 to 1.0).
        
        Args:
            task_id: Task ID
            progress: Progress value between 0.0 and 1.0
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                # Clamp progress between 0.0 and 1.0
                task.progress = max(0.0, min(1.0, progress))
                logger.debug(f"📊 Task {task_id} progress: {task.progress * 100:.1f}%")
    
    def get_tasks_by_tenant(self, tenant: str) -> list[Task]:
        """Get all tasks for a tenant.
        
        Args:
            tenant: Tenant identifier
            
        Returns:
            List of tasks
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.tenant == tenant]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            if task.future:
                cancelled = task.future.cancel()
                if cancelled:
                    # Task was successfully cancelled before execution
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    task.future = None  # Clear future reference
                    logger.info(f"🛑 Task {task_id} cancelled (before execution)")
                    return True
                else:
                    # Task is already running - mark as cancelled
                    # The task thread will check this status and exit gracefully
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    task.error = "Task was cancelled while running"
                    # Don't clear future here - let the thread finish and clean it up
                    logger.info(f"🛑 Task {task_id} marked as cancelled (was running)")
                    return True
            else:
                # Task hasn't started yet (no future assigned)
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"🛑 Task {task_id} cancelled (pending)")
                return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status.
        
        Returns:
            Status dictionary
        """
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
        """Shutdown the task queue executor gracefully.
        
        Args:
            wait: If True, wait for running tasks to complete
            timeout: Maximum time to wait for shutdown (None = wait indefinitely)
        """
        logger.info("🛑 Shutting down task queue executor...")
        if wait:
            # Cancel all pending tasks
            with self.lock:
                for task in list(self.tasks.values()):
                    if task.status == TaskStatus.PENDING and task.future:
                        task.future.cancel()
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = datetime.now()
                        logger.info(f"🛑 Cancelled pending task {task.task_id} during shutdown")
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        logger.info("✅ Task queue executor shut down")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the global task queue instance.
    
    Returns:
        TaskQueue instance
    """
    global _task_queue
    if _task_queue is None:
        # Default to 5 workers, can be configured via env var
        import os
        max_workers = int(os.getenv("TASK_QUEUE_WORKERS", "5"))
        _task_queue = TaskQueue(max_workers=max_workers)
    return _task_queue


def submit_task(
    tenant: str,
    task_type: str,
    params: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Submit a task to the queue (for tool handlers).
    
    Args:
        tenant: Tenant identifier
        task_type: Type of task
        params: Task parameters
        metadata: Optional metadata
        
    Returns:
        Dict in same format as get_task_results() - includes all tasks grouped by status
        Also includes 'newly_created_task_id' field to identify which task was just created
        This allows the widget to immediately display the newly created task without
        needing a separate get_task_results() call
    """
    queue = get_task_queue()
    task_id = queue.submit_task(tenant, task_type, params, metadata)
    
    # Get all task results in the standard format
    result = get_task_results()
    
    # Add the newly created task_id so widget can identify it
    result["newly_created_task_id"] = task_id
    
    return result


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status (for tool handlers).
    
    Args:
        task_id: Task ID
        
    Returns:
        Dict with task status and result
    """
    # Get current tenant from context for security check
    current_tenant = get_current_data('tenant')
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    # Security check: ensure tenant can only access their own tasks
    if current_tenant and task.tenant != current_tenant:
        return {
            "success": False,
            "error": f"Task {task_id} not found"  # Don't reveal that task exists for other tenant
        }
    
    result = {
        "success": True,
        "task_id": task.task_id,
        "tenant": task.tenant,
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


def cancel_task(task_id: str) -> Dict[str, Any]:
    """Cancel a task (for tool handlers).
    
    Args:
        task_id: Task ID
        
    Returns:
        Dict with updated task list in same format as get_task_results (array format)
    """
    # Get current tenant from context for security check
    current_tenant = get_current_data('tenant')
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    # Security check: ensure tenant can only cancel their own tasks
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found"
        }
    
    if current_tenant and task.tenant != current_tenant:
        return {
            "success": False,
            "error": f"Task {task_id} not found"  # Don't reveal that task exists for other tenant
        }
    
    cancelled = queue.cancel_task(task_id)
    
    if cancelled:
        # Return updated task list in same format as get_task_results
        return get_task_results()
    else:
        return {
            "success": False,
            "error": f"Task {task_id} cannot be cancelled (status: {task.status.value})"
        }


def get_queue_status() -> Dict[str, Any]:
    """Get queue status (for tool handlers).
    
    Returns:
        Dict with queue statistics
    """
    queue = get_task_queue()
    return {
        "success": True,
        **queue.get_status()
    }


def get_task_results() -> Dict[str, Any]:
    """Get all tasks for the current tenant, grouped by status (for tool handlers).
    
    Returns:
        Dict with tasks grouped by status: pending, running, completed
    """
    # Get current tenant from context
    current_tenant = get_current_data('tenant')
    
    if not current_tenant:
        return {
            "success": False,
            "error": "No tenant context available"
        }
    
    queue = get_task_queue()
    tenant_tasks = queue.get_tasks_by_tenant(current_tenant)
    
    # Group tasks by status
    pending = []
    running = []
    completed = []
    
    for task in tenant_tasks:
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
                # Include result for completed tasks (if available)
                # Try to serialize result, but handle non-serializable data gracefully
                if task.result:
                    try:
                        # Try to serialize to ensure it's JSON-compatible
                        json.dumps(task.result)  # Test if serializable
                        task_data["result"] = task.result
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Task {task.task_id} result is not JSON-serializable: {e}")
                        task_data["result"] = {"error": "Result data is not serializable"}
                completed.append(task_data)
        except Exception as e:
            # If serializing a single task fails, log and skip it
            logger.error(f"Error serializing task {task.task_id}: {e}", exc_info=True)
            continue
    
    # Sort completed by completed_at (most recent first), limit to 3
    completed = sorted(
        completed,
        key=lambda t: t.get("completed_at") or t.get("created_at") or "",
        reverse=True
    )[:3]
    
    # Sort pending by created_at (oldest first)
    pending = sorted(pending, key=lambda t: t.get("created_at") or "")
    
    # Sort running by started_at (oldest first)
    running = sorted(running, key=lambda t: t.get("started_at") or t.get("created_at") or "")
    
    return {
        "success": True,
        "tenant": current_tenant,
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
    """Get available task types with their parameter schemas (for tool handlers).
    
    Returns:
        Dict with available task types and their metadata
    """
    queue = get_task_queue()
    task_types = queue.get_available_task_types()
    
    return {
        "success": True,
        "task_types": task_types,
        "message": f"Found {len(task_types)} available task type(s)"
    }

