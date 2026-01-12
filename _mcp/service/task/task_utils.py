"""Task management utilities for task handlers."""

from __future__ import annotations
from typing import Optional
from _mcp.context import get_current_data
from _mcp.service.task.task_manager import get_task_queue, TaskStatus
from _mcp.logger import app_logger as logger


def update_progress(progress: float, message: Optional[str] = None) -> bool:
    """Update progress for the current task."""
    task_id = get_current_data('task_id')
    if not task_id:
        logger.warning("⚠️  Cannot update progress: no task_id in context")
        return False
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        logger.warning(f"⚠️  Cannot update progress: task {task_id} not found")
        return False
    
    if task.status == TaskStatus.CANCELLED:
        logger.debug(f"⚠️  Task {task_id} is cancelled, ignoring progress update")
        return False
    
    queue.update_progress(task_id, progress)
    
    if message:
        logger.info(f"📊 Task {task_id}: {progress * 100:.1f}% - {message}")
    else:
        logger.debug(f"📊 Task {task_id}: {progress * 100:.1f}%")
    
    return True


def is_cancelled() -> bool:
    """Check if the current task has been cancelled."""
    task_id = get_current_data('task_id')
    if not task_id:
        return False
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        return False
    
    return task.status == TaskStatus.CANCELLED


def get_task_id() -> Optional[str]:
    """Get the current task ID from context."""
    return get_current_data('task_id')


def report_stage(stage: str, progress: float, message: Optional[str] = None) -> bool:
    """Report progress for a specific stage of task execution."""
    if message:
        return update_progress(progress, f"[{stage.upper()}] {message}")
    else:
        return update_progress(progress, f"Stage: {stage}")


def report_item_progress(current: int, total: int, stage: str = "processing") -> bool:
    """Report progress based on item count."""
    if total == 0:
        return update_progress(1.0, f"{stage}: No items to process")
    
    progress = min(1.0, current / total)
    message = f"{stage}: {current}/{total} items"
    
    return update_progress(progress, message)
