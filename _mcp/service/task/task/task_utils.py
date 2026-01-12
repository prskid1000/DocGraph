"""Task management utilities for task handlers.

This module provides utilities for task handlers to:
- Update task progress in a structured way
- Check if task has been cancelled
- Report progress at different stages
"""

from __future__ import annotations
from typing import Optional
from context import get_current_data
from service.task.task_manager import get_task_queue, TaskStatus
from logger import app_logger as logger


def update_progress(progress: float, message: Optional[str] = None) -> bool:
    """
    Update progress for the current task.
    
    This is the standard way for task handlers to update progress during execution.
    The task_id is automatically retrieved from the current context.
    
    Args:
        progress: Progress value between 0.0 and 1.0 (0% to 100%)
        message: Optional progress message for logging
        
    Returns:
        True if progress was updated, False if task not found or cancelled
        
    Example:
        # Update progress to 25%
        update_progress(0.25, "Downloading images...")
        
        # Update progress to 50%
        update_progress(0.50, "Analyzing image quality...")
    """
    task_id = get_current_data('task_id')
    if not task_id:
        logger.warning("⚠️  Cannot update progress: no task_id in context")
        return False
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        logger.warning(f"⚠️  Cannot update progress: task {task_id} not found")
        return False
    
    # Check if task is cancelled
    if task.status == TaskStatus.CANCELLED:
        logger.debug(f"⚠️  Task {task_id} is cancelled, ignoring progress update")
        return False
    
    # Update progress
    queue.update_progress(task_id, progress)
    
    if message:
        logger.info(f"📊 Task {task_id}: {progress * 100:.1f}% - {message}")
    else:
        logger.debug(f"📊 Task {task_id}: {progress * 100:.1f}%")
    
    return True


def is_cancelled() -> bool:
    """
    Check if the current task has been cancelled.
    
    Task handlers should periodically check this during long-running operations
    and exit gracefully if the task is cancelled.
    
    Returns:
        True if task is cancelled, False otherwise
        
    Example:
        for item in items:
            if is_cancelled():
                logger.info("Task cancelled, stopping processing")
                return {"success": False, "error": "Task was cancelled"}
            
            process_item(item)
            update_progress(processed_count / total_count)
    """
    task_id = get_current_data('task_id')
    if not task_id:
        return False
    
    queue = get_task_queue()
    task = queue.get_task(task_id)
    
    if not task:
        return False
    
    return task.status == TaskStatus.CANCELLED


def get_task_id() -> Optional[str]:
    """
    Get the current task ID from context.
    
    Returns:
        Task ID if available, None otherwise
    """
    return get_current_data('task_id')


def report_stage(stage: str, progress: float, message: Optional[str] = None) -> bool:
    """
    Report progress for a specific stage of task execution.
    
    This is a structured way to report progress at different stages.
    
    Args:
        stage: Stage name (e.g., "query", "download", "analysis", "finalization")
        progress: Progress value between 0.0 and 1.0
        message: Optional message describing the stage
        
    Returns:
        True if progress was updated, False otherwise
        
    Example:
        report_stage("query", 0.1, "Executing SQL query...")
        report_stage("download", 0.3, "Downloading images...")
        report_stage("analysis", 0.7, "Analyzing image quality...")
        report_stage("finalization", 1.0, "Task completed")
    """
    if message:
        return update_progress(progress, f"[{stage.upper()}] {message}")
    else:
        return update_progress(progress, f"Stage: {stage}")


def report_item_progress(current: int, total: int, stage: str = "processing") -> bool:
    """
    Report progress based on item count (e.g., images processed).
    
    This is a convenient helper for tasks that process multiple items.
    
    Args:
        current: Number of items processed so far
        total: Total number of items to process
        stage: Optional stage name for logging
        
    Returns:
        True if progress was updated, False otherwise
        
    Example:
        for idx, image in enumerate(images):
            process_image(image)
            report_item_progress(idx + 1, len(images), "analyzing")
    """
    if total == 0:
        return update_progress(1.0, f"{stage}: No items to process")
    
    progress = min(1.0, current / total)
    message = f"{stage}: {current}/{total} items"
    
    return update_progress(progress, message)

