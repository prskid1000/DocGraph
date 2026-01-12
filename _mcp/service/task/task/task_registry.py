"""Task handler registration for the task management system."""

from __future__ import annotations
from logger import app_logger as logger
from service.task.task_manager import get_task_queue


def register_task_handlers():
    """Register all task handlers with the task queue."""
    queue = get_task_queue()
    
    logger.info("✅ Task handlers registered successfully")
    
    # Add more task handler registrations here as needed
    # Example:
    # queue.register_handler(
    #     "ANOTHER_TASK_TYPE",
    #     another_task_handler,
    #     description="Description of what this task does",
    #     params_schema={...}
    # )

