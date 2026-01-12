"""Task management module for handling asynchronous tasks."""

from _mcp.service.task.task_manager import (
    submit_task,
    get_task_results,
    cancel_task,
    get_queue_status,
    get_task_queue,
    TaskStatus
)
from _mcp.service.task.task_registry import register_task_handlers
from _mcp.service.task.task_utils import (
    update_progress,
    is_cancelled,
    get_task_id,
    report_stage,
    report_item_progress
)

__all__ = [
    'submit_task',
    'get_task_results',
    'cancel_task',
    'get_queue_status',
    'get_task_queue',
    'TaskStatus',
    'register_task_handlers',
    'update_progress',
    'is_cancelled',
    'get_task_id',
    'report_stage',
    'report_item_progress'
]
