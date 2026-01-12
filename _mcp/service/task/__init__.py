"""Task management module for handling asynchronous tasks (DocGraph MCP)."""


from .task_manager import *
from .task_utils import *
from .task_registry import *

from .task_manager import (
    TaskStatus, Task, TaskQueue, get_task_queue, submit_task, get_task_status, cancel_task,
    get_queue_status, get_task_results, get_available_task_types
)
from .task_utils import (
    update_progress, is_cancelled, get_task_id, report_stage, report_item_progress
)
from .task_registry import (
    index_codebase_handler, register_task_handlers
)

__all__ = [
    # task_manager.py
    "TaskStatus", "Task", "TaskQueue", "get_task_queue", "submit_task", "get_task_status", "cancel_task",
    "get_queue_status", "get_task_results", "get_available_task_types",
    # task_utils.py
    "update_progress", "is_cancelled", "get_task_id", "report_stage", "report_item_progress",
    # task_registry.py
    "index_codebase_handler", "register_task_handlers",
]
