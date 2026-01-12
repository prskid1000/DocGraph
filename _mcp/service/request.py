"""HTTP request utilities for task management API endpoints."""

from __future__ import annotations
from typing import Any, Dict, Optional
from _mcp.context import get_current_data
from _mcp.logger import app_logger as logger


def get_task_status() -> Dict[str, Any]:
    """Get task status for the current codebase via API."""
    from _mcp.service.task.task_manager import get_task_results
    
    codebase_id = get_current_data('codebase_id')
    
    if not codebase_id:
        return {
            "success": False,
            "error": "No codebase_id context available"
        }
    
    try:
        result = get_task_results()
        return result
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def cancel_task_request(task_id: str) -> Dict[str, Any]:
    """Cancel a task via API."""
    from _mcp.service.task.task_manager import cancel_task
    
    current_codebase_id = get_current_data('codebase_id')
    
    if not current_codebase_id:
        return {
            "success": False,
            "error": "No codebase_id context available"
        }
    
    try:
        result = cancel_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return {
            "success": False,
            "error": str(e)
        }
