"""Request context management for DocGraph MCP Server."""

import threading
from typing import Optional, Any

# Thread-local storage for current request
_thread_local = threading.local()


def set_current_data(data: dict):
    """Set the current request data in thread-local storage."""
    _thread_local.data = data


def get_current_data(attribute: str) -> Any:
    """Get a specific attribute from the current request data."""
    data = getattr(_thread_local, 'data', None)
    if data and attribute in data:
        return data[attribute]
    return None


def clear_current_data():
    """Clear the current request data."""
    if hasattr(_thread_local, 'data'):
        delattr(_thread_local, 'data')
