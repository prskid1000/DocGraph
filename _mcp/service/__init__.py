"""Init file for MCP service module."""

from _mcp.service.tools import get_tool_definition, get_all_tool_definitions
from _mcp.service.handlers import (
    list_tools,
    list_resources,
    handle_read_resource,
    handle_call_tool,
    list_prompts,
    handle_get_prompt
)

__all__ = [
    "get_tool_definition",
    "get_all_tool_definitions",
    "list_tools",
    "list_resources",
    "handle_read_resource",
    "handle_call_tool",
    "list_prompts",
    "handle_get_prompt",
]
