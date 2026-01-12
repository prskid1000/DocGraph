"""Init file for MCP service module."""


from _mcp.service.tools import *
from _mcp.service.handlers import *
from _mcp.service.prompts import *
from _mcp.service.cloudwatch_logger import *
from _mcp.service.widgets import *
from _mcp.service.request import *

# Export all functions from all modules
from _mcp.service.tools import (
    ToolDefinition, get_tool_definition, get_all_tool_definitions,
    search_code_entities_handler, get_definition_handler, find_references_handler,
    get_call_graph_handler, get_dependencies_handler, query_graph_handler,
    get_context_handler, submit_task_handler, get_task_results_handler, cancel_task_handler,
    TOOL_DEFINITIONS
)
from _mcp.service.handlers import (
    get_widget_actual_filename, get_widget_base_url, get_widget_src_url, get_tool_meta,
    get_embedded_widget_resource, list_tools, list_resources, list_resource_templates,
    handle_read_resource, generate_tools_documentation, handle_call_tool, list_prompts,
    handle_get_prompt
)
from _mcp.service.prompts import (
    PromptArgument, PromptDefinition, PROMPT_DEFINITIONS, get_prompt_definition,
    list_prompts as list_prompt_definitions, get_prompt_message
)
from _mcp.service.cloudwatch_logger import log_to_cloudwatch
from _mcp.service.widgets import Widget, widgets
from _mcp.service.request import get_task_status, cancel_task_request

__all__ = [
    # tools.py
    "ToolDefinition", "get_tool_definition", "get_all_tool_definitions",
    "search_code_entities_handler", "get_definition_handler", "find_references_handler",
    "get_call_graph_handler", "get_dependencies_handler", "query_graph_handler",
    "get_context_handler", "submit_task_handler", "get_task_results_handler", "cancel_task_handler",
    "TOOL_DEFINITIONS",
    # handlers.py
    "get_widget_actual_filename", "get_widget_base_url", "get_widget_src_url", "get_tool_meta",
    "get_embedded_widget_resource", "list_tools", "list_resources", "list_resource_templates",
    "handle_read_resource", "generate_tools_documentation", "handle_call_tool", "list_prompts",
    "handle_get_prompt",
    # prompts.py
    "PromptArgument", "PromptDefinition", "PROMPT_DEFINITIONS", "get_prompt_definition",
    "list_prompt_definitions", "get_prompt_message",
    # cloudwatch_logger.py
    "log_to_cloudwatch",
    # widgets.py
    "Widget", "widgets",
    # request.py
    "get_task_status", "cancel_task_request",
]
