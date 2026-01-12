"""MCP request handlers for DocGraph MCP Server."""

from __future__ import annotations
from typing import Any, Dict, List
import mcp.types as types
from pydantic import ValidationError
from _mcp.service.tools import get_tool_definition, get_all_tool_definitions, TOOL_DEFINITIONS
from _mcp.service.widgets import Widget, widgets
from _mcp.context import get_current_data
from _mcp.config import Config
import logging
import glob
import pathlib

logger = logging.getLogger(__name__)

MIME_TYPE = "text/html+skybridge"

# Create widget lookup dictionaries
WIDGETS_BY_ID: Dict[str, Widget] = {widget.identifier: widget for widget in widgets}
WIDGETS_BY_URI: Dict[str, Widget] = {widget.template_uri: widget for widget in widgets}


def get_widget_actual_filename(widget_name: str) -> str:
    """Get the actual filename for a widget (handles hashed/versioned filenames).
    
    Uses the same two-tier lookup strategy as widget_loader:
    1. First tries exact match (e.g., 'SearchEntities.html')
    2. Falls back to glob pattern for versioned files (e.g., 'SearchEntities-*.html')
    
    Args:
        widget_name: Base widget name (e.g., 'SearchEntities')
    
    Returns:
        Actual filename (e.g., 'SearchEntities-a11054ff.html' or 'SearchEntities.html')
    """
    assets_dir = pathlib.Path(__file__).parent.parent / "widgets-assets"
    
    if not assets_dir.exists():
        # Fallback to base name if assets dir doesn't exist
        return f"{widget_name}.html"
    
    # Tier 1: Try exact match (e.g., 'SearchEntities.html')
    exact_match_path = assets_dir / f"{widget_name}.html"
    if exact_match_path.exists():
        return f"{widget_name}.html"
    
    # Tier 2: Fall back to glob pattern for versioned files (e.g., 'SearchEntities-*.html')
    glob_pattern = str(assets_dir / f"{widget_name}-*.html")
    matches = sorted(glob.glob(glob_pattern), reverse=True)  # Most recent first
    
    if matches:
        # Return just the filename (not full path)
        return pathlib.Path(matches[0]).name
    
    # Fallback to base name if not found
    return f"{widget_name}.html"


def get_widget_base_url() -> str:
    """Get the base URL for widget assets based on environment."""
    # Use configured host and port for widget server
    if Config.HOST:
        return f"http://{Config.HOST}:{Config.PORT}"
    
    # Fallback to localhost for development
    return "http://localhost:5500"


def get_widget_src_url(widget: Widget) -> str:
    """Get the source URL for a widget.
    
    Returns the actual production URL where the compiled widget HTML asset is hosted.
    This includes the hash/version suffix if present (e.g., SearchEntities-a11054ff.html).
    This should match the URL pattern where widgets-assets are served.
    """
    base_url = get_widget_base_url()
    
    # Extract widget name from template_uri (e.g., 'SearchEntities' from 'ui://widget/SearchEntities.html')
    widget_name = widget.template_uri.replace("ui://widget/", "").replace(".html", "")
    
    # Get the actual filename (with hash if present)
    actual_filename = get_widget_actual_filename(widget_name)
    
    # Return URL to widget asset (e.g., http://localhost:5500/widgets-assets/SearchEntities-a11054ff.html)
    return f"{base_url}/widgets-assets/{actual_filename}"


def get_tool_meta(widget: Widget) -> Dict[str, Any]:
    """Get tool metadata for a widget."""
    widget_src = get_widget_src_url(widget)
    widget_domain = get_widget_base_url()
    
    return {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
        "openai/widgetDomain": widget_domain,
        "openai/widgetCSP": {
            "connect_domains": [],
            "resource_domains": [],
        },
        "type": "iframe",
        "src": widget_src,
        "annotations": {
            "destructiveHint": False,
            "openWorldHint": False,
            "readOnlyHint": True,
        }
    }


def get_embedded_widget_resource(widget: Widget) -> types.EmbeddedResource:
    """Get embedded widget resource."""
    if not widget.html:
        logger.warning(f"⚠️  Widget {widget.identifier} has empty HTML - widget won't render")
    return types.EmbeddedResource(
        type="resource",
        resource=types.TextResourceContents(
            uri=widget.template_uri,
            mimeType=MIME_TYPE,
            text=widget.html,
            title=widget.title,
        ),
    )


async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    all_tools = {t.identifier: t for t in get_all_tool_definitions()}
    tools_list = []
    
    # Add tools with widgets
    for widget in widgets:
        if widget.identifier in all_tools:
            tool_def = all_tools[widget.identifier]
            tool_obj = types.Tool(
                name=widget.identifier,
                title=widget.title,
                description=tool_def.description if tool_def else widget.title,
                inputSchema=tool_def.input_schema if tool_def else {},
                _meta=get_tool_meta(widget),
            )
            tools_list.append(tool_obj)

    # Add tools without widgets
    widget_identifiers = {widget.identifier for widget in widgets}
    for tool_id, tool_def in all_tools.items():
        if tool_id not in widget_identifiers:
            tool_obj = types.Tool(
                name=tool_def.identifier,
                title=tool_def.title,
                description=tool_def.description,
                inputSchema=tool_def.input_schema,
            )
            tools_list.append(tool_obj)

    return tools_list


async def list_resources() -> list[types.Resource]:
    """List available resources (codebase documentation)."""
    codebase_id = get_current_data('codebase_id')
    
    resources = [
        types.Resource(
            uri=f"docgraph://codebase/{codebase_id}",
            name=f"Codebase: {codebase_id}",
            description="Knowledge graph for the indexed codebase",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="docgraph://tools/documentation",
            name="Tools Documentation",
            description="Documentation for all available MCP tools",
            mimeType="text/markdown"
        ),
    ]
    
    return resources


async def list_resource_templates() -> list[types.ResourceTemplate]:
    """List available resource templates (widgets)."""
    return [
        types.ResourceTemplate(
            name=widget.title,
            title=widget.title,
            uriTemplate=widget.template_uri,
            description=f"{widget.title} widget markup",
            mimeType=MIME_TYPE,
            _meta=get_tool_meta(widget),
        )
        for widget in widgets
    ]


async def handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    """Handle resource read requests."""
    uri = str(req.params.uri)
    
    # Check if it's a widget resource
    widget = WIDGETS_BY_URI.get(uri)
    if widget:
        contents = [
            types.TextResourceContents(
                uri=widget.template_uri,
                mimeType=MIME_TYPE,
                text=widget.html,
                _meta=get_tool_meta(widget),
            )
        ]
        return types.ServerResult(types.ReadResourceResult(contents=contents))
    
    # Handle documentation resources
    if uri == "docgraph://tools/documentation":
        content = generate_tools_documentation()
    elif uri.startswith("docgraph://codebase/"):
        codebase_id = uri.replace("docgraph://codebase/", "")
        content = f"Knowledge graph documentation for codebase: {codebase_id}\n\nUse the available tools to query this codebase."
    else:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {uri}"},
            )
        )
    
    return types.ServerResult(
        types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=uri,
                    mimeType="text/plain" if uri.startswith("docgraph://codebase/") else "text/markdown",
                    text=content,
                )
            ]
        )
    )


def generate_tools_documentation() -> str:
    """Generate documentation for all available tools."""
    doc = "# DocGraph MCP Tools Documentation\n\n"
    
    for tool_id, tool_def in TOOL_DEFINITIONS.items():
        doc += f"## {tool_def.title}\n\n"
        doc += f"**ID:** `{tool_def.identifier}`\n\n"
        doc += f"**Description:** {tool_def.description}\n\n"
        doc += "**Parameters:**\n\n"
        
        if "properties" in tool_def.input_schema:
            for prop_name, prop_schema in tool_def.input_schema["properties"].items():
                required = prop_name in tool_def.input_schema.get("required", [])
                req_str = "required" if required else "optional"
                doc += f"- `{prop_name}` ({prop_schema.get('type', 'unknown')}, {req_str}): {prop_schema.get('description', '')}\n"
        
        doc += "\n"
    
    return doc


async def handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    """Handle tool call requests."""
    name = req.params.name
    arguments = req.params.arguments or {}
    
    logger.info(f"Calling tool: {name}")
    logger.debug(f"Arguments: {arguments}")
    
    # Get tool definition
    tool_def = get_tool_definition(name)
    if not tool_def:
        logger.error(f"Tool definition not found: {name}")
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Tool definition not found: {name}",
                    )
                ],
                isError=True,
            )
        )
    
    # Get widget for this tool (optional)
    widget = WIDGETS_BY_ID.get(name)
    
    try:
        # Execute the tool
        result = tool_def.execute(arguments)
        logger.debug(f"Tool result: {result}")
        
        # Check for errors in the result
        if result.get("success") is False or "error" in result:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Tool returned error for {name}: {error_msg}")
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Error: {error_msg}",
                        )
                    ],
                    isError=True,
                )
            )
        
        # If no widget, return structured content only
        if widget is None:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text="Tool executed successfully",
                        )
                    ],
                    structuredContent=result,
                )
            )
        
        # Return with widget metadata
        widget_resource = get_embedded_widget_resource(widget)
        
        # Get CSP configuration and widget URLs
        widget_src = get_widget_src_url(widget)
        widget_domain = get_widget_base_url()
        
        meta: Dict[str, Any] = {
            "openai.com/widget": widget_resource.model_dump(mode="json"),
            "openai/outputTemplate": widget.template_uri,
            "openai/toolInvocation/invoking": widget.invoking,
            "openai/toolInvocation/invoked": widget.invoked,
            "openai/widgetAccessible": True,
            "openai/resultCanProduceWidget": True,
            "openai/widgetDomain": widget_domain,
            "openai/widgetCSP": {
                "connect_domains": [],
                "resource_domains": [],
            },
            "type": "iframe",
            "src": widget_src,
        }
        
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=widget.response_text,
                    )
                ],
                structuredContent=result,
                _meta=meta,
            )
        )
    except ValidationError as e:
        error_msg = f"Input validation error for tool {name}: {e.errors()}"
        logger.error(error_msg)
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Input validation error: {e.errors()}",
                    )
                ],
                isError=True,
            )
        )
    except Exception as e:
        error_msg = f"Tool execution error for {name}: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Tool execution error: {str(e)}",
                    )
                ],
                isError=True,
            )
        )


async def list_prompts() -> list[types.Prompt]:
    """List available prompts."""
    return [
        types.Prompt(
            name="codebase-overview",
            description="Get an overview of the codebase structure",
            arguments=[]
        ),
        types.Prompt(
            name="code-analysis",
            description="Analyze specific code patterns",
            arguments=[
                types.PromptArgument(
                    name="pattern",
                    description="Code pattern to analyze",
                    required=True
                )
            ]
        ),
    ]


async def handle_get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
    """Handle prompt requests."""
    name = req.params.name
    arguments = req.params.arguments or {}
    
    if name == "codebase-overview":
        return types.ServerResult(
            types.GetPromptResult(
                description="Codebase Overview",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text="Analyze the codebase structure and provide an overview"
                        )
                    )
                ]
            )
        )
    elif name == "code-analysis":
        pattern = arguments.get("pattern", "")
        return types.ServerResult(
            types.GetPromptResult(
                description="Code Pattern Analysis",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Analyze the following code pattern in the codebase: {pattern}"
                        )
                    )
                ]
            )
        )
    else:
        return types.ServerResult(
            types.GetPromptResult(
                description="Unknown prompt",
                messages=[],
            ),
            isError=True
        )
