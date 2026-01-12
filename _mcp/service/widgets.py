"""Widget definitions and HTML templates for the DocGraph MCP server."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from _mcp.logger import app_logger as logger
from _mcp.config import Config
from _mcp.service.utils.widget_loader import load_all_widgets_html
from _mcp.service.utils.widget_builder import ensure_widgets_built


@dataclass(frozen=True)
class Widget:
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str


# Widget base definitions
_widget_base_definitions: List[Dict] = [
    {
        "identifier": "search-code-entities",
        "title": "DocGraph: Search Code Entities",
        "template_uri": "ui://widget/SearchEntities.html",
        "invoking": "Searching code entities...",
        "invoked": "Search completed successfully",
        "response_text": "Code entity search completed."
    },
    {
        "identifier": "get-definition",
        "title": "DocGraph: Get Entity Definition",
        "template_uri": "ui://widget/GetDefinition.html",
        "invoking": "Fetching entity definition...",
        "invoked": "Definition retrieved successfully",
        "response_text": "Entity definition retrieved."
    },
    {
        "identifier": "find-references",
        "title": "DocGraph: Find References",
        "template_uri": "ui://widget/FindReferences.html",
        "invoking": "Finding references...",
        "invoked": "References found successfully",
        "response_text": "References retrieved."
    },
    {
        "identifier": "get-call-graph",
        "title": "DocGraph: Call Graph",
        "template_uri": "ui://widget/CallGraph.html",
        "invoking": "Building call graph...",
        "invoked": "Call graph generated successfully",
        "response_text": "Call graph generated."
    },
    {
        "identifier": "get-dependencies",
        "title": "DocGraph: Dependencies",
        "template_uri": "ui://widget/Dependencies.html",
        "invoking": "Analyzing dependencies...",
        "invoked": "Dependencies retrieved successfully",
        "response_text": "Dependencies analyzed."
    },
    {
        "identifier": "get-context",
        "title": "DocGraph: Code Context",
        "template_uri": "ui://widget/CodeContext.html",
        "invoking": "Loading code context...",
        "invoked": "Context loaded successfully",
        "response_text": "Code context retrieved."
    },
    {
        "identifier": "task-result",
        "title": "DocGraph: Task Results",
        "template_uri": "ui://widget/TaskResult.html",
        "invoking": "Loading task results...",
        "invoked": "Task results loaded successfully",
        "response_text": "Background task results retrieved."
    },
]


def _load_widgets() -> List[Widget]:
    """Load widgets with compiled HTML from build assets."""
    logger.info("🔧 Loading widgets with compiled HTML...")
    
    if not ensure_widgets_built():
        logger.warning("⚠️  Widgets build failed, using empty HTML")
    
    base_url = f"http://{Config.HOST}:{Config.PORT}" if Config.HOST else "http://localhost:5500"
    widget_html_map = load_all_widgets_html(_widget_base_definitions, base_url=base_url)
    
    widgets: List[Widget] = []
    
    for widget_def in _widget_base_definitions:
        identifier = widget_def["identifier"]
        html = widget_html_map.get(identifier, "")
        
        if not html:
            logger.warning(f"⚠️  No compiled HTML found for widget: {identifier}")
        else:
            logger.debug(f"✅ Widget {identifier} loaded with {len(html)} chars of HTML")
        
        widget = Widget(
            identifier=identifier,
            title=widget_def["title"],
            template_uri=widget_def["template_uri"],
            invoking=widget_def["invoking"],
            invoked=widget_def["invoked"],
            html=html,
            response_text=widget_def["response_text"],
        )
        widgets.append(widget)
    
    logger.info(f"✅ Loaded {len(widgets)} widgets")
    return widgets


# Initialize widgets on module import
widgets: List[Widget] = _load_widgets()
