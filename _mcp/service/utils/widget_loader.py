"""Widget loader that discovers and loads compiled widgets on server startup."""

from __future__ import annotations
import glob
import pathlib
from functools import lru_cache
from typing import Dict
from _mcp.logger import app_logger as logger


@lru_cache(maxsize=128)
def _load_widget_html(widget_name: str) -> str:
    """Load widget HTML using two-tier lookup strategy with LRU caching."""
    assets_dir = pathlib.Path(__file__).parent.parent.parent / "widgets-assets"
    
    if not assets_dir.exists():
        logger.warning(f"⚠️  Widget assets directory not found: {assets_dir}")
        return ""
    
    # Tier 1: Try exact match (e.g., 'search-entities.html')
    exact_match_path = assets_dir / f"{widget_name}.html"
    if exact_match_path.exists():
        try:
            with open(exact_match_path, "r", encoding="utf-8") as f:
                html = f.read()
            logger.debug(f"✅ Loaded {widget_name} HTML via exact match")
            return html
        except Exception as e:
            logger.warning(f"⚠️  Failed to read {exact_match_path}: {e}")
    
    # Tier 2: Fall back to glob pattern for versioned files (e.g., 'search-entities-*.html')
    glob_pattern = str(assets_dir / f"{widget_name}-*.html")
    matches = sorted(glob.glob(glob_pattern), reverse=True)
    
    if matches:
        versioned_path = pathlib.Path(matches[0])
        try:
            with open(versioned_path, "r", encoding="utf-8") as f:
                html = f.read()
            logger.debug(f"✅ Loaded {widget_name} HTML via glob pattern: {versioned_path.name}")
            return html
        except Exception as e:
            logger.warning(f"⚠️  Failed to read {versioned_path}: {e}")
    
    logger.warning(f"⚠️  Widget HTML not found for {widget_name}")
    return ""


def get_widget_html(widget_name: str, base_url: str = "") -> str:
    """Get HTML for a compiled widget."""
    return _load_widget_html(widget_name)


def _extract_widget_name_from_template_uri(template_uri: str) -> str:
    """Extract widget name from template URI."""
    if template_uri.startswith("ui://widget/"):
        name = template_uri[len("ui://widget/"):]
        if name.endswith(".html"):
            return name[:-5]
    return ""


def load_all_widgets_html(widget_definitions: list, base_url: str = "") -> Dict[str, str]:
    """Load HTML for all compiled widgets."""
    assets_dir = pathlib.Path(__file__).parent.parent.parent / "widgets-assets"
    
    if not assets_dir.exists():
        logger.warning(f"⚠️  Widget assets directory not found: {assets_dir}")
        return {}
    
    widget_html_map: Dict[str, str] = {}
    
    for widget_def in widget_definitions:
        template_uri = widget_def.get("template_uri", "")
        identifier = widget_def.get("identifier", "")
        
        if not template_uri or not identifier:
            continue
        
        widget_name = _extract_widget_name_from_template_uri(template_uri)
        
        if not widget_name:
            logger.warning(f"⚠️  Could not extract widget name from template_uri: {template_uri}")
            continue
        
        html = _load_widget_html(widget_name)
        
        if not html:
            logger.debug(f"⚠️  No HTML file found for widget: {widget_name} (identifier: {identifier})")
        else:
            widget_html_map[identifier] = html
    
    logger.info(f"✅ Loaded HTML for {len(widget_html_map)} widgets")
    
    for identifier, html in widget_html_map.items():
        html_len = len(html) if html else 0
        logger.debug(f"   Widget {identifier}: {html_len} chars")
    
    return widget_html_map
