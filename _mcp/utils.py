"""Utility functions for DocGraph MCP server."""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as JSON string.
    
    Args:
        data: Data to format
        indent: JSON indentation level
        
    Returns:
        JSON formatted string
    """
    if isinstance(data, BaseModel):
        return json.dumps(data.model_dump(), indent=indent)
    return json.dumps(data, indent=indent, default=str)


def parse_json(data: str) -> Any:
    """
    Parse JSON string safely.
    
    Args:
        data: JSON string to parse
        
    Returns:
        Parsed data or None if invalid
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None


def filter_dict(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to only include specified keys.
    
    Args:
        data: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in data.items() if k in keys}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to max length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_code(code: str) -> str:
    """
    Clean code string by removing extra whitespace.
    
    Args:
        code: Code to clean
        
    Returns:
        Cleaned code
    """
    lines = code.split('\n')
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def highlight_line(code: str, line_number: int, context_lines: int = 2) -> str:
    """
    Get code with highlighted line and context.
    
    Args:
        code: Code string
        line_number: Line to highlight (1-indexed)
        context_lines: Lines of context before and after
        
    Returns:
        Code snippet with context
    """
    lines = code.split('\n')
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)
    
    result = []
    for i in range(start, end):
        prefix = ">>> " if i == line_number - 1 else "    "
        result.append(f"{prefix}{i+1:4d}: {lines[i]}")
    
    return '\n'.join(result)
