"""JavaScript/TypeScript parser using tree-sitter."""
from pathlib import Path
from typing import List

from .tree_sitter_parser import TreeSitterParser
from .base import CodeEntity, Reference


class JavaScriptParser(TreeSitterParser):
    """JavaScript parser using tree-sitter."""
    
    def __init__(self):
        """Initialize JavaScript parser."""
        super().__init__("javascript", "tree_sitter_javascript")


class TypeScriptParser(TreeSitterParser):
    """TypeScript parser using tree-sitter."""
    
    def __init__(self):
        """Initialize TypeScript parser."""
        super().__init__("typescript", "tree_sitter_typescript")

