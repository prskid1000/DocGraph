"""Parser modules for multi-language code analysis."""
from .base import BaseParser, ParserFactory, CodeEntity, Reference
from .tree_sitter_parser import TreeSitterParser
from .python_parser import PythonParser
from .javascript_parser import JavaScriptParser, TypeScriptParser

# Register parsers
ParserFactory.register_parser("python", PythonParser)
ParserFactory.register_parser("javascript", JavaScriptParser)
ParserFactory.register_parser("typescript", TypeScriptParser)
ParserFactory.register_parser("java", lambda: TreeSitterParser("java", "tree_sitter_java"))
ParserFactory.register_parser("go", lambda: TreeSitterParser("go", "tree_sitter_go"))
ParserFactory.register_parser("rust", lambda: TreeSitterParser("rust", "tree_sitter_rust"))

__all__ = [
    "BaseParser",
    "ParserFactory",
    "CodeEntity",
    "Reference",
    "TreeSitterParser",
    "PythonParser",
    "JavaScriptParser",
    "TypeScriptParser",
]

