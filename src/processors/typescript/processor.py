"""TypeScript language processor - similar to JavaScript but with type information."""
from pathlib import Path
from typing import Tuple, Any
import tree_sitter
from tree_sitter import Language, Parser
import importlib

from ..base import LanguageProcessor
from ..javascript.processor import JavaScriptProcessor
from .extractor import TypeScriptEntityExtractor
from .reference_extractor import TypeScriptReferenceExtractor
from .reference_resolver import TypeScriptReferenceResolver
from .graph_builder import TypeScriptGraphBuilder
from .embedding_generator import TypeScriptEmbeddingGenerator


class TypeScriptProcessor(JavaScriptProcessor):
    """TypeScript language processor - extends JavaScript with type support."""
    
    def __init__(self):
        """Initialize TypeScript processor."""
        super().__init__()
        self.language = "typescript"
        # Override parser for TypeScript
        self.parser = self._load_typescript_parser()
        self.entity_extractor = TypeScriptEntityExtractor()
        self.reference_extractor = TypeScriptReferenceExtractor()
        self.graph_builder = TypeScriptGraphBuilder()
        self.embedding_generator = TypeScriptEmbeddingGenerator()
    
    def _load_typescript_parser(self):
        """Load tree-sitter TypeScript parser."""
        try:
            module = importlib.import_module("tree_sitter_typescript")
            lang_func = getattr(module, "language_typescript", None) or getattr(module, "language", None)
            if lang_func:
                lang_obj = lang_func()
                return Parser(Language(lang_obj))
        except ImportError:
            pass
        return None
    
    def create_reference_resolver(self, entity_container):
        """Create TypeScript reference resolver."""
        return TypeScriptReferenceResolver(entity_container)
