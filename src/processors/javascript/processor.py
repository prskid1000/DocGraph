"""JavaScript language processor."""
from pathlib import Path
from typing import Tuple, Any
import tree_sitter
from tree_sitter import Language, Parser
import importlib

from ..base import (
    LanguageProcessor,
    BaseEntityExtractor,
    BaseReferenceExtractor,
    BaseReferenceResolver,
    BaseGraphBuilder,
    BaseEmbeddingGenerator
)
from .extractor import JavaScriptEntityExtractor
from .reference_extractor import JavaScriptReferenceExtractor
from .reference_resolver import JavaScriptReferenceResolver
from .graph_builder import JavaScriptGraphBuilder
from .embedding_generator import JavaScriptEmbeddingGenerator


class JavaScriptProcessor(LanguageProcessor):
    """JavaScript language processor."""
    
    def __init__(self):
        """Initialize JavaScript processor."""
        super().__init__("javascript")
        self.parser = self._load_parser()
    
    def _load_parser(self):
        """Load tree-sitter JavaScript parser."""
        try:
            module = importlib.import_module("tree_sitter_javascript")
            lang_func = getattr(module, "language", None) or getattr(module, "language_javascript", None)
            if lang_func:
                lang_obj = lang_func()
                return Parser(Language(lang_obj))
        except ImportError:
            pass
        return None
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse a JavaScript file."""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            if self.parser:
                ast = self.parser.parse(source_code)
            else:
                ast = None
            return ast, source_code.decode('utf-8')
        except (PermissionError, IOError, OSError) as e:
            # Handle permission errors and other file access issues gracefully
            raise IOError(f"Cannot read file {file_path}: {e}") from e
    
    def create_entity_extractor(self) -> BaseEntityExtractor:
        """Create JavaScript entity extractor."""
        return JavaScriptEntityExtractor()
    
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create JavaScript reference extractor."""
        return JavaScriptReferenceExtractor()
    
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create JavaScript reference resolver."""
        return JavaScriptReferenceResolver(entity_container)
    
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create JavaScript graph builder."""
        return JavaScriptGraphBuilder()
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create JavaScript embedding generator."""
        return JavaScriptEmbeddingGenerator()
