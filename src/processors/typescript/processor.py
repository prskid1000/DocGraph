"""TypeScript language processor."""
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
from .extractor import TypeScriptEntityExtractor
from .reference_extractor import TypeScriptReferenceExtractor
from .reference_resolver import TypeScriptReferenceResolver
from .graph_builder import TypeScriptGraphBuilder
from .embedding_generator import TypeScriptEmbeddingGenerator


class TypeScriptProcessor(LanguageProcessor):
    """TypeScript language processor."""
    
    def __init__(self):
        """Initialize TypeScript processor."""
        super().__init__("typescript")
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
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse a TypeScript file."""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            if self.parser:
                ast = self.parser.parse(source_code)
            else:
                ast = None
            return ast, source_code.decode('utf-8')
        except (PermissionError, IOError, OSError) as e:
            raise IOError(f"Cannot read file {file_path}: {e}") from e
    
    def create_entity_extractor(self) -> BaseEntityExtractor:
        """Create TypeScript entity extractor."""
        return TypeScriptEntityExtractor()
    
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create TypeScript reference extractor."""
        return TypeScriptReferenceExtractor()
    
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create TypeScript reference resolver."""
        return TypeScriptReferenceResolver(entity_container)
    
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create TypeScript graph builder."""
        return TypeScriptGraphBuilder()
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create TypeScript embedding generator."""
        return TypeScriptEmbeddingGenerator()
