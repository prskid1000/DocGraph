"""Java language processor."""
from pathlib import Path
from typing import Tuple, Any
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
from .extractor import JavaEntityExtractor
from .reference_extractor import JavaReferenceExtractor
from .reference_resolver import JavaReferenceResolver
from .graph_builder import JavaGraphBuilder
from .embedding_generator import JavaEmbeddingGenerator


class JavaProcessor(LanguageProcessor):
    """Java language processor."""
    
    def __init__(self):
        """Initialize Java processor."""
        super().__init__("java")
        self.parser = self._load_java_parser()
        self.entity_extractor = JavaEntityExtractor()
        self.reference_extractor = JavaReferenceExtractor()
        self.graph_builder = JavaGraphBuilder()
        self.embedding_generator = JavaEmbeddingGenerator()
    
    def _load_java_parser(self):
        """Load tree-sitter Java parser."""
        try:
            module = importlib.import_module("tree_sitter_java")
            lang_func = getattr(module, "language", None) or getattr(module, "language_java", None)
            if lang_func:
                lang_obj = lang_func()
                return Parser(Language(lang_obj))
        except ImportError:
            pass
        return None
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse a Java file."""
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
        """Create Java entity extractor."""
        return JavaEntityExtractor()
    
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create Java reference extractor."""
        return JavaReferenceExtractor()
    
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create Java reference resolver."""
        return JavaReferenceResolver(entity_container)
    
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create Java graph builder."""
        return JavaGraphBuilder()
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create Java embedding generator."""
        return JavaEmbeddingGenerator()
