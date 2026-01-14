"""Kotlin language processor."""
import importlib
from pathlib import Path
from typing import Tuple, Any
from tree_sitter import Language, Parser

from ..base import (
    LanguageProcessor,
    BaseEntityExtractor,
    BaseReferenceExtractor,
    BaseReferenceResolver,
    BaseGraphBuilder,
    BaseEmbeddingGenerator
)
from .extractor import KotlinEntityExtractor
from .reference_extractor import KotlinReferenceExtractor
from .reference_resolver import KotlinReferenceResolver
from .graph_builder import KotlinGraphBuilder
from .embedding_generator import KotlinEmbeddingGenerator


class KotlinProcessor(LanguageProcessor):
    """Kotlin language processor."""
    
    def __init__(self):
        """Initialize Kotlin processor."""
        super().__init__("kotlin")
        self.parser = self._load_kotlin_parser()
        self.entity_extractor = KotlinEntityExtractor()
        self.reference_extractor = KotlinReferenceExtractor()
        self.graph_builder = KotlinGraphBuilder()
        self.embedding_generator = KotlinEmbeddingGenerator()
    
    def _load_kotlin_parser(self):
        """Load Java parser for Kotlin (Kotlin syntax is similar enough to Java for tree-sitter)."""
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
        """Parse a Kotlin file."""
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
        """Create Kotlin entity extractor."""
        return KotlinEntityExtractor()
    
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create Kotlin reference extractor."""
        return KotlinReferenceExtractor()
    
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create Kotlin reference resolver."""
        return KotlinReferenceResolver(entity_container)
    
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create Kotlin graph builder."""
        return KotlinGraphBuilder()
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create Kotlin embedding generator."""
        return KotlinEmbeddingGenerator()
