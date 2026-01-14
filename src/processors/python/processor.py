"""Python language processor."""
from pathlib import Path
from typing import Tuple, Any

from ..base import (
    LanguageProcessor,
    BaseEntityExtractor,
    BaseReferenceExtractor,
    BaseReferenceResolver,
    BaseGraphBuilder,
    BaseEmbeddingGenerator
)
from .parser import PythonParser
from .extractor import PythonEntityExtractor
from .reference_extractor import PythonReferenceExtractor
from .reference_resolver import PythonReferenceResolver
from .graph_builder import PythonGraphBuilder
from .embedding_generator import PythonEmbeddingGenerator


class PythonProcessor(LanguageProcessor):
    """Python language processor."""
    
    def __init__(self):
        """Initialize Python processor."""
        super().__init__("python")
        self.parser = PythonParser()
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            ast = self.parser.parse(source_code)
            return ast, source_code
        except (PermissionError, IOError, OSError) as e:
            # Handle permission errors and other file access issues gracefully
            raise IOError(f"Cannot read file {file_path}: {e}") from e
    
    def create_entity_extractor(self) -> BaseEntityExtractor:
        """Create Python entity extractor."""
        return PythonEntityExtractor()
    
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create Python reference extractor."""
        return PythonReferenceExtractor()
    
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create Python reference resolver."""
        return PythonReferenceResolver(entity_container)
    
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create Python graph builder."""
        return PythonGraphBuilder()
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create Python embedding generator."""
        return PythonEmbeddingGenerator()
