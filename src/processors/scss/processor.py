"""SCSS language processor."""
from pathlib import Path
from typing import Tuple, Any

from ..base import LanguageProcessor
from .extractor import SCSSEntityExtractor
from .reference_extractor import SCSSReferenceExtractor
from .reference_resolver import SCSSReferenceResolver
from .graph_builder import SCSSGraphBuilder
from .embedding_generator import SCSSEmbeddingGenerator


class SCSSProcessor(LanguageProcessor):
    """SCSS language processor."""
    
    def __init__(self):
        """Initialize SCSS processor."""
        super().__init__("scss")
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse SCSS file - return source code."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        return source_code, source_code
    
    def create_entity_extractor(self):
        """Create SCSS entity extractor."""
        return SCSSEntityExtractor()
    
    def create_reference_extractor(self):
        """Create SCSS reference extractor."""
        return SCSSReferenceExtractor()
    
    def create_reference_resolver(self, entity_container):
        """Create SCSS reference resolver."""
        return SCSSReferenceResolver(entity_container)
    
    def create_graph_builder(self):
        """Create SCSS graph builder."""
        return SCSSGraphBuilder()
    
    def create_embedding_generator(self):
        """Create SCSS embedding generator."""
        return SCSSEmbeddingGenerator()
