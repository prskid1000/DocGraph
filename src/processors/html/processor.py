"""HTML language processor."""
from pathlib import Path
from typing import Tuple, Any

from ..base import LanguageProcessor
from .extractor import HTMLEntityExtractor
from .reference_extractor import HTMLReferenceExtractor
from .reference_resolver import HTMLReferenceResolver
from .graph_builder import HTMLGraphBuilder
from .embedding_generator import HTMLEmbeddingGenerator


class HTMLProcessor(LanguageProcessor):
    """HTML language processor."""
    
    def __init__(self):
        """Initialize HTML processor."""
        super().__init__("html")
    
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse HTML file - just return source code."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        return source_code, source_code  # AST is just the source for HTML
    
    def create_entity_extractor(self):
        """Create HTML entity extractor."""
        return HTMLEntityExtractor()
    
    def create_reference_extractor(self):
        """Create HTML reference extractor."""
        return HTMLReferenceExtractor()
    
    def create_reference_resolver(self, entity_container):
        """Create HTML reference resolver."""
        return HTMLReferenceResolver(entity_container)
    
    def create_graph_builder(self):
        """Create HTML graph builder."""
        return HTMLGraphBuilder()
    
    def create_embedding_generator(self):
        """Create HTML embedding generator."""
        return HTMLEmbeddingGenerator()
