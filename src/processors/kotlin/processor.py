"""Kotlin language processor - similar to Java."""
from ..java.processor import JavaProcessor
from .extractor import KotlinEntityExtractor
from .reference_extractor import KotlinReferenceExtractor
from .reference_resolver import KotlinReferenceResolver
from .graph_builder import KotlinGraphBuilder
from .embedding_generator import KotlinEmbeddingGenerator


class KotlinProcessor(JavaProcessor):
    """Kotlin language processor."""
    
    def __init__(self):
        """Initialize Kotlin processor."""
        super().__init__()
        self.language = "kotlin"
        # Kotlin uses similar structure to Java but different syntax
        self.entity_extractor = KotlinEntityExtractor()
        self.reference_extractor = KotlinReferenceExtractor()
        self.graph_builder = KotlinGraphBuilder()
        self.embedding_generator = KotlinEmbeddingGenerator()
    
    def _load_java_parser(self):
        """Try to load Kotlin parser (may need tree-sitter-kotlin if available)."""
        # For now, we'll need to implement Kotlin parsing
        # This is a placeholder - Kotlin tree-sitter may not be available
        return None
    
    def create_reference_resolver(self, entity_container):
        """Create Kotlin reference resolver."""
        return KotlinReferenceResolver(entity_container)
