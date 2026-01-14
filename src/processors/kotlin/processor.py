"""Kotlin language processor - similar to Java."""
import importlib
from tree_sitter import Language, Parser

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
        """Load Java parser for Kotlin (Kotlin syntax is similar enough to Java for tree-sitter)."""
        # Use Java parser for Kotlin since tree-sitter-kotlin may not be available
        # Kotlin syntax is similar enough to Java that the Java parser can handle basic structures
        try:
            module = importlib.import_module("tree_sitter_java")
            lang_func = getattr(module, "language", None) or getattr(module, "language_java", None)
            if lang_func:
                lang_obj = lang_func()
                from tree_sitter import Parser, Language
                return Parser(Language(lang_obj))
        except ImportError:
            pass
        return None
    
    def create_reference_resolver(self, entity_container):
        """Create Kotlin reference resolver."""
        return KotlinReferenceResolver(entity_container)
