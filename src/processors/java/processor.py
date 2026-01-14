"""Java language processor."""
from pathlib import Path
from typing import Tuple, Any
from tree_sitter import Language, Parser
import importlib

from ..base import LanguageProcessor
from ..javascript.processor import JavaScriptProcessor
from .extractor import JavaEntityExtractor
from .reference_extractor import JavaReferenceExtractor
from .reference_resolver import JavaReferenceResolver
from .graph_builder import JavaGraphBuilder
from .embedding_generator import JavaEmbeddingGenerator


class JavaProcessor(JavaScriptProcessor):
    """Java language processor."""
    
    def __init__(self):
        """Initialize Java processor."""
        super().__init__()
        self.language = "java"
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
    
    def create_reference_resolver(self, entity_container):
        """Create Java reference resolver."""
        return JavaReferenceResolver(entity_container)
