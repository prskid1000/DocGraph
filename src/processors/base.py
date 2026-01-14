"""Base interfaces for language processors."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..parsers.base import CodeEntity, Reference


@dataclass
class ScopedReference(Reference):
    """Reference with scope information for better resolution."""
    scope: Optional[str] = None  # Parent class/function/module
    qualified_name: Optional[str] = None  # Fully qualified name
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


class BaseEntityExtractor(ABC):
    """Base class for extracting entities from parsed code."""
    
    @abstractmethod
    def extract_entities(self, ast: Any, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract all entities from AST.
        
        Args:
            ast: Parsed AST (language-specific).
            file_path: Path to source file.
            source_code: Original source code.
            
        Returns:
            List of extracted entities.
        """
        pass


class BaseReferenceExtractor(ABC):
    """Base class for extracting references with proper scoping."""
    
    @abstractmethod
    def extract_references(
        self, 
        ast: Any, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract all references from AST with scope information.
        
        Args:
            ast: Parsed AST (language-specific).
            file_path: Path to source file.
            source_code: Original source code.
            entities: List of entities in this file (for scoping).
            
        Returns:
            List of scoped references.
        """
        pass
    
    def _find_enclosing_entity(
        self, 
        line: int, 
        entities: List[CodeEntity]
    ) -> Optional[CodeEntity]:
        """Find the entity that encloses a given line.
        
        Args:
            line: Line number.
            entities: List of entities.
            
        Returns:
            Enclosing entity or None.
        """
        candidates = [
            e for e in entities 
            if e.start_line <= line <= e.end_line
        ]
        if not candidates:
            return None
        # Return the most nested (smallest range)
        return min(candidates, key=lambda e: (e.end_line - e.start_line, e.start_line))


class BaseReferenceResolver(ABC):
    """Base class for language-specific reference resolution."""
    
    def __init__(self, entity_container):
        """Initialize resolver.
        
        Args:
            entity_container: Entity container with all entities.
        """
        self.entity_container = entity_container
        self.resolved_references: Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]] = {}
        self.unresolved_references: List[ScopedReference] = []
    
    @abstractmethod
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve all references to their target entities.
        
        Args:
            references: List of scoped references to resolve.
            
        Returns:
            Dictionary mapping reference types to lists of (reference, target_entity) tuples.
        """
        pass


class BaseGraphBuilder(ABC):
    """Base class for building graph relationships."""
    
    @abstractmethod
    def build_relationships(
        self,
        entities: List[CodeEntity],
        references: List[ScopedReference],
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Build graph relationships from entities and references.
        
        Args:
            entities: List of entities in the file.
            references: List of scoped references.
            file_path: Path to the file.
            
        Returns:
            List of relationship dictionaries with keys:
            - from_id: Source entity/node ID
            - to_id: Target entity/node ID
            - type: Relationship type
            - properties: Additional properties
        """
        pass


class BaseEmbeddingGenerator(ABC):
    """Base class for generating embeddings (can be language-specific)."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for embedding.
        
        Args:
            entity: Code entity.
            
        Returns:
            Text representation.
        """
        parts = [f"{entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        if entity.parent:
            parts.append(f"Parent: {entity.parent}")
        
        return "\n".join(parts)


class LanguageProcessor(ABC):
    """Base class for language-specific processing pipeline."""
    
    def __init__(self, language: str):
        """Initialize language processor.
        
        Args:
            language: Language name (e.g., 'python', 'javascript').
        """
        self.language = language
        self.entity_extractor: BaseEntityExtractor = self.create_entity_extractor()
        self.reference_extractor: BaseReferenceExtractor = self.create_reference_extractor()
        self.reference_resolver: BaseReferenceResolver = None  # Set after all files processed
        self.graph_builder: BaseGraphBuilder = self.create_graph_builder()
        self.embedding_generator: BaseEmbeddingGenerator = self.create_embedding_generator()
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> Tuple[Any, str]:
        """Parse a file into AST and return source code.
        
        Args:
            file_path: Path to source file.
            
        Returns:
            Tuple of (AST, source_code).
        """
        pass
    
    @abstractmethod
    def create_entity_extractor(self) -> BaseEntityExtractor:
        """Create entity extractor for this language."""
        pass
    
    @abstractmethod
    def create_reference_extractor(self) -> BaseReferenceExtractor:
        """Create reference extractor for this language."""
        pass
    
    @abstractmethod
    def create_reference_resolver(self, entity_container) -> BaseReferenceResolver:
        """Create reference resolver for this language.
        
        Args:
            entity_container: Entity container with all entities.
        """
        pass
    
    @abstractmethod
    def create_graph_builder(self) -> BaseGraphBuilder:
        """Create graph builder for this language."""
        pass
    
    def create_embedding_generator(self) -> BaseEmbeddingGenerator:
        """Create embedding generator (default implementation)."""
        return BaseEmbeddingGenerator()
    
    def process_file(self, file_path: Path) -> Tuple[List[CodeEntity], List[ScopedReference]]:
        """Process a file and extract entities and references.
        
        Args:
            file_path: Path to source file.
            
        Returns:
            Tuple of (entities, references).
        """
        # Parse file
        ast, source_code = self.parse_file(file_path)
        
        # Share parser instance with extractors (for Python's metadata_wrapper)
        if hasattr(self, 'parser') and hasattr(self.entity_extractor, 'parser'):
            self.entity_extractor.parser = self.parser
        if hasattr(self, 'parser') and hasattr(self.reference_extractor, 'parser'):
            self.reference_extractor.parser = self.parser
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(ast, file_path, source_code)
        
        # Extract references (with scoping)
        references = self.reference_extractor.extract_references(
            ast, file_path, source_code, entities
        )
        
        return entities, references
