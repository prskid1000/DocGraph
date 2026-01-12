"""Entity extraction from parsed code."""
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from ..parsers.base import CodeEntity, Reference, BaseParser


class EntityExtractor:
    """Extracts and organizes code entities from parsed files."""
    
    def __init__(self):
        """Initialize entity extractor."""
        self.entities_by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.entities_by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.references: List[Reference] = []
    
    def extract_from_file(self, parser: BaseParser, file_path: Path) -> tuple[List[CodeEntity], List[Reference]]:
        """Extract entities and references from a file.
        
        Args:
            parser: Parser instance for the file's language.
            file_path: Path to the source file.
            
        Returns:
            Tuple of (entities, references).
        """
        entities, references = parser.parse_and_extract(file_path)
        
        file_path_str = str(file_path)
        self.entities_by_file[file_path_str].extend(entities)
        
        for entity in entities:
            self.entities_by_name[entity.name].append(entity)
        
        self.references.extend(references)
        
        return entities, references
    
    def get_entities_by_file(self, file_path: str) -> List[CodeEntity]:
        """Get all entities in a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of entities in the file.
        """
        return self.entities_by_file.get(file_path, [])
    
    def get_entities_by_name(self, name: str) -> List[CodeEntity]:
        """Get all entities with a given name.
        
        Args:
            name: Entity name.
            
        Returns:
            List of entities with the name.
        """
        return self.entities_by_name.get(name, [])
    
    def get_entities_by_type(self, entity_type: str) -> List[CodeEntity]:
        """Get all entities of a specific type.
        
        Args:
            entity_type: Entity type (e.g., 'class', 'function').
            
        Returns:
            List of entities of the type.
        """
        entities = []
        for file_entities in self.entities_by_file.values():
            entities.extend([e for e in file_entities if e.entity_type == entity_type])
        return entities
    
    def get_all_entities(self) -> List[CodeEntity]:
        """Get all extracted entities.
        
        Returns:
            List of all entities.
        """
        entities = []
        for file_entities in self.entities_by_file.values():
            entities.extend(file_entities)
        return entities
    
    def get_all_references(self) -> List[Reference]:
        """Get all extracted references.
        
        Returns:
            List of all references.
        """
        return self.references
    
    def clear(self):
        """Clear all extracted data."""
        self.entities_by_file.clear()
        self.entities_by_name.clear()
        self.references.clear()

