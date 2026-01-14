"""Simple entity container for processors."""
from typing import List, Dict
from collections import defaultdict

from ..parsers.base import CodeEntity


class EntityContainer:
    """Simple container for entities - replaces old EntityExtractor."""
    
    def __init__(self):
        """Initialize container."""
        self.entities_by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.entities_by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.entities: List[CodeEntity] = []
    
    def add_entity(self, entity: CodeEntity):
        """Add an entity to the container."""
        self.entities.append(entity)
        self.entities_by_file[entity.file_path].append(entity)
        self.entities_by_name[entity.name].append(entity)
    
    def add_entities(self, entities: List[CodeEntity]):
        """Add multiple entities."""
        for entity in entities:
            self.add_entity(entity)
    
    def get_all_entities(self) -> List[CodeEntity]:
        """Get all entities."""
        return self.entities
    
    def get_entities_by_file(self, file_path: str) -> List[CodeEntity]:
        """Get entities in a file."""
        return self.entities_by_file.get(file_path, [])
    
    def get_entities_by_name(self, name: str) -> List[CodeEntity]:
        """Get entities with a given name."""
        return self.entities_by_name.get(name, [])
