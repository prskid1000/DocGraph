"""Reference resolution for cross-file references and symbol resolution."""
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from ..parsers.base import CodeEntity, Reference
from .entity_extractor import EntityExtractor


class ReferenceResolver:
    """Resolves references to code entities across files."""
    
    def __init__(self, entity_extractor: EntityExtractor):
        """Initialize reference resolver.
        
        Args:
            entity_extractor: Entity extractor with extracted entities.
        """
        self.entity_extractor = entity_extractor
        self.resolved_references: Dict[str, List[Tuple[Reference, Optional[CodeEntity]]]] = defaultdict(list)
        self.unresolved_references: List[Reference] = []
    
    def resolve_references(self) -> Dict[str, List[Tuple[Reference, Optional[CodeEntity]]]]:
        """Resolve all references to their target entities.
        
        Returns:
            Dictionary mapping reference types to lists of (reference, target_entity) tuples.
        """
        references = self.entity_extractor.get_all_references()
        entities = self.entity_extractor.get_all_entities()
        
        # Build lookup maps
        entity_map = self._build_entity_map(entities)
        
        for ref in references:
            target_entity = self._resolve_reference(ref, entity_map)
            
            if target_entity:
                self.resolved_references[ref.reference_type].append((ref, target_entity))
            else:
                self.unresolved_references.append(ref)
        
        return self.resolved_references
    
    def _build_entity_map(self, entities: List[CodeEntity]) -> Dict[str, List[CodeEntity]]:
        """Build a map for quick entity lookup.
        
        Args:
            entities: List of all entities.
            
        Returns:
            Dictionary mapping entity names to lists of entities.
        """
        entity_map = defaultdict(list)
        
        for entity in entities:
            # Index by full name
            entity_map[entity.name].append(entity)
            
            # Index by qualified name (if has parent)
            if entity.parent:
                qualified_name = f"{entity.parent}.{entity.name}"
                entity_map[qualified_name].append(entity)
        
        return entity_map
    
    def _resolve_reference(self, ref: Reference, entity_map: Dict[str, List[CodeEntity]]) -> Optional[CodeEntity]:
        """Resolve a single reference to its target entity.
        
        Args:
            ref: Reference to resolve.
            entity_map: Map of entity names to entities.
            
        Returns:
            Target entity if found, None otherwise.
        """
        target_name = ref.to_entity
        
        # Direct name match
        if target_name in entity_map:
            candidates = entity_map[target_name]
            
            # If multiple candidates, prefer same file
            if len(candidates) == 1:
                return candidates[0]
            else:
                # Try to find in same file first
                same_file = [e for e in candidates if e.file_path == ref.file_path]
                if same_file:
                    return same_file[0]
                return candidates[0] if candidates else None
        
        # Try qualified name resolution for imports
        if ref.reference_type == 'imports':
            # Handle module imports - try to find module entity
            module_parts = target_name.split('.')
            if module_parts:
                # Try to find the module or package
                module_name = module_parts[0]
                if module_name in entity_map:
                    return entity_map[module_name][0]
        
        return None
    
    def get_call_graph(self, function_name: str) -> Dict[str, List[str]]:
        """Build call graph for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Dictionary with 'calls' (functions this calls) and 'called_by' (functions that call this).
        """
        calls = []
        called_by = []
        
        # Find the function entity
        function_entities = self.entity_extractor.get_entities_by_name(function_name)
        if not function_entities:
            return {'calls': [], 'called_by': []}
        
        function_entity = function_entities[0]
        
        # Find what this function calls
        for ref, target in self.resolved_references.get('calls', []):
            if ref.from_entity == function_entity.name or ref.file_path == function_entity.file_path:
                if target and target.entity_type == 'function':
                    calls.append(target.name)
        
        # Find what calls this function
        for ref, target in self.resolved_references.get('calls', []):
            if target and target.name == function_name:
                # Find the calling entity
                calling_entities = self.entity_extractor.get_entities_by_name(ref.from_entity)
                if calling_entities:
                    called_by.append(calling_entities[0].name)
        
        return {
            'calls': list(set(calls)),
            'called_by': list(set(called_by))
        }
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get all dependencies (imports) for a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of imported module names.
        """
        dependencies = []
        
        for ref, target in self.resolved_references.get('imports', []):
            if ref.file_path == file_path:
                dependencies.append(ref.to_entity)
        
        return list(set(dependencies))
    
    def get_inheritance_hierarchy(self, class_name: str) -> Dict[str, List[str]]:
        """Get inheritance hierarchy for a class.
        
        Args:
            class_name: Name of the class.
            
        Returns:
            Dictionary with 'parents' (superclasses) and 'children' (subclasses).
        """
        parents = []
        children = []
        
        # Find the class entity
        class_entities = self.entity_extractor.get_entities_by_name(class_name)
        if not class_entities:
            return {'parents': [], 'children': []}
        
        class_entity = class_entities[0]
        
        # Check metadata for base classes (Python)
        if class_entity.metadata and 'bases' in class_entity.metadata:
            parents.extend(class_entity.metadata['bases'])
        
        # Find inheritance relationships
        for ref, target in self.resolved_references.get('inherits', []):
            if ref.from_entity == class_name:
                if target:
                    parents.append(target.name)
            elif target and target.name == class_name:
                # Find the child class
                child_entities = self.entity_extractor.get_entities_by_name(ref.from_entity)
                if child_entities:
                    children.append(child_entities[0].name)
        
        return {
            'parents': list(set(parents)),
            'children': list(set(children))
        }

