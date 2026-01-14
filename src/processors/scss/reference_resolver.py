"""SCSS reference resolver."""
from ..base import BaseReferenceResolver, ScopedReference
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)


class SCSSReferenceResolver(BaseReferenceResolver):
    """SCSS reference resolver - resolves variables, mixins, and classes."""
    
    def __init__(self, entity_container):
        """Initialize SCSS reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build lookup structures for SCSS."""
        entities = self.entity_container.get_all_entities()
        
        self.by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        for entity in entities:
            self.by_name[entity.name].append(entity)
            self.by_file[entity.file_path].append(entity)
    
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve SCSS references to their target entities."""
        resolved = defaultdict(list)
        
        for ref in references:
            target_entity = self._resolve_reference(ref)
            resolved[ref.reference_type].append((ref, target_entity))
        
        self.resolved_references = resolved
        return resolved
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a SCSS reference."""
        target_name = ref.to_entity
        
        # Strategy 1: Same file lookup (most common for SCSS)
        same_file = self.by_file.get(ref.file_path, [])
        
        # For class references from @extend, try both with and without dot prefix
        # Entities store ".button" but references extract "button"
        search_names = [target_name]
        if ref.reference_type == 'references':
            # Try with dot prefix (entities have it)
            if not target_name.startswith('.'):
                search_names.append(f".{target_name}")
            # Try without dot (in case entity doesn't have it)
            if target_name.startswith('.'):
                search_names.append(target_name[1:])
        
        candidates = []
        for search_name in search_names:
            candidates.extend([e for e in same_file if e.name == search_name])
        
        if candidates:
            # Prefer matching entity type if context provides it
            if ref.reference_type == 'calls':
                # For @include, look for mixins (functions)
                func_candidates = [e for e in candidates if e.entity_type == 'function']
                if func_candidates:
                    return func_candidates[0]
            elif ref.reference_type == 'references':
                # For variable references, prefer variables
                var_candidates = [e for e in candidates if e.entity_type == 'variable']
                if var_candidates:
                    return var_candidates[0]
                # For @extend, look for classes
                class_candidates = [e for e in candidates if e.entity_type == 'class']
                if class_candidates:
                    return class_candidates[0]
            return candidates[0]
        
        # Strategy 2: Global name lookup (try all name variations)
        for search_name in search_names:
            candidates = self.by_name.get(search_name, [])
            if candidates:
                return candidates[0]
        
        return None
