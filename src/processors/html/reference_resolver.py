"""HTML reference resolver."""
from ..base import BaseReferenceResolver, ScopedReference
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)


class HTMLReferenceResolver(BaseReferenceResolver):
    """HTML reference resolver - resolves file paths and JavaScript references."""
    
    def __init__(self, entity_container):
        """Initialize HTML reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build lookup structures for HTML."""
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
        """Resolve HTML references."""
        resolved = defaultdict(list)
        
        for ref in references:
            if ref.reference_type in ['imports', 'contains']:
                # File path references - no entity target needed
                resolved[ref.reference_type].append((ref, None))
            else:
                # CALLS and REFERENCES - try to resolve to entities
                target_entity = self._resolve_reference(ref)
                resolved[ref.reference_type].append((ref, target_entity))
        
        self.resolved_references = resolved
        return resolved
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a JavaScript reference in HTML."""
        target_name = ref.to_entity
        
        # Strategy 1: Same file lookup
        same_file = self.by_file.get(ref.file_path, [])
        candidates = [e for e in same_file if e.name == target_name]
        if candidates:
            # Prefer matching entity type
            if ref.reference_type == 'calls':
                func_candidates = [e for e in candidates if e.entity_type == 'function']
                if func_candidates:
                    return func_candidates[0]
            elif ref.reference_type == 'references':
                var_candidates = [e for e in candidates if e.entity_type == 'variable']
                if var_candidates:
                    return var_candidates[0]
            return candidates[0]
        
        # Strategy 2: Global name lookup
        candidates = self.by_name.get(target_name, [])
        if candidates:
            return candidates[0]
        
        return None
