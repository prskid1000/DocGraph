"""JavaScript-specific reference resolver."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)


class JavaScriptReferenceResolver(BaseReferenceResolver):
    """JavaScript-specific reference resolver."""
    
    def __init__(self, entity_container):
        """Initialize JavaScript reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build lookup structures for JavaScript."""
        entities = self.entity_container.get_all_entities()
        
        self.by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_qualified_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_scope: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        for entity in entities:
            self.by_name[entity.name].append(entity)
            if entity.parent:
                qualified = f"{entity.parent}.{entity.name}"
                self.by_qualified_name[qualified].append(entity)
                self.by_scope[entity.parent].append(entity)
            self.by_file[entity.file_path].append(entity)
    
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve JavaScript references."""
        resolved = defaultdict(list)
        
        for ref in references:
            target = self._resolve_reference(ref)
            if target:
                resolved[ref.reference_type].append((ref, target))
            else:
                self.unresolved_references.append(ref)
        
        total_resolved = sum(len(v) for v in resolved.values())
        logger.info(f"JavaScript: Resolved {total_resolved}/{len(references)} references ({total_resolved/len(references)*100:.1f}%)" if references else "JavaScript: No references")
        
        self.resolved_references = resolved
        return resolved
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a JavaScript reference with improved strategies."""
        target_name = ref.to_entity
        
        # Strategy 1: Qualified name lookup (Class.method, module.function)
        if '.' in target_name:
            parts = target_name.split('.')
            if len(parts) == 2:
                parent_name, child_name = parts
                qualified = f"{parent_name}.{child_name}"
                candidates = self.by_qualified_name.get(qualified, [])
                if candidates:
                    return candidates[0]
        
        # Strategy 2: Scoped lookup (within class/function scope)
        if ref.scope:
            scoped_name = f"{ref.scope}.{target_name}"
            candidates = self.by_qualified_name.get(scoped_name, [])
            if candidates:
                return candidates[0]
            
            # Look for entities within the scope
            scope_entities = self.by_scope.get(ref.scope, [])
            candidates = [e for e in scope_entities if e.name == target_name]
            if candidates:
                # Prefer functions for 'calls' type, prefer variables for 'references'
                if ref.reference_type == 'calls':
                    func_candidates = [e for e in candidates if e.entity_type == 'function']
                    if func_candidates:
                        return func_candidates[0]
                return candidates[0]
        
        # Strategy 3: Same file lookup (most common for local references)
        same_file = self.by_file.get(ref.file_path, [])
        candidates = [e for e in same_file if e.name == target_name]
        if candidates:
            # Prefer matching entity type if context provides it
            if ref.context and 'expected_type' in ref.context:
                expected = ref.context['expected_type']
                type_map = {'function': 'function', 'class': 'class', 'variable': 'variable'}
                if expected in type_map:
                    type_candidates = [e for e in candidates if e.entity_type == type_map[expected]]
                    if type_candidates:
                        return type_candidates[0]
            return candidates[0]
        
        # Strategy 4: Global lookup across all files
        candidates = self.by_name.get(target_name, [])
        if candidates:
            # Prefer matching entity type
            if ref.context and 'expected_type' in ref.context:
                expected = ref.context['expected_type']
                type_map = {'function': 'function', 'class': 'class', 'variable': 'variable'}
                if expected in type_map:
                    type_candidates = [e for e in candidates if e.entity_type == type_map[expected]]
                    if type_candidates:
                        # Prefer same file if available
                        same_file_candidates = [e for e in type_candidates if e.file_path == ref.file_path]
                        if same_file_candidates:
                            return same_file_candidates[0]
                        return type_candidates[0]
            
            # Prefer same file if available
            same_file_candidates = [e for e in candidates if e.file_path == ref.file_path]
            if same_file_candidates:
                return same_file_candidates[0]
            
            return candidates[0]
        
        return None
