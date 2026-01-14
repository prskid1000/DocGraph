"""Python-specific reference resolver."""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)


class PythonReferenceResolver(BaseReferenceResolver):
    """Python-specific reference resolver with module path resolution."""
    
    def __init__(self, entity_container):
        """Initialize Python reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build comprehensive lookup structures for Python."""
        entities = self.entity_container.get_all_entities()
        
        self.by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_qualified_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_scope: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_module_path: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        for entity in entities:
            # Index by simple name
            self.by_name[entity.name].append(entity)
            
            # Index by qualified name
            if entity.parent:
                qualified = f"{entity.parent}.{entity.name}"
                self.by_qualified_name[qualified].append(entity)
                self.by_scope[entity.parent].append(entity)
            
            # Index by file
            self.by_file[entity.file_path].append(entity)
            
            # Index by module path
            module_paths = self._get_module_paths(entity.file_path)
            for path in module_paths:
                self.by_module_path[path].append(entity)
                # Also try with entity name
                self.by_module_path[f"{path}.{entity.name}"].append(entity)
    
    def _get_module_paths(self, file_path: str) -> List[str]:
        """Get possible Python module paths for a file."""
        paths = []
        path_obj = Path(file_path)
        
        stem = path_obj.stem
        if stem == '__init__':
            stem = path_obj.parent.name
        
        # Build module path from directory structure
        parts = list(path_obj.parts)
        if 'src' in parts:
            idx = parts.index('src')
            parts = parts[idx+1:]
        elif 'lib' in parts:
            idx = parts.index('lib')
            parts = parts[idx+1:]
        
        if parts:
            parts[-1] = stem
            module_path = '.'.join(parts)
            paths.append(module_path)
            paths.append(stem)
        
        return paths
    
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve Python references."""
        resolved = defaultdict(list)
        
        for ref in references:
            target = self._resolve_reference(ref)
            if target:
                resolved[ref.reference_type].append((ref, target))
            else:
                self.unresolved_references.append(ref)
        
        total_resolved = sum(len(v) for v in resolved.values())
        logger.info(f"Python: Resolved {total_resolved}/{len(references)} references ({total_resolved/len(references)*100:.1f}%)")
        
        self.resolved_references = resolved
        return resolved
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a Python reference."""
        target_name = ref.to_entity
        
        # Strategy 1: Qualified name from import
        if ref.qualified_name:
            candidates = self.by_qualified_name.get(ref.qualified_name, [])
            if not candidates:
                # Try module path resolution
                candidates = self.by_module_path.get(ref.qualified_name, [])
            if candidates:
                return self._disambiguate(candidates, ref)
        
        # Strategy 2: Scoped lookup
        if ref.scope:
            scoped_name = f"{ref.scope}.{target_name}"
            candidates = self.by_qualified_name.get(scoped_name, [])
            if candidates:
                return candidates[0]
            
            # Try entities within scope
            scope_entities = self.by_scope.get(ref.scope, [])
            candidates = [e for e in scope_entities if e.name == target_name]
            if candidates:
                return candidates[0]
        
        # Strategy 3: Same file
        same_file = self.by_file.get(ref.file_path, [])
        candidates = [e for e in same_file if e.name == target_name]
        if candidates:
            return self._disambiguate(candidates, ref)
        
        # Strategy 4: Global name
        candidates = self.by_name.get(target_name, [])
        if candidates:
            return self._disambiguate(candidates, ref)
        
        return None
    
    def _disambiguate(self, candidates: List[CodeEntity], ref: ScopedReference) -> Optional[CodeEntity]:
        """Disambiguate between candidates."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        # Prefer same file
        same_file = [e for e in candidates if e.file_path == ref.file_path]
        if same_file:
            if ref.scope:
                same_scope = [e for e in same_file if e.parent == ref.scope]
                if same_scope:
                    return same_scope[0]
            return same_file[0]
        
        return candidates[0]
