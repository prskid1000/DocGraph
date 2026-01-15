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
        
        # Filter out invalid references before resolving
        valid_references = []
        for ref in references:
            if self._should_resolve_reference(ref):
                valid_references.append(ref)
        
        for ref in valid_references:
            # For IMPORTS, allow them through even if they can't be resolved to entities
            # (they're file-to-file/module relationships, not entity relationships)
            if ref.reference_type == 'imports':
                resolved[ref.reference_type].append((ref, None))
            # For INHERITS, allow them through even if target can't be resolved
            # (the graph builder will try to find the base class by name)
            elif ref.reference_type == 'inherits':
                target = self._resolve_reference(ref)
                # Always add INHERITS, even if target is None (graph builder will search by name)
                resolved[ref.reference_type].append((ref, target))
            else:
                target = self._resolve_reference(ref)
                if target:
                    resolved[ref.reference_type].append((ref, target))
                else:
                    self.unresolved_references.append(ref)
        
        total_resolved = sum(len(v) for v in resolved.values())
        total_valid = len(valid_references)
        logger.info(f"Python: Resolved {total_resolved}/{total_valid} references ({total_resolved/total_valid*100:.1f}%)" if total_valid else "Python: No references")
        
        # Log unresolved references
        if self.unresolved_references:
            unresolved_details = []
            for ref in self.unresolved_references:
                detail = f"  - {ref.to_entity} (type: {ref.reference_type}, file: {ref.file_path}, line: {ref.line_number}"
                if ref.scope:
                    detail += f", scope: {ref.scope}"
                detail += ")"
                unresolved_details.append(detail)
            logger.info(f"Python: Unresolved references ({len(self.unresolved_references)}):\n" + "\n".join(unresolved_details))
        
        self.resolved_references = resolved
        return resolved
    
    def _should_resolve_reference(self, ref: ScopedReference) -> bool:
        """Check if a reference should be resolved."""
        target_name = ref.to_entity.strip()
        
        # Skip empty or invalid names
        if not target_name:
            return False
        
        # Skip Python keywords
        python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
            'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with',
            'yield', 'True', 'False', 'None', 'self', 'super'
        }
        if target_name in python_keywords:
            return False
        
        # Skip standard library modules and built-ins
        standard_library_modules = {
            'os', 'sys', 'json', 'math', 'random', 'datetime', 'time', 'collections',
            'itertools', 'functools', 'operator', 're', 'string', 'io', 'pathlib',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing', 'asyncio',
            'typing', 'dataclasses', 'enum', 'abc', 'collections.abc'
        }
        if ref.reference_type == 'imports':
            if target_name in standard_library_modules:
                return False
            # Also filter typing imports (List, Dict, Optional, etc.)
            typing_types = {'List', 'Dict', 'Optional', 'Union', 'Tuple', 'Set', 'FrozenSet',
                          'Callable', 'Iterable', 'Iterator', 'Generator', 'Any', 'TypeVar',
                          'Generic', 'Protocol', 'TypedDict', 'Literal', 'Final', 'ClassVar'}
            if target_name in typing_types:
                return False
        
        # Skip standard library methods that are expected to be unresolved
        standard_library_methods = {
            'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index', 'count',
            'sort', 'reverse', 'copy', 'get', 'set', 'keys', 'values', 'items',
            'update', 'popitem', 'setdefault', 'fromkeys', 'split', 'join', 'strip',
            'replace', 'find', 'index', 'count', 'startswith', 'endswith', 'lower',
            'upper', 'format', 'read', 'write', 'close', 'flush', 'seek', 'tell',
            'readline', 'readlines', 'writelines', 'open', 'print', 'input', 'len',
            'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'frozenset',
            'dumps', 'loads', 'dump', 'load', 'encode', 'decode'  # JSON methods
        }
        if ref.reference_type == 'calls' and target_name in standard_library_methods:
            return False
        
        return True
    
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
