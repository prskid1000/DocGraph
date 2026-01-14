"""Java reference resolver with package resolution."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)

# Java keywords that should not be resolved
JAVA_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'super',
    'package', 'import', 'class', 'interface', 'extends', 'implements',
    'public', 'private', 'protected', 'static', 'final', 'abstract',
    'synchronized', 'volatile', 'transient', 'native', 'strictfp',
    'enum', 'assert', 'goto', 'instanceof', 'null', 'true', 'false'
}

# Java standard library classes that are expected to be unresolved
JAVA_BUILTINS = {
    'System', 'String', 'Integer', 'Double', 'Float', 'Boolean', 'Character',
    'Object', 'Class', 'Exception', 'RuntimeException', 'Error',
    'List', 'ArrayList', 'LinkedList', 'Map', 'HashMap', 'Set', 'HashSet',
    'Collections', 'Arrays', 'Math', 'Random', 'Date', 'Calendar'
}


class JavaReferenceResolver(BaseReferenceResolver):
    """Java-specific reference resolver with package support."""
    
    def __init__(self, entity_container):
        """Initialize Java reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build lookup structures including package paths."""
        entities = self.entity_container.get_all_entities()
        
        self.by_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_qualified_name: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_file: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_scope: Dict[str, List[CodeEntity]] = defaultdict(list)
        self.by_package: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        for entity in entities:
            # Index by simple name
            self.by_name[entity.name].append(entity)
            
            # Index by qualified name (package.class or class.method)
            if entity.parent:
                qualified = f"{entity.parent}.{entity.name}"
                self.by_qualified_name[qualified].append(entity)
                self.by_scope[entity.parent].append(entity)
            
            # Index by file
            self.by_file[entity.file_path].append(entity)
            
            # Index by package
            package = self._extract_package_from_path(entity.file_path)
            if package:
                self.by_package[package].append(entity)
                # Also index by package.class
                if entity.entity_type == 'class':
                    self.by_qualified_name[f"{package}.{entity.name}"].append(entity)
    
    def _extract_package_from_path(self, file_path: str) -> Optional[str]:
        """Extract Java package from file path."""
        # Simple heuristic - look for java/ or src/ directory
        parts = file_path.replace('\\', '/').split('/')
        if 'java' in parts:
            idx = parts.index('java')
            package_parts = parts[idx+1:-1]  # Exclude filename
            return '.'.join(package_parts) if package_parts else None
        elif 'src' in parts:
            idx = parts.index('src')
            package_parts = parts[idx+1:-1]
            return '.'.join(package_parts) if package_parts else None
        return None
    
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve Java references."""
        resolved = defaultdict(list)
        
        # Filter out invalid references before resolving
        valid_references = []
        for ref in references:
            if self._should_resolve_reference(ref):
                valid_references.append(ref)
        
        for ref in valid_references:
            target = self._resolve_reference(ref)
            if target:
                resolved[ref.reference_type].append((ref, target))
            else:
                self.unresolved_references.append(ref)
        
        total_resolved = sum(len(v) for v in resolved.values())
        total_valid = len(valid_references)
        logger.info(f"Java: Resolved {total_resolved}/{total_valid} references ({total_resolved/total_valid*100:.1f}%)" if total_valid else "Java: No references")
        
        # Log unresolved references
        if self.unresolved_references:
            unresolved_details = []
            for ref in self.unresolved_references:
                detail = f"  - {ref.to_entity} (type: {ref.reference_type}, file: {ref.file_path}, line: {ref.line_number}"
                if ref.scope:
                    detail += f", scope: {ref.scope}"
                detail += ")"
                unresolved_details.append(detail)
            logger.info(f"Java: Unresolved references ({len(self.unresolved_references)}):\n" + "\n".join(unresolved_details))
        
        self.resolved_references = resolved
        return resolved
    
    def _should_resolve_reference(self, ref: ScopedReference) -> bool:
        """Check if a reference should be resolved."""
        target_name = ref.to_entity.strip()
        
        # Skip empty or invalid names
        if not target_name:
            return False
        
        # Skip keywords
        if target_name.lower() in JAVA_KEYWORDS:
            return False
        
        # Skip built-in standard library classes (unless it's an import)
        if target_name in JAVA_BUILTINS and ref.reference_type != 'imports':
            return False
        
        # Skip invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            return False
        
        return True
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a Java reference with package-aware strategies."""
        target_name = ref.to_entity.strip()
        
        # Clean up invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            target_name = target_name.replace('extends ', '').strip()
        
        # Strategy 1: Package-qualified name lookup (com.example.Class)
        if '.' in target_name:
            # Try as fully qualified name
            candidates = self.by_qualified_name.get(target_name, [])
            if candidates:
                return candidates[0]
            
            # Try package lookup
            if ref.reference_type == 'imports':
                candidates = self.by_package.get(target_name, [])
                if candidates:
                    return candidates[0]
        
        # Strategy 2: Scoped lookup (within class scope)
        if ref.scope:
            # Look for entities within the scope (methods, fields)
            scope_entities = self.by_scope.get(ref.scope, [])
            candidates = [e for e in scope_entities if e.name == target_name]
            if candidates:
                # Prefer methods for 'calls' type, prefer fields for 'references'
                if ref.reference_type == 'calls':
                    method_candidates = [e for e in candidates if e.entity_type == 'function']
                    if method_candidates:
                        return method_candidates[0]
                elif ref.reference_type == 'references':
                    field_candidates = [e for e in candidates if e.entity_type == 'variable']
                    if field_candidates:
                        return field_candidates[0]
                return candidates[0]
            
            # Try qualified name with scope (Class.method)
            scoped_name = f"{ref.scope}.{target_name}"
            candidates = self.by_qualified_name.get(scoped_name, [])
            if candidates:
                return candidates[0]
        
        # Strategy 3: Same file lookup (most common for local references)
        same_file = self.by_file.get(ref.file_path, [])
        candidates = [e for e in same_file if e.name == target_name]
        if candidates:
            # Prefer matching entity type
            if ref.reference_type == 'calls':
                method_candidates = [e for e in candidates if e.entity_type == 'function']
                if method_candidates:
                    return method_candidates[0]
            elif ref.reference_type == 'references':
                field_candidates = [e for e in candidates if e.entity_type == 'variable']
                if field_candidates:
                    return field_candidates[0]
            return candidates[0]
        
        # Strategy 4: Global lookup across all files (same package preferred)
        candidates = self.by_name.get(target_name, [])
        if candidates:
            # Prefer same file if available
            same_file_candidates = [e for e in candidates if e.file_path == ref.file_path]
            if same_file_candidates:
                return same_file_candidates[0]
            
            # Prefer same package
            ref_package = self._extract_package_from_path(ref.file_path)
            if ref_package:
                same_package_candidates = [
                    e for e in candidates 
                    if self._extract_package_from_path(e.file_path) == ref_package
                ]
                if same_package_candidates:
                    return same_package_candidates[0]
            
            return candidates[0]
        
        return None
