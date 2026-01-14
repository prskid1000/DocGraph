"""Kotlin reference resolver."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)

# Kotlin keywords that should not be resolved
KOTLIN_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'when', 'break', 'continue',
    'return', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'super',
    'package', 'import', 'class', 'interface', 'object', 'enum', 'sealed',
    'data', 'inline', 'infix', 'operator', 'override', 'open', 'abstract',
    'private', 'protected', 'internal', 'public', 'val', 'var', 'fun',
    'null', 'true', 'false', 'is', 'as', 'in', 'out', 'reified', 'companion'
}

# Kotlin standard library classes that are expected to be unresolved
KOTLIN_BUILTINS = {
    'String', 'Int', 'Double', 'Float', 'Boolean', 'Char', 'Long', 'Short', 'Byte',
    'Any', 'Unit', 'Nothing', 'List', 'MutableList', 'ArrayList', 'LinkedList',
    'Map', 'MutableMap', 'HashMap', 'Set', 'MutableSet', 'HashSet',
    'Array', 'Pair', 'Triple', 'Result', 'Exception', 'Error', 'IllegalArgumentException'
}


class KotlinReferenceResolver(BaseReferenceResolver):
    """Kotlin-specific reference resolver with package support."""
    
    def __init__(self, entity_container):
        """Initialize Kotlin reference resolver."""
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
            
            # Index by qualified name
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
        """Extract Kotlin package from file path."""
        # Simple heuristic - look for kotlin/ or src/ directory
        parts = file_path.replace('\\', '/').split('/')
        if 'kotlin' in parts:
            idx = parts.index('kotlin')
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
        """Resolve Kotlin references."""
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
        logger.info(f"Kotlin: Resolved {total_resolved}/{total_valid} references ({total_resolved/total_valid*100:.1f}%)" if total_valid else "Kotlin: No references")
        
        # Log unresolved references
        if self.unresolved_references:
            unresolved_details = []
            for ref in self.unresolved_references:
                detail = f"  - {ref.to_entity} (type: {ref.reference_type}, file: {ref.file_path}, line: {ref.line_number}"
                if ref.scope:
                    detail += f", scope: {ref.scope}"
                detail += ")"
                unresolved_details.append(detail)
            logger.info(f"Kotlin: Unresolved references ({len(self.unresolved_references)}):\n" + "\n".join(unresolved_details))
        
        self.resolved_references = resolved
        return resolved
    
    def _should_resolve_reference(self, ref: ScopedReference) -> bool:
        """Check if a reference should be resolved."""
        target_name = ref.to_entity.strip()
        
        # Skip empty or invalid names
        if not target_name:
            return False
        
        # Skip keywords
        if target_name.lower() in KOTLIN_KEYWORDS:
            return False
        
        # Skip built-in standard library classes (unless it's an import)
        if target_name in KOTLIN_BUILTINS and ref.reference_type != 'imports':
            return False
        
        # Skip invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            return False
        
        # Skip standard library methods that are expected to be unresolved
        standard_library_methods = {'isEmpty', 'isNotEmpty', 'size', 'get', 'set', 'add', 'remove', 'contains', 
                                     'indexOf', 'lastIndexOf', 'subList', 'clear', 'toString', 'equals', 'hashCode'}
        if ref.reference_type == 'calls' and target_name in standard_library_methods:
            return False
        
        return True
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a Kotlin reference with package-aware strategies."""
        target_name = ref.to_entity.strip()
        
        # Clean up invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            target_name = target_name.replace('extends ', '').strip()
        
        # Strategy 1: Package-qualified name lookup
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
        
        # Strategy 2: Scoped lookup (within class/object scope)
        if ref.scope:
            # First, determine if scope is a class or method
            scope_entity = None
            for entity in self.entity_container.get_all_entities():
                if entity.name == ref.scope and entity.file_path == ref.file_path:
                    scope_entity = entity
                    break
            
            # If scope is a method, also look in its parent class
            if scope_entity and scope_entity.entity_type == 'function' and scope_entity.parent:
                # Look in parent class scope
                parent_scope_entities = self.by_scope.get(scope_entity.parent, [])
                candidates = [e for e in parent_scope_entities if e.name == target_name]
                if candidates:
                    if ref.reference_type == 'calls':
                        func_candidates = [e for e in candidates if e.entity_type == 'function']
                        if func_candidates:
                            return func_candidates[0]
                    return candidates[0]
            
            # If scope is a top-level function (no parent), check all classes for the method
            # This handles cases like getName() called on a parameter in a top-level function
            if scope_entity and scope_entity.entity_type == 'function' and not scope_entity.parent:
                # Check all classes in the file for this method
                same_file = self.by_file.get(ref.file_path, [])
                all_classes = [e for e in same_file if e.entity_type == 'class']
                for cls in all_classes:
                    class_methods = [e for e in same_file if e.parent == cls.name and e.name == target_name and e.entity_type == 'function']
                    if class_methods:
                        return class_methods[0]
            
            # Look for entities within the scope
            scope_entities = self.by_scope.get(ref.scope, [])
            candidates = [e for e in scope_entities if e.name == target_name]
            if candidates:
                # Prefer functions for 'calls' type, prefer properties for 'references'
                if ref.reference_type == 'calls':
                    func_candidates = [e for e in candidates if e.entity_type == 'function']
                    if func_candidates:
                        return func_candidates[0]
                elif ref.reference_type == 'references':
                    prop_candidates = [e for e in candidates if e.entity_type == 'variable']
                    if prop_candidates:
                        return prop_candidates[0]
                return candidates[0]
            
            # Try qualified name with scope
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
                func_candidates = [e for e in candidates if e.entity_type == 'function']
                if func_candidates:
                    # Try to find the enclosing function and its class
                    # This handles cases where scope wasn't set correctly
                    enclosing_function = None
                    for entity in same_file:
                        if entity.entity_type == 'function' and entity.start_line <= ref.line_number <= entity.end_line:
                            enclosing_function = entity
                            break
                    
                    # If we have an enclosing function with a parent class, look there first
                    if enclosing_function and enclosing_function.parent:
                        # Look for methods in the same class
                        class_methods = [e for e in same_file if e.parent == enclosing_function.parent and e.name == target_name and e.entity_type == 'function']
                        if class_methods:
                            return class_methods[0]
                    
                    # Also check if scope was set but is a method - look in its parent class
                    # This handles cases where scope is set but resolution didn't work in Strategy 2
                    if ref.scope:
                        scope_entity = None
                        for entity in same_file:
                            if entity.name == ref.scope and entity.entity_type == 'function':
                                scope_entity = entity
                                break
                        if scope_entity and scope_entity.parent:
                            parent_class_methods = [e for e in same_file if e.parent == scope_entity.parent and e.name == target_name and e.entity_type == 'function']
                            if parent_class_methods:
                                return parent_class_methods[0]
                    
                    # Check all classes in the file for this method
                    # This handles cases like:
                    # 1. Methods on parameters (getName on base: BaseClass) - when scope is a top-level function
                    # 2. Methods in the same class when scope wasn't set correctly
                    # 3. Methods in other classes that might be called
                    all_classes = [e for e in same_file if e.entity_type == 'class']
                    for cls in all_classes:
                        class_methods = [e for e in same_file if e.parent == cls.name and e.name == target_name and e.entity_type == 'function']
                        if class_methods:
                            # Prefer methods in the same class as the enclosing function
                            if enclosing_function and enclosing_function.parent == cls.name:
                                return class_methods[0]
                            # For methods on parameters or when scope wasn't set, return any matching method
                            # (e.g., getName called on base: BaseClass parameter, or helperFunction in same class)
                            return class_methods[0]
                    
                    # If no scope or couldn't find in class, return first function candidate
                    return func_candidates[0]
            elif ref.reference_type == 'references':
                prop_candidates = [e for e in candidates if e.entity_type == 'variable']
                if prop_candidates:
                    return prop_candidates[0]
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
