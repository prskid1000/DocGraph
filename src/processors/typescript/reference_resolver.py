"""TypeScript reference resolver."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)

# TypeScript keywords that should not be resolved
TS_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'super',
    'typeof', 'instanceof', 'in', 'of', 'var', 'let', 'const', 'function',
    'class', 'extends', 'import', 'export', 'default', 'async', 'await',
    'yield', 'static', 'public', 'private', 'protected', 'abstract', 'interface',
    'type', 'enum', 'namespace', 'module', 'declare', 'as', 'is', 'keyof'
}

# TypeScript/JavaScript built-ins that are expected to be unresolved
TS_BUILTINS = {
    'JSON', 'Math', 'Date', 'Array', 'Object', 'String', 'Number', 'Boolean',
    'Promise', 'Set', 'Map', 'WeakMap', 'WeakSet', 'Symbol', 'RegExp', 'Error',
    'console', 'window', 'document', 'global', 'process', 'Buffer', 'exports',
    'module', 'require', 'fs', 'path', 'os', 'http', 'https', 'url', 'util'
}


class TypeScriptReferenceResolver(BaseReferenceResolver):
    """TypeScript-specific reference resolver with type support."""
    
    def __init__(self, entity_container):
        """Initialize TypeScript reference resolver."""
        super().__init__(entity_container)
        self._build_lookup_structures()
    
    def _build_lookup_structures(self):
        """Build lookup structures for TypeScript."""
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
        """Resolve TypeScript references."""
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
        logger.info(f"TypeScript: Resolved {total_resolved}/{total_valid} references ({total_resolved/total_valid*100:.1f}%)" if total_valid else "TypeScript: No references")
        
        # Log unresolved references
        if self.unresolved_references:
            unresolved_details = []
            for ref in self.unresolved_references:
                detail = f"  - {ref.to_entity} (type: {ref.reference_type}, file: {ref.file_path}, line: {ref.line_number}"
                if ref.scope:
                    detail += f", scope: {ref.scope}"
                detail += ")"
                unresolved_details.append(detail)
            logger.info(f"TypeScript: Unresolved references ({len(self.unresolved_references)}):\n" + "\n".join(unresolved_details))
        
        self.resolved_references = resolved
        return resolved
    
    def _should_resolve_reference(self, ref: ScopedReference) -> bool:
        """Check if a reference should be resolved."""
        target_name = ref.to_entity.strip()
        
        # Skip empty or invalid names
        if not target_name:
            return False
        
        # Skip keywords
        if target_name.lower() in TS_KEYWORDS:
            return False
        
        # Skip built-in globals (unless it's an import)
        if target_name in TS_BUILTINS and ref.reference_type != 'imports':
            return False
        
        # Skip invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            return False
        
        # Skip external module imports (relative paths, node modules, etc.)
        if ref.reference_type == 'imports':
            # Skip relative imports (./base, ../module, etc.)
            if target_name.startswith('./') or target_name.startswith('../'):
                return False
            # Skip common node modules that aren't in the codebase
            common_node_modules = {'fs', 'path', 'os', 'util', 'http', 'https', 'crypto', 'stream', 'events', 'buffer'}
            if target_name in common_node_modules:
                return False
        
        # Skip common object property names that are not entities
        common_object_properties = {'length', 'toString', 'valueOf', 'hasOwnProperty', 'constructor', 'prototype'}
        if target_name in common_object_properties and ref.reference_type == 'references':
            return False
        
        # For local variable references, check if they're parameters or class properties
        if ref.reference_type == 'references' and ref.scope:
            # Check if the scope is a function and if target is a parameter
            same_file = self.by_file.get(ref.file_path, [])
            scope_function = None
            for entity in same_file:
                if entity.name == ref.scope and entity.entity_type == 'function':
                    scope_function = entity
                    break
            
            # If scope is a function, check if target is a parameter
            if scope_function and scope_function.metadata:
                params = scope_function.metadata.get('parameters', [])
                if any(p.get('name') == target_name for p in params):
                    # It's a parameter reference - parameters are not entities, so skip
                    return False
            
            # Check if it's a class property
            scope_entities = self.by_scope.get(ref.scope, [])
            is_class_property = False
            for entity in scope_entities:
                if entity.name == target_name and entity.entity_type == 'variable' and entity.parent:
                    is_class_property = True
                    break
            
            # If it's not a parameter or property, check if it exists as an entity
            if not is_class_property:
                if not any(e.name == target_name for e in same_file):
                    # It's likely a local variable that doesn't need resolution
                    return False
        
        return True
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a TypeScript reference with module-aware strategies."""
        target_name = ref.to_entity.strip()
        
        # Clean up invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            target_name = target_name.replace('extends ', '').strip()
        
        # Handle property references - try to find the property in the class/scope
        if ref.context and ref.context.get('is_property') and ref.scope:
            scope_entities = self.by_scope.get(ref.scope, [])
            candidates = [e for e in scope_entities if e.name == target_name and e.entity_type in ['variable', 'parameter']]
            if candidates:
                return candidates[0]
            # Also check if the scope is a class and look for properties in that class
            class_entities = [e for e in self.by_name.get(ref.scope, []) if e.entity_type == 'class']
            if class_entities:
                class_properties = [e for e in scope_entities if e.parent == ref.scope and e.name == target_name]
                if class_properties:
                    return class_properties[0]
        
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
            # First, determine if scope is a class or method
            scope_entity = None
            for entity in self.entity_container.get_all_entities():
                if entity.name == ref.scope and entity.file_path == ref.file_path:
                    scope_entity = entity
                    break
            
            # If scope is a method, also look in its parent class and check for parameters
            if scope_entity and scope_entity.entity_type == 'function':
                # Check if target is a parameter of this function
                if scope_entity.metadata and scope_entity.metadata.get('parameters'):
                    params = scope_entity.metadata.get('parameters', [])
                    for param in params:
                        if param.get('name') == target_name:
                            # It's a parameter reference, try to resolve it as a variable in the scope
                            pass
                
                # Look in parent class scope if method has a parent
                if scope_entity.parent:
                    parent_scope_entities = self.by_scope.get(scope_entity.parent, [])
                    candidates = [e for e in parent_scope_entities if e.name == target_name]
                    if candidates:
                        if ref.reference_type == 'calls':
                            func_candidates = [e for e in candidates if e.entity_type == 'function']
                            if func_candidates:
                                return func_candidates[0]
                        return candidates[0]
            
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
