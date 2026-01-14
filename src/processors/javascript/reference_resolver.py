"""JavaScript-specific reference resolver."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ..base import BaseReferenceResolver, ScopedReference
from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)

# JavaScript/Java/Kotlin keywords that should not be resolved
JS_KEYWORDS = {
    # JavaScript/TypeScript keywords
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'super',
    'typeof', 'instanceof', 'in', 'of', 'var', 'let', 'const', 'function',
    'class', 'extends', 'import', 'export', 'default', 'async', 'await',
    'yield', 'static', 'public', 'private', 'protected', 'abstract', 'interface',
    # Java/Kotlin keywords
    'package', 'synchronized', 'volatile', 'transient', 'native', 'strictfp',
    'enum', 'assert', 'goto', 'implements', 'instanceof', 'synchronized',
    'transient', 'volatile', 'null', 'true', 'false'
}

# Built-in globals that are expected to be unresolved
JS_BUILTINS = {
    'JSON', 'Math', 'Date', 'Array', 'Object', 'String', 'Number', 'Boolean',
    'Promise', 'Set', 'Map', 'WeakMap', 'WeakSet', 'Symbol', 'RegExp', 'Error',
    'console', 'window', 'document', 'global', 'process', 'Buffer', 'exports',
    'module', 'require', 'fs', 'path', 'os', 'http', 'https', 'url', 'util'
}


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
        language_name = self._get_language_name()
        
        # Use logger from the actual class's module, not the parent
        actual_logger = logging.getLogger(self.__class__.__module__)
        
        # Filter out invalid references before resolving
        valid_references = []
        for ref in references:
            if self._should_resolve_reference(ref):
                valid_references.append(ref)
            else:
                # Skip keywords, built-ins, and invalid references
                pass
        
        for ref in valid_references:
            target = self._resolve_reference(ref)
            if target:
                resolved[ref.reference_type].append((ref, target))
            else:
                self.unresolved_references.append(ref)
        
        total_resolved = sum(len(v) for v in resolved.values())
        total_valid = len(valid_references)
        actual_logger.info(f"{language_name}: Resolved {total_resolved}/{total_valid} references ({total_resolved/total_valid*100:.1f}%)" if total_valid else f"{language_name}: No references")
        
        # Log unresolved references (only valid ones that failed to resolve)
        if self.unresolved_references:
            unresolved_details = []
            for ref in self.unresolved_references:
                detail = f"  - {ref.to_entity} (type: {ref.reference_type}, file: {ref.file_path}, line: {ref.line_number}"
                if ref.scope:
                    detail += f", scope: {ref.scope}"
                detail += ")"
                unresolved_details.append(detail)
            actual_logger.info(f"{language_name}: Unresolved references ({len(self.unresolved_references)}):\n" + "\n".join(unresolved_details))
        
        self.resolved_references = resolved
        return resolved
    
    def _get_language_name(self) -> str:
        """Get the language name for logging."""
        class_name = self.__class__.__name__
        # Extract language from class name (e.g., "TypeScriptReferenceResolver" -> "TypeScript")
        # Check JavaScript first to avoid matching "Java" in "JavaScript"
        if 'JavaScript' in class_name or ('Java' in class_name and 'Script' in class_name):
            return 'JavaScript'
        elif 'TypeScript' in class_name:
            return 'TypeScript'
        elif 'Kotlin' in class_name:
            return 'Kotlin'
        elif 'Java' in class_name:
            return 'Java'
        else:
            return 'JavaScript'  # Default fallback
    
    def _should_resolve_reference(self, ref: ScopedReference) -> bool:
        """Check if a reference should be resolved."""
        target_name = ref.to_entity.strip()
        
        # Skip empty or invalid names
        if not target_name:
            return False
        
        # Skip keywords
        if target_name.lower() in JS_KEYWORDS:
            return False
        
        # Skip built-in globals (unless it's an import)
        if target_name in JS_BUILTINS and ref.reference_type != 'imports':
            return False
        
        # Skip invalid inherits references (like "extends BaseClass")
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            return False
        
        # Skip property references that are clearly just property accesses
        # (these are usually not entities but properties of entities)
        if ref.context and ref.context.get('is_property'):
            # Only skip if it's a simple property name (not a qualified name)
            if '.' not in target_name and target_name[0].islower():
                # Check if it's likely a class property by looking in scope
                if ref.scope:
                    # If we can find the class, we might be able to resolve the property
                    # For now, let's try to resolve it
                    pass
        
        return True
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve a JavaScript reference with improved strategies."""
        target_name = ref.to_entity.strip()
        
        # Clean up invalid inherits references
        if ref.reference_type == 'inherits' and target_name.startswith('extends '):
            target_name = target_name.replace('extends ', '').strip()
        
        # Handle property references - try to find the property in the class/scope
        if ref.context and ref.context.get('is_property') and ref.scope:
            # Look for the property as a variable/parameter in the scope
            scope_entities = self.by_scope.get(ref.scope, [])
            # Check if it's a class property (variable in class scope)
            candidates = [e for e in scope_entities if e.name == target_name and e.entity_type in ['variable', 'parameter']]
            if candidates:
                return candidates[0]
            
            # Also check if the scope is a class and look for properties in that class
            # (properties might be defined via this.prop = value)
            class_entities = [e for e in self.by_name.get(ref.scope, []) if e.entity_type == 'class']
            if class_entities:
                class_entity = class_entities[0]
                # Look for variables with this class as parent
                class_properties = [e for e in scope_entities if e.parent == ref.scope and e.name == target_name]
                if class_properties:
                    return class_properties[0]
        
        # Original resolution logic continues...
        
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
