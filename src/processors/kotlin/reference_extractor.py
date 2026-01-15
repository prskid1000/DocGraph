"""Kotlin reference extractor."""
from pathlib import Path
from typing import List
import tree_sitter

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class KotlinReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from Kotlin code."""
    
    def extract_references(
        self, 
        ast, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract Kotlin references including package imports."""
        references = []
        file_path_str = str(file_path)
        current_scope = []
        
        if ast is None or not ast.root_node:
            return references
        
        def extract_kotlin_refs(node: tree_sitter.Node):
            # Track scope
            if node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(class_name)
                    
                    # Extract INHERITS relationships
                    superclass_node = node.child_by_field_name('superclass')
                    if superclass_node:
                        superclass_name = superclass_node.text.decode('utf-8') if hasattr(superclass_node.text, 'decode') else str(superclass_node.text)
                        superclass_name = superclass_name.replace('extends', '').replace(':', '').strip()
                        base_name = superclass_name.split('.')[-1].strip()
                        if base_name:
                            references.append(ScopedReference(
                                from_entity=class_name,
                                to_entity=base_name,
                                reference_type='inherits',
                                file_path=file_path_str,
                                line_number=node.start_point[0] + 1,
                                qualified_name=superclass_name,
                                context={'base_class': superclass_name}
                            ))
            
            elif node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(method_name)
            
            # Extract import statements
            if node.type == 'import_declaration':
                source = node.child_by_field_name('source')
                if source:
                    package_name = source.text.decode('utf-8') if hasattr(source.text, 'decode') else str(source.text)
                    references.append(ScopedReference(
                        from_entity=file_path_str,
                        to_entity=package_name,
                        reference_type='imports',
                        file_path=file_path_str,
                        line_number=node.start_point[0] + 1,
                        scope=current_scope[-1] if current_scope else None
                    ))
            
            # Extract constructor calls (ClassName() or ClassName(...))
            # In Kotlin, constructors are called like regular function calls
            # We need to detect when it's a constructor call vs method call
            # This is tricky - we'll look for patterns like ClassName() where ClassName is a class
            elif node.type == 'call_expression':
                # Check if this might be a constructor call
                # In Kotlin, constructor calls look like: ClassName() or ClassName(args)
                # The function part should be a type_identifier or user_type
                function_node = node.child_by_field_name('function')
                if function_node:
                    # Check if it's a type identifier (likely a constructor call)
                    if function_node.type in ['type_identifier', 'user_type', 'identifier']:
                        # Try to determine if this is a constructor call
                        # If the identifier matches a class name in scope, it's likely a constructor
                        class_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                        # For now, we'll treat capitalized identifiers as potential constructor calls
                        # This is a heuristic - in Kotlin, class names are typically PascalCase
                        if class_name and class_name[0].isupper() if class_name else False:
                            enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                            scope = current_scope[-1] if current_scope else None
                            references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else file_path_str,
                                to_entity=class_name,
                                reference_type='calls',
                                file_path=file_path_str,
                                line_number=node.start_point[0] + 1,
                                scope=scope,
                                context={'expected_type': 'class', 'is_constructor': True}
                            ))
                        else:
                            # Regular function call - handled below
                            pass
            
            # Extract method calls
            elif node.type == 'method_invocation':
                name_node = node.child_by_field_name('name')
                if name_node:
                    method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                    scope = current_scope[-1] if current_scope else None
                    references.append(ScopedReference(
                        from_entity=enclosing.name if enclosing else file_path_str,
                        to_entity=method_name,
                        reference_type='calls',
                        file_path=file_path_str,
                        line_number=node.start_point[0] + 1,
                        scope=scope,
                        context={'expected_type': 'function'}
                    ))
            
            for child in node.children:
                extract_kotlin_refs(child)
            
            # Pop scope
            if node.type == 'class_declaration':
                if current_scope:
                    current_scope.pop()
            elif node.type == 'method_declaration':
                if current_scope:
                    current_scope.pop()
        
        extract_kotlin_refs(ast.root_node)
        return references
