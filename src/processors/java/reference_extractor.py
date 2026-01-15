"""Java reference extractor."""
from pathlib import Path
from typing import List
import tree_sitter

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class JavaReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from Java code."""
    
    def extract_references(
        self, 
        ast, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract Java references including package imports."""
        references = []
        file_path_str = str(file_path)
        current_scope = []
        
        if ast is None or not ast.root_node:
            return references
        
        def extract_java_refs(node: tree_sitter.Node):
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
                        superclass_name = superclass_name.replace('extends', '').strip()
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
            
            # Extract constructor calls (new ClassName())
            elif node.type == 'object_creation_expression':
                # Try to find the type/class being instantiated
                type_node = node.child_by_field_name('type')
                if not type_node and node.child_count > 0:
                    # Fallback: first child might be the type
                    type_node = node.children[0]
                
                if type_node:
                    # Extract class name
                    if type_node.type == 'type_identifier':
                        class_name = type_node.text.decode('utf-8') if hasattr(type_node.text, 'decode') else str(type_node.text)
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
                    elif type_node.type == 'generic_type':
                        # Handle generic types like new ArrayList<String>()
                        type_identifier = type_node.child_by_field_name('type')
                        if type_identifier:
                            class_name = type_identifier.text.decode('utf-8') if hasattr(type_identifier.text, 'decode') else str(type_identifier.text)
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
                extract_java_refs(child)
            
            # Pop scope
            if node.type == 'class_declaration':
                if current_scope:
                    current_scope.pop()
            elif node.type == 'method_declaration':
                if current_scope:
                    current_scope.pop()
        
        extract_java_refs(ast.root_node)
        return references
