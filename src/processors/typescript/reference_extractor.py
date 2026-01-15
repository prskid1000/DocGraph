"""TypeScript reference extractor."""
from pathlib import Path
from typing import List
import tree_sitter

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class TypeScriptReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from TypeScript code."""
    
    def extract_references(
        self, 
        ast, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract references from TypeScript AST."""
        if ast is None:
            return []
        
        references = []
        file_path_str = str(file_path)
        current_scope = []
        
        def extract_from_node(node: tree_sitter.Node):
            # Track scope
            if node.type in ['class_declaration', 'class_expression', 'interface_declaration']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(name)
                    
                    # Extract INHERITS relationships from extends clause
                    # Try superclass field first (some tree-sitter versions)
                    superclass = node.child_by_field_name('superclass')
                    if not superclass:
                        # Try class_heritage child node (other tree-sitter versions)
                        for child in node.children:
                            if child.type == 'class_heritage':
                                # class_heritage contains the extends clause
                                # Look for identifier or member_expression in it
                                for heritage_child in child.children:
                                    if heritage_child.type in ['identifier', 'member_expression']:
                                        superclass = heritage_child
                                        break
                                break
                    
                    if superclass:
                        superclass_name = superclass.text.decode('utf-8') if hasattr(superclass.text, 'decode') else str(superclass.text)
                        base_name = superclass_name.split('.')[-1].strip()
                        references.append(ScopedReference(
                            from_entity=name,
                            to_entity=base_name,
                            reference_type='inherits',
                            file_path=file_path_str,
                            line_number=node.start_point[0] + 1,
                            scope=None,
                            qualified_name=superclass_name,
                            context={'base_class': superclass_name}
                        ))
            
            elif node.type in ['function_declaration', 'function_expression', 'method_definition']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(name)
            
            # Extract import statements (ES6)
            if node.type in ['import_statement', 'import_declaration']:
                source = node.child_by_field_name('source')
                if source:
                    module_name = source.text.decode('utf-8').strip('"\'') if hasattr(source.text, 'decode') else str(source.text).strip('"\'')
                    scope = current_scope[-1] if current_scope else None
                    references.append(ScopedReference(
                        from_entity=file_path_str,
                        to_entity=module_name,
                        reference_type='imports',
                        file_path=file_path_str,
                        line_number=node.start_point[0] + 1,
                        scope=scope
                    ))
            
            # Extract require() calls (CommonJS)
            # Note: Check for require() before processing as regular call_expression
            if node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node and function_node.type == 'identifier':
                    func_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                    if func_name == 'require':
                        # Extract module name from require() argument
                        arguments_node = node.child_by_field_name('arguments')
                        if arguments_node:
                            # Find string argument (might not be first child due to parentheses)
                            for arg_child in arguments_node.children:
                                if arg_child.type == 'string':
                                    module_name = arg_child.text.decode('utf-8').strip('"\'') if hasattr(arg_child.text, 'decode') else str(arg_child.text).strip('"\'')
                                    references.append(ScopedReference(
                                        from_entity=file_path_str,
                                        to_entity=module_name,
                                        reference_type='imports',
                                        file_path=file_path_str,
                                        line_number=node.start_point[0] + 1
                                    ))
                                    # Don't process this as a regular call_expression
                                    for child in node.children:
                                        extract_from_node(child)
                                    return
                                    break
            
            # Extract new expressions (constructor calls)
            elif node.type == 'new_expression':
                # Try to find constructor node (could be field 'constructor' or first child)
                constructor_node = node.child_by_field_name('constructor')
                if not constructor_node and node.child_count > 0:
                    # Fallback: first child is usually the constructor
                    constructor_node = node.children[0]
                
                if constructor_node:
                    # Handle different constructor patterns
                    if constructor_node.type == 'identifier':
                        # Simple constructor: new ClassName()
                        class_name = constructor_node.text.decode('utf-8') if hasattr(constructor_node.text, 'decode') else str(constructor_node.text)
                        scope = current_scope[-1] if current_scope else None
                        enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                        
                        references.append(ScopedReference(
                            from_entity=enclosing.name if enclosing else file_path_str,
                            to_entity=class_name,
                            reference_type='calls',
                            file_path=file_path_str,
                            line_number=node.start_point[0] + 1,
                            scope=scope,
                            context={'expected_type': 'class', 'is_constructor': True}
                        ))
                    elif constructor_node.type == 'member_expression':
                        # Qualified constructor: new obj.ClassName()
                        property_node = constructor_node.child_by_field_name('property')
                        if property_node:
                            class_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                            scope = current_scope[-1] if current_scope else None
                            enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                            
                            references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else file_path_str,
                                to_entity=class_name,
                                reference_type='calls',
                                file_path=file_path_str,
                                line_number=node.start_point[0] + 1,
                                scope=scope,
                                context={'expected_type': 'class', 'is_constructor': True}
                            ))
            
            # Extract function calls
            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    if function_node.type == 'identifier':
                        func_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                        scope = current_scope[-1] if current_scope else None
                        enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                        
                        references.append(ScopedReference(
                            from_entity=enclosing.name if enclosing else file_path_str,
                            to_entity=func_name,
                            reference_type='calls',
                            file_path=file_path_str,
                            line_number=node.start_point[0] + 1,
                            scope=scope,
                            context={'expected_type': 'function'}
                        ))
                    elif function_node.type == 'member_expression':
                        property_node = function_node.child_by_field_name('property')
                        if property_node:
                            method_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                            scope = current_scope[-1] if current_scope else None
                            enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                            
                            references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else file_path_str,
                                to_entity=method_name,
                                reference_type='calls',
                                file_path=file_path_str,
                                line_number=node.start_point[0] + 1,
                                scope=scope,
                                context={'expected_type': 'function', 'is_method': True}
                            ))
            
            # Extract identifier references
            elif node.type == 'identifier':
                parent = node.parent
                if parent and parent.type not in [
                    'variable_declarator', 'function_declaration', 'class_declaration', 
                    'property_definition', 'property_signature', 'method_definition',
                    'arrow_function', 'function_expression', 'class_expression', 'interface_declaration'
                ]:
                    # Skip if it's part of a member expression (obj.property) - we handle those separately
                    if parent.type == 'member_expression':
                        # Only extract if it's the object part, not the property part
                        property_node = parent.child_by_field_name('property')
                        if property_node and node == property_node:
                            # This is a property access, skip it (properties like 'length' are not entities)
                            pass
                        else:
                            # This is the object part, extract it
                            identifier_name = node.text.decode('utf-8') if hasattr(node.text, 'decode') else str(node.text)
                            scope = current_scope[-1] if current_scope else None
                            enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                            
                            references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else file_path_str,
                                to_entity=identifier_name,
                                reference_type='references',
                                file_path=file_path_str,
                                line_number=node.start_point[0] + 1,
                                scope=scope
                            ))
                    # Only extract if it's a usage (assignment target, expression, etc.)
                    elif parent.type in ['assignment_expression', 'binary_expression', 'unary_expression', 
                                      'return_statement', 'if_statement', 'while_statement', 'for_statement',
                                      'expression_statement']:
                        identifier_name = node.text.decode('utf-8') if hasattr(node.text, 'decode') else str(node.text)
                        scope = current_scope[-1] if current_scope else None
                        enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                        
                        references.append(ScopedReference(
                            from_entity=enclosing.name if enclosing else file_path_str,
                            to_entity=identifier_name,
                            reference_type='references',
                            file_path=file_path_str,
                            line_number=node.start_point[0] + 1,
                            scope=scope
                        ))
            
            # Extract member expressions (only for class properties, not object properties)
            elif node.type == 'member_expression':
                parent = node.parent
                if parent and parent.type != 'call_expression':
                    property_node = node.child_by_field_name('property')
                    object_node = node.child_by_field_name('object')
                    if property_node and property_node.type == 'property_identifier':
                        property_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                        # Skip common object properties that are not entities
                        common_object_properties = {'length', 'toString', 'valueOf', 'hasOwnProperty', 'constructor', 'prototype'}
                        if property_name not in common_object_properties:
                            # Only extract if it's a class property (this.property)
                            if object_node and object_node.type == 'this':
                                scope = current_scope[-1] if current_scope else None
                                enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                                
                                references.append(ScopedReference(
                                    from_entity=enclosing.name if enclosing else file_path_str,
                                    to_entity=property_name,
                                    reference_type='references',
                                    file_path=file_path_str,
                                    line_number=node.start_point[0] + 1,
                                    scope=scope,
                                    context={'is_property': True}
                                ))
            
            # Recurse
            for child in node.children:
                extract_from_node(child)
            
            # Pop scope
            if node.type in ['class_declaration', 'class_expression', 'interface_declaration', 
                            'function_declaration', 'function_expression', 'method_definition']:
                if current_scope:
                    current_scope.pop()
        
        if ast and ast.root_node:
            extract_from_node(ast.root_node)
        
        return references
