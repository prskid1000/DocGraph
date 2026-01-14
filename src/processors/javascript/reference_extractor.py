"""JavaScript reference extractor."""
from pathlib import Path
from typing import List
import tree_sitter

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class JavaScriptReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from JavaScript code."""
    
    def extract_references(
        self, 
        ast, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract references from JavaScript AST."""
        if ast is None:
            return []
        
        references = []
        file_path_str = str(file_path)
        current_scope = []
        
        def extract_from_node(node: tree_sitter.Node):
            # Track scope
            if node.type in ['class_declaration', 'class_expression']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(name)
                    
                    # Extract INHERITS relationships from extends clause
                    superclass = node.child_by_field_name('superclass')
                    if superclass:
                        superclass_name = superclass.text.decode('utf-8') if hasattr(superclass.text, 'decode') else str(superclass.text)
                        # Extract class name from superclass (handle qualified names)
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
            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node and function_node.type == 'identifier':
                    func_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                    if func_name == 'require':
                        # Extract module name from require() argument
                        arguments_node = node.child_by_field_name('arguments')
                        if arguments_node and arguments_node.child_count > 0:
                            arg_node = arguments_node.children[0]
                            if arg_node.type == 'string':
                                module_name = arg_node.text.decode('utf-8').strip('"\'') if hasattr(arg_node.text, 'decode') else str(arg_node.text).strip('"\'')
                                references.append(ScopedReference(
                                    from_entity=file_path_str,
                                    to_entity=module_name,
                                    reference_type='imports',
                                    file_path=file_path_str,
                                    line_number=node.start_point[0] + 1
                                ))
            
            # Extract function calls
            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    # Handle different function call patterns
                    if function_node.type == 'identifier':
                        # Simple function call: func()
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
                        # Method call: obj.method()
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
            
            # Extract identifier references (variables, properties, etc.)
            # Only extract if it's a usage, not a declaration
            elif node.type == 'identifier':
                parent = node.parent
                # Skip declarations and definitions
                if parent and parent.type not in [
                    'variable_declarator', 'function_declaration', 'class_declaration', 
                    'property_definition', 'property_signature', 'method_definition',
                    'arrow_function', 'function_expression', 'class_expression'
                ]:
                    # Only extract if it's a usage (assignment target, expression, etc.)
                    if parent.type in ['assignment_expression', 'binary_expression', 'unary_expression', 
                                      'return_statement', 'if_statement', 'while_statement', 'for_statement',
                                      'expression_statement', 'member_expression']:
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
            
            # Extract member expressions (obj.property) - but not if it's part of a call (already handled)
            elif node.type == 'member_expression':
                # Only extract if it's not a call expression
                parent = node.parent
                if parent and parent.type != 'call_expression':
                    property_node = node.child_by_field_name('property')
                    if property_node and property_node.type == 'property_identifier':
                        property_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
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
            if node.type in ['class_declaration', 'class_expression', 'function_declaration', 'function_expression', 'method_definition']:
                if current_scope:
                    current_scope.pop()
        
        if ast and ast.root_node:
            extract_from_node(ast.root_node)
        
        return references
