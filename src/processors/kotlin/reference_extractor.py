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
        
        extracted_calls = set()  # Track (line_number, to_entity) to avoid duplicates
        
        def extract_kotlin_refs(node: tree_sitter.Node):
            # Track scope
            # Handle both class_declaration and ERROR nodes that contain classes
            is_class_node = node.type == 'class_declaration'
            is_error_with_class = node.type == 'ERROR' and any(child.type == 'class' for child in node.children)
            
            if is_class_node or is_error_with_class:
                # For ERROR nodes, find the class identifier
                name_node = None
                if is_class_node:
                    name_node = node.child_by_field_name('name')
                else:
                    # In ERROR node, look for identifier after 'class' token
                    for i, child in enumerate(node.children):
                        if child.type == 'class' and i + 1 < len(node.children):
                            next_child = node.children[i + 1]
                            if next_child.type == 'identifier':
                                name_node = next_child
                                break
                
                if name_node:
                    class_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(class_name)
                    
                    # Extract INHERITS relationships
                    # Kotlin uses colon syntax: class Derived : BaseClass
                    superclass_node = None
                    if is_class_node:
                        superclass_node = node.child_by_field_name('superclass')
                    
                    if not superclass_node:
                        # Try to find supertype in children (Kotlin uses different structure)
                        # Look for type_identifier after colon or in supertype_list
                        # The inheritance colon comes AFTER the constructor parameters, not inside them
                        # Pattern: "class Name(params) : BaseClass" or "class Name : BaseClass"
                        # In ERROR nodes, the colon might be inside an ERROR child node
                        for i, child in enumerate(node.children):
                            # Look for ERROR node that contains both ')' and ':' (inheritance pattern)
                            if child.type == 'ERROR':
                                error_text = child.text.decode('utf-8') if hasattr(child.text, 'decode') else str(child.text)
                                # Check if this ERROR contains the inheritance pattern: ") :"
                                if ')' in error_text and ':' in error_text:
                                    # The next identifier after this ERROR should be the base class
                                    for j, next_child in enumerate(node.children[i+1:], start=i+1):
                                        if next_child.type in ['type_identifier', 'user_type', 'type_reference', 'identifier']:
                                            # Make sure it's not part of formal_parameters
                                            if next_child.type == 'identifier':
                                                # Check if it's followed by '(' (would be constructor call, not base class)
                                                if j+1 < len(node.children) and node.children[j+1].type != '(':
                                                    superclass_node = next_child
                                                    break
                                                elif j+1 >= len(node.children) or node.children[j+1].type != '(':
                                                    superclass_node = next_child
                                                    break
                                            else:
                                                superclass_node = next_child
                                                break
                                    if superclass_node:
                                        break
                            elif child.type == ':':
                                # Direct colon - check if it's for inheritance (after ')')
                                # Look backwards for ')'
                                found_closing_paren = False
                                for k in range(i-1, max(-1, i-10), -1):
                                    if node.children[k].type == ')':
                                        found_closing_paren = True
                                        break
                                
                                if found_closing_paren or i == 1:  # Colon right after class name (no constructor)
                                    # Next identifier should be the base class
                                    for j, next_child in enumerate(node.children[i+1:], start=i+1):
                                        if next_child.type in ['type_identifier', 'user_type', 'type_reference', 'identifier']:
                                            superclass_node = next_child
                                            break
                                    if superclass_node:
                                        break
                            elif child.type == 'supertype_clause':
                                # Next type_identifier or user_type should be the superclass
                                for next_child in child.children:
                                    if next_child.type in ['type_identifier', 'user_type', 'type_reference']:
                                        superclass_node = next_child
                                        break
                                if superclass_node:
                                    break
                            # Also check for supertype_list
                            elif child.type == 'supertype_list':
                                for subtype in child.children:
                                    if subtype.type in ['type_identifier', 'user_type', 'type_reference']:
                                        superclass_node = subtype
                                        break
                                if superclass_node:
                                    break
                    
                    if superclass_node:
                        # Extract the type identifier from superclass node
                        if superclass_node.type in ['type_identifier', 'user_type', 'type_reference', 'identifier']:
                            superclass_name = superclass_node.text.decode('utf-8') if hasattr(superclass_node.text, 'decode') else str(superclass_node.text)
                        else:
                            # If it's a container node, find the type_identifier inside it
                            type_id = None
                            for child in superclass_node.children:
                                if child.type in ['type_identifier', 'user_type', 'type_reference', 'identifier']:
                                    type_id = child
                                    break
                            if type_id:
                                superclass_name = type_id.text.decode('utf-8') if hasattr(type_id.text, 'decode') else str(type_id.text)
                            else:
                                superclass_name = superclass_node.text.decode('utf-8') if hasattr(superclass_node.text, 'decode') else str(superclass_node.text)
                        
                        superclass_name = superclass_name.replace('extends', '').replace(':', '').strip()
                        # Remove parentheses if present (e.g., "BaseClass(name)" -> "BaseClass")
                        if '(' in superclass_name:
                            superclass_name = superclass_name.split('(')[0].strip()
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
                # Kotlin imports may use scoped_identifier or identifier child, not 'source' field
                source = node.child_by_field_name('source')
                if not source:
                    # Look for scoped_identifier or identifier in children
                    for child in node.children:
                        if child.type in ['scoped_identifier', 'identifier', 'type_identifier']:
                            source = child
                            break
                
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
            
            # Extract constructor calls and function calls (ClassName() or func())
            # In Kotlin, constructors are called like regular function calls
            # We need to detect when it's a constructor call vs method call
            # Also handle call_expression inside ERROR nodes
            elif node.type == 'call_expression' or (node.type == 'ERROR' and any(child.type == 'call_expression' for child in node.children)):
                # If it's an ERROR node, find the call_expression inside it
                call_node = node if node.type == 'call_expression' else None
                if not call_node:
                    for child in node.children:
                        if child.type == 'call_expression':
                            call_node = child
                            break
                
                if call_node:
                    function_node = call_node.child_by_field_name('function')
                    if function_node:
                        # Handle member expressions (Utils.helperFunction(), instance.method())
                        if function_node.type == 'member_expression':
                            property_node = function_node.child_by_field_name('property')
                            object_node = function_node.child_by_field_name('object')
                            if property_node:
                                method_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                                # Skip common object properties that are not function calls
                                common_properties = {'length', 'toString', 'valueOf', 'hasOwnProperty', 'constructor', 'prototype'}
                                if method_name not in common_properties:
                                    line_num = call_node.start_point[0] + 1
                                    call_key = (line_num, method_name)
                                    if call_key not in extracted_calls:
                                        extracted_calls.add(call_key)
                                        enclosing = self._find_enclosing_entity(line_num, entities)
                                        scope = current_scope[-1] if current_scope else None
                                        
                                        # Try to extract object name for qualified name (e.g., Utils.helperFunction)
                                        qualified_name = None
                                        if object_node:
                                            object_name = object_node.text.decode('utf-8') if hasattr(object_node.text, 'decode') else str(object_node.text)
                                            # If object is an identifier, use it for qualified lookup
                                            if object_node.type == 'identifier':
                                                qualified_name = f"{object_name}.{method_name}"
                                        
                                        references.append(ScopedReference(
                                            from_entity=enclosing.name if enclosing else file_path_str,
                                            to_entity=method_name,
                                            reference_type='calls',
                                            file_path=file_path_str,
                                            line_number=line_num,
                                            scope=scope,
                                            qualified_name=qualified_name,
                                            context={'expected_type': 'function', 'is_method': True, 'object_name': object_node.text.decode('utf-8') if object_node and hasattr(object_node.text, 'decode') else (str(object_node.text) if object_node else None)}
                                        ))
                        # Handle simple identifiers (could be constructor or function call)
                        elif function_node.type in ['type_identifier', 'user_type', 'identifier']:
                            class_or_func_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                            # Skip keywords that are not function calls
                            kotlin_keywords = {'if', 'for', 'while', 'when', 'try', 'catch', 'finally', 'return', 'throw', 'break', 'continue'}
                            if class_or_func_name not in kotlin_keywords:
                                # Treat capitalized identifiers as potential constructor calls
                                # Otherwise treat as function calls
                                is_constructor = class_or_func_name and class_or_func_name[0].isupper() if class_or_func_name else False
                                line_num = call_node.start_point[0] + 1
                                call_key = (line_num, class_or_func_name)
                                if call_key not in extracted_calls:
                                    extracted_calls.add(call_key)
                                    enclosing = self._find_enclosing_entity(line_num, entities)
                                    scope = current_scope[-1] if current_scope else None
                                    references.append(ScopedReference(
                                        from_entity=enclosing.name if enclosing else file_path_str,
                                        to_entity=class_or_func_name,
                                        reference_type='calls',
                                        file_path=file_path_str,
                                        line_number=line_num,
                                        scope=scope,
                                        context={'expected_type': 'class' if is_constructor else 'function', 'is_constructor': is_constructor}
                                    ))
            
            # Extract method calls (method_invocation)
            # Also handle method_invocation inside ERROR nodes
            elif node.type == 'method_invocation' or (node.type == 'ERROR' and any(child.type == 'method_invocation' for child in node.children)):
                # If it's an ERROR node, find the method_invocation inside it
                method_node = node if node.type == 'method_invocation' else None
                if not method_node:
                    for child in node.children:
                        if child.type == 'method_invocation':
                            method_node = child
                            break
                
                if method_node:
                    name_node = method_node.child_by_field_name('name')
                    if name_node:
                        method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                        # Skip keywords and common object methods that are not function calls
                        kotlin_keywords = {'if', 'for', 'while', 'when', 'try', 'catch', 'finally', 'return', 'throw', 'break', 'continue'}
                        common_methods = {'isEmpty', 'isNotEmpty', 'length', 'toString', 'equals', 'hashCode'}
                        if method_name not in kotlin_keywords and method_name not in common_methods:
                            line_num = method_node.start_point[0] + 1
                            call_key = (line_num, method_name)
                            if call_key not in extracted_calls:
                                extracted_calls.add(call_key)
                                enclosing = self._find_enclosing_entity(line_num, entities)
                                scope = current_scope[-1] if current_scope else None
                                references.append(ScopedReference(
                                    from_entity=enclosing.name if enclosing else file_path_str,
                                    to_entity=method_name,
                                    reference_type='calls',
                                    file_path=file_path_str,
                                    line_number=line_num,
                                    scope=scope,
                                    context={'expected_type': 'function'}
                                ))
            
            # Recurse to children
            for child in node.children:
                extract_kotlin_refs(child)
            
            # Pop scope
            if node.type in ['class_declaration', 'ERROR']:
                # Check if this ERROR node contained a class
                if node.type == 'ERROR':
                    if any(child.type == 'class' for child in node.children):
                        if current_scope:
                            current_scope.pop()
                else:
                    if current_scope:
                        current_scope.pop()
            elif node.type == 'method_declaration':
                if current_scope:
                    current_scope.pop()
        
        extract_kotlin_refs(ast.root_node)
        return references
