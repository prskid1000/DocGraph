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
        
        # Build a set of function definition lines to skip (function name lines)
        # Also check source code for "fun" keyword to identify function definitions
        function_definition_lines = set()
        lines_with_fun = set()
        
        # First, scan source code for lines with "fun" keyword
        for i, line in enumerate(source_code.split('\n'), 1):
            if 'fun ' in line or ' fun ' in line:
                lines_with_fun.add(i)
        
        def find_function_definitions(node: tree_sitter.Node):
            # Check for method_declaration nodes (both direct and inside ERROR nodes)
            if node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    line_num = name_node.start_point[0] + 1
                    function_definition_lines.add(line_num)
            elif node.type == 'ERROR':
                # ERROR nodes might contain method_declaration nodes
                # Check if this ERROR node contains a call_expression that's on a line with "fun"
                node_line = node.start_point[0] + 1
                if node_line in lines_with_fun:
                    # This ERROR node is on a line with "fun" - check if it has a call_expression
                    # If so, the call_expression is likely the function definition, not a call
                    for child in node.children:
                        if child.type == 'call_expression':
                            call_line = child.start_point[0] + 1
                            if call_line in lines_with_fun:
                                # This call_expression is on a line with "fun" - it's a function definition
                                function_definition_lines.add(call_line)
                                break
            for child in node.children:
                find_function_definitions(child)
        
        if ast and ast.root_node:
            find_function_definitions(ast.root_node)
        
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
            # Also handle call_expression and method_invocation inside ERROR nodes
            # Note: The Java parser (used for Kotlin) creates method_invocation nodes, not call_expression
            elif (node.type == 'call_expression' or 
                  node.type == 'method_invocation' or
                  (node.type == 'ERROR' and any(child.type in ['call_expression', 'method_invocation'] for child in node.children))):
                # If it's an ERROR node, find the call_expression or method_invocation inside it
                call_node = node if node.type in ['call_expression', 'method_invocation'] else None
                if not call_node:
                    for child in node.children:
                        if child.type in ['call_expression', 'method_invocation']:
                            call_node = child
                            break
                
                if call_node:
                    line_num = call_node.start_point[0] + 1
                    function_node = call_node.child_by_field_name('function')
                    
                    # Skip if this call is on a function definition line
                    # Check if the line contains "fun" keyword and this is likely a function definition
                    should_skip = False
                    if line_num in function_definition_lines:
                        should_skip = True
                    elif line_num in lines_with_fun and function_node:
                        # Check if this call_expression is actually a function definition
                        # by checking if the function name appears right after "fun" on this line
                        if function_node.type == 'identifier':
                            func_name = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                            # Get the line text
                            source_lines = source_code.split('\n')
                            line_text = source_lines[line_num - 1] if line_num <= len(source_lines) else ''
                            # Check if "fun func_name(" appears on this line
                            import re
                            if re.search(rf'\bfun\s+{re.escape(func_name)}\s*\(', line_text):
                                should_skip = True
                    
                    if should_skip:
                        # This is likely a function definition, not a call - skip it
                        pass
                    elif function_node:
                        if function_node:
                            # Handle member expressions (Utils.helperFunction(), instance.method())
                            # Kotlin uses various node types: member_expression, navigation_expression, scoped_identifier, etc.
                            is_member_expr = function_node.type in ['member_expression', 'navigation_expression', 'scoped_identifier', 'user_type']
                            if is_member_expr:
                                # Try to extract property/method name and object name
                                property_node = function_node.child_by_field_name('property')
                                object_node = function_node.child_by_field_name('object')
                                
                                # If no property field, try to find the last identifier (method name)
                                if not property_node:
                                    # Look for identifier children - last one is usually the method name
                                    identifiers = [c for c in function_node.children if c.type == 'identifier']
                                    if len(identifiers) >= 2:
                                        object_node = identifiers[0]
                                        property_node = identifiers[-1]
                                    elif len(identifiers) == 1:
                                        # Might be a scoped call like Utils.helperFunction
                                        # Check parent structure
                                        property_node = identifiers[0]
                                
                                # Also try navigation_expression structure (object.method)
                                if not property_node and function_node.type == 'navigation_expression':
                                    # Navigation expression: object DOT method
                                    children = list(function_node.children)
                                    for i, child in enumerate(children):
                                        if child.type == '.' and i + 1 < len(children):
                                            # Next child should be the method name
                                            if i > 0:
                                                object_node = children[i-1]
                                            property_node = children[i+1]
                                            break
                                
                                # Also check for scoped_identifier (Utils.helperFunction pattern)
                                if not property_node and function_node.type == 'scoped_identifier':
                                    # scoped_identifier: Utils.helperFunction
                                    identifiers = [c for c in function_node.children if c.type == 'identifier']
                                    if len(identifiers) >= 2:
                                        object_node = identifiers[0]
                                        property_node = identifiers[-1]
                                    elif len(identifiers) == 1:
                                        # Might be a single identifier in scoped context
                                        property_node = identifiers[0]
                                
                                if property_node:
                                    method_name = property_node.text.decode('utf-8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                                    # Skip common object properties that are not function calls
                                    common_properties = {'length', 'toString', 'valueOf', 'hasOwnProperty', 'constructor', 'prototype'}
                                    if method_name not in common_properties:
                                        call_key = (line_num, method_name)
                                        if call_key not in extracted_calls:
                                            extracted_calls.add(call_key)
                                            enclosing = self._find_enclosing_entity(line_num, entities)
                                            scope = current_scope[-1] if current_scope else None
                                            
                                            # Try to extract object name for qualified name (e.g., Utils.helperFunction)
                                            qualified_name = None
                                            object_name_str = None
                                            if object_node:
                                                object_name_str = object_node.text.decode('utf-8') if hasattr(object_node.text, 'decode') else str(object_node.text)
                                                # If object is an identifier or type, use it for qualified lookup
                                                if object_node.type in ['identifier', 'type_identifier', 'user_type']:
                                                    qualified_name = f"{object_name_str}.{method_name}"
                                            elif function_node.type == 'scoped_identifier':
                                                # For scoped_identifier, try to extract from the node text
                                                node_text = function_node.text.decode('utf-8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                                                if '.' in node_text:
                                                    parts = node_text.split('.')
                                                    if len(parts) >= 2:
                                                        object_name_str = parts[0]
                                                        qualified_name = f"{object_name_str}.{method_name}"
                                            
                                            references.append(ScopedReference(
                                                from_entity=enclosing.name if enclosing else file_path_str,
                                                to_entity=method_name,
                                                reference_type='calls',
                                                file_path=file_path_str,
                                                line_number=line_num,
                                                scope=scope,
                                                qualified_name=qualified_name,
                                                context={'expected_type': 'function', 'is_method': True, 'object_name': object_name_str}
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
            
            # Extract method calls from assignment_expressions (e.g., val result = Utils.helperFunction(data))
            elif node.type == 'assignment_expression':
                # Look for method_invocation or call_expression in the right side of assignment
                # assignment_expression structure: left = right
                # Find the right side which might contain the method call
                for child in node.children:
                    if child.type in ['method_invocation', 'call_expression']:
                        # Process this as a method call
                        method_node = child
                        name_node = method_node.child_by_field_name('name')
                        if name_node:
                            method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                            line_num = method_node.start_point[0] + 1
                            
                            # Skip keywords and common object methods
                            kotlin_keywords = {'if', 'for', 'while', 'when', 'try', 'catch', 'finally', 'return', 'throw', 'break', 'continue'}
                            common_methods = {'isEmpty', 'isNotEmpty', 'length', 'toString', 'equals', 'hashCode'}
                            if method_name not in kotlin_keywords and method_name not in common_methods:
                                call_key = (line_num, method_name)
                                if call_key not in extracted_calls:
                                    extracted_calls.add(call_key)
                                    enclosing = self._find_enclosing_entity(line_num, entities)
                                    scope = current_scope[-1] if current_scope else None
                                    
                                    # Check for object.method pattern
                                    object_node = method_node.child_by_field_name('object')
                                    qualified_name = None
                                    object_name_str = None
                                    if object_node:
                                        object_name_str = object_node.text.decode('utf-8') if hasattr(object_node.text, 'decode') else str(object_node.text)
                                        qualified_name = f"{object_name_str}.{method_name}"
                                    
                                    references.append(ScopedReference(
                                        from_entity=enclosing.name if enclosing else file_path_str,
                                        to_entity=method_name,
                                        reference_type='calls',
                                        file_path=file_path_str,
                                        line_number=line_num,
                                        scope=scope,
                                        qualified_name=qualified_name,
                                        context={'expected_type': 'function', 'object_name': object_name_str}
                                    ))
                        break
            
            # Extract method calls (method_invocation)
            # Also handle method_invocation inside ERROR nodes (but not already processed in assignment_expression)
            elif node.type == 'method_invocation' or (node.type == 'ERROR' and any(child.type == 'method_invocation' for child in node.children)):
                # If it's an ERROR node, find the method_invocation inside it
                method_node = node if node.type == 'method_invocation' else None
                if not method_node:
                    for child in node.children:
                        if child.type == 'method_invocation':
                            method_node = child
                            break
                
                if method_node:
                    line_num = method_node.start_point[0] + 1
                    name_node = method_node.child_by_field_name('name')
                    if name_node:
                        method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                        
                        # Skip if this is a function definition (has "fun" on the same line)
                        should_skip = False
                        if line_num in function_definition_lines:
                            should_skip = True
                        elif line_num in lines_with_fun:
                            # Check if "fun method_name(" appears on this line
                            source_lines = source_code.split('\n')
                            line_text = source_lines[line_num - 1] if line_num <= len(source_lines) else ''
                            import re
                            if re.search(rf'\bfun\s+{re.escape(method_name)}\s*\(', line_text):
                                should_skip = True
                        
                        if not should_skip:
                            # Skip keywords and common object methods that are not function calls
                            kotlin_keywords = {'if', 'for', 'while', 'when', 'try', 'catch', 'finally', 'return', 'throw', 'break', 'continue'}
                            common_methods = {'isEmpty', 'isNotEmpty', 'length', 'toString', 'equals', 'hashCode'}
                            if method_name not in kotlin_keywords and method_name not in common_methods:
                                call_key = (line_num, method_name)
                                if call_key not in extracted_calls:
                                    extracted_calls.add(call_key)
                                    enclosing = self._find_enclosing_entity(line_num, entities)
                                    scope = current_scope[-1] if current_scope else None
                                    
                                    # Check for object.method pattern
                                    object_node = method_node.child_by_field_name('object')
                                    qualified_name = None
                                    object_name_str = None
                                    if object_node:
                                        object_name_str = object_node.text.decode('utf-8') if hasattr(object_node.text, 'decode') else str(object_node.text)
                                        qualified_name = f"{object_name_str}.{method_name}"
                                    
                                    references.append(ScopedReference(
                                        from_entity=enclosing.name if enclosing else file_path_str,
                                        to_entity=method_name,
                                        reference_type='calls',
                                        file_path=file_path_str,
                                        line_number=line_num,
                                        scope=scope,
                                        qualified_name=qualified_name,
                                        context={'expected_type': 'function', 'object_name': object_name_str}
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
