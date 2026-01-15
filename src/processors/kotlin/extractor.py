"""Kotlin entity extractor."""
from ...parsers.base import CodeEntity
from pathlib import Path
from typing import List
import tree_sitter

from ..base import BaseEntityExtractor


class KotlinEntityExtractor(BaseEntityExtractor):
    """Extracts entities from Kotlin code."""
    
    def extract_entities(self, ast, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract Kotlin entities (classes, functions, properties)."""
        entities = []
        file_path_str = str(file_path)
        
        if ast is None or not ast.root_node:
            return entities
        
        processed_node_ids = set()  # Track processed nodes to avoid duplicates
        
        def extract_kotlin_entities(node: tree_sitter.Node, parent=None):
            # Skip if already processed (to avoid duplicates from recursive calls)
            node_id = id(node)
            if node_id in processed_node_ids:
                return
            processed_node_ids.add(node_id)
            
            processed_children = False  # Track if children were already processed
            # Handle both class_declaration and ERROR nodes that contain classes
            is_class_node = node.type == 'class_declaration'
            is_error_with_class = node.type == 'ERROR' and any(child.type == 'class' for child in node.children)
            # Also handle object declarations (object Utils { ... })
            # In Java parser, objects might be parsed as class_declaration or ERROR nodes
            is_object_node = node.type == 'object_declaration' or (node.type == 'ERROR' and any(child.type == 'object' for child in node.children))
            # Check if it's an object by looking at the source code
            is_object_by_code = False
            if node.type in ['class_declaration', 'ERROR', 'local_variable_declaration']:
                node_text = node.text.decode('utf-8') if hasattr(node.text, 'decode') else str(node.text)
                # Check if this node contains "object" keyword followed by an identifier
                import re
                if re.search(r'\bobject\s+[A-Za-z_]', node_text):
                    is_object_by_code = True
            
            if is_class_node or is_error_with_class or is_object_node or is_object_by_code:
                # For ERROR nodes, find the class/object identifier
                name_node = None
                if is_class_node:
                    name_node = node.child_by_field_name('name')
                elif is_object_node and node.type == 'object_declaration':
                    name_node = node.child_by_field_name('name')
                elif is_object_by_code:
                    # For objects parsed as class_declaration, ERROR, or local_variable_declaration
                    node_text = node.text.decode('utf-8') if hasattr(node.text, 'decode') else str(node.text)
                    # Extract identifier after 'object' keyword from text using regex
                    import re
                    match = re.search(r'object\s+([A-Za-z_][A-Za-z0-9_]*)', node_text)
                    if match:
                        object_name = match.group(1)
                        # Find the identifier node matching this name (search recursively)
                        def find_identifier_recursive(n, target_name):
                            for c in n.children:
                                c_text = c.text.decode('utf-8') if hasattr(c.text, 'decode') else str(c.text)
                                if c.type == 'identifier' and c_text == target_name:
                                    return c
                                result = find_identifier_recursive(c, target_name)
                                if result:
                                    return result
                            return None
                        name_node = find_identifier_recursive(node, object_name)
                        # If still not found, we'll create the entity with the extracted name
                        if not name_node:
                            # Create a virtual name_node-like object
                            class VirtualNameNode:
                                def __init__(self, name):
                                    self.text = name.encode() if isinstance(name, str) else name
                            name_node = VirtualNameNode(object_name)
                else:
                    # In ERROR node, look for identifier after 'class' or 'object' token
                    for i, child in enumerate(node.children):
                        if (child.type == 'class' or child.type == 'object') and i + 1 < len(node.children):
                            next_child = node.children[i + 1]
                            if next_child.type == 'identifier':
                                name_node = next_child
                                break
                
                if name_node:
                    # Handle both real tree_sitter nodes and virtual nodes
                    if hasattr(name_node, 'text'):
                        name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    else:
                        name = str(name_node)
                    # Treat objects as classes for entity purposes
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='class',  # Objects are treated as classes
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent
                    ))
                    # Extract children with object/class name as parent
                    # Skip the recursive call at the end for these children
                    for child in node.children:
                        extract_kotlin_entities(child, name)
                    # Mark that we've processed children - don't process again at line 166
                    processed_children = True
            
            elif node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    signature = self._extract_kotlin_method_signature(node)
                    
                    # Extract parameters
                    parameters = []
                    params_node = node.child_by_field_name('parameters')
                    if params_node:
                        param_idx = 0
                        for child in params_node.children:
                            if child.type == 'formal_parameter':
                                param_name_node = child.child_by_field_name('name')
                                param_type_node = child.child_by_field_name('type')
                                if param_name_node:
                                    param_name = param_name_node.text.decode('utf-8') if hasattr(param_name_node.text, 'decode') else str(param_name_node.text)
                                    param_type = None
                                    if param_type_node:
                                        param_type = param_type_node.text.decode('utf-8') if hasattr(param_type_node.text, 'decode') else str(param_type_node.text)
                                    parameters.append({
                                        'name': param_name,
                                        'type': param_type,
                                        'position': param_idx
                                    })
                                    param_idx += 1
                    
                    # Extract return type
                    return_type = None
                    return_type_node = node.child_by_field_name('type')
                    if return_type_node:
                        return_type = return_type_node.text.decode('utf-8') if hasattr(return_type_node.text, 'decode') else str(return_type_node.text)
                    
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='function',
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        signature=signature,
                        parent=parent,
                        metadata={
                            'parameters': parameters,
                            'return_type': return_type
                        }
                    ))
            
            elif node.type == 'field_declaration' and parent:
                # Extract field/property names
                for child in node.children:
                    if child.type == 'variable_declarator':
                        name_node = child.child_by_field_name('name')
                        if name_node:
                            name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                            entities.append(CodeEntity(
                                name=name,
                                entity_type='variable',
                                file_path=file_path_str,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                start_column=node.start_point[1],
                                end_column=node.end_point[1],
                                parent=parent
                            ))
            
            # Only process children recursively if they weren't already processed
            # (e.g., if this is a class/object, children were already processed with correct parent)
            if not processed_children:
                for child in node.children:
                    extract_kotlin_entities(child, parent)
        
        extract_kotlin_entities(ast.root_node)
        return entities
    
    def _extract_kotlin_method_signature(self, node: tree_sitter.Node) -> str:
        """Extract Kotlin method signature."""
        name_node = node.child_by_field_name('name')
        name = name_node.text.decode('utf-8') if name_node and hasattr(name_node.text, 'decode') else 'method'
        params = node.child_by_field_name('parameters')
        param_text = params.text.decode('utf-8') if params and hasattr(params.text, 'decode') else '()'
        return f"{name}{param_text}"
