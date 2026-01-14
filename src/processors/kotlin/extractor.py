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
        
        def extract_kotlin_entities(node: tree_sitter.Node, parent=None):
            if node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='class',
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent
                    ))
                    for child in node.children:
                        extract_kotlin_entities(child, name)
            
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
