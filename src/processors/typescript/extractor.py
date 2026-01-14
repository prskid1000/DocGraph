"""TypeScript entity extractor."""
from ...parsers.base import CodeEntity
from pathlib import Path
from typing import List, Optional
import tree_sitter

from ..base import BaseEntityExtractor


class TypeScriptEntityExtractor(BaseEntityExtractor):
    """Extracts entities from TypeScript code including interfaces and types."""
    
    def extract_entities(self, ast, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract entities including TypeScript-specific ones."""
        entities = []
        file_path_str = str(file_path)
        
        if ast is None:
            return entities
        
        def extract_from_node(node: tree_sitter.Node, parent: Optional[str] = None):
            # Extract classes
            if node.type in ['class_declaration', 'class_expression']:
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
                        extract_from_node(child, name)
            
            # Extract interfaces
            elif node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='class',  # Treat interface as class-like
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent,
                        metadata={'is_interface': True}
                    ))
            
            # Extract type aliases
            elif node.type == 'type_alias_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='type',
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent
                    ))
            
            # Extract functions
            elif node.type in ['function_declaration', 'function_expression', 'arrow_function', 'method_definition']:
                name_node = node.child_by_field_name('name')
                name = None
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                elif node.type == 'arrow_function':
                    name = f"anonymous_{node.start_point[0]}"
                
                if name:
                    signature = self._extract_function_signature(node)
                    
                    # Extract parameters
                    parameters = []
                    params_node = node.child_by_field_name('parameters')
                    if params_node:
                        param_idx = 0
                        for child in params_node.children:
                            if child.type in ['identifier', 'required_parameter', 'optional_parameter']:
                                param_name_node = child.child_by_field_name('name') or child
                                if param_name_node:
                                    param_name = param_name_node.text.decode('utf-8') if hasattr(param_name_node.text, 'decode') else str(param_name_node.text)
                                    param_type = None
                                    type_node = child.child_by_field_name('type')
                                    if type_node:
                                        param_type = type_node.text.decode('utf-8') if hasattr(type_node.text, 'decode') else str(type_node.text)
                                    parameters.append({
                                        'name': param_name,
                                        'type': param_type,
                                        'position': param_idx
                                    })
                                    param_idx += 1
                    
                    # Extract return type
                    return_type = None
                    return_type_node = node.child_by_field_name('return_type')
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
            
            # Extract variables
            elif node.type == 'variable_declaration' and parent is None:
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
            
            # Recurse through children
            for child in node.children:
                extract_from_node(child, parent)
        
        if ast and ast.root_node:
            extract_from_node(ast.root_node)
        
        return entities
    
    def _extract_function_signature(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract function signature."""
        name_node = node.child_by_field_name('name')
        name = name_node.text.decode('utf-8') if name_node and hasattr(name_node.text, 'decode') else 'anonymous'
        params = node.child_by_field_name('parameters')
        param_text = params.text.decode('utf-8') if params and hasattr(params.text, 'decode') else '()'
        return f"{name}{param_text}"
