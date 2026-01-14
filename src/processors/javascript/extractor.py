"""JavaScript entity extractor."""
from pathlib import Path
from typing import List, Optional
import tree_sitter

from ..base import BaseEntityExtractor
from ...parsers.base import CodeEntity


class JavaScriptEntityExtractor(BaseEntityExtractor):
    """Extracts entities from JavaScript code."""
    
    def extract_entities(self, ast, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract entities from JavaScript AST."""
        if ast is None:
            return []
        
        entities = []
        file_path_str = str(file_path)
        
        def extract_from_node(node: tree_sitter.Node, parent: Optional[str] = None):
            if node.type in ['class_declaration', 'class_expression']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = node.text.decode('utf-8') if hasattr(node.text, 'decode') else str(node.text)
                    # Extract class name properly
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
                    # Recursively extract nested entities
                    for child in node.children:
                        extract_from_node(child, name)
            
            elif node.type in ['function_declaration', 'function_expression', 'arrow_function', 'method_definition']:
                name_node = node.child_by_field_name('name')
                name = None
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                elif node.type == 'arrow_function':
                    name = f"anonymous_{node.start_point[0]}"
                
                if name:
                    signature = self._extract_function_signature(node)
                    
                    # Extract parameters for HAS_PARAMETER relationships
                    parameters = []
                    params_node = node.child_by_field_name('parameters')
                    if params_node:
                        param_idx = 0
                        for child in params_node.children:
                            if child.type in ['identifier', 'required_parameter', 'optional_parameter']:
                                param_name_node = child.child_by_field_name('name') or child
                                if param_name_node:
                                    param_name = param_name_node.text.decode('utf-8') if hasattr(param_name_node.text, 'decode') else str(param_name_node.text)
                                    # Extract type annotation if present (TypeScript)
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
                    
                    # Extract return type for RETURNS relationship (TypeScript)
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
                            'parameters': parameters,  # For HAS_PARAMETER relationships
                            'return_type': return_type  # For RETURNS relationships
                        }
                    ))
            
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
