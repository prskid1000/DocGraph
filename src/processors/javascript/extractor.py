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
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='function',
                        file_path=file_path_str,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        signature=signature,
                        parent=parent
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
