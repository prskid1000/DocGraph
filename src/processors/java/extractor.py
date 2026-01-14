"""Java entity extractor."""
from ..javascript.extractor import JavaScriptEntityExtractor
from ...parsers.base import CodeEntity
from pathlib import Path
from typing import List
import tree_sitter


class JavaEntityExtractor(JavaScriptEntityExtractor):
    """Extracts entities from Java code."""
    
    def extract_entities(self, ast, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract Java entities (classes, methods, fields)."""
        entities = []
        file_path_str = str(file_path)
        
        if ast is None or not ast.root_node:
            return entities
        
        def extract_java_entities(node: tree_sitter.Node, parent=None):
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
                        extract_java_entities(child, name)
            
            elif node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    signature = self._extract_java_method_signature(node)
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
            
            elif node.type == 'field_declaration' and parent:
                # Extract field names
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
                extract_java_entities(child, parent)
        
        extract_java_entities(ast.root_node)
        return entities
    
    def _extract_java_method_signature(self, node: tree_sitter.Node) -> str:
        """Extract Java method signature."""
        name_node = node.child_by_field_name('name')
        name = name_node.text.decode('utf-8') if name_node and hasattr(name_node.text, 'decode') else 'method'
        params = node.child_by_field_name('parameters')
        param_text = params.text.decode('utf-8') if params and hasattr(params.text, 'decode') else '()'
        return f"{name}{param_text}"
