"""TypeScript entity extractor - extends JavaScript with interface/type support."""
from ..javascript.extractor import JavaScriptEntityExtractor
from ...parsers.base import CodeEntity
from pathlib import Path
from typing import List
import tree_sitter


class TypeScriptEntityExtractor(JavaScriptEntityExtractor):
    """Extracts entities from TypeScript code including interfaces and types."""
    
    def extract_entities(self, ast, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract entities including TypeScript-specific ones."""
        entities = super().extract_entities(ast, file_path, source_code)
        file_path_str = str(file_path)
        
        if ast is None or not ast.root_node:
            return entities
        
        def extract_typescript_entities(node: tree_sitter.Node, parent=None):
            # Extract interfaces
            if node.type == 'interface_declaration':
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
            
            for child in node.children:
                extract_typescript_entities(child, parent)
        
        extract_typescript_entities(ast.root_node)
        return entities
