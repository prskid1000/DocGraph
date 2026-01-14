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
            
            elif node.type in ['function_declaration', 'function_expression', 'method_definition']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    current_scope.append(name)
            
            # Extract import statements
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
            
            # Extract function calls
            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
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
