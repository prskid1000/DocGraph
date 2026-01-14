"""Java reference extractor."""
from ..javascript.reference_extractor import JavaScriptReferenceExtractor
from pathlib import Path
from typing import List
import tree_sitter

from ..base import ScopedReference
from ...parsers.base import CodeEntity


class JavaReferenceExtractor(JavaScriptReferenceExtractor):
    """Extracts references from Java code."""
    
    def extract_references(
        self, 
        ast, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract Java references including package imports."""
        references = []
        file_path_str = str(file_path)
        
        if ast is None or not ast.root_node:
            return references
        
        def extract_java_refs(node: tree_sitter.Node):
            # Extract import statements
            if node.type == 'import_declaration':
                source = node.child_by_field_name('source')
                if source:
                    package_name = source.text.decode('utf-8') if hasattr(source.text, 'decode') else str(source.text)
                    references.append(ScopedReference(
                        from_entity=file_path_str,
                        to_entity=package_name,
                        reference_type='imports',
                        file_path=file_path_str,
                        line_number=node.start_point[0] + 1
                    ))
            
            # Extract method calls
            elif node.type == 'method_invocation':
                name_node = node.child_by_field_name('name')
                if name_node:
                    method_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    enclosing = self._find_enclosing_entity(node.start_point[0] + 1, entities)
                    references.append(ScopedReference(
                        from_entity=enclosing.name if enclosing else file_path_str,
                        to_entity=method_name,
                        reference_type='calls',
                        file_path=file_path_str,
                        line_number=node.start_point[0] + 1,
                        context={'expected_type': 'function'}
                    ))
            
            for child in node.children:
                extract_java_refs(child)
        
        extract_java_refs(ast.root_node)
        return references
