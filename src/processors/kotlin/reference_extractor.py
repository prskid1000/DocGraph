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
        
        if ast is None or not ast.root_node:
            return references
        
        def extract_kotlin_refs(node: tree_sitter.Node):
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
            
            # Extract INHERITS relationships from class declarations
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                superclass_node = node.child_by_field_name('superclass')
                if name_node and superclass_node:
                    class_name = name_node.text.decode('utf-8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    superclass_name = superclass_node.text.decode('utf-8') if hasattr(superclass_node.text, 'decode') else str(superclass_node.text)
                    # Clean up superclass name
                    superclass_name = superclass_name.replace('extends', '').replace(':', '').strip()
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
                extract_kotlin_refs(child)
        
        extract_kotlin_refs(ast.root_node)
        return references
