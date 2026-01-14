"""SCSS reference extractor."""
from pathlib import Path
from typing import List
import re

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class SCSSReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from SCSS (@import, @use, @forward, @include)."""
    
    def extract_references(
        self, 
        ast: str, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract SCSS references."""
        references = []
        file_path_str = str(file_path)
        lines = source_code.split('\n')
        
        # Extract @import
        import_pattern = r'@import\s+["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            match = re.search(import_pattern, line)
            if match:
                import_path = match.group(1)
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=import_path,
                    reference_type='imports',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        # Extract @use
        use_pattern = r'@use\s+["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            match = re.search(use_pattern, line)
            if match:
                use_path = match.group(1)
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=use_path,
                    reference_type='imports',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        # Extract @include (mixin usage) - CALLS relationship
        include_pattern = r'@include\s+([a-zA-Z_-][\w-]*)'
        for i, line in enumerate(lines, 1):
            match = re.search(include_pattern, line)
            if match:
                mixin_name = match.group(1)
                enclosing = self._find_enclosing_entity(i, entities)
                references.append(ScopedReference(
                    from_entity=enclosing.name if enclosing else file_path_str,
                    to_entity=mixin_name,
                    reference_type='calls',
                    file_path=file_path_str,
                    line_number=i,
                    context={'expected_type': 'function'}
                ))
        
        # Extract variable references ($variable) - REFERENCES relationship
        variable_ref_pattern = r'\$([a-zA-Z_-][\w-]*)'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(variable_ref_pattern, line):
                var_name = match.group(1)
                # Skip if it's a variable definition (has : after it)
                if ':' not in line[:match.start()] or line[:match.start()].count(':') == 0:
                    enclosing = self._find_enclosing_entity(i, entities)
                    references.append(ScopedReference(
                        from_entity=enclosing.name if enclosing else file_path_str,
                        to_entity=var_name,
                        reference_type='references',
                        file_path=file_path_str,
                        line_number=i
                    ))
        
        # Extract @extend (class extension) - REFERENCES relationship
        extend_pattern = r'@extend\s+\.([a-zA-Z_-][\w-]*)'
        for i, line in enumerate(lines, 1):
            match = re.search(extend_pattern, line)
            if match:
                class_name = match.group(1)
                enclosing = self._find_enclosing_entity(i, entities)
                references.append(ScopedReference(
                    from_entity=enclosing.name if enclosing else file_path_str,
                    to_entity=class_name,
                    reference_type='references',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        return references
