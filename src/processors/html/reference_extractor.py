"""HTML reference extractor."""
from pathlib import Path
from typing import List
import re

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity


class HTMLReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from HTML (links, scripts, stylesheets)."""
    
    def extract_references(
        self, 
        ast: str, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract HTML references."""
        references = []
        file_path_str = str(file_path)
        lines = source_code.split('\n')
        
        # Extract links (href) - create both IMPORTS and CONTAINS
        href_pattern = r'href=["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(href_pattern, line):
                href = match.group(1)
                # Check if it's a local file reference (not external URL)
                if not href.startswith(('http://', 'https://', '//', 'data:', 'mailto:', '#')):
                    # Create CONTAINS relationship for file references
                    references.append(ScopedReference(
                        from_entity=file_path_str,
                        to_entity=href,
                        reference_type='contains',
                        file_path=file_path_str,
                        line_number=i
                    ))
                # Also create IMPORTS relationship
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=href,
                    reference_type='imports',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        # Extract script sources - create both IMPORTS and CONTAINS
        src_pattern = r'src=["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(src_pattern, line):
                src = match.group(1)
                # Check if it's a local file reference (not external URL)
                if not src.startswith(('http://', 'https://', '//', 'data:')):
                    # Create CONTAINS relationship for file references
                    references.append(ScopedReference(
                        from_entity=file_path_str,
                        to_entity=src,
                        reference_type='contains',
                        file_path=file_path_str,
                        line_number=i
                    ))
                # Also create IMPORTS relationship
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=src,
                    reference_type='imports',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        return references
