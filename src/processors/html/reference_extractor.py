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
        
        # Extract links (href)
        href_pattern = r'href=["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(href_pattern, line):
                href = match.group(1)
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=href,
                    reference_type='imports',  # Treat as import-like
                    file_path=file_path_str,
                    line_number=i
                ))
        
        # Extract script sources
        src_pattern = r'src=["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(src_pattern, line):
                src = match.group(1)
                references.append(ScopedReference(
                    from_entity=file_path_str,
                    to_entity=src,
                    reference_type='imports',
                    file_path=file_path_str,
                    line_number=i
                ))
        
        return references
