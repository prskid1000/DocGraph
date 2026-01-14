"""HTML entity extractor."""
from pathlib import Path
from typing import List
import re

from ..base import BaseEntityExtractor
from ...parsers.base import CodeEntity


class HTMLEntityExtractor(BaseEntityExtractor):
    """Extracts entities from HTML (elements, IDs, classes)."""
    
    def extract_entities(self, ast: str, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract HTML entities."""
        entities = []
        file_path_str = str(file_path)
        lines = source_code.split('\n')
        
        # Extract elements with IDs
        id_pattern = r'id=["\']([^"\']+)["\']'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(id_pattern, line):
                id_name = match.group(1)
                col_start = match.start()
                entities.append(CodeEntity(
                    name=id_name,
                    entity_type='variable',  # Treat ID as variable-like
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=col_start,
                    end_column=col_start + len(id_name),
                    metadata={'html_type': 'id'}
                ))
        
        # Extract script and style blocks
        script_pattern = r'<script[^>]*>'
        style_pattern = r'<style[^>]*>'
        for i, line in enumerate(lines, 1):
            if re.search(script_pattern, line):
                entities.append(CodeEntity(
                    name='script_block',
                    entity_type='function',
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=0,
                    end_column=len(line)
                ))
            if re.search(style_pattern, line):
                entities.append(CodeEntity(
                    name='style_block',
                    entity_type='variable',
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=0,
                    end_column=len(line)
                ))
        
        return entities
