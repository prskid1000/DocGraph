"""SCSS entity extractor."""
from pathlib import Path
from typing import List
import re

from ..base import BaseEntityExtractor
from ...parsers.base import CodeEntity


class SCSSEntityExtractor(BaseEntityExtractor):
    """Extracts entities from SCSS (selectors, mixins, variables)."""
    
    def extract_entities(self, ast: str, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract SCSS entities."""
        entities = []
        file_path_str = str(file_path)
        lines = source_code.split('\n')
        
        # Extract selectors (classes, IDs, elements)
        selector_pattern = r'^([.#]?[a-zA-Z_-][\w-]*)\s*[,\{]'
        for i, line in enumerate(lines, 1):
            match = re.match(selector_pattern, line.strip())
            if match:
                selector = match.group(1)
                entities.append(CodeEntity(
                    name=selector,
                    entity_type='class',  # Treat selector as class-like
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=0,
                    end_column=len(selector)
                ))
        
        # Extract variables ($variable)
        variable_pattern = r'\$([a-zA-Z_-][\w-]*)\s*:'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(variable_pattern, line):
                var_name = match.group(1)
                entities.append(CodeEntity(
                    name=var_name,
                    entity_type='variable',
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=match.start(),
                    end_column=match.end()
                ))
        
        # Extract mixins (@mixin)
        mixin_pattern = r'@mixin\s+([a-zA-Z_-][\w-]*)'
        for i, line in enumerate(lines, 1):
            match = re.search(mixin_pattern, line)
            if match:
                mixin_name = match.group(1)
                entities.append(CodeEntity(
                    name=mixin_name,
                    entity_type='function',
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=match.start(),
                    end_column=match.end()
                ))
        
        return entities
