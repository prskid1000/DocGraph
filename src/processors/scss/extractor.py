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
        # Need to find the full block range, not just the selector line
        # Also handle nested selectors (e.g., .container .nested)
        selector_pattern = r'^([.#]?[a-zA-Z_-][\w-]*)\s*[,\{]'
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check for nested selector (indented selector inside a block)
            stripped_line = line.strip()
            # Skip if it's just whitespace or a comment
            if not stripped_line or stripped_line.startswith('//'):
                i += 1
                continue
            
            match = re.match(selector_pattern, stripped_line)
            if match:
                selector = match.group(1)
                start_line = i + 1
                # Find the end of the block (matching braces)
                brace_count = 0
                end_line = start_line
                found_open_brace = False
                for j in range(i, len(lines)):
                    current_line = lines[j]
                    for char in current_line:
                        if char == '{':
                            brace_count += 1
                            found_open_brace = True
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and found_open_brace:
                                end_line = j + 1
                                break
                    if brace_count == 0 and found_open_brace:
                        break
                
                # If we didn't find a closing brace, use the start line
                if not found_open_brace:
                    end_line = start_line
                
                entities.append(CodeEntity(
                    name=selector,
                    entity_type='class',  # Treat selector as class-like
                    file_path=file_path_str,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=0,
                    end_column=len(selector)
                ))
                # Don't skip - continue to find nested selectors
            i += 1
        
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
