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
        
        # Extract JavaScript code from <script> tags for CALLS and REFERENCES
        in_script = False
        script_lines = []
        script_start_line = 0
        script_block_entity = None
        for i, line in enumerate(lines, 1):
            if re.search(r'<script[^>]*>', line):
                in_script = True
                script_start_line = i
                script_lines = []
                # Find the script_block entity for this script tag
                script_block_entity = self._find_enclosing_entity(i, entities)
            elif re.search(r'</script>', line):
                if in_script and script_lines:
                    # Process JavaScript code in script tag
                    script_code = '\n'.join(script_lines)
                    # Use script_block entity or file path as from_entity
                    from_entity_name = script_block_entity.name if script_block_entity else file_path_str
                    
                    # Extract function calls (CALLS)
                    call_pattern = r'(\w+)\s*\([^)]*\)'
                    for match in re.finditer(call_pattern, script_code):
                        func_name = match.group(1)
                        if func_name not in ['if', 'for', 'while', 'switch', 'return', 'new', 'typeof', 'instanceof']:
                            # Robust line number calculation
                            rel_line = match.string[:match.start()].count('\n')
                            try:
                                line_offset = script_lines.index(script_lines[rel_line])
                            except (ValueError, IndexError):
                                line_offset = rel_line if rel_line < len(script_lines) else 0
                            references.append(ScopedReference(
                                from_entity=from_entity_name,
                                to_entity=func_name,
                                reference_type='calls',
                                file_path=file_path_str,
                                line_number=script_start_line + line_offset
                            ))
                    # Extract variable references (REFERENCES) - but skip function definitions
                    var_ref_pattern = r'\b([a-zA-Z_$][\w$]*)\b'
                    for match in re.finditer(var_ref_pattern, script_code):
                        var_name = match.group(1)
                        # Skip keywords and function definitions
                        if var_name not in ['const', 'let', 'var', 'function', 'class', 'if', 'for', 'while', 'return', 'this', 'true', 'false', 'null', 'undefined', 'document', 'console', 'window']:
                            # Check if it's a function definition
                            text_before = script_code[:match.start()].rstrip()
                            if text_before.endswith('function') or text_before.endswith('const') or text_before.endswith('let') or text_before.endswith('var'):
                                continue
                            rel_line = match.string[:match.start()].count('\n')
                            try:
                                line_offset = script_lines.index(script_lines[rel_line])
                            except (ValueError, IndexError):
                                line_offset = rel_line if rel_line < len(script_lines) else 0
                            references.append(ScopedReference(
                                from_entity=from_entity_name,
                                to_entity=var_name,
                                reference_type='references',
                                file_path=file_path_str,
                                line_number=script_start_line + line_offset
                            ))
                in_script = False
                script_lines = []
                script_block_entity = None
            elif in_script:
                script_lines.append(line)
        
        return references
