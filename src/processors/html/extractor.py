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
        
        # Extract script and style blocks, and JavaScript entities
        script_pattern = r'<script[^>]*>'
        style_pattern = r'<style[^>]*>'
        in_script = False
        script_lines = []
        script_start_line = 0
        
        for i, line in enumerate(lines, 1):
            if re.search(script_pattern, line):
                in_script = True
                script_start_line = i
                script_lines = []
                entities.append(CodeEntity(
                    name='script_block',
                    entity_type='function',
                    file_path=file_path_str,
                    start_line=i,
                    end_line=i,
                    start_column=0,
                    end_column=len(line)
                ))
            elif re.search(r'</script>', line):
                if in_script and script_lines:
                    # Extract JavaScript functions and variables from script
                    script_code = '\n'.join(script_lines)
                    
                    # Extract function declarations with parameters
                    func_pattern = r'function\s+([a-zA-Z_$][\w$]*)\s*\(([^)]*)\)'
                    for match in re.finditer(func_pattern, script_code):
                        func_name = match.group(1)
                        params_str = match.group(2).strip()
                        rel_line = match.string[:match.start()].count('\n')
                        try:
                            line_offset = script_lines.index(script_lines[rel_line]) if rel_line < len(script_lines) else 0
                        except (ValueError, IndexError):
                            line_offset = rel_line if rel_line < len(script_lines) else 0
                        
                        # Extract parameters
                        parameters = []
                        if params_str:
                            param_names = [p.strip().split('=')[0].strip() for p in params_str.split(',')]
                            parameters = [{'name': p, 'position': i} for i, p in enumerate(param_names) if p]
                        
                        func_entity = CodeEntity(
                            name=func_name,
                            entity_type='function',
                            file_path=file_path_str,
                            start_line=script_start_line + line_offset,
                            end_line=script_start_line + line_offset,
                            start_column=0,
                            end_column=len(func_name),
                            metadata={'parameters': parameters} if parameters else {}
                        )
                        entities.append(func_entity)
                        
                        # Also create parameter entities
                        for param_data in parameters:
                            entities.append(CodeEntity(
                                name=param_data['name'],
                                entity_type='parameter',
                                file_path=file_path_str,
                                start_line=script_start_line + line_offset,
                                end_line=script_start_line + line_offset,
                                start_column=0,
                                end_column=len(param_data['name']),
                                metadata={'position': param_data['position']}
                            ))
                    
                    # Extract variable declarations (const, let, var)
                    var_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][\w$]*)'
                    for match in re.finditer(var_pattern, script_code):
                        var_name = match.group(1)
                        rel_line = match.string[:match.start()].count('\n')
                        try:
                            line_offset = script_lines.index(script_lines[rel_line]) if rel_line < len(script_lines) else 0
                        except (ValueError, IndexError):
                            line_offset = rel_line if rel_line < len(script_lines) else 0
                        entities.append(CodeEntity(
                            name=var_name,
                            entity_type='variable',
                            file_path=file_path_str,
                            start_line=script_start_line + line_offset,
                            end_line=script_start_line + line_offset,
                            start_column=0,
                            end_column=len(var_name)
                        ))
                
                in_script = False
                script_lines = []
            elif in_script:
                script_lines.append(line)
            
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
