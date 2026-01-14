"""Python entity extractor."""
from pathlib import Path
from typing import List
import libcst as cst

from ..base import BaseEntityExtractor
from ...parsers.base import CodeEntity
from .parser import PythonParser


class PythonEntityExtractor(BaseEntityExtractor):
    """Extracts entities from Python code."""
    
    def __init__(self):
        """Initialize extractor."""
        self.parser = None  # Will be set by processor after parsing
    
    def extract_entities(self, ast: cst.Module, file_path: Path, source_code: str) -> List[CodeEntity]:
        """Extract entities from Python AST."""
        # Parser will be set by base.py process_file() after parsing
        if not hasattr(self, 'parser') or not self.parser:
            return []  # Parser not set yet
        if not hasattr(self.parser, 'metadata_wrapper'):
            return []  # metadata_wrapper not created yet (should not happen if parser is shared correctly)
        
        entities = []
        file_path_str = str(file_path)
        
        class Visitor(cst.CSTVisitor):
            def __init__(self, extractor_instance, file_path):
                self.extractor = extractor_instance
                self.file_path = file_path
                self.entities = []
            
            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                pos = self.extractor.parser.get_position(node)
                docstring = self.extractor._extract_docstring(node)
                bases = [self.extractor.parser.get_node_code(base) for base in node.bases]
                
                self.entities.append(CodeEntity(
                    name=node.name.value,
                    entity_type='class',
                    file_path=self.file_path,
                    start_line=pos.start.line,
                    end_line=pos.end.line,
                    start_column=pos.start.column,
                    end_column=pos.end.column,
                    docstring=docstring,
                    metadata={
                        'decorators': [self.extractor.parser.get_node_code(d) for d in node.decorators],
                        'bases': bases
                    }
                ))
            
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                pos = self.extractor.parser.get_position(node)
                signature = self.extractor.parser.get_node_code(node)
                docstring = self.extractor._extract_docstring(node)
                
                # Get parent class if any
                parent = None
                parent_node = self.extractor.parser.get_parent(node)
                if parent_node and isinstance(parent_node, cst.ClassDef):
                    parent = parent_node.name.value
                
                # Extract parameters for HAS_PARAMETER relationships
                parameters = []
                if node.params:
                    for param_idx, param in enumerate(node.params.params):
                        param_name = param.name.value
                        param_type = None
                        param_default = None
                        
                        # Extract type annotation (handle safely)
                        if param.annotation:
                            try:
                                param_type = self.extractor.parser.get_node_code(param.annotation)
                            except Exception:
                                # If code generation fails, try to get the annotation name
                                if hasattr(param.annotation, 'annotation'):
                                    try:
                                        param_type = self.extractor.parser.get_node_code(param.annotation.annotation)
                                    except Exception:
                                        pass
                        
                        # Extract default value (handle safely)
                        if param.default:
                            try:
                                param_default = self.extractor.parser.get_node_code(param.default)
                            except Exception:
                                pass
                        
                        parameters.append({
                            'name': param_name,
                            'type': param_type,
                            'default': param_default,
                            'position': param_idx
                        })
                
                # Extract return type for RETURNS relationship
                return_type = None
                if node.returns:
                    try:
                        return_type = self.extractor.parser.get_node_code(node.returns)
                    except Exception:
                        # If code generation fails, try to extract type name
                        if hasattr(node.returns, 'annotation'):
                            try:
                                return_type = self.extractor.parser.get_node_code(node.returns.annotation)
                            except Exception:
                                pass
                
                self.entities.append(CodeEntity(
                    name=node.name.value,
                    entity_type='function',
                    file_path=self.file_path,
                    start_line=pos.start.line,
                    end_line=pos.end.line,
                    start_column=pos.start.column,
                    end_column=pos.end.column,
                    signature=signature.split('\n')[0],  # First line only
                    docstring=docstring,
                    parent=parent,
                    metadata={
                        'decorators': [self.extractor.parser.get_node_code(d) for d in node.decorators],
                        'async': node.asynchronous is not None,
                        'parameters': parameters,  # For HAS_PARAMETER relationships
                        'return_type': return_type  # For RETURNS relationships
                    }
                ))
            
            def visit_Assign(self, node: cst.Assign) -> None:
                # Extract all variable assignments (not just top-level)
                pos = self.extractor.parser.get_position(node)
                parent_node = self.extractor.parser.get_parent(node)
                
                # Get parent function/class for context
                parent_entity = None
                if parent_node:
                    # Walk up to find enclosing function or class
                    current = parent_node
                    while current:
                        if isinstance(current, cst.FunctionDef):
                            parent_entity = current.name.value
                            break
                        elif isinstance(current, cst.ClassDef):
                            parent_entity = current.name.value
                            break
                        current = self.extractor.parser.get_parent(current)
                
                for target in node.targets:
                    if isinstance(target.target, cst.Name):
                        self.entities.append(CodeEntity(
                            name=target.target.value,
                            entity_type='variable',
                            file_path=self.file_path,
                            start_line=pos.start.line,
                            end_line=pos.end.line,
                            start_column=pos.start.column,
                            end_column=pos.end.column,
                            parent=parent_entity
                        ))
        
        visitor = Visitor(self, file_path_str)
        ast.visit(visitor)
        return visitor.entities
    
    def _extract_docstring(self, node: cst.FunctionDef | cst.ClassDef) -> str | None:
        """Extract docstring from a function or class."""
        body = node.body
        if isinstance(body, cst.SimpleStatementSuite):
            statements = body.body
            if statements and isinstance(statements[0], cst.Expr):
                expr = statements[0].value
                if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
                    return expr.value.strip('"""\'\'\'')
        elif isinstance(body, cst.IndentedBlock):
            statements = body.body
            if statements and isinstance(statements[0], cst.SimpleStatementLine):
                stmt = statements[0].body[0]
                if isinstance(stmt, cst.Expr):
                    expr = stmt.value
                    if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
                        return expr.value.strip('"""\'\'\'')
        return None
