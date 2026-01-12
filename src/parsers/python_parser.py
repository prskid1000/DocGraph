"""Python-specific parser using libcst for enhanced analysis."""
from pathlib import Path
from typing import List
import libcst as cst
from libcst.metadata import PositionProvider, ParentNodeProvider

from .base import BaseParser, CodeEntity, Reference


class PythonParser(BaseParser):
    """Python parser using libcst for source code preservation."""
    
    def __init__(self):
        """Initialize Python parser."""
        super().__init__("python")
        self.metadata_wrapper = None
    
    def parse_file(self, file_path: Path) -> cst.Module:
        """Parse a Python file using libcst.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            libcst Module object.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Create metadata wrapper for position and parent information
        self.metadata_wrapper = cst.metadata.MetadataWrapper(
            cst.parse_module(source_code),
            cache={
                PositionProvider: PositionProvider(),
                ParentNodeProvider: ParentNodeProvider()
            }
        )
        
        return self.metadata_wrapper.module
    
    def extract_entities(self, ast: cst.Module, file_path: Path) -> List[CodeEntity]:
        """Extract entities from Python AST.
        
        Args:
            ast: libcst Module object.
            file_path: Path to the source file.
            
        Returns:
            List of extracted code entities.
        """
        entities = []
        file_path_str = str(file_path)
        
        class Visitor(cst.CSTVisitor):
            def __init__(self, parser_instance, file_path):
                self.parser = parser_instance
                self.file_path = file_path
                self.entities = []
            
            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                docstring = self.parser._extract_docstring_libcst(node)
                bases = [self.parser._get_node_code(base) for base in node.bases]
                
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
                        'decorators': [self.parser._get_node_code(d) for d in node.decorators],
                        'bases': bases
                    }
                ))
            
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                signature = self.parser._get_node_code(node)
                docstring = self.parser._extract_docstring_libcst(node)
                
                # Get parent class if any
                parent = None
                parent_metadata = self.parser.metadata_wrapper.resolve(ParentNodeProvider)
                if node in parent_metadata:
                    parent_node = parent_metadata[node]
                    if isinstance(parent_node, cst.ClassDef):
                        parent = parent_node.name.value
                
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
                        'decorators': [self.parser._get_node_code(d) for d in node.decorators],
                        'async': node.asynchronous is not None
                    }
                ))
            
            def visit_Assign(self, node: cst.Assign) -> None:
                # Only top-level assignments
                parent_metadata = self.parser.metadata_wrapper.resolve(ParentNodeProvider)
                if node in parent_metadata:
                    parent = parent_metadata[node]
                    if isinstance(parent, cst.Module):
                        pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                        for target in node.targets:
                            if isinstance(target.target, cst.Name):
                                self.entities.append(CodeEntity(
                                    name=target.target.value,
                                    entity_type='variable',
                                    file_path=self.file_path,
                                    start_line=pos.start.line,
                                    end_line=pos.end.line,
                                    start_column=pos.start.column,
                                    end_column=pos.end.column
                                ))
        
        visitor = Visitor(self, file_path_str)
        ast.visit(visitor)
        return visitor.entities
    
    def extract_references(self, ast: cst.Module, file_path: Path) -> List[Reference]:
        """Extract references from Python AST.
        
        Args:
            ast: libcst Module object.
            file_path: Path to the source file.
            
        Returns:
            List of extracted references.
        """
        references = []
        file_path_str = str(file_path)
        
        class Visitor(cst.CSTVisitor):
            def __init__(self, parser_instance, file_path):
                self.parser = parser_instance
                self.file_path = file_path
                self.references = []
            
            def visit_Import(self, node: cst.Import) -> None:
                pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                for alias in node.names:
                    module_name = alias.name.value
                    self.references.append(Reference(
                        from_entity=self.file_path,
                        to_entity=module_name,
                        reference_type='imports',
                        file_path=self.file_path,
                        line_number=pos.start.line
                    ))
            
            def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
                pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                if node.module:
                    module_name = self.parser._get_node_code(node.module)
                    self.references.append(Reference(
                        from_entity=self.file_path,
                        to_entity=module_name,
                        reference_type='imports',
                        file_path=self.file_path,
                        line_number=pos.start.line
                    ))
            
            def visit_Call(self, node: cst.Call) -> None:
                pos = self.parser.metadata_wrapper.resolve(PositionProvider)[node]
                if isinstance(node.func, cst.Name):
                    func_name = node.func.value
                    self.references.append(Reference(
                        from_entity=self.file_path,
                        to_entity=func_name,
                        reference_type='calls',
                        file_path=self.file_path,
                        line_number=pos.start.line
                    ))
        
        visitor = Visitor(self, file_path_str)
        ast.visit(visitor)
        return visitor.references
    
    def _get_node_code(self, node: cst.CSTNode) -> str:
        """Get source code for a node."""
        return self.metadata_wrapper.module.code_for_node(node)
    
    def _extract_docstring_libcst(self, node: cst.FunctionDef | cst.ClassDef) -> str | None:
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

