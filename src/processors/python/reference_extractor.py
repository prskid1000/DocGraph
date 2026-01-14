"""Python reference extractor with scoping."""
from pathlib import Path
from typing import List
import libcst as cst

from ..base import BaseReferenceExtractor, ScopedReference
from ...parsers.base import CodeEntity
from .parser import PythonParser


class PythonReferenceExtractor(BaseReferenceExtractor):
    """Extracts references from Python code with proper scoping."""
    
    def __init__(self):
        """Initialize extractor."""
        self.parser = PythonParser()
    
    def extract_references(
        self, 
        ast: cst.Module, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract references from Python AST with scope information."""
        references = []
        file_path_str = str(file_path)
        
        class Visitor(cst.CSTVisitor):
            def __init__(self, extractor_instance, file_path, entities):
                self.extractor = extractor_instance
                self.file_path = file_path
                self.entities = entities
                self.references = []
                self.current_scope = []  # Stack of enclosing entities
            
            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                self.current_scope.append(node.name.value)
                # Continue visiting children
                return True
            
            def leave_ClassDef(self, node: cst.ClassDef) -> None:
                if self.current_scope and self.current_scope[-1] == node.name.value:
                    self.current_scope.pop()
            
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                self.current_scope.append(node.name.value)
                return True
            
            def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
                if self.current_scope and self.current_scope[-1] == node.name.value:
                    self.current_scope.pop()
            
            def visit_Import(self, node: cst.Import) -> None:
                pos = self.extractor.parser.get_position(node)
                scope = self.current_scope[-1] if self.current_scope else None
                
                for alias in node.names:
                    module_name = alias.name.value
                    qualified = f"{module_name}.{alias.evaluated_name}" if alias.evaluated_name != alias.name.value else module_name
                    
                    self.references.append(ScopedReference(
                        from_entity=self.file_path,
                        to_entity=module_name,
                        reference_type='imports',
                        file_path=self.file_path,
                        line_number=pos.start.line,
                        scope=scope,
                        qualified_name=qualified,
                        context={'alias': alias.evaluated_name}
                    ))
            
            def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
                pos = self.extractor.parser.get_position(node)
                scope = self.current_scope[-1] if self.current_scope else None
                
                if node.module:
                    module_name = self.extractor.parser.get_node_code(node.module)
                    
                    # Handle imported names
                    for alias in node.names:
                        imported_name = alias.evaluated_name if alias.evaluated_name else alias.name.value
                        qualified = f"{module_name}.{imported_name}"
                        
                        self.references.append(ScopedReference(
                            from_entity=self.file_path,
                            to_entity=imported_name,
                            reference_type='imports',
                            file_path=self.file_path,
                            line_number=pos.start.line,
                            scope=scope,
                            qualified_name=qualified,
                            context={'module': module_name, 'alias': alias.evaluated_name}
                        ))
            
            def visit_Call(self, node: cst.Call) -> None:
                pos = self.extractor.parser.get_position(node)
                scope = self.current_scope[-1] if self.current_scope else None
                
                if isinstance(node.func, cst.Name):
                    func_name = node.func.value
                    enclosing = self.extractor._find_enclosing_entity(pos.start.line, self.entities)
                    
                    self.references.append(ScopedReference(
                        from_entity=enclosing.name if enclosing else self.file_path,
                        to_entity=func_name,
                        reference_type='calls',
                        file_path=self.file_path,
                        line_number=pos.start.line,
                        scope=scope,
                        context={'expected_type': 'function'}
                    ))
                elif isinstance(node.func, cst.Attribute):
                    # Handle method calls like obj.method()
                    attr_name = node.func.attr.value
                    self.references.append(ScopedReference(
                        from_entity=enclosing.name if (enclosing := self.extractor._find_enclosing_entity(pos.start.line, self.entities)) else self.file_path,
                        to_entity=attr_name,
                        reference_type='calls',
                        file_path=self.file_path,
                        line_number=pos.start.line,
                        scope=scope,
                        context={'expected_type': 'function', 'is_method': True}
                    ))
        
        visitor = Visitor(self, file_path_str, entities)
        ast.visit(visitor)
        return visitor.references
