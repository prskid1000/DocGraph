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
        self.parser = None  # Will be set by processor
    
    def extract_references(
        self, 
        ast: cst.Module, 
        file_path: Path, 
        source_code: str,
        entities: List[CodeEntity]
    ) -> List[ScopedReference]:
        """Extract references from Python AST with scope information."""
        # Parser will be set by base.py process_file() after parsing
        if not hasattr(self, 'parser') or not self.parser:
            return []  # Parser not set yet
        if not hasattr(self.parser, 'metadata_wrapper'):
            return []  # metadata_wrapper not created yet (should not happen if parser is shared correctly)
        
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
                
                # Extract INHERITS relationships from base classes
                pos = self.extractor.parser.get_position(node)
                for base in node.bases:
                    base_code = self.extractor.parser.get_node_code(base)
                    # Extract class name from base (handle qualified names like "module.Class")
                    base_name = base_code.split('.')[-1].strip()
                    
                    self.references.append(ScopedReference(
                        from_entity=node.name.value,
                        to_entity=base_name,
                        reference_type='inherits',
                        file_path=self.file_path,
                        line_number=pos.start.line,
                        scope=None,
                        qualified_name=base_code,
                        context={'base_class': base_code}
                    ))
                
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
                    # Check if it's a wildcard import (import *)
                    if isinstance(node.names, cst.ImportStar):
                        # For wildcard imports, we can't enumerate specific names
                        # Just record the module import
                        self.references.append(ScopedReference(
                            from_entity=self.file_path,
                            to_entity=module_name,
                            reference_type='imports',
                            file_path=self.file_path,
                            line_number=pos.start.line,
                            scope=scope,
                            qualified_name=module_name,
                            context={'module': module_name, 'is_wildcard': True}
                        ))
                    else:
                        # Regular import with specific names
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
            
            def visit_Assign(self, node: cst.Assign) -> None:
                """Extract variable references in assignments."""
                pos = self.extractor.parser.get_position(node)
                scope = self.current_scope[-1] if self.current_scope else None
                enclosing = self.extractor._find_enclosing_entity(pos.start.line, self.entities)
                
                # Extract references to variables on the right side
                def extract_from_value(value_node):
                    if isinstance(value_node, cst.Name):
                        var_name = value_node.value
                        # Check if it's a known variable
                        is_variable = any(e.name == var_name and e.entity_type == 'variable' for e in self.entities)
                        if is_variable:
                            self.references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else self.file_path,
                                to_entity=var_name,
                                reference_type='references',
                                file_path=self.file_path,
                                line_number=pos.start.line,
                                scope=scope,
                                context={'expected_type': 'variable'}
                            ))
                    elif isinstance(value_node, cst.Attribute):
                        # Handle attribute access like self.name
                        if isinstance(value_node.attr, cst.Name):
                            attr_name = value_node.attr.value
                            is_variable = any(e.name == attr_name and e.entity_type == 'variable' for e in self.entities)
                            if is_variable:
                                self.references.append(ScopedReference(
                                    from_entity=enclosing.name if enclosing else self.file_path,
                                    to_entity=attr_name,
                                    reference_type='references',
                                    file_path=self.file_path,
                                    line_number=pos.start.line,
                                    scope=scope,
                                    context={'expected_type': 'variable', 'is_attribute': True}
                                ))
                
                extract_from_value(node.value)
            
            def visit_Return(self, node: cst.Return) -> None:
                """Extract variable references in return statements."""
                if node.value:
                    pos = self.extractor.parser.get_position(node)
                    scope = self.current_scope[-1] if self.current_scope else None
                    enclosing = self.extractor._find_enclosing_entity(pos.start.line, self.entities)
                    
                    # Extract from return value
                    if isinstance(node.value, cst.Name):
                        var_name = node.value.value
                        is_variable = any(e.name == var_name and e.entity_type == 'variable' for e in self.entities)
                        if is_variable:
                            self.references.append(ScopedReference(
                                from_entity=enclosing.name if enclosing else self.file_path,
                                to_entity=var_name,
                                reference_type='references',
                                file_path=self.file_path,
                                line_number=pos.start.line,
                                scope=scope,
                                context={'expected_type': 'variable'}
                            ))
                    elif isinstance(node.value, cst.Dict):
                        # Extract from dict literals like {"name": self.name}
                        for element in node.value.elements:
                            if isinstance(element.value, (cst.Name, cst.Attribute)):
                                if isinstance(element.value, cst.Name):
                                    var_name = element.value.value
                                elif isinstance(element.value, cst.Attribute) and isinstance(element.value.attr, cst.Name):
                                    var_name = element.value.attr.value
                                else:
                                    continue
                                
                                is_variable = any(e.name == var_name and e.entity_type == 'variable' for e in self.entities)
                                if is_variable:
                                    self.references.append(ScopedReference(
                                        from_entity=enclosing.name if enclosing else self.file_path,
                                        to_entity=var_name,
                                        reference_type='references',
                                        file_path=self.file_path,
                                        line_number=pos.start.line,
                                        scope=scope,
                                        context={'expected_type': 'variable'}
                                    ))
        
        visitor = Visitor(self, file_path_str, entities)
        ast.visit(visitor)
        return visitor.references
