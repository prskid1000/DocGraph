"""Tree-sitter based parser implementation."""
import tree_sitter
from tree_sitter import Language, Parser
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib
import sys

from .base import BaseParser, CodeEntity, Reference


class TreeSitterParser(BaseParser):
    """Tree-sitter based parser for multiple languages."""
    
    def __init__(self, language: str, language_module: Optional[str] = None):
        """Initialize tree-sitter parser.
        
        Args:
            language: Language name (e.g., 'python', 'javascript').
            language_module: Optional module name for tree-sitter language.
        """
        super().__init__(language)
        self.language_module = language_module or f"tree_sitter_{language}"
        self.language_obj = self._load_language()
        # Use modern tree-sitter 0.20+ API with Parser(language=...)
        self.parser = Parser(self.language_obj)
    
    def _load_language(self) -> Language:
        """Load tree-sitter language module.
        
        Returns:
            Tree-sitter Language object.
            
        Raises:
            ImportError: If language module cannot be imported.
        """
        try:
            module = importlib.import_module(self.language_module)
            
            # Try multiple ways to get the language object from the module
            # 1. Try function like language_typescript() 
            func_name = f"language_{self.language}"
            if hasattr(module, func_name):
                capsule_or_lang = getattr(module, func_name)()
                # Wrap in Language() if it's a capsule (tree-sitter 0.20+)
                return Language(capsule_or_lang)
            
            # 2. Try language_* with alternative names (e.g., javascript, python)
            for attr_name in dir(module):
                if attr_name.startswith('language_'):
                    func = getattr(module, attr_name)
                    if callable(func):
                        try:
                            capsule_or_lang = func()
                            return Language(capsule_or_lang)
                        except:
                            continue
            
            # 3. Try LANGUAGE constant (older API)
            if hasattr(module, 'LANGUAGE'):
                lang_obj = module.LANGUAGE
                # If it's a capsule, wrap it
                if 'capsule' in str(type(lang_obj)):
                    return Language(lang_obj)
                return lang_obj
            
            # 4. Try language() method
            if hasattr(module, 'language'):
                capsule_or_lang = module.language()
                return Language(capsule_or_lang)
            
            raise ImportError(f"Language module {self.language_module} doesn't expose a language accessor")
        except ImportError as e:
            raise ImportError(
                f"Failed to import tree-sitter language {self.language_module}. "
                f"Install it with: pip install {self.language_module}"
            ) from e
    
    def parse_file(self, file_path: Path) -> tree_sitter.Tree:
        """Parse a source file into a tree-sitter AST.
        
        Args:
            file_path: Path to the source file.
            
        Returns:
            Tree-sitter Tree object.
        """
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        return self.parser.parse(source_code)
    
    def extract_entities(self, ast: tree_sitter.Tree, file_path: Path) -> List[CodeEntity]:
        """Extract code entities from tree-sitter AST.
        
        Args:
            ast: Tree-sitter Tree object.
            file_path: Path to the source file.
            
        Returns:
            List of extracted code entities.
        """
        entities = []
        file_path_str = str(file_path)
        
        # Language-specific extraction logic
        if self.language == 'python':
            entities.extend(self._extract_python_entities(ast.root_node, file_path_str))
        elif self.language in ['javascript', 'typescript']:
            entities.extend(self._extract_js_entities(ast.root_node, file_path_str))
        elif self.language == 'java':
            entities.extend(self._extract_java_entities(ast.root_node, file_path_str))
        elif self.language == 'go':
            entities.extend(self._extract_go_entities(ast.root_node, file_path_str))
        elif self.language == 'rust':
            entities.extend(self._extract_rust_entities(ast.root_node, file_path_str))
        
        return entities
    
    def _extract_python_entities(self, node: tree_sitter.Node, file_path: str, 
                                  parent: Optional[str] = None) -> List[CodeEntity]:
        """Extract entities from Python AST."""
        entities = []
        
        if node.type == 'class_definition':
            name = self._get_node_text(node.child_by_field_name('name'))
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='class',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent,
                    metadata={'decorators': self._extract_decorators(node)}
                ))
                # Recursively extract nested entities
                for child in node.children:
                    entities.extend(self._extract_python_entities(child, file_path, name))
        
        elif node.type == 'function_definition':
            name = self._get_node_text(node.child_by_field_name('name'))
            if name:
                signature = self._extract_function_signature(node)
                docstring = self._extract_docstring(node)
                entities.append(CodeEntity(
                    name=name,
                    entity_type='function',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    signature=signature,
                    docstring=docstring,
                    parent=parent,
                    metadata={'decorators': self._extract_decorators(node)}
                ))
        
        elif node.type == 'assignment' and parent is None:
            # Top-level variable
            target = node.child_by_field_name('left')
            if target:
                name = self._get_node_text(target)
                if name:
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='variable',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent
                    ))
        
        # Recursively process children
        for child in node.children:
            entities.extend(self._extract_python_entities(child, file_path, parent))
        
        return entities
    
    def _extract_js_entities(self, node: tree_sitter.Node, file_path: str,
                             parent: Optional[str] = None) -> List[CodeEntity]:
        """Extract entities from JavaScript/TypeScript AST."""
        entities = []
        
        if node.type in ['class_declaration', 'class_expression']:
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='class',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent
                ))
                # Recursively extract nested entities within class (don't do this at end)
                for child in node.children:
                    entities.extend(self._extract_js_entities(child, file_path, name))
                return entities  # Early return to avoid double-processing
        
        elif node.type in ['function_declaration', 'function_expression', 'arrow_function', 'method_definition']:
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name or node.type == 'arrow_function':
                if not name:
                    name = f"anonymous_{node.start_point[0]}"
                signature = self._extract_js_function_signature(node)
                entities.append(CodeEntity(
                    name=name,
                    entity_type='function',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    signature=signature,
                    parent=parent
                ))
            # Don't recurse into function bodies for now - we only extract top-level and class methods
            return entities
        
        elif node.type == 'variable_declaration' and parent is None:
            for declarator in node.children:
                if declarator.type == 'variable_declarator':
                    name_node = declarator.child_by_field_name('name')
                    name = self._get_node_text(name_node) if name_node else None
                    if name:
                        entities.append(CodeEntity(
                            name=name,
                            entity_type='variable',
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            start_column=node.start_point[1],
                            end_column=node.end_point[1],
                            parent=parent
                        ))
        
        # Recurse through children for top-level extraction
        for child in node.children:
            entities.extend(self._extract_js_entities(child, file_path, parent))
        
        return entities
    
    def _extract_java_entities(self, node: tree_sitter.Node, file_path: str,
                               parent: Optional[str] = None) -> List[CodeEntity]:
        """Extract entities from Java AST."""
        entities = []
        
        if node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='class',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent
                ))
                for child in node.children:
                    entities.extend(self._extract_java_entities(child, file_path, name))
        
        elif node.type == 'method_declaration':
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                signature = self._extract_java_method_signature(node)
                entities.append(CodeEntity(
                    name=name,
                    entity_type='function',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    signature=signature,
                    parent=parent
                ))
        
        for child in node.children:
            entities.extend(self._extract_java_entities(child, file_path, parent))
        
        return entities
    
    def _extract_go_entities(self, node: tree_sitter.Node, file_path: str,
                            parent: Optional[str] = None) -> List[CodeEntity]:
        """Extract entities from Go AST."""
        entities = []
        
        if node.type == 'type_declaration':
            type_spec = node.child_by_field_name('type')
            if type_spec and type_spec.type == 'struct_type':
                name_node = node.child_by_field_name('name')
                name = self._get_node_text(name_node) if name_node else None
                if name:
                    entities.append(CodeEntity(
                        name=name,
                        entity_type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_column=node.start_point[1],
                        end_column=node.end_point[1],
                        parent=parent
                    ))
        
        elif node.type == 'method_declaration' or node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='function',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent
                ))
        
        for child in node.children:
            entities.extend(self._extract_go_entities(child, file_path, parent))
        
        return entities
    
    def _extract_rust_entities(self, node: tree_sitter.Node, file_path: str,
                               parent: Optional[str] = None) -> List[CodeEntity]:
        """Extract entities from Rust AST."""
        entities = []
        
        if node.type == 'struct_item':
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='class',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent
                ))
        
        elif node.type == 'function_item':
            name_node = node.child_by_field_name('name')
            name = self._get_node_text(name_node) if name_node else None
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type='function',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    parent=parent
                ))
        
        for child in node.children:
            entities.extend(self._extract_rust_entities(child, file_path, parent))
        
        return entities
    
    def extract_references(self, ast: tree_sitter.Tree, file_path: Path) -> List[Reference]:
        """Extract references from tree-sitter AST.
        
        Args:
            ast: Tree-sitter Tree object.
            file_path: Path to the source file.
            
        Returns:
            List of extracted references.
        """
        references = []
        file_path_str = str(file_path)
        
        # Language-specific reference extraction
        if self.language == 'python':
            references.extend(self._extract_python_references(ast.root_node, file_path_str))
        elif self.language in ['javascript', 'typescript']:
            references.extend(self._extract_js_references(ast.root_node, file_path_str))
        # Add other languages as needed
        
        return references
    
    def _extract_python_references(self, node: tree_sitter.Node, file_path: str) -> List[Reference]:
        """Extract references from Python AST."""
        references = []
        
        # Extract import statements
        if node.type == 'import_statement' or node.type == 'import_from_statement':
            module_name = self._extract_import_name(node)
            if module_name:
                references.append(Reference(
                    from_entity=file_path,
                    to_entity=module_name,
                    reference_type='imports',
                    file_path=file_path,
                    line_number=node.start_point[0] + 1
                ))
        
        # Extract function calls
        if node.type == 'call':
            function_node = node.child_by_field_name('function')
            if function_node:
                func_name = self._get_node_text(function_node)
                if func_name:
                    references.append(Reference(
                        from_entity=file_path,
                        to_entity=func_name,
                        reference_type='calls',
                        file_path=file_path,
                        line_number=node.start_point[0] + 1
                    ))
        
        for child in node.children:
            references.extend(self._extract_python_references(child, file_path))
        
        return references
    
    def _extract_js_references(self, node: tree_sitter.Node, file_path: str) -> List[Reference]:
        """Extract references from JavaScript/TypeScript AST."""
        references = []
        
        # Extract import statements
        if node.type in ['import_statement', 'import_declaration']:
            source = node.child_by_field_name('source')
            if source:
                module_name = self._get_node_text(source).strip('"\'')
                references.append(Reference(
                    from_entity=file_path,
                    to_entity=module_name,
                    reference_type='imports',
                    file_path=file_path,
                    line_number=node.start_point[0] + 1
                ))
        
        # Extract function calls
        if node.type == 'call_expression':
            function_node = node.child_by_field_name('function')
            if function_node:
                func_name = self._get_node_text(function_node)
                if func_name:
                    references.append(Reference(
                        from_entity=file_path,
                        to_entity=func_name,
                        reference_type='calls',
                        file_path=file_path,
                        line_number=node.start_point[0] + 1
                    ))
        
        for child in node.children:
            references.extend(self._extract_js_references(child, file_path))
        
        return references
    
    # Helper methods
    def _get_node_text(self, node: tree_sitter.Node) -> Optional[str]:
        """Get text content of a node."""
        if node is None:
            return None
        return node.text.decode('utf-8') if node.text else None
    
    def _extract_decorators(self, node: tree_sitter.Node) -> List[str]:
        """Extract decorator names from a node."""
        decorators = []
        for child in node.children:
            if child.type == 'decorator':
                decorator_name = self._get_node_text(child)
                if decorator_name:
                    decorators.append(decorator_name.strip('@'))
        return decorators
    
    def _extract_docstring(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract docstring from a function or class node."""
        # Look for string literal as first statement
        body = node.child_by_field_name('body')
        if body and len(body.children) > 0:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0]
                if expr.type == 'string':
                    return self._get_node_text(expr).strip('"""\'\'\'')
        return None
    
    def _extract_function_signature(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract function signature."""
        name = self._get_node_text(node.child_by_field_name('name'))
        params = node.child_by_field_name('parameters')
        param_text = self._get_node_text(params) if params else '()'
        return f"{name}{param_text}"
    
    def _extract_js_function_signature(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract JavaScript function signature."""
        name_node = node.child_by_field_name('name')
        name = self._get_node_text(name_node) if name_node else 'anonymous'
        params = node.child_by_field_name('parameters')
        param_text = self._get_node_text(params) if params else '()'
        return f"{name}{param_text}"
    
    def _extract_java_method_signature(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract Java method signature."""
        name = self._get_node_text(node.child_by_field_name('name'))
        params = node.child_by_field_name('parameters')
        param_text = self._get_node_text(params) if params else '()'
        return_type = self._get_node_text(node.child_by_field_name('type'))
        return f"{return_type or 'void'} {name}{param_text}"
    
    def _extract_import_name(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract import module name."""
        if node.type == 'import_statement':
            module = node.child_by_field_name('module_name')
            return self._get_node_text(module) if module else None
        elif node.type == 'import_from_statement':
            module = node.child_by_field_name('module_name')
            return self._get_node_text(module) if module else None
        return None

