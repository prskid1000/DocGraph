"""Base parser interface and abstract classes."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CodeEntity:
    """Represents a code entity (class, function, variable, etc.)."""
    name: str
    entity_type: str  # 'class', 'function', 'variable', 'module', 'parameter', 'type'
    file_path: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class or module
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Reference:
    """Represents a reference to a code entity."""
    from_entity: str  # Entity making the reference
    to_entity: str    # Entity being referenced
    reference_type: str  # 'calls', 'references', 'imports', 'inherits'
    file_path: str
    line_number: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseParser(ABC):
    """Abstract base class for language parsers."""
    
    def __init__(self, language: str):
        """Initialize parser.
        
        Args:
            language: Language name (e.g., 'python', 'javascript').
        """
        self.language = language
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> Any:
        """Parse a source file into an AST.
        
        Args:
            file_path: Path to the source file.
            
        Returns:
            AST representation (language-specific).
        """
        pass
    
    @abstractmethod
    def extract_entities(self, ast: Any, file_path: Path) -> List[CodeEntity]:
        """Extract code entities from AST.
        
        Args:
            ast: Parsed AST.
            file_path: Path to the source file.
            
        Returns:
            List of extracted code entities.
        """
        pass
    
    @abstractmethod
    def extract_references(self, ast: Any, file_path: Path) -> List[Reference]:
        """Extract references from AST.
        
        Args:
            ast: Parsed AST.
            file_path: Path to the source file.
            
        Returns:
            List of extracted references.
        """
        pass
    
    def parse_and_extract(self, file_path: Path) -> tuple[List[CodeEntity], List[Reference]]:
        """Parse file and extract entities and references.
        
        Args:
            file_path: Path to the source file.
            
        Returns:
            Tuple of (entities, references).
        """
        ast = self.parse_file(file_path)
        entities = self.extract_entities(ast, file_path)
        references = self.extract_references(ast, file_path)
        return entities, references


class ParserFactory:
    """Factory for creating language-specific parsers."""
    
    _parsers: Dict[str, type] = {}
    
    @classmethod
    def register_parser(cls, language: str, parser_class: type):
        """Register a parser class for a language.
        
        Args:
            language: Language name.
            parser_class: Parser class implementing BaseParser.
        """
        cls._parsers[language] = parser_class
    
    @classmethod
    def create_parser(cls, language: str) -> BaseParser:
        """Create a parser instance for a language.
        
        Args:
            language: Language name.
            
        Returns:
            Parser instance.
            
        Raises:
            ValueError: If language is not supported.
        """
        if language not in cls._parsers:
            raise ValueError(f"Unsupported language: {language}")
        return cls._parsers[language]()
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language names.
        """
        return list(cls._parsers.keys())

