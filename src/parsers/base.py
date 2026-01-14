"""Base data structures for code entities and references."""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, field


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
