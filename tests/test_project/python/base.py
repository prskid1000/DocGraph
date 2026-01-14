"""Base module for testing."""
from typing import List, Dict

# DEFINES: File defines entities

class BaseClass:
    """Base class for inheritance."""
    def __init__(self, name: str):  # HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns None (implicit)
        self.name = name  # REFERENCES: References variable
    
    def get_name(self) -> str:  # HAS_PARAMETER: Function has parameter 'self', RETURNS: Returns type 'str'
        """Get name."""
        return self.name  # REFERENCES: References variable

# Module-level variable
GLOBAL_VAR = "test"  # DEFINES: Defines variable

def helper_function(data: List[str]) -> str:  # HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'str'
    """Helper function."""
    return data[0] if data else ""  # REFERENCES: References variable 'data'
