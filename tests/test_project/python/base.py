"""Base module for testing - no library functions used."""

# DEFINES: File defines entities

class BaseClass:
    """Base class for inheritance."""
    def __init__(self, name: str):  # HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns None (implicit)
        self.name = name  # REFERENCES: References variable 'name'
        self.value = 0  # REFERENCES: References variable 'value'
    
    def get_name(self) -> str:  # HAS_PARAMETER: Function has parameter 'self', RETURNS: Returns type 'str'
        """Get name."""
        return self.name  # REFERENCES: References variable 'self.name'
    
    def set_value(self, value: int) -> None:  # HAS_PARAMETER: Function has parameters 'self' and 'value', RETURNS: Returns type 'None'
        """Set value."""
        self.value = value  # REFERENCES: References variables 'self.value' and 'value'

# Module-level variable
GLOBAL_VAR = "test"  # DEFINES: Defines variable

# Module-level function
def helper_function(data: list) -> str:  # HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'str'
    """Helper function - custom implementation without library functions."""
    if len(data) == 0:  # REFERENCES: References variable 'data'
        return ""  # RETURNS: Returns string
    return data[0]  # REFERENCES: References variable 'data', RETURNS: Returns string

def calculate_sum(a: int, b: int) -> int:  # HAS_PARAMETER: Function has parameters 'a' and 'b', RETURNS: Returns type 'int'
    """Calculate sum without using library functions."""
    result = a + b  # REFERENCES: References variables 'a' and 'b'
    return result  # RETURNS: Returns type 'int'
