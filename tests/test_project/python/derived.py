"""Derived module demonstrating all relationships - no library functions used."""
from .base import BaseClass, helper_function, calculate_sum  # IMPORTS: File imports from module

# Create a custom module for testing imports
# This will be in a separate file
CUSTOM_CONSTANT = 42  # DEFINES: Defines variable

# DEFINES: File defines entities

class DerivedClass(BaseClass):  # INHERITS: Class inherits from BaseClass
    """Derived class."""
    
    def __init__(self, name: str, age: int):  # HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns None (implicit)
        super().__init__(name)  # CALLS: Calls parent constructor
        self.age = age  # REFERENCES: References variable 'age'
        my_var = 10  # DEFINES: Defines variable
        result = my_var  # REFERENCES: References variable 'my_var'
        self.value = calculate_sum(my_var, 5)  # CALLS: Calls function, REFERENCES: References variables
    
    def process(self, data: list) -> str:  # HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'str'
        """Process data."""
        result = helper_function(data)  # CALLS: Function calls another function
        self.name = result  # REFERENCES: References variable 'result'
        return result  # RETURNS: Returns type 'str'
    
    def get_info(self) -> dict:  # HAS_PARAMETER: Function has parameter 'self', RETURNS: Returns type 'dict'
        """Get info as dict."""
        info = {"name": self.name, "age": self.age}  # REFERENCES: References variables 'self.name' and 'self.age'
        return info  # RETURNS: Returns type 'dict'
    
    def compute(self, x: int, y: int) -> int:  # HAS_PARAMETER: Function has parameters 'x' and 'y', RETURNS: Returns type 'int'
        """Compute using helper."""
        return calculate_sum(x, y)  # CALLS: Calls function, RETURNS: Returns type 'int'

# Module-level variable
MODULE_VAR = "test"  # DEFINES: Defines variable

# Module-level function
def main() -> str:  # HAS_PARAMETER: Function has no parameters, RETURNS: Returns type 'str'
    """Main function."""
    instance = DerivedClass("test", 25)  # CALLS: Calls constructor
    result = instance.process(["data"])  # CALLS: Calls method
    info_dict = instance.get_info()  # CALLS: Calls method
    computed = instance.compute(10, 20)  # CALLS: Calls method
    return result  # RETURNS: Returns type 'str'
