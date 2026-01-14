"""Derived module demonstrating all relationships."""
from .base import BaseClass, helper_function  # IMPORTS: File imports from module
import json  # IMPORTS: File imports module

# DEFINES: File defines entities

class DerivedClass(BaseClass):  # INHERITS: Class inherits from BaseClass
    """Derived class."""
    
    def __init__(self, name: str, age: int):  # HAS_PARAMETER: Function has parameters 'name' and 'age'
        super().__init__(name)  # CALLS: Calls parent constructor
        self.age = age  # REFERENCES: References variable 'age'
        my_var = 10  # DEFINES: Defines variable
        result = my_var  # REFERENCES: References variable 'my_var'
    
    def process(self, data: List[str]) -> str:  # HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'str'
        """Process data."""
        result = helper_function(data)  # CALLS: Function calls another function
        self.name = result  # REFERENCES: References variable
        return result
    
    def get_info(self) -> dict:
        """Get info as dict."""
        return {"name": self.name, "age": self.age}  # REFERENCES: References variables

# Module-level function
def main():
    """Main function."""
    instance = DerivedClass("test", 25)  # CALLS: Calls constructor
    result = instance.process(["data"])  # CALLS: Calls method
    info = json.dumps(instance.get_info())  # CALLS: Calls external function
    return info
