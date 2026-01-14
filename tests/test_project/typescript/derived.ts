// Derived module demonstrating all relationships
import { BaseClass, helperFunction } from './base';  // IMPORTS: File imports from module
import * as fs from 'fs';  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    private age: number;  // DEFINES: Defines property
    
    constructor(name: string, age: number) {  // HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns void (implicit)
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References property
        const localVar = 10;  // DEFINES: Defines variable
        const result = localVar;  // REFERENCES: References variable 'localVar'
    }
    
    process(data: string[]): string {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'string'
        const result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References variable
        return result;  // RETURNS: Returns type 'string'
    }
    
    getInfo(): { name: string; age: number } {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns object type
        return { name: this.name, age: this.age };  // REFERENCES: References properties
    }
}

// Module-level variable
const MODULE_VAR: string = "test";  // DEFINES: Defines variable

// Module-level function
function main(): string {  // HAS_PARAMETER: Function has no parameters, RETURNS: Returns type 'string'
    const instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
    const result = instance.process(["data"]);  // CALLS: Calls method
    const info = JSON.stringify(instance.getInfo());  // CALLS: Calls external function
    return info;  // RETURNS: Returns type 'string'
}

export { DerivedClass, main };  // IMPORTS: Module exports
