// Derived module demonstrating all relationships - no library functions used
import { BaseClass, helperFunction, calculateSum, callHelper, SubClass } from './base';  // IMPORTS: File imports from module

// DEFINES: File defines entities

class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    private age: number;  // DEFINES: Defines property
    
    constructor(name: string, age: number) {  // HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns void (implicit)
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References property 'age'
        const localVar = 10;  // DEFINES: Defines variable
        const result = localVar;  // REFERENCES: References variable 'localVar'
        this.value = calculateSum(localVar, 5);  // CALLS: Calls function, REFERENCES: References variables
    }
    
    process(data: string[]): string {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'string'
        const result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References variable 'result'
        return result;  // RETURNS: Returns type 'string'
    }
    
    getInfo(): { name: string; age: number } {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns object type
        const info = { name: this.name, age: this.age };  // REFERENCES: References properties 'this.name' and 'this.age'
        return info;  // RETURNS: Returns object type
    }
    
    compute(x: number, y: number): number {  // HAS_PARAMETER: Function has parameters 'x' and 'y', RETURNS: Returns type 'number'
        return calculateSum(x, y);  // CALLS: Calls function, RETURNS: Returns type 'number'
    }
}

// Module-level variable
const MODULE_VAR: string = "test";  // DEFINES: Defines variable

// Module-level function
function main(): string {  // HAS_PARAMETER: Function has no parameters, RETURNS: Returns type 'string'
    const instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
    const result = instance.process(["data"]);  // CALLS: Calls method
    const info = instance.getInfo();  // CALLS: Calls method
    const computed = instance.compute(10, 20);  // CALLS: Calls method
    callHelper();  // CALLS: Calls imported function
    return result;  // RETURNS: Returns type 'string'
}

// Add a function that demonstrates REFERENCES and CALLS
function useCustomFunctions(): string {
    const base = new BaseClass("test");  // CALLS: Calls constructor, REFERENCES: BaseClass
    const name = base.getName();  // CALLS: Calls method, REFERENCES: References variable 'base'
    return name;  // RETURNS: Returns type 'string'
}

export { DerivedClass, main, useCustomFunctions };
