// Derived module demonstrating all relationships - no library functions used
const { BaseClass, helperFunction, calculateSum, callHelper, SubClass } = require('./base');  // IMPORTS: File imports from module

// DEFINES: File defines entities

class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    constructor(name, age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age'
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References variable 'age'
        const localVar = 10;  // DEFINES: Defines variable
        const result = localVar;  // REFERENCES: References variable 'localVar'
        this.value = calculateSum(localVar, 5);  // CALLS: Calls function, REFERENCES: References variables
    }
    
    process(data) {  // HAS_PARAMETER: Function has parameter 'data'
        const result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References variable 'result'
        return result;  // RETURNS: Returns value
    }
    
    getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        const info = { name: this.name, age: this.age };  // REFERENCES: References properties 'this.name' and 'this.age'
        return info;  // RETURNS: Returns object
    }
    
    compute(x, y) {  // HAS_PARAMETER: Function has parameters 'x' and 'y'
        return calculateSum(x, y);  // CALLS: Calls function, RETURNS: Returns value
    }
}

// Module-level variable
const MODULE_VAR = "test";  // DEFINES: Defines variable

// Module-level function
function main() {  // HAS_PARAMETER: Function has no parameters
    const instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
    const result = instance.process(["data"]);  // CALLS: Calls method
    const info = instance.getInfo();  // CALLS: Calls method
    const computed = instance.compute(10, 20);  // CALLS: Calls method
    callHelper();  // CALLS: Calls imported function
    return result;  // RETURNS: Returns value
}

// Add a function that demonstrates REFERENCES and CALLS
function useCustomFunctions() {
    const base = new BaseClass("test");  // CALLS: Calls constructor, REFERENCES: BaseClass
    const name = base.getName();  // CALLS: Calls method, REFERENCES: References variable 'base'
    return name;  // RETURNS: Returns value
}

module.exports = { DerivedClass, main, useCustomFunctions };
