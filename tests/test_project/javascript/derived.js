const { BaseClass, helperFunction } = require('./base');  // IMPORTS: File imports from module
const fs = require('fs');  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    constructor(name, age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age'
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References variable
        const localVar = 10;  // DEFINES: Defines variable
        const result = localVar;  // REFERENCES: References variable 'localVar'
    }
    
    process(data) {  // HAS_PARAMETER: Function has parameter 'data'
        const result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References variable
        return result;  // RETURNS: Returns value (implicit type)
    }
    
    getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        return { name: this.name, age: this.age };  // REFERENCES: References properties, RETURNS: Returns object
    }
}

// Module-level variable
const MODULE_VAR = "test";  // DEFINES: Defines variable

// Module-level function
function main() {  // HAS_PARAMETER: Function has no parameters
    const instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
    const result = instance.process(["data"]);  // CALLS: Calls method
    const info = JSON.stringify(instance.getInfo());  // CALLS: Calls external function
    return info;  // RETURNS: Returns value
}

module.exports = { DerivedClass, main };  // IMPORTS: Module exports

// Derived module demonstrating all relationships
const { BaseClass, helperFunction, callHelper, SubClass } = require('./base');  // IMPORTS: File imports from module
const fs = require('fs');  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    constructor(name, age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age'
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References variable
        const localVar = 10;  // DEFINES: Defines variable
        const result = localVar;  // REFERENCES: References variable 'localVar'
    }
    
    process(data) {  // HAS_PARAMETER: Function has parameter 'data'
        const result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References variable
        return result;  // RETURNS: Returns value (implicit type)
    }
    
    getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        return { name: this.name, age: this.age };  // REFERENCES: References properties, RETURNS: Returns object
    }
}

// Module-level variable
const MODULE_VAR = "test";  // DEFINES: Defines variable

// Module-level function
function main() {  // HAS_PARAMETER: Function has no parameters
    const instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
    const result = instance.process(["data"]);  // CALLS: Calls method
    const info = JSON.stringify(instance.getInfo());  // CALLS: Calls external function
    callHelper();  // CALLS: Calls imported function
    return info;  // RETURNS: Returns value
}

// Add a function that demonstrates REFERENCES and CALLS
function useFs() {
    fs.readFileSync('./somefile.txt');  // CALLS: Calls external function, REFERENCES: fs
}

module.exports = { DerivedClass, main, useFs };
