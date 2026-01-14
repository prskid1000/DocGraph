
// DEFINES: File defines entities

class BaseClass {
    constructor(name) {  // HAS_PARAMETER: Function has parameter 'name'
        this.name = name;  // REFERENCES: References variable
    }
    
    getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        return this.name;  // REFERENCES: References property
    }
}

// Module-level variable
const GLOBAL_VAR = "test";  // DEFINES: Defines variable

// Module-level function
function helperFunction(data) {  // HAS_PARAMETER: Function has parameter 'data'
    return data.length > 0 ? data[0] : "";  // REFERENCES: References variable 'data'
}

module.exports = { BaseClass, helperFunction };  // IMPORTS: Module exports

// Base module for testing

// DEFINES: File defines entities

class BaseClass {
    constructor(name) {  // HAS_PARAMETER: Function has parameter 'name'
        this.name = name;  // REFERENCES: References variable
    }
    
    getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        return this.name;  // REFERENCES: References property
    }
}

// Module-level variable
const GLOBAL_VAR = "test";  // DEFINES: Defines variable

// Module-level function
function helperFunction(data) {  // HAS_PARAMETER: Function has parameter 'data'
    if (data && data.length > 0) {
        return data[0];  // REFERENCES: References variable 'data'
    }
    return "";
}

// Add a function that calls another function (to ensure CALLS relationship)
function callHelper() {
    return helperFunction([GLOBAL_VAR]);  // CALLS: Calls helperFunction, REFERENCES: GLOBAL_VAR
}

// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass extends BaseClass {
    constructor(name) {
        super(name);  // CALLS: Calls parent constructor
    }
}

module.exports = { BaseClass, helperFunction, callHelper, SubClass };
