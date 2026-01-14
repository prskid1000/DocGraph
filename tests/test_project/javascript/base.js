// Base module for testing - no library functions used

// DEFINES: File defines entities

class BaseClass {
    constructor(name) {  // HAS_PARAMETER: Function has parameter 'name'
        this.name = name;  // REFERENCES: References variable 'name'
        this.value = 0;  // REFERENCES: References variable 'value'
    }
    
    getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit)
        return this.name;  // REFERENCES: References property 'this.name'
    }
    
    setValue(value) {  // HAS_PARAMETER: Function has parameters 'this' (implicit) and 'value'
        this.value = value;  // REFERENCES: References variables 'this.value' and 'value'
    }
}

// Module-level variable
const GLOBAL_VAR = "test";  // DEFINES: Defines variable

// Module-level function - custom implementation without library functions
function helperFunction(data) {  // HAS_PARAMETER: Function has parameter 'data'
    if (data.length === 0) {  // REFERENCES: References variable 'data'
        return "";  // RETURNS: Returns value
    }
    return data[0];  // REFERENCES: References variable 'data', RETURNS: Returns value
}

function calculateSum(a, b) {  // HAS_PARAMETER: Function has parameters 'a' and 'b'
    const result = a + b;  // REFERENCES: References variables 'a' and 'b'
    return result;  // RETURNS: Returns value
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

module.exports = { BaseClass, helperFunction, calculateSum, callHelper, SubClass };
