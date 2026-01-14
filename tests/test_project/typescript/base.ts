// Base module for testing - no library functions used

// DEFINES: File defines entities

class BaseClass {
    protected name: string;  // DEFINES: Defines property
    protected value: number;  // DEFINES: Defines property
    
    constructor(name: string) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References property 'name'
        this.value = 0;  // REFERENCES: References property 'value'
    }
    
    getName(): string {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'string'
        return this.name;  // REFERENCES: References property 'this.name'
    }
    
    setValue(value: number): void {  // HAS_PARAMETER: Function has parameters 'this' (implicit) and 'value', RETURNS: Returns type 'void'
        this.value = value;  // REFERENCES: References variables 'this.value' and 'value'
    }
}

// Module-level variable
const GLOBAL_VAR: string = "test";  // DEFINES: Defines variable

// Module-level function - custom implementation without library functions
function helperFunction(data: string[]): string {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'string'
    if (data.length === 0) {  // REFERENCES: References variable 'data'
        return "";  // RETURNS: Returns type 'string'
    }
    return data[0];  // REFERENCES: References variable 'data', RETURNS: Returns type 'string'
}

function calculateSum(a: number, b: number): number {  // HAS_PARAMETER: Function has parameters 'a' and 'b', RETURNS: Returns type 'number'
    const result = a + b;  // REFERENCES: References variables 'a' and 'b'
    return result;  // RETURNS: Returns type 'number'
}

// Add a function that calls another function (to ensure CALLS relationship)
function callHelper(): string {
    return helperFunction([GLOBAL_VAR]);  // CALLS: Calls helperFunction, REFERENCES: GLOBAL_VAR
}

// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass extends BaseClass {
    constructor(name: string) {
        super(name);  // CALLS: Calls parent constructor
    }
}

export { BaseClass, helperFunction, calculateSum, callHelper, SubClass };
