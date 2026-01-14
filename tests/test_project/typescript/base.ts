
// DEFINES: File defines entities

class BaseClass {
    protected name: string;  // DEFINES: Defines property
    
    constructor(name: string) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References property
    }
    
    getName(): string {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'string'
        return this.name;  // REFERENCES: References property
    }
}

// Module-level variable
const GLOBAL_VAR: string = "test";  // DEFINES: Defines variable

// Module-level function
function helperFunction(data: string[]): string {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'string'
    return data.length > 0 ? data[0] : "";  // REFERENCES: References variable 'data'
}

export { BaseClass, helperFunction };  // IMPORTS: Module exports

// Base module for testing

// DEFINES: File defines entities

class BaseClass {
    protected name: string;  // DEFINES: Defines property
    
    constructor(name: string) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References property
    }
    
    getName(): string {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'string'
        return this.name;  // REFERENCES: References property
    }
}

// Module-level variable
const GLOBAL_VAR: string = "test";  // DEFINES: Defines variable

// Module-level function
function helperFunction(data: string[]): string {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'string'
    if (data && data.length > 0) {
        return data[0];  // REFERENCES: References variable 'data'
    }
    return "";
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

export { BaseClass, helperFunction, callHelper, SubClass };
