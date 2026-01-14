package com.test;

// DEFINES: File defines entities

public class BaseClass {
    protected String name;  // DEFINES: Defines field
    
    public BaseClass(String name) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References field
    }
    
    public String getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return this.name;  // REFERENCES: References field
    }
}

// Base class for testing
package com.test;

// DEFINES: File defines entities

public class BaseClass {
    protected String name;  // DEFINES: Defines field
    
    public BaseClass(String name) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References field
    }
    
    public String getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return this.name;  // REFERENCES: References field
    }
}

// Add a function that calls another function (to ensure CALLS relationship)
class Util {
    public static String callHelper(BaseClass base) {
        return base.getName();  // CALLS: Calls getName, REFERENCES: base
    }
}

// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass extends BaseClass {
    public SubClass(String name) {
        super(name);  // CALLS: Calls parent constructor
    }
}
