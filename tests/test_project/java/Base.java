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
