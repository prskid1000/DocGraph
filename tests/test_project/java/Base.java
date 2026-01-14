package com.test;

import com.test.Utils;  // IMPORTS: File imports module

// DEFINES: File defines entities

public class BaseClass {
    protected String name;  // DEFINES: Defines field
    protected int value;  // DEFINES: Defines field
    
    public BaseClass(String name) {  // HAS_PARAMETER: Function has parameter 'name', RETURNS: Returns void (implicit)
        this.name = name;  // REFERENCES: References field 'name'
        this.value = 0;  // REFERENCES: References field 'value'
    }
    
    public String getName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return this.name;  // REFERENCES: References field 'this.name'
    }
    
    public void setValue(int value) {  // HAS_PARAMETER: Function has parameters 'this' (implicit) and 'value', RETURNS: Returns type 'void'
        this.value = value;  // REFERENCES: References fields 'this.value' and variable 'value'
    }
    
    public String getFormattedName() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return Utils.formatString(this.name);  // CALLS: Calls Utils method, REFERENCES: References field 'this.name', RETURNS: Returns type 'String'
    }
}


// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass extends BaseClass {
    public SubClass(String name) {
        super(name);  // CALLS: Calls parent constructor
    }
}
