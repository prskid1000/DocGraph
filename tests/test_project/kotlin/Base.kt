package com.test

import com.test.Utils  // IMPORTS: File imports module

// DEFINES: File defines entities

open class BaseClass(val name: String) {  // HAS_PARAMETER: Constructor has parameter 'name', DEFINES: Defines property
    protected var value: Int = 0  // DEFINES: Defines property
    
    fun getName(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return name  // REFERENCES: References property 'name'
    }
    
    fun setValue(value: Int): Unit {  // HAS_PARAMETER: Function has parameters 'this' (implicit) and 'value', RETURNS: Returns type 'Unit'
        this.value = value  // REFERENCES: References property 'this.value' and variable 'value'
    }
    
    fun getFormattedName(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return Utils.formatString(this.name)  // CALLS: Calls Utils method, REFERENCES: References property 'this.name', RETURNS: Returns type 'String'
    }
}


// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass(name: String) : BaseClass(name) {  // INHERITS: Class inherits from BaseClass
    init {
        // Constructor body
    }
}
