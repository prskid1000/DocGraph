package com.test

// DEFINES: File defines entities

open class BaseClass(val name: String) {  // HAS_PARAMETER: Constructor has parameter 'name', DEFINES: Defines property
    fun getName(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return name  // REFERENCES: References property
    }
}

// Base class for testing
package com.test

// DEFINES: File defines entities

open class BaseClass(val name: String) {  // HAS_PARAMETER: Constructor has parameter 'name', DEFINES: Defines property
    fun getName(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return name  // REFERENCES: References property
    }
}

// Add a function that calls another function (to ensure CALLS relationship)
fun callHelper(base: BaseClass): String {
    return base.getName()  // CALLS: Calls getName, REFERENCES: base
}

// Add a class that inherits from BaseClass (to ensure INHERITS relationship)
class SubClass(name: String) : BaseClass(name)
