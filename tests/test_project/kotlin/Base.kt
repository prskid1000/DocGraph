// Base class for testing
package com.test

// DEFINES: File defines entities

open class BaseClass(val name: String) {  // HAS_PARAMETER: Constructor has parameter 'name', DEFINES: Defines property
    fun getName(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return name  // REFERENCES: References property
    }
}
