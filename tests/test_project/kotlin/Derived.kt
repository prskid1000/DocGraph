package com.test

import java.util.List  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass(name: String, private val age: Int) : BaseClass(name) {  // INHERITS: Class inherits from BaseClass, HAS_PARAMETER: Constructor has parameters 'name' and 'age'
    private val localVar = 10  // DEFINES: Defines property
    
    fun process(data: List<String>): String {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        val result = helperFunction(data)  // CALLS: Function calls another function
        val localResult = localVar  // REFERENCES: References property 'localVar'
        return result  // RETURNS: Returns type 'String'
    }
    
    private fun helperFunction(items: List<String>): String {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        return if (items.isEmpty()) "" else items[0]  // REFERENCES: References variable 'items'
    }
    
    fun getInfo(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return "Name: $name, Age: $age"  // REFERENCES: References properties
    }
}

// Derived class demonstrating all relationships
package com.test

import java.util.List  // IMPORTS: File imports module
import java.util.ArrayList  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass(name: String, private val age: Int) : BaseClass(name) {  // INHERITS: Class inherits from BaseClass, HAS_PARAMETER: Constructor has parameters 'name' and 'age'
    private val localVar = 10  // DEFINES: Defines property
    
    fun process(data: List<String>): String {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        val result = helperFunction(data)  // CALLS: Function calls another function
        val localResult = localVar  // REFERENCES: References property 'localVar'
        return result  // RETURNS: Returns type 'String'
    }
    
    private fun helperFunction(items: List<String>): String {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        return if (items.isEmpty()) "" else items[0]  // REFERENCES: References variable 'items'
    }
    
    fun getInfo(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return "Name: $name, Age: $age"  // REFERENCES: References properties
    }
}

// Add a function that demonstrates REFERENCES and CALLS
fun main() {
    val instance = DerivedClass("test", 25)  // CALLS: Calls constructor
    val data = ArrayList<String>()
    data.add("data")
    val result = instance.process(data)  // CALLS: Calls method
    val info = instance.getInfo()  // CALLS: Calls method
    println(info)  // CALLS: Calls external function
}
