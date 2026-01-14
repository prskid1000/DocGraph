package com.test

import com.test.BaseClass  // IMPORTS: File imports module
import com.test.Utils  // IMPORTS: File imports module

// DEFINES: File defines entities

class DerivedClass(name: String, private val age: Int) : BaseClass(name) {  // INHERITS: Class inherits from BaseClass, HAS_PARAMETER: Constructor has parameters 'name' and 'age'
    private val localVar = 10  // DEFINES: Defines property
    
    fun process(data: Array<String>): String {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        val result = Utils.helperFunction(data)  // CALLS: Function calls another function
        val localResult = localVar  // REFERENCES: References property 'localVar'
        this.value = Utils.calculateSum(localVar, 5)  // CALLS: Calls function, REFERENCES: References variables
        return result  // RETURNS: Returns type 'String'
    }
    
    private fun helperFunction(items: Array<String>): String {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        return if (items.isEmpty()) "" else items[0]  // REFERENCES: References variable 'items', RETURNS: Returns type 'String'
    }
    
    fun getInfo(): String {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        val info = "Name: $name, Age: $age"  // REFERENCES: References properties 'name' and 'age'
        return info  // RETURNS: Returns type 'String'
    }
    
    fun compute(x: Int, y: Int): Int {  // HAS_PARAMETER: Function has parameters 'x' and 'y', RETURNS: Returns type 'Int'
        return Utils.calculateSum(x, y)  // CALLS: Calls function, RETURNS: Returns type 'Int'
    }
}

// Add a function that demonstrates REFERENCES and CALLS
fun main() {  // HAS_PARAMETER: Function has no parameters, RETURNS: Returns type 'Unit' (implicit)
    val instance = DerivedClass("test", 25)  // CALLS: Calls constructor
    val data = arrayOf("data")  // DEFINES: Defines variable
    val result = instance.process(data)  // CALLS: Calls method
    val info = instance.getInfo()  // CALLS: Calls method
    val computed = instance.compute(10, 20)  // CALLS: Calls method
        val helperResult = Utils.formatString(result)  // CALLS: Calls function, REFERENCES: References variable 'result'
}
