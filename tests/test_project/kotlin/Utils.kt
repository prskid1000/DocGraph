package com.test

// DEFINES: File defines entities

object Utils {
    fun helperFunction(data: Array<String>): String {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        if (data.isEmpty()) {  // REFERENCES: References variable 'data'
            return ""  // RETURNS: Returns type 'String'
        }
        return data[0]  // REFERENCES: References variable 'data', RETURNS: Returns type 'String'
    }
    
    fun calculateSum(a: Int, b: Int): Int {  // HAS_PARAMETER: Function has parameters 'a' and 'b', RETURNS: Returns type 'Int'
        val result = a + b  // REFERENCES: References variables 'a' and 'b'
        return result  // RETURNS: Returns type 'Int'
    }
    
    fun formatString(input: String): String {  // HAS_PARAMETER: Function has parameter 'input', RETURNS: Returns type 'String'
        return "Formatted: $input"  // REFERENCES: References variable 'input', RETURNS: Returns type 'String'
    }
}
