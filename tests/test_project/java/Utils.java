package com.test;

// DEFINES: File defines entities

public class Utils {
    public static String helperFunction(String[] data) {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        if (data.length == 0) {  // REFERENCES: References variable 'data'
            return "";  // RETURNS: Returns type 'String'
        }
        return data[0];  // REFERENCES: References variable 'data', RETURNS: Returns type 'String'
    }
    
    public static int calculateSum(int a, int b) {  // HAS_PARAMETER: Function has parameters 'a' and 'b', RETURNS: Returns type 'int'
        int result = a + b;  // REFERENCES: References variables 'a' and 'b'
        return result;  // RETURNS: Returns type 'int'
    }
    
    public static String formatString(String input) {  // HAS_PARAMETER: Function has parameter 'input', RETURNS: Returns type 'String'
        return "Formatted: " + input;  // REFERENCES: References variable 'input', RETURNS: Returns type 'String'
    }
}
