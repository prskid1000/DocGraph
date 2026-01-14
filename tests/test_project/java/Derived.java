package com.test;

import com.test.BaseClass;  // IMPORTS: File imports module
import com.test.Utils;  // IMPORTS: File imports module

// DEFINES: File defines entities

public class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    private int age;  // DEFINES: Defines field
    
    public DerivedClass(String name, int age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns void (implicit)
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References field 'age'
        int localVar = 10;  // DEFINES: Defines variable
        int result = localVar;  // REFERENCES: References variable 'localVar'
        this.value = Utils.calculateSum(localVar, 5);  // CALLS: Calls function, REFERENCES: References variables
    }
    
    public String process(String[] data) {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        String result = Utils.helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References field 'this.name' and variable 'result'
        return result;  // RETURNS: Returns type 'String'
    }
    
    private String helperFunction(String[] items) {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        if (items.length == 0) {  // REFERENCES: References variable 'items'
            return "";  // RETURNS: Returns type 'String'
        }
        return items[0];  // REFERENCES: References variable 'items', RETURNS: Returns type 'String'
    }
    
    public String getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        String info = "Name: " + this.name + ", Age: " + this.age;  // REFERENCES: References fields 'this.name' and 'this.age'
        return info;  // RETURNS: Returns type 'String'
    }
    
    public int compute(int x, int y) {  // HAS_PARAMETER: Function has parameters 'x' and 'y', RETURNS: Returns type 'int'
        return Utils.calculateSum(x, y);  // CALLS: Calls function, RETURNS: Returns type 'int'
    }
}

// Add a function that demonstrates REFERENCES and CALLS
class Main {
    public static void main(String[] args) {  // HAS_PARAMETER: Function has parameter 'args', RETURNS: Returns type 'void'
        DerivedClass instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
        String[] data = {"data"};  // DEFINES: Defines variable
        String result = instance.process(data);  // CALLS: Calls method
        String info = instance.getInfo();  // CALLS: Calls method
        int computed = instance.compute(10, 20);  // CALLS: Calls method
        String helperResult = Utils.formatString(result);  // CALLS: Calls function, REFERENCES: References variable 'result'
    }
}
