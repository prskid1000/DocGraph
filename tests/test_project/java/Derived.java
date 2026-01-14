package com.test;

import java.util.List;  // IMPORTS: File imports module
import java.util.ArrayList;  // IMPORTS: File imports module

// DEFINES: File defines entities

public class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    private int age;  // DEFINES: Defines field
    
    public DerivedClass(String name, int age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns void (implicit)
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References field
        int localVar = 10;  // DEFINES: Defines variable
        int result = localVar;  // REFERENCES: References variable 'localVar'
    }
    
    public String process(List<String> data) {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        String result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References field
        return result;  // RETURNS: Returns type 'String'
    }
    
    private String helperFunction(List<String> items) {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        return items.isEmpty() ? "" : items.get(0);  // REFERENCES: References variable 'items'
    }
    
    public String getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return "Name: " + this.name + ", Age: " + this.age;  // REFERENCES: References fields
    }
}

// Derived class demonstrating all relationships
package com.test;

import java.util.List;  // IMPORTS: File imports module
import java.util.ArrayList;  // IMPORTS: File imports module

// DEFINES: File defines entities

public class DerivedClass extends BaseClass {  // INHERITS: Class inherits from BaseClass
    private int age;  // DEFINES: Defines field
    
    public DerivedClass(String name, int age) {  // HAS_PARAMETER: Function has parameters 'name' and 'age', RETURNS: Returns void (implicit)
        super(name);  // CALLS: Calls parent constructor
        this.age = age;  // REFERENCES: References field
        int localVar = 10;  // DEFINES: Defines variable
        int result = localVar;  // REFERENCES: References variable 'localVar'
    }
    
    public String process(List<String> data) {  // HAS_PARAMETER: Function has parameter 'data', RETURNS: Returns type 'String'
        String result = helperFunction(data);  // CALLS: Function calls another function
        this.name = result;  // REFERENCES: References field
        return result;  // RETURNS: Returns type 'String'
    }
    
    private String helperFunction(List<String> items) {  // HAS_PARAMETER: Function has parameter 'items', RETURNS: Returns type 'String'
        return items.isEmpty() ? "" : items.get(0);  // REFERENCES: References variable 'items'
    }
    
    public String getInfo() {  // HAS_PARAMETER: Function has parameter 'this' (implicit), RETURNS: Returns type 'String'
        return "Name: " + this.name + ", Age: " + this.age;  // REFERENCES: References fields
    }
}

// Add a function that demonstrates REFERENCES and CALLS
class Main {
    public static void main(String[] args) {
        DerivedClass instance = new DerivedClass("test", 25);  // CALLS: Calls constructor
        List<String> data = new ArrayList<>();
        data.add("data");
        String result = instance.process(data);  // CALLS: Calls method
        String info = instance.getInfo();  // CALLS: Calls method
        System.out.println(info);  // CALLS: Calls external function
    }
}
