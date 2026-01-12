# Neo4j Browser Queries

Common queries to explore and analyze your codebase in Neo4j Browser.

## 📊 Statistics & Overview

### Count All Nodes by Type
```cypher
MATCH (n {codebase_id: 'rmapp'})
WITH labels(n)[0] as label, count(*) as count
RETURN label, count
ORDER BY count DESC
```

### Count All Relationships
```cypher
MATCH (a {codebase_id: 'rmapp'})-[r]->(b {codebase_id: 'rmapp'})
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC
```

### List All Codebases
```cypher
MATCH (n)
RETURN DISTINCT n.codebase_id as codebase_id, count(*) as nodes
ORDER BY codebase_id
```

### Database Overview
```cypher
MATCH (n {codebase_id: 'rmapp'})
RETURN 
  count(DISTINCT n) as total_nodes,
  count(DISTINCT labels(n)[0]) as node_types,
  count{(n)-[]->()} as total_relationships
```

## 🏗️ Entity Queries

### View All Classes
```cypher
MATCH (c:Class {codebase_id: 'rmapp'})
RETURN c.name, c.file_path, c.start_line
ORDER BY c.name
LIMIT 50
```

### View All Functions
```cypher
MATCH (f:Function {codebase_id: 'rmapp'})
RETURN f.name, f.file_path, f.signature, f.parent
ORDER BY f.name
LIMIT 50
```

### View Functions in a Specific File
```cypher
MATCH (f:Function {codebase_id: 'rmapp'})
WHERE f.file_path CONTAINS 'CustomerModel.ts'
RETURN f.name, f.signature, f.start_line, f.parent
ORDER BY f.start_line
```

### Find All Methods of a Class
```cypher
MATCH (f:Function {codebase_id: 'rmapp'})
WHERE f.parent = 'CustomerModel'
RETURN f.name, f.signature, f.start_line
ORDER BY f.start_line
```

### View All Variables
```cypher
MATCH (v:Variable {codebase_id: 'rmapp'})
RETURN v.name, v.file_path, v.start_line
ORDER BY v.name
LIMIT 50
```

## 🔍 Search Queries

### Find Entity by Name
```cypher
MATCH (n {codebase_id: 'rmapp'})
WHERE n.name = 'CustomerModel'
RETURN n, labels(n)[0] as type
```

### Search Entities by Name Pattern
```cypher
MATCH (n {codebase_id: 'rmapp'})
WHERE n.name CONTAINS 'Customer'
RETURN labels(n)[0] as type, n.name, n.file_path
ORDER BY type, n.name
LIMIT 50
```

### Find All Constructors
```cypher
MATCH (f:Function {codebase_id: 'rmapp'})
WHERE f.name = 'constructor'
RETURN f.parent as class_name, f.signature, f.file_path
ORDER BY f.parent
```

### Find Functions with Specific Signature Pattern
```cypher
MATCH (f:Function {codebase_id: 'rmapp'})
WHERE f.signature CONTAINS 'async'
RETURN f.name, f.signature, f.file_path
LIMIT 50
```

## 📁 File-Based Queries

### View All Files
```cypher
MATCH (f:File {codebase_id: 'rmapp'})
RETURN f.path, f.language
ORDER BY f.path
LIMIT 50
```

### Find Files by Language
```cypher
MATCH (f:File {codebase_id: 'rmapp'})
WHERE f.language = 'typescript'
RETURN f.path
ORDER BY f.path
LIMIT 50
```

### Count Entities per File
```cypher
MATCH (f:File {codebase_id: 'rmapp'})<-[:DEFINED_IN]-(n)
RETURN f.path, count(n) as entity_count
ORDER BY entity_count DESC
LIMIT 20
```

## 🔗 Relationship Queries

### View File Relationships
```cypher
MATCH (f:File {codebase_id: 'rmapp'})-[r:DEFINES]->(n)
RETURN f, r, n
LIMIT 25
```

### Find All References
```cypher
MATCH (a {codebase_id: 'rmapp'})-[r:REFERENCES]->(b {codebase_id: 'rmapp'})
RETURN a.name as from, b.name as to, type(r) as relationship
LIMIT 50
```

### Find What References a Specific Entity
```cypher
MATCH (source {codebase_id: 'rmapp'})-[r:REFERENCES]->(target {codebase_id: 'rmapp'})
WHERE target.name = 'CustomerModel'
RETURN source.name, source.file_path, labels(source)[0] as source_type
LIMIT 50
```

### Find What an Entity References
```cypher
MATCH (source {codebase_id: 'rmapp'})-[r:REFERENCES]->(target {codebase_id: 'rmapp'})
WHERE source.name = 'CustomerModel'
RETURN target.name, target.file_path, labels(target)[0] as target_type
LIMIT 50
```

## 🎯 Graph Visualization

### Visualize a Class and Its Methods
```cypher
MATCH (c:Class {codebase_id: 'rmapp', name: 'CustomerModel'})
OPTIONAL MATCH (c)<-[:DEFINED_IN]-(m:Function)
RETURN c, m
```

### Visualize File Dependencies (via references)
```cypher
MATCH (f1:File {codebase_id: 'rmapp'})<-[:DEFINED_IN]-(e1)
MATCH (e1)-[:REFERENCES]->(e2)-[:DEFINED_IN]->(f2:File {codebase_id: 'rmapp'})
WHERE f1.path CONTAINS 'CustomerModel.ts'
RETURN DISTINCT f1, f2
LIMIT 10
```

### Visualize Class Hierarchy
```cypher
MATCH path = (c:Class {codebase_id: 'rmapp'})-[:EXTENDS|IMPLEMENTS*1..3]->(parent:Class {codebase_id: 'rmapp'})
RETURN path
LIMIT 25
```

### Show Entity and Its Context
```cypher
MATCH (n {codebase_id: 'rmapp', name: 'CustomerModel'})
OPTIONAL MATCH (n)-[r]-(connected)
RETURN n, r, connected
LIMIT 50
```

## 📈 Analysis Queries

### Find Most Referenced Entities
```cypher
MATCH (target {codebase_id: 'rmapp'})<-[:REFERENCES]-(source)
RETURN labels(target)[0] as type, target.name, target.file_path, count(source) as reference_count
ORDER BY reference_count DESC
LIMIT 20
```

### Find Most Connected Files
```cypher
MATCH (f:File {codebase_id: 'rmapp'})
OPTIONAL MATCH (f)-[r]-()
RETURN f.path, count(r) as connection_count
ORDER BY connection_count DESC
LIMIT 20
```

### Find Entities with Most Methods/Functions
```cypher
MATCH (c:Class {codebase_id: 'rmapp'})
OPTIONAL MATCH (c)<-[:DEFINED_IN]-(m:Function)
RETURN c.name, c.file_path, count(m) as method_count
ORDER BY method_count DESC
LIMIT 20
```

### Find Files with Most Entities
```cypher
MATCH (f:File {codebase_id: 'rmapp'})
OPTIONAL MATCH (f)<-[:DEFINED_IN]-(entity)
RETURN f.path, count(entity) as entity_count
ORDER BY entity_count DESC
LIMIT 20
```

### Find Orphaned Entities (no relationships)
```cypher
MATCH (n {codebase_id: 'rmapp'})
WHERE NOT (n)-[]-()
RETURN labels(n)[0] as type, n.name, n.file_path
LIMIT 50
```

## 🧹 Maintenance Queries

### Delete Specific Codebase
```cypher
MATCH (n {codebase_id: 'rmapp'})
DETACH DELETE n
```

### Delete All Data (⚠️ Use with caution!)
```cypher
MATCH (n)
DETACH DELETE n
```

### Clear Codebase and Re-index
```cypher
// First delete existing data
MATCH (n {codebase_id: 'rmapp'})
DETACH DELETE n;

// Then run: python index_codebase.py /path/to/codebase --codebase-id rmapp
```

## 🔧 Schema & Index Queries

### View All Indexes
```cypher
SHOW INDEXES
```

### View All Constraints
```cypher
SHOW CONSTRAINTS
```

### Check Database Size
```cypher
MATCH (n)
RETURN count(n) as total_nodes, 
       count{()-[]->()} as total_relationships
```

## 💡 Tips

1. **Replace `codebase_id`**: Change `'rmapp'` to your actual codebase ID in all queries
2. **Adjust LIMIT**: Modify `LIMIT` values based on your needs
3. **Use EXPLAIN**: Prefix queries with `EXPLAIN` to see execution plan
4. **Use PROFILE**: Prefix with `PROFILE` to see execution statistics
5. **Visualize**: Click the graph icon in Neo4j Browser for visual exploration
6. **Export Results**: Use the download button to export query results as CSV/JSON

## 📚 Example Workflows

### Explore New Codebase
```cypher
// 1. Check what was indexed
MATCH (n {codebase_id: 'rmapp'})
RETURN labels(n)[0] as type, count(*) as count
ORDER BY count DESC;

// 2. View file structure
MATCH (f:File {codebase_id: 'rmapp'})
RETURN f.path
ORDER BY f.path
LIMIT 100;

// 3. Find main classes
MATCH (c:Class {codebase_id: 'rmapp'})
RETURN c.name, c.file_path
ORDER BY c.name
LIMIT 50;
```

### Analyze Dependencies
```cypher
// Find which files depend on CustomerModel
MATCH (f1:File)<-[:DEFINED_IN]-(e1:Class {name: 'CustomerModel'})
MATCH (e1)<-[:REFERENCES]-(e2)-[:DEFINED_IN]->(f2:File)
WHERE f1.codebase_id = 'rmapp' AND f2.codebase_id = 'rmapp'
RETURN DISTINCT f2.path as dependent_file
ORDER BY dependent_file;
```

### Find Code Patterns
```cypher
// Find all async functions
MATCH (f:Function {codebase_id: 'rmapp'})
WHERE f.signature CONTAINS 'async'
RETURN f.name, f.parent, f.file_path
LIMIT 50;
```
