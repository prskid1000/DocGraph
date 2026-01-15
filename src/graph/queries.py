"""Common Cypher queries for graph operations."""
from typing import List, Dict, Any, Optional


class GraphQueries:
    """Collection of common Cypher queries."""
    
    @staticmethod
    def find_entity_by_name(name: str, entity_type: Optional[str] = None) -> str:
        """Find entity by name.
        
        Args:
            name: Entity name.
            entity_type: Optional entity type filter (will be capitalized if needed).
            
        Returns:
            Cypher query string.
        """
        if entity_type:
            # Capitalize entity type to match schema (Class, Function, Variable, etc.)
            entity_type_capitalized = entity_type.capitalize()
            return f"""
                MATCH (n:{entity_type_capitalized} {{name: $name}})
                RETURN n
                LIMIT 10
            """
        else:
            return """
                MATCH (n)
                WHERE n.name = $name
                RETURN n
                LIMIT 10
            """
    
    @staticmethod
    def find_entities_in_file(file_path: str, codebase_id: Optional[str] = None) -> str:
        """Find all entities in a file.
        
        Args:
            file_path: Path to the file.
            codebase_id: Optional codebase filter.
            
        Returns:
            Cypher query string.
        """
        if codebase_id:
            return """
                MATCH (f:File {path: $file_path, codebase_id: $codebase_id})
                MATCH (f)-[:DEFINES]->(e)
                WHERE e.codebase_id = $codebase_id
                RETURN e
            """
        return """
            MATCH (f:File {path: $file_path})
            MATCH (f)-[:DEFINES]->(e)
            RETURN e
        """
    
    @staticmethod
    def find_references_to_entity(entity_name: str, entity_type: str) -> str:
        """Find all references to an entity.
        
        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity (will be capitalized if needed).
            
        Returns:
            Cypher query string.
        """
        # Capitalize entity type to match schema (Class, Function, Variable)
        entity_type_capitalized = entity_type.capitalize() if entity_type else "Function"
        
        return f"""
            MATCH (target:{entity_type_capitalized} {{name: $entity_name}})
            MATCH (source)-[r:CALLS|REFERENCES|INHERITS]->(target)
            RETURN source, r, target
        """
    
    @staticmethod
    def get_call_graph(function_name: str, depth: int = 2, codebase_id: Optional[str] = None) -> str:
        """Get call graph for a function.
        
        Args:
            function_name: Name of the function.
            depth: Maximum depth to traverse.
            codebase_id: Optional codebase filter.
            
        Returns:
            Cypher query string.
        """
        if codebase_id:
            # Use separate queries for calls and called_by to avoid WHERE clause issues with OPTIONAL MATCH
            # Filter codebase_id in the pattern itself for better performance
            return f"""
                MATCH (f:Function {{name: $function_name, codebase_id: $codebase_id}})
                OPTIONAL MATCH (f)-[:CALLS*1..{depth}]->(called:Function {{codebase_id: $codebase_id}})
                WITH f, collect(DISTINCT called) as called_list
                OPTIONAL MATCH (caller:Function {{codebase_id: $codebase_id}})-[:CALLS*1..{depth}]->(f)
                RETURN f, 
                       [c IN called_list WHERE c IS NOT NULL] as calls,
                       [c IN collect(DISTINCT caller) WHERE c IS NOT NULL] as called_by
            """
        return f"""
            MATCH (f:Function {{name: $function_name}})
            OPTIONAL MATCH path1 = (f)-[:CALLS*1..{depth}]->(called:Function)
            OPTIONAL MATCH path2 = (caller:Function)-[:CALLS*1..{depth}]->(f)
            RETURN f, 
                   collect(DISTINCT called) as calls,
                   collect(DISTINCT caller) as called_by
        """
    
    @staticmethod
    def get_dependencies(file_path: str, codebase_id: Optional[str] = None) -> str:
        """Get all dependencies for a file.
        
        Args:
            file_path: Path to the file.
            codebase_id: Optional codebase filter.
            
        Returns:
            Cypher query string.
        """
        if codebase_id:
            return """
                MATCH (f:File {path: $file_path, codebase_id: $codebase_id})
                MATCH (f)-[:IMPORTS]->(m:Module {codebase_id: $codebase_id})
                RETURN m
            """
        return """
            MATCH (f:File {path: $file_path})
            MATCH (f)-[:IMPORTS]->(m:Module)
            RETURN m
        """
    
    @staticmethod
    def get_inheritance_hierarchy(class_name: str) -> str:
        """Get inheritance hierarchy for a class.
        
        Args:
            class_name: Name of the class.
            
        Returns:
            Cypher query string.
        """
        return """
            MATCH (c:Class {name: $class_name})
            OPTIONAL MATCH (c)-[:INHERITS]->(parent:Class)
            OPTIONAL MATCH (child:Class)-[:INHERITS]->(c)
            RETURN c,
                   collect(DISTINCT parent) as parents,
                   collect(DISTINCT child) as children
        """
    
    @staticmethod
    def find_similar_entities(entity_name: str, entity_type: str, limit: int = 10) -> str:
        """Find similar entities (placeholder for semantic search integration).
        
        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity (will be capitalized if needed).
            limit: Maximum number of results.
            
        Returns:
            Cypher query string.
        """
        # Capitalize entity type to match schema (Class, Function, Variable, etc.)
        entity_type_capitalized = entity_type.capitalize()
        return f"""
            MATCH (e:{entity_type_capitalized})
            WHERE e.name CONTAINS $entity_name
               OR e.name =~ $entity_name
            RETURN e
            LIMIT $limit
        """
    
    @staticmethod
    def get_context_around_location(file_path: str, line_number: int, context_lines: int = 50) -> str:
        """Get context around a specific location.
        
        Args:
            file_path: Path to the file.
            line_number: Line number.
            context_lines: Number of context lines.
            
        Returns:
            Cypher query string.
        """
        start_line = max(1, line_number - context_lines)
        end_line = line_number + context_lines
        
        return """
            MATCH (f:File {path: $file_path})
            MATCH (f)-[:DEFINES]->(e)
            WHERE e.start_line >= $start_line AND e.end_line <= $end_line
            OPTIONAL MATCH (e)<-[:CALLS|REFERENCES]-(related)
            RETURN e, collect(DISTINCT related) as related_entities
        """

