"""Neo4j graph schema definitions."""
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class NodeSchema:
    """Schema definition for a graph node type."""
    label: str
    properties: List[str]
    indexes: List[str] = None
    
    def __post_init__(self):
        if self.indexes is None:
            self.indexes = []


@dataclass
class RelationshipSchema:
    """Schema definition for a graph relationship type."""
    type: str
    from_node: str
    to_node: str
    properties: List[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = []


class GraphSchema:
    """Defines the knowledge graph schema."""
    
    # Node types
    NODE_FILE = "File"
    NODE_CLASS = "Class"
    NODE_FUNCTION = "Function"
    NODE_VARIABLE = "Variable"
    NODE_MODULE = "Module"
    NODE_PARAMETER = "Parameter"
    NODE_TYPE = "Type"
    
    # Relationship types
    REL_DEFINES = "DEFINES"
    REL_INHERITS = "INHERITS"
    REL_CALLS = "CALLS"
    REL_REFERENCES = "REFERENCES"
    REL_IMPORTS = "IMPORTS"
    REL_HAS_PARAMETER = "HAS_PARAMETER"
    REL_RETURNS = "RETURNS"
    REL_CONTAINS = "CONTAINS"
    
    @staticmethod
    def get_node_schemas() -> Dict[str, NodeSchema]:
        """Get all node schemas.
        
        Returns:
            Dictionary mapping node labels to schemas.
        """
        return {
            GraphSchema.NODE_FILE: NodeSchema(
                label=GraphSchema.NODE_FILE,
                properties=["path", "name", "language", "size", "last_modified", "codebase_id"],
                indexes=["path", "name", "codebase_id"]
            ),
            GraphSchema.NODE_CLASS: NodeSchema(
                label=GraphSchema.NODE_CLASS,
                properties=["name", "signature", "docstring", "file_path", 
                           "start_line", "end_line", "start_column", "end_column",
                           "parent", "decorators", "bases", "codebase_id"],
                indexes=["name", "file_path", "codebase_id"]
            ),
            GraphSchema.NODE_FUNCTION: NodeSchema(
                label=GraphSchema.NODE_FUNCTION,
                properties=["name", "signature", "docstring", "file_path",
                           "start_line", "end_line", "start_column", "end_column",
                           "parent", "decorators", "async", "codebase_id"],
                indexes=["name", "file_path", "signature", "codebase_id"]
            ),
            GraphSchema.NODE_VARIABLE: NodeSchema(
                label=GraphSchema.NODE_VARIABLE,
                properties=["name", "file_path", "start_line", "end_line",
                           "start_column", "end_column", "parent", "codebase_id"],
                indexes=["name", "file_path", "codebase_id"]
            ),
            GraphSchema.NODE_MODULE: NodeSchema(
                label=GraphSchema.NODE_MODULE,
                properties=["name", "path"],
                indexes=["name", "path"]
            ),
            GraphSchema.NODE_PARAMETER: NodeSchema(
                label=GraphSchema.NODE_PARAMETER,
                properties=["name", "type", "default_value"],
                indexes=["name"]
            ),
            GraphSchema.NODE_TYPE: NodeSchema(
                label=GraphSchema.NODE_TYPE,
                properties=["name", "full_name"],
                indexes=["name", "full_name"]
            ),
        }
    
    @staticmethod
    def get_relationship_schemas() -> Dict[str, RelationshipSchema]:
        """Get all relationship schemas.
        
        Returns:
            Dictionary mapping relationship types to schemas.
        """
        return {
            GraphSchema.REL_DEFINES: RelationshipSchema(
                type=GraphSchema.REL_DEFINES,
                from_node=GraphSchema.NODE_FILE,
                to_node=f"{GraphSchema.NODE_CLASS}|{GraphSchema.NODE_FUNCTION}|{GraphSchema.NODE_VARIABLE}",
                properties=[]
            ),
            GraphSchema.REL_INHERITS: RelationshipSchema(
                type=GraphSchema.REL_INHERITS,
                from_node=GraphSchema.NODE_CLASS,
                to_node=GraphSchema.NODE_CLASS,
                properties=[]
            ),
            GraphSchema.REL_CALLS: RelationshipSchema(
                type=GraphSchema.REL_CALLS,
                from_node=GraphSchema.NODE_FUNCTION,
                to_node=GraphSchema.NODE_FUNCTION,
                properties=["line_number", "file_path"]
            ),
            GraphSchema.REL_REFERENCES: RelationshipSchema(
                type=GraphSchema.REL_REFERENCES,
                from_node=GraphSchema.NODE_FUNCTION,
                to_node=f"{GraphSchema.NODE_VARIABLE}|{GraphSchema.NODE_CLASS}",
                properties=["line_number", "file_path"]
            ),
            GraphSchema.REL_IMPORTS: RelationshipSchema(
                type=GraphSchema.REL_IMPORTS,
                from_node=GraphSchema.NODE_FILE,
                to_node=GraphSchema.NODE_MODULE,
                properties=["line_number"]
            ),
            GraphSchema.REL_HAS_PARAMETER: RelationshipSchema(
                type=GraphSchema.REL_HAS_PARAMETER,
                from_node=GraphSchema.NODE_FUNCTION,
                to_node=GraphSchema.NODE_PARAMETER,
                properties=["position"]
            ),
            GraphSchema.REL_RETURNS: RelationshipSchema(
                type=GraphSchema.REL_RETURNS,
                from_node=GraphSchema.NODE_FUNCTION,
                to_node=GraphSchema.NODE_TYPE,
                properties=[]
            ),
            GraphSchema.REL_CONTAINS: RelationshipSchema(
                type=GraphSchema.REL_CONTAINS,
                from_node=GraphSchema.NODE_FILE,
                to_node=GraphSchema.NODE_FILE,
                properties=[]
            ),
        }
    
    @staticmethod
    def get_create_index_queries() -> List[str]:
        """Get Cypher queries to create indexes.
        
        Returns:
            List of Cypher CREATE INDEX queries.
        """
        queries = []
        schemas = GraphSchema.get_node_schemas()
        
        for schema in schemas.values():
            for prop in schema.indexes:
                # Create index on property
                queries.append(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{schema.label}) ON (n.{prop})"
                )
        
        return queries

