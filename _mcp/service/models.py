"""Pydantic models and JSON schemas for DocGraph MCP tools."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# SEARCH ENTITIES
# ============================================================================
class SearchCodeEntitiesInput(BaseModel):
    """Input model for searching code entities."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase to search"
    )
    
    query: str = Field(
        ...,
        description="Search query text (semantic search for entities)"
    )
    
    entity_type: Optional[str] = Field(
        None,
        description="Optional filter by entity type. Supported: 'function', 'class', 'variable', 'module', 'parameter', 'type'. Case-insensitive (auto-capitalized for Neo4j)."
    )
    
    limit: int = Field(
        10,
        description="Maximum number of results to return"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


SEARCH_CODE_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase to search"
        },
        "query": {
            "type": "string",
            "description": "Search query text (semantic search)"
        },
        "entity_type": {
            "type": ["string", "null"],
            "description": "Filter by entity type. Supported: 'function', 'class', 'variable', 'module', 'parameter', 'type'. Case-insensitive."
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return"
        }
    },
    "required": ["codebase_id", "query"]
}


# ============================================================================
# GET DEFINITION
# ============================================================================
class GetDefinitionInput(BaseModel):
    """Input model for getting entity definition."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    entity_name: str = Field(
        ...,
        description="Specific name of the code entity (e.g., 'MyClass', 'calculate_total')"
    )
    
    file_path: Optional[str] = Field(
        None,
        description="File path for disambiguation (optional)"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


GET_DEFINITION_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "entity_name": {
            "type": "string",
            "description": "Name of the code entity"
        },
        "file_path": {
            "type": ["string", "null"],
            "description": "File path for disambiguation (optional)"
        }
    },
    "required": ["codebase_id", "entity_name"]
}


# ============================================================================
# FIND REFERENCES
# ============================================================================
class FindReferencesInput(BaseModel):
    """Input model for finding entity references."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    entity_name: str = Field(
        ...,
        description="Specific name of the entity to find references for (e.g., 'MyClass', 'calculate_total')"
    )
    
    entity_type: Optional[str] = Field(
        None,
        description="Optional entity type for disambiguation. Supported: 'function', 'class', 'variable'. Case-insensitive (auto-capitalized for Neo4j)."
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


FIND_REFERENCES_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "entity_name": {
            "type": "string",
            "description": "Name of the entity to find references for"
        },
        "entity_type": {
            "type": ["string", "null"],
            "description": "Optional entity type for disambiguation. Supported: 'function', 'class', 'variable'. Case-insensitive."
        }
    },
    "required": ["codebase_id", "entity_name"]
}


# ============================================================================
# CALL GRAPH
# ============================================================================
class GetCallGraphInput(BaseModel):
    """Input model for call graph analysis."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    function_name: str = Field(
        ...,
        description="Name of the function"
    )
    
    depth: int = Field(
        2,
        description="Maximum depth to traverse"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


GET_CALL_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "function_name": {
            "type": "string",
            "description": "Name of the function"
        },
        "depth": {
            "type": "integer",
            "description": "Maximum depth to traverse"
        }
    },
    "required": ["codebase_id", "function_name"]
}


# ============================================================================
# DEPENDENCIES
# ============================================================================
class GetDependenciesInput(BaseModel):
    """Input model for getting file dependencies."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    file_path: str = Field(
        ...,
        description="Path to the file"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


GET_DEPENDENCIES_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "file_path": {
            "type": "string",
            "description": "Path to the file"
        }
    },
    "required": ["codebase_id", "file_path"]
}


# ============================================================================
# QUERY GRAPH
# ============================================================================
class QueryGraphInput(BaseModel):
    """Input model for custom Cypher queries."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    cypher_query: str = Field(
        ...,
        description="Cypher query string. Available node labels: Class, Function, Variable, Module, Parameter, Type, File. Relationships: DEFINES, INHERITS, CALLS, REFERENCES, IMPORTS, HAS_PARAMETER, RETURNS, CONTAINS."
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


QUERY_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "cypher_query": {
            "type": "string",
            "description": "Cypher query string. Available node labels: Class, Function, Variable, Module, Parameter, Type, File. Available relationships: DEFINES, INHERITS, CALLS, REFERENCES, IMPORTS, HAS_PARAMETER, RETURNS, CONTAINS."
        }
    },
    "required": ["codebase_id", "cypher_query"]
}


# ============================================================================
# GET CONTEXT
# ============================================================================
class GetContextInput(BaseModel):
    """Input model for getting code context."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    file_path: str = Field(
        ...,
        description="Path to the file"
    )
    
    line_number: int = Field(
        ...,
        description="Line number"
    )
    
    context_lines: int = Field(
        50,
        description="Number of context lines to return"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


GET_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "file_path": {
            "type": "string",
            "description": "Path to the file"
        },
        "line_number": {
            "type": "integer",
            "description": "Line number"
        },
        "context_lines": {
            "type": "integer",
            "description": "Number of context lines to return"
        }
    },
    "required": ["codebase_id", "file_path", "line_number"]
}


# ============================================================================
# TASK MANAGEMENT
# ============================================================================
class SubmitTaskInput(BaseModel):
    """Input model for submitting a background task."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase to operate on"
    )
    
    task_type: str = Field(
        ...,
        description="Type of task to submit (e.g., 'INDEX_CODEBASE')"
    )
    
    params: Dict[str, Any] = Field(
        ...,
        description="Task-specific parameters"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


SUBMIT_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase to operate on"
        },
        "task_type": {
            "type": "string",
            "description": "Type of task to submit (e.g., 'INDEX_CODEBASE')"
        },
        "params": {
            "type": "object",
            "description": "Task-specific parameters"
        }
    },
    "required": ["codebase_id", "task_type", "params"]
}


class TaskResultInput(BaseModel):
    """Input model for viewing task results."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase to view tasks for"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


TASK_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase to view tasks for"
        }
    },
    "required": ["codebase_id"]
}


class CancelTaskInput(BaseModel):
    """Input model for cancelling a task."""
    
    codebase_id: str = Field(
        ...,
        description="ID of the codebase"
    )
    
    task_id: str = Field(
        ...,
        description="ID of the task to cancel"
    )
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


CANCEL_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "codebase_id": {
            "type": "string",
            "description": "ID of the codebase"
        },
        "task_id": {
            "type": "string",
            "description": "ID of the task to cancel"
        }
    },
    "required": ["codebase_id", "task_id"]
}
