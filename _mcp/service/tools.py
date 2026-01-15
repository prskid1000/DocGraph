"""Tool definitions for DocGraph MCP Server."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel
import json
from _mcp.logger import app_logger as logger
import sys
import re
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

from _mcp.service import models
from src.query.engine import QueryEngine
from _mcp.service.task import submit_task, get_task_results, cancel_task

# Shared instances dictionary (populated by application.py at startup)
shared_instances = {}


def parse_bracket_params(s: str) -> Dict[str, Any]:
    """Parse a simple bracketed param format into a dict.

    Supported line forms:
      [key] = value
      [key] = "value with spaces or backslashes"
      [key] = 'value'
    Trailing commas are ignored. Blank lines and lines starting with # or // are skipped.
    """
    out: Dict[str, Any] = {}
    if not isinstance(s, str):
        return out

    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        # drop trailing commas
        if line.endswith(','):
            line = line[:-1].rstrip()

        m = re.match(r'^\[?\s*(?P<key>[^\]\=]+?)\s*\]?\s*=\s*(?P<val>.*)$', line)
        if not m:
            continue

        key = m.group('key').strip()
        val = m.group('val').strip()

        # strip surrounding quotes if present
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        out[key] = val

    return out


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool with all its configuration."""

    identifier: str
    title: str
    description: str
    input_schema: Dict[str, Any]
    input_model: type[BaseModel]
    handler: Callable[..., Dict[str, Any]]

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with validated arguments."""
        # If `params` is provided as a string, parse it using the bracket-format parser.
        if isinstance(arguments, dict) and 'params' in arguments and isinstance(arguments['params'], str):
            arguments = dict(arguments)  # shallow copy
            arguments['params'] = parse_bracket_params(arguments['params'])

        # Validate input
        payload = self.input_model.model_validate(arguments)
        
        # Execute handler with validated data
        return self.handler(**payload.model_dump(by_alias=False))


def search_code_entities_handler(codebase_id: str, query: str, entity_type: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """Handle search_code_entities tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        results = engine.search_code_entities(query, entity_type=entity_type, limit=limit)
        
        return {
            "success": True,
            "data": results,
            "query": query,
            "entity_type": entity_type or "",
            "total_results": len(results),
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "entity_type": entity_type or "",
            "codebase_id": codebase_id
        }


def get_definition_handler(codebase_id: str, entity_name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Handle get_definition tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        result = engine.get_definition(entity_name, file_path=file_path)
        
        return {
            "success": True,
            "data": result,
            "entity_name": entity_name,
            "file_path": file_path,
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "entity_name": entity_name,
            "file_path": file_path,
            "codebase_id": codebase_id
        }


def find_references_handler(codebase_id: str, entity_name: str, entity_type: Optional[str] = None) -> Dict[str, Any]:
    """Handle find_references tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        results = engine.find_references(entity_name, entity_type=entity_type)
        
        return {
            "success": True,
            "data": results,
            "entity_name": entity_name,
            "entity_type": entity_type or "",
            "total_references": len(results),
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "entity_name": entity_name,
            "entity_type": entity_type or "",
            "codebase_id": codebase_id
        }


def get_call_graph_handler(codebase_id: str, function_name: str, depth: int = 2) -> Dict[str, Any]:
    """Handle get_call_graph tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        graph = engine.get_call_graph(function_name, depth=depth, codebase_id=codebase_id)
        
        # Transform data structure to match widget expectations
        # Widget expects: { name, type, file_path, line_number, children: [...] }
        function_data = graph.get('function')
        
        # Combine calls and called_by into children array
        calls = graph.get('calls', [])
        called_by = graph.get('called_by', [])
        children = []
        
        # Add "calls" (functions this function calls)
        for call in calls:
            if call:  # Filter out None values
                children.append({
                    'name': call.get('name', ''),
                    'type': 'Function',
                    'file_path': call.get('file_path', ''),
                    'line_number': call.get('start_line') or call.get('line_number'),
                    'id': call.get('id', ''),
                    'direction': 'calls',
                    **{k: v for k, v in call.items() if k not in ['name', 'type', 'file_path', 'line_number', 'start_line', 'id']}
                })
        
        # Add "called_by" (functions that call this function)
        for caller in called_by:
            if caller:  # Filter out None values
                children.append({
                    'name': caller.get('name', ''),
                    'type': 'Function',
                    'file_path': caller.get('file_path', ''),
                    'line_number': caller.get('start_line') or caller.get('line_number'),
                    'id': caller.get('id', ''),
                    'direction': 'called_by',
                    **{k: v for k, v in caller.items() if k not in ['name', 'type', 'file_path', 'line_number', 'start_line', 'id']}
                })
        
        # Create the root node structure - always return valid structure even if function not found
        # This allows widget to show "No call graph data found" instead of loading forever
        call_graph_data = {
            'name': function_data.get('name', function_name) if function_data else function_name,
            'type': 'Function',
            'file_path': function_data.get('file_path', '') if function_data else '',
            'line_number': (function_data.get('start_line') or function_data.get('line_number')) if function_data else None,
            'children': children
        }
        
        return {
            "success": True,
            "data": call_graph_data,
            "function_name": function_name,
            "depth": depth,
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "function_name": function_name,
            "depth": depth,
            "codebase_id": codebase_id
        }


def get_dependencies_handler(codebase_id: str, file_path: str) -> Dict[str, Any]:
    """Handle get_dependencies tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        deps = engine.get_dependencies(file_path, codebase_id=codebase_id)
        
        # Transform data to match widget expectations
        # Widget expects array of { name, type, file_path, line_number, id, ... }
        transformed_deps = []
        for dep in deps:
            if dep:  # Filter out None values
                transformed_deps.append({
                    'name': dep.get('name', dep.get('path', 'Unknown')),
                    'type': 'Module',
                    'file_path': dep.get('file_path', dep.get('path', '')),
                    'line_number': dep.get('start_line') or dep.get('line_number'),
                    'id': dep.get('id', ''),
                    **{k: v for k, v in dep.items() if k not in ['name', 'type', 'file_path', 'line_number', 'start_line', 'id', 'path']}
                })
        
        return {
            "success": True,
            "data": transformed_deps,
            "file_path": file_path,
            "codebase_id": codebase_id,
            "total_results": len(transformed_deps)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
            "codebase_id": codebase_id
        }


def query_graph_handler(codebase_id: str, cypher_query: str) -> Dict[str, Any]:
    """Handle query_graph tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        results = engine.query_graph(cypher_query)
        
        return {
            "success": True,
            "data": results,
            "query": cypher_query,
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": cypher_query,
            "codebase_id": codebase_id
        }


def get_context_handler(codebase_id: str, file_path: str, line_number: int, context_lines: int = 50) -> Dict[str, Any]:
    """Handle get_context tool call."""
    try:
        get_vector_db = shared_instances.get('get_vector_db')
        engine = QueryEngine(
            codebase_id=codebase_id,
            neo4j_client=shared_instances.get('neo4j_client'),
            vector_db=get_vector_db(codebase_id) if get_vector_db else None,
            embedding_model=shared_instances.get('embedding_model')
        )
        context = engine.get_context(file_path, line_number, context_lines=context_lines)
        
        return {
            "success": True,
            "data": context,
            "file_path": file_path,
            "line_number": line_number,
            "context_lines": context_lines,
            "codebase_id": codebase_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
            "line_number": line_number,
            "codebase_id": codebase_id
        }


def submit_task_handler(codebase_id: str, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle submit_task tool call - creates a background task."""
    # Use explicit codebase_id parameter (task_manager.submit_task expects codebase_id)
    return submit_task(codebase_id=codebase_id, task_type=task_type, params=params)


def get_task_results_handler(codebase_id: str) -> Dict[str, Any]:
    """Handle get_task_results tool call - returns all tasks for a codebase."""
    # Filter tasks by codebase_id (tenant)
    # Use explicit codebase_id (don't rely on request context)
    all_tasks = get_task_results(codebase_id)
    return all_tasks


def cancel_task_handler(codebase_id: str, task_id: str) -> Dict[str, Any]:
    """Handle cancel_task tool call - cancels a running/pending task."""
    # Use explicit codebase_id so we don't rely on request context
    return cancel_task(task_id=task_id, codebase_id=codebase_id)


# Tool definitions registry
TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "search-code-entities": ToolDefinition(
        identifier="search-code-entities",
        title="Search Code Entities",
        description="Search for code entities (functions, classes, variables) using semantic search. Returns matching entities with their definitions and locations.",
        input_schema=models.SEARCH_CODE_ENTITIES_SCHEMA,
        input_model=models.SearchCodeEntitiesInput,
        handler=search_code_entities_handler
    ),
    "get-definition": ToolDefinition(
        identifier="get-definition",
        title="Get Entity Definition",
        description="Get the complete definition and metadata for a specific code entity (function, class, or variable).",
        input_schema=models.GET_DEFINITION_SCHEMA,
        input_model=models.GetDefinitionInput,
        handler=get_definition_handler
    ),
    "find-references": ToolDefinition(
        identifier="find-references",
        title="Find References",
        description="Find all usages and references to a specific code entity throughout the codebase.",
        input_schema=models.FIND_REFERENCES_SCHEMA,
        input_model=models.FindReferencesInput,
        handler=find_references_handler
    ),
    "get-call-graph": ToolDefinition(
        identifier="get-call-graph",
        title="Get Call Graph",
        description="Get the call graph for a function showing what it calls and what calls it.",
        input_schema=models.GET_CALL_GRAPH_SCHEMA,
        input_model=models.GetCallGraphInput,
        handler=get_call_graph_handler
    ),
    "get-dependencies": ToolDefinition(
        identifier="get-dependencies",
        title="Get Dependencies",
        description="Get import/module dependencies for a specific file.",
        input_schema=models.GET_DEPENDENCIES_SCHEMA,
        input_model=models.GetDependenciesInput,
        handler=get_dependencies_handler
    ),
    "query-graph": ToolDefinition(
        identifier="query-graph",
        title="Query Graph",
        description="Execute a custom Cypher query on the knowledge graph for advanced queries.",
        input_schema=models.QUERY_GRAPH_SCHEMA,
        input_model=models.QueryGraphInput,
        handler=query_graph_handler
    ),
    "get-context": ToolDefinition(
        identifier="get-context",
        title="Get Code Context",
        description="Get the code context around a specific location including related entities and definitions.",
        input_schema=models.GET_CONTEXT_SCHEMA,
        input_model=models.GetContextInput,
        handler=get_context_handler
    ),
    # Task management tools
    "submit-task": ToolDefinition(
        identifier="submit-task",
        title="Submit Background Task",
        description="Create and submit a background task (e.g., INDEX_CODEBASE). Returns immediately with task ID. Use task-result to view progress.",
        input_schema=models.SUBMIT_TASK_SCHEMA,
        input_model=models.SubmitTaskInput,
        handler=submit_task_handler
    ),
    "task-result": ToolDefinition(
        identifier="task-result",
        title="View Task Results",
        description="View all background tasks grouped by status (pending, running, completed). Shows progress for running tasks and results for completed tasks.",
        input_schema=models.TASK_RESULT_SCHEMA,
        input_model=models.TaskResultInput,
        handler=get_task_results_handler
    ),
    "cancel-task": ToolDefinition(
        identifier="cancel-task",
        title="Cancel Task",
        description="Cancel a pending or running background task by task ID.",
        input_schema=models.CANCEL_TASK_SCHEMA,
        input_model=models.CancelTaskInput,
        handler=cancel_task_handler
    ),
}


def get_tool_definition(tool_id: str) -> Optional[ToolDefinition]:
    """Get a tool definition by ID."""
    return TOOL_DEFINITIONS.get(tool_id)


def get_all_tool_definitions() -> list[ToolDefinition]:
    """Get all tool definitions."""
    return list(TOOL_DEFINITIONS.values())
