# DocGraph

A fully local codebase knowledge graph system that extracts classes, functions, variables, and their references from multi-language codebases, storing them in Neo4j (graph database) and ChromaDB (vector database) for semantic search. Provides MCP server integration for LLM access.

## Features

- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Kotlin, HTML, SCSS
- **Knowledge Graph**: Extracts classes, functions, variables, and relationships
- **Hybrid Storage**: Neo4j for graph relationships, ChromaDB for semantic embeddings
- **MCP Server**: Model Context Protocol server for LLM integration
- **Fully Local**: All processing runs locally, no cloud dependencies
- **Incremental Updates**: Real-time updates as code changes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

CHROMADB_PERSIST_DIR=~/.docgraph/chromadb
CHROMADB_COLLECTION=code_entities

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# MCP Server HTTP Configuration
DOCGRAPH_HOST=127.0.0.1
DOCGRAPH_PORT=5500
```

3. Install Neo4j (local):
   - Download from https://neo4j.com/download/
   - Or use Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j:latest`

## Usage

### Index a Codebase

```bash
# Index with auto-generated codebase ID (uses directory name)
python index_codebase.py /path/to/codebase

# Index with custom codebase ID
python index_codebase.py /path/to/codebase --codebase-id myproject

# Index specific languages
python index_codebase.py /path/to/codebase --languages python javascript

# Clear existing data and re-index
python index_codebase.py /path/to/codebase --codebase-id myproject --clear
```

### Manage ChromaDB with Streamlit UI

```bash
# Start the Chroma DB manager UI
streamlit run chroma/app.py
```

**Features:**
- Browse, add, edit, delete documents
- Semantic search with auto-embedding
- Import/Export JSONL with batch processing
- Collection switching and administration

**Default Settings:**
- Collection base: `code_entities`
- Codebase ID: `default`
- Resolved collection name: `code_entities_default`

**To view a specific codebase's data:**

In the sidebar, update:
- `Collection base`: `code_entities`
- `Codebase ID`: your codebase ID (e.g., `rmapp`)

Or set environment variable:
```bash
$env:DOCGRAPH_CODEBASE="rmapp"
streamlit run chroma/app.py
```

### Start MCP Server

The MCP server now runs as a production-ready HTTP service using **FastMCP** with Server-Sent Events (SSE) for real-time streaming:

```bash
# Start server on default host/port (0.0.0.0:5500)
python -m _mcp.application

# Or directly run
python _mcp/application.py

# Start with custom codebase ID
DOCGRAPH_CODEBASE_ID=myproject python -m _mcp.application

# Start with custom host and port
DOCGRAPH_HOST=localhost DOCGRAPH_PORT=9000 python -m _mcp.application
```

**Server Endpoints:**
- **Health**: `http://localhost:5500/health`
- **Info**: `http://localhost:5500/info`
- **SSE Connection**: `http://localhost:5500/mcp`
- **Messages Endpoint**: `http://localhost:5500/mcp/messages` (POST)
- **Manifest**: `http://localhost:5500/.well-known/mcp.json`

**Example MCP Client Configuration (Claude Desktop):**
```json
{
  "mcpServers": {
    "docgraph": {
      "command": "python",
      "args": ["-m", "_mcp.application"],
      "env": {
        "DOCGRAPH_PORT": "5500",
        "NEO4J_URI": "neo4j://localhost:7687",
        "CHROMADB_PERSIST_DIR": "~/.docgraph/chromadb"
      }
    }
  }
}
```

**Important:** All tools now require a `codebase_id` parameter to specify which codebase to query. This allows the same server instance to manage multiple codebases.

### Available MCP Tools

The MCP server exposes the following tools for LLM integration:

#### `search_code_entities`
Search for code entities using semantic search.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase to search
- `query` (string, required): Search query text
- `entity_type` (string, optional): Filter by `function`, `class`, or `variable`
- `limit` (integer, optional): Maximum number of results (default: 10)

**Example:**
```python
# Search for authentication functions
results = await client.call_tool("search_code_entities", {
    "codebase_id": "myproject",
    "query": "authentication function",
    "entity_type": "function",
    "limit": 5
})
```

#### `get_definition`
Get definition and metadata for a code entity.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `entity_name` (string, required): Name of the entity
- `file_path` (string, optional): File path for disambiguation

**Example:**
```python
results = await client.call_tool("get_definition", {
    "codebase_id": "myproject",
    "entity_name": "authenticate",
    "file_path": "src/auth.py"
})
```

#### `find_references`
Find all references/usages of an entity.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `entity_name` (string, required): Name of the entity
- `entity_type` (string, optional): Type of entity - `function`, `class`, or `variable` (default: `function`)

**Example:**
```python
results = await client.call_tool("find_references", {
    "codebase_id": "myproject",
    "entity_name": "authenticate",
    "entity_type": "function"
})
```

#### `get_call_graph`
Get call graph for a function showing what it calls and what calls it.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `function_name` (string, required): Name of the function
- `depth` (integer, optional): Maximum depth to traverse (default: 2)

**Example:**
```python
results = await client.call_tool("get_call_graph", {
    "codebase_id": "myproject",
    "function_name": "process_request",
    "depth": 3
})
```

#### `get_dependencies`
Get import/module dependencies for a file.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `file_path` (string, required): Path to the file

**Example:**
```python
results = await client.call_tool("get_dependencies", {
    "codebase_id": "myproject",
    "file_path": "src/api/routes.py"
})
```

#### `query_graph`
Execute a custom Cypher query on the knowledge graph (advanced).

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `cypher_query` (string, required): Cypher query string

**Example:**
```python
results = await client.call_tool("query_graph", {
    "codebase_id": "myproject",
    "cypher_query": "MATCH (f:Function) WHERE f.name CONTAINS 'auth' RETURN f LIMIT 10"
})
```

#### `get_context`
Get relevant context around a code location including related entities.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `file_path` (string, required): Path to the file
- `line_number` (integer, required): Line number
- `context_lines` (integer, optional): Number of context lines (default: 50)

**Example:**
```python
results = await client.call_tool("get_context", {
    "codebase_id": "myproject",
    "file_path": "src/handlers.py",
    "line_number": 42,
    "context_lines": 20
})
```

#### `submit_task`
Submit a background task for long-running operations (e.g., codebase indexing).

**Parameters:**
- `codebase_id` (string, required): ID of the codebase to operate on
- `task_type` (string, required): Type of task - `INDEX_CODEBASE`, etc.
- `params` (object, required): Task-specific parameters

**Example:**
```python
# Submit codebase indexing task (runs in background)
result = await client.call_tool("submit_task", {
    "codebase_id": "myproject",
    "task_type": "INDEX_CODEBASE",
    "params": {
        "directory": "/path/to/project",
        "languages": ["python", "javascript"]
    }
})
# Returns task ID immediately, indexing runs in background
```

#### `task_result`
View all background tasks grouped by status (pending, running, completed).

**Parameters:**
- `codebase_id` (string, required): ID of the codebase to view tasks for

**Example:**
```python
# Check task status and results
results = await client.call_tool("task_result", {
    "codebase_id": "myproject"
})
# Returns: {pending: [], running: [], completed: []}
# Running tasks show progress bars in widget
# Completed tasks show results (files indexed, entities found, etc.)
```

#### `cancel_task`
Cancel a pending or running background task.

**Parameters:**
- `codebase_id` (string, required): ID of the codebase
- `task_id` (string, required): ID of the task to cancel

**Example:**
```python
result = await client.call_tool("cancel_task", {
    "codebase_id": "myproject",
    "task_id": "task-uuid-here"
})
```

### Background Task Management

DocGraph supports long-running operations as background tasks using a thread pool executor:

**Available Task Types:**
- `INDEX_CODEBASE`: Index a codebase with entity extraction and embedding generation

**Task Workflow:**
1. Submit task with `submit_task` tool (returns immediately with task ID)
2. Check progress with `task_result` tool (shows progress bar in widget)
3. View completed results with `task_result` tool (shows statistics and outputs)
4. Cancel if needed with `cancel_task` tool

**Task Limits (per codebase):**
- Maximum 1 running task
- Maximum 2 pending tasks
- Maximum 3 completed tasks retained (most recent)

**Example Background Indexing:**
```python
# 1. Submit indexing task
submit_result = await client.call_tool("submit_task", {
    "codebase_id": "bigproject",
    "task_type": "INDEX_CODEBASE",
    "params": {
        "directory": "/large/codebase",
        "languages": ["python", "javascript", "typescript"]
    }
})

# 2. Check progress (can call multiple times)
progress = await client.call_tool("task_result", {
    "codebase_id": "bigproject"
})
# Shows: running: [{task_id, progress: 0.65, message: "Processed 650/1000 files..."}]

# 3. Get final results when complete
final = await client.call_tool("task_result", {
    "codebase_id": "bigproject"
})
# Shows: completed: [{files_indexed: 1000, entities_found: 5432, ...}]
```

### Query the Knowledge Graph

```python
from src.query.engine import QueryEngine

engine = QueryEngine()
results = engine.search_code_entities("authentication function")
```

## Data Storage

### Storage Locations

- **Neo4j Graph Database**: Local instance at `neo4j://localhost:7687` (default)
  - Database: Single database (default: `neo4j`)
  - Data Segregation: Uses `codebase_id` property on all nodes to separate codebases
- **ChromaDB Vector Database**: `~/.docgraph/chromadb/` (configurable via `CHROMADB_PERSIST_DIR`)
  - Collections: Separate collection per codebase
  - Format: `{collection_base}_{codebase_id}` (default: `code_entities_{codebase_id}`)
  - Example: `code_entities_myproject`, `code_entities_rmapp`, `custom_code_anotherproject`

### Collection Naming Convention

ChromaDB collections follow the pattern: `{collection_base}_{codebase_id}`

**Default Pattern:**
```
code_entities_default
code_entities_rmapp
code_entities_myproject
```

**Components:**
- `collection_base`: Base collection name (default: `code_entities`), set by `CHROMADB_COLLECTION` env var
- `codebase_id`: Unique identifier for each codebase, set by `DOCGRAPH_CODEBASE` env var or `--codebase-id` flag

**Examples:**
```bash
# Creates: code_entities_project1
python index_codebase.py /path/to/project1 --codebase-id project1

# Creates: custom_base_project2
$env:CHROMADB_COLLECTION="custom_base"
python index_codebase.py /path/to/project2 --codebase-id project2
```

### Codebase Segregation

Each codebase is identified by a unique `codebase_id`. This ensures complete data isolation between different codebases.

**How It Works:**

1. **Graph Database (Neo4j)**:
   - All nodes have a `codebase_id` property
   - All queries filter by `codebase_id` to ensure isolation
   - Indexes include `codebase_id` for efficient filtering

2. **Vector Database (ChromaDB)**:
   - Each codebase uses a separate collection
   - Collection name: `{collection_base}_{codebase_id}` (e.g., `code_entities_rmapp`)
   - Metadata includes `codebase_id` for additional filtering

3. **Query Engine**:
   - Automatically filters by `codebase_id` in all queries
   - Can be overridden by passing `codebase_id` parameter

### Managing Multiple Codebases

```bash
# Index first codebase
python index_codebase.py /path/to/project1 --codebase-id project1

# Index second codebase (completely separate)
python index_codebase.py /path/to/project2 --codebase-id project2

# Query specific codebase
python -m src.mcp.server project1
```

Query in Python:
```python
from src.query.engine import QueryEngine

# Query specific codebase
engine = QueryEngine(codebase_id="project1")
results = engine.search_code_entities("authentication")
```

### Managing Codebase Data

**List All Codebases:**
```cypher
MATCH (n)
RETURN DISTINCT n.codebase_id as codebase_id
ORDER BY codebase_id
```

**Delete a Codebase:**
```bash
python index_codebase.py /path/to/project --codebase-id project1 --clear
```

Or directly in Neo4j:
```cypher
MATCH (n {codebase_id: 'project1'})
DETACH DELETE n
```

**Delete ChromaDB Collection:**
```python
from src.storage.vector_db import VectorDB

vector_db = VectorDB(codebase_id="project1")
vector_db.clear()
```

### Environment Variables

- `NEO4J_URI`: Neo4j connection URI (default: `neo4j://localhost:7687`)
- `NEO4J_DATABASE`: Database name (default: `neo4j`)
- `CHROMADB_PERSIST_DIR`: ChromaDB storage directory (default: `~/.docgraph/chromadb`)
- `DOCGRAPH_DATA_DIR`: Base data directory (default: `~/.docgraph/data`)

### Best Practices

1. **Use Descriptive Codebase IDs**: Use meaningful names like `project-name` or `repo-name`
2. **Consistent Naming**: Use the same `codebase_id` when re-indexing the same codebase
3. **Clear Before Re-indexing**: Use `--clear` flag when re-indexing to avoid duplicates
4. **Backup**: Neo4j and ChromaDB data can be backed up by copying their data directories

### Building and Deploying Widgets

```bash
# Install dependencies
cd _mcp/widgets
npm install

# Development server
npm run dev

# Production build
npm run build


## License

MIT



