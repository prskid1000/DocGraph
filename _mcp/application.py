"""DocGraph MCP Server - Main application file."""

from __future__ import annotations
import os
import sys
import logging
import json
from datetime import datetime
import asyncio

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
import mcp.types as types
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import pathlib
import time

from _mcp.config import Config
from _mcp.context import set_current_data, get_current_data
from _mcp.service.handlers import (
    list_tools,
    list_resources,
    list_resource_templates,
    handle_read_resource,
    handle_call_tool,
    list_prompts,
    handle_get_prompt
)
from _mcp.service.task import register_task_handlers

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize shared instances at startup
logger.info("🚀 Initializing DocGraph MCP Server")
logger.info(f"️  Neo4j: {Config.NEO4J_URI}")
logger.info(f"🔍 ChromaDB: {Config.CHROMADB_PERSIST_DIR}")

# Import and initialize shared database and model instances
from src.storage.neo4j_client import Neo4jClient
from src.storage.vector_db import VectorDB
from src.embeddings.models import EmbeddingModel
from src.embeddings.generator import EmbeddingGenerator

# Initialize shared instances (loaded once at startup)
logger.info("🔧 Initializing shared components...")
shared_neo4j_client = Neo4jClient()
logger.info("✅ Neo4j client initialized")

shared_embedding_model = EmbeddingModel()
logger.info("✅ Embedding model loaded")

# Cache for VectorDB instances by codebase_id
vector_db_cache = {}

def get_vector_db(codebase_id: str) -> VectorDB:
    """Get or create a cached VectorDB instance for a codebase."""
    if codebase_id not in vector_db_cache:
        vector_db_cache[codebase_id] = VectorDB(codebase_id=codebase_id)
        logger.debug(f"Created VectorDB instance for codebase: {codebase_id}")
    return vector_db_cache[codebase_id]

# Store shared instances for access by handlers
shared_instances = {
    'neo4j_client': shared_neo4j_client,
    'embedding_model': shared_embedding_model,
    'get_vector_db': get_vector_db,
}

# Make shared instances available to handlers
from _mcp.service import tools as tools_module
tools_module.shared_instances = shared_instances

mcp = FastMCP(
    name="docgraph",
    sse_path="/mcp",
    message_path="/mcp/messages",
    stateless_http=True,
    instructions="This is an MCP server for the DocGraph codebase knowledge graph. It provides tools to search, analyze, and understand code structure across multiple languages."
)

logger.info("✅ FastMCP initialized")

# Register task handlers
logger.info("📝 Registering task handlers...")
register_task_handlers()
logger.info("✅ Task handlers registered")


# Register the handlers with the MCP server
@mcp._mcp_server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """List all available tools."""
    return await list_tools()


@mcp._mcp_server.list_resources()
async def _list_resources() -> list[types.Resource]:
    """List all available resources."""
    return await list_resources()


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> list[types.ResourceTemplate]:
    """List all available resource templates."""
    return await list_resource_templates()


@mcp._mcp_server.list_prompts()
async def _list_prompts() -> list[types.Prompt]:
    """List all available prompts."""
    return await list_prompts()


# Register request handlers
mcp._mcp_server.request_handlers[types.CallToolRequest] = handle_call_tool
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = handle_read_resource
mcp._mcp_server.request_handlers[types.GetPromptRequest] = handle_get_prompt


# Get the Starlette application
application = mcp.streamable_http_app()


# ============================================================================
# MIDDLEWARE
# ============================================================================
class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to set request context for request ID and metadata."""
    
    async def dispatch(self, request: Request, call_next):
        """Set context for each request."""
        # Generate request ID
        request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Set context for this request
        set_current_data({
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process request
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            raise
        finally:
            # Clear context
            from _mcp.context import clear_current_data
            clear_current_data()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        # Skip logging for health check
        if request.url.path == "/":
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"[{get_current_data('request_id')}] "
                f"{request.method} {request.url.path} - "
                f"{response.status_code} ({process_time:.3f}s)"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = get_current_data('request_id') or ""
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{get_current_data('request_id')}] "
                f"{request.method} {request.url.path} - "
                f"Error ({process_time:.3f}s): {str(e)}"
            )
            raise


# Add middlewares (order matters - last added = first executed)
# 1. CORS (outermost)
try:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    logger.debug("✅ CORS middleware added")
except Exception as e:
    logger.warning(f"⚠️  Could not add CORS middleware: {e}")

# 2. Logging
application.add_middleware(RequestLoggingMiddleware)

# 3. Context
application.add_middleware(RequestContextMiddleware)

logger.debug("✅ Middlewares configured")


# ============================================================================
# ROUTES
# ============================================================================
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "docgraph-mcp",
        "environment": Config.ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    })


async def info_endpoint(request: Request) -> JSONResponse:
    """Server info endpoint."""
    return JSONResponse({
        "name": "DocGraph MCP Server",
        "version": "1.0.0",
        "service": "docgraph",
        "description": "Codebase knowledge graph MCP server",
        "mcp_enabled": True,
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "mcp_sse": "/mcp",
            "mcp_messages": "/mcp/messages",
            "manifest": "/.well-known/mcp.json"
        }
    })


async def mcp_manifest(request: Request) -> JSONResponse:
    """MCP manifest endpoint."""
    manifest = {
        "mcpServers": {
            "docgraph": {
                "command": "python",
                "args": ["-m", "mcp.application"],
                "env": {
                    "DOCGRAPH_HOST": Config.HOST,
                    "DOCGRAPH_PORT": str(Config.PORT),
                    "NEO4J_URI": Config.NEO4J_URI,
                    "CHROMADB_PERSIST_DIR": Config.CHROMADB_PERSIST_DIR
                }
            }
        }
    }
    return JSONResponse(manifest)


# Widget assets endpoint - serve compiled widget files
widgets_assets_path = pathlib.Path(__file__).parent / "widgets-assets"
if widgets_assets_path.exists():
    try:
        application.routes.append(
            Mount("/widgets-assets", app=StaticFiles(directory=str(widgets_assets_path)), name="widgets-assets")
        )
        logger.info(f"✅ Widget assets served from: {widgets_assets_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not mount widget assets: {e}")
else:
    logger.warning(f"⚠️  Widget assets directory not found: {widgets_assets_path}")

# Add routes
application.routes.append(
    Route("/health", health_check, methods=["GET"])
)
application.routes.append(
    Route("/info", info_endpoint, methods=["GET"])
)
application.routes.append(
    Route("/.well-known/mcp.json", mcp_manifest, methods=["GET"])
)

logger.info("✅ Routes configured")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting DocGraph MCP Server")
    logger.info(f"📡 Host: {Config.HOST}")
    logger.info(f"🔌 Port: {Config.PORT}")
    logger.info(f"💡 Visit http://{Config.HOST}:{Config.PORT}/info for server info")
    logger.info(f"💡 MCP endpoint: http://{Config.HOST}:{Config.PORT}/mcp")
    
    # Run with uvicorn
    uvicorn.run(
        application,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        access_log=False  # We have our own logging middleware
    )
