"""Configuration for DocGraph MCP Server."""

import os


class Config:
    """Configuration class for DocGraph MCP Server."""
    
    # Environment
    ENVIRONMENT = os.getenv("DOCGRAPH_ENV", "development").lower()
    
    # Server Configuration
    HOST = os.getenv("DOCGRAPH_HOST", "127.0.0.1")
    PORT = int(os.getenv("DOCGRAPH_PORT", "5500"))
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # ChromaDB Configuration
    CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "~/.docgraph/chromadb")
    CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "code_entities")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # OAuth Configuration (optional)
    OAUTH_ENABLED = os.getenv("OAUTH_ENABLED", "false").lower() == "true"
    OAUTH_ISSUER = os.getenv("OAUTH_ISSUER", None)
    OAUTH_AUDIENCE = os.getenv("OAUTH_AUDIENCE", None)
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
