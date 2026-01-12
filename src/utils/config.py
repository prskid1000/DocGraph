"""Configuration management for DocGraph."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_dir: Directory containing config files. Defaults to project root.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = config_dir
        self._languages_config = None
        self._load_languages_config()
    
    def _load_languages_config(self):
        """Load languages configuration from YAML."""
        config_file = self.config_dir / "languages.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._languages_config = yaml.safe_load(f)
        else:
            self._languages_config = {"languages": {}}
    
    def get_languages(self) -> Dict[str, Any]:
        """Get languages configuration.
        
        Returns:
            Dictionary of language configurations.
        """
        return self._languages_config.get("languages", {})
    
    def get_enabled_languages(self) -> Dict[str, Any]:
        """Get only enabled languages.
        
        Returns:
            Dictionary of enabled language configurations.
        """
        languages = self.get_languages()
        return {
            lang: config 
            for lang, config in languages.items() 
            if config.get("enabled", False)
        }
    
    def get_language_config(self, language: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript').
            
        Returns:
            Language configuration or None if not found.
        """
        return self.get_languages().get(language)
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j connection configuration.
        
        Returns:
            Dictionary with Neo4j connection parameters.
        """
        return {
            "uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "12891289"),
            "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        }
    
    def get_chromadb_config(self) -> Dict[str, Any]:
        """Get ChromaDB configuration.
        
        Returns:
            Dictionary with ChromaDB configuration.
        """
        return {
            "persist_directory": os.getenv(
                "CHROMADB_PERSIST_DIR", 
                str(Path.home() / ".docgraph" / "chromadb")
            ),
            "collection_name": os.getenv("CHROMADB_COLLECTION", "code_entities"),
        }
    
    def get_data_directory(self) -> Path:
        """Get base data directory for storing codebase data.
        
        Returns:
            Path to data directory.
        """
        data_dir = os.getenv(
            "DOCGRAPH_DATA_DIR",
            str(Path.home() / ".docgraph" / "data")
        )
        return Path(data_dir)
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration.
        
        Returns:
            Dictionary with embedding model configuration.
        """
        return {
            "model_name": os.getenv(
                "EMBEDDING_MODEL", 
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        }
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP server configuration.
        
        Returns:
            Dictionary with MCP server configuration.
        """
        return {
            "transport": os.getenv("MCP_TRANSPORT", "stdio"),
            "host": os.getenv("MCP_HOST", "localhost"),
            "port": int(os.getenv("MCP_PORT", "8000")),
        }


# Global config instance
config = Config()

