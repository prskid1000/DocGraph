
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.neo4j_client import Neo4jClient
from src.utils.config import config
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_all_data():
    """Clean all data from Neo4j and VectorDB regardless of codebase."""
    print("WARNING: This will delete ALL data from Neo4j and ChromaDB.")
    print("Are you sure you want to continue? (y/n)")
    response = input().lower()
    if response != 'y':
        print("Operation cancelled.")
        return

    # 1. Clean Neo4j
    logger.info("Cleaning Neo4j database...")
    try:
        neo4j_client = Neo4jClient()
        neo4j_client.clear_database()
        neo4j_client.close()
        logger.info("Neo4j database cleared.")
    except Exception as e:
        logger.error(f"Failed to clear Neo4j: {e}")

    # 2. Clean VectorDB (ChromaDB)
    logger.info("Cleaning VectorDB (ChromaDB)...")
    try:
        chromadb_config = config.get_chromadb_config()
        persist_directory = chromadb_config['persist_directory']
        
        if Path(persist_directory).exists():
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collections = client.list_collections()
            logger.info(f"Found {len(collections)} collections.")
            
            for collection in collections:
                logger.info(f"Deleting collection: {collection.name}")
                client.delete_collection(collection.name)
            
            logger.info("All ChromaDB collections deleted.")
        else:
            logger.warning(f"ChromaDB persistence directory not found: {persist_directory}")
            
    except Exception as e:
        logger.error(f"Failed to clear VectorDB: {e}")

    logger.info("Cleanup complete!")

if __name__ == "__main__":
    clean_all_data()
