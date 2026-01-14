"""ChromaDB vector database client."""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
import logging

from ..utils.config import config

logger = logging.getLogger(__name__)


class VectorDB:
    """Client for ChromaDB vector database operations."""
    
    def __init__(self, persist_directory: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 codebase_id: Optional[str] = None):
        """Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist database. Defaults to config.
            collection_name: Collection name. Defaults to config.
            codebase_id: Codebase identifier for collection segregation.
        """
        chromadb_config = config.get_chromadb_config()
        self.persist_directory = persist_directory or chromadb_config['persist_directory']
        base_collection_name = collection_name or chromadb_config['collection_name']
        self.codebase_id = codebase_id or "default"
        # Use codebase-specific collection name
        self.collection_name = f"{base_collection_name}_{self.codebase_id}"
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.debug(f"Initialized ChromaDB collection '{self.collection_name}' at {self.persist_directory}")
    
    def add_embeddings(self, ids: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]], documents: Optional[List[str]] = None):
        """Add embeddings to the collection (upserts if IDs already exist).
        
        Args:
            ids: List of IDs (duplicates will be handled by upsert).
            embeddings: List of embedding vectors.
            metadatas: List of metadata dictionaries.
            documents: Optional list of document texts.
        """
        try:
            # Use upsert to handle both new and existing IDs
            # Upsert will add new IDs and update existing ones
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.debug(f"Upserted {len(ids)} embeddings to collection")
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 10,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the collection for similar embeddings.
        
        Args:
            query_embeddings: Query embedding vectors.
            n_results: Number of results to return.
            where: Metadata filter.
            where_document: Document filter.
            
        Returns:
            Dictionary with 'ids', 'distances', 'metadatas', 'documents'.
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            logger.error(f"Error querying embeddings: {e}")
            raise
    
    def get(self, ids: Optional[List[str]] = None,
           where: Optional[Dict[str, Any]] = None,
           limit: Optional[int] = None,
           offset: Optional[int] = None) -> Dict[str, Any]:
        """Get embeddings by IDs or filter.
        
        Args:
            ids: List of IDs to retrieve.
            where: Metadata filter.
            limit: Maximum number of results.
            offset: Offset for pagination.
            
        Returns:
            Dictionary with 'ids', 'embeddings', 'metadatas', 'documents'.
        """
        try:
            results = self.collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    def update(self, ids: List[str], embeddings: Optional[List[List[float]]] = None,
              metadatas: Optional[List[Dict[str, Any]]] = None,
              documents: Optional[List[str]] = None):
        """Update embeddings in the collection.
        
        Args:
            ids: List of IDs to update.
            embeddings: Optional new embeddings.
            metadatas: Optional new metadatas.
            documents: Optional new documents.
        """
        try:
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.debug(f"Updated {len(ids)} embeddings")
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            raise
    
    def delete(self, ids: Optional[List[str]] = None,
              where: Optional[Dict[str, Any]] = None):
        """Delete embeddings from the collection.
        
        Args:
            ids: List of IDs to delete.
            where: Metadata filter for deletion.
        """
        try:
            self.collection.delete(ids=ids, where=where)
            logger.debug(f"Deleted embeddings")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise
    
    def count(self) -> int:
        """Get the number of embeddings in the collection.
        
        Returns:
            Number of embeddings.
        """
        return self.collection.count()
    
    def clear(self):
        """Clear all embeddings from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

