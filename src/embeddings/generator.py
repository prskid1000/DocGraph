"""Embedding generator for code entities."""
from typing import List, Dict, Any, Optional
import logging

from ..parsers.base import CodeEntity
from .models import EmbeddingModel
from ..storage.vector_db import VectorDB
from ..utils.config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for code entities and stores them in vector database."""
    
    def __init__(self, model: Optional[EmbeddingModel] = None, vector_db: Optional[VectorDB] = None,
                 codebase_id: Optional[str] = None):
        """Initialize embedding generator.
        
        Args:
            model: Embedding model instance. Created if None.
            vector_db: Vector database instance. Created if None.
            codebase_id: Codebase identifier for data segregation.
        """
        self.model = model or EmbeddingModel()
        self.codebase_id = codebase_id or "default"
        self.vector_db = vector_db or VectorDB(codebase_id=self.codebase_id)
        self.batch_size = config.get_embedding_config()['batch_size']
    
    def _entity_to_text(self, entity: CodeEntity) -> str:
        """Convert entity to text for embedding.
        
        Args:
            entity: Code entity.
            
        Returns:
            Text representation of the entity.
        """
        parts = []
        
        # Entity type and name
        parts.append(f"{entity.entity_type}: {entity.name}")
        
        # Signature if available
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        
        # Docstring if available
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        
        # Parent context
        if entity.parent:
            parts.append(f"Parent: {entity.parent}")
        
        # Metadata
        if entity.metadata:
            if 'decorators' in entity.metadata:
                parts.append(f"Decorators: {', '.join(entity.metadata['decorators'])}")
        
        return "\n".join(parts)
    
    def generate_embeddings(self, entities: List[CodeEntity], batch_size: Optional[int] = None) -> List[str]:
        """Generate embeddings for entities and store in vector database.
        
        Args:
            entities: List of code entities.
            batch_size: Optional batch size override.
            
        Returns:
            List of embedding IDs.
        """
        if not entities:
            return []
        
        # Deduplicate entities by generating IDs first
        seen_ids = set()
        unique_entities = []
        for entity in entities:
            entity_id = f"{entity.entity_type}:{entity.name}:{entity.file_path}:{entity.start_line}:{entity.start_column}"
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                unique_entities.append(entity)
        
        if len(unique_entities) < len(entities):
            logger.info(f"Deduplicated {len(entities)} entities to {len(unique_entities)} unique entities")
        
        batch_size = batch_size or self.batch_size
        embedding_ids = []
        
        # Process in batches
        for i in range(0, len(unique_entities), batch_size):
            batch = unique_entities[i:i + batch_size]
            
            # Convert entities to text
            texts = [self._entity_to_text(entity) for entity in batch]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} entities)")
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress=True)
            
            # Prepare IDs and metadata
            ids = []
            metadatas = []
            documents = []
            
            for entity, embedding, text in zip(batch, embeddings, texts):
                entity_id = f"{entity.entity_type}:{entity.name}:{entity.file_path}:{entity.start_line}:{entity.start_column}"
                ids.append(entity_id)
                
                metadata = {
                    'entity_type': entity.entity_type,
                    'name': entity.name,
                    'file_path': entity.file_path,
                    'start_line': entity.start_line,
                    'end_line': entity.end_line,
                    'codebase_id': self.codebase_id,
                }
                if entity.parent:
                    metadata['parent'] = entity.parent
                if entity.signature:
                    metadata['signature'] = entity.signature
                
                metadatas.append(metadata)
                documents.append(text)
            
            # Check for duplicates within batch and deduplicate
            batch_seen = set()
            unique_indices = []
            for idx, entity_id in enumerate(ids):
                if entity_id not in batch_seen:
                    batch_seen.add(entity_id)
                    unique_indices.append(idx)
            
            if len(unique_indices) < len(ids):
                logger.warning(f"Found {len(ids) - len(unique_indices)} duplicate IDs in batch, deduplicating")
            
            # Store in vector database (upsert handles both new and existing IDs)
            if unique_indices:
                unique_ids = [ids[i] for i in unique_indices]
                unique_embeddings = [embeddings[i] for i in unique_indices]
                unique_metadatas = [metadatas[i] for i in unique_indices]
                unique_documents = [documents[i] for i in unique_indices]
                
                self.vector_db.add_embeddings(
                    ids=unique_ids,
                    embeddings=unique_embeddings,
                    metadatas=unique_metadatas,
                    documents=unique_documents
                )
                embedding_ids.extend(unique_ids)
                logger.info(f"Stored {len(unique_ids)} embeddings")
        
        return embedding_ids
    
    def search_similar(self, query: str, n_results: int = 10,
                      entity_type: Optional[str] = None,
                      file_path: Optional[str] = None,
                      codebase_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar entities.
        
        Args:
            query: Search query text.
            n_results: Number of results to return.
            entity_type: Optional filter by entity type.
            file_path: Optional filter by file path.
            
        Returns:
            List of similar entities with metadata.
        """
        # Generate query embedding
        query_embedding = self.model.encode_single(query)
        
        # Build where filter with ChromaDB logical operators
        where_conditions = []
        if entity_type:
            where_conditions.append({'entity_type': entity_type})
        if file_path:
            where_conditions.append({'file_path': file_path})
        if codebase_id:
            where_conditions.append({'codebase_id': codebase_id})
        elif self.codebase_id:
            where_conditions.append({'codebase_id': self.codebase_id})
        
        # Construct where filter based on number of conditions
        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {'$and': where_conditions}
        
        # Query vector database
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        similar_entities = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, entity_id in enumerate(results['ids'][0]):
                similar_entities.append({
                    'id': entity_id,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'document': results['documents'][0][i] if results['documents'] else None
                })
        
        return similar_entities
    
    def get_embedding(self, entity_id: str) -> Optional[List[float]]:
        """Get embedding for an entity.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            Embedding vector or None.
        """
        results = self.vector_db.get(ids=[entity_id])
        if results['embeddings'] and len(results['embeddings']) > 0:
            return results['embeddings'][0]
        return None

