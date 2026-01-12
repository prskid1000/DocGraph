"""Query engine for hybrid graph + vector search."""
from typing import List, Dict, Any, Optional
import logging

from ..storage.neo4j_client import Neo4jClient
from ..storage.vector_db import VectorDB
from ..embeddings.generator import EmbeddingGenerator
from ..graph.queries import GraphQueries
from ..utils.config import config

logger = logging.getLogger(__name__)


class QueryEngine:
    """Hybrid query engine combining graph and vector search."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None,
                 vector_db: Optional[VectorDB] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 embedding_model: Optional['EmbeddingModel'] = None,
                 codebase_id: Optional[str] = None):
        """Initialize query engine.
        
        Args:
            neo4j_client: Neo4j client instance. Created if None.
            vector_db: Vector database instance. Created if None.
            embedding_generator: Embedding generator instance. Created if None.
            embedding_model: Shared embedding model instance. Used if embedding_generator is None.
            codebase_id: Codebase identifier for filtering queries.
        """
        self.codebase_id = codebase_id or "default"
        self.neo4j = neo4j_client or Neo4jClient()
        self.vector_db = vector_db or VectorDB(codebase_id=self.codebase_id)
        
        if embedding_generator:
            self.embedding_generator = embedding_generator
        else:
            # Use shared embedding model if provided, otherwise create new one
            self.embedding_generator = EmbeddingGenerator(
                model=embedding_model,
                vector_db=self.vector_db,
                codebase_id=self.codebase_id
            )
    
    def search_code_entities(self, query: str, entity_type: Optional[str] = None,
                           limit: int = 10, codebase_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for code entities using semantic search.
        
        Args:
            query: Search query text.
            entity_type: Optional filter by entity type.
            limit: Maximum number of results.
            codebase_id: Optional codebase filter (defaults to instance codebase_id).
            
        Returns:
            List of matching entities with metadata.
        """
        codebase_filter = codebase_id or self.codebase_id
        # Semantic search using vector database
        similar_entities = self.embedding_generator.search_similar(
            query=query,
            n_results=limit,
            entity_type=entity_type,
            codebase_id=codebase_filter
        )
        
        # Enrich with graph data
        enriched_results = []
        for entity in similar_entities:
            metadata = entity.get('metadata', {})
            entity_name = metadata.get('name')
            file_path = metadata.get('file_path')
            
            if entity_name and file_path:
                # Get full entity details from graph
                graph_entity = self.get_definition(entity_name, file_path)
                if graph_entity:
                    enriched_results.append({
                        **entity,
                        'graph_data': graph_entity
                    })
                else:
                    enriched_results.append(entity)
            else:
                enriched_results.append(entity)
        
        return enriched_results
    
    def get_definition(self, entity_name: str, file_path: Optional[str] = None,
                     codebase_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get definition and metadata for a code entity.
        
        Args:
            entity_name: Name of the entity.
            file_path: Optional file path for disambiguation.
            codebase_id: Optional codebase filter (defaults to instance codebase_id).
            
        Returns:
            Entity definition with metadata or None.
        """
        codebase_filter = codebase_id or self.codebase_id
        if file_path:
            query = f"""
                MATCH (n)
                WHERE n.name = $name AND n.file_path = $file_path AND n.codebase_id = $codebase_id
                RETURN n
                LIMIT 1
            """
            results = self.neo4j.execute_query(query, {
                'name': entity_name,
                'file_path': file_path,
                'codebase_id': codebase_filter
            })
        else:
            query = """
                MATCH (n)
                WHERE n.name = $name AND n.codebase_id = $codebase_id
                RETURN n
                LIMIT 10
            """
            results = self.neo4j.execute_query(query, {
                'name': entity_name,
                'codebase_id': codebase_filter
            })
        
        if results:
            node_data = dict(results[0]['n'])
            return node_data
        return None
    
    def find_references(self, entity_name: str, entity_type: str = 'function') -> List[Dict[str, Any]]:
        """Find all references/usages of an entity.
        
        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity.
            
        Returns:
            List of references with metadata.
        """
        query = GraphQueries.find_references_to_entity(entity_name, entity_type)
        results = self.neo4j.execute_query(query, {
            'entity_name': entity_name,
            'entity_type': entity_type
        })
        
        references = []
        for record in results:
            references.append({
                'source': dict(record['source']),
                'relationship': dict(record['r']),
                'target': dict(record['target'])
            })
        
        return references
    
    def get_call_graph(self, function_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get call graph for a function.
        
        Args:
            function_name: Name of the function.
            depth: Maximum depth to traverse.
            
        Returns:
            Dictionary with function, calls, and called_by.
        """
        query = GraphQueries.get_call_graph(function_name, depth)
        results = self.neo4j.execute_query(query, {'function_name': function_name})
        
        if results:
            record = results[0]
            return {
                'function': dict(record['f']),
                'calls': [dict(c) for c in record.get('calls', [])],
                'called_by': [dict(c) for c in record.get('called_by', [])]
            }
        return {'function': None, 'calls': [], 'called_by': []}
    
    def get_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """Get import/module dependencies for a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of module dependencies.
        """
        query = GraphQueries.get_dependencies(file_path)
        results = self.neo4j.execute_query(query, {'file_path': file_path})
        
        return [dict(record['m']) for record in results]
    
    def get_context(self, file_path: str, line_number: int,
                   context_lines: int = 50) -> Dict[str, Any]:
        """Get relevant context around a code location.
        
        Args:
            file_path: Path to the file.
            line_number: Line number.
            context_lines: Number of context lines.
            
        Returns:
            Dictionary with entities and related entities in context.
        """
        query = GraphQueries.get_context_around_location(file_path, line_number, context_lines)
        results = self.neo4j.execute_query(query, {
            'file_path': file_path,
            'line_number': line_number,
            'start_line': max(1, line_number - context_lines),
            'end_line': line_number + context_lines
        })
        
        entities = []
        related_entities = []
        
        for record in results:
            entities.append(dict(record['e']))
            related = record.get('related_entities', [])
            related_entities.extend([dict(r) for r in related])
        
        return {
            'file_path': file_path,
            'line_number': line_number,
            'entities': entities,
            'related_entities': list(set([str(r) for r in related_entities]))
        }
    
    def query_graph(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query.
        
        Args:
            cypher_query: Cypher query string.
            parameters: Query parameters.
            
        Returns:
            Query results.
        """
        return self.neo4j.execute_query(cypher_query, parameters or {})
    
    def hybrid_search(self, query: str, entity_type: Optional[str] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and graph traversal.
        
        Args:
            query: Search query.
            entity_type: Optional entity type filter.
            limit: Maximum results.
            
        Returns:
            List of results with both semantic and graph scores.
        """
        # Semantic search
        semantic_results = self.search_code_entities(query, entity_type, limit * 2)
        
        # Graph-based expansion
        # For each semantic result, find related entities via graph
        hybrid_results = []
        seen_ids = set()
        
        for result in semantic_results[:limit]:
            entity_id = result.get('id')
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            
            metadata = result.get('metadata', {})
            entity_name = metadata.get('name')
            
            if entity_name:
                # Get graph relationships
                references = self.find_references(entity_name, metadata.get('entity_type', 'function'))
                result['graph_references'] = references[:5]  # Limit to 5 references
            
            hybrid_results.append(result)
        
        return hybrid_results

