"""Graph builder for constructing Neo4j knowledge graph."""
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import hashlib
import logging

from ..parsers.base import CodeEntity, Reference
from .schema import GraphSchema
from .queries import GraphQueries

logger = logging.getLogger(__name__)

# Neo4j index property size limit (conservative to avoid indexing errors)
MAX_INDEXED_PROPERTY_SIZE = 5000


def _truncate_signature(signature: Optional[str]) -> Optional[str]:
    """Truncate signature if it exceeds Neo4j index limits.
    
    Args:
        signature: Function/method signature.
        
    Returns:
        Truncated signature or original if within limits.
    """
    if not signature or len(signature) <= MAX_INDEXED_PROPERTY_SIZE:
        return signature
    return signature[:MAX_INDEXED_PROPERTY_SIZE] + "..."


class GraphBuilder:
    """Builds Neo4j knowledge graph from extracted entities and references."""
    
    def __init__(self, neo4j_client=None, codebase_id: Optional[str] = None):
        """Initialize graph builder.
        
        Args:
            neo4j_client: Neo4j client instance (will be created if None).
            codebase_id: Unique identifier for the codebase being indexed.
        """
        self.neo4j_client = neo4j_client
        self.codebase_id = codebase_id or "default"
        self.entities: List[CodeEntity] = []
        self.references: List[Reference] = []
        self.node_ids: Dict[str, str] = {}  # Map entity to node ID
        self.pending_nodes: List[Dict[str, Any]] = []
        self.pending_relationships: List[Dict[str, Any]] = []
    
    def add_entities(self, entities: List[CodeEntity]):
        """Add entities to be inserted into the graph.
        
        Args:
            entities: List of code entities.
        """
        self.entities.extend(entities)
    
    def add_references(self, references: List[Reference]):
        """Add references to be inserted into the graph.
        
        Args:
            references: List of references.
        """
        self.references.extend(references)
    
    def add_references_from_resolved(self, resolved_references: List[tuple]):
        """Add resolved references (reference, target_entity tuples).
        
        Args:
            resolved_references: List of (Reference, CodeEntity) tuples.
        """
        self.resolved_references = resolved_references
    
    def _generate_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID for an entity.
        
        Args:
            entity: Code entity.
            
        Returns:
            Unique node ID string.
        """
        # Create unique ID based on entity properties, including column to handle multiple entities per line
        key = f"{entity.entity_type}:{entity.file_path}:{entity.name}:{entity.start_line}:{entity.start_column}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _entity_to_node(self, entity: CodeEntity) -> Dict[str, Any]:
        """Convert entity to Neo4j node properties.
        
        Args:
            entity: Code entity.
            
        Returns:
            Dictionary of node properties.
        """
        node_id = self._generate_node_id(entity)
        self.node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        
        # Map entity type to Neo4j label
        label_map = {
            'class': GraphSchema.NODE_CLASS,
            'function': GraphSchema.NODE_FUNCTION,
            'variable': GraphSchema.NODE_VARIABLE,
            'module': GraphSchema.NODE_MODULE,
            'parameter': GraphSchema.NODE_PARAMETER,
            'type': GraphSchema.NODE_TYPE,
        }
        
        label = label_map.get(entity.entity_type, 'Entity')
        
        properties = {
            'id': node_id,
            'name': entity.name,
            'file_path': entity.file_path,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'start_column': entity.start_column,
            'end_column': entity.end_column,
            'codebase_id': self.codebase_id,
        }
        
        if entity.signature:
            properties['signature'] = _truncate_signature(entity.signature)
        if entity.docstring:
            properties['docstring'] = entity.docstring
        if entity.parent:
            properties['parent'] = entity.parent
        if entity.metadata:
            properties.update(entity.metadata)
        
        return {
            'id': node_id,
            'label': label,
            'properties': properties
        }
    
    def _create_file_node(self, file_path: str, language: str) -> Dict[str, Any]:
        """Create a file node.
        
        Args:
            file_path: Path to the file.
            language: Programming language.
            
        Returns:
            File node dictionary.
        """
        path_obj = Path(file_path)
        file_id = hashlib.md5(file_path.encode()).hexdigest()
        
        return {
            'id': file_id,
            'label': GraphSchema.NODE_FILE,
            'properties': {
                'id': file_id,
                'path': file_path,
                'name': path_obj.name,
                'language': language,
                'size': path_obj.stat().st_size if path_obj.exists() else 0,
                'codebase_id': self.codebase_id,
            }
        }
    
    def _reference_to_relationship(self, ref: Reference, 
                                    resolved_target: Optional[CodeEntity] = None) -> Optional[Dict[str, Any]]:
        """Convert reference to Neo4j relationship.
        
        Args:
            ref: Reference object.
            resolved_target: Optional resolved target entity.
            
        Returns:
            Relationship dictionary or None if cannot be created.
        """
        # Map reference type to relationship type
        rel_map = {
            'calls': GraphSchema.REL_CALLS,
            'references': GraphSchema.REL_REFERENCES,
            'imports': GraphSchema.REL_IMPORTS,
            'inherits': GraphSchema.REL_INHERITS,
        }
        
        rel_type = rel_map.get(ref.reference_type)
        if not rel_type:
            return None
        
        # Find source and target node IDs
        source_id = None
        target_id = None
        
        # Source: find entity making the reference
        if ref.from_entity:
            # Try different entity types for source
            for entity_type in ['function', 'class', 'variable']:
                source_key = f"{entity_type}:{ref.from_entity}:{ref.file_path}"
                source_id = self.node_ids.get(source_key)
                if source_id:
                    break
        
        # Target: use resolved entity
        if resolved_target:
            target_key = f"{resolved_target.entity_type}:{resolved_target.name}:{resolved_target.file_path}"
            target_id = self.node_ids.get(target_key)
        
        if not source_id or not target_id:
            return None
        
        properties = {
            'line_number': ref.line_number,
            'file_path': ref.file_path,
        }
        if ref.metadata:
            properties.update(ref.metadata)
        
        return {
            'from_id': source_id,
            'to_id': target_id,
            'type': rel_type,
            'properties': properties
        }
    
    def build(self, batch_size: int = 1000, clear_existing: bool = False):
        """Build the graph by inserting nodes and relationships.
        
        Args:
            batch_size: Number of nodes/relationships to insert per batch.
            clear_existing: If True, delete existing nodes for this codebase first.
        """
        if not self.neo4j_client:
            from ..storage.neo4j_client import Neo4jClient
            self.neo4j_client = Neo4jClient()
        
        # Clear existing data for this codebase if requested
        if clear_existing:
            self._clear_codebase_data()
        
        # Create indexes
        self._create_indexes()
        
        # Group entities by file
        entities_by_file = defaultdict(list)
        for entity in self.entities:
            entities_by_file[entity.file_path].append(entity)
        
        # Create file nodes and entity nodes
        for file_path, file_entities in entities_by_file.items():
            # Determine language from file extension
            language = Path(file_path).suffix[1:] if Path(file_path).suffix else 'unknown'
            
            # Create file node
            file_node = self._create_file_node(file_path, language)
            self.pending_nodes.append(file_node)
            
            # Create entity nodes
            for entity in file_entities:
                node = self._entity_to_node(entity)
                self.pending_nodes.append(node)
                
                # Create DEFINES relationship
                self.pending_relationships.append({
                    'from_id': file_node['id'],
                    'to_id': node['id'],
                    'type': GraphSchema.REL_DEFINES,
                    'properties': {}
                })
        
        # Batch insert nodes
        self._batch_insert_nodes(batch_size)
        
        # Create relationships from resolved references
        if hasattr(self, 'resolved_references') and self.resolved_references:
            logger.info(f"Creating relationships from {len(self.resolved_references)} resolved references...")
            total_resolved = 0
            for ref, target in self.resolved_references:
                rel = self._reference_to_relationship(ref, target)
                if rel:
                    self.pending_relationships.append(rel)
                    total_resolved += 1
            logger.info(f"Created {total_resolved} relationships")
        else:
            logger.warning("No resolved references provided. Relationships will not be created.")
        
        # Batch insert relationships
        self._batch_insert_relationships(batch_size)
    
    def _create_indexes(self):
        """Create indexes in Neo4j."""
        queries = GraphSchema.get_create_index_queries()
        for query in queries:
            try:
                self.neo4j_client.execute_query(query)
            except Exception as e:
                # Index might already exist
                pass
    
    def _batch_insert_nodes(self, batch_size: int):
        """Insert nodes in batches.
        
        Args:
            batch_size: Number of nodes per batch.
        """
        for i in range(0, len(self.pending_nodes), batch_size):
            batch = self.pending_nodes[i:i + batch_size]
            self._insert_node_batch(batch)
    
    def _insert_node_batch(self, nodes: List[Dict[str, Any]]):
        """Insert a batch of nodes.
        
        Args:
            nodes: List of node dictionaries.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (n {id: node.id})
        SET n = node
        SET n:`%s`
        """
        
        # Group by label
        nodes_by_label = defaultdict(list)
        for node in nodes:
            nodes_by_label[node['label']].append(node)
        
        for label, label_nodes in nodes_by_label.items():
            formatted_query = query % label
            self.neo4j_client.execute_query(
                formatted_query,
                parameters={'nodes': [n['properties'] for n in label_nodes]}
            )
    
    def _batch_insert_relationships(self, batch_size: int):
        """Insert relationships in batches.
        
        Args:
            batch_size: Number of relationships per batch.
        """
        for i in range(0, len(self.pending_relationships), batch_size):
            batch = self.pending_relationships[i:i + batch_size]
            self._insert_relationship_batch(batch)
    
    def _insert_relationship_batch(self, relationships: List[Dict[str, Any]]):
        """Insert a batch of relationships.
        
        Args:
            relationships: List of relationship dictionaries.
        """
        query = """
        UNWIND $rels AS rel
        MATCH (from {id: rel.from_id})
        MATCH (to {id: rel.to_id})
        MERGE (from)-[r:`%s`]->(to)
        SET r = rel.properties
        """
        
        # Group by relationship type
        rels_by_type = defaultdict(list)
        for rel in relationships:
            rels_by_type[rel['type']].append(rel)
        
        for rel_type, type_rels in rels_by_type.items():
            formatted_query = query % rel_type
            self.neo4j_client.execute_query(
                formatted_query,
                parameters={'rels': [
                    {
                        'from_id': r['from_id'],
                        'to_id': r['to_id'],
                        'properties': r['properties']
                    }
                    for r in type_rels
                ]}
            )
    
    def _clear_codebase_data(self):
        """Clear all data for the current codebase."""
        query = """
        MATCH (n {codebase_id: $codebase_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        result = self.neo4j_client.execute_query(query, {'codebase_id': self.codebase_id})
        if result:
            logger.info(f"Cleared {result[0]['deleted']} nodes for codebase: {self.codebase_id}")
    
    def clear(self):
        """Clear pending nodes and relationships."""
        self.pending_nodes.clear()
        self.pending_relationships.clear()
        self.node_ids.clear()

