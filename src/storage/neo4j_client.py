"""Neo4j database client."""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import logging

from ..utils.config import config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j database operations."""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None):
        """Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI. Defaults to config value.
            user: Neo4j username. Defaults to config value.
            password: Neo4j password. Defaults to config value.
            database: Neo4j database name. Defaults to config value.
        """
        neo4j_config = config.get_neo4j_config()
        self.uri = uri or neo4j_config['uri']
        self.user = user or neo4j_config['user']
        self.password = password or neo4j_config['password']
        self.database = database or neo4j_config['database']
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._verify_connectivity()
    
    def _verify_connectivity(self):
        """Verify connection to Neo4j."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            query: Cypher query string.
            parameters: Query parameters.
            
        Returns:
            List of result records as dictionaries.
        """
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow graceful shutdown
            logger.warning("Query interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query[:200]}...")  # Truncate long queries
            if parameters and len(str(parameters)) < 500:
                logger.error(f"Parameters: {parameters}")
            raise
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write query (transaction).
        
        Args:
            query: Cypher query string.
            parameters: Query parameters.
            
        Returns:
            List of result records.
        """
        if parameters is None:
            parameters = {}
        
        def work(tx):
            result = tx.run(query, parameters)
            return list(result)
        
        try:
            with self.driver.session(database=self.database) as session:
                # Use execute_write for Neo4j driver 5.x (replaces write_transaction)
                result = session.execute_write(work)
                return [dict(record) for record in result]
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow graceful shutdown
            logger.warning("Write query interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error executing write query: {e}")
            logger.error(f"Query: {query[:200]}...")  # Truncate long queries
            if parameters and len(str(parameters)) < 500:
                logger.error(f"Parameters: {parameters}")
            raise
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a single node.
        
        Args:
            label: Node label.
            properties: Node properties.
            
        Returns:
            Node ID.
        """
        props_str = ', '.join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET {props_str}
        RETURN id(n) as node_id
        """
        
        result = self.execute_query(query, {'id': properties.get('id'), **properties})
        return result[0]['node_id'] if result else None
    
    def create_relationship(self, from_id: str, to_id: str, rel_type: str,
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes.
        
        Args:
            from_id: Source node ID.
            to_id: Target node ID.
            rel_type: Relationship type.
            properties: Relationship properties.
            
        Returns:
            True if successful.
        """
        if properties is None:
            properties = {}
        
        props_str = ', '.join([f"r.{k} = ${k}" for k in properties.keys()]) if properties else ""
        set_clause = f"SET {props_str}" if props_str else ""
        
        query = f"""
        MATCH (from {{id: $from_id}})
        MATCH (to {{id: $to_id}})
        MERGE (from)-[r:{rel_type}]->(to)
        {set_clause}
        RETURN r
        """
        
        params = {'from_id': from_id, 'to_id': to_id, **properties}
        result = self.execute_query(query, params)
        return len(result) > 0
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID.
            
        Returns:
            Node properties or None.
        """
        query = """
        MATCH (n {id: $node_id})
        RETURN n
        """
        
        result = self.execute_query(query, {'node_id': node_id})
        if result:
            return dict(result[0]['n'])
        return None
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships.
        
        Args:
            node_id: Node ID.
            
        Returns:
            True if successful.
        """
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        result = self.execute_query(query, {'node_id': node_id})
        return result[0]['deleted'] > 0 if result else False
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logger.warning("Database cleared")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

