"""LLM integration interface for context retrieval."""
from typing import List, Dict, Any, Optional
import logging

from .engine import QueryEngine

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for LLM context retrieval and RAG-style queries."""
    
    def __init__(self, query_engine: Optional[QueryEngine] = None):
        """Initialize LLM interface.
        
        Args:
            query_engine: Query engine instance. Created if None.
        """
        self.query_engine = query_engine or QueryEngine()
    
    def get_context_for_query(self, query: str, max_results: int = 5,
                             entity_type: Optional[str] = None) -> str:
        """Get formatted context for an LLM query.
        
        Args:
            query: Natural language query.
            max_results: Maximum number of results to include.
            entity_type: Optional entity type filter.
            
        Returns:
            Formatted context string for LLM.
        """
        # Search for relevant entities
        results = self.query_engine.search_code_entities(
            query=query,
            entity_type=entity_type,
            limit=max_results
        )
        
        # Format context
        context_parts = []
        context_parts.append(f"Relevant code entities for query: '{query}'\n")
        context_parts.append("=" * 80)
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            graph_data = result.get('graph_data', {})
            
            entity_name = metadata.get('name') or graph_data.get('name', 'Unknown')
            entity_type = metadata.get('entity_type') or graph_data.get('entity_type', 'unknown')
            file_path = metadata.get('file_path') or graph_data.get('file_path', 'Unknown')
            
            context_parts.append(f"\n{i}. {entity_type.upper()}: {entity_name}")
            context_parts.append(f"   File: {file_path}")
            
            if graph_data.get('signature'):
                context_parts.append(f"   Signature: {graph_data['signature']}")
            
            if graph_data.get('docstring'):
                docstring = graph_data['docstring'].strip()
                if len(docstring) > 200:
                    docstring = docstring[:200] + "..."
                context_parts.append(f"   Documentation: {docstring}")
            
            if graph_data.get('start_line'):
                context_parts.append(f"   Location: Lines {graph_data['start_line']}-{graph_data.get('end_line', '?')}")
        
        return "\n".join(context_parts)
    
    def get_rag_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get RAG-style context with entities and relationships.
        
        Args:
            query: Query text.
            top_k: Number of top results.
            
        Returns:
            Dictionary with entities, relationships, and formatted context.
        """
        # Hybrid search
        results = self.query_engine.hybrid_search(query, limit=top_k)
        
        entities = []
        relationships = []
        
        for result in results:
            metadata = result.get('metadata', {})
            graph_data = result.get('graph_data', {})
            
            entity = {
                'name': metadata.get('name') or graph_data.get('name'),
                'type': metadata.get('entity_type') or graph_data.get('entity_type'),
                'file_path': metadata.get('file_path') or graph_data.get('file_path'),
                'signature': graph_data.get('signature'),
                'docstring': graph_data.get('docstring'),
            }
            entities.append(entity)
            
            # Add relationships
            graph_refs = result.get('graph_references', [])
            for ref in graph_refs:
                relationships.append({
                    'from': ref.get('source', {}).get('name'),
                    'to': ref.get('target', {}).get('name'),
                    'type': ref.get('relationship', {}).get('type', 'RELATED')
                })
        
        # Format context
        context = self._format_rag_context(entities, relationships)
        
        return {
            'query': query,
            'entities': entities,
            'relationships': relationships,
            'context': context,
            'count': len(entities)
        }
    
    def _format_rag_context(self, entities: List[Dict[str, Any]],
                           relationships: List[Dict[str, Any]]) -> str:
        """Format entities and relationships into context string.
        
        Args:
            entities: List of entities.
            relationships: List of relationships.
            
        Returns:
            Formatted context string.
        """
        parts = []
        parts.append("CODE CONTEXT")
        parts.append("=" * 80)
        
        # Entities section
        parts.append("\nENTITIES:")
        for entity in entities:
            parts.append(f"\n- {entity['type'].upper()}: {entity['name']}")
            if entity.get('file_path'):
                parts.append(f"  File: {entity['file_path']}")
            if entity.get('signature'):
                parts.append(f"  Signature: {entity['signature']}")
            if entity.get('docstring'):
                doc = entity['docstring'].strip()
                if len(doc) > 150:
                    doc = doc[:150] + "..."
                parts.append(f"  Docs: {doc}")
        
        # Relationships section
        if relationships:
            parts.append("\nRELATIONSHIPS:")
            for rel in relationships[:10]:  # Limit to 10 relationships
                parts.append(f"  {rel['from']} --[{rel['type']}]--> {rel['to']}")
        
        return "\n".join(parts)
    
    def get_code_snippet_context(self, file_path: str, line_number: int,
                                 context_lines: int = 50) -> str:
        """Get formatted context for a specific code location.
        
        Args:
            file_path: Path to the file.
            line_number: Line number.
            context_lines: Number of context lines.
            
        Returns:
            Formatted context string.
        """
        context_data = self.query_engine.get_context(file_path, line_number, context_lines)
        
        parts = []
        parts.append(f"CODE CONTEXT: {file_path}:{line_number}")
        parts.append("=" * 80)
        
        # Entities in context
        entities = context_data.get('entities', [])
        if entities:
            parts.append("\nEntities in context:")
            for entity in entities:
                parts.append(f"  - {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'unknown')})")
                if entity.get('signature'):
                    parts.append(f"    {entity['signature']}")
        
        # Related entities
        related = context_data.get('related_entities', [])
        if related:
            parts.append("\nRelated entities:")
            for rel in related[:5]:
                parts.append(f"  - {rel}")
        
        return "\n".join(parts)

