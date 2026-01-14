"""Python graph builder."""
from typing import List, Dict, Any
import hashlib

from ..base import BaseGraphBuilder
from ...parsers.base import CodeEntity
from ...graph.schema import GraphSchema
from ..base import ScopedReference


class PythonGraphBuilder(BaseGraphBuilder):
    """Builds graph relationships for Python code."""
    
    def build_relationships(
        self,
        entities: List[CodeEntity],
        references: List[ScopedReference],
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Build relationships for Python entities and references."""
        relationships = []
        node_ids = {}
        
        # Generate node IDs for entities
        for entity in entities:
            node_id = self._generate_node_id(entity)
            node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        
        # Build relationships from references
        for ref in references:
            # Find source entity
            source_key = f"function:{ref.from_entity}:{ref.file_path}"
            if source_key not in node_ids:
                source_key = f"class:{ref.from_entity}:{ref.file_path}"
            if source_key not in node_ids:
                continue
            
            source_id = node_ids[source_key]
            
            # For now, we'll resolve target in the main graph builder
            # This is a placeholder - actual resolution happens later
            relationships.append({
                'from_id': source_id,
                'to_entity': ref.to_entity,
                'type': self._map_reference_type(ref.reference_type),
                'properties': {
                    'line_number': ref.line_number,
                    'scope': ref.scope,
                    'qualified_name': ref.qualified_name
                }
            })
        
        return relationships
    
    def _generate_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID."""
        key = f"{entity.entity_type}:{entity.file_path}:{entity.name}:{entity.start_line}:{entity.start_column}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _map_reference_type(self, ref_type: str) -> str:
        """Map reference type to graph relationship type."""
        mapping = {
            'calls': GraphSchema.REL_CALLS,
            'references': GraphSchema.REL_REFERENCES,
            'imports': GraphSchema.REL_IMPORTS,
            'inherits': GraphSchema.REL_INHERITS,
        }
        return mapping.get(ref_type, GraphSchema.REL_REFERENCES)
