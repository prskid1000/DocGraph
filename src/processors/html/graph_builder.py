"""HTML graph builder."""
from typing import List, Dict, Any
import hashlib

from ..base import BaseGraphBuilder
from ...parsers.base import CodeEntity
from ...graph.schema import GraphSchema
from ..base import ScopedReference


class HTMLGraphBuilder(BaseGraphBuilder):
    """Builds graph relationships for HTML."""
    
    def build_relationships(
        self,
        entities: List[CodeEntity],
        references: List[ScopedReference],
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Build relationships for HTML."""
        relationships = []
        node_ids = {}
        
        for entity in entities:
            node_id = self._generate_node_id(entity)
            node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        
        for ref in references:
            relationships.append({
                'from_id': node_ids.get(f"variable:{file_path}:{file_path}", "file_node"),
                'to_entity': ref.to_entity,
                'type': GraphSchema.REL_IMPORTS,
                'properties': {
                    'line_number': ref.line_number
                }
            })
        
        return relationships
    
    def _generate_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID."""
        key = f"{entity.entity_type}:{entity.file_path}:{entity.name}:{entity.start_line}:{entity.start_column}"
        return hashlib.md5(key.encode()).hexdigest()
