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
        """Build relationships for HTML, supporting CONTAINS, CALLS, REFERENCES, IMPORTS."""
        relationships = []
        node_ids = {}
        for entity in entities:
            node_id = self._generate_node_id(entity)
            node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        # Default node id for file
        file_node_id = node_ids.get(f"file:{file_path}:{file_path}", "file_node")
        for ref in references:
            rel_type = ref.reference_type.lower()
            if rel_type == "imports":
                rel_schema = GraphSchema.REL_IMPORTS
            elif rel_type == "contains":
                rel_schema = GraphSchema.REL_CONTAINS
            elif rel_type == "calls":
                rel_schema = GraphSchema.REL_CALLS
            elif rel_type == "references":
                rel_schema = GraphSchema.REL_REFERENCES
            else:
                rel_schema = GraphSchema.REL_IMPORTS  # fallback
            relationships.append({
                'from_id': file_node_id,
                'to_entity': ref.to_entity,
                'type': rel_schema,
                'properties': {
                    'line_number': ref.line_number
                }
            })
        return relationships
    
    def _generate_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID."""
        key = f"{entity.entity_type}:{entity.file_path}:{entity.name}:{entity.start_line}:{entity.start_column}"
        return hashlib.md5(key.encode()).hexdigest()
