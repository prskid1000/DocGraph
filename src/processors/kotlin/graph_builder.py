"""Kotlin graph builder."""
from typing import List, Dict, Any
import hashlib

from ..base import BaseGraphBuilder
from ...parsers.base import CodeEntity
from ...graph.schema import GraphSchema
from ..base import ScopedReference


class KotlinGraphBuilder(BaseGraphBuilder):
    """Builds graph relationships for Kotlin code."""
    
    def build_relationships(
        self,
        entities: List[CodeEntity],
        references: List[ScopedReference],
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Build relationships for Kotlin."""
        relationships = []
        node_ids = {}
        
        for entity in entities:
            node_id = self._generate_node_id(entity)
            node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        
        for ref in references:
            source_id = None
            
            # Handle IMPORTS - source is the file
            if ref.reference_type == 'imports':
                # For imports, from_entity is typically the file path
                if ref.from_entity == ref.file_path:
                    # Use file path hash as source ID
                    source_id = hashlib.md5(ref.file_path.encode()).hexdigest()
                else:
                    # Try to find file entity
                    file_key = f"file:{ref.from_entity}:{ref.file_path}"
                    source_id = node_ids.get(file_key)
                    if not source_id:
                        source_id = hashlib.md5(ref.file_path.encode()).hexdigest()
            else:
                # For other relationship types, try to find source entity
                # Try multiple entity types
                for entity_type in ['function', 'class', 'variable']:
                    source_key = f"{entity_type}:{ref.from_entity}:{ref.file_path}"
                    if source_key in node_ids:
                        source_id = node_ids[source_key]
                        break
                
                # If not found by exact match, try partial match
                if not source_id:
                    for key, node_id in node_ids.items():
                        if (key.startswith(f"function:{ref.from_entity}:") or 
                            key.startswith(f"class:{ref.from_entity}:") or
                            key.startswith(f"variable:{ref.from_entity}:")):
                            if ref.file_path in key:
                                source_id = node_id
                                break
                
                # If source is a file path (for top-level references), use file as source
                if not source_id and ref.from_entity == ref.file_path:
                    source_id = hashlib.md5(ref.file_path.encode()).hexdigest()
            
            # Skip if we still can't find a source
            if not source_id:
                continue
            
            relationships.append({
                'from_id': source_id,
                'to_entity': ref.to_entity,
                'type': self._map_reference_type(ref.reference_type),
                'properties': {
                    'line_number': ref.line_number,
                    'scope': ref.scope
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
