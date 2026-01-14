"""TypeScript embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class TypeScriptEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for TypeScript entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for TypeScript entity."""
        parts = [f"TypeScript {entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        if entity.parent:
            parts.append(f"Defined in: {entity.parent}")
        if entity.metadata and entity.metadata.get('is_interface'):
            parts.append("Type: interface")
        
        return "\n".join(parts)
