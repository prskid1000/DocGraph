"""SCSS embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class SCSSEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for SCSS entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for SCSS entity."""
        parts = [f"SCSS {entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        
        return "\n".join(parts)
