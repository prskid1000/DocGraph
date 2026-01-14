"""Java embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class JavaEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for Java entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for Java entity."""
        parts = [f"Java {entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        if entity.parent:
            parts.append(f"Defined in: {entity.parent}")
        
        return "\n".join(parts)
