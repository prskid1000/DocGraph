"""JavaScript embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class JavaScriptEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for JavaScript entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for JavaScript entity."""
        parts = [f"JavaScript {entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        if entity.parent:
            parts.append(f"Defined in: {entity.parent}")
        
        return "\n".join(parts)
