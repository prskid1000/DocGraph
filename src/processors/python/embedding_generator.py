"""Python embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class PythonEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for Python entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for Python entity."""
        parts = [f"Python {entity.entity_type}: {entity.name}"]
        
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        if entity.parent:
            parts.append(f"Defined in: {entity.parent}")
        if entity.metadata:
            if 'decorators' in entity.metadata:
                parts.append(f"Decorators: {', '.join(entity.metadata['decorators'])}")
            if 'bases' in entity.metadata:
                parts.append(f"Inherits from: {', '.join(entity.metadata['bases'])}")
        
        return "\n".join(parts)
