"""HTML embedding generator."""
from ..base import BaseEmbeddingGenerator
from ...parsers.base import CodeEntity


class HTMLEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generates embeddings for HTML entities."""
    
    def generate_embedding_text(self, entity: CodeEntity) -> str:
        """Generate text representation for HTML entity."""
        parts = [f"HTML {entity.entity_type}: {entity.name}"]
        
        if entity.metadata and 'html_type' in entity.metadata:
            parts.append(f"Type: {entity.metadata['html_type']}")
        
        return "\n".join(parts)
