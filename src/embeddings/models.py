"""Embedding model loading and management."""
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer
import logging

from ..utils.config import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the model. Defaults to config.
            device: Device to run on ('cpu' or 'cuda'). Defaults to config.
        """
        embedding_config = config.get_embedding_config()
        self.model_name = model_name or embedding_config['model_name']
        self.device = device or embedding_config['device']
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded")
    
    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress: bool = False) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            
        Returns:
            List of embedding vectors.
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string.
            
        Returns:
            Embedding vector.
        """
        return self.encode([text])[0]

