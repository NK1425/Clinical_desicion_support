"""
Text Embeddings Module
Handles conversion of text to vector embeddings using sentence-transformers
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingEngine:
    """Handles text embedding operations using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: len(texts) x embedding_dimension)
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dimension


# Singleton instance for reuse
_embedding_engine = None

def get_embedding_engine(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingEngine:
    """Get or create the embedding engine singleton"""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine(model_name)
    return _embedding_engine
