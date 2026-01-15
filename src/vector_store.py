"""
Vector Store Module
Handles FAISS vector database operations for RAG pipeline
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from .embeddings import get_embedding_engine


class VectorStore:
    """FAISS-based vector store for medical knowledge retrieval"""
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize the vector store
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            index_path: Path to save/load the index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.embedding_engine = get_embedding_engine()
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Metadata storage (maps index position to document info)
        self.documents: List[Dict] = []
        
        # Load existing index if path provided and exists
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata' keys
        """
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_engine.embed_texts(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store document metadata
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def add_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a single text to the vector store
        
        Args:
            text: Text content to add
            metadata: Optional metadata dict
        """
        doc = {'content': text, 'metadata': metadata or {}}
        self.add_documents([doc])
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_engine.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    'content': self.documents[idx]['content'],
                    'metadata': self.documents[idx].get('metadata', {}),
                    'score': float(1 / (1 + dist))  # Convert distance to similarity score
                }
                results.append(result)
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the index and documents to disk"""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path provided for saving")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        
        # Save documents metadata
        with open(os.path.join(save_path, "documents.json"), 'w') as f:
            json.dump(self.documents, f)
        
        print(f"Saved index to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the index and documents from disk"""
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No path provided for loading")
        
        index_file = os.path.join(load_path, "index.faiss")
        docs_file = os.path.join(load_path, "documents.json")
        
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            
        if os.path.exists(docs_file):
            with open(docs_file, 'r') as f:
                self.documents = json.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def clear(self) -> None:
        """Clear all documents from the store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print("Vector store cleared")
    
    @property
    def count(self) -> int:
        """Get total number of documents"""
        return self.index.ntotal


# Singleton instance
_vector_store = None

def get_vector_store(index_path: str = "./data/faiss_index") -> VectorStore:
    """Get or create the vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(index_path=index_path)
    return _vector_store
