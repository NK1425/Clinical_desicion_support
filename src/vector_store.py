"""
Vector Store Module
Handles FAISS vector database operations for RAG pipeline.
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional

from .embeddings import get_embedding_engine
from .logging_config import get_logger, timed

log = get_logger("vector_store")


class VectorStore:
    """FAISS-based vector store for medical knowledge retrieval."""

    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path
        self.embedding_engine = get_embedding_engine()
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict] = []

        if index_path and os.path.exists(index_path):
            self.load(index_path)

    @timed(name="vector_store.add_documents")
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store."""
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_engine.embed_texts(texts)
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(documents)
        log.info(f"Added {len(documents)} documents. Total: {self.index.ntotal}")

    def add_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add a single text to the vector store."""
        doc = {"content": text, "metadata": metadata or {}}
        self.add_documents([doc])

    @timed(name="vector_store.search")
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_engine.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx].get("metadata", {}),
                    "score": float(1 / (1 + dist)),
                }
                results.append(result)

        return results

    def save(self, path: Optional[str] = None) -> None:
        """Save the index and documents to disk."""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path provided for saving")

        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))

        with open(os.path.join(save_path, "documents.json"), "w") as f:
            json.dump(self.documents, f)

        log.info(f"Saved index ({self.index.ntotal} vectors) to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load the index and documents from disk."""
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No path provided for loading")

        index_file = os.path.join(load_path, "index.faiss")
        docs_file = os.path.join(load_path, "documents.json")

        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)

        if os.path.exists(docs_file):
            with open(docs_file, "r") as f:
                self.documents = json.load(f)

        log.info(f"Loaded index with {self.index.ntotal} vectors")

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        log.info("Vector store cleared")

    @property
    def count(self) -> int:
        """Get total number of documents."""
        return self.index.ntotal


# Singleton instance
_vector_store = None


def get_vector_store(index_path: str = "./data/faiss_index") -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(index_path=index_path)
    return _vector_store
