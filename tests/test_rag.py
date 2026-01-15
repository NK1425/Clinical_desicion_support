"""
Unit Tests for Clinical Decision Support System
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMedicalAPIs:
    """Test openFDA API integration"""
    
    def test_fda_client_import(self):
        """Test FDA client can be imported"""
        from src.medical_apis import OpenFDAClient
        client = OpenFDAClient()
        assert client is not None
    
    def test_drug_search(self):
        """Test drug search functionality"""
        from src.medical_apis import OpenFDAClient
        client = OpenFDAClient()
        result = client.search_drug("aspirin", limit=1)
        # Should return a dict (may or may not have results depending on API)
        assert isinstance(result, dict)
    
    def test_adverse_events(self):
        """Test adverse events lookup"""
        from src.medical_apis import OpenFDAClient
        client = OpenFDAClient()
        result = client.get_adverse_events("metformin", limit=5)
        assert isinstance(result, dict)
        assert 'drug_name' in result


class TestEmbeddings:
    """Test embedding functionality"""
    
    def test_embedding_engine_import(self):
        """Test embedding engine can be imported"""
        from src.embeddings import EmbeddingEngine
        assert EmbeddingEngine is not None
    
    @pytest.mark.slow
    def test_embedding_generation(self):
        """Test embedding generation (requires model download)"""
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        embedding = engine.embed_text("Test medical text")
        assert embedding is not None
        assert len(embedding) == 384  # MiniLM dimension


class TestVectorStore:
    """Test FAISS vector store"""
    
    def test_vector_store_import(self):
        """Test vector store can be imported"""
        from src.vector_store import VectorStore
        assert VectorStore is not None
    
    @pytest.mark.slow
    def test_vector_store_operations(self):
        """Test basic vector store operations"""
        from src.vector_store import VectorStore
        store = VectorStore()
        
        # Add document
        store.add_text("Test medical document about diabetes management")
        assert store.count == 1
        
        # Search
        results = store.search("diabetes", k=1)
        assert len(results) == 1
        assert 'content' in results[0]


class TestConfig:
    """Test configuration"""
    
    def test_config_import(self):
        """Test config can be imported"""
        from src.config import settings
        assert settings is not None
    
    def test_default_values(self):
        """Test default configuration values"""
        from src.config import settings
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.chunk_size == 1000


class TestRAGPipeline:
    """Test RAG pipeline"""
    
    def test_rag_import(self):
        """Test RAG pipeline can be imported"""
        from src.rag_pipeline import RAGPipeline
        assert RAGPipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
