"""
Tests for the FAISS Vector Store.
"""
import os
import json
import tempfile
import pytest
from unittest.mock import patch


class TestVectorStoreOperations:
    """Test FAISS vector store CRUD operations."""

    def test_init_empty(self, mock_vector_store):
        assert mock_vector_store.count == 0
        assert mock_vector_store.documents == []

    def test_add_single_text(self, mock_vector_store):
        mock_vector_store.add_text("Test medical document")
        assert mock_vector_store.count == 1

    def test_add_documents(self, mock_vector_store, sample_documents):
        mock_vector_store.add_documents(sample_documents)
        assert mock_vector_store.count == 2

    def test_add_documents_with_metadata(self, mock_vector_store):
        docs = [
            {"content": "Diabetes info", "metadata": {"source": "ADA", "category": "endocrinology"}},
        ]
        mock_vector_store.add_documents(docs)
        assert mock_vector_store.documents[0]["metadata"]["source"] == "ADA"

    def test_search_returns_results(self, populated_vector_store):
        results = populated_vector_store.search("diabetes treatment", k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_result_structure(self, populated_vector_store):
        results = populated_vector_store.search("hypertension", k=1)
        assert len(results) == 1
        result = results[0]
        assert "content" in result
        assert "metadata" in result
        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_search_empty_store(self, mock_vector_store):
        results = mock_vector_store.search("anything", k=5)
        assert results == []

    def test_search_k_larger_than_store(self, mock_vector_store):
        mock_vector_store.add_text("Only document")
        results = mock_vector_store.search("document", k=10)
        assert len(results) == 1

    def test_clear(self, populated_vector_store):
        assert populated_vector_store.count > 0
        populated_vector_store.clear()
        assert populated_vector_store.count == 0
        assert populated_vector_store.documents == []

    def test_save_and_load(self, populated_vector_store, mock_embedding_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            populated_vector_store.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "index.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "documents.json"))

            # Load into new store
            with patch("src.vector_store.get_embedding_engine", return_value=mock_embedding_engine):
                from src.vector_store import VectorStore
                new_store = VectorStore(dimension=384, index_path=tmpdir)
                assert new_store.count == populated_vector_store.count

    def test_save_without_path_raises(self, mock_vector_store):
        with pytest.raises(ValueError, match="No path provided"):
            mock_vector_store.save(None)

    def test_count_property(self, mock_vector_store, sample_documents):
        assert mock_vector_store.count == 0
        mock_vector_store.add_documents(sample_documents)
        assert mock_vector_store.count == 2
