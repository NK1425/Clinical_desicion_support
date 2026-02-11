"""
Tests for the RAG Pipeline — end-to-end with mocked LLM.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.rag_pipeline import RAGPipeline
from src.medical_apis import MedicalDataAggregator


class TestRAGPipeline:
    """Test RAG pipeline with mocked dependencies."""

    def test_init(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        assert pipeline.vector_store is populated_vector_store
        assert pipeline.llm_handler is mock_llm_handler

    def test_query_basic(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        result = pipeline.query("How to manage type 2 diabetes?")

        assert "response" in result
        assert "retrieved_documents" in result
        assert "query" in result
        assert "timings" in result
        assert len(result["retrieved_documents"]) > 0
        mock_llm_handler.generate_response.assert_called_once()

    def test_query_with_patient_info(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        result = pipeline.query(
            question="What medications should I consider?",
            patient_info={"age": 65, "gender": "Male", "medical_history": ["Diabetes", "Hypertension"]},
        )

        assert result["patient_info"]["age"] == 65
        call_args = mock_llm_handler.generate_response.call_args
        assert "Age: 65" in call_args.kwargs.get("query", call_args[1].get("query", ""))

    def test_query_with_medications(self, populated_vector_store, mock_llm_handler):
        aggregator = MagicMock(spec=MedicalDataAggregator)
        aggregator.get_comprehensive_drug_report.return_value = {
            "drugs": {"metformin": {"found": True, "warnings": ["Test warning"]}},
            "potential_interactions": [],
        }

        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
            medical_aggregator=aggregator,
        )
        result = pipeline.query(
            question="Check my medications",
            medications=["metformin"],
        )

        assert result["drug_information"] is not None
        assert "metformin" in result["drug_information"]["drugs"]

    def test_query_timings(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        result = pipeline.query("Test query")
        timings = result["timings"]
        assert "retrieval_seconds" in timings
        assert "llm_seconds" in timings
        assert timings["retrieval_seconds"] >= 0
        assert timings["llm_seconds"] >= 0

    def test_get_knowledge_base_stats(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        stats = pipeline.get_knowledge_base_stats()
        assert stats["total_documents"] == populated_vector_store.count
        assert stats["llm_available"] is True
        assert stats["llm_model"] == "mock-model"

    def test_extract_medications(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        text = "Patient is taking metformin 500mg and lisinopril 10mg daily"
        meds = pipeline.extract_medications(text)
        assert "Metformin" in meds
        assert "Lisinopril" in meds

    def test_extract_medications_empty(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        meds = pipeline.extract_medications("No medications mentioned here")
        assert isinstance(meds, list)

    def test_quick_drug_check(self, populated_vector_store, mock_llm_handler):
        aggregator = MagicMock(spec=MedicalDataAggregator)
        aggregator.get_comprehensive_drug_report.return_value = {
            "drugs": {},
            "potential_interactions": [],
        }

        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
            medical_aggregator=aggregator,
        )
        result = pipeline.quick_drug_check(["aspirin"])
        aggregator.get_comprehensive_drug_report.assert_called_with(["aspirin"])

    def test_format_context_empty(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        context = pipeline._format_context([])
        assert "No relevant documents" in context

    def test_format_context_with_docs(self, populated_vector_store, mock_llm_handler):
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            llm_handler=mock_llm_handler,
        )
        docs = [
            {"content": "Test content", "metadata": {"source": "Test"}, "score": 0.95},
        ]
        context = pipeline._format_context(docs)
        assert "Test content" in context
        assert "0.95" in context


class TestRAGPipelineConfig:
    """Test RAG pipeline configuration."""

    def test_config_import(self):
        from src.config import settings
        assert settings is not None
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_config_defaults(self):
        from src.config import settings
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200

    def test_clinical_system_prompt(self):
        from src.config import CLINICAL_SYSTEM_PROMPT
        assert "Clinical Decision Support" in CLINICAL_SYSTEM_PROMPT
