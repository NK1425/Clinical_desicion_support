"""
Shared test fixtures for Clinical Decision Support System tests.
"""
import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_embedding_engine():
    """Mock embedding engine that returns deterministic vectors."""
    engine = MagicMock()
    engine.embedding_dimension = 384
    engine.get_dimension.return_value = 384

    def _embed_text(text):
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(384).astype("float32")

    def _embed_texts(texts, **kwargs):
        return np.array([_embed_text(t) for t in texts]).astype("float32")

    engine.embed_text = _embed_text
    engine.embed_texts = _embed_texts
    return engine


@pytest.fixture
def mock_vector_store(mock_embedding_engine):
    """Create a VectorStore with mocked embeddings."""
    with patch("src.vector_store.get_embedding_engine", return_value=mock_embedding_engine):
        from src.vector_store import VectorStore
        store = VectorStore(dimension=384)
        return store


@pytest.fixture
def populated_vector_store(mock_vector_store):
    """VectorStore pre-populated with test documents."""
    docs = [
        {
            "content": "Metformin is the first-line treatment for type 2 diabetes. Target HbA1c less than 7%.",
            "metadata": {"source": "ADA Guidelines", "category": "endocrinology"},
        },
        {
            "content": "Lisinopril is an ACE inhibitor used for hypertension. Target BP less than 130/80.",
            "metadata": {"source": "ACC/AHA Guidelines", "category": "cardiology"},
        },
        {
            "content": "Sepsis Hour-1 Bundle: blood cultures, broad-spectrum antibiotics, lactate, 30mL/kg fluids.",
            "metadata": {"source": "Surviving Sepsis Campaign", "category": "critical care"},
        },
        {
            "content": "Aspirin 325mg and P2Y12 inhibitor for acute coronary syndrome. Door-to-balloon under 90 minutes.",
            "metadata": {"source": "ACS Guidelines", "category": "cardiology"},
        },
        {
            "content": "COPD exacerbation: bronchodilators, prednisone 40mg x 5 days, antibiotics if purulent sputum.",
            "metadata": {"source": "GOLD Guidelines", "category": "pulmonology"},
        },
    ]
    mock_vector_store.add_documents(docs)
    return mock_vector_store


@pytest.fixture
def mock_llm_handler():
    """Mock LLM handler that returns a fixed response."""
    handler = MagicMock()
    handler.is_available.return_value = True
    handler.model = "mock-model"
    handler.active_provider = "mock"
    handler.generate_response.return_value = (
        "## Clinical Assessment\n"
        "Based on the provided information, the recommended approach is...\n\n"
        "## Recommendations\n"
        "1. Follow evidence-based guidelines\n"
        "2. Monitor patient response\n\n"
        "**Disclaimer:** Consult qualified healthcare professionals."
    )
    return handler


@pytest.fixture
def mock_fda_response():
    """Mock openFDA API response for drug search."""
    return {
        "results": [
            {
                "indications_and_usage": ["Treatment of type 2 diabetes"],
                "dosage_and_administration": ["500mg twice daily"],
                "warnings": ["Risk of lactic acidosis in renal impairment"],
                "contraindications": ["eGFR less than 30 mL/min"],
                "drug_interactions": ["May interact with carbonic anhydrase inhibitors"],
                "openfda": {
                    "brand_name": ["Glucophage"],
                    "generic_name": ["metformin hydrochloride"],
                },
            }
        ],
        "meta": {"results": {"total": 1}},
    }


@pytest.fixture
def mock_adverse_events_response():
    """Mock openFDA adverse events response."""
    return {
        "results": [
            {
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "Nausea"},
                        {"reactionmeddrapt": "Diarrhea"},
                    ]
                }
            },
            {
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "Lactic acidosis"},
                    ]
                }
            },
        ],
        "meta": {"results": {"total": 100}},
    }


@pytest.fixture
def sample_documents():
    """Sample medical documents for testing."""
    return [
        {
            "content": "Type 2 diabetes requires lifestyle modifications and metformin as first-line therapy.",
            "metadata": {"source": "test", "category": "endocrinology"},
        },
        {
            "content": "Hypertension management includes ACE inhibitors and lifestyle changes.",
            "metadata": {"source": "test", "category": "cardiology"},
        },
    ]
