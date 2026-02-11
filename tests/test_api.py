"""
Tests for FastAPI endpoints using httpx TestClient.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.rag_pipeline import RAGPipeline
from src.medical_apis import OpenFDAClient


@pytest.fixture
def client(populated_vector_store, mock_llm_handler):
    """Create a test client with mocked pipeline."""
    mock_aggregator = MagicMock()
    mock_aggregator.get_comprehensive_drug_report.return_value = {
        "drugs": {}, "potential_interactions": [],
    }

    mock_pipeline = RAGPipeline(
        vector_store=populated_vector_store,
        llm_handler=mock_llm_handler,
        medical_aggregator=mock_aggregator,
    )

    mock_fda = MagicMock(spec=OpenFDAClient)
    mock_fda.get_drug_info_summary.return_value = {
        "drug_name": "aspirin", "found": True,
        "indications": ["Pain"], "warnings": ["GI bleeding"],
        "contraindications": [], "interactions": [],
        "common_adverse_events": ["Nausea"], "dosage": [],
    }
    mock_fda.get_adverse_events.return_value = {
        "drug_name": "aspirin", "adverse_events": ["Nausea"], "total_reports": 50,
    }

    mock_image = MagicMock()
    mock_image.is_available.return_value = False

    with patch("api.main.get_rag_pipeline", return_value=mock_pipeline), \
         patch("api.main.get_fda_client", return_value=mock_fda), \
         patch("api.main.get_image_processor", return_value=mock_image), \
         patch("api.main.settings") as mock_settings:
        mock_settings.api_key = ""  # No auth required for tests
        # Need to reimport to pick up patches
        import importlib
        import api.main
        importlib.reload(api.main)
        yield TestClient(api.main.app)


class TestRootEndpoint:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "Clinical Decision Support" in data["message"]
        assert "version" in data


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "knowledge_base_docs" in data
        assert "llm_available" in data


class TestClinicalQueryEndpoint:
    def test_query_success(self, client):
        resp = client.post("/api/query", json={
            "question": "How to manage diabetes?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "response" in data

    def test_query_with_patient_info(self, client):
        resp = client.post("/api/query", json={
            "question": "Treatment options?",
            "patient_info": {"age": 55, "gender": "Female"},
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_query_with_medications(self, client):
        resp = client.post("/api/query", json={
            "question": "Check interactions",
            "medications": ["metformin", "lisinopril"],
        })
        assert resp.status_code == 200

    def test_query_empty_question(self, client):
        resp = client.post("/api/query", json={"question": ""})
        assert resp.status_code == 422  # Validation error

    def test_query_too_short(self, client):
        resp = client.post("/api/query", json={"question": "ab"})
        assert resp.status_code == 422


class TestDrugEndpoints:
    def test_drug_check(self, client):
        resp = client.post("/api/drug-check", json={
            "drug_names": ["aspirin"],
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_drug_info(self, client):
        resp = client.get("/api/drug/aspirin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["drug_name"] == "aspirin"

    def test_adverse_events(self, client):
        resp = client.get("/api/adverse-events/aspirin?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


class TestStatsEndpoint:
    def test_stats(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "total_documents" in data["data"]


class TestMetricsEndpoint:
    def test_prometheus_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "cdss_" in resp.text or "request" in resp.text.lower()
