"""
Tests for Medical APIs (openFDA integration).
"""
import pytest
from unittest.mock import patch, MagicMock
import requests

from src.medical_apis import OpenFDAClient, MedicalDataAggregator


class TestOpenFDAClient:
    """Test openFDA API client with mocked HTTP responses."""

    def setup_method(self):
        self.client = OpenFDAClient()

    @patch.object(requests.Session, "get")
    def test_search_drug_success(self, mock_get, mock_fda_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_fda_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.search_drug("metformin", limit=1)
        assert "results" in result
        assert len(result["results"]) == 1

    @patch.object(requests.Session, "get")
    def test_search_drug_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Timed out")
        result = self.client.search_drug("metformin")
        assert "error" in result
        assert "timed out" in result["error"].lower()

    @patch.object(requests.Session, "get")
    def test_search_drug_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        result = self.client.search_drug("metformin")
        assert "error" in result

    @patch.object(requests.Session, "get")
    def test_get_drug_interactions_success(self, mock_get, mock_fda_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_fda_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.get_drug_interactions("metformin")
        assert result["drug_name"] == "metformin"
        assert "drug_interactions" in result

    @patch.object(requests.Session, "get")
    def test_get_drug_interactions_no_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.get_drug_interactions("unknowndrug")
        assert result["drug_name"] == "unknowndrug"
        assert "message" in result

    @patch.object(requests.Session, "get")
    def test_get_adverse_events_success(self, mock_get, mock_adverse_events_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_adverse_events_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.get_adverse_events("metformin", limit=5)
        assert result["drug_name"] == "metformin"
        assert "adverse_events" in result
        assert len(result["adverse_events"]) > 0
        assert "Nausea" in result["adverse_events"]

    @patch.object(requests.Session, "get")
    def test_get_adverse_events_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        result = self.client.get_adverse_events("metformin")
        assert "error" in result
        assert result["adverse_events"] == []

    @patch.object(requests.Session, "get")
    def test_get_drug_info_summary(self, mock_get, mock_fda_response, mock_adverse_events_response):
        mock_resp = MagicMock()
        # search_drug, get_drug_interactions, get_adverse_events all use session.get
        mock_resp.json.side_effect = [
            mock_fda_response,  # search_drug
            mock_fda_response,  # get_drug_interactions
            mock_adverse_events_response,  # get_adverse_events
        ]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.get_drug_info_summary("metformin")
        assert result["drug_name"] == "metformin"
        assert result["found"] is True
        assert len(result["indications"]) > 0


class TestMedicalDataAggregator:
    """Test medical data aggregation."""

    @patch.object(OpenFDAClient, "get_drug_info_summary")
    def test_comprehensive_report_single_drug(self, mock_summary):
        mock_summary.return_value = {
            "drug_name": "aspirin",
            "found": True,
            "indications": ["Pain relief"],
            "warnings": ["GI bleeding risk"],
            "contraindications": [],
            "interactions": [],
            "common_adverse_events": ["GI upset"],
            "dosage": [],
        }

        aggregator = MedicalDataAggregator()
        report = aggregator.get_comprehensive_drug_report(["aspirin"])
        assert "aspirin" in report["drugs"]
        assert report["drugs"]["aspirin"]["found"] is True

    @patch.object(OpenFDAClient, "get_drug_info_summary")
    @patch.object(OpenFDAClient, "get_drug_interactions")
    def test_comprehensive_report_multi_drug(self, mock_interactions, mock_summary):
        mock_summary.return_value = {
            "drug_name": "test",
            "found": True,
            "indications": [],
            "warnings": [],
            "contraindications": [],
            "interactions": [],
            "common_adverse_events": [],
            "dosage": [],
        }
        mock_interactions.return_value = {
            "drug_name": "aspirin",
            "drug_interactions": ["May interact with warfarin"],
        }

        aggregator = MedicalDataAggregator()
        report = aggregator.get_comprehensive_drug_report(["aspirin", "warfarin"])
        assert "aspirin" in report["drugs"]
        assert "warfarin" in report["drugs"]
        assert "potential_interactions" in report
