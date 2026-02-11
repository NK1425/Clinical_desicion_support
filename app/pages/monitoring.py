"""
Streamlit Monitoring Dashboard
Real-time system health, query latency, and knowledge base statistics.
"""
import streamlit as st
import requests
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="CDSS Monitoring", page_icon="📊", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")


def fetch_health():
    """Fetch health status from API."""
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def fetch_stats():
    """Fetch system stats from API."""
    try:
        resp = requests.get(f"{API_URL}/api/stats", timeout=5)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except requests.RequestException:
        return None


def fetch_metrics():
    """Fetch Prometheus metrics and parse key values."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        resp.raise_for_status()
        return _parse_prometheus_metrics(resp.text)
    except requests.RequestException:
        return None


def _parse_prometheus_metrics(text: str) -> dict:
    """Parse Prometheus text format into a simple dict of metric name -> value."""
    metrics = {}
    for line in text.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split(" ")
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics


def main():
    st.title("📊 System Monitoring Dashboard")
    st.caption("Real-time health, performance, and knowledge base metrics")

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(0.1)
        st.rerun()

    # --- Health Status ---
    st.header("System Health")
    health = fetch_health()

    if health:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("API Status", "Online", delta="healthy")
        col2.metric("Knowledge Base", f"{health.get('knowledge_base_docs', 0)} docs")
        col3.metric("LLM Status", "Active" if health.get("llm_available") else "Offline")
        col4.metric("LLM Model", health.get("llm_model", "N/A"))
    else:
        st.error("API is not reachable. Make sure the API server is running on " + API_URL)
        st.info("Start the API with: `uvicorn api.main:app --reload`")

    st.divider()

    # --- Prometheus Metrics ---
    st.header("Performance Metrics")
    metrics = fetch_metrics()

    if metrics:
        col1, col2, col3 = st.columns(3)

        # Request counts
        total_requests = sum(v for k, v in metrics.items() if k.startswith("cdss_request_count_total"))
        col1.metric("Total Requests", int(total_requests))

        # Error counts
        total_errors = sum(v for k, v in metrics.items() if k.startswith("cdss_api_errors_total"))
        col2.metric("Total Errors", int(total_errors))

        # Knowledge base size
        kb_size = metrics.get("cdss_knowledge_base_documents_total", 0)
        col3.metric("KB Documents", int(kb_size))

        # Latency histograms (show sum/count = avg)
        st.subheader("Latency Averages")
        latency_data = {}

        req_sum = metrics.get("cdss_request_latency_seconds_sum", 0)
        req_count = metrics.get("cdss_request_latency_seconds_count", 0)
        if req_count > 0:
            latency_data["Request Avg (s)"] = round(req_sum / req_count, 3)

        rag_sum = metrics.get("cdss_rag_retrieval_latency_seconds_sum", 0)
        rag_count = metrics.get("cdss_rag_retrieval_latency_seconds_count", 0)
        if rag_count > 0:
            latency_data["RAG Retrieval Avg (s)"] = round(rag_sum / rag_count, 3)

        llm_sum = metrics.get("cdss_llm_generation_latency_seconds_sum", 0)
        llm_count = metrics.get("cdss_llm_generation_latency_seconds_count", 0)
        if llm_count > 0:
            latency_data["LLM Generation Avg (s)"] = round(llm_sum / llm_count, 3)

        if latency_data:
            cols = st.columns(len(latency_data))
            for i, (label, value) in enumerate(latency_data.items()):
                cols[i].metric(label, f"{value:.3f}s")
        else:
            st.info("No latency data yet. Make some queries to generate metrics.")
    else:
        st.info("Metrics not available. API may not be running.")

    st.divider()

    # --- Knowledge Base Stats ---
    st.header("Knowledge Base")
    stats = fetch_stats()

    if stats:
        st.json(stats)
    else:
        # Fallback: try to read from local vector store
        try:
            from src.data_ingestion import get_ingestion_stats
            from src.vector_store import get_vector_store
            vs = get_vector_store()
            local_stats = get_ingestion_stats(vs)
            st.metric("Total Documents", local_stats["total_documents"])
            st.metric("Unique Sources", local_stats["unique_sources"])

            if local_stats.get("categories"):
                st.subheader("Documents by Category")
                import pandas as pd
                cat_df = pd.DataFrame(
                    list(local_stats["categories"].items()),
                    columns=["Category", "Count"],
                ).sort_values("Count", ascending=False)
                st.bar_chart(cat_df.set_index("Category"))
        except Exception:
            st.info("Connect to API or initialize vector store to see knowledge base stats.")

    st.divider()

    # --- Quick API Test ---
    st.header("Quick API Test")
    test_query = st.text_input("Test query:", placeholder="e.g., How to manage type 2 diabetes?")
    if st.button("Send Query"):
        if test_query:
            with st.spinner("Querying..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/api/query",
                        json={"question": test_query},
                        timeout=30,
                    )
                    data = resp.json()
                    if data.get("success"):
                        st.success("Query successful!")
                        st.markdown(data["response"])
                        if data.get("timings"):
                            st.json(data["timings"])
                    else:
                        st.error(f"Query failed: {data}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
