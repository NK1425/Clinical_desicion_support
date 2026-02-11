"""
Prometheus Metrics & Monitoring Helpers
Exposes real metrics via prometheus-client for the Clinical Decision Support API.
"""
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --- Counters ---
REQUEST_COUNT = Counter(
    "cdss_request_count_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

API_ERRORS = Counter(
    "cdss_api_errors_total",
    "Total API errors by type",
    ["error_type"],
)

# --- Histograms ---
REQUEST_LATENCY = Histogram(
    "cdss_request_latency_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "cdss_rag_retrieval_latency_seconds",
    "RAG retrieval step latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

LLM_GENERATION_LATENCY = Histogram(
    "cdss_llm_generation_latency_seconds",
    "LLM generation step latency",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# --- Gauges ---
KNOWLEDGE_BASE_DOCS = Gauge(
    "cdss_knowledge_base_documents_total",
    "Total documents in the knowledge base",
)


def get_metrics() -> bytes:
    """Generate Prometheus-format metrics output."""
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get the correct content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record an HTTP request in metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_rag_retrieval(duration: float):
    """Record RAG retrieval latency."""
    RAG_RETRIEVAL_LATENCY.observe(duration)


def record_llm_generation(duration: float):
    """Record LLM generation latency."""
    LLM_GENERATION_LATENCY.observe(duration)


def record_error(error_type: str):
    """Record an API error."""
    API_ERRORS.labels(error_type=error_type).inc()


def update_knowledge_base_size(count: int):
    """Update the knowledge base document gauge."""
    KNOWLEDGE_BASE_DOCS.set(count)
