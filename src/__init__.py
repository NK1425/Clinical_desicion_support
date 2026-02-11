"""
Clinical Decision Support System - Source Package
"""
from .config import settings
from .logging_config import setup_logging, get_logger
from .embeddings import EmbeddingEngine, get_embedding_engine
from .vector_store import VectorStore, get_vector_store
from .medical_apis import OpenFDAClient, MedicalDataAggregator, get_fda_client, get_medical_aggregator
from .llm_handler import LLMHandler, get_llm_handler
from .rag_pipeline import RAGPipeline, get_rag_pipeline
from .langchain_rag import LangChainRAG, get_langchain_rag

__all__ = [
    "settings",
    "setup_logging",
    "get_logger",
    "EmbeddingEngine",
    "get_embedding_engine",
    "VectorStore",
    "get_vector_store",
    "OpenFDAClient",
    "MedicalDataAggregator",
    "get_fda_client",
    "get_medical_aggregator",
    "LLMHandler",
    "get_llm_handler",
    "RAGPipeline",
    "get_rag_pipeline",
    "LangChainRAG",
    "get_langchain_rag",
]
