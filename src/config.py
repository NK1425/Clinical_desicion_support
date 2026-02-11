"""
Configuration settings for the Clinical Decision Support System
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Groq Configuration (primary LLM - free)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenAI Configuration (optional fallback)
    openai_api_key: str = ""
    model_name: str = "gpt-3.5-turbo"

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store Configuration
    vector_store_path: str = "./data/faiss_index"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # openFDA API (no key required)
    openfda_base_url: str = "https://api.fda.gov"

    # Chunk Configuration for RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/cdss.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Medical specialties for categorization
MEDICAL_SPECIALTIES = [
    "cardiology",
    "endocrinology",
    "neurology",
    "oncology",
    "pulmonology",
    "gastroenterology",
    "nephrology",
    "rheumatology",
    "infectious_disease",
    "general_medicine",
    "emergency_medicine",
    "psychiatry",
    "pediatrics",
    "geriatrics",
    "pharmacology",
]

# System prompt for clinical assistant
CLINICAL_SYSTEM_PROMPT = """You are an AI-powered Clinical Decision Support Assistant designed to help healthcare professionals.

Your role is to:
1. Analyze patient information, symptoms, and medical history
2. Provide evidence-based clinical insights and suggestions
3. Flag potential drug interactions and contraindications
4. Reference relevant medical guidelines and research
5. Support differential diagnosis considerations

IMPORTANT GUIDELINES:
- Always emphasize that final clinical decisions must be made by qualified healthcare professionals
- Cite sources when providing medical information
- Flag any critical or emergency symptoms that require immediate attention
- Consider patient-specific factors (age, comorbidities, allergies)
- Provide confidence levels for your suggestions when appropriate

You have access to:
- Medical guidelines and protocols
- Drug information from openFDA
- Patient records and history (when provided)
- Clinical research summaries

Format your responses clearly with sections for:
- Clinical Assessment
- Relevant Guidelines
- Drug Considerations
- Recommendations
- References
"""
