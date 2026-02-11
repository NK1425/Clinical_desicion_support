# Clinical Decision Support Assistant

An AI-powered clinical decision support system using RAG (Retrieval-Augmented Generation), LangChain, and real-time medical APIs to assist healthcare professionals with evidence-based clinical insights.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Architecture

```
                    +------------------+
                    |   Streamlit UI   |
                    |  (Dashboard +    |
                    |   Monitoring)    |
                    +--------+---------+
                             |
                    +--------+---------+
                    |   FastAPI API    |
                    |  Auth | Metrics  |
                    |  Rate Limiting   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------+------+ +----+----+ +-------+--------+
     | RAG Pipeline   | | LangChain| | Medical APIs  |
     | (FAISS +       | | RAG Chain| | (openFDA,     |
     |  Embeddings)   | | (Groq/  | |  PubMed)      |
     +--------+------+ | OpenAI) | +----------------+
              |         +----+----+
     +--------+------+       |
     | Knowledge Base |  +---+---+
     | 120+ Medical   |  |  LLM  |
     | Guidelines     |  | Groq  |
     | + PubMed       |  |Llama  |
     +-----------------+ +-------+
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RAG Framework | **LangChain** (`langchain`, `langchain-community`, `langchain-groq`) |
| LLM | **Llama 3.3 70B** via Groq (free) / GPT-4 via OpenAI (optional) |
| Vector Database | **FAISS** with sentence-transformers (`all-MiniLM-L6-v2`) |
| API Backend | **FastAPI** with API key auth, rate limiting, Prometheus metrics |
| Dashboard | **Streamlit** with monitoring page |
| Medical APIs | **openFDA** (drug info), **PubMed** (abstract ingestion) |
| Logging | **Loguru** (structured JSON logging with correlation IDs) |
| Testing | **pytest** with coverage, mocked dependencies |
| CI/CD | **GitHub Actions** (lint, test, Docker build) |
| Containerization | **Docker** (multi-stage) + Docker Compose |

## Features

- **RAG Pipeline**: Retrieves relevant medical guidelines from a 120+ document knowledge base using FAISS semantic search, then generates contextualized clinical responses via LLM
- **LangChain Integration**: Full LangChain RAG chain with `ChatGroq`, `FAISS` vectorstore wrapper, `RecursiveCharacterTextSplitter`, and custom clinical prompts
- **Real-time Drug Data**: Queries openFDA for drug interactions, adverse events, warnings, and contraindications
- **PubMed Ingestion**: Automated pipeline to download and index PubMed abstracts for any medical condition
- **API Security**: Header-based API key authentication and in-memory sliding window rate limiting
- **Prometheus Metrics**: Request count, latency histograms, RAG/LLM timing, knowledge base size at `/metrics`
- **Monitoring Dashboard**: Streamlit page showing real-time system health, performance metrics, and knowledge base statistics
- **RAG Evaluation**: Precision@K, MRR, and latency benchmarks against 32 curated Q&A pairs
- **Medical Image Analysis**: Optional BLIP-2 integration for medical image interpretation

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/clinical-decision-support.git
cd clinical-decision-support
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free at https://console.groq.com)
```

### 3. Initialize knowledge base

```bash
python src/init_vectorstore.py
# Optional: also ingest PubMed abstracts
python src/init_vectorstore.py --include-pubmed
```

### 4. Run

```bash
# API server
uvicorn api.main:app --reload

# Streamlit dashboard (separate terminal)
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000  |  Dashboard: http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| POST | `/api/query` | Clinical query (RAG pipeline) |
| POST | `/api/drug-check` | Multi-drug interaction check |
| GET | `/api/drug/{name}` | Drug information lookup |
| GET | `/api/adverse-events/{name}` | Adverse event reports |
| GET | `/api/stats` | Knowledge base statistics |
| GET | `/metrics` | Prometheus metrics |

## Data Pipeline

The knowledge base combines three sources:

1. **Curated Guidelines** (`data/medical_guidelines/`): 37 markdown files covering cardiology, endocrinology, neurology, pulmonology, nephrology, pharmacology, emergency medicine, psychiatry, oncology, pediatrics, and geriatrics
2. **Core Knowledge** (`src/init_vectorstore.py`): 13 inline clinical guidelines for baseline coverage
3. **PubMed Abstracts** (`src/pubmed_ingestion.py`): Automated ingestion from PubMed API with source attribution

```bash
# Ingest PubMed abstracts for specific conditions
python -m src.pubmed_ingestion --conditions "diabetes,hypertension,COPD" --max-per-condition 30
```

## Evaluation

```bash
# Run RAG evaluation
python -m src.evaluation
```

Evaluates retrieval quality against 32 Q&A pairs. Metrics:
- Precision@1, @3, @5
- Mean Reciprocal Rank (MRR)
- Retrieval latency (P50, P95, P99)

Results saved to `results/eval_latest.json`.

## Testing

```bash
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing
```

## Project Structure

```
├── src/
│   ├── config.py              # Settings (Groq, OpenAI, FAISS config)
│   ├── logging_config.py      # Loguru structured logging
│   ├── embeddings.py          # Sentence-transformers embedding engine
│   ├── vector_store.py        # FAISS vector database
│   ├── llm_handler.py         # LLM handler (Groq → OpenAI → fallback)
│   ├── rag_pipeline.py        # Core RAG pipeline with timing
│   ├── langchain_rag.py       # LangChain RAG chain (ChatGroq + FAISS)
│   ├── medical_apis.py        # openFDA API client
│   ├── data_ingestion.py      # Document chunking and indexing
│   ├── pubmed_ingestion.py    # PubMed abstract ingestion
│   ├── evaluation.py          # RAG evaluation framework
│   ├── init_vectorstore.py    # Knowledge base initialization
│   └── image_processor.py     # BLIP-2 image analysis (optional)
├── api/
│   ├── main.py                # FastAPI app (auth, rate limiting, metrics)
│   └── monitoring.py          # Prometheus metrics definitions
├── app/
│   ├── streamlit_app.py       # Main Streamlit dashboard
│   └── pages/
│       └── monitoring.py      # Monitoring dashboard page
├── data/
│   ├── medical_guidelines/    # 37 curated clinical guideline files
│   └── faiss_index/           # FAISS index + document metadata
├── tests/
│   ├── conftest.py            # Shared test fixtures
│   ├── test_rag.py            # RAG pipeline tests
│   ├── test_vector_store.py   # Vector store tests
│   ├── test_medical_apis.py   # Medical API tests (mocked)
│   ├── test_api.py            # FastAPI endpoint tests
│   └── evaluation_data.json   # 32 evaluation Q&A pairs
├── .github/workflows/ci.yml   # CI pipeline
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # API + Streamlit services
└── requirements.txt           # Python dependencies
```

## License

MIT
