# 🏥 Clinical Decision Support Assistant

An AI-powered clinical decision support system leveraging Large Language Models, RAG pipelines, and real-time medical APIs to assist healthcare professionals with diagnostic insights.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Multimodal Processing**: Handles patient records, clinical notes, and medical images
- **RAG Pipeline**: Retrieval-Augmented Generation using FAISS for accurate medical information retrieval
- **Real-time Data**: Integrates with openFDA API for drug information and adverse events
- **Interactive Dashboard**: Streamlit-based UI for easy interaction
- **RESTful API**: FastAPI backend for integration with existing systems
- **Containerized**: Docker support for easy deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Clinical Decision Support System                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 Patient Data      🖼️ Medical Images     📝 Clinical Notes │
│         │                    │                    │          │
│         └────────────────────┼────────────────────┘          │
│                              ▼                               │
│              ┌───────────────────────────┐                   │
│              │   Multimodal Processor    │                   │
│              │   (BLIP-2 + Text Parser)  │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                 │
│              ┌───────────────────────────┐                   │
│              │      RAG Pipeline         │                   │
│              │  ┌───────┐  ┌──────────┐  │                   │
│              │  │ FAISS │  │ Medical  │  │                   │
│              │  │ Index │  │ Knowledge│  │                   │
│              │  └───────┘  └──────────┘  │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                 │
│              ┌───────────────────────────┐                   │
│              │   Real-time APIs          │                   │
│              │  (openFDA, PubMed)        │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                 │
│              ┌───────────────────────────┐                   │
│              │      LLM Engine           │                   │
│              │    (GPT-4 / LLaMA)        │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                 │
│              ┌───────────────────────────┐                   │
│              │    FastAPI Backend        │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                 │
│              ┌───────────────────────────┐                   │
│              │   Streamlit Dashboard     │                   │
│              └───────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API Key
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NK1425/clinical-decision-support.git
   cd clinical-decision-support
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Initialize the vector store**
   ```bash
   python src/init_vectorstore.py
   ```

### Running the Application

#### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run app/streamlit_app.py
```
Access at: `http://localhost:8501`

#### Option 2: FastAPI Backend
```bash
uvicorn api.main:app --reload
```
Access API docs at: `http://localhost:8000/docs`

#### Option 3: Docker
```bash
docker-compose up --build
```

## 📁 Project Structure

```
clinical-decision-support/
├── README.md                 # Documentation
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker compose
│
├── data/
│   ├── medical_guidelines.txt    # CDC/WHO guidelines
│   ├── drug_interactions.json    # Drug interaction data
│   └── sample_patients.json      # Synthetic patient data
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── embeddings.py             # Text embeddings
│   ├── vector_store.py           # FAISS operations
│   ├── rag_pipeline.py           # RAG implementation
│   ├── llm_handler.py            # LLM interactions
│   ├── image_processor.py        # BLIP-2 image analysis
│   ├── medical_apis.py           # openFDA integration
│   └── init_vectorstore.py       # Initialize vector DB
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   └── routes/
│       ├── query.py              # Query endpoints
│       └── health.py             # Health check
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
└── tests/
    └── test_rag.py               # Unit tests
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | Yes |
| `MODEL_NAME` | LLM model to use | No (default: gpt-3.5-turbo) |
| `EMBEDDING_MODEL` | Embedding model | No (default: all-MiniLM-L6-v2) |

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Submit clinical query |
| POST | `/api/analyze-image` | Analyze medical image |
| GET | `/api/drug/{drug_name}` | Get drug information |
| GET | `/api/health` | Health check |

## 🧪 Example Usage

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "patient_info": "65-year-old male with Type 2 diabetes",
        "symptoms": "fatigue, increased thirst, blurred vision",
        "current_medications": ["Metformin 500mg", "Lisinopril 10mg"]
    }
)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"patient_info": "65-year-old male", "symptoms": "chest pain"}'
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | OpenAI GPT-4 / GPT-3.5 |
| **Vision** | BLIP-2 (Salesforce) |
| **RAG Framework** | LangChain |
| **Vector Database** | FAISS |
| **Real-time Data** | openFDA API |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker |

## ⚠️ Disclaimer

This is an **academic/portfolio project** for demonstration purposes only. It should **NOT** be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Nitish Kumar Manthri**
- LinkedIn: [nitish-kumar-6b6925303](https://www.linkedin.com/in/nitish-kumar-6b6925303)
- GitHub: [NK1425](https://github.com/NK1425)
