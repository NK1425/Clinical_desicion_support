"""
FastAPI Application — Clinical Decision Support API
Features: API key auth, rate limiting, Prometheus metrics, structured error responses.
"""
import time
import os
import sys
from collections import defaultdict
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import get_rag_pipeline
from src.medical_apis import get_fda_client
from src.image_processor import get_image_processor
from src.config import settings
from src.logging_config import get_logger, correlation_id
from api.monitoring import (
    get_metrics, get_metrics_content_type,
    record_request, record_rag_retrieval, record_llm_generation,
    record_error, update_knowledge_base_size,
)

log = get_logger("api")

# --- FastAPI App ---
app = FastAPI(
    title="Clinical Decision Support API",
    description="AI-powered clinical decision support with RAG pipeline, real-time medical data, and LangChain integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiter (in-memory sliding window) ---
_rate_limit_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds


def _check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit. Returns True if allowed."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    _rate_limit_store[client_id] = [
        t for t in _rate_limit_store[client_id] if t > window_start
    ]
    if len(_rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_limit_store[client_id].append(now)
    return True


# --- Auth Dependency ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if configured."""
    configured_key = settings.api_key
    if not configured_key:
        return  # No key configured = open access
    if x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Middleware ---
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Log requests, enforce rate limiting, record metrics."""
    import uuid
    cid = uuid.uuid4().hex[:12]
    correlation_id.set(cid)

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        record_error("rate_limit")
        return Response(
            content='{"detail":"Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    endpoint = request.url.path
    record_request(request.method, endpoint, response.status_code, duration)
    log.info(
        f"{request.method} {endpoint} -> {response.status_code} ({duration:.3f}s)",
    )
    response.headers["X-Correlation-ID"] = cid
    return response


# --- Request/Response Models ---
class PatientInfo(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class ClinicalQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    patient_info: Optional[PatientInfo] = None
    medications: Optional[List[str]] = None


class DrugQuery(BaseModel):
    drug_names: List[str] = Field(..., min_length=1, max_length=10)


class ImageAnalysisRequest(BaseModel):
    image_path: str
    image_type: Optional[str] = "general"
    question: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_type: str = "internal_error"


class HealthResponse(BaseModel):
    status: str
    knowledge_base_docs: int
    llm_available: bool
    llm_model: str
    image_processor_available: bool


# --- Initialize components ---
rag_pipeline = get_rag_pipeline()
fda_client = get_fda_client()
image_processor = get_image_processor()


# --- Endpoints ---
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Clinical Decision Support API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "metrics": "/metrics",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    stats = rag_pipeline.get_knowledge_base_stats()
    update_knowledge_base_size(stats["total_documents"])
    return HealthResponse(
        status="healthy",
        knowledge_base_docs=stats["total_documents"],
        llm_available=stats["llm_available"],
        llm_model=stats["llm_model"],
        image_processor_available=image_processor.is_available(),
    )


@app.post("/api/query", tags=["Clinical Query"], dependencies=[Depends(verify_api_key)])
async def clinical_query(query: ClinicalQuery):
    """Process a clinical query through the RAG pipeline."""
    try:
        patient_info = None
        if query.patient_info:
            patient_info = {
                "age": query.patient_info.age,
                "gender": query.patient_info.gender,
                "medical_history": query.patient_info.medical_history,
                "allergies": query.patient_info.allergies,
            }

        result = rag_pipeline.query(
            question=query.question,
            patient_info=patient_info,
            medications=query.medications,
        )

        # Record RAG metrics
        timings = result.get("timings", {})
        if "retrieval_seconds" in timings:
            record_rag_retrieval(timings["retrieval_seconds"])
        if "llm_seconds" in timings:
            record_llm_generation(timings["llm_seconds"])

        return {
            "success": True,
            "response": result["response"],
            "sources": len(result["retrieved_documents"]),
            "medications_analyzed": query.medications,
            "timings": timings,
        }

    except ValueError as e:
        record_error("validation_error")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        record_error("runtime_error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drug-check", tags=["Drug Information"], dependencies=[Depends(verify_api_key)])
async def drug_check(query: DrugQuery):
    """Check drug information and interactions."""
    try:
        result = rag_pipeline.quick_drug_check(query.drug_names)
        return {"success": True, "data": result}
    except ValueError as e:
        record_error("validation_error")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/drug/{drug_name}", tags=["Drug Information"])
async def get_drug_info(drug_name: str):
    """Get detailed information for a specific drug."""
    try:
        result = fda_client.get_drug_info_summary(drug_name)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/adverse-events/{drug_name}", tags=["Drug Information"])
async def get_adverse_events(drug_name: str, limit: int = 10):
    """Get adverse event reports for a drug."""
    try:
        result = fda_client.get_adverse_events(drug_name, limit=limit)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/analyze-image", tags=["Image Analysis"], dependencies=[Depends(verify_api_key)])
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze a medical image using BLIP-2."""
    try:
        if request.question:
            result = image_processor.answer_question(request.image_path, request.question)
        else:
            result = image_processor.get_clinical_findings(
                request.image_path, image_type=request.image_type
            )
        return {"success": result.get("success", False), "data": result}
    except FileNotFoundError:
        record_error("file_not_found")
        raise HTTPException(status_code=404, detail="Image file not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/stats", tags=["Statistics"])
async def get_stats():
    """Get system statistics."""
    stats = rag_pipeline.get_knowledge_base_stats()
    update_knowledge_base_size(stats["total_documents"])
    return {"success": True, "data": stats}


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
