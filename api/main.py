"""
FastAPI Application - Clinical Decision Support API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import get_rag_pipeline
from src.medical_apis import get_fda_client
from src.image_processor import get_image_processor


# Initialize FastAPI app
app = FastAPI(
    title="Clinical Decision Support API",
    description="AI-powered clinical decision support system with RAG pipeline and real-time medical data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class PatientInfo(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class ClinicalQuery(BaseModel):
    question: str
    patient_info: Optional[PatientInfo] = None
    medications: Optional[List[str]] = None


class DrugQuery(BaseModel):
    drug_names: List[str]


class ImageAnalysisRequest(BaseModel):
    image_path: str
    image_type: Optional[str] = "general"
    question: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    knowledge_base_docs: int
    llm_available: bool
    image_processor_available: bool


# Initialize components
rag_pipeline = get_rag_pipeline()
fda_client = get_fda_client()
image_processor = get_image_processor()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Clinical Decision Support API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    stats = rag_pipeline.get_knowledge_base_stats()
    return HealthResponse(
        status="healthy",
        knowledge_base_docs=stats['total_documents'],
        llm_available=stats['llm_available'],
        image_processor_available=image_processor.is_available()
    )


@app.post("/api/query", tags=["Clinical Query"])
async def clinical_query(query: ClinicalQuery):
    """
    Process a clinical query through the RAG pipeline
    
    - Retrieves relevant medical guidelines
    - Fetches real-time drug information (if medications provided)
    - Generates AI-powered clinical insights
    """
    try:
        patient_info = None
        if query.patient_info:
            patient_info = {
                'age': query.patient_info.age,
                'gender': query.patient_info.gender,
                'medical_history': query.patient_info.medical_history,
                'allergies': query.patient_info.allergies
            }
        
        result = rag_pipeline.query(
            question=query.question,
            patient_info=patient_info,
            medications=query.medications
        )
        
        return {
            "success": True,
            "response": result['response'],
            "sources": len(result['retrieved_documents']),
            "medications_analyzed": query.medications
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drug-check", tags=["Drug Information"])
async def drug_check(query: DrugQuery):
    """
    Check drug information and interactions
    
    - Fetches real-time data from openFDA
    - Checks for potential drug interactions
    - Returns warnings and contraindications
    """
    try:
        result = rag_pipeline.quick_drug_check(query.drug_names)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drug/{drug_name}", tags=["Drug Information"])
async def get_drug_info(drug_name: str):
    """
    Get detailed information for a specific drug
    """
    try:
        result = fda_client.get_drug_info_summary(drug_name)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/adverse-events/{drug_name}", tags=["Drug Information"])
async def get_adverse_events(drug_name: str, limit: int = 10):
    """
    Get adverse event reports for a drug
    """
    try:
        result = fda_client.get_adverse_events(drug_name, limit=limit)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-image", tags=["Image Analysis"])
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze a medical image using BLIP-2
    
    Note: Requires transformers and torch to be installed
    """
    try:
        if request.question:
            result = image_processor.answer_question(request.image_path, request.question)
        else:
            result = image_processor.get_clinical_findings(
                request.image_path, 
                image_type=request.image_type
            )
        
        return {
            "success": result.get('success', False),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", tags=["Statistics"])
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_pipeline.get_knowledge_base_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
