"""
RAG Pipeline Module
Retrieval-Augmented Generation pipeline for clinical decision support
"""
from typing import List, Dict, Optional
from .vector_store import get_vector_store, VectorStore
from .llm_handler import get_llm_handler, LLMHandler
from .medical_apis import get_medical_aggregator, MedicalDataAggregator
import re


class RAGPipeline:
    """
    RAG Pipeline for Clinical Decision Support
    
    Combines:
    1. Vector store retrieval (FAISS)
    2. Real-time medical APIs (openFDA)
    3. LLM generation (GPT-4/GPT-3.5)
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_handler: Optional[LLMHandler] = None,
        medical_aggregator: Optional[MedicalDataAggregator] = None
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            vector_store: FAISS vector store instance
            llm_handler: LLM handler instance
            medical_aggregator: Medical data aggregator instance
        """
        self.vector_store = vector_store or get_vector_store()
        self.llm_handler = llm_handler or get_llm_handler()
        self.medical_aggregator = medical_aggregator or get_medical_aggregator()
    
    def query(
        self,
        question: str,
        patient_info: Optional[Dict] = None,
        medications: Optional[List[str]] = None,
        num_results: int = 5
    ) -> Dict:
        """
        Process a clinical query through the RAG pipeline
        
        Args:
            question: Clinical question or query
            patient_info: Optional patient information
            medications: List of current medications
            num_results: Number of documents to retrieve
            
        Returns:
            Dict containing response and sources
        """
        # Step 1: Retrieve relevant documents from vector store
        retrieved_docs = self.vector_store.search(question, k=num_results)
        context = self._format_context(retrieved_docs)
        
        # Step 2: Get drug information if medications provided
        drug_info = None
        if medications:
            drug_info = self.medical_aggregator.get_comprehensive_drug_report(medications)
        
        # Step 3: Build enhanced query with patient info
        enhanced_query = self._build_enhanced_query(question, patient_info)
        
        # Step 4: Generate response using LLM
        response = self.llm_handler.generate_response(
            query=enhanced_query,
            context=context,
            drug_info=drug_info
        )
        
        return {
            'response': response,
            'retrieved_documents': retrieved_docs,
            'drug_information': drug_info,
            'query': question,
            'patient_info': patient_info,
            'medications': medications
        }
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            score = doc.get('score', 0)
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', 'Unknown')
            
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{content}\n(From: {source})")
        
        return "\n\n".join(context_parts)
    
    def _build_enhanced_query(self, question: str, patient_info: Optional[Dict]) -> str:
        """Build enhanced query with patient context"""
        query_parts = [question]
        
        if patient_info:
            patient_context = []
            if 'age' in patient_info:
                patient_context.append(f"Age: {patient_info['age']}")
            if 'gender' in patient_info:
                patient_context.append(f"Gender: {patient_info['gender']}")
            if 'medical_history' in patient_info:
                history = patient_info['medical_history']
                if isinstance(history, list):
                    history = ', '.join(history)
                patient_context.append(f"Medical History: {history}")
            if 'allergies' in patient_info:
                allergies = patient_info['allergies']
                if isinstance(allergies, list):
                    allergies = ', '.join(allergies)
                patient_context.append(f"Allergies: {allergies}")
            
            if patient_context:
                query_parts.append(f"\nPatient Context:\n" + "\n".join(patient_context))
        
        return "\n".join(query_parts)
    
    def extract_medications(self, text: str) -> List[str]:
        """
        Extract medication names from clinical text
        
        Args:
            text: Clinical text containing medication mentions
            
        Returns:
            List of extracted medication names
        """
        # Common medication patterns
        common_meds = [
            'metformin', 'lisinopril', 'amlodipine', 'metoprolol', 'omeprazole',
            'simvastatin', 'atorvastatin', 'levothyroxine', 'gabapentin', 'losartan',
            'hydrochlorothiazide', 'furosemide', 'prednisone', 'albuterol', 'insulin',
            'aspirin', 'ibuprofen', 'acetaminophen', 'warfarin', 'clopidogrel'
        ]
        
        found_meds = []
        text_lower = text.lower()
        
        for med in common_meds:
            if med in text_lower:
                found_meds.append(med.capitalize())
        
        # Also look for patterns like "taking [medication]" or "[medication] mg"
        mg_pattern = r'(\w+)\s*\d+\s*mg'
        matches = re.findall(mg_pattern, text_lower)
        for match in matches:
            if match.capitalize() not in found_meds:
                found_meds.append(match.capitalize())
        
        return found_meds
    
    def quick_drug_check(self, drug_names: List[str]) -> Dict:
        """
        Quick drug interaction and safety check
        
        Args:
            drug_names: List of drug names to check
            
        Returns:
            Drug safety information
        """
        return self.medical_aggregator.get_comprehensive_drug_report(drug_names)
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': self.vector_store.count,
            'llm_available': self.llm_handler.is_available(),
            'llm_model': self.llm_handler.model
        }


# Singleton instance
_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
