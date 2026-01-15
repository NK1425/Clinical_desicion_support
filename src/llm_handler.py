"""
LLM Handler Module
Manages interactions with Large Language Models (OpenAI GPT-4/GPT-3.5)
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI
from .config import settings, CLINICAL_SYSTEM_PROMPT


class LLMHandler:
    """Handles LLM interactions for clinical decision support"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM handler
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Model name to use
        """
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = CLINICAL_SYSTEM_PROMPT
        self.conversation_history: List[Dict] = []
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.client is not None
    
    def generate_response(
        self, 
        query: str, 
        context: Optional[str] = None,
        drug_info: Optional[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate a clinical response using the LLM
        
        Args:
            query: User's clinical query
            context: Retrieved context from RAG pipeline
            drug_info: Drug information from openFDA
            temperature: Response randomness (lower = more focused)
            max_tokens: Maximum response length
            
        Returns:
            Generated clinical response
        """
        if not self.is_available():
            return self._generate_fallback_response(query, context, drug_info)
        
        # Build the prompt with context
        user_message = self._build_prompt(query, context, drug_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}\n\n{self._generate_fallback_response(query, context, drug_info)}"
    
    def _build_prompt(
        self, 
        query: str, 
        context: Optional[str] = None, 
        drug_info: Optional[Dict] = None
    ) -> str:
        """Build the complete prompt with all available information"""
        
        prompt_parts = []
        
        # Add clinical query
        prompt_parts.append(f"## Clinical Query\n{query}")
        
        # Add RAG context if available
        if context:
            prompt_parts.append(f"\n## Relevant Medical Information (from knowledge base)\n{context}")
        
        # Add drug information if available
        if drug_info:
            prompt_parts.append(f"\n## Drug Information (from openFDA)\n{self._format_drug_info(drug_info)}")
        
        prompt_parts.append("\n## Instructions\nBased on the above information, provide a comprehensive clinical assessment. Include relevant guidelines, potential concerns, and recommendations.")
        
        return "\n".join(prompt_parts)
    
    def _format_drug_info(self, drug_info: Dict) -> str:
        """Format drug information for the prompt"""
        formatted = []
        
        if 'drugs' in drug_info:
            for drug_name, info in drug_info['drugs'].items():
                formatted.append(f"\n### {drug_name}")
                if info.get('found'):
                    if info.get('warnings'):
                        formatted.append(f"**Warnings:** {info['warnings'][0][:500]}...")
                    if info.get('contraindications'):
                        formatted.append(f"**Contraindications:** {info['contraindications'][0][:500]}...")
                    if info.get('common_adverse_events'):
                        formatted.append(f"**Common Adverse Events:** {', '.join(info['common_adverse_events'][:10])}")
                else:
                    formatted.append("No FDA data found for this drug.")
        
        if drug_info.get('potential_interactions'):
            formatted.append(f"\n### Potential Drug Interactions\n{chr(10).join(drug_info['potential_interactions'])}")
        
        return "\n".join(formatted)
    
    def _generate_fallback_response(
        self, 
        query: str, 
        context: Optional[str] = None, 
        drug_info: Optional[Dict] = None
    ) -> str:
        """Generate a fallback response when LLM is not available"""
        
        response_parts = [
            "## Clinical Decision Support Response",
            "(Note: LLM unavailable - showing retrieved information only)",
            f"\n### Query: {query}"
        ]
        
        if context:
            response_parts.append(f"\n### Relevant Medical Guidelines:\n{context}")
        
        if drug_info and 'drugs' in drug_info:
            response_parts.append("\n### Drug Information:")
            for drug_name, info in drug_info['drugs'].items():
                response_parts.append(f"\n**{drug_name}:**")
                if info.get('common_adverse_events'):
                    response_parts.append(f"- Adverse events: {', '.join(info['common_adverse_events'][:5])}")
        
        response_parts.append("\n\n⚠️ **Disclaimer:** Please consult with qualified healthcare professionals for medical decisions.")
        
        return "\n".join(response_parts)
    
    def analyze_symptoms(self, symptoms: List[str], patient_info: Optional[Dict] = None) -> str:
        """
        Analyze a list of symptoms
        
        Args:
            symptoms: List of symptoms
            patient_info: Optional patient information (age, gender, history)
            
        Returns:
            Clinical analysis of symptoms
        """
        query = f"Patient presenting with the following symptoms: {', '.join(symptoms)}."
        
        if patient_info:
            query += f"\n\nPatient Information:"
            if 'age' in patient_info:
                query += f"\n- Age: {patient_info['age']}"
            if 'gender' in patient_info:
                query += f"\n- Gender: {patient_info['gender']}"
            if 'medical_history' in patient_info:
                query += f"\n- Medical History: {', '.join(patient_info['medical_history'])}"
        
        query += "\n\nPlease provide:\n1. Possible differential diagnoses\n2. Recommended initial workup\n3. Red flags to watch for"
        
        return self.generate_response(query)


# Singleton instance
_llm_handler = None

def get_llm_handler() -> LLMHandler:
    """Get or create LLM handler singleton"""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler(model=settings.model_name)
    return _llm_handler
