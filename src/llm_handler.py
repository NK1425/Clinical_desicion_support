"""
LLM Handler Module
Manages interactions with Large Language Models.
LLM fallback chain: Groq (Llama 3.3 70B) → OpenAI → Fallback (retrieval-only)
"""
import os
from typing import List, Dict, Optional

from .config import settings, CLINICAL_SYSTEM_PROMPT
from .logging_config import get_logger, timed

log = get_logger("llm_handler")


class LLMHandler:
    """Handles LLM interactions for clinical decision support."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.openai_api_key = openai_api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.groq_model = settings.groq_model
        self.openai_model = model or settings.model_name

        self.groq_client = None
        self.openai_client = None
        self.active_provider = None

        # Try Groq first (free)
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.active_provider = "groq"
                log.info(f"LLM initialized: Groq ({self.groq_model})")
            except Exception as e:
                log.warning(f"Failed to initialize Groq client: {e}")

        # Fallback to OpenAI
        if self.active_provider is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.active_provider = "openai"
                log.info(f"LLM initialized: OpenAI ({self.openai_model})")
            except Exception as e:
                log.warning(f"Failed to initialize OpenAI client: {e}")

        if self.active_provider is None:
            log.warning("No LLM provider available — retrieval-only mode")

        self.system_prompt = CLINICAL_SYSTEM_PROMPT

    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return self.active_provider is not None

    @property
    def model(self) -> str:
        """Get the active model name."""
        if self.active_provider == "groq":
            return self.groq_model
        if self.active_provider == "openai":
            return self.openai_model
        return "none"

    @timed(name="llm_handler.generate_response")
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        drug_info: Optional[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        """
        Generate a clinical response using the best available LLM.

        Fallback chain: Groq → OpenAI → retrieval-only
        """
        if not self.is_available():
            return self._generate_fallback_response(query, context, drug_info)

        user_message = self._build_prompt(query, context, drug_info)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Try Groq
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                log.error(f"Groq generation failed: {e}")

        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                log.error(f"OpenAI generation failed: {e}")

        return self._generate_fallback_response(query, context, drug_info)

    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        drug_info: Optional[Dict] = None,
    ) -> str:
        """Build the complete prompt with all available information."""
        prompt_parts = [f"## Clinical Query\n{query}"]

        if context:
            prompt_parts.append(f"\n## Relevant Medical Information (from knowledge base)\n{context}")

        if drug_info:
            prompt_parts.append(f"\n## Drug Information (from openFDA)\n{self._format_drug_info(drug_info)}")

        prompt_parts.append(
            "\n## Instructions\nBased on the above information, provide a comprehensive "
            "clinical assessment. Include relevant guidelines, potential concerns, and recommendations."
        )
        return "\n".join(prompt_parts)

    def _format_drug_info(self, drug_info: Dict) -> str:
        """Format drug information for the prompt."""
        formatted = []
        if "drugs" in drug_info:
            for drug_name, info in drug_info["drugs"].items():
                formatted.append(f"\n### {drug_name}")
                if info.get("found"):
                    if info.get("warnings"):
                        formatted.append(f"**Warnings:** {info['warnings'][0][:500]}...")
                    if info.get("contraindications"):
                        formatted.append(f"**Contraindications:** {info['contraindications'][0][:500]}...")
                    if info.get("common_adverse_events"):
                        formatted.append(f"**Common Adverse Events:** {', '.join(info['common_adverse_events'][:10])}")
                else:
                    formatted.append("No FDA data found for this drug.")

        if drug_info.get("potential_interactions"):
            formatted.append(f"\n### Potential Drug Interactions\n{chr(10).join(drug_info['potential_interactions'])}")

        return "\n".join(formatted)

    def _generate_fallback_response(
        self,
        query: str,
        context: Optional[str] = None,
        drug_info: Optional[Dict] = None,
    ) -> str:
        """Generate a response when no LLM is available."""
        response_parts = [
            "## Clinical Decision Support Response",
            "(Note: LLM unavailable — showing retrieved information only)",
            f"\n### Query: {query}",
        ]

        if context:
            response_parts.append(f"\n### Relevant Medical Guidelines:\n{context}")

        if drug_info and "drugs" in drug_info:
            response_parts.append("\n### Drug Information:")
            for drug_name, info in drug_info["drugs"].items():
                response_parts.append(f"\n**{drug_name}:**")
                if info.get("common_adverse_events"):
                    response_parts.append(f"- Adverse events: {', '.join(info['common_adverse_events'][:5])}")

        response_parts.append(
            "\n\n**Disclaimer:** Please consult with qualified healthcare professionals for medical decisions."
        )
        return "\n".join(response_parts)

    def analyze_symptoms(self, symptoms: List[str], patient_info: Optional[Dict] = None) -> str:
        """Analyze a list of symptoms."""
        query = f"Patient presenting with the following symptoms: {', '.join(symptoms)}."

        if patient_info:
            query += "\n\nPatient Information:"
            if "age" in patient_info:
                query += f"\n- Age: {patient_info['age']}"
            if "gender" in patient_info:
                query += f"\n- Gender: {patient_info['gender']}"
            if "medical_history" in patient_info:
                query += f"\n- Medical History: {', '.join(patient_info['medical_history'])}"

        query += "\n\nPlease provide:\n1. Possible differential diagnoses\n2. Recommended initial workup\n3. Red flags to watch for"
        return self.generate_response(query)


# Singleton instance
_llm_handler = None


def get_llm_handler() -> LLMHandler:
    """Get or create LLM handler singleton."""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler()
    return _llm_handler
