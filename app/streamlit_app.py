"""
Clinical Decision Support Assistant
AI-Powered Medical Intelligence Platform
"""
import streamlit as st
import sys
import os
import requests
from typing import Dict, List, Optional
from datetime import datetime

# Fix path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import components with graceful fallbacks
RAG_AVAILABLE = False
APIS_IMPORTED = False

try:
    from src.rag_pipeline import get_rag_pipeline
    RAG_AVAILABLE = True
except ImportError:
    pass

try:
    from src.medical_apis import get_fda_client, get_medical_aggregator
    APIS_IMPORTED = True
except ImportError:
    pass

try:
    from src.config import settings
except ImportError:
    settings = None

st.set_page_config(
    page_title="Clinical Decision Support Assistant",
    page_icon="🏥",
    layout="wide",
)

# --- Inline FDA Client Fallback (for Streamlit Cloud without full deps) ---
class SimpleFDAClient:
    """Lightweight FDA client for when full src package is not available."""
    BASE_URL = "https://api.fda.gov"

    def search_drug(self, drug_name: str, limit: int = 5) -> Dict:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/drug/label.json",
                params={"search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"', "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"error": "API request failed", "results": []}

    def get_drug_info_summary(self, drug_name: str) -> Dict:
        summary = {"drug_name": drug_name, "found": False, "indications": [], "dosage": [],
                    "warnings": [], "contraindications": [], "interactions": [], "common_adverse_events": []}
        data = self.search_drug(drug_name, limit=1)
        if "results" in data and data["results"]:
            r = data["results"][0]
            summary["found"] = True
            summary["indications"] = r.get("indications_and_usage", ["Not available"])
            summary["dosage"] = r.get("dosage_and_administration", ["Not available"])
            summary["warnings"] = r.get("warnings", ["Not available"])
            summary["contraindications"] = r.get("contraindications", ["Not available"])
            summary["interactions"] = r.get("drug_interactions", ["Not available"])
        return summary

    def get_adverse_events(self, drug_name: str, limit: int = 10) -> Dict:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/drug/event.json",
                params={"search": f'patient.drug.medicinalproduct:"{drug_name}"', "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            events = []
            for result in data.get("results", []):
                for reaction in result.get("patient", {}).get("reaction", []):
                    events.append(reaction.get("reactionmeddrapt", "Unknown"))
            return {"drug_name": drug_name, "adverse_events": list(set(events))[:20],
                    "total_reports": data.get("meta", {}).get("results", {}).get("total", 0)}
        except Exception:
            return {"drug_name": drug_name, "adverse_events": [], "error": "API request failed"}


# Disease database for symptom checker
DISEASE_DATABASE = {
    "Type 2 Diabetes": {"symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision", "weight loss"],
        "risk_factors": ["obesity", "family history", "sedentary lifestyle", "age >45"],
        "tests": ["Fasting glucose", "HbA1c", "Oral glucose tolerance test"],
        "treatment": "Metformin first-line, lifestyle modifications, monitoring HbA1c"},
    "Hypertension": {"symptoms": ["headache", "dizziness", "chest pain", "shortness of breath", "visual changes"],
        "risk_factors": ["obesity", "high sodium diet", "smoking", "family history", "age"],
        "tests": ["Blood pressure measurement", "BMP", "Urinalysis", "ECG"],
        "treatment": "ACE inhibitors, ARBs, CCBs, or thiazide diuretics; target <130/80"},
    "Acute Coronary Syndrome": {"symptoms": ["chest pain", "diaphoresis", "dyspnea", "nausea", "jaw pain", "arm pain"],
        "risk_factors": ["smoking", "diabetes", "hypertension", "hyperlipidemia", "family history"],
        "tests": ["ECG", "Troponin", "CXR", "Echocardiogram"],
        "treatment": "Aspirin, P2Y12 inhibitor, anticoagulation, PCI if STEMI"},
    "COPD": {"symptoms": ["chronic cough", "dyspnea", "sputum production", "wheezing"],
        "risk_factors": ["smoking", "occupational exposure", "alpha-1 antitrypsin deficiency"],
        "tests": ["Spirometry (FEV1/FVC)", "CXR", "ABG"],
        "treatment": "Bronchodilators (LAMA/LABA), ICS if eosinophilic, smoking cessation"},
    "Sepsis": {"symptoms": ["fever", "tachycardia", "hypotension", "altered mental status", "tachypnea"],
        "risk_factors": ["immunosuppression", "chronic disease", "recent surgery", "indwelling devices"],
        "tests": ["Blood cultures", "Lactate", "CBC", "Procalcitonin"],
        "treatment": "Hour-1 bundle: cultures, broad-spectrum antibiotics, 30mL/kg fluids, vasopressors if needed"},
    "Heart Failure": {"symptoms": ["dyspnea", "orthopnea", "PND", "peripheral edema", "fatigue", "weight gain"],
        "risk_factors": ["CAD", "hypertension", "diabetes", "valvular disease"],
        "tests": ["BNP/NT-proBNP", "Echocardiogram", "CXR", "ECG"],
        "treatment": "GDMT: ACEi/ARB/ARNI + beta-blocker + MRA + SGLT2i"},
}


@st.cache_resource
def load_components():
    """Load RAG pipeline and FDA client with fallbacks."""
    rag = None
    fda = None

    if RAG_AVAILABLE:
        try:
            rag = get_rag_pipeline()
        except Exception:
            pass

    if APIS_IMPORTED:
        try:
            fda = get_fda_client()
        except Exception:
            pass

    if fda is None:
        fda = SimpleFDAClient()

    return rag, fda


def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0.5rem; }
        .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">🏥 Clinical Decision Support Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered clinical insights using RAG, LangChain, and real-time medical data</p>', unsafe_allow_html=True)

    rag_pipeline, fda_client = load_components()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        if rag_pipeline:
            stats = rag_pipeline.get_knowledge_base_stats()
            col1, col2 = st.columns(2)
            col1.metric("📚 Docs", stats["total_documents"])
            col2.metric("🤖 LLM", "✅" if stats["llm_available"] else "⚠️")
            if stats["llm_available"]:
                st.caption(f"Model: {stats['llm_model']}")
            else:
                st.caption("Set GROQ_API_KEY for LLM responses")
        else:
            st.warning("RAG pipeline not available")
            st.caption("Run `python src/init_vectorstore.py` to initialize")

        st.divider()
        st.header("👤 Patient Info")
        age = st.number_input("Age", 0, 120, 0)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])
        medical_history = st.text_area("Medical History", placeholder="e.g., Diabetes, Hypertension")
        current_meds = st.text_area("Current Medications", placeholder="e.g., Metformin 500mg")

        st.divider()
        st.caption("⚠️ For educational purposes only. Not for actual medical decisions.")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🩺 Clinical Query", "💊 Drug Lookup", "🔍 Symptom Checker"])

    # --- Clinical Query Tab ---
    with tab1:
        query = st.text_area(
            "Enter your clinical question:", height=100,
            placeholder="e.g., What is the recommended management for Type 2 diabetes?",
        )

        if st.button("🔍 Analyze", type="primary", key="analyze"):
            if query and rag_pipeline:
                with st.spinner("Analyzing..."):
                    patient_info = {}
                    if age > 0:
                        patient_info["age"] = age
                    if gender != "Not specified":
                        patient_info["gender"] = gender
                    if medical_history:
                        patient_info["medical_history"] = [h.strip() for h in medical_history.split(",")]

                    medications = [m.strip() for m in current_meds.split(",") if m.strip()] if current_meds else None

                    result = rag_pipeline.query(
                        question=query,
                        patient_info=patient_info if patient_info else None,
                        medications=medications,
                    )

                    st.markdown("### 📋 Clinical Assessment")
                    st.markdown(result["response"])

                    # Show timing info
                    timings = result.get("timings", {})
                    if timings:
                        tcols = st.columns(len(timings))
                        for i, (key, val) in enumerate(timings.items()):
                            label = key.replace("_seconds", "").replace("_", " ").title()
                            tcols[i].metric(label, f"{val:.3f}s")

                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(result["retrieved_documents"], 1):
                            st.markdown(f"**Source {i}** (Score: {doc['score']:.2f})")
                            st.caption(doc["content"][:200] + "...")
            elif not rag_pipeline:
                st.warning("RAG pipeline not available. Initialize the knowledge base first.")

    # --- Drug Lookup Tab ---
    with tab2:
        drug_name = st.text_input("Enter drug name:", placeholder="e.g., Metformin")

        if st.button("🔍 Search Drug", type="primary", key="drug_search"):
            if drug_name:
                with st.spinner(f"Searching {drug_name}..."):
                    info = fda_client.get_drug_info_summary(drug_name)

                    if info.get("found"):
                        st.success(f"Found: {drug_name}")

                        if info.get("indications"):
                            st.markdown("### 📋 Indications")
                            text = info["indications"][0]
                            st.write(text[:500] + "..." if len(text) > 500 else text)

                        if info.get("warnings"):
                            st.markdown("### ⚠️ Warnings")
                            text = info["warnings"][0]
                            st.warning(text[:500] + "..." if len(text) > 500 else text)

                        if info.get("contraindications"):
                            st.markdown("### 🚫 Contraindications")
                            text = info["contraindications"][0]
                            st.error(text[:500] + "..." if len(text) > 500 else text)

                        if info.get("common_adverse_events"):
                            st.markdown("### 💊 Common Side Effects")
                            st.write(", ".join(info["common_adverse_events"][:15]))
                    else:
                        st.warning(f"No data found for '{drug_name}'")

    # --- Symptom Checker Tab ---
    with tab3:
        st.markdown("### Symptom-Based Differential Diagnosis")
        st.caption("Select symptoms to get possible diagnoses from the clinical database")

        all_symptoms = sorted(set(
            s for d in DISEASE_DATABASE.values() for s in d["symptoms"]
        ))
        selected_symptoms = st.multiselect("Select symptoms:", all_symptoms)

        if selected_symptoms and st.button("Check Symptoms", type="primary", key="symptom_check"):
            matches = []
            for disease, info in DISEASE_DATABASE.items():
                overlap = set(selected_symptoms) & set(info["symptoms"])
                if overlap:
                    score = len(overlap) / len(info["symptoms"])
                    matches.append((disease, info, score, overlap))

            matches.sort(key=lambda x: x[2], reverse=True)

            if matches:
                st.markdown("### Possible Diagnoses")
                for disease, info, score, overlap in matches:
                    with st.expander(f"**{disease}** — {score:.0%} symptom match"):
                        st.markdown(f"**Matching symptoms:** {', '.join(overlap)}")
                        st.markdown(f"**Risk factors:** {', '.join(info['risk_factors'])}")
                        st.markdown(f"**Recommended tests:** {', '.join(info['tests'])}")
                        st.markdown(f"**Treatment:** {info['treatment']}")
            else:
                st.info("No matching conditions found for the selected symptoms.")


if __name__ == "__main__":
    main()
