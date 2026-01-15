"""
Streamlit Dashboard - Clinical Decision Support System
Real-time medical data + LLM-powered insights
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.medical_apis import (
    get_fda_client, 
    get_pubmed_client, 
    get_rxnorm_client,
    get_medical_aggregator
)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Clinical Decision Support Assistant", page_icon="🏥", layout="wide")

@st.cache_resource
def load_clients():
    return get_fda_client(), get_pubmed_client(), get_rxnorm_client(), get_medical_aggregator()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    if api_key and OPENAI_AVAILABLE:
        return OpenAI(api_key=api_key)
    return None

def generate_llm_response(client, query, context):
    if not client:
        return None
    
    context_text = ""
    if context.get('research'):
        context_text += "\n## PubMed Research:\n"
        for a in context['research'][:3]:
            context_text += f"- {a['title']} ({a['year']}): {a['abstract'][:200]}...\n"
    if context.get('drug_info'):
        context_text += "\n## FDA Drug Info:\n"
        for drug, info in context['drug_info'].items():
            if info.get('found'):
                context_text += f"- {drug}: {info.get('indications', [''])[0][:150]}...\n"
    if context.get('interactions'):
        context_text += "\n## Drug Interactions:\n"
        for i in context['interactions']:
            context_text += f"- {i.get('description', '')}\n"
    if context.get('patient_info'):
        context_text += f"\n## Patient: {context['patient_info']}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Clinical Decision Support AI. Provide evidence-based medical insights. Always note that final decisions must be made by healthcare professionals. Format with clear sections."},
                {"role": "user", "content": f"Query: {query}\n\nReal-time Data:\n{context_text}\n\nProvide: 1) Clinical Assessment 2) Key Findings 3) Recommendations 4) Warnings if any"}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

def main():
    st.markdown("## 🏥 Clinical Decision Support Assistant")
    st.caption("Real-time APIs + AI-powered clinical insights")
    
    fda, pubmed, rxnorm, aggregator = load_clients()
    llm = get_openai_client()
    
    with st.sidebar:
        st.header("🔌 Status")
        st.success("✅ openFDA")
        st.success("✅ PubMed")
        st.success("✅ RxNorm")
        st.success("✅ OpenAI LLM") if llm else st.warning("⚠️ LLM - Add API key")
        st.divider()
        st.header("👤 Patient")
        age = st.number_input("Age", 0, 120, 0)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])
        history = st.text_area("Medical History", placeholder="Diabetes, HTN")
        meds = st.text_area("Medications", placeholder="Metformin, Lisinopril")
        st.divider()
        st.caption("⚠️ Educational only. Not for real medical decisions.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Query", "🔬 PubMed", "💊 Drug Info", "⚠️ Interactions"])
    
    with tab1:
        st.markdown("### 🤖 AI Clinical Query")
        query = st.text_area("Ask any clinical question:", placeholder="What are treatment options for Type 2 diabetes with hypertension?", height=100)
        med_list = [m.strip() for m in meds.split(',') if m.strip()] if meds else []
        
        if st.button("🔍 Analyze", type="primary"):
            if query:
                with st.spinner("Gathering data & generating AI response..."):
                    research = pubmed.search_articles(query, 5).get('articles', [])
                    drug_info = {m: fda.get_drug_info_summary(m) for m in med_list} if med_list else {}
                    interactions = rxnorm.get_interactions(med_list).get('interactions', []) if len(med_list) >= 2 else []
                    patient = f"Age:{age}, Gender:{gender}, History:{history}" if age > 0 else ""
                    
                    st.success(f"✅ Found {len(research)} articles, {len(drug_info)} drugs analyzed")
                    
                    st.markdown("---")
                    st.markdown("### 🤖 AI Clinical Assessment")
                    
                    if llm:
                        response = generate_llm_response(llm, query, {'research': research, 'drug_info': drug_info, 'interactions': interactions, 'patient_info': patient})
                        st.markdown(response)
                    else:
                        st.warning("Add OPENAI_API_KEY in Streamlit secrets for AI insights")
                    
                    st.markdown("---")
                    st.markdown("### 📚 Research Sources")
                    for a in research[:3]:
                        with st.expander(a['title'][:60] + "..."):
                            st.write(f"**{a['journal']}** ({a['year']})")
                            st.write(a['abstract'])
                            if a.get('url'): st.markdown(f"[PubMed Link]({a['url']})")
    
    with tab2:
        st.markdown("### 🔬 PubMed Search")
        q = st.text_input("Search:", placeholder="diabetes treatment 2024")
        n = st.slider("Results", 3, 10, 5)
        if st.button("Search", key="pub"):
            if q:
                results = pubmed.search_articles(q, n).get('articles', [])
                st.success(f"Found {len(results)}")
                for a in results:
                    with st.expander(a['title'][:70]):
                        st.write(f"{a['journal']} ({a['year']})")
                        st.write(a['abstract'])
                        if a.get('url'): st.markdown(f"[Link]({a['url']})")
    
    with tab3:
        st.markdown("### 💊 Drug Lookup")
        drug = st.text_input("Drug name:", placeholder="Metformin")
        if st.button("Search", key="drug"):
            if drug:
                info = fda.get_drug_info_summary(drug)
                if info.get('found'):
                    st.success(f"Found: {drug}")
                    if info.get('indications'): st.markdown(f"**Indications:** {info['indications'][0][:500]}...")
                    if info.get('warnings'): st.warning(f"**Warnings:** {info['warnings'][0][:500]}...")
                    if info.get('common_adverse_events'): st.write(f"**Side Effects:** {', '.join(info['common_adverse_events'][:10])}")
                else:
                    st.warning("Not found")
    
    with tab4:
        st.markdown("### ⚠️ Interaction Check")
        drugs = st.text_area("Enter drugs (one per line):", placeholder="Warfarin\nAspirin")
        if st.button("Check", key="inter"):
            dl = [d.strip() for d in drugs.split('\n') if d.strip()]
            if len(dl) >= 2:
                result = rxnorm.get_interactions(dl)
                if result.get('interactions'):
                    st.error(f"Found {len(result['interactions'])} interactions!")
                    for i in result['interactions']:
                        st.write(f"⚠️ {i.get('description')}")
                else:
                    st.success("No interactions found")

if __name__ == "__main__":
    main()
