"""
Clinical Decision Support Assistant
Premium Apple-Inspired Design + Free Gemini AI
"""
import streamlit as st
import sys
import os
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.medical_apis import get_fda_client, get_pubmed_client, get_rxnorm_client

st.set_page_config(page_title="Clinical AI Assistant", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

# Premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* {font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
.stApp {background: linear-gradient(180deg, #0a0a0f 0%, #111118 100%);}
#MainMenu, footer, header {visibility: hidden;}
.main .block-container {padding: 2rem 3rem; max-width: 1100px;}

.hero {text-align: center; padding: 50px 20px 30px;}
.hero h1 {font-size: 3rem; font-weight: 700; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;}
.hero p {font-size: 1.1rem; color: #64748b; font-weight: 400;}

.glass-card {background: rgba(255,255,255,0.02); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 28px; margin: 16px 0;}
.glass-card:hover {border-color: rgba(99,102,241,0.3);}
.glass-card h3 {color: #fff; margin: 0 0 6px; font-weight: 600; font-size: 1.15rem;}
.glass-card p {color: #64748b; margin: 0; font-size: 0.9rem;}

.result-card {background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 18px; margin: 10px 0;}
.result-card h5 {color: #8b5cf6; font-size: 0.95rem; font-weight: 600; margin: 0 0 8px;}
.result-card p {color: #94a3b8; font-size: 0.88rem; line-height: 1.55; margin: 0;}

.ai-card {background: linear-gradient(145deg, rgba(99,102,241,0.08), rgba(139,92,246,0.03)); border: 1px solid rgba(99,102,241,0.2); border-radius: 18px; padding: 24px; margin: 20px 0;}

.status-pill {display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; margin: 3px;}
.status-on {background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25);}
.status-off {background: rgba(234,179,8,0.12); color: #eab308; border: 1px solid rgba(234,179,8,0.25);}

.success-box {background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25); border-radius: 12px; padding: 14px 18px; color: #22c55e; font-weight: 500;}
.warning-box {background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.25); border-radius: 12px; padding: 14px 18px; color: #eab308;}
.error-box {background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); border-radius: 12px; padding: 14px 18px; color: #ef4444;}

.stButton > button {background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 12px 28px !important; font-weight: 600 !important; box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;}
.stButton > button:hover {transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; color: white !important; padding: 14px !important;}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {border-color: rgba(99,102,241,0.4) !important;}

.stTabs [data-baseweb="tab-list"] {background: rgba(255,255,255,0.02); border-radius: 14px; padding: 5px; gap: 3px;}
.stTabs [data-baseweb="tab"] {background: transparent; border-radius: 10px; color: #64748b; font-weight: 500; padding: 10px 20px;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important;}

.streamlit-expanderHeader {background: rgba(255,255,255,0.02) !important; border-radius: 10px !important;}
[data-testid="stSidebar"] {background: rgba(0,0,0,0.95) !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clients():
    return get_fda_client(), get_pubmed_client(), get_rxnorm_client()

def call_gemini(prompt, api_key):
    """Call Gemini API with correct model name"""
    if not api_key:
        return None
    
    # Use gemini-2.0-flash (current free model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}
        }, timeout=30)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"AI Error: {str(e)}"

def generate_response(query, research, drug_info, interactions, patient, api_key):
    context = f"""You are an expert Clinical Decision Support AI.

QUERY: {query}
PATIENT: {patient if patient else 'Not provided'}

📚 RESEARCH ({len(research)} articles):
"""
    for i, a in enumerate(research[:3], 1):
        context += f"\n{i}. {a['title']} ({a['year']})\n   {a['abstract'][:180]}...\n"
    
    if drug_info:
        context += "\n💊 MEDICATIONS:\n"
        for d, info in drug_info.items():
            if info.get('found'):
                context += f"• {d}: {info.get('indications', [''])[0][:120]}...\n"
    
    if interactions:
        context += "\n⚠️ INTERACTIONS:\n"
        for i in interactions:
            context += f"• {i.get('description', '')}\n"
    
    context += """

Provide a clinical assessment with these sections:
## 🔍 Clinical Assessment
## 📋 Key Research Findings
## 💡 Recommendations  
## ⚠️ Important Warnings

Note: Educational purposes only. Consult healthcare professionals for medical decisions."""
    
    return call_gemini(context, api_key)

def main():
    fda, pubmed, rxnorm = load_clients()
    
    # Get API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY")
        except:
            pass
    
    # Hero
    st.markdown("""
        <div class="hero">
            <h1>Clinical AI Assistant</h1>
            <p>Intelligent medical insights • Real-time data • Evidence-based</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Status
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<span class="status-pill status-on">● PubMed</span>', unsafe_allow_html=True)
    c2.markdown('<span class="status-pill status-on">● FDA</span>', unsafe_allow_html=True)
    c3.markdown('<span class="status-pill status-on">● RxNorm</span>', unsafe_allow_html=True)
    c4.markdown(f'<span class="status-pill {"status-on" if gemini_key else "status-off"}">● Gemini AI</span>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✨ AI Assistant", "📚 Research", "💊 Medications", "⚡ Interactions"])
    
    with tab1:
        st.markdown("""
            <div class="glass-card">
                <h3>Ask any medical question</h3>
                <p>Get AI insights backed by real-time PubMed, FDA & RxNorm data</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("👤 Patient Context (Optional)"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 0, 120, 0)
            gender = c2.selectbox("Gender", ["Not specified", "Male", "Female"])
            history = c1.text_input("Medical History", placeholder="Diabetes, HTN")
            meds = c2.text_input("Medications", placeholder="Metformin, Lisinopril")
        
        query = st.text_area("Your question", placeholder="What are treatment options for Type 2 diabetes with cardiovascular disease?", height=100, label_visibility="collapsed")
        
        if st.button("✨ Analyze", use_container_width=False):
            if query:
                patient = f"Age: {age}, Gender: {gender}, History: {history}" if age > 0 else ""
                med_list = [m.strip() for m in meds.split(',') if m.strip()] if meds else []
                
                with st.spinner("Searching medical databases..."):
                    research = pubmed.search_articles(query, 5).get('articles', [])
                    drug_info = {m: fda.get_drug_info_summary(m) for m in med_list} if med_list else {}
                    interactions = rxnorm.get_interactions(med_list).get('interactions', []) if len(med_list) >= 2 else []
                
                st.markdown(f'<div class="success-box">✅ Found {len(research)} articles{f" • {len(drug_info)} drugs" if drug_info else ""}</div>', unsafe_allow_html=True)
                
                if gemini_key:
                    with st.spinner("Generating AI analysis..."):
                        response = generate_response(query, research, drug_info, interactions, patient, gemini_key)
                    
                    if response and not response.startswith("AI Error"):
                        st.markdown(f'<div class="ai-card">{response}</div>', unsafe_allow_html=True)
                    else:
                        st.error(response)
                else:
                    st.markdown("""
                        <div class="warning-box">
                            ⚠️ Add GEMINI_API_KEY to enable AI insights<br>
                            <small>Free key: <a href="https://aistudio.google.com/apikey" target="_blank">Google AI Studio</a></small>
                        </div>
                    """, unsafe_allow_html=True)
                
                if research:
                    st.markdown("### 📚 Sources")
                    for a in research[:4]:
                        st.markdown(f"""
                            <div class="result-card">
                                <h5>{a['title']}</h5>
                                <p style="color:#8b5cf6;font-size:0.8rem;margin-bottom:6px;">{a['journal']} • {a['year']}</p>
                                <p>{a['abstract'][:220]}...</p>
                                <a href="{a['url']}" target="_blank" style="color:#8b5cf6;font-size:0.85rem;">View on PubMed →</a>
                            </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="glass-card"><h3>Search Medical Literature</h3><p>Access millions of PubMed articles</p></div>', unsafe_allow_html=True)
        q = st.text_input("Search", placeholder="SGLT2 inhibitors heart failure", label_visibility="collapsed", key="search_q")
        if st.button("🔍 Search", key="btn_search"):
            if q:
                with st.spinner("Searching..."):
                    results = pubmed.search_articles(q, 6).get('articles', [])
                if results:
                    st.markdown(f'<div class="success-box">Found {len(results)} articles</div>', unsafe_allow_html=True)
                    for a in results:
                        st.markdown(f'<div class="result-card"><h5>{a["title"]}</h5><p style="color:#8b5cf6;font-size:0.8rem;">{a["journal"]} • {a["year"]}</p><p>{a["abstract"][:180]}...</p><a href="{a["url"]}" target="_blank" style="color:#8b5cf6;">View →</a></div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="glass-card"><h3>Drug Information</h3><p>Official FDA drug data</p></div>', unsafe_allow_html=True)
        drug = st.text_input("Drug name", placeholder="Metformin", label_visibility="collapsed", key="drug_q")
        if st.button("🔍 Search", key="btn_drug"):
            if drug:
                with st.spinner("Fetching..."):
                    info = fda.get_drug_info_summary(drug)
                if info.get('found'):
                    st.markdown(f'<div class="success-box">✅ {drug.upper()}</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    if info.get('indications'): c1.info(f"**Indications**\n\n{info['indications'][0][:400]}...")
                    if info.get('warnings'): c2.warning(f"**Warnings**\n\n{info['warnings'][0][:400]}...")
                    if info.get('common_adverse_events'): st.write("**Side Effects:** " + ", ".join(info['common_adverse_events'][:10]))
                else:
                    st.warning("Not found. Try generic name.")
    
    with tab4:
        st.markdown('<div class="glass-card"><h3>Drug Interaction Checker</h3><p>Check dangerous combinations</p></div>', unsafe_allow_html=True)
        drugs = st.text_area("Medications (one per line)", placeholder="Warfarin\nAspirin", height=100, label_visibility="collapsed", key="inter_q")
        if st.button("⚡ Check", key="btn_inter"):
            dl = [d.strip() for d in drugs.split('\n') if d.strip()]
            if len(dl) >= 2:
                with st.spinner("Checking..."):
                    result = rxnorm.get_interactions(dl)
                if result.get('interactions'):
                    st.markdown(f'<div class="error-box">⚠️ {len(result["interactions"])} interaction(s) found</div>', unsafe_allow_html=True)
                    for i in result['interactions']:
                        st.markdown(f'<div class="result-card" style="border-color:rgba(239,68,68,0.25);"><h5 style="color:#ef4444;">⚠️ Warning</h5><p><b>Severity:</b> {i.get("severity","Unknown")}</p><p>{i.get("description","")}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ No known interactions</div>', unsafe_allow_html=True)
            else:
                st.warning("Enter at least 2 medications")
    
    st.markdown('<div style="text-align:center;padding:40px 0 20px;color:#475569;font-size:0.8rem;"><p>⚠️ Educational only. Consult healthcare professionals.</p><p>Built by Nitish Kumar Manthri</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
