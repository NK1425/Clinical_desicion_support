"""
Clinical Decision Support Assistant
Premium Design + 3 LLM Options (OpenAI, Gemini, Groq) with Auto-Fallback
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

def get_api_keys():
    """Get API keys from Streamlit secrets"""
    keys = {
        'openai': None,
        'gemini': None,
        'groq': None
    }
    
    # Try Streamlit secrets
    try:
        keys['openai'] = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    except:
        pass
    
    try:
        keys['gemini'] = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("gemini_api_key")
    except:
        pass
    
    try:
        keys['groq'] = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    except:
        pass
    
    # Also try environment variables
    keys['openai'] = keys['openai'] or os.getenv("OPENAI_API_KEY")
    keys['gemini'] = keys['gemini'] or os.getenv("GEMINI_API_KEY")
    keys['groq'] = keys['groq'] or os.getenv("GROQ_API_KEY")
    
    return keys

def call_openai(prompt, api_key):
    """Call OpenAI API"""
    if not api_key:
        return None, "No OpenAI key"
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1500
        }, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], None
    except Exception as e:
        return None, f"OpenAI Error: {str(e)}"

def call_groq(prompt, api_key):
    """Call Groq API (Free & Fast!)"""
    if not api_key:
        return None, "No Groq key"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1500
        }, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], None
    except Exception as e:
        return None, f"Groq Error: {str(e)}"

def call_gemini(prompt, api_key):
    """Call Gemini API"""
    if not api_key:
        return None, "No Gemini key"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}
        }, timeout=30)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'], None
    except Exception as e:
        return None, f"Gemini Error: {str(e)}"

def call_llm_with_fallback(prompt, keys):
    """Try all LLMs in order: Groq (fastest) -> OpenAI -> Gemini"""
    errors = []
    
    # Try Groq first (fastest, free)
    if keys['groq']:
        result, error = call_groq(prompt, keys['groq'])
        if result:
            return result, "Groq (Llama 3.1 70B)"
        errors.append(error)
    
    # Try OpenAI second
    if keys['openai']:
        result, error = call_openai(prompt, keys['openai'])
        if result:
            return result, "OpenAI (GPT-3.5)"
        errors.append(error)
    
    # Try Gemini last
    if keys['gemini']:
        result, error = call_gemini(prompt, keys['gemini'])
        if result:
            return result, "Google Gemini"
        errors.append(error)
    
    # All failed
    return None, errors

def build_prompt(query, research, drug_info, interactions, patient):
    """Build the clinical prompt"""
    prompt = f"""You are an expert Clinical Decision Support AI Assistant.

CLINICAL QUERY: {query}

PATIENT INFORMATION: {patient if patient else 'Not provided'}

📚 RELEVANT RESEARCH FROM PUBMED ({len(research)} articles found):
"""
    for i, a in enumerate(research[:3], 1):
        prompt += f"""
{i}. "{a['title']}" 
   Journal: {a['journal']} ({a['year']})
   Key Finding: {a['abstract'][:200]}...
"""
    
    if drug_info:
        prompt += "\n💊 MEDICATION INFORMATION FROM FDA:\n"
        for d, info in drug_info.items():
            if info.get('found'):
                ind = info.get('indications', [''])[0][:150] if info.get('indications') else ''
                warn = info.get('warnings', [''])[0][:100] if info.get('warnings') else ''
                prompt += f"• {d.upper()}: {ind}... Warnings: {warn}...\n"
    
    if interactions:
        prompt += "\n⚠️ DRUG INTERACTIONS DETECTED:\n"
        for inter in interactions:
            prompt += f"• {inter.get('description', 'Unknown interaction')}\n"
    
    prompt += """

Based on this real-time medical data, provide a comprehensive clinical assessment.

FORMAT YOUR RESPONSE WITH THESE SECTIONS:

## 🔍 Clinical Assessment
[Brief overview of the clinical situation based on the query and patient info]

## 📋 Key Research Findings
[Summarize the most relevant findings from the PubMed articles]

## 💡 Evidence-Based Recommendations
[Specific, actionable recommendations based on current guidelines]

## ⚠️ Important Warnings & Considerations
[Any warnings, contraindications, drug interactions, or special considerations]

IMPORTANT: Always emphasize that this is for educational purposes only and final medical decisions must be made by qualified healthcare professionals."""
    
    return prompt

def main():
    fda, pubmed, rxnorm = load_clients()
    keys = get_api_keys()
    
    has_any_key = any([keys['openai'], keys['gemini'], keys['groq']])
    
    # Hero
    st.markdown("""
        <div class="hero">
            <h1>Clinical AI Assistant</h1>
            <p>Intelligent medical insights • Real-time data • Evidence-based</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Status pills
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<span class="status-pill status-on">● PubMed</span>', unsafe_allow_html=True)
    c2.markdown('<span class="status-pill status-on">● FDA</span>', unsafe_allow_html=True)
    c3.markdown('<span class="status-pill status-on">● RxNorm</span>', unsafe_allow_html=True)
    
    # Show which AI is available
    ai_available = []
    if keys['groq']: ai_available.append("Groq")
    if keys['openai']: ai_available.append("OpenAI")
    if keys['gemini']: ai_available.append("Gemini")
    
    if ai_available:
        c4.markdown(f'<span class="status-pill status-on">● AI ({", ".join(ai_available)})</span>', unsafe_allow_html=True)
    else:
        c4.markdown('<span class="status-pill status-off">● AI (No keys)</span>', unsafe_allow_html=True)
    
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
            history = c1.text_input("Medical History", placeholder="Diabetes, HTN, Asthma")
            meds = c2.text_input("Medications", placeholder="Metformin, Lisinopril")
        
        query = st.text_area(
            "Your question", 
            placeholder="What are the current treatment guidelines for Type 2 diabetes in patients with cardiovascular disease?", 
            height=100, 
            label_visibility="collapsed"
        )
        
        if st.button("✨ Analyze", use_container_width=False):
            if query:
                # Build patient info
                patient = ""
                if age > 0: patient += f"Age: {age}, "
                if gender != "Not specified": patient += f"Gender: {gender}, "
                if history: patient += f"Medical History: {history}, "
                if meds: patient += f"Current Medications: {meds}"
                
                med_list = [m.strip() for m in meds.split(',') if m.strip()] if meds else []
                
                # Fetch data from APIs
                with st.spinner("🔍 Searching medical databases..."):
                    research = pubmed.search_articles(query, 5).get('articles', [])
                    drug_info = {m: fda.get_drug_info_summary(m) for m in med_list} if med_list else {}
                    interactions = rxnorm.get_interactions(med_list).get('interactions', []) if len(med_list) >= 2 else []
                
                # Show results summary
                summary = f"✅ Found {len(research)} research articles"
                if drug_info: summary += f" • {len(drug_info)} medications analyzed"
                if interactions: summary += f" • {len(interactions)} interactions detected"
                st.markdown(f'<div class="success-box">{summary}</div>', unsafe_allow_html=True)
                
                # Generate AI response
                if has_any_key:
                    with st.spinner("🤖 Generating AI analysis..."):
                        prompt = build_prompt(query, research, drug_info, interactions, patient)
                        response, provider = call_llm_with_fallback(prompt, keys)
                    
                    if response:
                        st.markdown(f"""
                            <div class="ai-card">
                                <small style="color:#64748b;">✨ Powered by {provider}</small>
                                <br><br>
                                {response}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"All AI providers failed. Errors: {provider}")
                else:
                    st.markdown("""
                        <div class="warning-box">
                            ⚠️ <b>No AI keys configured.</b> Add at least one API key in Streamlit Secrets:<br><br>
                            <code>GROQ_API_KEY = "gsk_..."</code> (Free: console.groq.com/keys)<br>
                            <code>OPENAI_API_KEY = "sk-..."</code> (Paid: platform.openai.com)<br>
                            <code>GEMINI_API_KEY = "AI..."</code> (Free: aistudio.google.com/apikey)
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show sources
                if research:
                    st.markdown("### 📚 Research Sources")
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
        st.markdown('<div class="glass-card"><h3>Search Medical Literature</h3><p>Access millions of PubMed articles in real-time</p></div>', unsafe_allow_html=True)
        q = st.text_input("Search", placeholder="SGLT2 inhibitors heart failure outcomes", label_visibility="collapsed", key="search_q")
        if st.button("🔍 Search", key="btn_search"):
            if q:
                with st.spinner("Searching PubMed..."):
                    results = pubmed.search_articles(q, 8).get('articles', [])
                if results:
                    st.markdown(f'<div class="success-box">Found {len(results)} articles</div>', unsafe_allow_html=True)
                    for a in results:
                        st.markdown(f'<div class="result-card"><h5>{a["title"]}</h5><p style="color:#8b5cf6;font-size:0.8rem;">{a["journal"]} • {a["year"]}</p><p>{a["abstract"][:200]}...</p><a href="{a["url"]}" target="_blank" style="color:#8b5cf6;">View →</a></div>', unsafe_allow_html=True)
                else:
                    st.warning("No articles found. Try different search terms.")
    
    with tab3:
        st.markdown('<div class="glass-card"><h3>Drug Information</h3><p>Official FDA drug data and safety information</p></div>', unsafe_allow_html=True)
        drug = st.text_input("Drug name", placeholder="Metformin, Lisinopril, Atorvastatin", label_visibility="collapsed", key="drug_q")
        if st.button("🔍 Search", key="btn_drug"):
            if drug:
                with st.spinner("Fetching FDA data..."):
                    info = fda.get_drug_info_summary(drug)
                if info.get('found'):
                    st.markdown(f'<div class="success-box">✅ Found: {drug.upper()}</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    if info.get('indications'): 
                        c1.markdown("**📋 Indications**")
                        c1.info(info['indications'][0][:500] + "...")
                    if info.get('warnings'): 
                        c2.markdown("**⚠️ Warnings**")
                        c2.warning(info['warnings'][0][:500] + "...")
                    if info.get('contraindications'):
                        st.markdown("**🚫 Contraindications**")
                        st.error(info['contraindications'][0][:400] + "...")
                    if info.get('common_adverse_events'): 
                        st.markdown("**Common Side Effects:** " + ", ".join(info['common_adverse_events'][:12]))
                else:
                    st.warning(f"No FDA data found for '{drug}'. Try the generic name.")
    
    with tab4:
        st.markdown('<div class="glass-card"><h3>Drug Interaction Checker</h3><p>Check for potentially dangerous drug combinations</p></div>', unsafe_allow_html=True)
        drugs = st.text_area("Medications (one per line)", placeholder="Warfarin\nAspirin\nIbuprofen", height=120, label_visibility="collapsed", key="inter_q")
        if st.button("⚡ Check Interactions", key="btn_inter"):
            dl = [d.strip() for d in drugs.split('\n') if d.strip()]
            if len(dl) >= 2:
                with st.spinner("Checking RxNorm database..."):
                    result = rxnorm.get_interactions(dl)
                if result.get('interactions'):
                    st.markdown(f'<div class="error-box">⚠️ Found {len(result["interactions"])} potential interaction(s)!</div>', unsafe_allow_html=True)
                    for i in result['interactions']:
                        st.markdown(f"""
                            <div class="result-card" style="border-color:rgba(239,68,68,0.25);">
                                <h5 style="color:#ef4444;">⚠️ Interaction Warning</h5>
                                <p><b>Severity:</b> {i.get("severity","Unknown")}</p>
                                <p>{i.get("description","No description available")}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ No known interactions found between these medications.</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter at least 2 medications to check for interactions.")
    
    # Footer
    st.markdown("""
        <div style="text-align:center;padding:40px 0 20px;color:#475569;font-size:0.8rem;">
            <p>⚠️ For educational purposes only. Always consult qualified healthcare professionals.</p>
            <p>Built by Nitish Kumar Manthri</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
