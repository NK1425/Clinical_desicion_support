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

# Page config
st.set_page_config(
    page_title="Clinical AI Assistant", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Apple-inspired CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 50%, #0a0a0a 100%);
    }
    
    /* Hide default elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 60px 20px;
        margin-bottom: 40px;
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 15px;
        letter-spacing: -1px;
    }
    
    .hero p {
        font-size: 1.3rem;
        color: #888;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 32px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.05));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .feature-card h4 {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .feature-card p {
        color: #888;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }
    
    .result-card h5 {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .result-card p {
        color: #aaa;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* AI Response card */
    .ai-response {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.08), rgba(240, 147, 251, 0.03));
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
    }
    
    .ai-response h3 {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Status pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 4px;
    }
    
    .status-online {
        background: rgba(52, 199, 89, 0.15);
        color: #34c759;
        border: 1px solid rgba(52, 199, 89, 0.3);
    }
    
    .status-offline {
        background: rgba(255, 149, 0, 0.15);
        color: #ff9500;
        border: 1px solid rgba(255, 149, 0, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 16px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        padding: 6px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #888;
        font-weight: 500;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Success/Warning/Error */
    .success-msg {
        background: rgba(52, 199, 89, 0.1);
        border: 1px solid rgba(52, 199, 89, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        color: #34c759;
        font-weight: 500;
    }
    
    .warning-msg {
        background: rgba(255, 149, 0, 0.1);
        border: 1px solid rgba(255, 149, 0, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        color: #ff9500;
    }
    
    .error-msg {
        background: rgba(255, 59, 48, 0.1);
        border: 1px solid rgba(255, 59, 48, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        color: #ff3b30;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Loading animation */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .loading {
        background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 12px;
        height: 20px;
        margin: 8px 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clients():
    return get_fda_client(), get_pubmed_client(), get_rxnorm_client()

def call_gemini_api(prompt, api_key):
    """Call Google Gemini API (Free!)"""
    if not api_key:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1500
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"AI Error: {str(e)}"

def generate_clinical_response(query, research, drug_info, interactions, patient_info, api_key):
    """Generate clinical response using Gemini"""
    
    context = f"""You are an expert Clinical Decision Support AI Assistant.

QUERY: {query}

PATIENT INFO: {patient_info if patient_info else 'Not provided'}

REAL-TIME DATA:

📚 PUBMED RESEARCH ({len(research)} articles found):
"""
    for i, article in enumerate(research[:3], 1):
        context += f"""
{i}. {article['title']} ({article['year']})
   Journal: {article['journal']}
   Key Finding: {article['abstract'][:200]}...
"""
    
    if drug_info:
        context += "\n💊 FDA DRUG INFORMATION:\n"
        for drug, info in drug_info.items():
            if info.get('found'):
                context += f"• {drug}: {info.get('indications', [''])[0][:150]}...\n"
    
    if interactions:
        context += "\n⚠️ DRUG INTERACTIONS:\n"
        for inter in interactions:
            context += f"• {inter.get('description', 'Unknown')}\n"
    
    context += """

Based on this real-time medical data, provide a comprehensive clinical assessment.

FORMAT YOUR RESPONSE AS:

## 🔍 Clinical Assessment
[Brief overview of the clinical situation]

## 📋 Key Findings from Research
[Summarize the most relevant research findings]

## 💡 Evidence-Based Recommendations
[Specific, actionable recommendations]

## ⚠️ Important Considerations
[Warnings, contraindications, or special considerations]

## 📚 References
[List the research sources used]

IMPORTANT: Always note that this is for educational purposes and final decisions must be made by qualified healthcare professionals.
"""
    
    return call_gemini_api(context, api_key)

def main():
    fda, pubmed, rxnorm = load_clients()
    
    # Get API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY")
        except:
            pass
    
    # Hero Section
    st.markdown("""
        <div class="hero">
            <h1>Clinical AI Assistant</h1>
            <p>Intelligent medical insights powered by real-time data</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<span class="status-pill status-online">● PubMed</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-pill status-online">● FDA</span>', unsafe_allow_html=True)
    with col3:
        st.markdown('<span class="status-pill status-online">● RxNorm</span>', unsafe_allow_html=True)
    with col4:
        if gemini_key:
            st.markdown('<span class="status-pill status-online">● Gemini AI</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-offline">● AI (Add Key)</span>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✨ AI Assistant", "📚 Research", "💊 Medications", "⚡ Interactions"])
    
    # Tab 1: AI Assistant
    with tab1:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #fff; margin: 0 0 8px 0; font-weight: 600;">Ask anything about medicine</h3>
                <p style="color: #666; margin: 0; font-size: 0.95rem;">Get AI-powered insights backed by real-time research from PubMed, FDA, and RxNorm</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Patient info (collapsible)
        with st.expander("👤 Add Patient Context (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 0, 120, 0)
                gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])
            with col2:
                history = st.text_input("Medical History", placeholder="Diabetes, Hypertension")
                meds = st.text_input("Current Medications", placeholder="Metformin, Lisinopril")
        
        # Query input
        query = st.text_area(
            "Your clinical question",
            placeholder="Example: What are the current treatment guidelines for Type 2 diabetes in patients with cardiovascular disease?",
            height=100,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze = st.button("✨ Analyze", use_container_width=True)
        
        if analyze and query:
            # Build patient info
            patient_info = ""
            if age > 0: patient_info += f"Age: {age}, "
            if gender != "Not specified": patient_info += f"Gender: {gender}, "
            if history: patient_info += f"History: {history}, "
            if meds: patient_info += f"Medications: {meds}"
            
            med_list = [m.strip() for m in meds.split(',') if m.strip()] if meds else []
            
            # Progress
            progress = st.empty()
            
            progress.markdown("""
                <div class="glass-card" style="text-align: center;">
                    <p style="color: #667eea; margin: 0;">🔍 Searching medical databases...</p>
                    <div class="loading"></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Fetch data
            research = pubmed.search_articles(query, 5).get('articles', [])
            drug_info = {m: fda.get_drug_info_summary(m) for m in med_list} if med_list else {}
            interactions = rxnorm.get_interactions(med_list).get('interactions', []) if len(med_list) >= 2 else []
            
            progress.empty()
            
            # Results summary
            st.markdown(f"""
                <div class="success-msg">
                    ✅ Found {len(research)} research articles
                    {f' • {len(drug_info)} medications analyzed' if drug_info else ''}
                    {f' • {len(interactions)} interactions detected' if interactions else ''}
                </div>
            """, unsafe_allow_html=True)
            
            # AI Response
            if gemini_key:
                with st.spinner(""):
                    ai_response = generate_clinical_response(
                        query, research, drug_info, interactions, patient_info, gemini_key
                    )
                
                if ai_response and not ai_response.startswith("AI Error"):
                    st.markdown(f"""
                        <div class="ai-response">
                            <h3>🤖 AI Clinical Assessment</h3>
                            {ai_response}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(ai_response)
            else:
                st.markdown("""
                    <div class="warning-msg">
                        ⚠️ Add your free Gemini API key to enable AI insights.<br>
                        <small>Get your free key at: <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></small>
                    </div>
                """, unsafe_allow_html=True)
            
            # Research sources
            if research:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📚 Research Sources")
                
                for article in research[:4]:
                    st.markdown(f"""
                        <div class="result-card">
                            <h5>{article['title']}</h5>
                            <p style="color: #667eea; font-size: 0.85rem; margin-bottom: 8px;">
                                {article['journal']} • {article['year']}
                            </p>
                            <p>{article['abstract'][:250]}...</p>
                            <a href="{article['url']}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">
                                View on PubMed →
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
    
    # Tab 2: Research
    with tab2:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #fff; margin: 0 0 8px 0;">Search Medical Literature</h3>
                <p style="color: #666; margin: 0;">Access millions of peer-reviewed articles from PubMed</p>
            </div>
        """, unsafe_allow_html=True)
        
        search_q = st.text_input("Search PubMed", placeholder="e.g., SGLT2 inhibitors heart failure", label_visibility="collapsed")
        
        if st.button("🔍 Search", key="search_research"):
            if search_q:
                with st.spinner("Searching..."):
                    results = pubmed.search_articles(search_q, 8).get('articles', [])
                
                if results:
                    st.markdown(f'<div class="success-msg">Found {len(results)} articles</div>', unsafe_allow_html=True)
                    
                    for article in results:
                        st.markdown(f"""
                            <div class="result-card">
                                <h5>{article['title']}</h5>
                                <p style="color: #667eea; font-size: 0.85rem;">{article['journal']} • {article['year']}</p>
                                <p>{article['abstract'][:200]}...</p>
                                <a href="{article['url']}" target="_blank" style="color: #667eea;">View →</a>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No results found")
    
    # Tab 3: Medications
    with tab3:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #fff; margin: 0 0 8px 0;">Drug Information</h3>
                <p style="color: #666; margin: 0;">Official FDA drug data and safety information</p>
            </div>
        """, unsafe_allow_html=True)
        
        drug_q = st.text_input("Enter medication name", placeholder="e.g., Metformin", label_visibility="collapsed")
        
        if st.button("🔍 Search", key="search_drug"):
            if drug_q:
                with st.spinner("Fetching FDA data..."):
                    info = fda.get_drug_info_summary(drug_q)
                
                if info.get('found'):
                    st.markdown(f'<div class="success-msg">✅ Found: {drug_q.upper()}</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if info.get('indications'):
                            st.markdown("**📋 Indications**")
                            st.info(info['indications'][0][:500] + "...")
                        
                        if info.get('dosage'):
                            st.markdown("**💊 Dosage**")
                            st.info(info['dosage'][0][:500] + "...")
                    
                    with col2:
                        if info.get('warnings'):
                            st.markdown("**⚠️ Warnings**")
                            st.warning(info['warnings'][0][:400] + "...")
                        
                        if info.get('contraindications'):
                            st.markdown("**🚫 Contraindications**")
                            st.error(info['contraindications'][0][:300] + "...")
                    
                    if info.get('common_adverse_events'):
                        st.markdown("**Common Side Effects:**")
                        st.write(", ".join(info['common_adverse_events'][:12]))
                else:
                    st.warning(f"No FDA data for '{drug_q}'. Try the generic name.")
    
    # Tab 4: Interactions
    with tab4:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color: #fff; margin: 0 0 8px 0;">Drug Interaction Checker</h3>
                <p style="color: #666; margin: 0;">Check for potentially dangerous drug combinations</p>
            </div>
        """, unsafe_allow_html=True)
        
        drugs_input = st.text_area("Enter medications (one per line)", placeholder="Warfarin\nAspirin\nIbuprofen", height=120, label_visibility="collapsed")
        
        if st.button("⚡ Check Interactions", key="check_inter"):
            if drugs_input:
                drug_list = [d.strip() for d in drugs_input.split('\n') if d.strip()]
                
                if len(drug_list) >= 2:
                    with st.spinner("Checking..."):
                        result = rxnorm.get_interactions(drug_list)
                    
                    if result.get('interactions'):
                        st.markdown(f"""
                            <div class="error-msg">
                                ⚠️ Found {len(result['interactions'])} potential interaction(s)
                            </div>
                        """, unsafe_allow_html=True)
                        
                        for inter in result['interactions']:
                            st.markdown(f"""
                                <div class="result-card" style="border-color: rgba(255, 59, 48, 0.3);">
                                    <h5 style="color: #ff3b30;">⚠️ Interaction Warning</h5>
                                    <p><strong>Severity:</strong> {inter.get('severity', 'Unknown')}</p>
                                    <p>{inter.get('description', 'No details available')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-msg">✅ No known interactions found</div>', unsafe_allow_html=True)
                else:
                    st.warning("Enter at least 2 medications")
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0; color: #444; font-size: 0.85rem;">
            <p>⚠️ For educational purposes only. Always consult healthcare professionals.</p>
            <p style="margin-top: 8px;">Built with ❤️ by Nitish Kumar Manthri</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
