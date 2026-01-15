"""
Clinical Decision Support Assistant - Beautiful UI
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.medical_apis import get_fda_client, get_pubmed_client, get_rxnorm_client, get_medical_aggregator
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Clinical Decision Support", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);}
.custom-card {background: linear-gradient(145deg, rgba(0,212,255,0.1), rgba(0,212,255,0.02)); border: 1px solid rgba(0,212,255,0.3); border-radius: 15px; padding: 25px; margin: 15px 0;}
.success-card {background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.4); border-radius: 12px; padding: 20px; margin: 10px 0;}
.warning-card {background: rgba(255,170,0,0.1); border: 1px solid rgba(255,170,0,0.4); border-radius: 12px; padding: 20px; margin: 10px 0;}
.error-card {background: rgba(255,71,87,0.1); border: 1px solid rgba(255,71,87,0.4); border-radius: 12px; padding: 20px; margin: 10px 0;}
.research-card {background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin: 10px 0;}
.stButton > button {background: linear-gradient(90deg, #00d4ff, #0099ff) !important; color: white !important; border: none !important; border-radius: 25px !important; font-weight: 600 !important;}
h1,h2,h3 {color: #00d4ff !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clients():
    return get_fda_client(), get_pubmed_client(), get_rxnorm_client(), get_medical_aggregator()

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try: key = st.secrets.get("OPENAI_API_KEY")
        except: pass
    return OpenAI(api_key=key) if key and OPENAI_AVAILABLE else None

def generate_response(client, query, ctx):
    if not client: return None
    txt = ""
    if ctx.get('research'):
        txt += "\n## Research:\n" + "\n".join([f"- {a['title']} ({a['year']})" for a in ctx['research'][:3]])
    if ctx.get('drug_info'):
        txt += "\n## Drugs:\n" + "\n".join([f"- {d}" for d in ctx['drug_info'].keys()])
    if ctx.get('interactions'):
        txt += "\n## Interactions:\n" + "\n".join([f"- {i.get('description','')}" for i in ctx['interactions']])
    try:
        r = client.chat.completions.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You are a Clinical AI. Give evidence-based insights with sections: 🔍 Assessment, 📋 Findings, 💡 Recommendations, ⚠️ Warnings. Note: consult real doctors."},
            {"role": "user", "content": f"Query: {query}\nData: {txt}"}
        ], temperature=0.3, max_tokens=1200)
        return r.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    st.markdown("<h1 style='text-align:center;'>🏥 Clinical Decision Support</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>AI-Powered • Real-Time Data • Evidence-Based</p>", unsafe_allow_html=True)
    
    fda, pubmed, rxnorm, agg = load_clients()
    llm = get_openai_client()
    
    with st.sidebar:
        st.markdown("### ⚡ Status")
        st.success("✅ openFDA"); st.success("✅ PubMed"); st.success("✅ RxNorm")
        st.success("✅ AI") if llm else st.warning("⚠️ AI - Add key")
        st.markdown("---")
        st.markdown("### 👤 Patient")
        age = st.number_input("Age", 0, 120, 0)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])
        history = st.text_area("History", placeholder="Diabetes, HTN")
        meds = st.text_area("Medications", placeholder="Metformin, Lisinopril")
        st.markdown("---")
        st.caption("⚠️ Educational only")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Query", "🔬 Research", "💊 Drugs", "⚠️ Interactions"])
    
    with tab1:
        st.markdown("<div class='custom-card'><h3>🤖 AI Clinical Assistant</h3><p style='color:#888'>Ask any medical question</p></div>", unsafe_allow_html=True)
        query = st.text_area("Your question:", placeholder="What are treatment options for Type 2 diabetes?", height=100)
        mlist = [m.strip() for m in meds.split(',') if m.strip()] if meds else []
        
        if st.button("🔍 Analyze", type="primary"):
            if query:
                with st.spinner("Analyzing..."):
                    research = pubmed.search_articles(query, 5).get('articles', [])
                    drug_info = {m: fda.get_drug_info_summary(m) for m in mlist} if mlist else {}
                    interactions = rxnorm.get_interactions(mlist).get('interactions', []) if len(mlist) >= 2 else []
                    
                    st.markdown(f"<div class='success-card'>✅ Found {len(research)} articles</div>", unsafe_allow_html=True)
                    
                    st.markdown("### 🤖 AI Assessment")
                    if llm:
                        resp = generate_response(llm, query, {'research': research, 'drug_info': drug_info, 'interactions': interactions})
                        st.markdown(f"<div class='custom-card'>{resp}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Add OPENAI_API_KEY for AI insights")
                    
                    if research:
                        st.markdown("### 📚 Sources")
                        for a in research[:3]:
                            with st.expander(a['title'][:60] + "..."):
                                st.write(f"**{a['journal']}** ({a['year']})")
                                st.write(a['abstract'])
                                st.markdown(f"[View]({a['url']})")
    
    with tab2:
        st.markdown("<div class='custom-card'><h3>🔬 PubMed Search</h3></div>", unsafe_allow_html=True)
        q = st.text_input("Search:", placeholder="diabetes guidelines 2024")
        if st.button("Search", key="s2"):
            if q:
                res = pubmed.search_articles(q, 5).get('articles', [])
                for a in res:
                    st.markdown(f"<div class='research-card'><b>{a['title']}</b><br><small>{a['journal']} ({a['year']})</small><br>{a['abstract'][:200]}... <a href='{a['url']}'>Link</a></div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='custom-card'><h3>💊 Drug Lookup</h3></div>", unsafe_allow_html=True)
        drug = st.text_input("Drug:", placeholder="Metformin")
        if st.button("Search", key="s3"):
            if drug:
                info = fda.get_drug_info_summary(drug)
                if info.get('found'):
                    st.success(f"Found: {drug}")
                    if info.get('indications'): st.info(info['indications'][0][:500])
                    if info.get('warnings'): st.warning(info['warnings'][0][:500])
                    if info.get('common_adverse_events'): st.write("Side effects: " + ", ".join(info['common_adverse_events'][:10]))
    
    with tab4:
        st.markdown("<div class='custom-card'><h3>⚠️ Interaction Checker</h3></div>", unsafe_allow_html=True)
        drugs = st.text_area("Drugs (one per line):", placeholder="Warfarin\nAspirin")
        if st.button("Check", key="s4"):
            dl = [d.strip() for d in drugs.split('\n') if d.strip()]
            if len(dl) >= 2:
                res = rxnorm.get_interactions(dl)
                if res.get('interactions'):
                    st.error(f"Found {len(res['interactions'])} interactions!")
                    for i in res['interactions']:
                        st.markdown(f"<div class='warning-card'>⚠️ {i.get('description')}</div>", unsafe_allow_html=True)
                else:
                    st.success("No interactions found")

if __name__ == "__main__":
    main()
