"""
Streamlit Dashboard - Clinical Decision Support System
Real-time medical data from openFDA, PubMed, and RxNorm APIs
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

st.set_page_config(
    page_title="Clinical Decision Support Assistant",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_clients():
    """Load API clients"""
    return (
        get_fda_client(),
        get_pubmed_client(),
        get_rxnorm_client(),
        get_medical_aggregator()
    )

def main():
    st.markdown("## 🏥 Clinical Decision Support Assistant")
    st.caption("Real-time medical data from openFDA, PubMed & RxNorm APIs")
    
    fda_client, pubmed_client, rxnorm_client, aggregator = load_clients()
    
    # Sidebar
    with st.sidebar:
        st.header("🔌 Data Sources")
        st.success("✅ openFDA - Drug Info")
        st.success("✅ PubMed - Research")
        st.success("✅ RxNorm - Interactions")
        
        st.divider()
        st.header("👤 Patient Info")
        age = st.number_input("Age", 0, 120, 0)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])
        medical_history = st.text_area("Medical History", placeholder="e.g., Diabetes, Hypertension")
        current_meds = st.text_area("Current Medications", placeholder="e.g., Metformin, Lisinopril")
        
        st.divider()
        st.caption("⚠️ For educational purposes only.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Research Search", 
        "💊 Drug Lookup", 
        "⚠️ Interaction Checker",
        "📋 Clinical Query"
    ])
    
    # Tab 1: PubMed Research Search
    with tab1:
        st.markdown("### 🔬 Search Medical Research (PubMed)")
        st.caption("Search millions of medical research articles in real-time")
        
        research_query = st.text_input(
            "Search medical literature:",
            placeholder="e.g., Type 2 diabetes treatment guidelines 2024"
        )
        
        num_results = st.slider("Number of results", 3, 10, 5)
        
        if st.button("🔍 Search PubMed", type="primary", key="pubmed_search"):
            if research_query:
                with st.spinner("Searching PubMed..."):
                    results = pubmed_client.search_articles(research_query, max_results=num_results)
                    
                    if results.get('articles'):
                        st.success(f"Found {len(results['articles'])} articles")
                        
                        for i, article in enumerate(results['articles'], 1):
                            with st.expander(f"📄 {article['title'][:80]}..." if len(article['title']) > 80 else f"📄 {article['title']}"):
                                st.markdown(f"**Journal:** {article['journal']} ({article['year']})")
                                st.markdown(f"**Abstract:** {article['abstract']}")
                                if article['url']:
                                    st.markdown(f"[🔗 View on PubMed]({article['url']})")
                    else:
                        st.warning("No articles found. Try different search terms.")
    
    # Tab 2: Drug Lookup (openFDA)
    with tab2:
        st.markdown("### 💊 Drug Information (openFDA)")
        st.caption("Real-time drug data from FDA database")
        
        drug_name = st.text_input(
            "Enter drug name:",
            placeholder="e.g., Metformin, Lisinopril, Atorvastatin"
        )
        
        if st.button("🔍 Search Drug", type="primary", key="drug_search"):
            if drug_name:
                with st.spinner(f"Fetching {drug_name} info from FDA..."):
                    info = fda_client.get_drug_info_summary(drug_name)
                    
                    if info.get('found'):
                        st.success(f"✅ Found: {drug_name}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if info.get('indications') and info['indications'][0] != 'Not available':
                                st.markdown("### 📋 Indications")
                                text = info['indications'][0]
                                st.write(text[:800] + "..." if len(text) > 800 else text)
                            
                            if info.get('dosage') and info['dosage'][0] != 'Not available':
                                st.markdown("### 💊 Dosage")
                                text = info['dosage'][0]
                                st.write(text[:800] + "..." if len(text) > 800 else text)
                        
                        with col2:
                            if info.get('warnings') and info['warnings'][0] != 'Not available':
                                st.markdown("### ⚠️ Warnings")
                                text = info['warnings'][0]
                                st.warning(text[:600] + "..." if len(text) > 600 else text)
                            
                            if info.get('contraindications') and info['contraindications'][0] != 'Not available':
                                st.markdown("### 🚫 Contraindications")
                                text = info['contraindications'][0]
                                st.error(text[:400] + "..." if len(text) > 400 else text)
                        
                        if info.get('common_adverse_events'):
                            st.markdown("### 💉 Reported Adverse Events")
                            st.write(", ".join(info['common_adverse_events'][:15]))
                        
                        if info.get('interactions') and info['interactions'][0] != 'Not available':
                            st.markdown("### 🔄 Drug Interactions")
                            text = info['interactions'][0]
                            st.info(text[:600] + "..." if len(text) > 600 else text)
                    else:
                        st.warning(f"No FDA data found for '{drug_name}'. Try the generic name.")
    
    # Tab 3: Drug Interaction Checker (RxNorm)
    with tab3:
        st.markdown("### ⚠️ Drug Interaction Checker (RxNorm)")
        st.caption("Check for interactions between multiple drugs using NIH RxNorm database")
        
        st.markdown("Enter drugs to check (one per line):")
        drugs_text = st.text_area(
            "Drugs:",
            placeholder="Warfarin\nAspirin\nIbuprofen",
            height=150,
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Check Interactions", type="primary", key="interaction_check"):
            if drugs_text:
                drug_list = [d.strip() for d in drugs_text.strip().split('\n') if d.strip()]
                
                if len(drug_list) < 2:
                    st.warning("Please enter at least 2 drugs to check interactions.")
                else:
                    with st.spinner(f"Checking interactions between {len(drug_list)} drugs..."):
                        # First verify drugs exist
                        st.markdown("#### 🔍 Verifying drugs...")
                        valid_drugs = []
                        for drug in drug_list:
                            info = rxnorm_client.get_drug_info(drug)
                            if info.get('found'):
                                st.write(f"✅ {drug} - Found")
                                valid_drugs.append(drug)
                            else:
                                st.write(f"❌ {drug} - Not found in RxNorm")
                        
                        if len(valid_drugs) >= 2:
                            st.markdown("#### ⚠️ Interaction Results")
                            interactions = rxnorm_client.get_interactions(valid_drugs)
                            
                            if interactions.get('interactions'):
                                st.error(f"Found {len(interactions['interactions'])} potential interaction(s)!")
                                
                                for inter in interactions['interactions']:
                                    with st.expander(f"⚠️ {' + '.join(inter.get('drugs', ['Unknown']))}"):
                                        st.markdown(f"**Severity:** {inter.get('severity', 'Unknown')}")
                                        st.markdown(f"**Description:** {inter.get('description', 'No description')}")
                            else:
                                st.success("✅ No known interactions found between these drugs.")
                        else:
                            st.warning("Need at least 2 valid drugs to check interactions.")
    
    # Tab 4: Clinical Query (Combined)
    with tab4:
        st.markdown("### 📋 Clinical Query")
        st.caption("Combines research + drug info + interactions")
        
        query = st.text_area(
            "Enter your clinical question:",
            placeholder="e.g., What is the first-line treatment for hypertension in diabetic patients?",
            height=100
        )
        
        # Get medications from sidebar
        medications = [m.strip() for m in current_meds.split(',')] if current_meds else []
        
        if st.button("🔍 Analyze", type="primary", key="clinical_query"):
            if query:
                with st.spinner("Gathering real-time medical data..."):
                    result = aggregator.clinical_query(query, medications if medications else None)
                    
                    # Show research
                    st.markdown("### 🔬 Relevant Research (PubMed)")
                    if result.get('research'):
                        for article in result['research'][:3]:
                            st.markdown(f"**📄 {article['title']}**")
                            st.caption(f"{article['journal']} ({article['year']})")
                            st.write(article['abstract'][:300] + "...")
                            if article['url']:
                                st.markdown(f"[View on PubMed]({article['url']})")
                            st.divider()
                    else:
                        st.info("No research articles found for this query.")
                    
                    # Show drug info if medications provided
                    if result.get('drug_info'):
                        st.markdown("### 💊 Medication Information (openFDA)")
                        for drug, info in result['drug_info'].items():
                            if info.get('found'):
                                with st.expander(f"💊 {drug}"):
                                    if info.get('warnings') and info['warnings'][0] != 'Not available':
                                        st.warning(f"**Warnings:** {info['warnings'][0][:300]}...")
                                    if info.get('common_adverse_events'):
                                        st.write(f"**Side effects:** {', '.join(info['common_adverse_events'][:5])}")
                    
                    # Show interactions
                    if result.get('interactions'):
                        st.markdown("### ⚠️ Drug Interactions (RxNorm)")
                        for inter in result['interactions']:
                            st.error(f"**{' + '.join(inter.get('drugs', []))}:** {inter.get('description', 'Unknown')}")
                    
                    # Patient context
                    if age > 0 or gender != "Not specified" or medical_history:
                        st.markdown("### 👤 Patient Context")
                        context = []
                        if age > 0: context.append(f"Age: {age}")
                        if gender != "Not specified": context.append(f"Gender: {gender}")
                        if medical_history: context.append(f"History: {medical_history}")
                        st.info(" | ".join(context))

if __name__ == "__main__":
    main()
