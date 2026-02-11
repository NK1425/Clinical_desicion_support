"""
System Monitoring Dashboard
Real-time health, performance, and knowledge base statistics.
"""
import streamlit as st
import requests
import time
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(page_title="CDSS Monitoring", page_icon="📊", layout="wide")

# ── Custom CSS matching the main app style ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.monitor-header {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.5rem;
}
.monitor-subtitle {
    text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 1rem;
}
.status-card {
    background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0;
}
.status-online { color: #22c55e; font-weight: 600; }
.status-offline { color: #ef4444; font-weight: 600; }
.status-warning { color: #eab308; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def check_api_status(name, url, timeout=5):
    """Check if an API endpoint is reachable."""
    try:
        resp = requests.get(url, timeout=timeout)
        return {
            "status": "online",
            "response_time_ms": round(resp.elapsed.total_seconds() * 1000),
            "status_code": resp.status_code,
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "response_time_ms": timeout * 1000, "status_code": None}
    except requests.exceptions.RequestException:
        return {"status": "offline", "response_time_ms": None, "status_code": None}


def check_llm_status():
    """Check LLM availability."""
    groq_key = None
    openai_key = None
    gemini_key = None

    # Check Streamlit secrets
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    # Check environment
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY", "")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY", "")
    if not gemini_key:
        gemini_key = os.getenv("GEMINI_API_KEY", "")

    providers = []
    if groq_key:
        providers.append(("Groq (Llama 3.3 70B)", "llama-3.3-70b-versatile"))
    if openai_key:
        providers.append(("OpenAI (GPT-4o-mini)", "gpt-4o-mini"))
    if gemini_key:
        providers.append(("Google Gemini", "gemini-1.5-flash"))

    return providers


def main():
    st.markdown('<p class="monitor-header">📊 System Monitoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="monitor-subtitle">Real-time health, performance, and knowledge base metrics</p>', unsafe_allow_html=True)

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    # ── External API Health Checks ──
    st.markdown("### 🌐 External API Health")
    st.caption("Live connectivity checks to all medical data sources")

    apis_to_check = [
        ("openFDA", "https://api.fda.gov/drug/label.json?limit=1"),
        ("PubMed (NCBI)", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=test&retmode=json&retmax=1"),
        ("RxNorm", "https://rxnav.nlm.nih.gov/REST/drugs.json?name=aspirin"),
        ("ClinicalTrials.gov", "https://clinicaltrials.gov/api/v2/studies?pageSize=1&format=json"),
        ("Zippopotam.us (Geo)", "https://api.zippopotam.us/us/10001"),
    ]

    if st.button("🔄 Run Health Checks", type="primary"):
        cols = st.columns(len(apis_to_check))
        for i, (name, url) in enumerate(apis_to_check):
            with cols[i]:
                with st.spinner(f"Checking {name}..."):
                    result = check_api_status(name, url)

                if result["status"] == "online" and result["status_code"] == 200:
                    st.success(f"**{name}**")
                    st.metric("Response", f"{result['response_time_ms']}ms")
                elif result["status"] == "timeout":
                    st.warning(f"**{name}**")
                    st.caption("Timeout")
                else:
                    st.error(f"**{name}**")
                    st.caption(f"Status: {result['status_code'] or 'unreachable'}")
    else:
        st.info("Click **Run Health Checks** to test connectivity to all medical APIs.")

    st.divider()

    # ── LLM Provider Status ──
    st.markdown("### 🤖 LLM Provider Status")
    providers = check_llm_status()

    if providers:
        cols = st.columns(len(providers))
        for i, (name, model) in enumerate(providers):
            with cols[i]:
                st.success(f"**{name}**")
                st.caption(f"Model: `{model}`")
        st.caption(f"Fallback chain: {' → '.join(p[0].split(' (')[0] for p in providers)} → Retrieval-only")
    else:
        st.warning("No LLM API keys configured. The app will work in retrieval-only mode.")
        st.caption("Set `GROQ_API_KEY` in Streamlit secrets or environment for AI-powered responses.")

    st.divider()

    # ── Knowledge Base Stats ──
    st.markdown("### 📚 Knowledge Base")

    try:
        from src.vector_store import get_vector_store
        vs = get_vector_store()
        doc_count = vs.count if hasattr(vs, "count") else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", doc_count)
        col2.metric("Embedding Model", "all-MiniLM-L6-v2")
        col3.metric("Vector Store", "FAISS")

        if doc_count > 0:
            try:
                from src.data_ingestion import get_ingestion_stats
                local_stats = get_ingestion_stats(vs)
                if local_stats.get("categories"):
                    st.subheader("Documents by Category")
                    import pandas as pd
                    cat_df = pd.DataFrame(
                        list(local_stats["categories"].items()),
                        columns=["Category", "Count"],
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(cat_df.set_index("Category"))
            except Exception:
                pass
    except Exception:
        st.info("Vector store not initialized. Run `python src/init_vectorstore.py` to build the knowledge base.")

    st.divider()

    # ── System Info ──
    st.markdown("### ⚙️ System Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Platform", sys.platform)
    col2.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check key packages
    pkg_status = {}
    for pkg in ["langchain", "faiss", "sentence_transformers", "groq", "streamlit"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "installed")
            pkg_status[pkg] = ver
        except ImportError:
            pkg_status[pkg] = "not installed"

    col3.metric("LangChain", pkg_status.get("langchain", "?"))
    col4.metric("Streamlit", pkg_status.get("streamlit", "?"))

    with st.expander("All Package Versions"):
        for pkg, ver in pkg_status.items():
            icon = "✅" if ver != "not installed" else "❌"
            st.text(f"{icon} {pkg}: {ver}")

    st.divider()

    # ── Quick API Test ──
    st.markdown("### 🧪 Quick Drug Lookup Test")
    st.caption("Test the openFDA API directly from here")

    test_drug = st.text_input("Drug name:", placeholder="e.g., metformin")
    if st.button("Test Lookup") and test_drug:
        with st.spinner(f"Looking up {test_drug}..."):
            try:
                resp = requests.get(
                    "https://api.fda.gov/drug/label.json",
                    params={"search": f'openfda.generic_name:"{test_drug}"', "limit": 1},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results"):
                        result = data["results"][0]
                        st.success(f"Found: {test_drug}")
                        if result.get("indications_and_usage"):
                            st.markdown("**Indications:**")
                            st.write(result["indications_and_usage"][0][:300] + "...")
                    else:
                        st.warning(f"No results for '{test_drug}'")
                else:
                    st.error(f"API returned status {resp.status_code}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    # Footer
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not for actual medical decisions.")


if __name__ == "__main__":
    main()
