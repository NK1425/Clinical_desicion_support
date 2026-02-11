"""
Initialize Vector Store
Loads curated medical guidelines and optional PubMed abstracts into the FAISS vector store.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore
from src.data_ingestion import ingest_directory, ingest_inline_documents, get_ingestion_stats
from src.config import settings
from src.logging_config import get_logger

log = get_logger("init_vectorstore")

# Core medical guidelines embedded directly for baseline knowledge
CORE_MEDICAL_KNOWLEDGE = [
    {
        "content": """Type 2 Diabetes Management Guidelines:
        - First-line therapy: Metformin is recommended as initial pharmacologic therapy for type 2 diabetes.
        - Target HbA1c: Generally less than 7% for most adults, individualized based on patient factors.
        - Lifestyle modifications: Medical nutrition therapy and physical activity (150 min/week moderate intensity).
        - Monitoring: HbA1c testing at least twice yearly if stable, quarterly if therapy changed.
        - Complications screening: Annual dilated eye exam, annual foot exam, annual urine albumin-to-creatinine ratio.""",
        "metadata": {"source": "ADA Standards of Care", "category": "endocrinology", "condition": "diabetes"}
    },
    {
        "content": """Metformin Prescribing Information:
        - Starting dose: 500mg once or twice daily with meals
        - Maximum dose: 2550mg daily in divided doses
        - Contraindications: eGFR <30 mL/min, acute/chronic metabolic acidosis
        - Hold before contrast: Discontinue at time of or before iodinated contrast procedures
        - Common side effects: GI upset, diarrhea, nausea (usually transient)
        - Rare but serious: Lactic acidosis (risk factors: renal impairment, hepatic disease, hypoxia)""",
        "metadata": {"source": "FDA Drug Label", "category": "pharmacology", "drug": "metformin"}
    },
    {
        "content": """Hypertension Management Guidelines:
        - Stage 1 HTN: BP 130-139/80-89 mmHg - lifestyle modification, consider medication if high ASCVD risk
        - Stage 2 HTN: BP >=140/90 mmHg - medication recommended in addition to lifestyle changes
        - First-line agents: ACE inhibitors, ARBs, calcium channel blockers, thiazide diuretics
        - Target BP: <130/80 mmHg for most adults with confirmed hypertension
        - Lifestyle: DASH diet, sodium restriction (<2300mg/day), regular exercise, weight management""",
        "metadata": {"source": "ACC/AHA Hypertension Guidelines", "category": "cardiology", "condition": "hypertension"}
    },
    {
        "content": """ACE Inhibitor Considerations:
        - Common ACE inhibitors: Lisinopril, Enalapril, Ramipril, Benazepril
        - Contraindications: Pregnancy, history of angioedema, bilateral renal artery stenosis
        - Monitor: Potassium levels, renal function (especially first few weeks)
        - Drug interactions: NSAIDs may reduce efficacy, potassium-sparing diuretics increase hyperkalemia risk
        - Common side effects: Dry cough (10-15%), hyperkalemia, acute kidney injury
        - Serious adverse effects: Angioedema (0.1-0.2%), especially higher risk in Black patients""",
        "metadata": {"source": "Clinical Pharmacology", "category": "pharmacology", "drug_class": "ACE inhibitors"}
    },
    {
        "content": """Chest Pain Evaluation - Acute Coronary Syndrome:
        - STEMI: ST elevation >=1mm in 2 contiguous leads, activate cath lab immediately
        - NSTEMI: Elevated troponin without ST elevation, risk stratify with TIMI/GRACE score
        - Unstable Angina: Negative troponin, concerning symptoms (rest pain, new/worsening)
        - Initial management: Aspirin 325mg, anticoagulation, consider P2Y12 inhibitor
        - High-risk features: Ongoing chest pain, hemodynamic instability, arrhythmias, heart failure
        - Time-sensitive: Door-to-balloon time <90 minutes for STEMI""",
        "metadata": {"source": "ACC/AHA STEMI/NSTEMI Guidelines", "category": "cardiology", "condition": "ACS"}
    },
    {
        "content": """Heart Failure Management:
        - HFrEF (EF <=40%): Guideline-directed medical therapy (GDMT) includes ACEi/ARB/ARNI, beta-blocker, MRA, SGLT2i
        - HFpEF (EF >=50%): SGLT2 inhibitors, diuretics for congestion, treat underlying conditions
        - Monitoring: Daily weights, sodium restriction (<2g/day), fluid restriction if severe
        - Warning signs: Weight gain >3 lbs in 1 day or >5 lbs in 1 week, worsening dyspnea, edema
        - Avoid: NSAIDs, non-dihydropyridine CCBs (HFrEF), excessive fluid intake""",
        "metadata": {"source": "ACC/AHA Heart Failure Guidelines", "category": "cardiology", "condition": "heart failure"}
    },
    {
        "content": """COPD Management Guidelines:
        - GOLD Classification: Based on FEV1% predicted and symptoms (CAT/mMRC scores)
        - Initial therapy: LAMA or LABA monotherapy for most patients
        - Escalation: LAMA + LABA if persistent symptoms, add ICS if eosinophils >=300
        - Acute exacerbation: Short-acting bronchodilators, systemic corticosteroids (5-7 days), antibiotics if purulent sputum
        - Prevention: Smoking cessation (most important), influenza and pneumococcal vaccines
        - Oxygen therapy: If PaO2 <=55 mmHg or SpO2 <=88% at rest""",
        "metadata": {"source": "GOLD COPD Guidelines", "category": "pulmonology", "condition": "COPD"}
    },
    {
        "content": """Asthma Management:
        - Step 1: SABA as needed (mild intermittent)
        - Step 2: Low-dose ICS (mild persistent)
        - Step 3: Low-dose ICS + LABA or medium-dose ICS
        - Step 4: Medium-dose ICS + LABA
        - Step 5: High-dose ICS + LABA, consider add-on therapy (tiotropium, biologics)
        - Control assessment: Daytime symptoms, nighttime awakening, SABA use, activity limitation
        - Exacerbation management: Systemic corticosteroids, frequent SABA, consider hospitalization if severe""",
        "metadata": {"source": "GINA Asthma Guidelines", "category": "pulmonology", "condition": "asthma"}
    },
    {
        "content": """Chronic Kidney Disease Management:
        - Staging: Based on GFR (G1-G5) and albuminuria (A1-A3)
        - Blood pressure target: <130/80 mmHg, ACEi/ARB preferred if proteinuria
        - Diabetes in CKD: SGLT2 inhibitors (if eGFR >20), GLP-1 RA as second-line
        - Anemia: Target Hgb 10-11.5 g/dL, iron supplementation first, then ESA if needed
        - Mineral bone disease: Phosphorus restriction, phosphate binders, vitamin D
        - Nephrology referral: eGFR <30, rapid decline (>5 mL/min/year), significant proteinuria""",
        "metadata": {"source": "KDIGO CKD Guidelines", "category": "nephrology", "condition": "CKD"}
    },
    {
        "content": """Common Critical Drug Interactions:
        - Warfarin + NSAIDs: Increased bleeding risk, avoid combination
        - ACEi + Potassium supplements: Hyperkalemia risk, monitor closely
        - Statins + Macrolides: Increased statin levels, myopathy risk
        - Metformin + Contrast dye: Hold metformin, risk of lactic acidosis
        - SSRIs + MAOIs: Serotonin syndrome, contraindicated
        - Digoxin + Amiodarone: Increased digoxin levels, reduce dose by 50%
        - Fluoroquinolones + QT-prolonging drugs: Risk of torsades de pointes""",
        "metadata": {"source": "Clinical Drug Interactions Database", "category": "pharmacology", "topic": "drug interactions"}
    },
    {
        "content": """Sepsis Recognition and Management (Sepsis-3):
        - Definition: Life-threatening organ dysfunction caused by dysregulated response to infection
        - qSOFA (quick screen): RR >=22, altered mentation, SBP <=100 mmHg (>=2 suggests sepsis)
        - SOFA score: For ICU patients, assesses 6 organ systems
        - Hour-1 Bundle: Measure lactate, blood cultures before antibiotics, broad-spectrum antibiotics, 30 mL/kg crystalloid if hypotensive/lactate >=4, vasopressors if MAP <65 after fluids
        - Target: MAP >=65 mmHg, lactate normalization, urine output >0.5 mL/kg/hr""",
        "metadata": {"source": "Surviving Sepsis Campaign", "category": "critical care", "condition": "sepsis"}
    },
    {
        "content": """Stroke Recognition and Management:
        - FAST screening: Face drooping, Arm weakness, Speech difficulty, Time to call 911
        - tPA window: Within 4.5 hours of symptom onset for eligible patients
        - Thrombectomy: Within 24 hours for large vessel occlusion (with imaging selection)
        - Blood pressure: Permissive hypertension if not receiving tPA (up to 220/120)
        - If receiving tPA: Maintain BP <180/105 mmHg
        - Acute workup: CT head (rule out hemorrhage), CTA/MRA, labs, ECG""",
        "metadata": {"source": "AHA/ASA Stroke Guidelines", "category": "neurology", "condition": "stroke"}
    },
    {
        "content": """Insulin Management for Inpatient Hyperglycemia:
        - Target glucose: 140-180 mg/dL for most hospitalized patients
        - Basal-bolus insulin preferred over sliding scale alone
        - Basal insulin: Glargine or detemir; calculate 0.2-0.4 units/kg/day
        - Bolus insulin: Rapid-acting (lispro, aspart) before meals
        - Correction factor: 1 unit per 50 mg/dL above target (adjust based on insulin sensitivity)
        - Hold oral hypoglycemics in acute illness; transition back before discharge
        - Hypoglycemia protocol: <70 mg/dL - give 15-20g fast-acting carbohydrate, recheck in 15 min""",
        "metadata": {"source": "ADA Inpatient Guidelines", "category": "endocrinology", "condition": "inpatient diabetes"}
    },
]


def initialize_vectorstore(include_pubmed: bool = False, pubmed_conditions: list = None):
    """Initialize the vector store with medical knowledge."""

    print("=" * 60)
    print("Initializing Clinical Decision Support Knowledge Base")
    print("=" * 60)

    # Create vector store
    index_path = settings.vector_store_path
    vector_store = VectorStore(index_path=index_path)
    vector_store.clear()

    # 1. Load core inline documents
    print(f"\nLoading {len(CORE_MEDICAL_KNOWLEDGE)} core medical documents...")
    ingest_inline_documents(CORE_MEDICAL_KNOWLEDGE, vector_store)

    # 2. Load curated guidelines from data/medical_guidelines/
    guidelines_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "medical_guidelines"
    )
    if os.path.exists(guidelines_dir):
        print(f"\nLoading curated guidelines from {guidelines_dir}...")
        count = ingest_directory(guidelines_dir, vector_store)
        print(f"  Loaded {count} chunks from curated guidelines")
    else:
        print(f"\nNo guidelines directory found at {guidelines_dir}")

    # 3. Optionally ingest PubMed abstracts
    if include_pubmed:
        try:
            from src.pubmed_ingestion import ingest_conditions
            conditions = pubmed_conditions or [
                "diabetes management", "hypertension treatment",
                "heart failure guidelines", "COPD treatment",
                "sepsis management",
            ]
            print(f"\nIngesting PubMed abstracts for: {conditions}")
            stats = ingest_conditions(conditions, vector_store, max_per_condition=20)
            print(f"  PubMed: {stats['total_articles']} articles, {stats['total_chunks']} chunks")
        except Exception as e:
            print(f"  PubMed ingestion skipped: {e}")

    # Save the index
    vector_store.save()

    # Print stats
    stats = get_ingestion_stats(vector_store)
    print("\n" + "=" * 60)
    print("Knowledge Base Initialized Successfully!")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Categories: {stats['categories']}")
    print(f"Index saved to: {index_path}")
    print("=" * 60)

    # Test search
    print("\nTesting search functionality...")
    test_queries = [
        "How to manage type 2 diabetes?",
        "What are the signs of sepsis?",
        "Metformin side effects and contraindications",
        "Atrial fibrillation anticoagulation",
        "Acute kidney injury management",
    ]

    for query in test_queries:
        results = vector_store.search(query, k=2)
        print(f"\nQuery: {query}")
        if results:
            print(f"  Top result (score: {results[0]['score']:.3f}): {results[0]['content'][:100]}...")

    print("\nVector store ready for use!")
    return vector_store


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initialize the clinical knowledge base")
    parser.add_argument("--include-pubmed", action="store_true", help="Also ingest PubMed abstracts")
    parser.add_argument("--pubmed-conditions", type=str, default=None,
                        help="Comma-separated PubMed conditions")
    args = parser.parse_args()

    pubmed_conditions = None
    if args.pubmed_conditions:
        pubmed_conditions = [c.strip() for c in args.pubmed_conditions.split(",")]

    initialize_vectorstore(
        include_pubmed=args.include_pubmed,
        pubmed_conditions=pubmed_conditions,
    )
