"""
Clinical Decision Support Assistant
Futuristic AI-Powered Medical Intelligence Platform
"""
import streamlit as st
import sys
import os
import requests
import tempfile
import re
import math
from typing import Dict, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

# Fix path for Streamlit Cloud
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Try to import medical_apis, fall back to inline implementation if not available
try:
    from src.medical_apis import (
        get_fda_client, get_pubmed_client, get_rxnorm_client,
        get_medical_aggregator, get_clinical_trials_client,
        get_disease_client, get_pharmacy_finder
    )
    APIS_IMPORTED = True
except ImportError:
    APIS_IMPORTED = False

# Optional imports
IMAGE_PROCESSOR_AVAILABLE = False
get_image_processor = None
RAG_AVAILABLE = False
get_rag_pipeline = None

try:
    from src.image_processor import get_image_processor
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError:
    pass

try:
    from src.rag_pipeline import get_rag_pipeline
    RAG_AVAILABLE = True
except ImportError:
    pass


# ========================================
# INLINE API CLIENTS (Fallback if import fails)
# ========================================
if not APIS_IMPORTED:
    class OpenFDAClient:
        """Client for openFDA API"""
        BASE_URL = "https://api.fda.gov"

        def __init__(self):
            self.session = requests.Session()

        def search_drug(self, drug_name: str, limit: int = 5) -> Dict:
            endpoint = f"{self.BASE_URL}/drug/label.json"
            params = {
                'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                'limit': limit
            }
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except:
                return {'error': 'API request failed', 'results': []}

        def get_adverse_events(self, drug_name: str, limit: int = 10) -> Dict:
            endpoint = f"{self.BASE_URL}/drug/event.json"
            params = {'search': f'patient.drug.medicinalproduct:"{drug_name}"', 'limit': limit}
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                events = []
                if 'results' in data:
                    for result in data['results']:
                        for reaction in result.get('patient', {}).get('reaction', []):
                            events.append(reaction.get('reactionmeddrapt', 'Unknown'))
                return {'drug_name': drug_name, 'adverse_events': list(set(events))[:20]}
            except:
                return {'error': 'API request failed'}

        def get_drug_info_summary(self, drug_name: str) -> Dict:
            drug_data = self.search_drug(drug_name, limit=1)
            adverse = self.get_adverse_events(drug_name, limit=5)
            summary = {
                'drug_name': drug_name, 'found': False, 'indications': [], 'dosage': [],
                'warnings': [], 'contraindications': [], 'interactions': [], 'common_adverse_events': []
            }
            if 'results' in drug_data and len(drug_data['results']) > 0:
                result = drug_data['results'][0]
                summary['found'] = True
                summary['indications'] = result.get('indications_and_usage', ['Not available'])
                summary['dosage'] = result.get('dosage_and_administration', ['Not available'])
                summary['warnings'] = result.get('warnings', ['Not available'])
                summary['contraindications'] = result.get('contraindications', ['Not available'])
                summary['interactions'] = result.get('drug_interactions', ['Not available'])
            if 'adverse_events' in adverse:
                summary['common_adverse_events'] = adverse['adverse_events']
            return summary

    class PubMedClient:
        """Client for PubMed API"""
        BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        def __init__(self):
            self.session = requests.Session()

        def search_articles(self, query: str, max_results: int = 5, recent_only: bool = False) -> Dict:
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            enhanced_query = query
            if recent_only:
                from datetime import datetime
                year = datetime.now().year
                enhanced_query = f"({query}) AND ({year - 5}[PDAT] : {year}[PDAT])"
            params = {'db': 'pubmed', 'term': enhanced_query, 'retmax': min(max_results, 50), 'retmode': 'json', 'sort': 'relevance'}
            try:
                response = self.session.get(search_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                ids = data.get('esearchresult', {}).get('idlist', [])
                if not ids:
                    return {'query': query, 'articles': [], 'count': 0}
                articles = self._fetch_article_details(ids)
                return {'query': query, 'articles': articles, 'count': len(articles)}
            except:
                return {'error': 'PubMed API request failed', 'articles': []}

        def _fetch_article_details(self, ids: List[str]) -> List[Dict]:
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            params = {'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'}
            try:
                response = self.session.get(fetch_url, params=params, timeout=15)
                response.raise_for_status()
                articles = []
                root = ET.fromstring(response.content)
                for article in root.findall('.//PubmedArticle'):
                    title_elem = article.find('.//ArticleTitle')
                    abstract_elem = article.find('.//AbstractText')
                    year_elem = article.find('.//PubDate/Year')
                    journal_elem = article.find('.//Journal/Title')
                    pmid_elem = article.find('.//PMID')
                    articles.append({
                        'title': title_elem.text if title_elem is not None else 'No title',
                        'abstract': (abstract_elem.text[:500] + '...' if abstract_elem is not None and abstract_elem.text and len(abstract_elem.text) > 500
                                    else (abstract_elem.text if abstract_elem is not None and abstract_elem.text else 'No abstract')),
                        'year': year_elem.text if year_elem is not None else 'Unknown',
                        'journal': journal_elem.text if journal_elem is not None else 'Unknown',
                        'pmid': pmid_elem.text if pmid_elem is not None else None,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else None
                    })
                return articles
            except:
                return []

    class RxNormClient:
        """Client for RxNorm API"""
        BASE_URL = "https://rxnav.nlm.nih.gov/REST"

        def __init__(self):
            self.session = requests.Session()

        def get_drug_info(self, drug_name: str) -> Dict:
            search_url = f"{self.BASE_URL}/drugs.json"
            params = {'name': drug_name}
            try:
                response = self.session.get(search_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                concepts = data.get('drugGroup', {}).get('conceptGroup', [])
                drugs = []
                for group in concepts:
                    if 'conceptProperties' in group:
                        for prop in group['conceptProperties']:
                            drugs.append({'rxcui': prop.get('rxcui'), 'name': prop.get('name')})
                return {'drug_name': drug_name, 'found': len(drugs) > 0, 'drugs': drugs[:5]}
            except:
                return {'error': 'RxNorm API request failed'}

        def get_interactions(self, drug_names: List[str]) -> Dict:
            rxcuis = []
            for drug in drug_names:
                info = self.get_drug_info(drug)
                if info.get('drugs'):
                    rxcuis.append(info['drugs'][0]['rxcui'])
            if len(rxcuis) < 2:
                return {'interactions': [], 'message': 'Need at least 2 valid drugs'}
            interaction_url = f"{self.BASE_URL}/interaction/list.json"
            params = {'rxcuis': '+'.join(rxcuis)}
            try:
                response = self.session.get(interaction_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                interactions = []
                for group in data.get('fullInteractionTypeGroup', []):
                    for itype in group.get('fullInteractionType', []):
                        for pair in itype.get('interactionPair', []):
                            interactions.append({
                                'severity': pair.get('severity', 'Unknown'),
                                'description': pair.get('description', 'No description'),
                                'drugs': [d.get('minConceptItem', {}).get('name', 'Unknown') for d in pair.get('interactionConcept', [])]
                            })
                return {'drug_names': drug_names, 'interactions_found': len(interactions) > 0, 'interactions': interactions}
            except:
                return {'error': 'Interaction check failed', 'interactions': []}

    class ClinicalTrialsClient:
        """Client for ClinicalTrials.gov API"""
        BASE_URL = "https://clinicaltrials.gov/api/v2"

        def __init__(self):
            self.session = requests.Session()

        def search_trials(self, condition: str, status: str = "RECRUITING", limit: int = 10) -> Dict:
            endpoint = f"{self.BASE_URL}/studies"
            params = {'query.cond': condition, 'filter.overallStatus': status, 'pageSize': limit, 'format': 'json'}
            try:
                response = self.session.get(endpoint, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                trials = []
                for study in data.get('studies', [])[:limit]:
                    protocol = study.get('protocolSection', {})
                    identification = protocol.get('identificationModule', {})
                    status_module = protocol.get('statusModule', {})
                    desc = protocol.get('descriptionModule', {})
                    trials.append({
                        'nct_id': identification.get('nctId', ''),
                        'title': identification.get('briefTitle', 'No title'),
                        'status': status_module.get('overallStatus', 'Unknown'),
                        'phase': status_module.get('phases', []),
                        'summary': desc.get('briefSummary', ''),
                        'url': f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}"
                    })
                return {'condition': condition, 'trials': trials, 'count': len(trials)}
            except:
                return {'error': 'ClinicalTrials API request failed', 'trials': []}

    class DiseaseInfoClient:
        """Client for disease information"""
        def __init__(self):
            self.session = requests.Session()

        def get_disease_info(self, disease_name: str) -> Dict:
            return {'disease_name': disease_name, 'research_articles': []}

        def suggest_medications(self, disease_name: str) -> Dict:
            return {'disease': disease_name, 'suggested_medications': []}

    class PharmacyFinderClient:
        """Client for finding pharmacies using Overpass API with bounding box"""
        def __init__(self):
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': 'MedAI-ClinicalSupport/1.0 (Clinical Decision Support App)'})

        def find_nearby_pharmacies(self, lat: float, lon: float, drug_name: str = None, radius: int = 8000) -> Dict:
            """Find pharmacies within radius (meters) using Overpass API with bounding box"""
            try:
                # Convert radius to degrees (approximate)
                # 1 degree latitude = ~111km, 1 degree longitude varies by latitude
                radius_km = radius / 1000
                lat_offset = radius_km / 111.0
                lon_offset = radius_km / (111.0 * abs(math.cos(math.radians(lat))) + 0.001)

                # Create bounding box: south, west, north, east
                south = lat - lat_offset
                north = lat + lat_offset
                west = lon - lon_offset
                east = lon + lon_offset

                # Use Overpass API with bounding box for precise results
                overpass_url = "https://overpass-api.de/api/interpreter"

                # Query using bounding box - more reliable than 'around'
                overpass_query = f"""
[out:json][timeout:30];
(
  node["amenity"="pharmacy"]({south},{west},{north},{east});
  way["amenity"="pharmacy"]({south},{west},{north},{east});
  node["healthcare"="pharmacy"]({south},{west},{north},{east});
  node["shop"="chemist"]({south},{west},{north},{east});
);
out body center;
"""

                response = self.session.post(overpass_url, data={'data': overpass_query}, timeout=35)
                response.raise_for_status()
                data = response.json()

                pharmacies = []
                max_distance_km = radius_km * 1.2  # Allow 20% buffer

                for element in data.get('elements', []):
                    # Get coordinates
                    if element.get('type') == 'way':
                        elem_lat = element.get('center', {}).get('lat')
                        elem_lon = element.get('center', {}).get('lon')
                    else:
                        elem_lat = element.get('lat')
                        elem_lon = element.get('lon')

                    # Skip if no valid coordinates
                    if elem_lat is None or elem_lon is None:
                        continue

                    # Calculate distance and strictly filter
                    distance = self._calculate_distance(lat, lon, elem_lat, elem_lon)

                    # STRICT FILTER: Skip any results beyond the search radius
                    if distance > max_distance_km:
                        continue

                    tags = element.get('tags', {})

                    # Build name - prioritize specific pharmacy names
                    name = tags.get('name') or tags.get('brand') or tags.get('operator') or 'Pharmacy'

                    # Build address from tags
                    address_parts = []
                    if tags.get('addr:housenumber'):
                        address_parts.append(tags.get('addr:housenumber'))
                    if tags.get('addr:street'):
                        address_parts.append(tags.get('addr:street'))
                    if tags.get('addr:city'):
                        address_parts.append(tags.get('addr:city'))
                    if tags.get('addr:state'):
                        address_parts.append(tags.get('addr:state'))
                    if tags.get('addr:postcode'):
                        address_parts.append(tags.get('addr:postcode'))

                    address = ', '.join(address_parts) if address_parts else None

                    # If no address from tags, try reverse geocoding for first few results
                    if not address and len(pharmacies) < 10:
                        address = self._reverse_geocode(elem_lat, elem_lon)

                    if not address:
                        address = f"Near {lat:.4f}, {lon:.4f}"

                    # Get additional info
                    phone = tags.get('phone') or tags.get('contact:phone') or ''
                    website = tags.get('website') or tags.get('contact:website') or ''
                    opening_hours = tags.get('opening_hours') or ''

                    pharmacies.append({
                        'name': name,
                        'address': address,
                        'latitude': elem_lat,
                        'longitude': elem_lon,
                        'distance_km': round(distance, 2),
                        'distance_miles': round(distance * 0.621371, 2),
                        'phone': phone,
                        'website': website,
                        'opening_hours': opening_hours
                    })

                # Sort by distance
                pharmacies.sort(key=lambda x: x['distance_km'])

                return {
                    'pharmacies': pharmacies[:15],
                    'count': len(pharmacies),
                    'location': {'lat': lat, 'lon': lon},
                    'radius_km': radius_km,
                    'drug_searched': drug_name
                }
            except Exception as e:
                return {'error': f'Pharmacy search failed: {str(e)}', 'pharmacies': []}

        def _reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
            """Get address from coordinates"""
            try:
                endpoint = "https://nominatim.openstreetmap.org/reverse"
                params = {'lat': lat, 'lon': lon, 'format': 'json', 'addressdetails': 1}
                response = self.session.get(endpoint, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    addr = data.get('address', {})
                    parts = []
                    if addr.get('house_number'):
                        parts.append(addr['house_number'])
                    if addr.get('road'):
                        parts.append(addr['road'])
                    if addr.get('city') or addr.get('town') or addr.get('village'):
                        parts.append(addr.get('city') or addr.get('town') or addr.get('village'))
                    if addr.get('state'):
                        parts.append(addr['state'])
                    if addr.get('postcode'):
                        parts.append(addr['postcode'])
                    return ', '.join(parts) if parts else None
            except:
                pass
            return None

        def geocode_address(self, address: str) -> Dict:
            """Convert address/ZIP code to coordinates with improved accuracy"""
            try:
                # Clean up the address
                address = address.strip()

                # If it looks like a ZIP code, add USA context
                if address.isdigit() and len(address) == 5:
                    address = f"{address}, USA"

                endpoint = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': address,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1,
                    'countrycodes': 'us'  # Prioritize US results
                }

                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data:
                    location = data[0]
                    addr_details = location.get('address', {})

                    # Build a clean display name
                    display_parts = []
                    if addr_details.get('city') or addr_details.get('town') or addr_details.get('village'):
                        display_parts.append(addr_details.get('city') or addr_details.get('town') or addr_details.get('village'))
                    if addr_details.get('state'):
                        display_parts.append(addr_details.get('state'))
                    if addr_details.get('postcode'):
                        display_parts.append(addr_details.get('postcode'))

                    display_name = ', '.join(display_parts) if display_parts else location.get('display_name', address)

                    return {
                        'success': True,
                        'latitude': float(location.get('lat', 0)),
                        'longitude': float(location.get('lon', 0)),
                        'display_name': display_name,
                        'city': addr_details.get('city') or addr_details.get('town') or addr_details.get('village', ''),
                        'state': addr_details.get('state', ''),
                        'postcode': addr_details.get('postcode', '')
                    }
                return {'success': False, 'error': 'Location not found. Please try a different address or ZIP code.'}
            except Exception as e:
                return {'success': False, 'error': f'Geocoding failed: {str(e)}'}

        def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate distance between two coordinates in kilometers (Haversine formula)"""
            R = 6371  # Earth's radius in km
            lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
            delta_lat, delta_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Singleton getters
    _fda_client = None
    _pubmed_client = None
    _rxnorm_client = None
    _trials_client = None
    _disease_client = None
    _pharmacy_finder = None

    def get_fda_client():
        global _fda_client
        if _fda_client is None:
            _fda_client = OpenFDAClient()
        return _fda_client

    def get_pubmed_client():
        global _pubmed_client
        if _pubmed_client is None:
            _pubmed_client = PubMedClient()
        return _pubmed_client

    def get_rxnorm_client():
        global _rxnorm_client
        if _rxnorm_client is None:
            _rxnorm_client = RxNormClient()
        return _rxnorm_client

    def get_clinical_trials_client():
        global _trials_client
        if _trials_client is None:
            _trials_client = ClinicalTrialsClient()
        return _trials_client

    def get_disease_client():
        global _disease_client
        if _disease_client is None:
            _disease_client = DiseaseInfoClient()
        return _disease_client

    def get_pharmacy_finder():
        global _pharmacy_finder
        if _pharmacy_finder is None:
            _pharmacy_finder = PharmacyFinderClient()
        return _pharmacy_finder

    def get_medical_aggregator():
        return None

st.set_page_config(
    page_title="MedAI - Clinical Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# COMPREHENSIVE DISEASE DATABASE
# ========================================
DISEASE_DATABASE = {
    # Cardiovascular
    "hypertension": {
        "name": "Hypertension (High Blood Pressure)",
        "category": "Cardiovascular",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "A chronic condition where blood pressure against artery walls is consistently too high, potentially leading to heart disease, stroke, and other complications.",
        "symptoms": ["Headaches", "Shortness of breath", "Nosebleeds", "Dizziness", "Chest pain", "Vision problems", "Fatigue"],
        "causes": ["Genetics", "Obesity", "High sodium diet", "Lack of exercise", "Stress", "Alcohol consumption", "Smoking", "Age"],
        "risk_factors": ["Family history", "Age over 65", "African ancestry", "Obesity", "Sedentary lifestyle", "High salt diet"],
        "treatments": ["Lifestyle modifications", "ACE inhibitors", "Calcium channel blockers", "Diuretics", "Beta-blockers", "ARBs"],
        "medications": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol", "Valsartan"],
        "prevention": ["Maintain healthy weight", "Exercise regularly", "Reduce sodium intake", "Limit alcohol", "Manage stress", "Don't smoke"],
        "complications": ["Heart attack", "Stroke", "Heart failure", "Kidney disease", "Vision loss", "Dementia"],
        "when_to_seek_help": "If blood pressure exceeds 180/120 mmHg, or if experiencing severe headache, chest pain, or vision changes"
    },
    "heart failure": {
        "name": "Heart Failure (Congestive Heart Failure)",
        "category": "Cardiovascular",
        "severity": "Severe",
        "criticality": 9,
        "description": "A chronic condition where the heart cannot pump blood efficiently enough to meet the body's needs.",
        "symptoms": ["Shortness of breath", "Fatigue", "Swollen legs/ankles", "Rapid heartbeat", "Persistent cough", "Wheezing", "Reduced exercise ability", "Sudden weight gain"],
        "causes": ["Coronary artery disease", "High blood pressure", "Previous heart attack", "Cardiomyopathy", "Heart valve disease", "Diabetes"],
        "risk_factors": ["Age over 65", "Previous heart conditions", "Diabetes", "Obesity", "Sleep apnea"],
        "treatments": ["Medications", "Lifestyle changes", "Device implants (ICD, pacemaker)", "Heart surgery", "Heart transplant"],
        "medications": ["Furosemide", "Carvedilol", "Enalapril", "Spironolactone", "Entresto", "Digoxin"],
        "prevention": ["Control blood pressure", "Manage diabetes", "Maintain healthy weight", "Exercise", "Avoid smoking/alcohol"],
        "complications": ["Kidney damage", "Liver damage", "Arrhythmias", "Pulmonary hypertension", "Death"],
        "when_to_seek_help": "Immediately if experiencing sudden severe shortness of breath, chest pain, or fainting"
    },
    "coronary artery disease": {
        "name": "Coronary Artery Disease (CAD)",
        "category": "Cardiovascular",
        "severity": "Severe",
        "criticality": 9,
        "description": "The most common type of heart disease caused by plaque buildup in the coronary arteries, reducing blood flow to the heart.",
        "symptoms": ["Chest pain (angina)", "Shortness of breath", "Fatigue", "Heart attack symptoms", "Pain radiating to arm/jaw"],
        "causes": ["Atherosclerosis", "High cholesterol", "High blood pressure", "Smoking", "Diabetes", "Inflammation"],
        "risk_factors": ["Age", "Male gender", "Family history", "Smoking", "High cholesterol", "Diabetes", "Obesity"],
        "treatments": ["Lifestyle changes", "Medications", "Angioplasty with stent", "Coronary bypass surgery"],
        "medications": ["Aspirin", "Statins (Atorvastatin)", "Beta-blockers", "Nitroglycerin", "Clopidogrel"],
        "prevention": ["Don't smoke", "Control cholesterol", "Manage blood pressure", "Exercise", "Healthy diet", "Maintain weight"],
        "complications": ["Heart attack", "Heart failure", "Arrhythmias", "Cardiac arrest"],
        "when_to_seek_help": "Call 911 immediately for chest pain, especially with sweating, nausea, or arm/jaw pain"
    },
    "atrial fibrillation": {
        "name": "Atrial Fibrillation (AFib)",
        "category": "Cardiovascular",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "An irregular and often rapid heart rhythm that can lead to blood clots, stroke, and heart failure.",
        "symptoms": ["Irregular heartbeat", "Heart palpitations", "Fatigue", "Shortness of breath", "Dizziness", "Chest discomfort"],
        "causes": ["High blood pressure", "Heart disease", "Thyroid disorders", "Sleep apnea", "Excessive alcohol", "Caffeine"],
        "risk_factors": ["Age over 60", "Heart conditions", "High blood pressure", "Obesity", "Family history"],
        "treatments": ["Rate control medications", "Rhythm control", "Blood thinners", "Cardioversion", "Ablation"],
        "medications": ["Warfarin", "Eliquis (Apixaban)", "Metoprolol", "Diltiazem", "Amiodarone", "Digoxin"],
        "prevention": ["Control blood pressure", "Limit caffeine/alcohol", "Maintain healthy weight", "Exercise"],
        "complications": ["Stroke", "Heart failure", "Blood clots", "Cognitive decline"],
        "when_to_seek_help": "If experiencing rapid heartbeat with chest pain, severe dizziness, or signs of stroke"
    },

    # Metabolic/Endocrine
    "diabetes type 2": {
        "name": "Type 2 Diabetes Mellitus",
        "category": "Metabolic/Endocrine",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "A chronic condition affecting how the body processes blood sugar (glucose), characterized by insulin resistance.",
        "symptoms": ["Increased thirst", "Frequent urination", "Increased hunger", "Fatigue", "Blurred vision", "Slow-healing wounds", "Numbness in hands/feet"],
        "causes": ["Insulin resistance", "Genetics", "Obesity", "Physical inactivity", "Poor diet"],
        "risk_factors": ["Obesity", "Age over 45", "Family history", "Sedentary lifestyle", "Prediabetes", "Gestational diabetes history"],
        "treatments": ["Lifestyle modifications", "Oral medications", "Injectable medications", "Insulin therapy", "Bariatric surgery"],
        "medications": ["Metformin", "Glipizide", "Januvia (Sitagliptin)", "Jardiance (Empagliflozin)", "Ozempic (Semaglutide)", "Trulicity", "Insulin"],
        "prevention": ["Maintain healthy weight", "Exercise regularly", "Eat balanced diet", "Monitor blood sugar if at risk"],
        "complications": ["Heart disease", "Stroke", "Kidney disease", "Neuropathy", "Retinopathy", "Foot problems", "Skin conditions"],
        "when_to_seek_help": "If experiencing extreme thirst, confusion, very high blood sugar, or diabetic ketoacidosis symptoms"
    },
    "diabetes type 1": {
        "name": "Type 1 Diabetes Mellitus",
        "category": "Metabolic/Endocrine",
        "severity": "Severe",
        "criticality": 8,
        "description": "An autoimmune condition where the pancreas produces little or no insulin.",
        "symptoms": ["Extreme thirst", "Frequent urination", "Unintended weight loss", "Fatigue", "Blurred vision", "Mood changes"],
        "causes": ["Autoimmune destruction of insulin-producing cells", "Genetics", "Environmental triggers"],
        "risk_factors": ["Family history", "Genetics", "Age (peaks in children 4-7 and 10-14)", "Geography"],
        "treatments": ["Insulin therapy (required)", "Blood sugar monitoring", "Carbohydrate counting", "Healthy eating"],
        "medications": ["Rapid-acting insulin (Humalog, Novolog)", "Long-acting insulin (Lantus, Levemir)", "Insulin pump therapy"],
        "prevention": "Cannot be prevented as it's an autoimmune condition",
        "complications": ["Hypoglycemia", "Diabetic ketoacidosis", "Heart disease", "Neuropathy", "Nephropathy", "Retinopathy"],
        "when_to_seek_help": "Immediately for signs of diabetic ketoacidosis: nausea, vomiting, abdominal pain, fruity breath"
    },
    "hypothyroidism": {
        "name": "Hypothyroidism (Underactive Thyroid)",
        "category": "Metabolic/Endocrine",
        "severity": "Mild to Moderate",
        "criticality": 5,
        "description": "A condition where the thyroid gland doesn't produce enough thyroid hormones.",
        "symptoms": ["Fatigue", "Weight gain", "Cold sensitivity", "Dry skin", "Depression", "Constipation", "Muscle weakness", "Slow heart rate"],
        "causes": ["Hashimoto's thyroiditis", "Thyroid surgery", "Radiation therapy", "Medications", "Iodine deficiency"],
        "risk_factors": ["Female gender", "Age over 60", "Autoimmune disease", "Family history", "Previous thyroid surgery"],
        "treatments": ["Thyroid hormone replacement therapy"],
        "medications": ["Levothyroxine (Synthroid)", "Liothyronine (Cytomel)"],
        "prevention": "Regular thyroid screening for high-risk individuals",
        "complications": ["Goiter", "Heart problems", "Mental health issues", "Myxedema coma", "Infertility"],
        "when_to_seek_help": "If experiencing severe symptoms like extreme fatigue, confusion, or very slow heart rate"
    },

    # Respiratory
    "asthma": {
        "name": "Asthma",
        "category": "Respiratory",
        "severity": "Mild to Severe",
        "criticality": 6,
        "description": "A chronic inflammatory disease of the airways causing wheezing, breathlessness, chest tightness, and coughing.",
        "symptoms": ["Wheezing", "Shortness of breath", "Chest tightness", "Coughing", "Difficulty sleeping due to breathing", "Rapid breathing"],
        "causes": ["Genetic factors", "Environmental allergens", "Respiratory infections", "Air pollution", "Exercise", "Cold air"],
        "risk_factors": ["Family history", "Allergies", "Obesity", "Smoking exposure", "Occupational exposures"],
        "treatments": ["Quick-relief inhalers", "Long-term control medications", "Allergy medications", "Bronchial thermoplasty"],
        "medications": ["Albuterol", "Fluticasone (Flovent)", "Salmeterol", "Montelukast (Singulair)", "Budesonide", "Prednisone"],
        "prevention": ["Identify and avoid triggers", "Get vaccinated", "Monitor breathing", "Use air purifier"],
        "complications": ["Severe asthma attacks", "Permanent airway narrowing", "Medication side effects", "Sleep problems"],
        "when_to_seek_help": "Immediately for severe breathing difficulty, blue lips/fingernails, or inhaler not providing relief"
    },
    "copd": {
        "name": "Chronic Obstructive Pulmonary Disease (COPD)",
        "category": "Respiratory",
        "severity": "Moderate to Severe",
        "criticality": 8,
        "description": "A chronic inflammatory lung disease causing obstructed airflow from the lungs, including emphysema and chronic bronchitis.",
        "symptoms": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Excess mucus", "Fatigue", "Frequent respiratory infections"],
        "causes": ["Smoking (primary cause)", "Long-term exposure to air pollutants", "Genetic factors (alpha-1 antitrypsin deficiency)"],
        "risk_factors": ["Smoking", "Age over 40", "Occupational dust exposure", "Genetics", "Asthma"],
        "treatments": ["Bronchodilators", "Inhaled steroids", "Pulmonary rehabilitation", "Oxygen therapy", "Surgery"],
        "medications": ["Tiotropium (Spiriva)", "Salmeterol", "Fluticasone", "Prednisone", "Roflumilast"],
        "prevention": ["Don't smoke or quit smoking", "Avoid lung irritants", "Get vaccinated", "Regular check-ups"],
        "complications": ["Respiratory infections", "Heart problems", "Lung cancer", "Pulmonary hypertension", "Depression"],
        "when_to_seek_help": "For sudden worsening of symptoms, inability to catch breath, or confusion"
    },
    "pneumonia": {
        "name": "Pneumonia",
        "category": "Respiratory",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "An infection that inflames air sacs in one or both lungs, which may fill with fluid.",
        "symptoms": ["Cough with phlegm", "Fever", "Chills", "Difficulty breathing", "Chest pain", "Fatigue", "Nausea/vomiting"],
        "causes": ["Bacteria", "Viruses", "Fungi", "Aspiration"],
        "risk_factors": ["Age under 2 or over 65", "Weakened immune system", "Chronic diseases", "Smoking", "Hospitalization"],
        "treatments": ["Antibiotics (bacterial)", "Antivirals (viral)", "Antifungals (fungal)", "Supportive care", "Hospitalization if severe"],
        "medications": ["Amoxicillin", "Azithromycin", "Levofloxacin", "Tamiflu (viral)", "Fluconazole (fungal)"],
        "prevention": ["Get vaccinated", "Practice good hygiene", "Don't smoke", "Keep immune system strong"],
        "complications": ["Bacteremia", "Breathing difficulty", "Lung abscess", "Pleural effusion", "Death"],
        "when_to_seek_help": "For high fever, severe breathing difficulty, confusion, or bluish skin color"
    },

    # Neurological
    "migraine": {
        "name": "Migraine",
        "category": "Neurological",
        "severity": "Moderate",
        "criticality": 5,
        "description": "A neurological condition causing intense, debilitating headaches, often with nausea, vomiting, and sensitivity to light/sound.",
        "symptoms": ["Severe throbbing headache", "Nausea/vomiting", "Light sensitivity", "Sound sensitivity", "Aura", "Visual disturbances", "Dizziness"],
        "causes": ["Genetic factors", "Brain chemical imbalances", "Triggers (stress, foods, hormones)", "Nerve pathway changes"],
        "risk_factors": ["Family history", "Female gender", "Hormonal changes", "Stress", "Sleep changes"],
        "treatments": ["Acute medications", "Preventive medications", "Lifestyle changes", "Botox injections", "CGRP inhibitors"],
        "medications": ["Sumatriptan", "Rizatriptan", "Topiramate", "Propranolol", "Amitriptyline", "Aimovig", "Ubrelvy"],
        "prevention": ["Identify and avoid triggers", "Regular sleep schedule", "Stress management", "Regular exercise", "Stay hydrated"],
        "complications": ["Chronic migraine", "Status migrainosus", "Migrainous infarction", "Medication overuse headache"],
        "when_to_seek_help": "For sudden severe headache, headache with fever/stiff neck, or worst headache of your life"
    },
    "alzheimer's disease": {
        "name": "Alzheimer's Disease",
        "category": "Neurological",
        "severity": "Severe",
        "criticality": 9,
        "description": "A progressive neurological disorder causing brain cells to degenerate and die, leading to dementia.",
        "symptoms": ["Memory loss", "Confusion", "Difficulty with familiar tasks", "Language problems", "Disorientation", "Mood changes", "Personality changes"],
        "causes": ["Brain protein abnormalities (plaques and tangles)", "Genetic factors", "Age-related brain changes"],
        "risk_factors": ["Age over 65", "Family history", "Down syndrome", "Head trauma", "Heart health factors"],
        "treatments": ["Cholinesterase inhibitors", "Memantine", "Behavioral interventions", "Supportive care"],
        "medications": ["Donepezil (Aricept)", "Rivastigmine", "Galantamine", "Memantine (Namenda)", "Aducanumab (Aduhelm)"],
        "prevention": ["Mental stimulation", "Physical exercise", "Social engagement", "Heart-healthy diet", "Quality sleep"],
        "complications": ["Complete dependence", "Infections", "Falls", "Malnutrition", "Death"],
        "when_to_seek_help": "When memory problems interfere with daily life or for sudden changes in behavior/personality"
    },
    "parkinson's disease": {
        "name": "Parkinson's Disease",
        "category": "Neurological",
        "severity": "Severe",
        "criticality": 8,
        "description": "A progressive nervous system disorder affecting movement, causing tremors, stiffness, and slowing of movement.",
        "symptoms": ["Tremor", "Slowed movement (bradykinesia)", "Rigid muscles", "Impaired posture/balance", "Speech changes", "Writing changes"],
        "causes": ["Loss of dopamine-producing neurons", "Genetic mutations", "Environmental triggers", "Lewy bodies"],
        "risk_factors": ["Age over 60", "Male gender", "Family history", "Toxin exposure", "Head trauma"],
        "treatments": ["Medications", "Deep brain stimulation", "Physical therapy", "Occupational therapy", "Speech therapy"],
        "medications": ["Levodopa/Carbidopa (Sinemet)", "Dopamine agonists (Pramipexole)", "MAO-B inhibitors", "Amantadine"],
        "prevention": "No proven prevention, but exercise and caffeine may reduce risk",
        "complications": ["Cognitive problems", "Depression", "Sleep disorders", "Swallowing/eating problems", "Falls"],
        "when_to_seek_help": "For significant changes in symptoms, falls, or mood/cognitive changes"
    },
    "epilepsy": {
        "name": "Epilepsy",
        "category": "Neurological",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "A neurological disorder characterized by recurrent seizures due to abnormal electrical brain activity.",
        "symptoms": ["Seizures", "Temporary confusion", "Staring spells", "Uncontrollable jerking movements", "Loss of consciousness", "Anxiety", "Deja vu"],
        "causes": ["Genetic factors", "Brain injury", "Brain tumors", "Stroke", "Infections", "Developmental disorders"],
        "risk_factors": ["Family history", "Head injuries", "Stroke", "Dementia", "Brain infections", "Childhood seizures"],
        "treatments": ["Anti-seizure medications", "Surgery", "Vagus nerve stimulation", "Ketogenic diet", "Deep brain stimulation"],
        "medications": ["Levetiracetam (Keppra)", "Lamotrigine", "Valproic acid", "Carbamazepine", "Phenytoin", "Topiramate"],
        "prevention": ["Prevent head injuries", "Get adequate sleep", "Avoid alcohol/drugs", "Take medications as prescribed"],
        "complications": ["Status epilepticus", "Sudden unexpected death", "Falls/injuries", "Drowning", "Emotional issues"],
        "when_to_seek_help": "For seizure lasting more than 5 minutes, repeated seizures, or seizure with pregnancy/diabetes"
    },

    # Mental Health
    "depression": {
        "name": "Major Depressive Disorder",
        "category": "Mental Health",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "A mood disorder causing persistent feelings of sadness, hopelessness, and loss of interest in activities.",
        "symptoms": ["Persistent sadness", "Loss of interest", "Sleep changes", "Appetite changes", "Fatigue", "Guilt/worthlessness", "Difficulty concentrating", "Thoughts of death"],
        "causes": ["Brain chemistry imbalances", "Genetics", "Hormonal changes", "Trauma", "Chronic illness", "Substance abuse"],
        "risk_factors": ["Family history", "Trauma", "Major life changes", "Chronic illness", "Certain medications", "Substance abuse"],
        "treatments": ["Psychotherapy", "Medications", "Brain stimulation therapies", "Lifestyle changes", "Support groups"],
        "medications": ["Sertraline (Zoloft)", "Fluoxetine (Prozac)", "Escitalopram (Lexapro)", "Bupropion (Wellbutrin)", "Venlafaxine (Effexor)"],
        "prevention": ["Stress management", "Reach out to support", "Early treatment", "Long-term maintenance treatment"],
        "complications": ["Self-harm", "Suicide", "Substance abuse", "Relationship problems", "Physical health problems"],
        "when_to_seek_help": "Immediately for thoughts of suicide or self-harm. Call 988 (Suicide & Crisis Lifeline)"
    },
    "anxiety disorder": {
        "name": "Generalized Anxiety Disorder",
        "category": "Mental Health",
        "severity": "Mild to Moderate",
        "criticality": 5,
        "description": "A mental health disorder characterized by persistent and excessive worry about various aspects of life.",
        "symptoms": ["Excessive worry", "Restlessness", "Fatigue", "Difficulty concentrating", "Irritability", "Muscle tension", "Sleep problems"],
        "causes": ["Brain chemistry", "Genetics", "Personality", "Life experiences", "Trauma"],
        "risk_factors": ["Family history", "Trauma", "Chronic illness", "Substance abuse", "Female gender"],
        "treatments": ["Psychotherapy (CBT)", "Medications", "Relaxation techniques", "Lifestyle changes"],
        "medications": ["Buspirone", "Escitalopram", "Sertraline", "Venlafaxine", "Benzodiazepines (short-term)"],
        "prevention": ["Stress management", "Regular exercise", "Adequate sleep", "Limit caffeine/alcohol", "Seek help early"],
        "complications": ["Depression", "Substance abuse", "Digestive problems", "Chronic pain", "Social isolation"],
        "when_to_seek_help": "When anxiety interferes with daily life or for thoughts of self-harm"
    },
    "bipolar disorder": {
        "name": "Bipolar Disorder",
        "category": "Mental Health",
        "severity": "Severe",
        "criticality": 8,
        "description": "A mental health condition causing extreme mood swings including emotional highs (mania) and lows (depression).",
        "symptoms": ["Manic episodes", "Depressive episodes", "Elevated mood", "Decreased need for sleep", "Racing thoughts", "Impulsive behavior", "Suicidal thoughts"],
        "causes": ["Genetic factors", "Brain structure differences", "Neurotransmitter imbalances"],
        "risk_factors": ["Family history", "High stress", "Drug/alcohol abuse", "Major life changes"],
        "treatments": ["Mood stabilizers", "Antipsychotics", "Antidepressants", "Psychotherapy", "Electroconvulsive therapy"],
        "medications": ["Lithium", "Valproate", "Lamotrigine", "Quetiapine", "Olanzapine", "Aripiprazole"],
        "prevention": "Cannot prevent, but early treatment and medication adherence help manage symptoms",
        "complications": ["Suicide", "Substance abuse", "Legal/financial problems", "Relationship difficulties", "Work problems"],
        "when_to_seek_help": "Immediately for suicidal thoughts. Call 988 (Suicide & Crisis Lifeline)"
    },

    # Gastrointestinal
    "gerd": {
        "name": "Gastroesophageal Reflux Disease (GERD)",
        "category": "Gastrointestinal",
        "severity": "Mild to Moderate",
        "criticality": 4,
        "description": "A chronic digestive disease where stomach acid frequently flows back into the esophagus, causing irritation.",
        "symptoms": ["Heartburn", "Regurgitation", "Difficulty swallowing", "Chest pain", "Chronic cough", "Hoarseness", "Feeling of lump in throat"],
        "causes": ["Weak lower esophageal sphincter", "Hiatal hernia", "Obesity", "Pregnancy", "Delayed stomach emptying"],
        "risk_factors": ["Obesity", "Hiatal hernia", "Pregnancy", "Smoking", "Eating large meals", "Lying down after eating"],
        "treatments": ["Lifestyle changes", "Antacids", "H2 blockers", "Proton pump inhibitors", "Surgery"],
        "medications": ["Omeprazole (Prilosec)", "Esomeprazole (Nexium)", "Famotidine (Pepcid)", "Ranitidine", "Sucralfate"],
        "prevention": ["Maintain healthy weight", "Avoid trigger foods", "Don't lie down after meals", "Elevate head of bed", "Don't smoke"],
        "complications": ["Esophagitis", "Esophageal stricture", "Barrett's esophagus", "Esophageal cancer"],
        "when_to_seek_help": "For severe chest pain, difficulty swallowing, or vomiting blood"
    },
    "irritable bowel syndrome": {
        "name": "Irritable Bowel Syndrome (IBS)",
        "category": "Gastrointestinal",
        "severity": "Mild to Moderate",
        "criticality": 4,
        "description": "A common disorder affecting the large intestine, causing cramping, abdominal pain, bloating, gas, and changes in bowel habits.",
        "symptoms": ["Abdominal pain/cramping", "Bloating", "Gas", "Diarrhea", "Constipation", "Mucus in stool"],
        "causes": ["Muscle contractions in intestine", "Nervous system abnormalities", "Gut microbiome changes", "Infection", "Stress"],
        "risk_factors": ["Age under 50", "Female gender", "Family history", "Mental health issues", "Food intolerances"],
        "treatments": ["Dietary changes", "Stress management", "Medications", "Probiotics", "Mental health therapies"],
        "medications": ["Loperamide", "Linaclotide", "Lubiprostone", "Rifaximin", "Antidepressants (low dose)"],
        "prevention": ["Manage stress", "Identify trigger foods", "Regular exercise", "Adequate sleep", "Eat regular meals"],
        "complications": ["Quality of life impact", "Mood disorders", "Food avoidance"],
        "when_to_seek_help": "For persistent changes in bowel habits, weight loss, rectal bleeding, or severe pain"
    },
    "crohn's disease": {
        "name": "Crohn's Disease",
        "category": "Gastrointestinal",
        "severity": "Moderate to Severe",
        "criticality": 7,
        "description": "A type of inflammatory bowel disease causing inflammation of the digestive tract, leading to abdominal pain, severe diarrhea, fatigue, and malnutrition.",
        "symptoms": ["Diarrhea", "Abdominal pain/cramping", "Blood in stool", "Fatigue", "Weight loss", "Fever", "Mouth sores"],
        "causes": ["Immune system malfunction", "Genetics", "Environmental factors"],
        "risk_factors": ["Age under 30", "Family history", "Smoking", "NSAIDs", "Certain ethnicities"],
        "treatments": ["Anti-inflammatory drugs", "Immune suppressors", "Biologics", "Antibiotics", "Surgery"],
        "medications": ["Mesalamine", "Prednisone", "Azathioprine", "Infliximab (Remicade)", "Adalimumab (Humira)", "Ustekinumab"],
        "prevention": "Cannot prevent, but treatment can reduce flares and maintain remission",
        "complications": ["Bowel obstruction", "Ulcers", "Fistulas", "Anal fissures", "Malnutrition", "Colon cancer"],
        "when_to_seek_help": "For severe abdominal pain, blood in stool, ongoing diarrhea, or unexplained fever"
    },

    # Musculoskeletal
    "rheumatoid arthritis": {
        "name": "Rheumatoid Arthritis",
        "category": "Musculoskeletal",
        "severity": "Moderate to Severe",
        "criticality": 6,
        "description": "An autoimmune disorder that primarily affects joints, causing painful swelling that can lead to bone erosion and joint deformity.",
        "symptoms": ["Joint pain", "Joint swelling", "Joint stiffness", "Fatigue", "Fever", "Loss of appetite", "Symmetric joint involvement"],
        "causes": ["Autoimmune response", "Genetic factors", "Environmental triggers", "Hormonal factors"],
        "risk_factors": ["Female gender", "Age 40-60", "Family history", "Smoking", "Obesity", "Environmental exposures"],
        "treatments": ["DMARDs", "Biologics", "Steroids", "Physical therapy", "Surgery"],
        "medications": ["Methotrexate", "Hydroxychloroquine", "Sulfasalazine", "Adalimumab", "Etanercept", "Prednisone"],
        "prevention": "Cannot prevent, but early treatment can slow progression",
        "complications": ["Osteoporosis", "Rheumatoid nodules", "Carpal tunnel syndrome", "Heart problems", "Lung disease"],
        "when_to_seek_help": "For new joint swelling, increased pain/stiffness, or signs of infection"
    },
    "osteoarthritis": {
        "name": "Osteoarthritis",
        "category": "Musculoskeletal",
        "severity": "Mild to Moderate",
        "criticality": 4,
        "description": "The most common form of arthritis, occurring when cartilage that cushions the ends of bones wears down over time.",
        "symptoms": ["Joint pain", "Stiffness", "Tenderness", "Loss of flexibility", "Bone spurs", "Swelling", "Grating sensation"],
        "causes": ["Joint damage over time", "Aging", "Obesity", "Joint injuries", "Repetitive stress", "Genetics"],
        "risk_factors": ["Older age", "Obesity", "Joint injuries", "Repetitive stress", "Genetics", "Bone deformities"],
        "treatments": ["Exercise", "Weight management", "Physical therapy", "Medications", "Injections", "Joint replacement surgery"],
        "medications": ["Acetaminophen", "NSAIDs (Ibuprofen, Naproxen)", "Duloxetine", "Cortisone injections", "Hyaluronic acid injections"],
        "prevention": ["Maintain healthy weight", "Stay active", "Protect joints from injury", "Control blood sugar"],
        "complications": ["Severe pain", "Decreased mobility", "Sleep problems", "Depression"],
        "when_to_seek_help": "For joint pain that doesn't improve, joint deformity, or inability to use the joint"
    },
    "osteoporosis": {
        "name": "Osteoporosis",
        "category": "Musculoskeletal",
        "severity": "Moderate",
        "criticality": 5,
        "description": "A bone disease that occurs when the body loses too much bone, makes too little bone, or both, causing bones to become weak and brittle.",
        "symptoms": ["Back pain", "Loss of height", "Stooped posture", "Bone fractures", "Often no symptoms until fracture"],
        "causes": ["Bone loss faster than bone creation", "Hormonal changes", "Calcium/Vitamin D deficiency", "Certain medications"],
        "risk_factors": ["Female gender", "Age", "Small body frame", "Family history", "Low calcium intake", "Smoking", "Excessive alcohol"],
        "treatments": ["Bisphosphonates", "Hormone therapy", "Bone-building medications", "Lifestyle modifications"],
        "medications": ["Alendronate (Fosamax)", "Risedronate", "Ibandronate", "Zoledronic acid", "Denosumab (Prolia)", "Teriparatide"],
        "prevention": ["Adequate calcium and Vitamin D", "Regular exercise", "Avoid smoking", "Limit alcohol", "Fall prevention"],
        "complications": ["Bone fractures (hip, spine, wrist)", "Height loss", "Chronic pain", "Disability"],
        "when_to_seek_help": "After any fall or injury, for sudden severe back pain, or if at high risk"
    },

    # Infectious Diseases
    "influenza": {
        "name": "Influenza (Flu)",
        "category": "Infectious Disease",
        "severity": "Mild to Severe",
        "criticality": 5,
        "description": "A contagious respiratory illness caused by influenza viruses that infect the nose, throat, and lungs.",
        "symptoms": ["Fever", "Cough", "Sore throat", "Body aches", "Headache", "Fatigue", "Runny nose", "Chills"],
        "causes": ["Influenza A, B, or C viruses", "Spread through respiratory droplets"],
        "risk_factors": ["Age under 5 or over 65", "Chronic conditions", "Weakened immune system", "Pregnancy", "Obesity"],
        "treatments": ["Rest and fluids", "Antiviral medications", "Over-the-counter symptom relief", "Hospitalization if severe"],
        "medications": ["Oseltamivir (Tamiflu)", "Zanamivir (Relenza)", "Baloxavir (Xofluza)", "Acetaminophen", "Ibuprofen"],
        "prevention": ["Annual flu vaccination", "Hand washing", "Avoid touching face", "Avoid sick people", "Cover coughs/sneezes"],
        "complications": ["Pneumonia", "Bronchitis", "Sinus infections", "Ear infections", "Myocarditis", "Encephalitis"],
        "when_to_seek_help": "For difficulty breathing, chest pain, confusion, severe vomiting, or symptoms that improve then worsen"
    },
    "covid-19": {
        "name": "COVID-19",
        "category": "Infectious Disease",
        "severity": "Mild to Severe",
        "criticality": 7,
        "description": "A respiratory illness caused by the SARS-CoV-2 coronavirus, ranging from mild to severe disease.",
        "symptoms": ["Fever", "Cough", "Shortness of breath", "Fatigue", "Body aches", "Loss of taste/smell", "Sore throat", "Headache", "Congestion"],
        "causes": ["SARS-CoV-2 virus", "Spread through respiratory droplets and aerosols"],
        "risk_factors": ["Older age", "Underlying conditions", "Unvaccinated status", "Obesity", "Immunocompromised"],
        "treatments": ["Supportive care", "Antiviral medications", "Monoclonal antibodies", "Steroids", "Oxygen/ventilation if severe"],
        "medications": ["Paxlovid", "Remdesivir", "Dexamethasone", "Molnupiravir", "Baricitinib"],
        "prevention": ["Vaccination", "Masking in high-risk settings", "Hand hygiene", "Good ventilation", "Testing when symptomatic"],
        "complications": ["Pneumonia", "ARDS", "Blood clots", "Long COVID", "Multi-organ failure", "Death"],
        "when_to_seek_help": "For difficulty breathing, persistent chest pain, confusion, inability to stay awake, or bluish lips/face"
    },
    "urinary tract infection": {
        "name": "Urinary Tract Infection (UTI)",
        "category": "Infectious Disease",
        "severity": "Mild to Moderate",
        "criticality": 4,
        "description": "An infection in any part of the urinary system, most commonly affecting the bladder and urethra.",
        "symptoms": ["Burning urination", "Frequent urination", "Urgent need to urinate", "Cloudy urine", "Blood in urine", "Pelvic pain", "Strong-smelling urine"],
        "causes": ["Bacteria (usually E. coli)", "Sexual activity", "Catheter use", "Urinary tract abnormalities"],
        "risk_factors": ["Female anatomy", "Sexual activity", "Certain birth control", "Menopause", "Urinary tract abnormalities", "Catheter use"],
        "treatments": ["Antibiotics", "Increased fluid intake", "Pain relief", "Preventive antibiotics for recurrent UTIs"],
        "medications": ["Trimethoprim-sulfamethoxazole", "Nitrofurantoin", "Ciprofloxacin", "Fosfomycin", "Phenazopyridine (pain relief)"],
        "prevention": ["Drink plenty of fluids", "Wipe front to back", "Urinate after intercourse", "Avoid irritating products"],
        "complications": ["Recurrent infections", "Kidney infection", "Sepsis (if untreated)", "Pregnancy complications"],
        "when_to_seek_help": "For fever, back pain, nausea/vomiting, or symptoms not improving with treatment"
    },

    # Cancer
    "breast cancer": {
        "name": "Breast Cancer",
        "category": "Oncology",
        "severity": "Severe",
        "criticality": 9,
        "description": "Cancer that forms in the cells of the breasts, most commonly beginning in the milk ducts or lobules.",
        "symptoms": ["Breast lump", "Breast shape/size change", "Nipple changes", "Nipple discharge", "Skin dimpling", "Redness/pitting of skin"],
        "causes": ["Genetic mutations (BRCA1, BRCA2)", "Hormonal factors", "Lifestyle factors", "Environmental factors"],
        "risk_factors": ["Female gender", "Age", "Family history", "Genetic mutations", "Hormone therapy", "Obesity", "Alcohol"],
        "treatments": ["Surgery", "Radiation therapy", "Chemotherapy", "Hormone therapy", "Targeted therapy", "Immunotherapy"],
        "medications": ["Tamoxifen", "Anastrozole", "Trastuzumab (Herceptin)", "Pertuzumab", "Palbociclib", "Chemotherapy drugs"],
        "prevention": ["Maintain healthy weight", "Exercise", "Limit alcohol", "Consider genetic testing if high risk", "Regular screening"],
        "complications": ["Metastasis", "Lymphedema", "Treatment side effects", "Recurrence"],
        "when_to_seek_help": "Immediately for any breast lump, changes in breast appearance, or nipple discharge"
    },
    "lung cancer": {
        "name": "Lung Cancer",
        "category": "Oncology",
        "severity": "Severe",
        "criticality": 10,
        "description": "A type of cancer that begins in the lungs, most often in people who smoke.",
        "symptoms": ["Persistent cough", "Coughing up blood", "Shortness of breath", "Chest pain", "Hoarseness", "Weight loss", "Bone pain", "Headache"],
        "causes": ["Smoking (primary cause)", "Secondhand smoke", "Radon exposure", "Asbestos", "Air pollution", "Genetic factors"],
        "risk_factors": ["Smoking", "Secondhand smoke exposure", "Radon exposure", "Family history", "Radiation therapy to chest"],
        "treatments": ["Surgery", "Radiation therapy", "Chemotherapy", "Targeted therapy", "Immunotherapy", "Palliative care"],
        "medications": ["Cisplatin", "Carboplatin", "Pembrolizumab (Keytruda)", "Osimertinib", "Crizotinib", "Bevacizumab"],
        "prevention": ["Don't smoke or quit smoking", "Avoid secondhand smoke", "Test home for radon", "Avoid carcinogens at work"],
        "complications": ["Metastasis", "Breathing difficulty", "Fluid accumulation", "Bleeding", "Pain"],
        "when_to_seek_help": "For persistent cough, coughing up blood, unexplained weight loss, or chest pain"
    },

    # Kidney
    "chronic kidney disease": {
        "name": "Chronic Kidney Disease (CKD)",
        "category": "Nephrology",
        "severity": "Moderate to Severe",
        "criticality": 8,
        "description": "A long-term condition where the kidneys don't work as well as they should, gradually losing function over time.",
        "symptoms": ["Nausea", "Vomiting", "Loss of appetite", "Fatigue", "Sleep problems", "Changes in urination", "Swelling", "Muscle cramps"],
        "causes": ["Diabetes", "High blood pressure", "Glomerulonephritis", "Polycystic kidney disease", "Prolonged urinary tract obstruction"],
        "risk_factors": ["Diabetes", "High blood pressure", "Heart disease", "Smoking", "Obesity", "Family history", "Age over 60"],
        "treatments": ["Treating underlying cause", "Blood pressure management", "Managing complications", "Dialysis", "Kidney transplant"],
        "medications": ["ACE inhibitors", "ARBs", "Diuretics", "Erythropoietin", "Phosphate binders", "Vitamin D supplements"],
        "prevention": ["Control diabetes and blood pressure", "Maintain healthy weight", "Don't smoke", "Limit NSAIDs", "Regular check-ups"],
        "complications": ["Fluid retention", "Anemia", "Heart disease", "Bone disease", "Kidney failure", "Death"],
        "when_to_seek_help": "For significant swelling, severe fatigue, confusion, or significant decrease in urination"
    },
    "kidney stones": {
        "name": "Kidney Stones",
        "category": "Nephrology",
        "severity": "Mild to Moderate",
        "criticality": 5,
        "description": "Hard deposits made of minerals and salts that form inside the kidneys.",
        "symptoms": ["Severe side/back pain", "Pain radiating to groin", "Painful urination", "Pink/red/brown urine", "Nausea/vomiting", "Frequent urination"],
        "causes": ["Not drinking enough water", "Diet high in protein/sodium/sugar", "Obesity", "Certain medical conditions", "Family history"],
        "risk_factors": ["Dehydration", "High-protein diet", "High-sodium diet", "Obesity", "Family history", "Certain medical conditions"],
        "treatments": ["Increased water intake", "Pain management", "Medical therapy", "Lithotripsy", "Ureteroscopy", "Surgery"],
        "medications": ["NSAIDs", "Alpha-blockers (Tamsulosin)", "Potassium citrate", "Allopurinol", "Thiazide diuretics"],
        "prevention": ["Drink plenty of water", "Limit sodium", "Limit animal protein", "Get enough calcium from food", "Limit oxalate-rich foods"],
        "complications": ["Recurring stones", "Urinary tract infection", "Kidney damage", "Obstruction"],
        "when_to_seek_help": "For severe pain with nausea/vomiting, fever with pain, blood in urine, or difficulty passing urine"
    }
}

# Disease name variations for fuzzy matching
DISEASE_ALIASES = {
    "high blood pressure": "hypertension",
    "hbp": "hypertension",
    "blood pressure": "hypertension",
    "bp": "hypertension",
    "chf": "heart failure",
    "congestive heart failure": "heart failure",
    "cad": "coronary artery disease",
    "heart disease": "coronary artery disease",
    "afib": "atrial fibrillation",
    "irregular heartbeat": "atrial fibrillation",
    "type 2 diabetes": "diabetes type 2",
    "t2d": "diabetes type 2",
    "type ii diabetes": "diabetes type 2",
    "diabetes mellitus": "diabetes type 2",
    "sugar": "diabetes type 2",
    "high sugar": "diabetes type 2",
    "blood sugar": "diabetes type 2",
    "type 1 diabetes": "diabetes type 1",
    "t1d": "diabetes type 1",
    "juvenile diabetes": "diabetes type 1",
    "underactive thyroid": "hypothyroidism",
    "thyroid": "hypothyroidism",
    "breathing problems": "asthma",
    "chronic bronchitis": "copd",
    "emphysema": "copd",
    "lung disease": "copd",
    "headache": "migraine",
    "migraines": "migraine",
    "alzheimers": "alzheimer's disease",
    "dementia": "alzheimer's disease",
    "memory loss": "alzheimer's disease",
    "parkinsons": "parkinson's disease",
    "tremor": "parkinson's disease",
    "seizures": "epilepsy",
    "convulsions": "epilepsy",
    "sad": "depression",
    "depressed": "depression",
    "sadness": "depression",
    "feeling low": "depression",
    "worry": "anxiety disorder",
    "anxiety": "anxiety disorder",
    "nervous": "anxiety disorder",
    "panic": "anxiety disorder",
    "bipolar": "bipolar disorder",
    "manic depression": "bipolar disorder",
    "acid reflux": "gerd",
    "heartburn": "gerd",
    "reflux": "gerd",
    "ibs": "irritable bowel syndrome",
    "stomach problems": "irritable bowel syndrome",
    "bowel problems": "irritable bowel syndrome",
    "crohns": "crohn's disease",
    "inflammatory bowel": "crohn's disease",
    "ra": "rheumatoid arthritis",
    "joint pain": "rheumatoid arthritis",
    "arthritis": "osteoarthritis",
    "oa": "osteoarthritis",
    "bone loss": "osteoporosis",
    "brittle bones": "osteoporosis",
    "flu": "influenza",
    "seasonal flu": "influenza",
    "corona": "covid-19",
    "coronavirus": "covid-19",
    "covid": "covid-19",
    "uti": "urinary tract infection",
    "bladder infection": "urinary tract infection",
    "ckd": "chronic kidney disease",
    "kidney failure": "chronic kidney disease",
    "renal disease": "chronic kidney disease",
    "stones": "kidney stones",
    "renal stones": "kidney stones"
}

# ========================================
# FUTURISTIC CSS
# ========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 25%, #161b22 50%, #0d1117 75%, #0a0a0f 100%);
    background-attachment: fixed;
}

/* Animated gradient overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(217, 70, 239, 0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1600px;
    position: relative;
    z-index: 1;
}

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 60px 20px 40px;
    animation: fadeInUp 0.8s ease-out;
}

.hero-icon {
    font-size: 4rem;
    margin-bottom: 20px;
    animation: pulse 2s ease-in-out infinite;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 30%, #a855f7 60%, #d946ef 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 15px;
    letter-spacing: -2px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.3rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}

.hero-tagline {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 300;
}

/* Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 24px;
    padding: 32px;
    margin: 16px 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-4px);
    box-shadow:
        0 20px 40px rgba(0, 0, 0, 0.3),
        0 0 60px rgba(99, 102, 241, 0.1);
}

.glass-card h3 {
    color: #ffffff;
    font-weight: 700;
    font-size: 1.4rem;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.glass-card p {
    color: #94a3b8;
    font-size: 1rem;
    line-height: 1.7;
    margin: 0;
}

/* Search Container */
.search-container {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.04));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 28px;
    padding: 40px;
    margin: 30px 0;
    position: relative;
    overflow: hidden;
}

.search-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.05) 0%, transparent 50%);
    animation: rotate 20s linear infinite;
}

/* Disease Card */
.disease-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 28px;
    margin: 16px 0;
    transition: all 0.3s ease;
}

.disease-card:hover {
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-2px);
}

.disease-card h4 {
    color: #a78bfa;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0 0 12px 0;
}

.disease-card p {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* Info Grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

/* Severity Badge */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.severity-low {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.severity-moderate {
    background: rgba(234, 179, 8, 0.15);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

.severity-high {
    background: rgba(249, 115, 22, 0.15);
    color: #f97316;
    border: 1px solid rgba(249, 115, 22, 0.3);
}

.severity-critical {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Symptom Tag */
.symptom-tag {
    display: inline-block;
    padding: 6px 14px;
    margin: 4px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    color: #a5b4fc;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}

.symptom-tag:hover {
    background: rgba(99, 102, 241, 0.2);
    transform: scale(1.05);
}

/* Medication Card */
.med-card {
    background: linear-gradient(145deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.02));
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s ease;
}

.med-card:hover {
    transform: translateX(8px);
    border-color: rgba(34, 197, 94, 0.4);
}

/* Warning Box */
.warning-box {
    background: linear-gradient(145deg, rgba(234, 179, 8, 0.1), rgba(234, 179, 8, 0.02));
    border: 1px solid rgba(234, 179, 8, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #fcd34d;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.danger-box {
    background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #fca5a5;
}

.success-box {
    background: linear-gradient(145deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.02));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #86efac;
}

.info-box {
    background: linear-gradient(145deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.02));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #93c5fd;
}

/* AI Card */
.ai-card {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.06));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 24px;
    padding: 32px;
    margin: 24px 0;
    box-shadow:
        0 8px 32px rgba(99, 102, 241, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.ai-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #d946ef, #8b5cf6, #6366f1);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}

/* Interaction Cards */
.interaction-severe {
    background: linear-gradient(145deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.04)) !important;
    border-left: 4px solid #ef4444 !important;
}

.interaction-moderate {
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.04)) !important;
    border-left: 4px solid #f59e0b !important;
}

.interaction-mild {
    background: linear-gradient(145deg, rgba(234, 179, 8, 0.08), rgba(234, 179, 8, 0.02)) !important;
    border-left: 4px solid #eab308 !important;
}

/* Status Pills */
.status-container {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 12px;
    margin: 20px 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.status-pill:hover {
    transform: scale(1.05);
}

.status-on {
    background: rgba(34, 197, 94, 0.12);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.25);
}

.status-off {
    background: rgba(234, 179, 8, 0.12);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.25);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px 36px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.5) !important;
}

/* Input Fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    color: white !important;
    padding: 16px 20px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 20px;
    padding: 8px;
    gap: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 14px;
    color: #94a3b8;
    font-weight: 500;
    padding: 14px 28px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.03);
    color: #e2e8f0;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

/* Pharmacy Card */
.pharmacy-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 24px;
    margin: 12px 0;
    transition: all 0.3s ease;
}

.pharmacy-card:hover {
    border-color: rgba(34, 197, 94, 0.3);
    transform: translateX(8px);
}

/* Result Card */
.result-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 24px;
    margin: 14px 0;
    transition: all 0.3s ease;
}

.result-card:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-2px);
}

.result-card h5 {
    color: #a78bfa;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 12px 0;
}

.result-card p {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* Criticality Meter */
.criticality-meter {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.criticality-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Footer */
.footer {
    text-align: center;
    padding: 50px 20px 30px;
    color: #64748b;
    font-size: 0.9rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 80px;
}

.footer a {
    color: #8b5cf6;
    text-decoration: none;
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
    background: rgba(99, 102, 241, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ========================================
# HELPER FUNCTIONS
# ========================================

@st.cache_resource
def load_clients():
    """Load all medical API clients"""
    clients = {
        'fda': get_fda_client(),
        'pubmed': get_pubmed_client(),
        'rxnorm': get_rxnorm_client(),
        'aggregator': get_medical_aggregator(),
        'trials': get_clinical_trials_client(),
        'disease': get_disease_client(),
        'pharmacy': get_pharmacy_finder(),
    }

    if IMAGE_PROCESSOR_AVAILABLE and get_image_processor:
        clients['image_processor'] = get_image_processor()
    else:
        clients['image_processor'] = None

    if RAG_AVAILABLE and get_rag_pipeline:
        clients['rag'] = get_rag_pipeline()
    else:
        clients['rag'] = None

    return clients

def get_api_keys():
    """Get API keys from Streamlit secrets"""
    keys = {'openai': None, 'gemini': None, 'groq': None}

    try:
        keys['openai'] = st.secrets.get("OPENAI_API_KEY", None)
    except:
        pass

    try:
        keys['gemini'] = st.secrets.get("GEMINI_API_KEY", None)
    except:
        pass

    try:
        keys['groq'] = st.secrets.get("GROQ_API_KEY", None)
    except:
        pass

    return keys

def call_llm_with_fallback(prompt: str, keys: dict):
    """Try all LLMs in order: Groq -> OpenAI -> Gemini"""
    errors = []

    # Try Groq first (fastest)
    if keys.get('groq'):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {keys['groq']}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 3000
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'], "Groq (Llama 3.3 70B)"
        except Exception as e:
            errors.append(f"Groq: {str(e)[:80]}")

    # Try OpenAI
    if keys.get('openai'):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {keys['openai']}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 3000
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'], "OpenAI (GPT-4o-mini)"
        except Exception as e:
            errors.append(f"OpenAI: {str(e)[:80]}")

    # Try Gemini
    if keys.get('gemini'):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={keys['gemini']}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 3000}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text'], "Google Gemini"
        except Exception as e:
            errors.append(f"Gemini: {str(e)[:80]}")

    return None, " | ".join(errors) if errors else "No API keys configured"

def find_disease(query: str) -> Optional[Dict]:
    """Find disease in database with fuzzy matching"""
    query_lower = query.lower().strip()

    # Direct match
    if query_lower in DISEASE_DATABASE:
        return DISEASE_DATABASE[query_lower]

    # Check aliases
    if query_lower in DISEASE_ALIASES:
        return DISEASE_DATABASE[DISEASE_ALIASES[query_lower]]

    # Partial match in disease names
    for disease_key, disease_data in DISEASE_DATABASE.items():
        if query_lower in disease_key or disease_key in query_lower:
            return disease_data

    # Partial match in aliases
    for alias, disease_key in DISEASE_ALIASES.items():
        if query_lower in alias or alias in query_lower:
            return DISEASE_DATABASE[disease_key]

    # Search by symptoms
    for disease_key, disease_data in DISEASE_DATABASE.items():
        symptoms_lower = [s.lower() for s in disease_data.get('symptoms', [])]
        for symptom in symptoms_lower:
            if query_lower in symptom or symptom in query_lower:
                return disease_data

    return None

def get_severity_color(criticality: int) -> tuple:
    """Get color based on criticality score"""
    if criticality <= 4:
        return "#22c55e", "Low"
    elif criticality <= 6:
        return "#eab308", "Moderate"
    elif criticality <= 8:
        return "#f97316", "High"
    else:
        return "#ef4444", "Critical"

def get_interaction_class(severity: str) -> str:
    """Get CSS class for interaction severity"""
    severity_lower = severity.lower()
    if 'major' in severity_lower or 'severe' in severity_lower or 'high' in severity_lower:
        return 'interaction-severe'
    elif 'moderate' in severity_lower:
        return 'interaction-moderate'
    return 'interaction-mild'

def build_disease_ai_prompt(disease_data: Dict, query: str) -> str:
    """Build AI prompt for disease analysis"""
    return f"""You are an expert medical AI assistant. Provide a comprehensive, easy-to-understand summary about the following condition.

CONDITION: {disease_data['name']}
USER QUERY: {query}

MEDICAL DATA:
- Category: {disease_data.get('category', 'Unknown')}
- Severity: {disease_data.get('severity', 'Unknown')}
- Description: {disease_data.get('description', 'No description')}
- Symptoms: {', '.join(disease_data.get('symptoms', [])[:8])}
- Causes: {', '.join(disease_data.get('causes', [])[:6])}
- Treatments: {', '.join(disease_data.get('treatments', [])[:6])}
- Medications: {', '.join(disease_data.get('medications', [])[:6])}

Please provide:
## Overview
A brief, patient-friendly summary of this condition.

## Key Symptoms to Watch
The most important symptoms patients should be aware of.

## Understanding the Causes
What leads to this condition in simple terms.

## Treatment Options
Modern treatment approaches available.

## Lifestyle Recommendations
Practical tips for managing or preventing this condition.

## When to Seek Medical Help
Clear guidance on when to see a doctor.

IMPORTANT: Write in a compassionate, easy-to-understand manner. Avoid overly technical jargon. Always emphasize consulting healthcare professionals for medical decisions.
"""

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    clients = load_clients()
    keys = get_api_keys()
    has_any_key = any([keys['openai'], keys['gemini'], keys['groq']])

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">🧬</div>
            <h1 class="hero-title">MedAI</h1>
            <p class="hero-subtitle">Clinical Intelligence Platform</p>
            <p class="hero-tagline">AI-Powered Medical Insights • Real-Time Research • Evidence-Based Care</p>
        </div>
    """, unsafe_allow_html=True)

    # Status Pills
    st.markdown('<div class="status-container">', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        st.markdown('<span class="status-pill status-on">● PubMed</span>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<span class="status-pill status-on">● FDA Database</span>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<span class="status-pill status-on">● RxNorm</span>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<span class="status-pill status-on">● Clinical Trials</span>', unsafe_allow_html=True)
    with cols[4]:
        ai_status = "status-on" if has_any_key else "status-off"
        ai_text = "AI Active" if has_any_key else "AI (No Key)"
        st.markdown(f'<span class="status-pill {ai_status}">● {ai_text}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Disease Search",
        "🤖 AI Assistant",
        "💊 Medications",
        "⚡ Interactions",
        "📚 Research",
        "📍 Find Pharmacy"
    ])

    # ========================================
    # TAB 1: DISEASE SEARCH
    # ========================================
    with tab1:
        st.markdown("""
            <div class="glass-card">
                <h3>🔍 Search Any Disease or Condition</h3>
                <p>Get comprehensive information about symptoms, causes, treatments, medications, and more.
                Our AI provides easy-to-understand summaries backed by medical databases.</p>
            </div>
        """, unsafe_allow_html=True)

        # Search Input
        search_query = st.text_input(
            "Search for a disease, condition, or symptom",
            placeholder="e.g., diabetes, high blood pressure, headache, chest pain, anxiety...",
            key="disease_search"
        )

        # Quick search suggestions
        st.markdown("**Quick searches:** ", unsafe_allow_html=True)
        quick_cols = st.columns(8)
        quick_searches = ["Diabetes", "Hypertension", "Asthma", "Depression", "Migraine", "Arthritis", "GERD", "UTI"]

        for i, disease in enumerate(quick_searches):
            with quick_cols[i]:
                if st.button(disease, key=f"quick_{disease}"):
                    search_query = disease.lower()

        if search_query:
            # Find disease in database
            disease_data = find_disease(search_query)

            if disease_data:
                st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Found:</strong> {disease_data['name']}
                    </div>
                """, unsafe_allow_html=True)

                # Disease Header
                color, severity_label = get_severity_color(disease_data.get('criticality', 5))

                st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
                            <div>
                                <h3 style="margin-bottom: 8px;">{disease_data['name']}</h3>
                                <p style="color: #8b5cf6; font-weight: 500; margin-bottom: 12px;">{disease_data.get('category', 'General')}</p>
                            </div>
                            <div style="text-align: right;">
                                <span class="severity-badge" style="background: {color}22; color: {color}; border-color: {color}55;">
                                    ● {disease_data.get('severity', 'Unknown')}
                                </span>
                                <div class="criticality-meter" style="width: 150px; margin-top: 10px;">
                                    <div class="criticality-fill" style="width: {disease_data.get('criticality', 5) * 10}%; background: {color};"></div>
                                </div>
                                <p style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Criticality: {disease_data.get('criticality', 5)}/10</p>
                            </div>
                        </div>
                        <p style="margin-top: 16px; font-size: 1.05rem; line-height: 1.7;">{disease_data.get('description', '')}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Information Grid
                col1, col2 = st.columns(2)

                with col1:
                    # Symptoms
                    st.markdown("""
                        <div class="disease-card">
                            <h4>🩺 Common Symptoms</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    symptoms_html = "".join([f'<span class="symptom-tag">{s}</span>' for s in disease_data.get('symptoms', [])])
                    st.markdown(f'<div style="margin: -10px 0 20px 0;">{symptoms_html}</div>', unsafe_allow_html=True)

                    # Causes
                    st.markdown("""
                        <div class="disease-card">
                            <h4>🔬 Causes & Risk Factors</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    causes = disease_data.get('causes', [])
                    risk_factors = disease_data.get('risk_factors', [])
                    for cause in causes[:5]:
                        st.markdown(f"- {cause}")
                    if risk_factors:
                        st.markdown("**Risk Factors:**")
                        for rf in risk_factors[:4]:
                            st.markdown(f"- {rf}")

                with col2:
                    # Treatments
                    st.markdown("""
                        <div class="disease-card">
                            <h4>💉 Treatment Options</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    for treatment in disease_data.get('treatments', []):
                        st.markdown(f"✓ {treatment}")

                    # Medications
                    st.markdown("""
                        <div class="disease-card">
                            <h4>💊 Common Medications</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    for med in disease_data.get('medications', [])[:6]:
                        st.markdown(f"""
                            <div class="med-card">
                                <span style="color: #22c55e;">💊</span>
                                <span style="color: #e2e8f0;">{med}</span>
                            </div>
                        """, unsafe_allow_html=True)

                # Prevention & Complications
                col3, col4 = st.columns(2)

                with col3:
                    prevention = disease_data.get('prevention', [])
                    if prevention:
                        st.markdown("""
                            <div class="disease-card">
                                <h4>🛡️ Prevention</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        if isinstance(prevention, list):
                            for p in prevention:
                                st.markdown(f"• {p}")
                        else:
                            st.markdown(prevention)

                with col4:
                    complications = disease_data.get('complications', [])
                    if complications:
                        st.markdown("""
                            <div class="disease-card">
                                <h4>⚠️ Potential Complications</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        for c in complications[:5]:
                            st.markdown(f"• {c}")

                # When to Seek Help
                when_to_seek = disease_data.get('when_to_seek_help', '')
                if when_to_seek:
                    st.markdown(f"""
                        <div class="warning-box">
                            <span style="font-size: 1.5rem;">⚕️</span>
                            <div>
                                <strong>When to Seek Medical Help</strong>
                                <p style="margin-top: 8px; color: #fef3c7;">{when_to_seek}</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # AI Summary
                if has_any_key:
                    st.markdown("### 🤖 AI Medical Summary")
                    with st.spinner("Generating comprehensive AI analysis..."):
                        prompt = build_disease_ai_prompt(disease_data, search_query)
                        response, provider = call_llm_with_fallback(prompt, keys)

                        if response:
                            st.markdown(f"""
                                <div class="ai-card">
                                    <small style="color: #64748b;">✨ Powered by {provider}</small>
                                    <div style="margin-top: 16px; color: #e2e8f0; line-height: 1.8;">
                                        {response}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"AI Error: {provider}")

                # Related Research
                st.markdown("### 📚 Related Research")
                with st.spinner("Fetching latest research..."):
                    research = clients['pubmed'].search_articles(disease_data['name'], max_results=5).get('articles', [])

                    if research:
                        for article in research[:3]:
                            st.markdown(f"""
                                <div class="result-card">
                                    <h5>{article.get('title', 'No title')}</h5>
                                    <p style="color: #8b5cf6; font-size: 0.85rem; margin-bottom: 8px;">
                                        {article.get('journal', 'Unknown')} • {article.get('year', 'Unknown')}
                                    </p>
                                    <p>{article.get('abstract', 'No abstract')[:250]}...</p>
                                    <a href="{article.get('url', '#')}" target="_blank" style="color: #a78bfa; text-decoration: none;">
                                        📖 Read on PubMed →
                                    </a>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                # Disease not found in database - search APIs
                st.markdown(f"""
                    <div class="info-box">
                        <strong>🔍 Searching medical databases for "{search_query}"...</strong>
                    </div>
                """, unsafe_allow_html=True)

                # Search PubMed
                with st.spinner("Searching medical literature..."):
                    research = clients['pubmed'].search_articles(search_query, max_results=8).get('articles', [])
                    disease_info = clients['disease'].get_disease_info(search_query)

                    if research:
                        st.markdown(f"""
                            <div class="success-box">
                                Found {len(research)} related research articles
                            </div>
                        """, unsafe_allow_html=True)

                        # AI Analysis
                        if has_any_key:
                            st.markdown("### 🤖 AI Analysis")
                            with st.spinner("Generating analysis..."):
                                research_summary = "\n".join([f"- {a.get('title', '')}" for a in research[:5]])
                                prompt = f"""Based on the following medical research about "{search_query}", provide a helpful summary for a patient:

Research Articles:
{research_summary}

Please provide:
1. What this condition/topic is about
2. Key findings from research
3. Important considerations for patients
4. When to consult a healthcare provider

Write in a patient-friendly manner and emphasize consulting healthcare professionals."""

                                response, provider = call_llm_with_fallback(prompt, keys)
                                if response:
                                    st.markdown(f"""
                                        <div class="ai-card">
                                            <small style="color: #64748b;">✨ Powered by {provider}</small>
                                            <div style="margin-top: 16px; color: #e2e8f0; line-height: 1.8;">{response}</div>
                                        </div>
                                    """, unsafe_allow_html=True)

                        # Display research
                        st.markdown("### 📚 Research Articles")
                        for article in research:
                            st.markdown(f"""
                                <div class="result-card">
                                    <h5>{article.get('title', 'No title')}</h5>
                                    <p style="color: #8b5cf6; font-size: 0.85rem; margin-bottom: 8px;">
                                        {article.get('journal', 'Unknown')} • {article.get('year', 'Unknown')}
                                    </p>
                                    <p>{article.get('abstract', 'No abstract')[:300]}...</p>
                                    <a href="{article.get('url', '#')}" target="_blank" style="color: #a78bfa;">
                                        📖 Read Full Article →
                                    </a>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="warning-box">
                                No results found. Try different keywords or check spelling.
                            </div>
                        """, unsafe_allow_html=True)

    # ========================================
    # TAB 2: AI ASSISTANT
    # ========================================
    with tab2:
        st.markdown("""
            <div class="glass-card">
                <h3>🤖 AI Clinical Assistant</h3>
                <p>Ask any medical question and get AI-powered insights backed by real-time research data,
                drug information, and clinical guidelines.</p>
            </div>
        """, unsafe_allow_html=True)

        # Patient Context
        with st.expander("👤 Patient Information (Optional)", expanded=False):
            col1, col2 = st.columns(2)
            age = col1.number_input("Age", 0, 120, 0, key="ai_age")
            gender = col2.selectbox("Gender", ["Not specified", "Male", "Female", "Other"], key="ai_gender")
            history = col1.text_input("Medical History", placeholder="Diabetes, Hypertension...", key="ai_history")
            meds = col2.text_input("Current Medications", placeholder="Metformin, Lisinopril...", key="ai_meds")
            allergies = st.text_input("Allergies", placeholder="Penicillin, Sulfa...", key="ai_allergies")

        # Query Input
        query = st.text_area(
            "Your clinical question",
            placeholder="What are the best treatment options for Type 2 diabetes in elderly patients with kidney disease?",
            height=120,
            key="ai_query"
        )

        if st.button("✨ Analyze", key="ai_analyze"):
            if query:
                # Build patient context
                patient_context = ""
                if age > 0:
                    patient_context = f"Age: {age}, Gender: {gender}"
                if history:
                    patient_context += f", Medical History: {history}"
                if allergies:
                    patient_context += f", Allergies: {allergies}"

                med_list = [m.strip() for m in meds.split(',') if m.strip()] if meds else []

                progress = st.progress(0)
                status = st.empty()

                # Search Research
                status.text("🔍 Searching medical databases...")
                progress.progress(25)
                research = clients['pubmed'].search_articles(query, max_results=10).get('articles', [])

                # Drug Information
                progress.progress(50)
                drug_info = {}
                if med_list:
                    status.text("💊 Analyzing medications...")
                    drug_info = {m: clients['fda'].get_drug_info_summary(m) for m in med_list}

                # Interactions
                progress.progress(75)
                interactions = []
                if len(med_list) >= 2:
                    status.text("⚡ Checking interactions...")
                    interactions = clients['rxnorm'].get_interactions(med_list).get('interactions', [])

                progress.progress(100)
                status.empty()

                # Summary
                summary_parts = [f"✅ Found {len(research)} research articles"]
                if drug_info:
                    summary_parts.append(f"{len(drug_info)} medications analyzed")
                if interactions:
                    summary_parts.append(f"⚠️ {len(interactions)} interactions detected")

                st.markdown(f'<div class="success-box">{" • ".join(summary_parts)}</div>', unsafe_allow_html=True)

                # AI Response
                if has_any_key:
                    status.text("🤖 Generating AI analysis...")

                    # Build comprehensive prompt
                    prompt = f"""You are an expert Clinical Decision Support AI assistant.

CLINICAL QUERY: {query}

PATIENT CONTEXT: {patient_context if patient_context else 'Not provided'}

RESEARCH ARTICLES ({len(research)} found):
"""
                    for i, a in enumerate(research[:5], 1):
                        prompt += f"{i}. {a.get('title', 'No title')} ({a.get('year', 'Unknown')})\n"
                        prompt += f"   {a.get('abstract', 'No abstract')[:200]}...\n\n"

                    if drug_info:
                        prompt += "\nMEDICATION INFORMATION:\n"
                        for d, info in drug_info.items():
                            if info.get('found'):
                                prompt += f"- {d}: "
                                if info.get('indications'):
                                    prompt += f"Indications: {info['indications'][0][:150]}...\n"

                    if interactions:
                        prompt += "\nDRUG INTERACTIONS:\n"
                        for inter in interactions:
                            prompt += f"- {inter.get('severity', 'Unknown')} Severity: {inter.get('description', '')[:100]}...\n"

                    prompt += """

Provide a comprehensive clinical response with:
## 🔍 Clinical Assessment
## 📋 Research-Based Findings
## 💊 Medication Considerations
## ⚠️ Important Warnings
## 💡 Evidence-Based Recommendations
## 📚 References

IMPORTANT: This is for educational purposes. Always recommend consulting healthcare professionals."""

                    response, provider = call_llm_with_fallback(prompt, keys)
                    status.empty()

                    if response:
                        st.markdown(f"""
                            <div class="ai-card">
                                <small style="color: #64748b;">✨ Powered by {provider}</small>
                                <div style="margin-top: 16px; color: #e2e8f0; line-height: 1.8;">{response}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"AI Error: {provider}")
                else:
                    st.markdown("""
                        <div class="warning-box">
                            ⚠️ <strong>AI features require API keys.</strong><br><br>
                            Add keys in Streamlit Secrets:<br>
                            • GROQ_API_KEY (Free: console.groq.com/keys)<br>
                            • OPENAI_API_KEY<br>
                            • GEMINI_API_KEY (Free: aistudio.google.com/apikey)
                        </div>
                    """, unsafe_allow_html=True)

                # Research Results
                if research:
                    st.markdown("### 📚 Supporting Research")
                    for a in research[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <h5>{a.get('title', 'No title')}</h5>
                                <p style="color: #8b5cf6; font-size: 0.85rem;">{a.get('journal', 'Unknown')} • {a.get('year', 'Unknown')}</p>
                                <p>{a.get('abstract', 'No abstract')[:200]}...</p>
                                <a href="{a.get('url', '#')}" target="_blank" style="color: #a78bfa;">📖 View →</a>
                            </div>
                        """, unsafe_allow_html=True)

    # ========================================
    # TAB 3: MEDICATIONS
    # ========================================
    with tab3:
        st.markdown("""
            <div class="glass-card">
                <h3>💊 Drug Information Center</h3>
                <p>Get comprehensive information about any medication including uses, dosage, warnings, side effects, and more.</p>
            </div>
        """, unsafe_allow_html=True)

        drug_name = st.text_input("Enter medication name", placeholder="Metformin, Lisinopril, Atorvastatin...", key="drug_search")

        if st.button("🔍 Search Drug", key="drug_btn"):
            if drug_name:
                with st.spinner(f"Fetching information for {drug_name}..."):
                    info = clients['fda'].get_drug_info_summary(drug_name)

                if info.get('found'):
                    st.markdown(f"""
                        <div class="success-box">
                            ✅ Found comprehensive information for <strong>{drug_name.upper()}</strong>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        if info.get('indications'):
                            st.markdown("### 📋 Indications & Uses")
                            st.markdown(info['indications'][0][:600] + "..." if len(info['indications'][0]) > 600 else info['indications'][0])

                        if info.get('dosage'):
                            st.markdown("### 💉 Dosage & Administration")
                            st.markdown(info['dosage'][0][:600] + "..." if len(info['dosage'][0]) > 600 else info['dosage'][0])

                    with col2:
                        if info.get('warnings'):
                            st.markdown("### ⚠️ Warnings")
                            st.markdown(f"""
                                <div class="warning-box">
                                    {info['warnings'][0][:500]}...
                                </div>
                            """, unsafe_allow_html=True)

                        if info.get('contraindications'):
                            st.markdown("### 🚫 Contraindications")
                            st.markdown(info['contraindications'][0][:400] + "..." if len(info['contraindications'][0]) > 400 else info['contraindications'][0])

                    if info.get('common_adverse_events'):
                        st.markdown("### 🔴 Common Side Effects")
                        effects_html = "".join([f'<span class="symptom-tag" style="background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.2); color: #fca5a5;">{e}</span>' for e in info['common_adverse_events'][:12]])
                        st.markdown(f'<div>{effects_html}</div>', unsafe_allow_html=True)

                    if info.get('interactions'):
                        st.markdown("### ⚡ Drug Interactions")
                        st.markdown(info['interactions'][0][:500] + "..." if len(info['interactions'][0]) > 500 else info['interactions'][0])
                else:
                    st.markdown("""
                        <div class="warning-box">
                            Drug not found. Try the generic name or check spelling.
                        </div>
                    """, unsafe_allow_html=True)

    # ========================================
    # TAB 4: INTERACTIONS
    # ========================================
    with tab4:
        st.markdown("""
            <div class="glass-card">
                <h3>⚡ Drug Interaction Checker</h3>
                <p>Check for potentially dangerous drug combinations. Enter your medications to identify interactions and safety concerns.</p>
            </div>
        """, unsafe_allow_html=True)

        drugs_input = st.text_area(
            "Enter medications (one per line or comma-separated)",
            placeholder="Warfarin\nAspirin\nIbuprofen\n\nor: Warfarin, Aspirin, Ibuprofen",
            height=150,
            key="interaction_drugs"
        )

        if st.button("⚡ Check Interactions", key="interaction_btn"):
            # Parse drugs
            drug_list = []
            for line in drugs_input.split('\n'):
                drug_list.extend([d.strip() for d in line.split(',') if d.strip()])

            if len(drug_list) >= 2:
                with st.spinner("Analyzing drug interactions..."):
                    result = clients['rxnorm'].get_interactions(drug_list)

                interactions = result.get('interactions', [])

                if interactions:
                    # Count by severity
                    severe_count = sum(1 for i in interactions if 'major' in i.get('severity', '').lower() or 'severe' in i.get('severity', '').lower())
                    moderate_count = sum(1 for i in interactions if 'moderate' in i.get('severity', '').lower())

                    if severe_count > 0:
                        st.markdown(f"""
                            <div class="danger-box">
                                <strong>🚨 CRITICAL: {severe_count} severe interaction(s) detected!</strong>
                                <p style="margin-top: 8px;">These combinations may be dangerous. Consult your healthcare provider immediately.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    elif moderate_count > 0:
                        st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠️ {len(interactions)} interaction(s) found ({moderate_count} moderate)</strong>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Interaction Details")

                    for inter in interactions:
                        severity = inter.get('severity', 'Unknown')
                        severity_class = get_interaction_class(severity)
                        drugs_involved = ', '.join(inter.get('drugs', []))
                        description = inter.get('description', 'No description available')

                        severity_icon = "🔴" if 'major' in severity.lower() or 'severe' in severity.lower() else "🟡" if 'moderate' in severity.lower() else "🟢"

                        st.markdown(f"""
                            <div class="result-card {severity_class}">
                                <h5>{severity_icon} {severity} Severity Interaction</h5>
                                <p><strong>Drugs:</strong> {drugs_involved}</p>
                                <p style="margin-top: 12px;">{description}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-box">
                            ✅ <strong>No known interactions detected</strong> between the listed medications.
                            <p style="margin-top: 8px; color: #86efac;">Always inform your healthcare provider about all medications you're taking.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box">
                        Please enter at least 2 medications to check for interactions.
                    </div>
                """, unsafe_allow_html=True)

    # ========================================
    # TAB 5: RESEARCH
    # ========================================
    with tab5:
        st.markdown("""
            <div class="glass-card">
                <h3>📚 Medical Research & Clinical Trials</h3>
                <p>Search millions of PubMed articles and find active clinical trials for any condition.</p>
            </div>
        """, unsafe_allow_html=True)

        research_query = st.text_input("Search medical literature", placeholder="SGLT2 inhibitors heart failure outcomes", key="research_search")

        col1, col2 = st.columns([3, 1])
        with col2:
            recent_only = st.checkbox("Recent only (5 years)", value=True)

        if st.button("🔍 Search Research", key="research_btn"):
            if research_query:
                with st.spinner("Searching PubMed and ClinicalTrials.gov..."):
                    articles = clients['pubmed'].search_articles(research_query, max_results=15, recent_only=recent_only).get('articles', [])
                    trials = clients['trials'].search_trials(research_query, limit=5).get('trials', [])

                if articles:
                    st.markdown(f"""
                        <div class="success-box">
                            Found {len(articles)} research articles and {len(trials)} clinical trials
                        </div>
                    """, unsafe_allow_html=True)

                    # Clinical Trials
                    if trials:
                        st.markdown("### 🔬 Active Clinical Trials")
                        for trial in trials:
                            status_color = "#22c55e" if trial.get('status') == 'RECRUITING' else "#eab308"
                            st.markdown(f"""
                                <div class="result-card">
                                    <h5>{trial.get('title', 'No title')}</h5>
                                    <p style="margin-bottom: 8px;">
                                        <span style="color: {status_color}; font-weight: 600;">● {trial.get('status', 'Unknown')}</span>
                                        <span style="color: #64748b;"> | Phase: {', '.join(trial.get('phase', ['Unknown']))}</span>
                                    </p>
                                    <p>{trial.get('summary', '')[:250]}...</p>
                                    <a href="{trial.get('url', '#')}" target="_blank" style="color: #a78bfa;">View Trial Details →</a>
                                </div>
                            """, unsafe_allow_html=True)

                    # Research Articles
                    st.markdown("### 📄 Research Articles")
                    for article in articles:
                        st.markdown(f"""
                            <div class="result-card">
                                <h5>{article.get('title', 'No title')}</h5>
                                <p style="color: #8b5cf6; font-size: 0.85rem; margin-bottom: 8px;">
                                    {article.get('journal', 'Unknown')} • {article.get('year', 'Unknown')}
                                </p>
                                <p>{article.get('abstract', 'No abstract')[:300]}...</p>
                                <a href="{article.get('url', '#')}" target="_blank" style="color: #a78bfa;">📖 Read on PubMed →</a>
                            </div>
                        """, unsafe_allow_html=True)

    # ========================================
    # TAB 6: FIND PHARMACY
    # ========================================
    with tab6:
        st.markdown("""
            <div class="glass-card">
                <h3>📍 Find Nearby Pharmacies</h3>
                <p>Locate pharmacies near you. Enter your ZIP code or full address to find the closest pharmacy locations with contact information.</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            address = st.text_input(
                "Enter ZIP code or address",
                placeholder="e.g., 90210 or 123 Main St, Los Angeles, CA",
                key="pharmacy_address",
                help="Enter a 5-digit ZIP code or full street address"
            )

        with col2:
            drug_needed = st.text_input(
                "Medication needed (optional)",
                placeholder="e.g., Metformin",
                key="pharmacy_drug"
            )

        with col3:
            search_radius = st.selectbox(
                "Search radius",
                options=["5 miles", "10 miles", "15 miles", "25 miles"],
                index=1,
                key="pharmacy_radius"
            )

        # Convert miles to meters for API
        radius_map = {"5 miles": 8000, "10 miles": 16000, "15 miles": 24000, "25 miles": 40000}
        radius_meters = radius_map.get(search_radius, 16000)

        if st.button("🔍 Find Pharmacies", key="pharmacy_btn", use_container_width=True):
            if address:
                with st.spinner("📍 Locating your area and searching for pharmacies..."):
                    # Geocode address
                    geocode = clients['pharmacy'].geocode_address(address)

                    if geocode.get('success'):
                        lat, lon = geocode['latitude'], geocode['longitude']

                        # Show location confirmation
                        location_display = geocode.get('display_name', address)
                        st.markdown(f"""
                            <div class="info-box">
                                📍 Searching near: <strong>{location_display}</strong>
                            </div>
                        """, unsafe_allow_html=True)

                        result = clients['pharmacy'].find_nearby_pharmacies(lat, lon, drug_needed, radius=radius_meters)

                        if result.get('pharmacies'):
                            pharmacy_count = len(result['pharmacies'])
                            st.markdown(f"""
                                <div class="success-box">
                                    ✅ Found <strong>{pharmacy_count}</strong> pharmacies within {search_radius} of your location
                                </div>
                            """, unsafe_allow_html=True)

                            for idx, pharm in enumerate(result['pharmacies'], 1):
                                distance_miles = pharm.get('distance_miles', pharm.get('distance_km', 0) * 0.621371)
                                distance_km = pharm.get('distance_km', 0)
                                name = pharm.get('name', 'Pharmacy')
                                address_text = pharm.get('address', 'Address not available')
                                phone = pharm.get('phone', '')
                                website = pharm.get('website', '')
                                hours = pharm.get('opening_hours', '')

                                # Build additional info line
                                extra_info = []
                                if phone:
                                    extra_info.append(f"📞 {phone}")
                                if hours:
                                    extra_info.append(f"🕐 {hours}")

                                extra_html = f'<p style="color: #64748b; font-size: 0.85rem; margin-top: 8px;">{" • ".join(extra_info)}</p>' if extra_info else ''

                                website_html = f'<a href="{website}" target="_blank" style="color: #8b5cf6; font-size: 0.85rem; text-decoration: none;">🌐 Website</a>' if website else ''

                                st.markdown(f"""
                                    <div class="pharmacy-card">
                                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
                                            <div style="flex: 1;">
                                                <h4 style="color: #22c55e; margin: 0 0 8px 0;">
                                                    <span style="color: #64748b; font-size: 0.9rem;">#{idx}</span> 🏥 {name}
                                                </h4>
                                                <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem;">{address_text}</p>
                                                {extra_html}
                                                {website_html}
                                            </div>
                                            <div style="text-align: right; min-width: 100px;">
                                                <span style="color: #8b5cf6; font-weight: 700; font-size: 1.2rem;">
                                                    {distance_miles:.1f} mi
                                                </span>
                                                <p style="color: #64748b; font-size: 0.8rem; margin: 4px 0 0 0;">
                                                    ({distance_km:.1f} km)
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                            # Add note about data source
                            st.markdown("""
                                <p style="color: #64748b; font-size: 0.8rem; text-align: center; margin-top: 20px;">
                                    📊 Data sourced from OpenStreetMap. Some pharmacies may not be listed.
                                    For the most accurate results, also check Google Maps or call ahead.
                                </p>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="warning-box">
                                    <strong>No pharmacies found within {search_radius}.</strong><br>
                                    Try increasing the search radius or entering a different location.
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="danger-box">
                                <strong>Could not find location.</strong><br>
                                {geocode.get('error', 'Please check your ZIP code or address and try again.')}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box">
                        Please enter a ZIP code or address to search for pharmacies.
                    </div>
                """, unsafe_allow_html=True)

    # ========================================
    # FOOTER
    # ========================================
    st.markdown("""
        <div class="footer">
            <p style="font-weight: 600; color: #94a3b8; margin-bottom: 12px;">⚠️ Important Medical Disclaimer</p>
            <p>This application is for <strong>educational and informational purposes only</strong>.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.</p>
            <p style="margin-top: 12px;">Always consult qualified healthcare professionals for medical decisions.</p>
            <p style="margin-top: 20px; color: #475569;">
                Built with ❤️ by Nitish Kumar Manthri |
                <span style="color: #8b5cf6;">MedAI Clinical Decision Support System</span>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
