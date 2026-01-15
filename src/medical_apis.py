"""
Medical APIs Module
Real-time integration with openFDA, PubMed, and RxNorm APIs
"""
import requests
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


class OpenFDAClient:
    """Client for openFDA API - Drug information and adverse events"""
    
    BASE_URL = "https://api.fda.gov"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def search_drug(self, drug_name: str, limit: int = 5) -> Dict:
        """Search for drug information by name"""
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
    
    def get_drug_interactions(self, drug_name: str) -> Dict:
        """Get drug interaction information"""
        endpoint = f"{self.BASE_URL}/drug/label.json"
        params = {'search': f'openfda.brand_name:"{drug_name}"', 'limit': 1}
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                return {
                    'drug_name': drug_name,
                    'drug_interactions': result.get('drug_interactions', ['No data']),
                    'warnings': result.get('warnings', ['No data']),
                    'contraindications': result.get('contraindications', ['No data'])
                }
            return {'drug_name': drug_name, 'message': 'No data found'}
        except:
            return {'error': 'API request failed'}
    
    def get_adverse_events(self, drug_name: str, limit: int = 10) -> Dict:
        """Get adverse event reports for a drug"""
        endpoint = f"{self.BASE_URL}/drug/event.json"
        params = {
            'search': f'patient.drug.medicinalproduct:"{drug_name}"',
            'limit': limit
        }
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            events = []
            if 'results' in data:
                for result in data['results']:
                    patient = result.get('patient', {})
                    reactions = patient.get('reaction', [])
                    for reaction in reactions:
                        events.append(reaction.get('reactionmeddrapt', 'Unknown'))
            return {
                'drug_name': drug_name,
                'adverse_events': list(set(events))[:20],
                'total_reports': data.get('meta', {}).get('results', {}).get('total', 0)
            }
        except:
            return {'error': 'API request failed'}
    
    def get_drug_info_summary(self, drug_name: str) -> Dict:
        """Get comprehensive drug information summary"""
        drug_data = self.search_drug(drug_name, limit=1)
        adverse = self.get_adverse_events(drug_name, limit=5)
        
        summary = {
            'drug_name': drug_name,
            'found': False,
            'indications': [],
            'dosage': [],
            'warnings': [],
            'contraindications': [],
            'interactions': [],
            'common_adverse_events': []
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
    """Client for PubMed/NCBI API - Medical research papers"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        self.session = requests.Session()
    
    def search_articles(self, query: str, max_results: int = 5) -> Dict:
        """Search PubMed for medical articles"""
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        try:
            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            ids = data.get('esearchresult', {}).get('idlist', [])
            
            if not ids:
                return {'query': query, 'articles': [], 'count': 0}
            
            articles = self._fetch_article_details(ids)
            
            return {
                'query': query,
                'articles': articles,
                'count': len(articles)
            }
        except:
            return {'error': 'PubMed API request failed', 'articles': []}
    
    def _fetch_article_details(self, ids: List[str]) -> List[Dict]:
        """Fetch details for a list of PubMed IDs"""
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        
        try:
            response = self.session.get(fetch_url, params=fetch_params, timeout=15)
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
                    'abstract': abstract_elem.text[:500] + '...' if abstract_elem is not None and abstract_elem.text and len(abstract_elem.text) > 500 else (abstract_elem.text if abstract_elem is not None and abstract_elem.text else 'No abstract'),
                    'year': year_elem.text if year_elem is not None else 'Unknown',
                    'journal': journal_elem.text if journal_elem is not None else 'Unknown',
                    'pmid': pmid_elem.text if pmid_elem is not None else None,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else None
                })
            
            return articles
        except:
            return []


class RxNormClient:
    """Client for RxNorm API - Drug names and interactions"""
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """Get RxNorm drug information"""
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
                        drugs.append({
                            'rxcui': prop.get('rxcui'),
                            'name': prop.get('name'),
                            'synonym': prop.get('synonym', ''),
                            'tty': prop.get('tty')
                        })
            
            return {
                'drug_name': drug_name,
                'found': len(drugs) > 0,
                'drugs': drugs[:5]
            }
        except:
            return {'error': 'RxNorm API request failed'}
    
    def get_interactions(self, drug_names: List[str]) -> Dict:
        """Check for drug-drug interactions"""
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
            interaction_groups = data.get('fullInteractionTypeGroup', [])
            
            for group in interaction_groups:
                for itype in group.get('fullInteractionType', []):
                    for pair in itype.get('interactionPair', []):
                        interactions.append({
                            'severity': pair.get('severity', 'Unknown'),
                            'description': pair.get('description', 'No description'),
                            'drugs': [d.get('minConceptItem', {}).get('name', 'Unknown') 
                                     for d in pair.get('interactionConcept', [])]
                        })
            
            return {
                'drug_names': drug_names,
                'interactions_found': len(interactions) > 0,
                'interactions': interactions
            }
        except:
            return {'error': 'Interaction check failed', 'interactions': []}


class MedicalDataAggregator:
    """Aggregates data from multiple medical sources"""
    
    def __init__(self):
        self.fda_client = OpenFDAClient()
        self.pubmed_client = PubMedClient()
        self.rxnorm_client = RxNormClient()
    
    def get_comprehensive_drug_report(self, drug_names: List[str]) -> Dict:
        """Get comprehensive report for multiple drugs"""
        report = {
            'drugs': {},
            'interactions': [],
            'potential_interactions': []
        }
        
        for drug in drug_names:
            report['drugs'][drug] = self.fda_client.get_drug_info_summary(drug)
        
        if len(drug_names) >= 2:
            interactions = self.rxnorm_client.get_interactions(drug_names)
            report['interactions'] = interactions.get('interactions', [])
            for inter in report['interactions']:
                report['potential_interactions'].append(inter.get('description', ''))
        
        return report
    
    def clinical_query(self, query: str, medications: List[str] = None) -> Dict:
        """Process a clinical query with real-time data"""
        result = {
            'query': query,
            'research': [],
            'drug_info': {},
            'interactions': []
        }
        
        result['research'] = self.pubmed_client.search_articles(query, max_results=5).get('articles', [])
        
        if medications:
            for med in medications:
                result['drug_info'][med] = self.fda_client.get_drug_info_summary(med)
            
            if len(medications) >= 2:
                interactions = self.rxnorm_client.get_interactions(medications)
                result['interactions'] = interactions.get('interactions', [])
        
        return result


# Singleton instances
_fda_client = None
_pubmed_client = None
_rxnorm_client = None
_aggregator = None

def get_fda_client() -> OpenFDAClient:
    global _fda_client
    if _fda_client is None:
        _fda_client = OpenFDAClient()
    return _fda_client

def get_pubmed_client() -> PubMedClient:
    global _pubmed_client
    if _pubmed_client is None:
        _pubmed_client = PubMedClient()
    return _pubmed_client

def get_rxnorm_client() -> RxNormClient:
    global _rxnorm_client
    if _rxnorm_client is None:
        _rxnorm_client = RxNormClient()
    return _rxnorm_client

def get_medical_aggregator() -> MedicalDataAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = MedicalDataAggregator()
    return _aggregator
