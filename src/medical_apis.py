"""
Medical APIs Module
Integrates with openFDA, PubMed, and RxNorm for real-time medical information.
"""
import math
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from .logging_config import get_logger, timed

log = get_logger("medical_apis")


class OpenFDAClient:
    """Client for openFDA API — Drug information and adverse events."""

    BASE_URL = "https://api.fda.gov"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    @timed(name="openfda.search_drug")
    def search_drug(self, drug_name: str, limit: int = 5) -> Dict:
        """Search for drug information by name."""
        endpoint = f"{self.BASE_URL}/drug/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": limit,
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            log.warning(f"Timeout searching drug: {drug_name}")
            return {"error": "Request timed out", "results": []}
        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error searching drug {drug_name}: {e}")
            return {"error": f"Connection error: {e}", "results": []}
        except requests.exceptions.HTTPError as e:
            log.warning(f"HTTP error searching drug {drug_name}: {e}")
            return {"error": str(e), "results": []}
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed for drug {drug_name}: {e}")
            return {"error": str(e), "results": []}

    @timed(name="openfda.get_drug_interactions")
    def get_drug_interactions(self, drug_name: str) -> Dict:
        """Get drug interaction information."""
        endpoint = f"{self.BASE_URL}/drug/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}"',
            "limit": 1,
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return {
                    "drug_name": drug_name,
                    "drug_interactions": result.get("drug_interactions", ["No interaction data available"]),
                    "warnings": result.get("warnings", ["No warnings available"]),
                    "contraindications": result.get("contraindications", ["No contraindications available"]),
                }
            return {"drug_name": drug_name, "message": "No data found"}

        except requests.exceptions.Timeout:
            log.warning(f"Timeout getting interactions for: {drug_name}")
            return {"drug_name": drug_name, "error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to get interactions for {drug_name}: {e}")
            return {"drug_name": drug_name, "error": str(e)}

    @timed(name="openfda.get_adverse_events")
    def get_adverse_events(self, drug_name: str, limit: int = 10) -> Dict:
        """Get adverse event reports for a drug."""
        endpoint = f"{self.BASE_URL}/drug/event.json"
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": limit,
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            events = []
            if "results" in data:
                for result in data["results"]:
                    patient = result.get("patient", {})
                    reactions = patient.get("reaction", [])
                    for reaction in reactions:
                        events.append(reaction.get("reactionmeddrapt", "Unknown"))

            return {
                "drug_name": drug_name,
                "adverse_events": list(set(events))[:20],
                "total_reports": data.get("meta", {}).get("results", {}).get("total", 0),
            }

        except requests.exceptions.Timeout:
            log.warning(f"Timeout getting adverse events for: {drug_name}")
            return {"drug_name": drug_name, "error": "Request timed out", "adverse_events": []}
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to get adverse events for {drug_name}: {e}")
            return {"drug_name": drug_name, "error": str(e), "adverse_events": []}

    def get_drug_info_summary(self, drug_name: str) -> Dict:
        """Get comprehensive drug information summary."""
        summary = {
            "drug_name": drug_name,
            "found": False,
            "indications": [],
            "dosage": [],
            "warnings": [],
            "contraindications": [],
            "interactions": [],
            "common_adverse_events": [],
        }

        drug_data = self.search_drug(drug_name, limit=1)
        interactions = self.get_drug_interactions(drug_name)
        adverse = self.get_adverse_events(drug_name, limit=5)

        if "results" in drug_data and len(drug_data["results"]) > 0:
            result = drug_data["results"][0]
            summary["found"] = True
            summary["indications"] = result.get("indications_and_usage", ["Not available"])
            summary["dosage"] = result.get("dosage_and_administration", ["Not available"])
            summary["warnings"] = result.get("warnings", ["Not available"])
            summary["contraindications"] = result.get("contraindications", ["Not available"])

        if "drug_interactions" in interactions:
            summary["interactions"] = interactions["drug_interactions"]

        if "adverse_events" in adverse:
            summary["common_adverse_events"] = adverse["adverse_events"]

        return summary


class PubMedClient:
    """Client for PubMed/NCBI API — Medical research papers."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.session = requests.Session()

    def search_articles(self, query: str, max_results: int = 5, recent_only: bool = False) -> Dict:
        """Search PubMed for medical articles."""
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        enhanced_query = query
        if recent_only:
            from datetime import datetime
            year = datetime.now().year
            enhanced_query = f"({query}) AND ({year - 5}[PDAT] : {year}[PDAT])"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": min(max_results, 50),
            "retmode": "json",
            "sort": "relevance",
        }

        try:
            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            ids = data.get("esearchresult", {}).get("idlist", [])

            if not ids:
                return {"query": query, "articles": [], "count": 0}

            articles = self._fetch_article_details(ids)
            return {"query": query, "articles": articles, "count": len(articles)}
        except requests.exceptions.RequestException as e:
            log.error(f"PubMed search failed: {e}")
            return {"error": "PubMed API request failed", "articles": []}

    def _fetch_article_details(self, ids: List[str]) -> List[Dict]:
        """Fetch details for a list of PubMed IDs."""
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}

        try:
            response = self.session.get(fetch_url, params=fetch_params, timeout=15)
            response.raise_for_status()

            articles = []
            root = ET.fromstring(response.content)

            for article in root.findall(".//PubmedArticle"):
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                year_elem = article.find(".//PubDate/Year")
                journal_elem = article.find(".//Journal/Title")
                pmid_elem = article.find(".//PMID")

                abstract_text = ""
                if abstract_elem is not None and abstract_elem.text:
                    abstract_text = abstract_elem.text[:500] + "..." if len(abstract_elem.text) > 500 else abstract_elem.text

                articles.append({
                    "title": title_elem.text if title_elem is not None else "No title",
                    "abstract": abstract_text or "No abstract",
                    "year": year_elem.text if year_elem is not None else "Unknown",
                    "journal": journal_elem.text if journal_elem is not None else "Unknown",
                    "pmid": pmid_elem.text if pmid_elem is not None else None,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else None,
                })

            return articles
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            log.error(f"PubMed fetch failed: {e}")
            return []


class RxNormClient:
    """Client for RxNorm API — Drug names and interactions."""

    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    def __init__(self):
        self.session = requests.Session()

    def get_drug_info(self, drug_name: str) -> Dict:
        """Get RxNorm drug information."""
        search_url = f"{self.BASE_URL}/drugs.json"
        params = {"name": drug_name}

        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            concepts = data.get("drugGroup", {}).get("conceptGroup", [])
            drugs = []
            for group in concepts:
                if "conceptProperties" in group:
                    for prop in group["conceptProperties"]:
                        drugs.append({
                            "rxcui": prop.get("rxcui"),
                            "name": prop.get("name"),
                            "synonym": prop.get("synonym", ""),
                            "tty": prop.get("tty"),
                        })

            return {"drug_name": drug_name, "found": len(drugs) > 0, "drugs": drugs[:5]}
        except requests.exceptions.RequestException as e:
            log.error(f"RxNorm lookup failed for {drug_name}: {e}")
            return {"error": "RxNorm API request failed"}

    def get_interactions(self, drug_names: List[str]) -> Dict:
        """Check for drug-drug interactions."""
        rxcuis = []
        for drug in drug_names:
            info = self.get_drug_info(drug)
            if info.get("drugs"):
                rxcuis.append(info["drugs"][0]["rxcui"])

        if len(rxcuis) < 2:
            return {"interactions": [], "message": "Need at least 2 valid drugs"}

        interaction_url = f"{self.BASE_URL}/interaction/list.json"
        params = {"rxcuis": "+".join(rxcuis)}

        try:
            response = self.session.get(interaction_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            interactions = []
            interaction_groups = data.get("fullInteractionTypeGroup", [])

            for group in interaction_groups:
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        interactions.append({
                            "severity": pair.get("severity", "Unknown"),
                            "description": pair.get("description", "No description"),
                            "drugs": [
                                d.get("minConceptItem", {}).get("name", "Unknown")
                                for d in pair.get("interactionConcept", [])
                            ],
                        })

            return {
                "drug_names": drug_names,
                "interactions_found": len(interactions) > 0,
                "interactions": interactions,
            }
        except requests.exceptions.RequestException as e:
            log.error(f"RxNorm interaction check failed: {e}")
            return {"error": "Interaction check failed", "interactions": []}


class MedicalDataAggregator:
    """Aggregates data from multiple medical sources."""

    def __init__(self):
        self.fda_client = OpenFDAClient()
        self.pubmed_client = PubMedClient()
        self.rxnorm_client = RxNormClient()

    def get_comprehensive_drug_report(self, drug_names: List[str]) -> Dict:
        """Get comprehensive report for multiple drugs."""
        report = {"drugs": {}, "interactions": [], "potential_interactions": []}

        for drug in drug_names:
            report["drugs"][drug] = self.fda_client.get_drug_info_summary(drug)

        if len(drug_names) >= 2:
            interactions = self.rxnorm_client.get_interactions(drug_names)
            report["interactions"] = interactions.get("interactions", [])
            for inter in report["interactions"]:
                report["potential_interactions"].append(inter.get("description", ""))

        return report

    def clinical_query(self, query: str, medications: List[str] = None) -> Dict:
        """Process a clinical query with real-time data."""
        result = {"query": query, "research": [], "drug_info": {}, "interactions": []}

        result["research"] = self.pubmed_client.search_articles(query, max_results=5).get("articles", [])

        if medications:
            for med in medications:
                result["drug_info"][med] = self.fda_client.get_drug_info_summary(med)

            if len(medications) >= 2:
                interactions = self.rxnorm_client.get_interactions(medications)
                result["interactions"] = interactions.get("interactions", [])

        return result


# Singleton instances
_fda_client = None
_pubmed_client = None
_rxnorm_client = None
_aggregator = None


def get_fda_client() -> OpenFDAClient:
    """Get OpenFDA client singleton."""
    global _fda_client
    if _fda_client is None:
        _fda_client = OpenFDAClient()
    return _fda_client


def get_pubmed_client() -> PubMedClient:
    """Get PubMed client singleton."""
    global _pubmed_client
    if _pubmed_client is None:
        _pubmed_client = PubMedClient()
    return _pubmed_client


def get_rxnorm_client() -> RxNormClient:
    """Get RxNorm client singleton."""
    global _rxnorm_client
    if _rxnorm_client is None:
        _rxnorm_client = RxNormClient()
    return _rxnorm_client


def get_medical_aggregator() -> MedicalDataAggregator:
    """Get medical data aggregator singleton."""
    global _aggregator
    if _aggregator is None:
        _aggregator = MedicalDataAggregator()
    return _aggregator


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        self.session = requests.Session()

    @timed(name="clinicaltrials.search")
    def search_trials(self, condition: str, status: str = "RECRUITING", limit: int = 10) -> Dict:
        """Search for clinical trials by condition."""
        endpoint = f"{self.BASE_URL}/studies"
        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "pageSize": limit,
            "format": "json",
        }
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            trials = []
            for study in data.get("studies", [])[:limit]:
                protocol = study.get("protocolSection", {})
                identification = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                desc = protocol.get("descriptionModule", {})
                trials.append({
                    "nct_id": identification.get("nctId", ""),
                    "title": identification.get("briefTitle", "No title"),
                    "status": status_module.get("overallStatus", "Unknown"),
                    "phase": status_module.get("phases", []),
                    "summary": desc.get("briefSummary", ""),
                    "url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}",
                })
            return {"condition": condition, "trials": trials, "count": len(trials)}
        except requests.exceptions.RequestException as e:
            log.error(f"ClinicalTrials search failed: {e}")
            return {"error": "ClinicalTrials API request failed", "trials": []}


class DiseaseInfoClient:
    """Client for disease information lookups."""

    def __init__(self):
        self.session = requests.Session()

    def get_disease_info(self, disease_name: str) -> Dict:
        """Get disease information."""
        return {"disease_name": disease_name, "research_articles": []}

    def suggest_medications(self, disease_name: str) -> Dict:
        """Suggest medications for a disease."""
        return {"disease": disease_name, "suggested_medications": []}


class PharmacyFinderClient:
    """Client for finding US pharmacies including major chains via Overpass API."""

    PHARMACY_CHAINS = {
        "walgreens": {"name": "Walgreens", "website": "https://www.walgreens.com/storelocator/find.jsp"},
        "cvs": {"name": "CVS Pharmacy", "website": "https://www.cvs.com/store-locator/landing"},
        "walmart": {"name": "Walmart Pharmacy", "website": "https://www.walmart.com/store-finder"},
        "kroger": {"name": "Kroger Pharmacy", "website": "https://www.kroger.com/stores/search"},
        "rite_aid": {"name": "Rite Aid", "website": "https://www.riteaid.com/locations"},
        "costco": {"name": "Costco Pharmacy", "website": "https://www.costco.com/warehouse-locations"},
        "sams_club": {"name": "Sam's Club Pharmacy", "website": "https://www.samsclub.com/locator"},
        "publix": {"name": "Publix Pharmacy", "website": "https://www.publix.com/locations"},
        "heb": {"name": "H-E-B Pharmacy", "website": "https://www.heb.com/store-locations"},
        "target": {"name": "Target (CVS)", "website": "https://www.target.com/store-locator/find-stores"},
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MedAI-ClinicalSupport/1.0 (contact@medai.health)",
            "Accept": "application/json",
        })

    @timed(name="pharmacy.find_nearby")
    def find_nearby_pharmacies(self, lat: float, lon: float, drug_name: str = None, radius: int = 16000) -> Dict:
        """Find pharmacies within radius (meters) using Overpass API."""
        try:
            radius_km = radius / 1000
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
[out:json][timeout:60];
(
  node["amenity"="pharmacy"](around:{radius},{lat},{lon});
  way["amenity"="pharmacy"](around:{radius},{lat},{lon});
  node["shop"="pharmacy"](around:{radius},{lat},{lon});
  node["shop"="chemist"](around:{radius},{lat},{lon});
  node["healthcare"="pharmacy"](around:{radius},{lat},{lon});
);
out body center;
"""
            response = self.session.post(overpass_url, data={"data": overpass_query}, timeout=60)
            response.raise_for_status()
            data = response.json()

            pharmacies = []
            seen_locations = set()

            for element in data.get("elements", []):
                if element.get("type") == "way":
                    elem_lat = element.get("center", {}).get("lat")
                    elem_lon = element.get("center", {}).get("lon")
                else:
                    elem_lat = element.get("lat")
                    elem_lon = element.get("lon")

                if elem_lat is None or elem_lon is None:
                    continue

                loc_key = f"{round(elem_lat, 4)},{round(elem_lon, 4)}"
                if loc_key in seen_locations:
                    continue
                seen_locations.add(loc_key)

                distance = self._calculate_distance(lat, lon, elem_lat, elem_lon)
                if distance > radius_km * 1.1:
                    continue

                tags = element.get("tags", {})
                name = tags.get("name") or tags.get("brand") or tags.get("operator") or "Pharmacy"

                addr_parts = []
                if tags.get("addr:housenumber"):
                    addr_parts.append(tags["addr:housenumber"])
                if tags.get("addr:street"):
                    addr_parts.append(tags["addr:street"])
                if tags.get("addr:city"):
                    addr_parts.append(tags["addr:city"])
                if tags.get("addr:state"):
                    addr_parts.append(tags["addr:state"])
                if tags.get("addr:postcode"):
                    addr_parts.append(tags["addr:postcode"])

                address = ", ".join(addr_parts) if addr_parts else f"Location: {elem_lat:.4f}, {elem_lon:.4f}"
                chain_type = self._identify_chain(name)

                pharmacies.append({
                    "name": name,
                    "address": address,
                    "latitude": elem_lat,
                    "longitude": elem_lon,
                    "distance_km": round(distance, 2),
                    "distance_miles": round(distance * 0.621371, 2),
                    "phone": tags.get("phone") or tags.get("contact:phone") or "",
                    "website": tags.get("website") or "",
                    "hours": tags.get("opening_hours") or "",
                    "chain": chain_type,
                })

            pharmacies.sort(key=lambda x: x["distance_km"])
            return {
                "pharmacies": pharmacies[:25],
                "count": len(pharmacies),
                "search_location": {"lat": lat, "lon": lon},
                "radius_miles": round(radius_km * 0.621371, 1),
            }
        except requests.exceptions.RequestException as e:
            log.error(f"Pharmacy search failed: {e}")
            return {"error": str(e), "pharmacies": []}

    def _identify_chain(self, name: str) -> str:
        """Identify pharmacy chain from name."""
        name_lower = name.lower()
        chains = [
            ("walgreen", "Walgreens"), ("cvs", "CVS"), ("walmart", "Walmart"),
            ("kroger", "Kroger"), ("rite aid", "Rite Aid"), ("riteaid", "Rite Aid"),
            ("costco", "Costco"), ("sam's", "Sam's Club"), ("sams", "Sam's Club"),
            ("publix", "Publix"), ("target", "Target"), ("heb", "H-E-B"), ("h-e-b", "H-E-B"),
        ]
        for keyword, chain_name in chains:
            if keyword in name_lower:
                return chain_name
        return "Independent"

    def geocode_address(self, address: str) -> Dict:
        """Convert US ZIP code to coordinates using Zippopotam.us API."""
        try:
            zip_code = address.strip()
            if not (zip_code.isdigit() and len(zip_code) == 5):
                return {"success": False, "error": "Please enter a valid 5-digit US ZIP code (e.g., 38119)"}

            endpoint = f"https://api.zippopotam.us/us/{zip_code}"
            response = self.session.get(endpoint, timeout=10)

            if response.status_code == 404:
                return {"success": False, "error": f"ZIP code {zip_code} not found. Please check and try again."}

            response.raise_for_status()
            data = response.json()

            if data and "places" in data and len(data["places"]) > 0:
                place = data["places"][0]
                city = place.get("place name", "")
                state = place.get("state", "")
                state_abbr = place.get("state abbreviation", "")
                lat = float(place.get("latitude", 0))
                lon = float(place.get("longitude", 0))
                display = f"{city}, {state_abbr} {zip_code}"
                return {
                    "success": True, "latitude": lat, "longitude": lon,
                    "display_name": display, "city": city, "state": state,
                    "state_abbr": state_abbr, "zip": zip_code,
                }
            return {"success": False, "error": f"Could not find location for ZIP code {zip_code}"}
        except requests.exceptions.RequestException:
            return {"success": False, "error": "Network error. Please try again."}

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in kilometers using Haversine formula."""
        R = 6371
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def get_chain_websites(self) -> Dict:
        """Return pharmacy chain websites for drug lookup."""
        return self.PHARMACY_CHAINS


# Additional singleton instances
_clinical_trials_client = None
_disease_client = None
_pharmacy_finder = None


def get_clinical_trials_client() -> ClinicalTrialsClient:
    """Get ClinicalTrials client singleton."""
    global _clinical_trials_client
    if _clinical_trials_client is None:
        _clinical_trials_client = ClinicalTrialsClient()
    return _clinical_trials_client


def get_disease_client() -> DiseaseInfoClient:
    """Get DiseaseInfo client singleton."""
    global _disease_client
    if _disease_client is None:
        _disease_client = DiseaseInfoClient()
    return _disease_client


def get_pharmacy_finder() -> PharmacyFinderClient:
    """Get PharmacyFinder client singleton."""
    global _pharmacy_finder
    if _pharmacy_finder is None:
        _pharmacy_finder = PharmacyFinderClient()
    return _pharmacy_finder
