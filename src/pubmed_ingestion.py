"""
PubMed Abstract Ingestion
Downloads and indexes PubMed abstracts for key medical conditions.

Usage:
    python -m src.pubmed_ingestion --conditions "diabetes,hypertension,COPD" --max-per-condition 30
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime

import requests

from .logging_config import get_logger, timed
from .data_ingestion import chunk_documents
from .vector_store import VectorStore
from .config import settings

log = get_logger("pubmed_ingestion")

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 30) -> List[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query
        max_results: Maximum number of results

    Returns:
        List of PMID strings
    """
    params = {
        "db": "pubmed",
        "term": f"{query} AND review[pt]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(PUBMED_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        log.info(f"Found {len(pmids)} PMIDs for query '{query}'")
        return pmids
    except requests.RequestException as e:
        log.error(f"PubMed search failed for '{query}': {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Fetch abstracts for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of dicts with title, abstract, journal, year, pmid
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }

    try:
        resp = requests.get(PUBMED_FETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"PubMed fetch failed: {e}")
        return []

    articles = []
    try:
        root = ET.fromstring(resp.content)
        for article_elem in root.findall(".//PubmedArticle"):
            article = _parse_article(article_elem)
            if article and article.get("abstract"):
                articles.append(article)
    except ET.ParseError as e:
        log.error(f"XML parsing failed: {e}")
        return []

    log.info(f"Fetched {len(articles)} abstracts from {len(pmids)} PMIDs")
    return articles


def _parse_article(elem) -> Optional[Dict]:
    """Parse a PubmedArticle XML element."""
    try:
        medline = elem.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "unknown"

        article = medline.find("Article")
        if article is None:
            return None

        # Title
        title_elem = article.find("ArticleTitle")
        title = title_elem.text if title_elem is not None else "Untitled"

        # Abstract
        abstract_elem = article.find(".//AbstractText")
        if abstract_elem is None:
            # Try multiple AbstractText sections
            abstract_parts = article.findall(".//AbstractText")
            if abstract_parts:
                abstract = " ".join(
                    (part.text or "") for part in abstract_parts
                )
            else:
                abstract = ""
        else:
            abstract = abstract_elem.text or ""

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

        # Year
        year_elem = article.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article.find(".//PubDate/MedlineDate")
        year = year_elem.text[:4] if year_elem is not None and year_elem.text else "Unknown"

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
        }
    except (AttributeError, TypeError) as e:
        log.debug(f"Failed to parse article: {e}")
        return None


def articles_to_documents(articles: List[Dict], condition: str) -> List[Dict]:
    """Convert PubMed articles to document dicts for indexing."""
    documents = []
    for article in articles:
        content = f"# {article['title']}\n\n{article['abstract']}"
        metadata = {
            "source": f"PubMed PMID:{article['pmid']}",
            "category": _condition_to_category(condition),
            "type": "pubmed_abstract",
            "pmid": article["pmid"],
            "journal": article["journal"],
            "year": article["year"],
            "condition": condition,
            "last_updated": datetime.now().isoformat(),
        }
        documents.append({"content": content, "metadata": metadata})
    return documents


def _condition_to_category(condition: str) -> str:
    """Map condition name to medical category."""
    mapping = {
        "diabetes": "endocrinology",
        "hypertension": "cardiology",
        "heart failure": "cardiology",
        "atrial fibrillation": "cardiology",
        "copd": "pulmonology",
        "asthma": "pulmonology",
        "pneumonia": "infectious_disease",
        "sepsis": "infectious_disease",
        "stroke": "neurology",
        "epilepsy": "neurology",
        "ckd": "nephrology",
        "chronic kidney disease": "nephrology",
        "depression": "psychiatry",
        "cancer": "oncology",
    }
    return mapping.get(condition.lower(), "general_medicine")


@timed(name="pubmed_ingestion.ingest_conditions")
def ingest_conditions(
    conditions: List[str],
    vector_store: VectorStore,
    max_per_condition: int = 30,
) -> Dict:
    """
    Ingest PubMed abstracts for multiple conditions.

    Args:
        conditions: List of medical condition search terms
        vector_store: VectorStore to index into
        max_per_condition: Max abstracts per condition

    Returns:
        Stats dict with counts per condition
    """
    stats = {"conditions": {}, "total_articles": 0, "total_chunks": 0}

    for condition in conditions:
        log.info(f"Ingesting PubMed abstracts for: {condition}")
        pmids = search_pubmed(condition, max_results=max_per_condition)
        if not pmids:
            stats["conditions"][condition] = {"articles": 0, "chunks": 0}
            continue

        articles = fetch_abstracts(pmids)
        documents = articles_to_documents(articles, condition)
        chunks = chunk_documents(documents)
        vector_store.add_documents(chunks)

        stats["conditions"][condition] = {
            "articles": len(articles),
            "chunks": len(chunks),
        }
        stats["total_articles"] += len(articles)
        stats["total_chunks"] += len(chunks)

        # Rate limiting - PubMed asks for max 3 requests/second without API key
        time.sleep(0.5)

    log.info(
        f"PubMed ingestion complete: {stats['total_articles']} articles, "
        f"{stats['total_chunks']} chunks"
    )
    return stats


def main():
    """CLI entry point for PubMed ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PubMed abstracts into the clinical knowledge base"
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="diabetes,hypertension,COPD,heart failure,sepsis",
        help="Comma-separated list of conditions to search",
    )
    parser.add_argument(
        "--max-per-condition",
        type=int,
        default=30,
        help="Maximum abstracts per condition",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to vector store index",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    index_path = args.index_path or settings.vector_store_path

    print(f"PubMed Ingestion")
    print(f"Conditions: {conditions}")
    print(f"Max per condition: {args.max_per_condition}")
    print(f"Index path: {index_path}")
    print("=" * 60)

    vector_store = VectorStore(index_path=index_path)
    stats = ingest_conditions(conditions, vector_store, args.max_per_condition)
    vector_store.save()

    print("\nResults:")
    for condition, info in stats["conditions"].items():
        print(f"  {condition}: {info['articles']} articles, {info['chunks']} chunks")
    print(f"\nTotal: {stats['total_articles']} articles, {stats['total_chunks']} chunks")
    print(f"Vector store total: {vector_store.count} documents")


if __name__ == "__main__":
    main()
