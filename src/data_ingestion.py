"""
Document Ingestion Pipeline
Loads, chunks, embeds, and indexes medical documents into the FAISS vector store.
"""
import os
import glob
import json
from typing import List, Dict, Optional
from datetime import datetime

from .logging_config import get_logger, timed
from .vector_store import VectorStore
from .config import settings

log = get_logger("data_ingestion")


def load_markdown_files(directory: str) -> List[Dict]:
    """
    Load all markdown files from a directory.

    Args:
        directory: Path to directory containing .md files

    Returns:
        List of dicts with 'content', 'metadata' keys
    """
    documents = []
    md_files = sorted(glob.glob(os.path.join(directory, "*.md")))

    for filepath in md_files:
        filename = os.path.basename(filepath)
        category = _infer_category(filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError as e:
            log.warning(f"Failed to read {filepath}: {e}")
            continue

        if not content:
            log.warning(f"Skipping empty file: {filepath}")
            continue

        documents.append({
            "content": content,
            "metadata": {
                "source": filename,
                "category": category,
                "type": "guideline",
                "last_updated": datetime.now().isoformat(),
                "file_path": filepath,
            },
        })
        log.debug(f"Loaded {filename} ({len(content)} chars, category={category})")

    log.info(f"Loaded {len(documents)} markdown files from {directory}")
    return documents


def _infer_category(filename: str) -> str:
    """Infer medical category from filename."""
    name = filename.lower().replace(".md", "").replace("_", " ")
    category_map = {
        "diabetes": "endocrinology",
        "thyroid": "endocrinology",
        "adrenal": "endocrinology",
        "hypertension": "cardiology",
        "cad": "cardiology",
        "acs": "cardiology",
        "stemi": "cardiology",
        "nstemi": "cardiology",
        "atrial": "cardiology",
        "heart failure": "cardiology",
        "dvt": "cardiology",
        "anticoagulation": "cardiology",
        "copd": "pulmonology",
        "asthma": "pulmonology",
        "pneumonia": "pulmonology",
        "stroke": "neurology",
        "tbi": "neurology",
        "seizure": "neurology",
        "sepsis": "infectious_disease",
        "uti": "infectious_disease",
        "aki": "nephrology",
        "ckd": "nephrology",
        "electrolyte": "nephrology",
        "drug interaction": "pharmacology",
        "pain": "pharmacology",
        "transfusion": "hematology",
        "anaphylaxis": "emergency_medicine",
        "gi bleeding": "gastroenterology",
        "depression": "psychiatry",
        "suicide": "psychiatry",
        "cancer": "oncology",
        "oncologic": "oncology",
        "pediatric": "pediatrics",
        "geriatric": "geriatrics",
        "falls": "geriatrics",
        "polypharmacy": "geriatrics",
        "pregnancy": "obstetrics",
        "gestational": "obstetrics",
        "palliative": "palliative_care",
    }
    for keyword, category in category_map.items():
        if keyword in name:
            return category
    return "general_medicine"


def chunk_document(
    document: Dict,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Dict]:
    """
    Split a document into overlapping chunks.

    Args:
        document: Dict with 'content' and 'metadata'
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunked documents with inherited metadata
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    content = document["content"]
    metadata = document.get("metadata", {})

    if len(content) <= chunk_size:
        return [document]

    chunks = []
    # Split on paragraph boundaries first
    paragraphs = content.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single paragraph exceeds chunk_size, split it
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk = current_chunk + " " + word if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        result.append({"content": chunk_text.strip(), "metadata": chunk_meta})

    return result


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Dict]:
    """Chunk a list of documents."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    log.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


@timed(name="data_ingestion.ingest_directory")
def ingest_directory(
    directory: str,
    vector_store: VectorStore,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> int:
    """
    Ingest all markdown files from a directory into the vector store.

    Args:
        directory: Path to directory containing .md files
        vector_store: VectorStore instance to add documents to
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        Number of chunks indexed
    """
    documents = load_markdown_files(directory)
    if not documents:
        log.warning(f"No documents found in {directory}")
        return 0

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    log.info(f"Indexed {len(chunks)} chunks from {len(documents)} files")
    return len(chunks)


def ingest_inline_documents(
    documents: List[Dict],
    vector_store: VectorStore,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> int:
    """
    Ingest inline document dicts (e.g. from init_vectorstore.py) into the vector store.

    Args:
        documents: List of dicts with 'content' and 'metadata'
        vector_store: VectorStore instance
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Number of chunks indexed
    """
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    log.info(f"Indexed {len(chunks)} chunks from {len(documents)} inline documents")
    return len(chunks)


def get_ingestion_stats(vector_store: VectorStore) -> Dict:
    """Get statistics about the ingested knowledge base."""
    categories = {}
    sources = set()
    for doc in vector_store.documents:
        meta = doc.get("metadata", {})
        cat = meta.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        sources.add(meta.get("source", "unknown"))

    return {
        "total_documents": vector_store.count,
        "categories": categories,
        "unique_sources": len(sources),
        "sources": sorted(sources),
    }
