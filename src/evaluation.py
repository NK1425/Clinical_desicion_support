"""
RAG Evaluation Framework
Measures retrieval quality and response latency for the clinical decision support system.

Usage:
    python -m src.evaluation
"""
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore, get_vector_store
from src.config import settings
from src.logging_config import get_logger

log = get_logger("evaluation")


def load_evaluation_data(path: str = None) -> List[Dict]:
    """Load evaluation Q&A pairs from JSON file."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "evaluation_data.json",
        )
    with open(path, "r") as f:
        return json.load(f)


def evaluate_retrieval(
    vector_store: VectorStore,
    eval_data: List[Dict],
    k_values: List[int] = None,
) -> Dict:
    """
    Evaluate retrieval quality.

    Metrics:
    - Precision@K: Fraction of retrieved docs containing expected keywords
    - MRR: Mean Reciprocal Rank of first relevant document
    - Latency: P50, P95, P99 retrieval time

    Args:
        vector_store: Initialized vector store
        eval_data: List of evaluation items
        k_values: List of K values for Precision@K

    Returns:
        Dict with evaluation results
    """
    if k_values is None:
        k_values = [1, 3, 5]

    max_k = max(k_values)
    precision_scores = {k: [] for k in k_values}
    mrr_scores = []
    latencies = []

    for item in eval_data:
        query = item["query"]
        expected_sources = [s.lower() for s in item.get("expected_sources", [])]
        expected_answer = item.get("expected_answer", "").lower()

        # Time the retrieval
        start = time.perf_counter()
        results = vector_store.search(query, k=max_k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        # Check relevance of each result
        relevance = []
        for doc in results:
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})
            source = str(metadata.get("source", "")).lower()
            category = str(metadata.get("category", "")).lower()

            is_relevant = False
            # Check if expected keywords appear in content or metadata
            for expected in expected_sources:
                if expected in content or expected in source or expected in category:
                    is_relevant = True
                    break
            # Also check if expected answer appears in content
            if expected_answer and expected_answer in content:
                is_relevant = True

            relevance.append(is_relevant)

        # Precision@K
        for k in k_values:
            top_k_relevant = relevance[:k]
            if top_k_relevant:
                precision_scores[k].append(sum(top_k_relevant) / k)
            else:
                precision_scores[k].append(0.0)

        # MRR
        rr = 0.0
        for i, rel in enumerate(relevance):
            if rel:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

    # Aggregate
    latencies.sort()
    n = len(latencies)

    results = {
        "num_queries": len(eval_data),
        "precision": {},
        "mrr": _mean(mrr_scores),
        "latency": {
            "p50_seconds": latencies[int(n * 0.5)] if n else 0,
            "p95_seconds": latencies[int(n * 0.95)] if n else 0,
            "p99_seconds": latencies[int(n * 0.99)] if n else 0,
            "mean_seconds": _mean(latencies),
        },
    }

    for k in k_values:
        results["precision"][f"P@{k}"] = round(_mean(precision_scores[k]), 4)

    return results


def _mean(values: List[float]) -> float:
    """Calculate mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_evaluation(
    eval_data_path: str = None,
    index_path: str = None,
    output_dir: str = None,
) -> Dict:
    """
    Run full evaluation and save results.

    Args:
        eval_data_path: Path to evaluation data JSON
        index_path: Path to vector store index
        output_dir: Directory to save results

    Returns:
        Evaluation results dict
    """
    # Load eval data
    eval_data = load_evaluation_data(eval_data_path)
    log.info(f"Loaded {len(eval_data)} evaluation queries")

    # Load vector store
    idx_path = index_path or settings.vector_store_path
    vector_store = VectorStore(index_path=idx_path)
    log.info(f"Vector store loaded: {vector_store.count} documents")

    if vector_store.count == 0:
        log.error("Vector store is empty. Run init_vectorstore.py first.")
        return {"error": "Empty vector store"}

    # Run evaluation
    log.info("Running retrieval evaluation...")
    results = evaluate_retrieval(vector_store, eval_data)
    results["timestamp"] = datetime.now().isoformat()
    results["vector_store_size"] = vector_store.count

    # Save results
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as latest
    latest_file = os.path.join(output_dir, "eval_latest.json")
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_results(results: Dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("RAG Evaluation Results")
    print("=" * 60)
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Queries evaluated: {results.get('num_queries', 0)}")
    print(f"Vector store size: {results.get('vector_store_size', 0)}")

    print("\n--- Retrieval Precision ---")
    for metric, value in results.get("precision", {}).items():
        status = "PASS" if value >= 0.6 else "NEEDS IMPROVEMENT"
        print(f"  {metric}: {value:.4f}  [{status}]")

    print(f"\n--- Mean Reciprocal Rank ---")
    mrr = results.get("mrr", 0)
    print(f"  MRR: {mrr:.4f}")

    print(f"\n--- Retrieval Latency ---")
    latency = results.get("latency", {})
    print(f"  P50: {latency.get('p50_seconds', 0):.4f}s")
    print(f"  P95: {latency.get('p95_seconds', 0):.4f}s")
    print(f"  P99: {latency.get('p99_seconds', 0):.4f}s")
    print(f"  Mean: {latency.get('mean_seconds', 0):.4f}s")

    print("=" * 60)


def main():
    """CLI entry point."""
    results = run_evaluation()
    if "error" not in results:
        print_results(results)
    else:
        print(f"Evaluation failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
