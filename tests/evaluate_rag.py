"""Évaluation RAGAS du pipeline (nécessite MISTRAL_API_KEY + annotated_qa.json)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

try:
    from config.settings import MISTRAL_API_KEY
except ImportError:
    MISTRAL_API_KEY = None

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS_FILE = ROOT / "annotated_qa.json"
RESULTS_FILE = ROOT / "ragas_evaluation_results.csv"

EVALUATION_LIMIT = 3
RETRIEVAL_K = 15
MIN_FAITHFULNESS = 0.50
MIN_ANSWER_RELEVANCY = 0.40


def load_benchmarks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    if not isinstance(benchmarks, list):
        return []
    return benchmarks[:EVALUATION_LIMIT]


def compute_mean(values: Any) -> float:
    if isinstance(values, (int, float)):
        return float(values)
    if values is None:
        return 0.0
    valid = []
    for v in values:
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if n == n:
            valid.append(n)
    return sum(valid) / len(valid) if valid else 0.0


def run_ragas_evaluation() -> None:
    """Lance l'évaluation RAGAS — nécessite MISTRAL_API_KEY + deps ragas/instructor."""
    try:
        from ragas import RunConfig, evaluate, SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
    except ImportError as e:
        logger.error("ragas non installé : %s", e)
        return

    try:
        from api.faiss_retriever import FAISSRetriever
        from api.mistral_qa import generate_response
    except Exception as e:
        logger.error("Modules RAG indisponibles : %s", e)
        return

    if not BENCHMARKS_FILE.exists():
        logger.error("Benchmark absent : %s", BENCHMARKS_FILE)
        return

    benchmarks = load_benchmarks(BENCHMARKS_FILE)
    if not benchmarks:
        raise ValueError("Benchmark vide.")

    logger.info("Évaluation de %d questions...", len(benchmarks))

    retriever = FAISSRetriever.get_instance()
    samples = []
    for item in benchmarks:
        question = item["question"]
        docs = retriever.retrieve(question, k=RETRIEVAL_K)
        contexts = [d.page_content for d in docs]
        answer = generate_response(question, k=RETRIEVAL_K)
        sample_data = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if item.get("ground_truth") is not None:
            sample_data["reference"] = item["ground_truth"]
        if item.get("reference_contexts"):
            sample_data["reference_contexts"] = item["reference_contexts"]
        samples.append(SingleTurnSample(**sample_data))
        time.sleep(0.5)

    run_config = RunConfig(max_workers=1, timeout=120)
    metrics = [Faithfulness(), AnswerRelevancy()]
    results = evaluate(dataset=samples, metrics=metrics, run_config=run_config)

    output_path = RESULTS_FILE
    logger.info("Résultats : %s", results.scores)
    print("\n===============================")
    print("🏆 RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("===============================")
    for name, values in results.scores.items():
        print(f"{name}: {compute_mean(values):.3f}")
    print("===============================")


if __name__ == "__main__":
    run_ragas_evaluation()
