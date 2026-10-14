"""Évaluation légère du système RAG avec l'API moderne de Ragas.

Le LLM d'évaluation est construit avec l'adaptateur Instructor natif de
Ragas. Il ne faut pas le remplacer par ``LangchainLLMWrapper`` : les
métriques de la collection (Faithfulness, etc.) refusent ce wrapper legacy.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from config.settings import MISTRAL_API_KEY
except ImportError:
    from settings import MISTRAL_API_KEY

from ragas import RunConfig, evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from langchain_huggingface import HuggingFaceEmbeddings

from scripts.faiss_retriever import FAISSRetriever
from scripts.mistral_qa import generate_response

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from mistralai import Mistral
import instructor

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_FILE = ROOT_DIR / "annotated_qa.json"
RESULTS_FILE = ROOT_DIR / "ragas_evaluation_results.csv"

# Le modèle est volontairement le même que celui configuré pour la génération.
EVALUATOR_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
EVALUATION_LIMIT = 3
RETRIEVAL_K = 15
run_config = RunConfig(max_workers=1, timeout=120)

MIN_FAITHFULNESS = 0.50
MIN_ANSWER_RELEVANCY = 0.40


def load_benchmarks(path: Path) -> list[dict[str, Any]]:
    """Charge le benchmark JSON et limite l'évaluation au test léger (3 cas)."""
    with path.open("r", encoding="utf-8") as file:
        benchmarks = json.load(file)
    if not isinstance(benchmarks, list):
        raise ValueError("Le benchmark doit contenir une liste d'exemples.")
    return benchmarks[:EVALUATION_LIMIT]


def compute_mean(values: Any) -> float:
    """Calcule une moyenne en ignorant les valeurs absentes/NaN."""
    if isinstance(values, (int, float)):
        return float(values)
    if values is None:
        return 0.0
    valid = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:  # NaN != NaN
            valid.append(number)
    return sum(valid) / len(valid) if valid else 0.0


def build_evaluator_llm() -> InstructorLLM:
    """Construit le LLM évaluateur Ragas avec le SDK officiel Mistral.

    ``instructor.from_mistral`` ajoute le support des sorties structurées
    Instructor au client Mistral. ``open-mistral-7b`` ne repose pas sur le
    tool-calling / json-schema côté API, donc on utilise le mode markdown JSON,
    valide pour le provider Mistral dans instructor v2.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY est requis dans .env ou l'environnement.")

    import instructor
    from mistralai import Mistral

    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    instructor_client = instructor.from_mistral(
        mistral_client,
        mode=instructor.Mode.MD_JSON,
    )
    return InstructorLLM(
        client=instructor_client,
        model=EVALUATOR_MODEL,
        provider="mistral",
        temperature=0,
    )


def build_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """Conserve les embeddings locaux BGE déjà utilisés par le projet.

    Ragas accepte encore ce wrapper pour les métriques qui ont besoin
    d'embeddings (AnswerRelevancy). Il est indépendant du wrapper LLM legacy,
    qui n'est volontairement plus utilisé.
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation() -> None:
    if not BENCHMARKS_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {BENCHMARKS_FILE}")

    benchmarks = load_benchmarks(BENCHMARKS_FILE)
    if not benchmarks:
        raise ValueError("Benchmark vide.")

    logger.info(
        "Préparation de l'évaluation Ragas sur %d exemple(s)...", len(benchmarks)
    )

    retriever = FAISSRetriever.get_instance()
    samples: list[SingleTurnSample] = []
    has_reference_contexts = any(item.get("reference_contexts") for item in benchmarks)

    for item in benchmarks:
        question = item["question"]
        documents = retriever.retrieve(question, k=RETRIEVAL_K)
        contexts = [document.page_content for document in documents]

        # La génération réutilise le pipeline existant et le même k=15.
        answer = generate_response(question, k=RETRIEVAL_K)
        sample_data: dict[str, Any] = {
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

    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    if not has_reference_contexts:
        logger.warning("Aucun contexte de référence → métriques de contexte ignorées.")

    logger.info("Calcul des scores Ragas via %s...", EVALUATOR_MODEL)
    evaluation_dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    output_path = RESULTS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_pandas().to_csv(output_path, index=False)
    logger.info("Résultats sauvegardés dans %s", output_path)

    print("\n===============================")
    print("🏆 RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("===============================")
    for metric_name, values in results.scores.items():
        print(f"  • {metric_name}: {compute_mean(values):.3f}")
    print("===============================")

    logger.info("Vérification des seuils de qualité...")
    failed = False
    for metric_name, threshold in (
        ("faithfulness", MIN_FAITHFULNESS),
        ("answer_relevancy", MIN_ANSWER_RELEVANCY),
    ):
        average = compute_mean(results.scores.get(metric_name))
        print(f"  • {metric_name}: {average:.3f} (seuil: {threshold:.2f})")
        if average < threshold:
            logger.error("%s trop bas : %.3f < %.2f", metric_name, average, threshold)
            failed = True

    if failed:
        raise SystemExit("❌ Évaluation échouée : scores sous les seuils.")
    logger.info("✅ Tous les scores sont au-dessus des seuils.")


if __name__ == "__main__":
    run_ragas_evaluation()