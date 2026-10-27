"""Évalue le pipeline RAG avec RAGAS (à placer dans ``tests/``).

Exécution depuis la racine du dépôt : ``python tests/evaluate_rag.py``.
L'évaluation réelle appelle le modèle Mistral configuré dans ``api.mistral_qa``.
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# === CONFIGURATION SSL (AVANT tous les imports qui utilisent HTTPS) ===
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Configuration globale de httpx pour utiliser les certificats système
import ssl
import httpx

# Créer un contexte SSL qui utilise les certificats système + certifi
_ssl_context = ssl.create_default_context()
_ssl_context.load_verify_locations(certifi.where())

# Monkey-patch httpx pour utiliser ce contexte par défaut
_original_client_init = httpx.Client.__init__
_original_async_client_init = httpx.AsyncClient.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", _ssl_context)
    _original_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", _ssl_context)
    _original_async_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

# Alternative avec truststore si disponible
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import ragas
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings


# === CHOIX DU LLM POUR L'ÉVALUATION ===
# Option 1: Mistral (si quota disponible) - décommente si besoin
# from langchain_mistralai import ChatMistralAI

# Option 2: Ollama (local, gratuit, pas de rate limit) - RECOMMANDÉ
from langchain_ollama import ChatOllama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS_FILE = ROOT / "annotated_qa.json"
RESULTS_FILE = ROOT / "ragas_evaluation_results.csv"

EVALUATION_LIMIT = 3  # On peut remettre 3 avec Ollama
RETRIEVAL_K = 5

# Délai entre les appels (moins critique avec Ollama)
RATE_LIMIT_DELAY = 0.5

from config.settings import MISTRAL_API_KEY


def create_evaluation_llm():
    """
    Crée le LLM pour l'évaluation RAGAS.
    Utilise Ollama (local) par défaut pour éviter les rate limits.
    """
    # === OPTION 1: Ollama (recommandé, gratuit, local) ===
    logger.info("Utilisation d'Ollama (modèle local) pour l'évaluation RAGAS")
    return ChatOllama(
        model="mistral",  # ou "llama3.2", "phi3", etc. selon ce qui est installé
        temperature=0,
        # num_ctx=4096,  # Augmente le contexte si besoin
    )

    # === OPTION 2: Mistral (décommente si tu as du quota) ===
    # logger.info("Utilisation de Mistral pour l'évaluation RAGAS")
    # return ChatMistralAI(
    #     api_key=MISTRAL_API_KEY,
    #     model="mistral-small-latest",
    #     temperature=0,
    #     max_retries=5,
    #     timeout=120,
    # )


def load_benchmarks(path: Path) -> list[dict[str, Any]]:
    """Charge au plus EVALUATION_LIMIT questions annotées."""
    with path.open("r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    if not isinstance(benchmarks, list):
        raise ValueError(f"Le benchmark doit être une liste JSON : {path}")
    if any(not isinstance(item, dict) for item in benchmarks):
        raise ValueError("Chaque entrée du benchmark doit être un objet JSON.")
    return benchmarks[:EVALUATION_LIMIT]


def compute_mean(values: Any) -> float | None:
    """Moyenne des scores numériques observés; None s'il n'y en a aucun."""
    if isinstance(values, (int, float)):
        number = float(values)
        return number if math.isfinite(number) else None
    if values is None or isinstance(values, (str, bytes, dict)):
        return None
    try:
        iterator: Iterable[Any] = iter(values)
    except TypeError:
        return None
    valid: list[float] = []
    for value in iterator:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            valid.append(number)
    return sum(valid) / len(valid) if valid else None


def _load_ragas_components() -> tuple[Any, list[Any], Any, bool]:
    """Importe evaluate, métriques et format de dataset selon la version RAGAS."""
    evaluate = getattr(ragas, "evaluate", None)
    if evaluate is None:
        from ragas import evaluate

    metrics_module = __import__("ragas.metrics", fromlist=["metrics"])

    # Faithfulness
    faithfulness_cls = getattr(metrics_module, "Faithfulness", None)
    if faithfulness_cls is None:
        faithfulness_cls = getattr(metrics_module, "faithfulness", None)

    # AnswerRelevancy
    relevancy_cls = getattr(metrics_module, "AnswerRelevancy", None)
    if relevancy_cls is None:
        relevancy_cls = getattr(metrics_module, "ResponseRelevancy", None)
    if relevancy_cls is None:
        relevancy_cls = getattr(metrics_module, "answer_relevancy", None)

    # ContextPrecision (NOUVEAU)
    context_precision_cls = getattr(metrics_module, "ContextPrecision", None)
    if context_precision_cls is None:
        context_precision_cls = getattr(metrics_module, "context_precision", None)
    if context_precision_cls is None:
        # Essayer l'import depuis ragas.metrics.collections (nouvelle API)
        try:
            from ragas.metrics.collections import ContextPrecision as context_precision_cls
        except ImportError:
            context_precision_cls = None

    # Vérification des métriques obligatoires
    if faithfulness_cls is None or relevancy_cls is None:
        raise ImportError("RAGAS n'expose pas les métriques Faithfulness et Answer/ResponseRelevancy.")

    # Construction de la liste des métriques
    metrics_list = [faithfulness_cls(), relevancy_cls()]
    if context_precision_cls is not None:
        metrics_list.append(context_precision_cls())
        logger.info("Métrique ContextPrecision ajoutée")
    else:
        logger.warning("ContextPrecision non disponible dans cette version de RAGAS")

    modern_dataset = getattr(ragas, "EvaluationDataset", None)
    return evaluate, metrics_list, modern_dataset, modern_dataset is not None


def _make_dataset(rows: list[dict[str, Any]], dataset_cls: Any, modern: bool) -> Any:
    """Prépare le schéma RAGAS v0.1 (datasets) ou le schéma single-turn récent."""
    if modern:
        if hasattr(dataset_cls, "from_list"):
            return dataset_cls.from_list(rows)
        sample_cls = getattr(__import__("ragas"), "SingleTurnSample", None)
        if sample_cls is None:
            raise ImportError("La version RAGAS détectée n'expose pas SingleTurnSample.")
        return dataset_cls(samples=[sample_cls(**row) for row in rows])

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Le format RAGAS historique nécessite le paquet 'datasets'.") from exc
    return Dataset.from_list(rows)


def _run_config_kwargs(evaluate: Any) -> dict[str, Any]:
    """N'ajoute RunConfig que si la signature installée le prend en charge."""
    try:
        signature = inspect.signature(evaluate)
    except (TypeError, ValueError):
        return {}
    parameters = signature.parameters
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if "run_config" not in parameters and not accepts_extra:
        return {}
    try:
        import ragas
        run_config_cls = getattr(ragas, "RunConfig", None)
        if run_config_cls is None:
            from ragas.run_config import RunConfig as run_config_cls
        config_params = inspect.signature(run_config_cls).parameters
        kwargs: dict[str, Any] = {}
        if "max_workers" in config_params:
            kwargs["max_workers"] = 1  # 1 seul worker pour la stabilité
        if "timeout" in config_params:
            kwargs["timeout"] = 600  # 10 minutes pour Ollama (plus lent)
        if "max_retries" in config_params:
            kwargs["max_retries"] = 3
        if "max_wait" in config_params:
            kwargs["max_wait"] = 60
        return {"run_config": run_config_cls(**kwargs)}
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}


def _score_rows(results: Any) -> list[dict[str, Any]]:
    """Normalise les retours EvaluationResult usuels sans inventer de score."""
    to_pandas = getattr(results, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        return frame.to_dict(orient="records")

    scores = getattr(results, "scores", results)
    if isinstance(scores, dict):
        lengths = [len(value) for value in scores.values()
                   if isinstance(value, (list, tuple))]
        row_count = max(lengths, default=1)
        rows: list[dict[str, Any]] = []
        for index in range(row_count):
            row: dict[str, Any] = {}
            for name, values in scores.items():
                row[name] = values[index] if isinstance(values, (list, tuple)) and index < len(values) else (
                    values if not isinstance(values, (list, tuple)) else None
                )
            rows.append(row)
        return rows
    if isinstance(scores, (list, tuple)):
        return [dict(row) if isinstance(row, dict) else {"score": row} for row in scores]
    return []


def _save_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Enregistre les scores réellement renvoyés par RAGAS au format CSV."""
    if not rows:
        logger.warning("RAGAS n'a fourni aucune ligne de score; aucun CSV écrit.")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_ragas_evaluation() -> None:
    """Évalue les questions réelles sans faire d'appels si les prérequis manquent."""
    # LLM pour l'évaluation (Ollama local)
    evaluation_llm = LangchainLLMWrapper(create_evaluation_llm())

    # Embeddings pour l'évaluation (local aussi)
    evaluation_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
        )
    )

    try:
        evaluate, metrics, dataset_cls, modern = _load_ragas_components()
    except ImportError as exc:
        logger.error("RAGAS ou ses métriques ne sont pas disponibles : %s", exc)
        return

    try:
        from api.faiss_retriever import FAISSRetriever
        from api.mistral_qa import generate_response
    except Exception as exc:
        logger.error("Modules RAG indisponibles : %s", exc)
        return

    if not BENCHMARKS_FILE.exists():
        logger.error("Benchmark absent : %s", BENCHMARKS_FILE)
        return

    benchmarks = load_benchmarks(BENCHMARKS_FILE)
    if not benchmarks:
        raise ValueError("Benchmark vide.")
    if any(not isinstance(item.get("question"), str) or not item["question"].strip()
           for item in benchmarks):
        raise ValueError("Chaque entrée du benchmark doit avoir un champ 'question' non vide.")

    logger.info("Évaluation de %d questions...", len(benchmarks))
    retriever = FAISSRetriever.get_instance()
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(benchmarks):
        question = item["question"]
        logger.info("Question %d/%d : %s", idx + 1, len(benchmarks), question[:80])

        docs = retriever.retrieve(question, k=RETRIEVAL_K)
        contexts = [doc.page_content for doc in docs]
        answer = generate_response(question, k=RETRIEVAL_K)
        if not isinstance(answer, str):
            raise TypeError("generate_response doit retourner une chaîne.")

        # Construction de la ligne selon la version de RAGAS
        if modern:
            row = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
            # reference = ground_truth de la réponse
            if item.get("ground_truth") is not None:
                row["reference"] = item["ground_truth"]
            # reference_contexts = contextes idéaux pour ContextPrecision
            if item.get("reference_contexts"):
                row["reference_contexts"] = item["reference_contexts"]
        else:
            row = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
            if item.get("ground_truth") is not None:
                row["ground_truth"] = item["ground_truth"]
            if item.get("reference_contexts"):
                row["reference_contexts"] = item["reference_contexts"]

        rows.append(row)

        if idx < len(benchmarks) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    dataset = _make_dataset(rows, dataset_cls, modern)

    logger.info("Lancement de l'évaluation RAGAS avec Ollama (peut prendre 5-15 minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluation_llm,
        embeddings=evaluation_embeddings,
        raise_exceptions=False,
        **_run_config_kwargs(evaluate),
    )
    score_rows = _score_rows(results)
    _save_results(score_rows, RESULTS_FILE)

    print("\n===============================")
    print("RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("===============================")

    # Définit score_source à partir de results
    score_source = getattr(results, "scores", results)

    # Liste complète des métriques à afficher
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]

    for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric_name in score_source:
            mean = compute_mean(score_source[metric_name])
            if mean is not None:
                print(f"{metric_name}: {mean:.3f}")
            else:
                print(f"{metric_name}: échec/indisponible")
        else:
            print(f"{metric_name}: non disponible")

    print("===============================")

    # Affichage détaillé par question si disponible
    if score_rows:
        print("\n--- DÉTAIL PAR QUESTION ---")
        for i, row in enumerate(score_rows, 1):
            print(f"\nQuestion {i}:")
            for metric_name in metric_names:
                value = row.get(metric_name, "N/A")
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.3f}")
                else:
                    print(f"  {metric_name}: {value}")
        print("===============================")

    if score_rows:
        logger.info("Résultats enregistrés : %s", RESULTS_FILE)


if __name__ == "__main__":
    run_ragas_evaluation()