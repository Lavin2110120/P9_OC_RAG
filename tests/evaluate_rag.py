# # """Évaluation Ragas avec un LLM local (Ollama)."""
# # import json
# # import logging
# # import os
# # import time
# # from pathlib import Path

# # from datasets import Dataset
# # from langchain_community.llms.ollama import Ollama
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from ragas import evaluate, RunConfig
# # from ragas.metrics import answer_relevancy, faithfulness

# # run_config = RunConfig(max_workers=1, timeout=None)

# # from scripts.faiss_retriever import FAISSRetriever
# # from scripts.mistral_qa import generate_response

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # ROOT_DIR = Path(__file__).resolve().parents[1]
# # BENCHMARKS_FILE = ROOT_DIR / "annotated_qa.json"

# # # --- Configuration du LLM local ---
# # # Modèle Ollama à utiliser pour l'évaluation Ragas. Seulement pour l'évaluation!
# # RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "qwen2.5:0.5b")
# # OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# # def load_benchmarks(path: Path) -> list[dict]:
# #     """Charge le fichier de benchmark annoté."""
# #     with path.open("r", encoding="utf-8") as f:
# #         return json.load(f)


# # def run_ragas_evaluation() -> None:
# #     """Prépare les exemples puis lance l'évaluation Ragas.

# #     L'évaluateur utilise désormais un LLM local via Ollama.
# #     """
# #     if not BENCHMARKS_FILE.exists():
# #         logger.error("Fichier de benchmark introuvable : %s", BENCHMARKS_FILE)
# #         return

# #     benchmarks = load_benchmarks(BENCHMARKS_FILE)

# #     retriever = FAISSRetriever()
# #     questions = []
# #     answers = []
# #     contexts = []
# #     ground_truths = []

# #     logger.info("Préparation de l'évaluation Ragas sur %d exemple(s)...", len(benchmarks))

# #     for item in benchmarks:
# #         q = item["question"]
# #         gt = item["ground_truth"]

# #         # Récupération des passages depuis l'index FAISS
# #         docs = retriever.retrieve(q, k=3)
# #         context_chunks = [d.page_content for d in docs]

# #         # Génération de la réponse à évaluer
# #         # Si besoin, remplace cette étape par une génération locale.
# #         ans = generate_response(q)

# #         questions.append(q)
# #         answers.append(ans)
# #         contexts.append(context_chunks)
# #         ground_truths.append(gt)

# #         # Petit délai volontaire pour stabiliser les enchaînements
# #         time.sleep(1)

# #     data = {
# #         "user_input": questions,
# #         "response": answers,
# #         "retrieved_contexts": contexts,
# #         "reference": ground_truths,
# #     }
# #     dataset = Dataset.from_dict(data)

# #     # --- Évaluateur local ---
# #     evaluator_llm = Ollama(
# #     model=RAGAS_LLM_MODEL,
# #     base_url=OLLAMA_BASE_URL,
# #     temperature=0,
# #     timeout=600,)
# #     evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# #     logger.info(
# #         "Calcul des scores Ragas en cours avec le modèle local '%s'...",
# #         RAGAS_LLM_MODEL,
# #     )

# #     results = evaluate(
# #         dataset=dataset,
# #         metrics=[faithfulness, answer_relevancy],
# #         llm=evaluator_llm,
# #         embeddings=evaluator_embeddings,
# #         run_config=run_config,
# #     )

# #     print("\n===============================")
# #     print("🏆 RÉSULTATS DE L'ÉVALUATION RAGAS")
# #     print("===============================")
# #     print(results)

# #     df_results = results.to_pandas()
# #     df_results.to_csv("ragas_evaluation_results.csv", index=False)
# #     print("✅ Résultats sauvegardés dans ragas_evaluation_results.csv")


# # if __name__ == "__main__":
# #     run_ragas_evaluation()

# """
# Évaluation Ragas du système RAG.

# Ce script :
# - Charge le benchmark annoté (questions / références).
# - Récupère les contextes via le retriever FAISS.
# - Génère les réponses via l'API Mistral.
# - Évalue les réponses avec Ragas en utilisant un LLM local (Ollama).
# - Calcule plusieurs métriques :
#     * faithfulness (fidélité de la réponse au contexte)
#     * answer_relevancy (pertinence de la réponse)
#     * context_precision (précision du retriever)
#     * context_recall (rappel du retriever)
#     * answer_correctness (exactitude par rapport à la référence)
# - Vérifie les seuils de qualité et échoue si un score est trop bas.
# - Sauvegarde les résultats dans ragas_evaluation_results.csv.

# Le LLM de génération reste l'API Mistral, comme demandé dans le projet.
# Le LLM d'évaluation peut être modifié via la variable d'environnement
# RAGAS_LLM_MODEL (par défaut : phi3:mini).
# """
# import json
# import logging
# import os
# import time
# from pathlib import Path

# from datasets import Dataset
# from langchain_community.llms.ollama import Ollama
# from langchain_huggingface import HuggingFaceEmbeddings
# from ragas import evaluate, RunConfig
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )
# # answer_correctness peut nécessiter une configuration supplémentaire
# try:
#     from ragas.metrics import answer_correctness
#     HAS_ANSWER_CORRECTNESS = True
# except ImportError:
#     HAS_ANSWER_CORRECTNESS = False
#     logging.warning("answer_correctness non disponible, métrique ignorée.")

# from scripts.faiss_retriever import FAISSRetriever
# from scripts.mistral_qa import generate_response

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# ROOT_DIR = Path(__file__).resolve().parents[1]
# BENCHMARKS_FILE = ROOT_DIR / "annotated_qa.json"

# # --- Configuration du LLM d'évaluation ---
# # Modèle Ollama utilisé uniquement pour l'évaluation Ragas
# RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "phi3:mini")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# # Seuils minimaux pour validation (à ajuster selon le projet)
# MIN_FAITHFULNESS = 0.70
# MIN_ANSWER_RELEVANCY = 0.70
# MIN_CONTEXT_PRECISION = 0.60
# MIN_CONTEXT_RECALL = 0.60
# MIN_ANSWER_CORRECTNESS = 0.60

# run_config = RunConfig(max_workers=1, timeout=None)


# def load_benchmarks(path: Path) -> list[dict]:
#     """Charge le fichier de benchmark annoté."""
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def run_ragas_evaluation() -> None:
#     """Prépare les exemples puis lance l'évaluation Ragas."""
#     if not BENCHMARKS_FILE.exists():
#         logger.error("Fichier de benchmark introuvable : %s", BENCHMARKS_FILE)
#         raise FileNotFoundError(f"Fichier de benchmark introuvable : {BENCHMARKS_FILE}")

#     benchmarks = load_benchmarks(BENCHMARKS_FILE)
#     if not benchmarks:
#         logger.error("Le fichier de benchmark est vide.")
#         raise ValueError("Benchmark vide, impossible d'évaluer.")

#     logger.info("Préparation de l'évaluation Ragas sur %d exemple(s)...", len(benchmarks))

#     retriever = FAISSRetriever()
#     questions, answers, contexts, ground_truths = [], [], [], []

#     for item in benchmarks:
#         q = item["question"]
#         gt = item["ground_truth"]

#         # Récupération des passages depuis l'index FAISS
#         docs = retriever.retrieve(q, k=3)
#         context_chunks = [d.page_content for d in docs]

#         # Génération de la réponse à évaluer
#         ans = generate_response(q)

#         questions.append(q)
#         answers.append(ans)
#         contexts.append(context_chunks)
#         ground_truths.append(gt)

#         # Petit délai volontaire pour stabiliser les enchaînements
#         time.sleep(1)

#     data = {
#         "user_input": questions,
#         "response": answers,
#         "retrieved_contexts": contexts,
#         "reference": ground_truths,
#     }
#     dataset = Dataset.from_dict(data)

#     # --- Évaluateur local ---
#     evaluator_llm = Ollama(
#         model=RAGAS_LLM_MODEL,
#         base_url=OLLAMA_BASE_URL,
#         # temperature=0,
#         timeout=600,
#     )
#     evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#     logger.info(
#         "Calcul des scores Ragas en cours avec le modèle local '%s'...",
#         RAGAS_LLM_MODEL,
#     )

#     # Choix des métriques selon disponibilité
#     metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
#     if HAS_ANSWER_CORRECTNESS:
#         metrics.append(answer_correctness)

#     results = evaluate(
#         dataset=dataset,
#         metrics=metrics,
#         llm=evaluator_llm,
#         embeddings=evaluator_embeddings,
#         run_config=run_config,
#     )

#     # Affichage
#     print("\n===============================")
#     print("🏆 RÉSULTATS DE L'ÉVALUATION RAGAS")
#     print("===============================")
#     scores = results
#     print(scores)

#     # Sauvegarde CSV
#     df_results = results.to_pandas()
#     df_results.to_csv("ragas_evaluation_results.csv", index=False)
#     print("✅ Résultats sauvegardés dans ragas_evaluation_results.csv")

#     # --- Vérification des seuils ---
#     logger.info("Vérification des seuils de qualité...")
#     failed = False

#     if scores["faithfulness"] < MIN_FAITHFULNESS:
#         logger.error("faithfulness trop bas : %.3f < %.2f", scores["faithfulness"], MIN_FAITHFULNESS)
#         failed = True
#     if scores["answer_relevancy"] < MIN_ANSWER_RELEVANCY:
#         logger.error("answer_relevancy trop bas : %.3f < %.2f", scores["answer_relevancy"], MIN_ANSWER_RELEVANCY)
#         failed = True
#     if scores["context_precision"] < MIN_CONTEXT_PRECISION:
#         logger.error("context_precision trop bas : %.3f < %.2f", scores["context_precision"], MIN_CONTEXT_PRECISION)
#         failed = True
#     if scores["context_recall"] < MIN_CONTEXT_RECALL:
#         logger.error("context_recall trop bas : %.3f < %.2f", scores["context_recall"], MIN_CONTEXT_RECALL)
#         failed = True
#     if HAS_ANSWER_CORRECTNESS and scores["answer_correctness"] < MIN_ANSWER_CORRECTNESS:
#         logger.error("answer_correctness trop bas : %.3f < %.2f", scores["answer_correctness"], MIN_ANSWER_CORRECTNESS)
#         failed = True

#     if failed:
#         raise SystemExit("❌ Évaluation échouée : certains scores sont en dessous des seuils.")
#     else:
#         logger.info("✅ Tous les scores sont au-dessus des seuils.")


# if __name__ == "__main__":
#     run_ragas_evaluation()

"""
Évaluation Ragas du système RAG.

Ce script :
- Charge le benchmark annoté (questions / références).
- Récupère les contextes via le retriever FAISS.
- Génère les réponses via la chaîne complète (API Mistral + retriever).
- Évalue les réponses avec Ragas en utilisant l'API Mistral.
- Calcule plusieurs métriques :
    * faithfulness (fidélité de la réponse au contexte)
    * answer_relevancy (pertinence de la réponse)
    * context_precision (précision du retriever)
    * context_recall (rappel du retriever)
- Vérifie les seuils de qualité et échoue si un score est trop bas.
- Sauvegarde les résultats dans ragas_evaluation_results.csv.

Le LLM de génération et d'évaluation utilisent l'API Mistral.
"""
import json
import logging
import os
import time
from pathlib import Path

# Désactiver les logs HTTP trop verbeux
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from scripts.faiss_retriever import FAISSRetriever
from scripts.mistral_qa import generate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_FILE = ROOT_DIR / "annotated_qa.json"

# --- Configuration de l'évaluateur ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY est requis dans .env ou variable d'environnement.")

# Seuils minimaux pour validation
MIN_FAITHFULNESS = 0.50
MIN_ANSWER_RELEVANCY = 0.50
MIN_CONTEXT_PRECISION = 0.60
MIN_CONTEXT_RECALL = 0.60

# Timeout Ragas désactivé (les appels API sont rapides)
run_config = RunConfig(max_workers=1, timeout=None)


def load_benchmarks(path: Path) -> list[dict]:
    """Charge le fichier de benchmark annoté."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_ragas_evaluation() -> None:
    """Prépare les exemples puis lance l'évaluation Ragas."""
    if not BENCHMARKS_FILE.exists():
        logger.error("Fichier de benchmark introuvable : %s", BENCHMARKS_FILE)
        raise FileNotFoundError(
            f"Fichier de benchmark introuvable : {BENCHMARKS_FILE}"
        )

    benchmarks = load_benchmarks(BENCHMARKS_FILE)
    if not benchmarks:
        logger.error("Le fichier de benchmark est vide.")
        raise ValueError("Benchmark vide, impossible d'évaluer.")

    logger.info(
        "Préparation de l'évaluation Ragas sur %d exemple(s)...",
        len(benchmarks)
    )

    retriever = FAISSRetriever()
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in benchmarks:
        q = item["question"]
        gt = item["ground_truth"]

        docs = retriever.retrieve(q, k=3)
        context_chunks = [d.page_content for d in docs]

        ans = generate_response(q)

        questions.append(q)
        answers.append(ans)
        contexts.append(context_chunks)
        ground_truths.append(gt)

        time.sleep(0.5)  # Léger délai entre appels API

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # --- Évaluateur via API Mistral ---
    evaluator_llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model_name="mistral-tiny",
        temperature=0,
    )

    # Un seul chargement des embeddings
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    logger.info("Calcul des scores Ragas en cours via l'API Mistral...")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    # Affichage des scores
    print("\n===============================")
    print("🏆 RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("===============================")
    scores = results
    print(scores)

    # Sauvegarde CSV
    df_results = results.to_pandas()
    df_results.to_csv("ragas_evaluation_results.csv", index=False)
    print("✅ Résultats sauvegardés dans ragas_evaluation_results.csv")

    # --- Vérification des seuils ---
    logger.info("Vérification des seuils de qualité...")
    failed = False

    checks = [
        ("faithfulness", MIN_FAITHFULNESS),
        ("answer_relevancy", MIN_ANSWER_RELEVANCY),
        ("context_precision", MIN_CONTEXT_PRECISION),
        ("context_recall", MIN_CONTEXT_RECALL),
    ]

    for metric_name, threshold in checks:
        if scores[metric_name] < threshold:
            logger.error(
                "%s trop bas : %.3f < %.2f",
                metric_name,
                scores[metric_name],
                threshold,
            )
            failed = True

    if failed:
        raise SystemExit(
            "❌ Évaluation échouée : certains scores sont en dessous des seuils."
        )
    else:
        logger.info("✅ Tous les scores sont au-dessus des seuils.")


if __name__ == "__main__":
    run_ragas_evaluation()