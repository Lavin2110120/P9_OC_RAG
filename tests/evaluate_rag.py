from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)
from api.rag_system import RAGSystem
import pandas as pd

# --- Préparation des données ---
def prepare_test_data():
    """Prépare un jeu de test avec des questions/réponses annotées."""
    test_cases = [
        {
            "question": "Quels sont les concerts à La Vapeur en octobre 2026 ?",
            "answer": "Les concerts à La Vapeur en octobre 2026 incluent Ghinzu le 16 octobre à 18h30.",
            "context": ["Concert de Ghinzu le 16 octobre 2026 à La Vapeur (Rock)."],
        },
        {
            "question": "Y a-t-il des événements de jazz à Dijon ?",
            "answer": "Oui, Sam Sauvage donnera un concert de jazz le 28 novembre 2026 à La Vapeur.",
            "context": ["Sam Sauvage, jazz, le 28 novembre 2026 à La Vapeur."],
        },
    ]
    return Dataset.from_pandas(pd.DataFrame(test_cases))

# --- Évaluation ---
def evaluate_rag_system():
    """Évalue le système RAG avec Ragas."""
    rag_system = RAGSystem()
    test_data = prepare_test_data()

    # Génération des réponses par le système RAG
    questions = test_data["question"]
    answers = []
    contexts = []

    for question in questions:
        response = rag_system.generate_response(question)
        answers.append(response)
        contexts.append(rag_system.retrieve(question, k=3))

    # Création du dataset pour Ragas
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": test_data["answer"],
    })

    # Évaluation
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_relevancy, context_recall],
    )
    return result

if __name__ == "__main__":
    evaluation_result = evaluate_rag_system()
    print(evaluation_result)