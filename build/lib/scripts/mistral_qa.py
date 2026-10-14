"""
Chaîne RAG complète avec Mistral, prompt strict et retriever optimisé.
"""
import logging
import os
from datetime import datetime
from functools import partial

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from scripts.faiss_retriever import FAISSRetriever

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Clé API Mistral
# --------------------------------------------------------------------------- #
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY est requis dans .env ou l'environnement.")

# --------------------------------------------------------------------------- #
# Modèle de langage (réutilisé)
# --------------------------------------------------------------------------- #
LLM = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model_name="open-mistral-7b",
    temperature=0.1,          # très basse pour rester fidèle au contexte
)

# --------------------------------------------------------------------------- #
# Date du jour (localisée si possible)
# --------------------------------------------------------------------------- #
def _get_localized_date() -> str:
    """Retourne la date du jour en français, ou en anglais par fallback."""
    try:
        import locale
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")  # Linux / macOS
    except locale.Error:
        pass   # en cas d'échec, on garde la locale par défaut
    return datetime.now().strftime("%d %B %Y")

AUJOURD_HUI = _get_localized_date()

# --------------------------------------------------------------------------- #
# Prompt strict (hallucinations minimisées)
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = f"""
Tu es un assistant spécialisé dans la recommandation d'événements culturels.
Aujourd'hui, nous sommes le {AUJOURD_HUI}.

RÈGLE IMPÉRATIVE : réponds UNIQUEMENT à partir du contexte ci-dessous.
Chaque information que tu donnes DOIT être directement tirée du contexte.
Si le contexte ne contient pas la réponse, dis explicitement :
"Je ne dispose pas de cette information dans les événements disponibles."
Ne mentionne aucun événement absent du contexte.
Pour chaque fait important, cite entre parenthèses un extrait du contexte.

Contexte :
{{context}}

Question : {{question}}

Réponse (basée uniquement sur le contexte) :
"""

PROMPT = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# --------------------------------------------------------------------------- #
# Fonction de génération principale
# --------------------------------------------------------------------------- #
def generate_response(query: str, k: int = 15) -> str:
    """
    Interroge le système RAG et retourne la réponse.

    Args:
        query: Question de l'utilisateur.
        k: Nombre de documents à récupérer (défaut 15, modulable).

    Returns:
        Réponse textuelle générée par le LLM.
    """
    # Récupère le retriever singleton (pas de rechargement inutile)
    retriever = FAISSRetriever.get_instance()

    # Crée une fonction de retrieval avec la valeur de k souhaitée
    retrieve_fn = partial(retriever.retrieve, k=k)

    # Construit la chaîne pour cet appel
    chain = (
        {"context": retrieve_fn, "question": RunnablePassthrough()}
        | PROMPT
        | LLM
        | StrOutputParser()
    )

    logger.info("Génération de réponse pour : '%s' (k=%d)", query, k)
    response = chain.invoke(query)
    logger.debug("Réponse obtenue : %s", response[:200])
    return response


# --------------------------------------------------------------------------- #
# Fonction principale (compatibilité avec les tests)
# --------------------------------------------------------------------------- #
def main():
    """Point d'entrée principal."""
    question = "Quels sont les prochains concerts de rap ou pop à La Vapeur ?"
    print(generate_response(question, k=5))


# --------------------------------------------------------------------------- #
# Exécution manuelle
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()