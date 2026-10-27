"""Génération de réponse via Mistral en s'appuyant sur le retriever FAISS."""
import logging
import os
from datetime import datetime
from functools import partial

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from api.faiss_retriever import FAISSRetriever
import ssl
import httpx
import truststore

load_dotenv()

logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

LLM = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model_name="open-mistral-7b",
    temperature=0.2,
)

_client = LLM.client

LLM.client = httpx.Client(
    base_url=_client.base_url,
    headers=dict(_client.headers),
    timeout=LLM.timeout,
    verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
)

_client.close()

def _get_localized_date() -> str:
    try:
        import locale
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
        return datetime.now().strftime("%d %B %Y")
    except locale.Error:
        return datetime.now().strftime("%d %B %Y")


AUJOURD_HUI = _get_localized_date()

PROMPT_TEMPLATE = f"""
Aujourd'hui, nous sommes le {AUJOURD_HUI}.

Tu es un assistant pour la ville de Dijon. Réponds en français.
Tu disposes d'extraits d'événements (contexte) : utilise-les pour répondre.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Ne mentionne aucun événement absent du contexte.
Ne suppose jamais de période temporelle si l'utilisateur ne la précise pas.
Si tu vois des événements à différentes dates, liste-les tous sans filtrer.

Contexte :
{{context}}

Question : {{question}}

Réponse (basée uniquement sur le contexte) :
"""

PROMPT = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def generate_response(query: str, k: int = 5) -> str:
    """Interroge le système RAG et retourne la réponse du LLM."""
    retriever = FAISSRetriever.get_instance()
    retrieve_fn = partial(retriever.retrieve, k=k)

    chain = (
        {"context": RunnableLambda(retrieve_fn) | RunnableLambda(_format_docs),"question": RunnablePassthrough(),}
        | PROMPT
        | LLM
        | StrOutputParser()
    )
    return chain.invoke(query)


def main(question: str = "Quels concerts à Dijon cette semaine ?") -> None:
    print(generate_response(question, k=5))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Quels concerts à Dijon cette semaine ?"
    main(q)
