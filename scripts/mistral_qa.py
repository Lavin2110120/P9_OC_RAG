from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from scripts.faiss_retriever import FAISSRetriever
from dotenv import load_dotenv
import os


# Chargez .env
load_dotenv()

# Récupérez la clé API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY est requis dans .env.")

# Initialisation du retriever
retriever = FAISSRetriever().get_retriever()

# Modèle Mistral
model = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model_name="mistral-tiny",  # ou "mistral-small", "mistral-medium"
    temperature=0.7,
)

# Template de prompt pour générer des réponses augmentées
template = """
Tu es un assistant spécialisé dans la recommandation d'événements culturels.
Utilise les informations suivantes pour répondre à la question de l'utilisateur.
Si les informations ne sont pas suffisantes, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse : """
prompt = ChatPromptTemplate.from_template(template)

# Chaîne de traitement : Recherche + Génération
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

def generate_response(query: str) -> str:
    """Génère une réponse augmentée pour une requête utilisateur."""
    return chain.invoke(query)