"""Vérifie que l'environnement est correctement installé.

Usage : python scripts/check_env.py
"""
import os
import sys

from dotenv import load_dotenv


def check(label, func):
    try:
        result = func()
        print(f"✅ {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"❌ {label} — {type(e).__name__}: {e}")
        return False


def test_faiss():
    import faiss
    import numpy as np
    index = faiss.IndexFlatL2(8)
    index.add(np.random.rand(10, 8).astype("float32"))
    return f"version {faiss.__version__}, {index.ntotal} vecteurs indexés"


def test_langchain_faiss():
    from langchain_community.vectorstores import FAISS  # noqa: F401
    import langchain
    return f"langchain {langchain.__version__}"


def test_hf_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
    return "import OK"


def test_mistral_sdk():
    from mistralai import Mistral  # noqa: F401
    return "import OK"


def test_langchain_mistral():
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # noqa: F401
    return "import OK"


def test_fastapi():
    import fastapi
    return f"version {fastapi.__version__}"


def test_mistral_api():
    """Appel réel à l'API (nécessite la clé dans .env)."""
    from mistralai import Mistral
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY absente du fichier .env")
    client = Mistral(api_key=key)
    resp = client.embeddings.create(model="mistral-embed", inputs=["Bonjour"])
    return f"embedding de dimension {len(resp.data[0].embedding)}"


if __name__ == "__main__":
    load_dotenv()
    print(f"Python {sys.version.split()[0]}\n")
    results = [
        check("FAISS", test_faiss),
        check("LangChain + FAISS", test_langchain_faiss),
        check("HuggingFace Embeddings", test_hf_embeddings),
        check("SDK Mistral", test_mistral_sdk),
        check("LangChain Mistral", test_langchain_mistral),
        check("FastAPI", test_fastapi),
        check("Appel API Mistral", test_mistral_api),
    ]
    print(f"\n{sum(results)}/{len(results)} vérifications réussies")
    sys.exit(0 if all(results) else 1)