import pytest
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

from config import settings

INDEX_DIR = settings.ROOT_DIR / "data" / "faiss_index"


@pytest.fixture(scope="module")
def store():
    if not INDEX_DIR.exists():
        pytest.skip("Index non construit : lancer build_index.py")
    emb = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=settings.MISTRAL_API_KEY)
    return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)


def test_index_non_vide(store):
    assert store.index.ntotal > 0


def test_dimension_embedding(store):
    assert store.index.d == 1024  # mistral-embed


def test_recherche_retourne_des_resultats(store):
    res = store.similarity_search("concert à Dijon", k=3)
    assert len(res) == 3
    assert all("title" in d.metadata for d in res)