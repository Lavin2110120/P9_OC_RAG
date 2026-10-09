# 🎭 Puls-Events — Système RAG de recommandation d'événements culturels

**POC d'un assistant conversationnel basé sur RAG (Retrieval-Augmented Generation) pour répondre aux questions sur les événements culturels (source : [Open Agenda](https://openagenda.com/)).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1f4e79?style=flat-square&logo=facebook)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/)

---

## 🎯 **Objectifs**
✅ **Système RAG fonctionnel** avec LangChain, Mistral (ou BGE) et FAISS.
✅ **API REST** (FastAPI) pour interagir avec le système.
✅ **Index vectoriel** optimisé pour la recherche sémantique.
✅ **Dockerisé** pour un déploiement facile.

---

## 📁 **Structure du projet**
```bash
puls-events-rag/
├── api/                  # API FastAPI
│   ├── main.py           # Endpoint `/search`
│   └── __init__.py
├── scripts/              # Scripts de traitement des données
│   ├── build_faiss_index.py  # Construction de l'index FAISS
│   ├── chunks_events.py  # Découpage des événements en chunks
│   └── verif_vectorize_events.py
├── data/                 # Données (non versionnées)
│   ├── raw/              # Données brutes (Open Agenda)
│   ├── processed/        # Données nettoyées et chunkées
│   └── vector_store/     # Index FAISS + métadonnées
│       ├── faiss_index.bin
│       └── metadata.parquet
├── tests/                # Tests unitaires
│   ├── test_search.py    # Tests de l'API
│   └── conftest.py
├── docs/                 # Documentation
│   └── rapport_technique.md
├── Dockerfile            # Conteneurisation
├── requirements.txt      # Dépendances Python
├── .env.example          # Variables d'environnement
└── README.md             # Ce fichier
🚀 Installation
✅ Prérequis

    Python 3.10 ou 3.11 (recommandé : 3.10 pour la compatibilité avec FAISS).
    Docker (optionnel, pour le déploiement).
    Git (pour cloner le dépôt).

📥 Étapes d'installation
1. Cloner le dépôt

git clone https://github.com/votre-utilisateur/puls-events-rag.git
cd puls-events-rag

2. Créer un environnement virtuel

python -m venv env
source env/bin/activate  # Linux/Mac
# OU
.\env\Scripts\activate   # Windows

3. Installer les dépendances

pip install --upgrade pip
pip install -r requirements.txt

4. Préparer les données

    Télécharger les données Open Agenda (ex: events.parquet) et les placer dans data/raw/.
    Exécuter les scripts de traitement :

    # Découper les événements en chunks
    python -m scripts.chunks_events

    # Construire l'index FAISS
    python -m scripts.build_faiss_index

    →
    Résultat : data/vector_store/faiss_index.bin et metadata.parquet sont générés.

🧪 Tester le système
1. Tester l'index FAISS en local

python -m scripts.build_faiss_index

→ Sortie attendue :

🔍 Requête: 'concert à La Vapeur'
1. Thylacine (le 2027-02-18) - Distance: 0.4131
2. Sinclair, Aïssa Mallouk (le 2026-11-06) - Distance: 0.4488

2. Lancer l'API FastAPI

uvicorn api.main:app --reload

→ L'API est disponible sur : http://localhost:8000
→ Documentation Swagger : http://localhost:8000/docs
Exemple de requête avec curl

curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "concert à La Vapeur", "k": 3}'

→ Réponse attendue :

{
  "query": "concert à La Vapeur",
  "results": [
    {
      "title": "Thylacine",
      "date": "2027-02-18",
      "location": "La Vapeur",
      "distance": 0.4131
    },
    {
      "title": "Sinclair, Aïssa Mallouk",
      "date": "2026-11-06",
      "location": "La Vapeur",
      "distance": 0.4488
    }
  ],
  "count": 2
}

3. Exécuter les tests unitaires

pytest tests/

→ Sortie attendue :

test_search.py::test_search_success PASSED
test_search.py::test_search_invalid_k PASSED
test_search.py::test_search_empty_query PASSED

🐳 Déploiement avec Docker
1. Construire l'image

docker build -t puls-events-rag .

2. Lancer le conteneur

docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \  # Monte le dossier data pour persister l'index
  puls-events-rag

→ L'API est disponible sur : http://localhost:8000
3. Utiliser Docker Compose (optionnel)

Créez un fichier docker-compose.yml :

version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1

Puis lancez :

docker-compose up

📊 Architecture technique
Composant 	Technologie 	Rôle
Backend 	FastAPI 	API REST pour les requêtes.
Embeddings 	BGE (BAAI/bge-small-en-v1.5) 	Vectorisation des textes.
Index vectoriel 	FAISS (IndexFlatL2) 	Recherche rapide par similarité.
Données 	Parquet (Pandas) 	Stockage des métadonnées.
Conteneurisation 	Docker 	Déploiement portable.
🔄 Workflow RAG

    Requête utilisateur → "concert à La Vapeur"
    Vectorisation → Embedding de la requête avec BGE.
    Recherche FAISS → Trouver les 3 événements les plus proches.
    Réponse → Retourner les titres, dates et lieux.

📈 Évaluation et métriques
Jeu de tests annotés

Un fichier tests/test_rag_quality.py est prévu pour évaluer la qualité des réponses :

    Précision@k : % de réponses pertinentes dans le top-k.
    Rappel : % d'événements pertinents retrouvés.
    F1-score : Moyenne harmonique de précision et rappel.

Exemple de test annoté :

# tests/test_rag_quality.py
TEST_CASES = [
    {
        "query": "concert à La Vapeur",
        "expected_titles": ["Thylacine", "Sinclair"],
        "k": 3
    },
    {
        "query": "exposition Dijon",
        "expected_titles": ["Expo Photo", "Salon du Livre"],
        "k": 2
    }
]

🛠 Configuration
Variables d'environnement

Créez un fichier .env à la racine du projet :

# .env
HF_TOKEN=your_huggingface_token  # Optionnel (pour éviter les warnings)
LOG_LEVEL=INFO

Fichier requirements.txt

fastapi>=0.104.0
uvicorn>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
pytest>=7.4.0
httpx>=0.25.0

📚 Rapport technique

Un rapport détaillé est disponible dans docs/rapport_technique.md, incluant :

    Choix technologiques (pourquoi FAISS ? pourquoi BGE ?).
    Benchmark des modèles d'embedding testés.
    Pistes d'amélioration :
        Utiliser Mistral-7B pour la génération (au lieu de BGE seul).
        Ajouter un système de cache pour les requêtes fréquentes.
        Optimiser l'index FAISS pour de plus grands datasets (IVF, PQ, etc.).
        Ajouter un frontend (Streamlit, React).

🤝 Contribuer

    Forker le dépôt.
    Créer une branche (git checkout -b feature/ma-fonctionnalité).
    Commiter vos changements (git commit -m "Ajout de X").
    Pousser (git push origin feature/ma-fonctionnalité).
    Ouvrir une Pull Request.

📜 Licence

Ce projet est sous licence MIT. Voir LICENSE pour plus de détails.