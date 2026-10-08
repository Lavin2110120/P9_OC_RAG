# Puls-Events — Assistant RAG de recommandation d'événements culturels

POC d'un chatbot qui répond aux questions sur les événements culturels
(source : API Open Agenda) grâce à un système RAG : LangChain + Mistral + FAISS.

## Objectifs
- Démontrer la faisabilité technique d'un chatbot RAG
- Exposer le système via une API REST (FastAPI)
- Évaluer la qualité des réponses sur un jeu de test annoté

## Structure du projet
```
├── api/            # API FastAPI (endpoints /ask, /rebuild)
├── scripts/        # Collecte, nettoyage, construction de l'index
├── src/rag/        # Logique RAG (retriever, prompt, chaîne)
├── data/           # Données brutes, nettoyées et index FAISS (non versionnés)
├── tests/          # Tests unitaires et évaluation
├── docs/           # Rapport technique et présentation
├── notebooks/      # Explorations
├── .env.example    # Modèle des variables d'environnement
└── requirements.txt
```

## Installation

### Prérequis
- Python 3.11 ou 3.12
- Une clé API Mistral (https://console.mistral.ai)

### Étapes
```bash
git clone <url-du-repo>
cd puls-events-rag

python -m venv env
source env/bin/activate        # Windows : env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env           # puis renseigner les clés API
python scripts/check_env.py    # vérifie l'installation
```

## Note sur les versions
Les imports historiques (`langchain.vectorstores`, `langchain.embeddings`,
`MistralClient`) sont obsolètes dans les versions actuelles. Ce projet utilise :
- `langchain_community.vectorstores.FAISS`
- `langchain_huggingface.HuggingFaceEmbeddings`
- `mistralai.Mistral` (SDK v1)

## Utilisation
*(à compléter : construction de l'index, lancement de l'API, Docker)*