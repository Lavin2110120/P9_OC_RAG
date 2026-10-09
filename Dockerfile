# --- Base Image ---
FROM python:3.12-slim

# --- Métadonnées ---
LABEL description="Conteneur Docker pour le système RAG avec FAISS et Mistral"

# --- Variables d'environnement ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# --- Répertoire de travail ---
WORKDIR /app

# --- Copier les fichiers nécessaires ---
COPY pyproject.toml .
COPY .env .
COPY . .

# --- Installer les dépendances ---
RUN pip install --no-cache-dir -r pyproject.toml

# --- Installer FAISS (via pip ou compilation) ---
# Option 1 : Utiliser faiss-cpu (recommandé pour Docker)
RUN pip install faiss-cpu

# --- Construire l'index FAISS au démarrage (si nécessaire) ---
# Cette étape est optionnelle : vous pouvez construire l'index à la volée ou le pré-construire.
# Ici, on suppose que l'index est déjà construit et copié dans le conteneur.
COPY data/vector_store/ ./data/vector_store/

# --- Exposer le port de l'API ---
EXPOSE 8000

# --- Commande de démarrage ---
# 1. Attendre que les dépendances (comme Mistral) soient prêtes (si nécessaire).
# 2. Lancer l'API FastAPI.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]