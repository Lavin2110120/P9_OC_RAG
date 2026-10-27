# =============================================================================
# Puls-Events RAG — image Docker (Linux, python:3.12-slim)
# =============================================================================
# Build :
#   docker build -t puls-events-rag .
# Run   :
#   docker run -p 8000:8000 --env-file .env puls-events-rag
# =============================================================================

# ---------- Stage 1 : builder (compilation des dépendances + pré-téléchargement modèle) ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache

WORKDIR /app

# build-essential nécessaire pour certaines wheels (faiss-cpu / sentence-transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copie pyproject.toml seul pour profiter du cache Docker des layers pip
COPY pyproject.toml ./

# Installation des dépendances core (FastAPI, FAISS, Mistral, …)
RUN pip install --no-cache-dir ".[core]"


ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='${HF_HOME}')" \
    && du -sh ${HF_HOME}

# ---------- Stage 2 : runner (image finale légère) ----------
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache

WORKDIR /app

# Dépendances runtime : libgomp1 pour OpenMP (faiss / sentence-transformers),
# curl pour le HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Récupération du virtualenv depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Récupération du cache HuggingFace (modèles pré-téléchargés)
COPY --from=builder /opt/hf_cache /opt/hf_cache

# Code source de l'application (api/ config/ scripts/ ui/)
# Les .env, data/raw, tests, docs sont exclus par .dockerignore.
COPY api/      ./api/
COPY config/   ./config/
COPY scripts/  ./scripts/
COPY ui/       ./ui/

# Aucun secret (OPENAGENDA_API_KEY, MISTRAL_API_KEY) n'est copié dans l'image :
# ils doivent être passés via --env-file .env ou -e au `docker run`.

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /opt/hf_cache
USER appuser

EXPOSE 8000

# Healthcheck simple sur /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

# Lancement de l'API (cohérent avec `uvicorn api.main:app`)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
