# ============================
# Stage 1 : builder
# ============================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer uniquement les outils de compilation nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un environnement virtuel (sera copié dans le runner)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier d'abord les fichiers de définition des dépendances (pour le cache)
COPY pyproject.toml .

# Installer les dépendances du projet (groupe core)
RUN pip install --no-cache-dir ".[core]"

# ============================
# Stage 2 : runner
# ============================
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copier l'environnement virtuel complet depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier le code source de l'application
COPY . .

# Optionnel : installer des bibliothèques runtime légères si nécessaire
# RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]