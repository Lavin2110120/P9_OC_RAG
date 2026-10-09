# --- Base ---
FROM python:3.10-slim

# --- Variables d'environnement ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Dépendances système ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Répertoire de travail ---
WORKDIR /app

# --- Installation des dépendances Python ---
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copie du code ---
COPY . .

# --- Ports ---
EXPOSE 8000

# --- Commande par défaut ---
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]