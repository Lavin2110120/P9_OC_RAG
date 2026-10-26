# =============================================================================
# Puls-Events RAG — Makefile (Linux/macOS, GNU Make)
# Sur Windows / PowerShell, utilisez les commandes équivalentes listées en
# commentaire sous chaque cible, ou bien `make` via WSL / Git-Bash.
# =============================================================================

.PHONY: help install install-dev install-ui install-all fetch clean-data vectorize pipeline api ui test evaluate docker-build docker-run docker-shell clean all

# ---------- Aide ----------
help:
	@echo "Cibles disponibles :"
	@echo "  install        : pip install -e .[core]"
	@echo "  install-dev    : pip install -e .[core,dev]"
	@echo "  install-ui     : pip install -e .[ui]"
	@echo "  fetch          : python -m scripts.fetch_events"
	@echo "  clean-data     : python -m scripts.clean_events"
	@echo "  vectorize      : python -m scripts.vectorize_events"
	@echo "  pipeline       : fetch + clean-data + vectorize"
	@echo "  api            : uvicorn api.main:app --reload"
	@echo "  ui             : streamlit run ui/app.py"
	@echo "  test           : pytest tests/ -v"
	@echo "  evaluate       : python -m scripts.evaluate_rag"
	@echo "  docker-build   : docker build -t puls-events-rag ."
	@echo "  docker-run     : docker run -p 8000:8000 --env-file .env puls-events-rag"
	@echo "  clean          : supprime caches Python + __pycache__ + .pytest_cache"

# ---------- Installation ----------
# PS> : pip install -e ".[core]"
install:
	pip install --upgrade pip
	pip install -e ".[core]"

# PS> : pip install -e ".[core,dev]"
install-dev: install
	pip install -e ".[dev]"

# PS> : pip install -e ".[ui]"
install-ui: install
	pip install -e ".[ui]"

# PS> : pip install -e ".[core,dev,ui]"
install-all: install-dev install-ui

# ---------- Pipeline de données ----------
# PS> : python -m scripts.fetch_events
fetch:
	python -m scripts.fetch_events

# PS> : python -m scripts.clean_events
clean-data:
	python -m scripts.clean_events

# PS> : python -m scripts.vectorize_events
vectorize:
	python -m scripts.vectorize_events

# PS> : python -m scripts.fetch_events ; python -m scripts.clean_events ; python -m scripts.vectorize_events
pipeline: fetch clean-data vectorize

# ---------- Lancement de l'API et de l'UI ----------
# PS> : uvicorn api.main:app --reload
api:
	uvicorn api.main:app --reload

# Alias historique
run: api

# PS> : streamlit run ui/app.py
# (depuis la racine du projet)
ui:
	streamlit run ui/app.py

# Alias historique
streamlit: ui

# ---------- Tests & évaluation ----------
# PS> : pytest tests/ -v
test:
	pytest tests/ -v

# PS> : python -m scripts.evaluate_rag
evaluate:
	python -m scripts.evaluate_rag

# ---------- Docker ----------
# PS> : docker build -t puls-events-rag .
docker-build:
	docker build -t puls-events-rag .

# PS> : docker run -p 8000:8000 --env-file .env puls-events-rag
# Monte data/ en volume pour persister l'index FAISS entre les rebuilds.
docker-run:
	docker run -p 8000:8000 --env-file .env \
		-v $(PWD)/data:/app/data \
		puls-events-rag

# Ouvre un shell dans le conteneur pour debug
docker-shell:
	docker run --rm -it --env-file .env \
		-v $(PWD)/data:/app/data \
		--entrypoint /bin/bash \
		puls-events-rag

# ---------- Nettoyage ----------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf data/vector_store/*
	rm -rf *.egg-info build dist

# ---------- Cible par défaut ----------
all: pipeline test evaluate
