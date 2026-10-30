# =============================================================================
# Puls-Events RAG — Makefile (Linux/macOS, GNU Make)
# Sur Windows / PowerShell, utilisez les commandes équivalentes listées en
# commentaire sous chaque cible, ou bien `make` via WSL / Git-Bash.
# =============================================================================

.PHONY: help install install-dev install-ui install-all fetch clean-data vectorize pipeline api ui test evaluate docker-build docker-run docker-shell clean purge-data

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
	@echo "  clean          : supprime caches Python + .pytest_cache + .coverage + *.csv"
	@echo "  purge-data     : supprime le dossier data/ (refetch complet nécessaire)"

# ---------- Installation ----------
# PS> : pip install -e ".[core]"
install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[core]"

# PS> : pip install -e ".[core,dev]"
install-dev: install
	python -m pip install -e ".[dev]"

# PS> : pip install -e ".[ui]"
install-ui: install
	python -m pip install -e ".[ui]"

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
	pytest tests/ -v --tb=short --cov=scripts --cov=api --cov-report=term-missing

# PS> : python -m scripts.evaluate_rag
evaluate:
	python -m tests.evaluate_rag

# ---------- Docker ----------
# PS> : docker build -t puls-events-rag .
docker-build:
	docker build -t puls-events-rag .

# PS> : docker run -p 8000:8000 --env-file .env puls-events-rag
docker-run:
	docker run --rm -p 8000:8000 --env-file .env --mount "type=bind,source=$(CURDIR)/data,target=/app/data" puls-events-rag

# ---------- Nettoyage ----------
# PS> : python -c "..." (supprime caches, .pytest_cache, .coverage, *.csv)
clean:
	python -c "import pathlib, shutil; \
	[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; \
	[shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('htmlcov'), pathlib.Path('build'), pathlib.Path('dist')]]; \
	[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]; \
	[p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('*.csv')]; \
	[pathlib.Path(p).unlink(missing_ok=True) for p in ['.coverage']]"

# Purge explicite des données (force un refetch complet) — attention, irréversible
# PS> : powershell -Command "Remove-Item -Recurse -Force data"
purge-data:
	powershell -Command "Remove-Item -Recurse -Force data -ErrorAction SilentlyContinue"
