.PHONY: all pipeline run chat test evaluate docker-build docker-run clean

all: pipeline chat test evaluate docker-build

# 1. Pipeline de données : ingestion, nettoyage, puis indexation
pipeline:
	python -m scripts.fetch_events
	python -m scripts.clean_events
	python -m scripts.vectorize_events

# 2. Exécution locale de la API
run:
	uvicorn api.main:app --reload

# 2.5 Test en ligne de commande 
chat:
	python -m scripts.faiss_retriever
	python -m scripts.mistral_qa

# 3. Tests unitaires et d'évaluation
test:
	pytest tests/ -v

evaluate:
	python -m tests.evaluate_rag

# 4. Commandes Docker
docker-build:
	docker build -t puls-events-rag .

docker-run:
	docker run -p 8000:8000 --env-file .env puls-events-rag

#5. Interface
streamlit:
	streamlit run ui/app.py

clean:
	rm -rf data/vector_store/*