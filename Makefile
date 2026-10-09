.PHONY: chunks index clean

chunks:
    python scripts/chunk_events.py

index: chunks
    python scripts/build_faiss_index.py

clean:
    rm -rf data/vector_store/*
    rm -f data/processed/chunks.parquet

all: clean chunks index