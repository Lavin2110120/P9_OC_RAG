import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Charger l'index et les métadonnées
index = faiss.read_index("data/vector_store/faiss_index.bin")
metadata = pd.read_parquet("data/vector_store/metadata.parquet")

# Charger le modèle d'embedding
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Tester une requête
query = "concert à La Vapeur en octobre"
query_embedding = model.encode([query], normalize_embeddings=True)

# Rechercher les 3 événements les plus proches
D, I = index.search(query_embedding, k=3)

# Afficher les résultats
print(f"Requête: '{query}'\n")
for i, (dist, idx) in enumerate(zip(D[0], I[0])):
    event = metadata.iloc[idx]
    print(f"{i+1}. {event['title']} (le {event['date_begin'].strftime('%Y-%m-%d')}) - Distance: {dist:.4f}")