from scripts.build_faiss_index import verify_index, build_index, load_data
import faiss
import pandas as pd

def test_index_construction():
    df = load_data()
    index, metadata = build_index(df)
    verify_index(index, metadata)