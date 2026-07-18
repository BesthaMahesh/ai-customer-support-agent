import os
from typing import List
from sentence_transformers import SentenceTransformer

# Global cache for the embedding model to avoid loading it repeatedly
_model = None

def get_model() -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    global _model
    if _model is None:
        # 'all-MiniLM-L6-v2' maps sentences to a 384-dimensional dense vector space
        # It is fast, lightweight, and runs locally on CPU
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for a single string."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of strings in batch."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
