import os
from typing import List
from fastembed import TextEmbedding

# Global cache for the embedding model to avoid loading it repeatedly
_model = None

def get_model() -> TextEmbedding:
    """Loads and caches the FastEmbed TextEmbedding model.
    
    ONNX Runtime is used for fast and memory-efficient CPU inference.
    """
    global _model
    if _model is None:
        # 'BAAI/bge-small-en-v1.5' maps sentences to a 384-dimensional dense vector space.
        # It is fast, lightweight (approx. 45MB), and runs efficiently on CPU.
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for a single string."""
    model = get_model()
    # model.embed returns a generator of numpy arrays
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of strings in batch."""
    model = get_model()
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]
