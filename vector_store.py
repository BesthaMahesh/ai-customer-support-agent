import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
import config
from utils import get_logger

logger = get_logger("VectorStore")

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        """Initializes FAISS L2 flat index and internal metadata/embeddings lists."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []  # Store raw embeddings to support rebuilds/deletions

    def add_documents(self, embeddings: List[List[float]], docs_metadata: List[Dict[str, Any]], filename: str) -> None:
        """Adds document embeddings and metadata incrementally.
        
        If a file with the same filename is already indexed, it deletes its existing 
        records first, ensuring only the latest version of the file exists in the database.
        """
        if not embeddings:
            return
            
        logger.info(f"Incrementally adding {len(embeddings)} chunks for '{filename}'")
        
        # 1. Handle duplication by removing old metadata/embeddings associated with the filename
        self.remove_document(filename)
        
        # 2. Append new records
        self.embeddings.extend(embeddings)
        self.metadata.extend(docs_metadata)
        
        # 3. Rebuild the FAISS index to keep it in sync
        self.rebuild_index()

    def remove_document(self, filename: str) -> None:
        """Removes all chunks and embeddings belonging to the specified filename."""
        logger.info(f"Removing '{filename}' from index if it exists...")
        initial_count = len(self.metadata)
        
        # Keep items that do not match the filename
        indices_to_keep = [i for i, meta in enumerate(self.metadata) if meta["source"] != filename]
        
        if len(indices_to_keep) < initial_count:
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            self.embeddings = [self.embeddings[i] for i in indices_to_keep]
            logger.info(f"Removed {initial_count - len(indices_to_keep)} chunks for '{filename}'")
        
    def rebuild_index(self) -> None:
        """Rebuilds the FAISS index using currently active embeddings in memory."""
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.embeddings:
            vectors = np.array(self.embeddings).astype('float32')
            self.index.add(vectors)
        logger.info(f"FAISS Index rebuilt. Total active vectors: {self.index.ntotal}")

    def search_l2(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Standard L2 similarity search."""
        if self.index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(distances[0][i])))
        return results

    def search_mmr(self, query_embedding: List[float], k: int = 5, fetch_k: int = 20, lambda_val: float = 0.5) -> List[Tuple[Dict[str, Any], float]]:
        """Performs Maximal Marginal Relevance (MMR) search.
        
        Balances relevance (similarity to query) and diversity (similarity to already selected chunks).
        """
        if self.index.ntotal == 0:
            return []

        # 1. Fetch top fetch_k candidate vectors using standard L2 similarity
        vector = np.array([query_embedding]).astype('float32')
        fetch_k = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(vector, fetch_k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            candidates.append(idx)
            
        if not candidates:
            return []
            
        k = min(k, len(candidates))
        
        # Calculate similarity scores to query for candidates
        # CosineSimilarity = 1.0 - (L2_Distance / 2.0)
        query_sims = {idx: 1.0 - (distances[0][i] / 2.0) for i, idx in enumerate(candidates)}
        
        # Get normalized vectors of candidates to calculate cross-similarities
        candidate_embeddings = {}
        for idx in candidates:
            emb = np.array(self.embeddings[idx])
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            candidate_embeddings[idx] = emb
            
        selected_indices = []
        
        # 2. MMR Iterative selection loop
        while len(selected_indices) < k:
            best_mmr = -float('inf')
            best_candidate = -1
            
            for cand in candidates:
                if cand in selected_indices:
                    continue
                    
                sim_to_query = query_sims[cand]
                
                if not selected_indices:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max([
                        np.dot(candidate_embeddings[cand], candidate_embeddings[sel])
                        for sel in selected_indices
                    ])
                    
                # MMR formula
                mmr_score = lambda_val * sim_to_query - (1.0 - lambda_val) * max_sim_to_selected
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_candidate = cand
                    
            if best_candidate == -1:
                break
                
            selected_indices.append(best_candidate)
            
        # 3. Retrieve L2 distances for selected indices
        results = []
        for idx in selected_indices:
            dist = 2.0  # default
            for i, c_idx in enumerate(indices[0]):
                if c_idx == idx:
                    dist = float(distances[0][i])
                    break
            results.append((self.metadata[idx], dist))
            
        return results

    def save(self) -> None:
        """Saves index and parallel metadata/embeddings lists to disk."""
        os.makedirs(config.DB_DIR, exist_ok=True)
        faiss.write_index(self.index, config.INDEX_PATH)
        with open(config.META_PATH, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "embeddings": self.embeddings
            }, f)
        logger.info(f"Vector Database saved to disk. Vectors: {self.index.ntotal}")

    def load(self) -> bool:
        """Loads index and metadata/embeddings from disk. Handles legacy formats."""
        if os.path.exists(config.INDEX_PATH) and os.path.exists(config.META_PATH):
            try:
                self.index = faiss.read_index(config.INDEX_PATH)
                with open(config.META_PATH, "rb") as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        self.metadata = data.get("metadata", [])
                        self.embeddings = data.get("embeddings", [])
                    else:
                        # Legacy format fallback
                        self.metadata = data
                        # Fallback empty embeddings
                        self.embeddings = []
                        
                logger.info(f"Vector Database loaded from disk. Vectors: {self.index.ntotal}")
                return True
            except Exception as e:
                logger.error(f"Error loading Vector Database: {e}")
                return False
        return False

    def clear(self) -> None:
        """Resets the FAISS index and deletes all DB files on disk."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.embeddings = []
        if os.path.exists(config.INDEX_PATH):
            try:
                os.remove(config.INDEX_PATH)
            except Exception:
                pass
        if os.path.exists(config.META_PATH):
            try:
                os.remove(config.META_PATH)
            except Exception:
                pass
        logger.info("Vector Database successfully cleared.")
