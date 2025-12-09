from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    """
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

from dataclasses import dataclass
import numpy as np

@dataclass
class RetrievedItem:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = None

class BaselineRetriever(BaseRetriever):
    """
    Baseline retriever using simple embeddings (dummy for now).
    """
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index_map = {} # id -> text
        self.embeddings = {} # id -> vector

    def index(self, tables: Dict[str, str]):
        """
        Index a dictionary of {table_id: table_text}.
        """
        self.index_map = tables
        print(f"Indexing {len(tables)} tables...")
        for tid, text in tables.items():
            # Dummy embedding: deterministic random based on length
            np.random.seed(len(text)) 
            self.embeddings[tid] = np.random.randn(self.embedding_dim)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve top-k tables for a query.
        """
        # Dummy query embedding
        np.random.seed(len(query))
        query_vec = np.random.randn(self.embedding_dim)
        
        scores = []
        for tid, emb in self.embeddings.items():
            # Cosine similarity
            score = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            scores.append((tid, score))
            
        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_scores = scores[:top_k]
        
        results = []
        for tid, score in top_k_scores:
            # Check if we have stored metadata for this doc
            doc_metadata = {"origin": "baseline_retriever"}
            if hasattr(self, '_corpus_metadata') and tid in self._corpus_metadata:
                doc_metadata = self._corpus_metadata[tid]
                
            results.append(RetrievedItem(
                id=tid,
                text=self.index_map[tid],
                score=float(score),
                metadata=doc_metadata
            ))
            
        return results
