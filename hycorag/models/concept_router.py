from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .concept_distill import HierarchicalConcepts

@dataclass
class RoutedConcepts:
    semantic: torch.Tensor      # (K_sem, D)
    structural: torch.Tensor    # (K_str, D)
    contextual: torch.Tensor    # (K_ctx, D)
    weights: torch.Tensor       # (K_total,) similarity weights
    meta: Optional[Dict[str, Any]] = None

class ConceptRouter(nn.Module):
    """
    Routes queries to the most relevant concepts.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Query projection to match concept space
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def route(self, query_embedding: torch.Tensor, concepts: HierarchicalConcepts, top_k: int = 4) -> RoutedConcepts:
        """
        Selects concepts based on the query.
        
        Args:
            query_embedding: Tensor of shape (1, hidden_dim)
            concepts: HierarchicalConcepts object
            top_k: Number of concepts to select per type (simplified)
            
        Returns:
            RoutedConcepts object.
        """
        q_emb = self.query_proj(query_embedding) # (1, D)
        
        def select_top_k(concept_tensor: torch.Tensor, k: int):
            # concept_tensor: (N, D)
            if concept_tensor.size(0) == 0:
                return torch.empty(0, concept_tensor.size(1)), torch.empty(0)
                
            # Compute similarity: (1, D) @ (D, N) -> (1, N)
            sim = torch.matmul(q_emb, concept_tensor.t()).squeeze(0)
            
            k = min(k, concept_tensor.size(0))
            topk_vals, topk_inds = torch.topk(sim, k)
            
            selected = concept_tensor[topk_inds]
            return selected, topk_vals

        # Route for each type
        # TODO: Adaptive k based on query type?
        sem_sel, sem_w = select_top_k(concepts.semantic, top_k)
        str_sel, str_w = select_top_k(concepts.structural, top_k)
        ctx_sel, ctx_w = select_top_k(concepts.contextual, top_k)
        
        all_weights = torch.cat([sem_w, str_w, ctx_w])
        
        return RoutedConcepts(
            semantic=sem_sel,
            structural=str_sel,
            contextual=ctx_sel,
            weights=all_weights,
            meta={"top_k": top_k}
        )
