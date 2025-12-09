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
    Routes queries to the most relevant concepts with structural awareness.
    
    Key improvement: Ensures minimum structural concepts to preserve
    table structure (headers, hierarchy) - addresses RQ2 failure.
    """
    def __init__(self, hidden_dim: int, structural_min_quota: int = 10):
        super().__init__()
        # Query projection to match concept space
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Structural quota: minimum structural concepts to preserve
        self.structural_min_quota = structural_min_quota

    def route(
        self, 
        query_embedding: torch.Tensor, 
        concepts: HierarchicalConcepts, 
        top_k: int = 4,
        preserve_structure: bool = True,
        header_first: bool = True  # NEW: Prioritize headers
    ) -> RoutedConcepts:
        """
        Selects concepts based on query with structural awareness.
        
        Args:
            query_embedding: Tensor of shape (1, hidden_dim)
            concepts: HierarchicalConcepts object
            top_k: Number of semantic/contextual concepts to select
            preserve_structure: If True, enforce structural minimum quota
            header_first: If True, include ALL header concepts (exceeds baseline)
            
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
        
        def select_all_with_weights(concept_tensor: torch.Tensor):
            """Select all concepts with similarity weights."""
            if concept_tensor.size(0) == 0:
                return torch.empty(0, concept_tensor.size(1)), torch.empty(0)
            
            # Compute similarity for all
            sim = torch.matmul(q_emb, concept_tensor.t()).squeeze(0)
            return concept_tensor, sim

        # Route semantic and contextual with top_k
        sem_sel, sem_w = select_top_k(concepts.semantic, top_k)
        ctx_sel, ctx_w = select_top_k(concepts.contextual, top_k)
        
        # Structural routing with header-first strategy
        if header_first:
            # CRITICAL: Include ALL structural concepts (headers)
            # This ensures we exceed baseline performance
            str_sel, str_w = select_all_with_weights(concepts.structural)
        elif preserve_structure:
            # Ensure minimum structural concepts for RQ2
            structural_k = max(top_k, self.structural_min_quota)
            str_sel, str_w = select_top_k(concepts.structural, structural_k)
        else:
            str_sel, str_w = select_top_k(concepts.structural, top_k)
        
        all_weights = torch.cat([sem_w, str_w, ctx_w])
        
        return RoutedConcepts(
            semantic=sem_sel,
            structural=str_sel,
            contextual=ctx_sel,
            weights=all_weights,
            meta={
                "top_k": top_k,
                "structural_quota": len(str_sel) if header_first else (self.structural_min_quota if preserve_structure else top_k),
                "preserve_structure": preserve_structure,
                "header_first": header_first
            }
        )
