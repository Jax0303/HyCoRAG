from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .table_encoder import TableEncoder

@dataclass
class HierarchicalConcepts:
    semantic: torch.Tensor      # (N_sem, D)
    structural: torch.Tensor    # (N_str, D)
    contextual: torch.Tensor    # (N_ctx, D)
    meta: Optional[Dict[str, Any]] = None   # Metadata about origin

class HybridConceptDistiller(nn.Module):
    """
    Distills hybrid concepts (semantic, structural, contextual) from tables.
    """
    def __init__(self, hidden_dim: int, table_encoder: Optional[TableEncoder] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.table_encoder = table_encoder if table_encoder else TableEncoder(hidden_dim=hidden_dim)
        
        # Projections for distillation (simple pooling/projection for now)
        self.semantic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.structural_proj = nn.Linear(hidden_dim, hidden_dim)
        self.contextual_proj = nn.Linear(hidden_dim, hidden_dim)

    def distill(self, 
                table_image: Optional[torch.Tensor] = None, 
                table_structure: Optional[Dict[str, Any]] = None, 
                context_text: Optional[str] = None) -> HierarchicalConcepts:
        """
        Main distillation method.
        """
        # 1. Get raw embeddings from TableEncoder
        raw_embeddings = self.table_encoder.encode_hybrid_table(
            table_image, table_structure, context_text
        )
        
        # 2. Distill Semantic Concepts (Granular)
        # We treat each cell's text embedding as a base semantic unit, 
        # but also aggregate them by row to form "Row Concepts".
        # Current TableEncoder returns cell_embeddings (N_cells, D)
        cell_embs = raw_embeddings['cell_embeddings']
        
        # Project each cell as a potential semantic concept (might be too many, but good for RQ1 ablation)
        semantic_concepts = self.semantic_proj(cell_embs)
        
        # 3. Distill Structural Concepts (Explicit)
        # Use row/col embeddings directly. 
        # TableEncoder generates row_embeddings (Unique per row index) and col_embeddings.
        # We want "Row Structure" and "Column Structure" concepts.
        
        # For 'row_embeddings', TableEncoder currently returns per-cell row embedding.
        # We should unique/mean-pool them to get actual Row Vectors.
        # But TableEncoder logic depends on implementation. 
        # Let's assume raw_embeddings['row_embeddings'] is (N_cells, D) matching cells.
        # We really want 1 concept per Row and 1 per Column.
        
        # Helper to deduplicate embeddings (simple approximation for v1)
        # In a real model, we'd use scatter_mean based on indices.
        # Here we just treat the set of unique vectors as structure concepts.
        row_embs = raw_embeddings['row_embeddings']
        col_embs = raw_embeddings['col_embeddings']
        
        # Structural Concepts = Row Concepts + Col Concepts
        # We project them into structural space
        str_row = self.structural_proj(row_embs) # Projects (N_cells, D)
        str_col = self.structural_proj(col_embs)
        
        # We simply concatenate all for now as the pool of structural concepts.
        # Ideally we would only keep unique ones, BUT:
        # If we keep all, the ConceptRouter can select the specific "Cell Structure" context.
        structural_concepts = torch.cat([str_row, str_col], dim=0)
        
        # 4. Distill Contextual Concepts (Global context)
        ctx_emb = raw_embeddings['context_embedding']
        contextual_concepts = self.contextual_proj(ctx_emb)
        
        return HierarchicalConcepts(
            semantic=semantic_concepts,       # N_cells concepts
            structural=structural_concepts,   # 2 * N_cells concepts (Row + Col view)
            contextual=contextual_concepts,   # 1 concept
            meta={"origin": "distilled_granular"}
        )

    def precompute_for_corpus(self, tables: List[Dict[str, Any]]) -> Dict[str, HierarchicalConcepts]:
        """
        Precomputes distilled concepts for a corpus.
        """
        print(f"[HybridConceptDistiller] Distilling concepts for {len(tables)} tables...")
        # TODO: Implement batch distillation
        return {}

    def adapt_to_new_domain(self, 
                          domain_id: str, 
                          examples: List[Any], 
                          feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Online / self-evolving adaptation hook.
        
        Args:
            domain_id: Identifier for the new domain.
            examples: List of QA examples or new tables.
            feedback: Optional feedback signal (e.g., from user or environment).
        """
        # TODO: Implement parameter updates or adapter training
        print(f"[HybridConceptDistiller] Adapting to domain: {domain_id}")
