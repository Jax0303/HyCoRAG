import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from transformers import AutoModel, AutoTokenizer

class TableEncoder(nn.Module):
    """
    Encoder to extract embeddings from table content (text/structure/image).
    v1: Uses Transformers for text and Embeddings for structure.
    """
    def __init__(self, 
                 input_dim: int = 768, 
                 hidden_dim: int = 768,
                 model_name: str = "prajjwal1/bert-tiny",
                 max_rows: int = 100,
                 max_cols: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Text Encoder (BERT-tiny for speed)
        # Note: bert-tiny dim is 128
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.text_dim = self.text_model.config.hidden_size
        self.text_projector = nn.Linear(self.text_dim, hidden_dim)

        # 2. Structural Encoders (Learnable Embeddings)
        self.row_embedding = nn.Embedding(max_rows, hidden_dim)
        self.col_embedding = nn.Embedding(max_cols, hidden_dim)
        # Span embedding (optional, for merged cells)
        self.span_embedding = nn.Embedding(10, hidden_dim) 

        # 3. Image Encoder Stub (ResNet-like)
        # TODO: Replace with actual Table-LLaVA or similar visual encoder
        self.image_projector = nn.Linear(512, hidden_dim) 

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of strings."""
        if not texts:
            return torch.zeros(0, self.hidden_dim)
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        # Move to device if model is on device (TODO: robust device handling)
        if next(self.text_model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use CLS token
            cls_emb = outputs.last_hidden_state[:, 0, :]
            
        return self.text_projector(cls_emb)

    def encode_hybrid_table(self, 
                          table_image: Optional[torch.Tensor] = None, 
                          table_structure: Optional[Dict[str, Any]] = None, 
                          context_text: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Encodes table inputs into a dictionary of embeddings.
        """
        embeddings = {}
        device = next(self.text_model.parameters()).device
        
        # 1. Image Encoding (Stub)
        if table_image is not None:
            # Assume table_image is already a feature vector or we project random
            # For now, generate random if not provided in right shape
            if table_image.dim() < 2:
                img_feat = torch.randn(1, 512).to(device)
            else:
                img_feat = table_image.to(device)
            embeddings['image_feat'] = self.image_projector(img_feat)
        
        # 2. Structure & Content Encoding
        # We need "Rows" and "Columns" concepts.
        # Ideally, table_structure provides explicit lists of cells with coordinates.
        
        if table_structure and "cells" in table_structure:
            # List of {text, row, col, rowspan, colspan}
            cells = table_structure['cells']
            
            # Batch text encoding
            cell_texts = [c.get('text', '') for c in cells]
            cell_text_embs = self.encode_text(cell_texts) # (N, D)
            
            # Structural embeddings
            row_indices = torch.tensor([min(c.get('row', 0), 99) for c in cells], device=device)
            col_indices = torch.tensor([min(c.get('col', 0), 49) for c in cells], device=device)
            
            row_embs = self.row_embedding(row_indices)
            col_embs = self.col_embedding(col_indices)
            
            # Combine: Cell Representation = Text + Row + Col
            full_cell_embs = cell_text_embs + row_embs + col_embs
            
            embeddings['cell_embeddings'] = full_cell_embs
            # Aggregate to get Row/Col embeddings for concepts
            # (Simplified: Just take unique row/col embeddings from vocabulary for now, 
            #  or mean pool cells. Let's return raw structure embeddings for Distiller to aggregate)
            embeddings['row_embeddings'] = row_embs
            embeddings['col_embeddings'] = col_embs
            
        elif table_structure and "text" in table_structure:
            # Fallback: Treat whole text as one cell
            text_emb = self.encode_text([table_structure['text']])
            embeddings['cell_embeddings'] = text_emb
            embeddings['row_embeddings'] = torch.zeros_like(text_emb)
            embeddings['col_embeddings'] = torch.zeros_like(text_emb)
            
        else:
            # Fallback Defaults (Empty)
            dummy = torch.zeros(1, self.hidden_dim).to(device)
            embeddings['cell_embeddings'] = dummy
            embeddings['row_embeddings'] = dummy
            embeddings['col_embeddings'] = dummy
            
        # 3. Context Encoding
        if context_text:
            embeddings['context_embedding'] = self.encode_text([context_text])
        else:
            embeddings['context_embedding'] = torch.zeros(1, self.hidden_dim).to(device)
            
        return embeddings

    def forward(self, table_input):
        return self.encode_hybrid_table()
