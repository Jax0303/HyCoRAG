"""
Cell-Header Attention Encoder

독창적 기여 #1: 셀-헤더 관계를 명시적으로 모델링
- TabRAG: VLM 기반 구조 추출 후 독립적 처리
- HyCoRAG: 셀과 헤더의 관계를 어텐션으로 명시적 인코딩

목표: 셀 정확도 8.2% → 35%+ 달성
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HeaderInfo:
    """헤더 정보 구조"""
    text: str
    level: int  # 계층 레벨 (0=최상위)
    parent_id: Optional[int] = None
    row_idx: Optional[int] = None
    col_idx: Optional[int] = None
    is_row_header: bool = False
    is_col_header: bool = False


@dataclass
class CellInfo:
    """셀 정보 구조"""
    text: str
    row: int
    col: int
    row_header_ids: List[int]  # 관련 행 헤더들
    col_header_ids: List[int]  # 관련 열 헤더들
    is_header: bool = False


class HeaderHierarchyEncoder(nn.Module):
    """
    헤더 계층 구조를 Transformer로 인코딩
    
    핵심 아이디어:
    - 부모-자식 헤더 관계를 명시적으로 학습
    - 계층 레벨 정보를 위치 인코딩으로 추가
    """
    
    def __init__(self, hidden_dim: int = 768, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 계층 레벨 임베딩 (최대 5단계 계층)
        self.level_embedding = nn.Embedding(5, hidden_dim)
        
        # Transformer Encoder for hierarchy
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.hierarchy_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(
        self, 
        header_embeddings: torch.Tensor,  # (num_headers, hidden_dim)
        header_levels: torch.Tensor,      # (num_headers,)
        parent_mask: Optional[torch.Tensor] = None  # (num_headers, num_headers)
    ) -> torch.Tensor:
        """
        Args:
            header_embeddings: 헤더 텍스트 임베딩
            header_levels: 각 헤더의 계층 레벨
            parent_mask: 부모-자식 관계 마스크 (optional)
        
        Returns:
            계층 정보가 인코딩된 헤더 표현
        """
        # 1. 레벨 임베딩 추가
        level_embs = self.level_embedding(header_levels)
        enriched_headers = header_embeddings + level_embs
        
        # 2. Transformer로 계층 관계 학습
        # parent_mask가 있으면 사용 (부모만 attend)
        hierarchical_headers = self.hierarchy_encoder(
            enriched_headers.unsqueeze(0),  # (1, num_headers, hidden_dim)
            mask=parent_mask
        ).squeeze(0)
        
        return hierarchical_headers


class CellHeaderAttentionEncoder(nn.Module):
    """
    셀-헤더 관계 명시적 모델링
    
    핵심 메커니즘:
    1. 각 셀이 관련 헤더들에 어텐션
    2. 헤더 컨텍스트를 셀 표현에 통합
    3. 셀 = 값 임베딩 + 헤더 컨텍스트 + 위치 정보
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head attention: 셀 → 헤더
        self.cell_to_header_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 위치 인코딩 (행/열)
        self.row_pos_embedding = nn.Embedding(100, hidden_dim)
        self.col_pos_embedding = nn.Embedding(100, hidden_dim)
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        cell_value_embeddings: torch.Tensor,  # (num_cells, hidden_dim)
        header_embeddings: torch.Tensor,      # (num_headers, hidden_dim)
        cell_to_header_map: torch.Tensor,     # (num_cells, max_headers) - 각 셀의 관련 헤더 인덱스
        cell_positions: torch.Tensor          # (num_cells, 2) - [row, col]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cell_value_embeddings: 셀 값의 텍스트 임베딩
            header_embeddings: 계층 인코딩된 헤더 임베딩
            cell_to_header_map: 각 셀이 참조하는 헤더 인덱스들
            cell_positions: 셀의 (행, 열) 위치
        
        Returns:
            (enriched_cell_embeddings, attention_weights)
        """
        num_cells = cell_value_embeddings.size(0)
        device = cell_value_embeddings.device
        
        # 1. 위치 인코딩 추가
        row_pos = self.row_pos_embedding(cell_positions[:, 0].clamp(0, 99))
        col_pos = self.col_pos_embedding(cell_positions[:, 1].clamp(0, 99))
        positional_encoding = row_pos + col_pos
        
        cell_with_pos = cell_value_embeddings + positional_encoding
        
        # 2. 각 셀에 대해 관련 헤더들에 어텐션
        enriched_cells = []
        all_attention_weights = []
        
        for i in range(num_cells):
            # 이 셀과 관련된 헤더들 추출
            relevant_header_ids = cell_to_header_map[i]
            # -1은 패딩, 제거
            valid_mask = relevant_header_ids >= 0
            relevant_header_ids = relevant_header_ids[valid_mask]
            
            if len(relevant_header_ids) == 0:
                # 관련 헤더 없으면 원본 사용
                enriched_cells.append(cell_with_pos[i])
                all_attention_weights.append(torch.zeros(1, device=device))
                continue
            
            # 관련 헤더 임베딩
            relevant_headers = header_embeddings[relevant_header_ids]  # (num_relevant, hidden_dim)
            
            # 어텐션: 셀(query) → 헤더들(key, value)
            cell_query = cell_with_pos[i].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
            header_kv = relevant_headers.unsqueeze(0)  # (1, num_relevant, hidden_dim)
            
            attended_context, attn_weights = self.cell_to_header_attention(
                query=cell_query,
                key=header_kv,
                value=header_kv
            )
            # attended_context: (1, 1, hidden_dim)
            # attn_weights: (1, 1, num_relevant)
            
            # 3. 셀 값 + 헤더 컨텍스트 융합
            combined = torch.cat([
                cell_with_pos[i].unsqueeze(0),
                attended_context.squeeze(0)
            ], dim=-1)  # (1, hidden_dim * 2)
            
            enriched_cell = self.fusion(combined).squeeze(0)  # (hidden_dim,)
            
            enriched_cells.append(enriched_cell)
            all_attention_weights.append(attn_weights.squeeze(0))
        
        enriched_cell_embeddings = torch.stack(enriched_cells)  # (num_cells, hidden_dim)
        
        return enriched_cell_embeddings, all_attention_weights


class CellHeaderAwareDistiller(nn.Module):
    """
    Cell-Header Attention을 통합한 개선된 Concept Distiller
    
    기존 HybridConceptDistiller 대비 개선점:
    - 셀과 헤더의 관계를 명시적으로 모델링
    - 계층 구조 보존
    - 셀 정확도 향상 목표
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        text_encoder: Optional[nn.Module] = None,
        use_hierarchy_encoding: bool = True,
        use_cell_header_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hierarchy_encoding = use_hierarchy_encoding
        self.use_cell_header_attention = use_cell_header_attention
        
        # 텍스트 인코더 (외부에서 주입 가능)
        self.text_encoder = text_encoder
        
        # 헤더 계층 인코더
        if use_hierarchy_encoding:
            self.header_hierarchy_encoder = HeaderHierarchyEncoder(hidden_dim)
        
        # 셀-헤더 어텐션 인코더
        if use_cell_header_attention:
            self.cell_header_attention = CellHeaderAttentionEncoder(hidden_dim)
        
        # 개념 투영 레이어
        self.semantic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.structural_proj = nn.Linear(hidden_dim, hidden_dim)
        self.contextual_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """텍스트를 임베딩으로 변환 (외부 인코더 사용)"""
        if self.text_encoder is not None:
            return self.text_encoder.encode_text(texts)
        else:
            # Fallback: 랜덤 임베딩 (실제로는 BERT 등 사용)
            return torch.randn(len(texts), self.hidden_dim)
    
    def extract_headers_and_cells(
        self, 
        table_structure: Dict[str, Any]
    ) -> Tuple[List[HeaderInfo], List[CellInfo]]:
        """
        테이블 구조에서 헤더와 셀 정보 추출
        
        Args:
            table_structure: 테이블 구조 정보
                - 'cells': 셀 리스트 (각 셀은 text, row, col, is_header 포함)
                - 'headers': (optional) 명시적 헤더 정보
        
        Returns:
            (headers, cells)
        """
        headers = []
        cells = []
        
        if 'cells' not in table_structure:
            return headers, cells
        
        raw_cells = table_structure['cells']
        
        # 1. 헤더 추출 (is_header=True인 셀들)
        header_cells = [c for c in raw_cells if c.get('is_header', False)]
        
        for i, hc in enumerate(header_cells):
            header = HeaderInfo(
                text=hc.get('text', ''),
                level=hc.get('level', 0),
                parent_id=hc.get('parent_id'),
                row_idx=hc.get('row'),
                col_idx=hc.get('col'),
                is_row_header=hc.get('is_row_header', False),
                is_col_header=hc.get('is_col_header', True)
            )
            headers.append(header)
        
        # 2. 데이터 셀 추출
        data_cells = [c for c in raw_cells if not c.get('is_header', False)]
        
        for dc in data_cells:
            # 이 셀과 관련된 헤더 찾기
            row_header_ids = []
            col_header_ids = []
            
            for i, h in enumerate(headers):
                # 열 헤더: 같은 열에 있는 헤더
                if h.is_col_header and h.col_idx == dc.get('col'):
                    col_header_ids.append(i)
                # 행 헤더: 같은 행에 있는 헤더
                if h.is_row_header and h.row_idx == dc.get('row'):
                    row_header_ids.append(i)
            
            cell = CellInfo(
                text=dc.get('text', ''),
                row=dc.get('row', 0),
                col=dc.get('col', 0),
                row_header_ids=row_header_ids,
                col_header_ids=col_header_ids,
                is_header=False
            )
            cells.append(cell)
        
        return headers, cells
    
    def distill(
        self,
        table_structure: Dict[str, Any],
        context_text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Cell-Header Attention을 사용한 개념 추출
        
        Returns:
            {
                'semantic_concepts': 셀 개념 (헤더 컨텍스트 포함),
                'structural_concepts': 헤더 개념 (계층 인코딩),
                'contextual_concepts': 전역 컨텍스트,
                'attention_weights': 셀-헤더 어텐션 가중치
            }
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        # 1. 헤더와 셀 추출
        headers, cells = self.extract_headers_and_cells(table_structure)
        
        if len(headers) == 0 or len(cells) == 0:
            # Fallback: 빈 텐서 반환
            dummy = torch.zeros(1, self.hidden_dim, device=device)
            return {
                'semantic_concepts': dummy,
                'structural_concepts': dummy,
                'contextual_concepts': dummy,
                'attention_weights': None
            }
        
        # 2. 헤더 임베딩 및 계층 인코딩
        header_texts = [h.text for h in headers]
        header_embeddings = self.encode_text(header_texts).to(device)
        
        if self.use_hierarchy_encoding:
            header_levels = torch.tensor([h.level for h in headers], device=device)
            hierarchical_headers = self.header_hierarchy_encoder(
                header_embeddings,
                header_levels
            )
        else:
            hierarchical_headers = header_embeddings
        
        # 3. 셀 임베딩
        cell_texts = [c.text for c in cells]
        cell_value_embeddings = self.encode_text(cell_texts).to(device)
        
        # 4. 셀-헤더 매핑 준비
        max_headers_per_cell = max(
            len(c.row_header_ids) + len(c.col_header_ids) 
            for c in cells
        ) if cells else 1
        
        cell_to_header_map = torch.full(
            (len(cells), max_headers_per_cell), 
            -1, 
            dtype=torch.long,
            device=device
        )
        
        for i, cell in enumerate(cells):
            all_header_ids = cell.row_header_ids + cell.col_header_ids
            for j, hid in enumerate(all_header_ids):
                if j < max_headers_per_cell:
                    cell_to_header_map[i, j] = hid
        
        cell_positions = torch.tensor(
            [[c.row, c.col] for c in cells],
            dtype=torch.long,
            device=device
        )
        
        # 5. Cell-Header Attention 적용
        if self.use_cell_header_attention:
            enriched_cells, attention_weights = self.cell_header_attention(
                cell_value_embeddings,
                hierarchical_headers,
                cell_to_header_map,
                cell_positions
            )
        else:
            enriched_cells = cell_value_embeddings
            attention_weights = None
        
        # 6. 개념 투영
        semantic_concepts = self.semantic_proj(enriched_cells)
        structural_concepts = self.structural_proj(hierarchical_headers)
        
        # 7. 컨텍스트 개념
        if context_text:
            context_emb = self.encode_text([context_text]).to(device)
            contextual_concepts = self.contextual_proj(context_emb)
        else:
            contextual_concepts = torch.zeros(1, self.hidden_dim, device=device)
        
        return {
            'semantic_concepts': semantic_concepts,
            'structural_concepts': structural_concepts,
            'contextual_concepts': contextual_concepts,
            'attention_weights': attention_weights,
            'num_cells': len(cells),
            'num_headers': len(headers)
        }
