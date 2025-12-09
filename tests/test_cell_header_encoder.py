"""
Cell-Header Attention 단위 테스트

목적:
1. CellHeaderAttentionEncoder의 기본 동작 검증
2. 셀-헤더 관계 모델링 확인
3. 통합 전 기능 검증
"""

import sys
sys.path.insert(0, '/home/user/HyCoRAG')

import torch
from hycorag.models.cell_header_encoder import (
    HeaderHierarchyEncoder,
    CellHeaderAttentionEncoder,
    CellHeaderAwareDistiller,
    HeaderInfo,
    CellInfo
)


class TestHeaderHierarchyEncoder:
    """헤더 계층 인코더 테스트"""
    
    def test_basic_encoding(self):
        """기본 계층 인코딩 테스트"""
        encoder = HeaderHierarchyEncoder(hidden_dim=768)
        
        # 3개 헤더, 2단계 계층
        num_headers = 3
        header_embeddings = torch.randn(num_headers, 768)
        header_levels = torch.tensor([0, 1, 1])  # 최상위 1개, 하위 2개
        
        output = encoder(header_embeddings, header_levels)
        
        assert output.shape == (num_headers, 768)
        assert not torch.isnan(output).any()
    
    def test_different_levels(self):
        """다양한 계층 레벨 테스트"""
        encoder = HeaderHierarchyEncoder(hidden_dim=768)
        
        # 5단계 계층
        num_headers = 5
        header_embeddings = torch.randn(num_headers, 768)
        header_levels = torch.tensor([0, 1, 1, 2, 2])
        
        output = encoder(header_embeddings, header_levels)
        
        assert output.shape == (num_headers, 768)
        # 계층 정보가 반영되었는지 확인 (원본과 달라야 함)
        assert not torch.allclose(output, header_embeddings, atol=0.1)


class TestCellHeaderAttentionEncoder:
    """셀-헤더 어텐션 인코더 테스트"""
    
    def test_basic_attention(self):
        """기본 어텐션 동작 테스트"""
        encoder = CellHeaderAttentionEncoder(hidden_dim=768)
        
        # 5개 셀, 3개 헤더
        num_cells = 5
        num_headers = 3
        
        cell_embeddings = torch.randn(num_cells, 768)
        header_embeddings = torch.randn(num_headers, 768)
        
        # 각 셀은 최대 2개 헤더 참조
        cell_to_header_map = torch.tensor([
            [0, 1],  # 셀 0: 헤더 0, 1
            [1, 2],  # 셀 1: 헤더 1, 2
            [0, -1], # 셀 2: 헤더 0만
            [2, -1], # 셀 3: 헤더 2만
            [0, 1],  # 셀 4: 헤더 0, 1
        ])
        
        cell_positions = torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1], [2, 0]
        ])
        
        enriched_cells, attn_weights = encoder(
            cell_embeddings,
            header_embeddings,
            cell_to_header_map,
            cell_positions
        )
        
        assert enriched_cells.shape == (num_cells, 768)
        assert len(attn_weights) == num_cells
        # 헤더 컨텍스트가 추가되어 원본과 달라야 함
        assert not torch.allclose(enriched_cells, cell_embeddings, atol=0.1)
    
    def test_no_headers(self):
        """헤더 없는 셀 처리 테스트"""
        encoder = CellHeaderAttentionEncoder(hidden_dim=768)
        
        num_cells = 2
        num_headers = 2
        
        cell_embeddings = torch.randn(num_cells, 768)
        header_embeddings = torch.randn(num_headers, 768)
        
        # 모든 셀이 헤더 없음
        cell_to_header_map = torch.full((num_cells, 2), -1)
        cell_positions = torch.tensor([[0, 0], [0, 1]])
        
        enriched_cells, attn_weights = encoder(
            cell_embeddings,
            header_embeddings,
            cell_to_header_map,
            cell_positions
        )
        
        assert enriched_cells.shape == (num_cells, 768)
        # 헤더 없으면 위치 정보만 추가 (크게 변하지 않음)
        assert torch.allclose(enriched_cells, cell_embeddings, atol=1.0)


class TestCellHeaderAwareDistiller:
    """통합 Distiller 테스트"""
    
    def test_basic_distillation(self):
        """기본 개념 추출 테스트"""
        distiller = CellHeaderAwareDistiller(
            hidden_dim=768,
            use_hierarchy_encoding=True,
            use_cell_header_attention=True
        )
        
        # 간단한 테이블 구조
        table_structure = {
            'cells': [
                # 헤더
                {'text': 'Year', 'row': 0, 'col': 0, 'is_header': True, 
                 'level': 0, 'is_col_header': True},
                {'text': 'Revenue', 'row': 0, 'col': 1, 'is_header': True, 
                 'level': 0, 'is_col_header': True},
                # 데이터 셀
                {'text': '2023', 'row': 1, 'col': 0, 'is_header': False},
                {'text': '100M', 'row': 1, 'col': 1, 'is_header': False},
                {'text': '2024', 'row': 2, 'col': 0, 'is_header': False},
                {'text': '120M', 'row': 2, 'col': 1, 'is_header': False},
            ]
        }
        
        result = distiller.distill(table_structure)
        
        assert 'semantic_concepts' in result
        assert 'structural_concepts' in result
        assert 'contextual_concepts' in result
        
        # 4개 데이터 셀 → 4개 semantic concepts
        assert result['semantic_concepts'].shape[0] == 4
        # 2개 헤더 → 2개 structural concepts
        assert result['structural_concepts'].shape[0] == 2
    
    def test_hierarchical_headers(self):
        """계층적 헤더 처리 테스트"""
        distiller = CellHeaderAwareDistiller(hidden_dim=768)
        
        # 2단계 계층 헤더
        table_structure = {
            'cells': [
                # 최상위 헤더
                {'text': 'Financial', 'row': 0, 'col': 0, 'is_header': True, 
                 'level': 0, 'is_col_header': True},
                # 하위 헤더
                {'text': 'Revenue', 'row': 1, 'col': 0, 'is_header': True, 
                 'level': 1, 'parent_id': 0, 'is_col_header': True},
                {'text': 'Profit', 'row': 1, 'col': 1, 'is_header': True, 
                 'level': 1, 'parent_id': 0, 'is_col_header': True},
                # 데이터
                {'text': '100M', 'row': 2, 'col': 0, 'is_header': False},
                {'text': '20M', 'row': 2, 'col': 1, 'is_header': False},
            ]
        }
        
        result = distiller.distill(table_structure)
        
        # 3개 헤더 (1 최상위 + 2 하위)
        assert result['structural_concepts'].shape[0] == 3
        # 2개 데이터 셀
        assert result['semantic_concepts'].shape[0] == 2
    
    def test_empty_table(self):
        """빈 테이블 처리 테스트"""
        distiller = CellHeaderAwareDistiller(hidden_dim=768)
        
        table_structure = {'cells': []}
        
        result = distiller.distill(table_structure)
        
        # 빈 테이블도 처리 가능해야 함
        assert result['semantic_concepts'].shape == (1, 768)
        assert result['structural_concepts'].shape == (1, 768)


def test_integration():
    """전체 파이프라인 통합 테스트"""
    print("\n" + "="*70)
    print("Cell-Header Attention Integration Test")
    print("="*70)
    
    # 1. Distiller 생성
    distiller = CellHeaderAwareDistiller(
        hidden_dim=768,
        use_hierarchy_encoding=True,
        use_cell_header_attention=True
    )
    
    # 2. 실제와 유사한 테이블 구조
    table_structure = {
        'cells': [
            # 열 헤더
            {'text': 'Company', 'row': 0, 'col': 0, 'is_header': True, 
             'level': 0, 'is_col_header': True},
            {'text': 'Q1 Revenue', 'row': 0, 'col': 1, 'is_header': True, 
             'level': 0, 'is_col_header': True},
            {'text': 'Q2 Revenue', 'row': 0, 'col': 2, 'is_header': True, 
             'level': 0, 'is_col_header': True},
            
            # 데이터 행 1
            {'text': 'Apple', 'row': 1, 'col': 0, 'is_header': False},
            {'text': '$90B', 'row': 1, 'col': 1, 'is_header': False},
            {'text': '$95B', 'row': 1, 'col': 2, 'is_header': False},
            
            # 데이터 행 2
            {'text': 'Google', 'row': 2, 'col': 0, 'is_header': False},
            {'text': '$70B', 'row': 2, 'col': 1, 'is_header': False},
            {'text': '$75B', 'row': 2, 'col': 2, 'is_header': False},
        ]
    }
    
    # 3. 개념 추출
    result = distiller.distill(table_structure, context_text="Tech company revenues")
    
    print(f"\n✓ Headers detected: {result['num_headers']}")
    print(f"✓ Data cells detected: {result['num_cells']}")
    print(f"✓ Semantic concepts shape: {result['semantic_concepts'].shape}")
    print(f"✓ Structural concepts shape: {result['structural_concepts'].shape}")
    print(f"✓ Contextual concepts shape: {result['contextual_concepts'].shape}")
    
    if result['attention_weights'] is not None:
        print(f"✓ Attention weights computed: {len(result['attention_weights'])} cells")
    
    # 검증
    assert result['num_headers'] == 3
    assert result['num_cells'] == 6
    assert result['semantic_concepts'].shape == (6, 768)
    assert result['structural_concepts'].shape == (3, 768)
    
    print("\n" + "="*70)
    print("✅ All integration tests passed!")
    print("="*70)


if __name__ == '__main__':
    # 통합 테스트 실행
    test_integration()
    print("\n✅ Cell-Header Attention module is ready!")

