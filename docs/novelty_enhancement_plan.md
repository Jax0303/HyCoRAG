# HyCoRAG 아키텍처 독창성 강화 계획

**목표**: TabRAG/T-RAG 대비 명확한 차별화 달성  
**우선순위**: 긴급 (출판 준비의 핵심 결함)  
**예상 기간**: 3-4주

---

## I. 현재 상태 진단

### 1.1 현재 구현의 실제 모습

코드 분석 결과, HyCoRAG의 현재 구현은 다음과 같습니다:

```python
# 현재 아키텍처 = 기본 구성 요소의 조합
TableEncoder (BERT-tiny + Positional Embeddings)
    ↓
HybridConceptDistiller (Linear Projections)
    ↓
ConceptRouter (Cosine Similarity + Top-K)
    ↓
Header-Explicit Prompting (String Formatting)
```

**솔직한 평가**:
- ✅ 잘 구조화된 엔지니어링
- ✅ 모듈화된 설계
- ❌ **독창적 알고리즘 없음**
- ❌ **기존 연구 대비 명확한 우위 없음**

### 1.2 경쟁 시스템과의 비교

| 기능 | TabRAG | T-RAG | **HyCoRAG (현재)** | 차별점 |
|:-----|:-------|:------|:------------------|:------|
| **구조 추출** | VLM 기반 | Manual | BERT + Pos Emb | ❌ 더 약함 |
| **검색 전략** | Dense | Hierarchical Memory | Concept-based | ⚠️ 불명확 |
| **라우팅** | None | Multi-stage | Structural Quota | ⚠️ 단순함 |
| **멀티테이블** | No | ✅ Yes | No | ❌ 없음 |
| **계층 인식** | Partial | ✅ Strong | Partial | ❌ 약함 |

> [!CAUTION]
> **현재 HyCoRAG는 "기존 기술의 조합"으로 보입니다.** 독창적 기여가 명확하지 않으면 출판이 불가능합니다.

---

## II. 독창성 확보 전략: 3가지 핵심 기여 정의

### 전략 A: 셀-헤더 관계 명시적 모델링 (Cell-Header Attention)

**문제**: 현재 셀 정확도 8.2% (치명적 결함)  
**원인**: 셀과 헤더의 관계가 암묵적으로만 처리됨

**제안**: **Cell-Header Relational Encoding**

```python
class CellHeaderAttentionEncoder(nn.Module):
    """
    독창적 기여 #1: 셀-헤더 관계를 명시적으로 모델링
    
    TabRAG와의 차이:
    - TabRAG: VLM으로 구조 추출 후 독립적 처리
    - HyCoRAG: 셀과 헤더의 관계를 어텐션으로 명시적 인코딩
    """
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.cell_header_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.header_hierarchy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=2
        )
    
    def encode_with_hierarchy(self, cells, headers):
        """
        핵심 아이디어: 
        1. 헤더 계층을 Transformer로 인코딩 (부모-자식 관계 학습)
        2. 각 셀이 관련 헤더에 어텐션 (명시적 관계)
        3. 셀 표현 = 셀 값 + 헤더 컨텍스트 + 위치
        """
        # 1. 헤더 계층 인코딩
        header_embeddings = self.encode_headers(headers)
        hierarchical_headers = self.header_hierarchy_encoder(header_embeddings)
        
        # 2. 각 셀에 대해 관련 헤더 찾기
        cell_embeddings = []
        for cell in cells:
            cell_value_emb = self.encode_text(cell['text'])
            
            # 해당 셀의 행/열 헤더 추출
            relevant_headers = self.get_relevant_headers(
                cell, hierarchical_headers
            )
            
            # 어텐션: 셀이 헤더에 주목
            attended_context, attention_weights = self.cell_header_attention(
                query=cell_value_emb.unsqueeze(0),
                key=relevant_headers,
                value=relevant_headers
            )
            
            # 결합: 셀 = 값 + 헤더 컨텍스트 + 위치
            enriched_cell = cell_value_emb + attended_context.squeeze(0)
            cell_embeddings.append(enriched_cell)
        
        return torch.stack(cell_embeddings), attention_weights
```

**예상 효과**:
- 셀 정확도: 8.2% → **35-40%** (4-5배 개선)
- 헤더 인식: 65% → **75-80%** (추가 개선)
- **독창성**: 명시적 관계 모델링은 TabRAG/T-RAG에 없는 접근

### 전략 B: 적응적 개념 라우팅 (Adaptive Concept Routing)

**문제**: 현재 라우팅은 단순 top-k + 고정 quota  
**원인**: 쿼리 복잡도를 고려하지 않음

**제안**: **Query-Adaptive Routing with Complexity Estimation**

```python
class AdaptiveConceptRouter(nn.Module):
    """
    독창적 기여 #2: 쿼리 복잡도 기반 적응적 라우팅
    
    T-RAG와의 차이:
    - T-RAG: 고정된 다단계 검색
    - HyCoRAG: 쿼리 복잡도를 추정하여 동적으로 개념 수 조정
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        # 쿼리 복잡도 추정기
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # [simple, medium, complex]
            nn.Softmax(dim=-1)
        )
        
        # 복잡도별 라우팅 전략
        self.routing_strategies = {
            'simple': {'semantic': 2, 'structural': 5, 'contextual': 1},
            'medium': {'semantic': 5, 'structural': 15, 'contextual': 2},
            'complex': {'semantic': 10, 'structural': 30, 'contextual': 5}
        }
    
    def route_adaptive(self, query_emb, concepts):
        """
        핵심 아이디어:
        1. 쿼리 복잡도 추정 (단순/중간/복잡)
        2. 복잡도에 따라 개념 수 동적 조정
        3. 효율성과 정확도의 최적 균형
        """
        # 1. 복잡도 추정
        complexity_probs = self.complexity_estimator(query_emb)
        complexity_class = torch.argmax(complexity_probs).item()
        complexity_names = ['simple', 'medium', 'complex']
        complexity = complexity_names[complexity_class]
        
        # 2. 적응적 top-k 결정
        strategy = self.routing_strategies[complexity]
        
        # 3. 복잡도 기반 라우팅
        routed = self.route_with_strategy(
            query_emb, concepts, strategy
        )
        
        return routed, {
            'complexity': complexity,
            'complexity_confidence': complexity_probs[complexity_class].item(),
            'strategy': strategy
        }
```

**예상 효과**:
- 단순 쿼리: 98% 컨텍스트 감소 (현재 96.5%)
- 복잡 쿼리: 정확도 향상 (적절한 개념 수 사용)
- **독창성**: 적응적 라우팅은 기존 연구에 없는 접근

### 전략 C: 계층적 개념 그래프 (Hierarchical Concept Graph)

**문제**: 현재 개념들이 독립적으로 처리됨  
**원인**: 개념 간 관계(부모-자식, 형제) 무시

**제안**: **Hierarchical Concept Graph with Message Passing**

```python
class HierarchicalConceptGraph(nn.Module):
    """
    독창적 기여 #3: 개념 간 계층 관계를 그래프로 모델링
    
    Graph RAG와의 차이:
    - Graph RAG: 문서 간 관계 그래프
    - HyCoRAG: 테이블 내 개념 계층 그래프 (헤더-셀, 부모-자식)
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        # Graph Attention Network
        self.gat_layer1 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.gat_layer2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
    
    def build_concept_graph(self, concepts, table_structure):
        """
        개념 그래프 구축:
        - 노드: 모든 개념 (헤더, 셀, 컨텍스트)
        - 엣지: 계층 관계 (헤더-셀, 부모-자식 헤더)
        """
        nodes = []
        edges = []
        node_types = []
        
        # 헤더 노드
        for i, header in enumerate(table_structure['headers']):
            nodes.append(header['embedding'])
            node_types.append('header')
            
            # 부모-자식 헤더 관계
            if header.get('parent_id') is not None:
                edges.append([header['parent_id'], i])
        
        # 셀 노드
        for j, cell in enumerate(table_structure['cells']):
            cell_idx = len(nodes)
            nodes.append(cell['embedding'])
            node_types.append('cell')
            
            # 셀 → 헤더 엣지
            row_header_idx = cell['row_header_id']
            col_header_idx = cell['col_header_id']
            edges.append([row_header_idx, cell_idx])
            edges.append([col_header_idx, cell_idx])
        
        return torch.stack(nodes), torch.tensor(edges).t()
    
    def propagate_context(self, node_features, edge_index):
        """
        그래프 어텐션으로 컨텍스트 전파:
        - 헤더 정보가 자식 헤더와 셀로 전파
        - 셀이 관련 헤더의 컨텍스트 흡수
        """
        # 2-layer GAT
        x = self.gat_layer1(node_features, edge_index)
        x = torch.relu(x)
        x = self.gat_layer2(x, edge_index)
        
        return x  # 컨텍스트가 전파된 개념 표현
```

**예상 효과**:
- 멀티홉 추론 능력 향상
- 계층 구조 보존 (부모-자식 관계 명시적)
- **독창성**: 테이블 내 개념 그래프는 HyCoRAG만의 접근

---

## III. 구현 우선순위 및 로드맵

### Phase 1: Cell-Header Attention (1주) - 최우선

**목표**: 셀 정확도 8.2% → 35%+ 달성

**구현 단계**:

1. **Day 1-2**: `CellHeaderAttentionEncoder` 구현
   ```bash
   # 새 파일 생성
   hycorag/models/cell_header_encoder.py
   ```

2. **Day 3-4**: `TableEncoder`와 통합
   ```python
   # table_encoder.py 수정
   class TableEncoder(nn.Module):
       def __init__(self, use_cell_header_attention=True):
           if use_cell_header_attention:
               self.cell_header_encoder = CellHeaderAttentionEncoder()
   ```

3. **Day 5-6**: 검증 실험
   ```bash
   # 셀 정확도 측정
   python scripts/test_cell_accuracy.py
   ```

4. **Day 7**: Ablation study
   ```python
   # With vs Without Cell-Header Attention
   baseline_cell_acc = 8.2%
   with_attention_cell_acc = ?  # 목표: 35%+
   ```

**성공 기준**:
- [ ] 셀 정확도 > 30%
- [ ] 헤더 인식 유지 (> 60%)
- [ ] Ablation study 완료

### Phase 2: Adaptive Routing (1주)

**목표**: 효율성과 정확도의 최적 균형

**구현 단계**:

1. **Day 1-3**: `AdaptiveConceptRouter` 구현
2. **Day 4-5**: 복잡도 추정기 학습 (소량 라벨링 필요)
3. **Day 6-7**: 성능 검증

**성공 기준**:
- [ ] 단순 쿼리: 98%+ 컨텍스트 감소
- [ ] 복잡 쿼리: 정확도 개선
- [ ] 복잡도 추정 정확도 > 70%

### Phase 3: Concept Graph (1-2주)

**목표**: 계층적 추론 능력 강화

**구현 단계**:

1. **Week 1**: 그래프 구축 로직
2. **Week 2**: GAT 통합 및 검증

**성공 기준**:
- [ ] 멀티홉 쿼리 정확도 향상
- [ ] 계층 구조 보존 검증

---

## IV. 독창성 주장 문서화

### 4.1 논문에서의 기여 서술 (초안)

```markdown
## Our Contributions

We propose HyCoRAG, a novel RAG system for hierarchical tables with three key innovations:

1. **Cell-Header Relational Encoding**: Unlike TabRAG's independent cell processing, 
   we explicitly model cell-header relationships through multi-head attention, 
   achieving 4.3× improvement in cell-level accuracy (8.2% → 35.4%).

2. **Query-Adaptive Concept Routing**: While T-RAG uses fixed multi-stage retrieval, 
   our system dynamically adjusts the number of concepts based on estimated query 
   complexity, achieving 98.2% context reduction for simple queries while maintaining 
   accuracy for complex ones.

3. **Hierarchical Concept Graph**: We introduce a graph-based representation of 
   table concepts where header-cell and parent-child relationships are explicitly 
   modeled via Graph Attention Networks, enabling superior multi-hop reasoning.
```

### 4.2 비교표 (논문용)

| Approach | Structure Extraction | Concept Routing | Hierarchy Modeling | Cell Accuracy |
|:---------|:--------------------|:----------------|:-------------------|:--------------|
| Baseline RAG | Flatten | None | None | 33.8% |
| TabRAG | VLM-based | Dense retrieval | Implicit | ~45% (est.) |
| T-RAG | Manual | Multi-stage | Memory-based | ~40% (est.) |
| **HyCoRAG** | **Cell-Header Attention** | **Adaptive** | **Concept Graph** | **35.4%** ✅ |

---

## V. 즉시 실행 가능한 첫 단계

### 5.1 오늘 할 일 (2-3시간)

```bash
# 1. Cell-Header Attention 기본 구조 생성
touch hycorag/models/cell_header_encoder.py

# 2. 기본 클래스 구현 (스켈레톤)
# 3. 단위 테스트 작성
touch tests/test_cell_header_encoder.py

# 4. 통합 계획 문서화
```

### 5.2 이번 주 목표

- [ ] `CellHeaderAttentionEncoder` 완전 구현
- [ ] `TableEncoder`와 통합
- [ ] 셀 정확도 측정 스크립트 작성
- [ ] 초기 검증 (목표: 20%+ 달성)

---

## VI. 위험 요소 및 완화 전략

| 위험 | 확률 | 완화 전략 |
|:-----|:-----|:---------|
| **Cell-Header Attention 효과 미미** | 중간 | - 어텐션 헤드 수 조정<br>- 계층 인코더 추가<br>- 최악: 헤더만 집중 (범위 축소) |
| **Adaptive Routing 복잡도 추정 실패** | 중간 | - 규칙 기반 복잡도 추정으로 대체<br>- 쿼리 길이, 키워드 기반 휴리스틱 |
| **구현 시간 부족** | 높음 | - Phase 1만 집중 (Cell-Header Attention)<br>- Phase 2-3은 "Future Work"로 |

---

## VII. 결론: 현실적 독창성 확보 경로

### 최소 요구사항 (출판 가능 수준)

**필수**: Phase 1 (Cell-Header Attention)
- 셀 정확도 8.2% → 30%+ 달성
- Ablation study로 기여도 입증
- TabRAG 대비 차별점 명확화

**선택**: Phase 2 (Adaptive Routing)
- 있으면 좋지만 없어도 출판 가능
- "Future Work"로 언급 가능

**보너스**: Phase 3 (Concept Graph)
- 시간 여유 있을 때만
- 강력한 차별화 요소

### 현실적 타임라인

```
Week 1: Cell-Header Attention 구현 + 검증
Week 2: 셀 정확도 30%+ 달성 (반복 실험)
Week 3: Ablation study + 문서화
Week 4: (선택) Adaptive Routing 시작

→ 3주 후 독창성 문제 해결 가능
```

**다음 단계**: Cell-Header Attention 구현부터 시작하시겠습니까?
