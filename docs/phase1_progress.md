# Phase 1: Cell-Header Attention 구현 진행 상황

## ✅ 완료된 작업

### 1. 핵심 모듈 구현 (100%)

- [x] `HeaderHierarchyEncoder` - 헤더 계층 구조 인코딩
  - Transformer 기반 계층 관계 학습
  - 레벨 임베딩으로 부모-자식 관계 표현
  - 파일: `hycorag/models/cell_header_encoder.py`

- [x] `CellHeaderAttentionEncoder` - 셀-헤더 관계 모델링
  - Multi-head attention으로 셀이 헤더에 주목
  - 위치 인코딩 (행/열) 추가
  - 셀 = 값 + 헤더 컨텍스트 + 위치

- [x] `CellHeaderAwareDistiller` - 통합 Distiller
  - 기존 `HybridConceptDistiller` 대체
  - 헤더와 셀 자동 추출
  - 셀-헤더 매핑 자동 생성

### 2. 테스트 및 검증 (100%)

- [x] 단위 테스트 작성
  - `HeaderHierarchyEncoder` 테스트
  - `CellHeaderAttentionEncoder` 테스트
  - `CellHeaderAwareDistiller` 통합 테스트

- [x] 통합 테스트 실행 성공
  ```
  ✓ Headers detected: 3
  ✓ Data cells detected: 6
  ✓ Semantic concepts shape: torch.Size([6, 768])
  ✓ Structural concepts shape: torch.Size([3, 768])
  ✓ Attention weights computed: 6 cells
  ```

### 3. 측정 도구 준비 (100%)

- [x] 셀 정확도 측정 스크립트
  - 파일: `scripts/test_cell_accuracy.py`
  - 답변에서 셀 값 추출 로직
  - Baseline vs Cell-Header Attention 비교

## 🔄 진행 중인 작업

### 4. RAG 파이프라인 통합 (0%)

**다음 단계**:

1. `HyCoRAGPipeline`에 `CellHeaderAwareDistiller` 통합
2. 기존 `HybridConceptDistiller` 대체 옵션 추가
3. Ablation mode 추가: `"cell_header_attention"`

**예상 코드**:
```python
# hycorag/rag/pipeline.py 수정
class HyCoRAGPipeline(BaseRAGPipeline):
    def __init__(
        self, 
        mode: Literal["baseline", "distill_only", "full", "cell_header_attention"] = "full",
        use_cell_header_attention: bool = False  # NEW
    ):
        if use_cell_header_attention:
            from ..models.cell_header_encoder import CellHeaderAwareDistiller
            self.distiller = CellHeaderAwareDistiller(
                hidden_dim=768,
                text_encoder=encoder
            )
        else:
            self.distiller = HybridConceptDistiller(...)
```

### 5. 성능 검증 (0%)

**목표**: 셀 정확도 8.2% → 35%+

**검증 계획**:
- [ ] 50개 샘플로 빠른 검증
- [ ] Baseline vs Cell-Header Attention 비교
- [ ] 통계적 유의성 확인 (나중에)

## 📋 남은 작업 (우선순위)

### 우선순위 1: RAG 통합 (1-2일)

- [ ] `pipeline.py` 수정
- [ ] `use_cell_header_attention` 플래그 추가
- [ ] 기존 코드 호환성 유지

### 우선순위 2: 실험 실행 (1-2일)

- [ ] 50개 샘플로 초기 검증
- [ ] 셀 정확도 측정
- [ ] 헤더 인식 유지 확인

### 우선순위 3: 결과 분석 (1일)

- [ ] 성공 사례 분석
- [ ] 실패 사례 분석
- [ ] 개선 방향 도출

### 우선순위 4: 반복 개선 (필요시)

- [ ] Attention 헤드 수 조정
- [ ] 계층 인코더 레이어 수 조정
- [ ] 융합 방식 개선

## 🎯 Phase 1 성공 기준

| 지표 | 현재 | 목표 | 상태 |
|:-----|:-----|:-----|:-----|
| **셀 정확도** | 8.2% | 35%+ | ⏳ 측정 대기 |
| **헤더 인식** | 65% | 60%+ (유지) | ⏳ 측정 대기 |
| **모듈 구현** | ✅ | ✅ | ✅ 완료 |
| **테스트 통과** | ✅ | ✅ | ✅ 완료 |

## 📊 예상 타임라인

```
Day 1-2 (완료): 모듈 구현 + 테스트 ✅
Day 3-4 (다음): RAG 통합 + 초기 검증
Day 5-6 (예정): 성능 측정 + 분석
Day 7 (예정): 반복 개선 (필요시)
```

## 🚀 즉시 실행 가능한 다음 단계

### 옵션 A: RAG 파이프라인 통합 (권장)

```bash
# 1. pipeline.py 수정
# 2. 통합 테스트
python scripts/test_hybrid_strategy.py --use-cell-header-attention

# 3. 셀 정확도 측정
python scripts/test_cell_accuracy.py
```

### 옵션 B: 독립 검증 먼저

```bash
# Cell-Header Attention만 독립적으로 검증
python scripts/test_cell_accuracy.py

# 결과 확인 후 RAG 통합 결정
```

## 💡 핵심 인사이트

### 구현 완료된 독창적 기여

1. **명시적 셀-헤더 관계 모델링**
   - TabRAG: VLM 기반 독립 처리
   - HyCoRAG: Multi-head attention으로 관계 학습

2. **계층적 헤더 인코딩**
   - 부모-자식 관계를 Transformer로 명시적 학습
   - 레벨 임베딩으로 계층 정보 보존

3. **위치 인식 셀 표현**
   - 행/열 위치 임베딩
   - 셀 값 + 헤더 컨텍스트 + 위치 융합

### 예상 효과

- **셀 정확도**: 8.2% → 35%+ (4배 개선 목표)
- **헤더 인식**: 65% 유지 또는 향상
- **독창성**: TabRAG/T-RAG 대비 명확한 차별화

## ⚠️ 주의사항

1. **텍스트 인코더 공유**
   - `CellHeaderAwareDistiller`는 `TableEncoder`의 `encode_text` 메서드 사용
   - 통합 시 인코더 인스턴스 공유 필요

2. **테이블 구조 요구사항**
   - `cells` 리스트 필수
   - 각 셀에 `is_header`, `row`, `col` 필요
   - RealHiTBench 데이터셋 구조 확인 필요

3. **성능 오버헤드**
   - Attention 연산 추가로 약간의 지연 시간 증가 예상
   - 배치 처리로 최적화 가능

## 📝 다음 회의 안건

1. RAG 통합 방식 결정
   - 기존 코드 대체 vs 옵션 추가
   - Ablation study 설계

2. 성능 목표 재확인
   - 35% 달성 가능성
   - 실패 시 대안 (헤더만 집중)

3. 타임라인 조정
   - 1주 vs 2주
   - 다른 작업과의 우선순위
