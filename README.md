# HyCoRAG: Hybrid Concept-Aware RAG for Complex Table Understanding

**Status**: Research Prototype - Infrastructure Complete, Validation In Progress

HyCoRAG is a concept-based RAG pipeline for complex hierarchical tables. It decomposes tables into **semantic, structural, and contextual concepts** and routes query-relevant subsets to reduce context length.

## 🎯 Research Questions

### RQ1: Context Efficiency + Performance Preservation
> Can hybrid concept decomposition and routing reduce context length while **maintaining QA performance**?

**Current Status**: 
- ✅ Context reduction demonstrated (97% on RealHiTBench)
- ⚠️ Performance preservation not yet validated (LLM integration pending large-scale eval)

### RQ2: Structural Error Reduction
> Can explicit structural concepts (header paths, cell spans) reduce structural errors compared to flattened approaches?

**Current Status**:
- ✅ Metrics implemented (header path match, unit match)
- ❌ Baseline vs HyCoRAG comparison not yet conducted

### Future: Self-Evolving Concept Space (Planned)
> Can the system adapt concept representations across domain shifts without catastrophic forgetting?

**Current Status**: 
- ✅ Adapter architecture designed
- ❌ Sequential domain training not yet implemented

## 📊 Current Results

### Context Reduction (RealHiTBench, 10 samples)

| Mode | Context Length | Concept Count | Reduction |
|:-----|:---------------|:--------------|:----------|
| Baseline (Flatten) | 15,444 tokens | - | - |
| Distill Only | 476 tokens | 5,559 | 96.9% ↓ |
| **Full (Routing)** | **461 tokens** | **15** | **97.0% ↓** |

**Interpretation**: Concept routing achieves 33× context compression. **Performance impact not yet measured.**

### Adapter Efficiency (Continual Learning Demo)

| Component | Parameters | Overhead |
|:----------|:-----------|:---------|
| Base Encoder | 5.0M | - |
| Domain Adapters (2×) | 201K | **4.03%** |

## 🛠️ Implementation Status

### Core Components ✅
- [x] **TableEncoder**: BERT-tiny + learnable structural embeddings
- [x] **Concept Decomposition**: Granular semantic/structural/contextual separation
- [x] **Concept Router**: Query-conditioned Top-k selection
- [x] **Ablation Framework**: Baseline / Distill-only / Full modes
- [x] **RealHiTBench Integration**: HTML parsing with structure preservation

### Evaluation Infrastructure ✅
- [x] Multi-stage pipeline (Text → Multimodal → Hierarchical)
- [x] Unified `QAExample` schema across datasets
- [x] Structural metrics (header path, unit match)
- [x] Local LLM integration (Qwen2.5-3B)

### Pending Validation ⚠️
- [ ] **RQ1 Full Validation**: EM/F1 comparison (Baseline vs HyCoRAG, n≥100)
- [ ] **RQ2 Quantification**: Structural error rate measurement
- [ ] **Distillation Objective**: KD-style loss for true knowledge transfer
- [ ] **Self-Evolving Training**: Sequential domain adaptation with forgetting metrics

## 🚀 Usage

### Quick Start
```bash
# Install dependencies
pip install -e .
pip install transformers accelerate beautifulsoup4

# Clone RealHiTBench dataset
git clone https://huggingface.co/datasets/spzy/RealHiTBench

# Run experiments
python scripts/run_hycorag.py --stage table_hier --dataset realhitbench --mode baseline --max_samples 10
python scripts/run_hycorag.py --stage table_hier --dataset realhitbench --mode full --max_samples 10
```

### Experiment Stages
- **Stage 0**: Text QA (HotpotQA, NQ Open) - baseline debugging
- **Stage 1**: Multimodal Tables (MMTab, ComTQA) - visual + text
- **Stage 2**: Hierarchical Tables (RealHiTBench) - complex structure
- **Stage 3**: Interleaved (RAG-IGBench) - mixed modality

## 📂 Project Structure
```
hycorag/
├── data/           # Dataset adapters (HotpotQA, MMTab, RealHiTBench, etc.)
├── models/         # TableEncoder, ConceptDistiller, ConceptRouter
├── rag/            # Pipeline, Retriever, LLM clients
├── evaluation/     # Experiment runners, metrics, structure analyzer
scripts/            # CLI entry points, debug tools
```

## 🔬 Next Steps (Priority Order)

### 1. Complete RQ1 Validation
**Goal**: Prove context reduction doesn't hurt performance

- [ ] Run 100+ sample experiments (Baseline vs HyCoRAG)
- [ ] Measure EM/F1, context length, latency
- [ ] Statistical significance testing

### 2. RQ2 Quantification
**Goal**: Measure structural error reduction

- [ ] Extract ground truth header paths (50 samples)
- [ ] Compare header/cell/unit error rates
- [ ] Ablation: Structural concepts on/off

### 3. Strengthen "Distillation" Claim
**Goal**: Add true knowledge transfer objective

- [ ] Implement teacher-student KD loss
- [ ] Train concept representations to match LLM hidden states
- [ ] Measure concept quality (alignment, coverage)

### 4. Self-Evolving Prototype (Future Work)
**Goal**: Demonstrate continual learning

- [ ] Sequential training: HotpotQA → MMTab → RealHiTBench
- [ ] Measure forward/backward transfer
- [ ] Catastrophic forgetting analysis

## ⚠️ Known Limitations

### Current Implementation
- **Small-scale validation**: 10-20 samples per experiment
- **Dummy metrics**: EM/F1 = 0.0 due to answer normalization issues
- **Header extraction**: RealHiTBench uses `<td>` for headers, not `<th>`
- **No training loop**: Concept representations are not optimized

### Research Gaps
- **RQ1**: Context reduction shown, but performance preservation not validated
- **RQ2**: Metrics designed, but no baseline comparison conducted
- **Distillation**: Currently more "decomposition + selection" than true distillation
- **Self-Evolving**: Architecture designed, but no sequential training implemented

## 📝 Citation

```bibtex
@software{hycorag2024,
  title={HyCoRAG: Hybrid Concept-Aware RAG for Complex Table Understanding},
  author={[Your Name]},
  year={2024},
  note={Research Prototype},
  url={https://github.com/Jax0303/HyCoRAG}
}
```

## 📄 License

MIT License

---

**Honest Assessment**: This is a well-structured research prototype with solid infrastructure for RQ1/RQ2 validation. Context reduction is demonstrated, but performance preservation and structural error reduction require further experimentation. Self-evolving capabilities are planned but not yet implemented.
