"""
P1: RQ2 Structural Error Quantification
Measures header path match, cell reference accuracy, unit match.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.data.datasets import RealHiTBenchDataset
from hycorag.rag.retriever import BaselineRetriever
from hycorag.rag.pipeline import BaselineRAGPipeline, HyCoRAGPipeline
from hycorag.rag.llm_client import LocalLLMClient
from hycorag.models.table_encoder import TableEncoder
from hycorag.models.concept_distill import HybridConceptDistiller
from hycorag.models.concept_router import ConceptRouter
from hycorag.evaluation.structure_analyzer import (
    extract_header_hierarchy,
    analyze_structural_coverage
)
from hycorag.evaluation.metrics import calculate_header_path_match, calculate_unit_match
import os
import json

print("="*70)
print("P1: RQ2 Structural Error Quantification")
print("="*70)

# Configuration
MAX_SAMPLES = 50
SAVE_RESULTS = True

# Load dataset
print(f"\nLoading RealHiTBench ({MAX_SAMPLES} samples)...")
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=MAX_SAMPLES)
print(f"Loaded {len(dataset)} samples")

# Initialize models
print("\nInitializing models...")
retriever_baseline = BaselineRetriever()
retriever_hycorag = BaselineRetriever()
llm = LocalLLMClient(model_name="Qwen/Qwen2.5-3B-Instruct")

encoder = TableEncoder(hidden_dim=768)
distiller = HybridConceptDistiller(hidden_dim=768, table_encoder=encoder)
router = ConceptRouter(hidden_dim=768)

# Create pipelines
baseline_pipeline = BaselineRAGPipeline(retriever_baseline, llm)
hycorag_pipeline = HyCoRAGPipeline(retriever_hycorag, llm, distiller, router, mode="full")

# Pre-index corpus
print("\nIndexing corpus...")
corpus = {}
corpus_metadata = {}
for i, item in enumerate(dataset):
    for j, doc in enumerate(item.documents):
        doc_id = f"doc_{item.qid}_{j}"
        corpus[doc_id] = doc
        if item.metadata and 'structure' in item.metadata:
            corpus_metadata[doc_id] = item.metadata

baseline_pipeline.retriever.index(corpus)
baseline_pipeline.retriever._corpus_metadata = corpus_metadata
hycorag_pipeline.retriever.index(corpus)
hycorag_pipeline.retriever._corpus_metadata = corpus_metadata

# Extract ground truth header paths
print("\nExtracting ground truth header paths...")
ground_truth_paths = []
valid_samples = []

for i, sample in enumerate(dataset):
    html_path = f"RealHiTBench/html/{sample.metadata['file']}.html"
    if os.path.exists(html_path):
        header_paths = extract_header_hierarchy(html_path)
        if header_paths:  # Only include if we extracted headers
            ground_truth_paths.append(header_paths)
            valid_samples.append(sample)
            print(f"  [{i+1}/{len(dataset)}] {sample.metadata['file']}: {len(header_paths)} cells")

print(f"\nValid samples with headers: {len(valid_samples)}/{len(dataset)}")

# Run experiments
print("\n" + "="*70)
print("Running Baseline...")
print("="*70)

baseline_results = {"answers": [], "structural_metrics": []}

for i, sample in enumerate(valid_samples):
    print(f"[{i+1}/{len(valid_samples)}] {sample.question[:60]}...")
    result = baseline_pipeline.run(sample.question, top_k=3)
    
    # Structural analysis
    cells = sample.metadata.get('structure', {}).get('cells', [])
    metrics = analyze_structural_coverage(
        result.answer,
        ground_truth_paths[i],
        cells
    )
    
    baseline_results["answers"].append(result.answer)
    baseline_results["structural_metrics"].append(metrics)

print("\n" + "="*70)
print("Running HyCoRAG...")
print("="*70)

hycorag_results = {"answers": [], "structural_metrics": []}

for i, sample in enumerate(valid_samples):
    print(f"[{i+1}/{len(valid_samples)}] {sample.question[:60]}...")
    result = hycorag_pipeline.run(sample.question, top_k=3)
    
    # Structural analysis
    cells = sample.metadata.get('structure', {}).get('cells', [])
    metrics = analyze_structural_coverage(
        result.answer,
        ground_truth_paths[i],
        cells
    )
    
    hycorag_results["answers"].append(result.answer)
    hycorag_results["structural_metrics"].append(metrics)

# Aggregate metrics
print("\n" + "="*70)
print("P1 RESULTS: RQ2 Structural Error Analysis")
print("="*70)

def avg_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

baseline_avg = avg_metrics(baseline_results["structural_metrics"])
hycorag_avg = avg_metrics(hycorag_results["structural_metrics"])

print("\n### Baseline Structural Metrics ###")
for k, v in baseline_avg.items():
    print(f"  {k}: {v:.3f}")

print("\n### HyCoRAG Structural Metrics ###")
for k, v in hycorag_avg.items():
    print(f"  {k}: {v:.3f}")

print("\n### Improvement (HyCoRAG - Baseline) ###")
for k in baseline_avg.keys():
    improvement = hycorag_avg[k] - baseline_avg[k]
    pct = (improvement / max(baseline_avg[k], 0.001)) * 100
    print(f"  {k}: {improvement:+.3f} ({pct:+.1f}%)")

# Calculate error rates
baseline_error_rate = 1 - baseline_avg.get("header_mention_rate", 0)
hycorag_error_rate = 1 - hycorag_avg.get("header_mention_rate", 0)
error_reduction = (baseline_error_rate - hycorag_error_rate) / max(baseline_error_rate, 0.001) * 100

print(f"\n### Error Reduction ###")
print(f"  Header error rate: {baseline_error_rate:.1%} → {hycorag_error_rate:.1%}")
print(f"  Relative reduction: {error_reduction:.1f}%")

# Save results
if SAVE_RESULTS:
    results = {
        "baseline": baseline_avg,
        "hycorag": hycorag_avg,
        "improvement": {k: hycorag_avg[k] - baseline_avg[k] for k in baseline_avg.keys()},
        "error_reduction_pct": error_reduction,
        "config": {
            "max_samples": MAX_SAMPLES,
            "valid_samples": len(valid_samples),
            "dataset": "RealHiTBench"
        }
    }
    
    with open("p1_rq2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to p1_rq2_results.json")

print("\n" + "="*70)
print("P1 Validation Complete!")
print("="*70)
