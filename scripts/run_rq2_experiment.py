"""
RQ2 Structural Error Analysis Experiment
Compares Baseline vs HyCoRAG on structural understanding metrics.
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
import os

print("="*70)
print("RQ2: Structural Error Analysis")
print("="*70)

# Load dataset
print("\nLoading RealHiTBench (20 samples for analysis)...")
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=20)
print(f"Loaded {len(dataset)} samples")

# Initialize models
print("\nInitializing models...")
retriever = BaselineRetriever()
llm = LocalLLMClient(model_name="Qwen/Qwen2.5-3B-Instruct")
encoder = TableEncoder(hidden_dim=768)
distiller = HybridConceptDistiller(hidden_dim=768, table_encoder=encoder)
router = ConceptRouter(hidden_dim=768)

# Create pipelines
baseline_pipeline = BaselineRAGPipeline(retriever, llm)
hycorag_pipeline = HyCoRAGPipeline(retriever, llm, distiller, router, mode="full")

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

# Run experiments
print("\n" + "="*70)
print("Running Baseline...")
print("="*70)

baseline_results = []
for i, sample in enumerate(dataset[:10]):  # First 10 for speed
    print(f"Sample {i+1}/10: {sample.question[:60]}...")
    result = baseline_pipeline.run(sample.question, top_k=3)
    baseline_results.append(result.answer)

print("\n" + "="*70)
print("Running HyCoRAG...")
print("="*70)

hycorag_results = []
for i, sample in enumerate(dataset[:10]):
    print(f"Sample {i+1}/10: {sample.question[:60]}...")
    result = hycorag_pipeline.run(sample.question, top_k=3)
    hycorag_results.append(result.answer)

# Analyze structural coverage
print("\n" + "="*70)
print("Analyzing Structural Coverage...")
print("="*70)

baseline_metrics = []
hycorag_metrics = []

for i, sample in enumerate(dataset[:10]):
    # Extract header paths
    html_path = f"RealHiTBench/html/{sample.metadata['file']}.html"
    if os.path.exists(html_path):
        header_paths = extract_header_hierarchy(html_path)
        cells = sample.metadata.get('structure', {}).get('cells', [])
        
        b_metrics = analyze_structural_coverage(
            baseline_results[i],
            header_paths,
            cells
        )
        h_metrics = analyze_structural_coverage(
            hycorag_results[i],
            header_paths,
            cells
        )
        
        baseline_metrics.append(b_metrics)
        hycorag_metrics.append(h_metrics)

# Aggregate results
def avg_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

baseline_avg = avg_metrics(baseline_metrics)
hycorag_avg = avg_metrics(hycorag_metrics)

print("\n" + "="*70)
print("RQ2 RESULTS: Structural Error Analysis")
print("="*70)

print("\nBaseline Metrics:")
for k, v in baseline_avg.items():
    print(f"  {k}: {v:.3f}")

print("\nHyCoRAG Metrics:")
for k, v in hycorag_avg.items():
    print(f"  {k}: {v:.3f}")

print("\nImprovement (HyCoRAG - Baseline):")
for k in baseline_avg.keys():
    improvement = hycorag_avg[k] - baseline_avg[k]
    pct = (improvement / max(baseline_avg[k], 0.001)) * 100
    print(f"  {k}: {improvement:+.3f} ({pct:+.1f}%)")

print("\n" + "="*70)
print("RQ2 Analysis Complete!")
print("="*70)
