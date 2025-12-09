"""
Quick RQ2 re-validation with improved routing.
Tests structural quota fix on 20 samples.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.data.datasets import RealHiTBenchDataset
from hycorag.rag.retriever import BaselineRetriever
from hycorag.rag.pipeline import HyCoRAGPipeline
from hycorag.rag.llm_client import LocalLLMClient
from hycorag.models.table_encoder import TableEncoder
from hycorag.models.concept_distill import HybridConceptDistiller
from hycorag.models.concept_router import ConceptRouter
from hycorag.evaluation.structure_analyzer import (
    extract_header_hierarchy,
    analyze_structural_coverage
)
import os
import json

print("="*70)
print("RQ2 Re-validation with Improved Routing")
print("="*70)

# Load dataset
print("\nLoading RealHiTBench (20 samples)...")
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=20)

# Initialize models with IMPROVED router
print("\nInitializing models with structural quota...")
retriever = BaselineRetriever()
llm = LocalLLMClient(model_name="Qwen/Qwen2.5-3B-Instruct")
encoder = TableEncoder(hidden_dim=768)
distiller = HybridConceptDistiller(hidden_dim=768, table_encoder=encoder)

# NEW: Router with structural minimum quota
router_improved = ConceptRouter(hidden_dim=768, structural_min_quota=20)

# Create pipeline
hycorag_pipeline = HyCoRAGPipeline(retriever, llm, distiller, router_improved, mode="full")

# Index corpus
print("\nIndexing corpus...")
corpus = {}
corpus_metadata = {}
for i, item in enumerate(dataset):
    for j, doc in enumerate(item.documents):
        doc_id = f"doc_{item.qid}_{j}"
        corpus[doc_id] = doc
        if item.metadata and 'structure' in item.metadata:
            corpus_metadata[doc_id] = item.metadata

hycorag_pipeline.retriever.index(corpus)
hycorag_pipeline.retriever._corpus_metadata = corpus_metadata

# Extract ground truth
print("\nExtracting ground truth...")
ground_truth_paths = []
valid_samples = []

for i, sample in enumerate(dataset):
    html_path = f"RealHiTBench/html/{sample.metadata['file']}.html"
    if os.path.exists(html_path):
        header_paths = extract_header_hierarchy(html_path)
        if header_paths:
            ground_truth_paths.append(header_paths)
            valid_samples.append(sample)

print(f"Valid samples: {len(valid_samples)}")

# Run improved HyCoRAG
print("\n" + "="*70)
print("Running HyCoRAG with Structural Quota...")
print("="*70)

results = {"structural_metrics": [], "concept_counts": []}

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
    
    results["structural_metrics"].append(metrics)
    results["concept_counts"].append(result.metadata.get("total_concepts", 0))

# Aggregate
def avg_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

improved_avg = avg_metrics(results["structural_metrics"])
avg_concepts = sum(results["concept_counts"]) / len(results["concept_counts"])

print("\n" + "="*70)
print("RESULTS: Improved Routing")
print("="*70)

print("\n### Structural Metrics (Improved) ###")
for k, v in improved_avg.items():
    print(f"  {k}: {v:.3f}")

print(f"\n### Concept Stats ###")
print(f"  Avg concepts routed: {avg_concepts:.1f}")

# Compare with P1 baseline results
print("\n### Comparison with Previous (P1) ###")
p1_baseline = {
    "header_mention_rate": 0.318,
    "cell_value_accuracy": 0.338
}
p1_hycorag_old = {
    "header_mention_rate": 0.076,
    "cell_value_accuracy": 0.028
}

print(f"\nHeader Mention Rate:")
print(f"  Baseline (P1): {p1_baseline['header_mention_rate']:.3f}")
print(f"  HyCoRAG Old (P1): {p1_hycorag_old['header_mention_rate']:.3f} (-76%)")
print(f"  HyCoRAG Improved: {improved_avg.get('header_mention_rate', 0):.3f}")

improvement = improved_avg.get('header_mention_rate', 0) - p1_hycorag_old['header_mention_rate']
print(f"  Improvement: {improvement:+.3f} ({improvement/p1_hycorag_old['header_mention_rate']*100:+.1f}%)")

# Save
with open("rq2_improved_results.json", "w") as f:
    json.dump({
        "improved": improved_avg,
        "avg_concepts": avg_concepts,
        "p1_comparison": {
            "baseline": p1_baseline,
            "old_hycorag": p1_hycorag_old,
            "improvement": improvement
        }
    }, f, indent=2)

print(f"\n✓ Results saved to rq2_improved_results.json")
print("\n" + "="*70)
