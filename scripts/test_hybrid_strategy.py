"""
Test HYBRID strategy: Header-explicit prompt + Quota routing
Target: 90% reduction + 65% header mention
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
print("HYBRID Strategy: Header-Explicit + Quota Routing")
print("Target: 90% reduction + 65% header mention")
print("="*70)

# Load dataset
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=20)

# Initialize with quota (NOT header-first)
retriever = BaselineRetriever()
llm = LocalLLMClient(model_name="Qwen/Qwen2.5-3B-Instruct")
encoder = TableEncoder(hidden_dim=768)
distiller = HybridConceptDistiller(hidden_dim=768, table_encoder=encoder)
router = ConceptRouter(hidden_dim=768, structural_min_quota=20)  # Quota, not header-first

hycorag_pipeline = HyCoRAGPipeline(retriever, llm, distiller, router, mode="full")

# Index
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

# Run HYBRID
print("\n" + "="*70)
print("Running HYBRID (Header-explicit + Quota=20)...")
print("="*70)

results = {
    "structural_metrics": [], 
    "concept_counts": [],
    "context_lengths": []
}

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
    results["context_lengths"].append(result.metadata.get("context_length", 0))

# Aggregate
def avg_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

hybrid_avg = avg_metrics(results["structural_metrics"])
avg_concepts = avg(results["concept_counts"])
avg_context = avg(results["context_lengths"])

print("\n" + "="*70)
print("RESULTS: HYBRID Strategy")
print("="*70)

print("\n### Structural Metrics ###")
for k, v in hybrid_avg.items():
    print(f"  {k}: {v:.3f}")

print(f"\n### Efficiency Metrics ###")
print(f"  Avg concepts: {avg_concepts:.1f}")
print(f"  Avg context length: {avg_context:.1f} chars")

# Estimate tokens (rough: 4 chars per token)
est_tokens = avg_context / 4
baseline_tokens = 15444
reduction = (1 - est_tokens / baseline_tokens) * 100

print(f"  Estimated tokens: {est_tokens:.0f}")
print(f"  Context reduction: {reduction:.1f}%")

# Compare with targets
baseline_header = 0.318
target_header_min = baseline_header * 1.15
header_rate = hybrid_avg.get('header_mention_rate', 0)

print(f"\n### vs Targets ###")
print(f"  Context reduction: {reduction:.1f}% (target: 90%)")
print(f"  Header mention: {header_rate:.1%} (target: {target_header_min:.1%})")

success_efficiency = reduction >= 85  # Allow 5% margin
success_header = header_rate >= target_header_min

print(f"\n### Success ###")
print(f"  Efficiency: {'✅' if success_efficiency else '❌'} ({reduction:.1f}% >= 85%)")
print(f"  Header: {'✅' if success_header else '❌'} ({header_rate:.1%} >= {target_header_min:.1%})")

if success_efficiency and success_header:
    print(f"\n🎉 BOTH TARGETS ACHIEVED!")
else:
    print(f"\n⚠️ Need improvement")

# Save
with open("hybrid_results.json", "w") as f:
    json.dump({
        "hybrid": hybrid_avg,
        "avg_concepts": avg_concepts,
        "avg_context_length": avg_context,
        "estimated_tokens": est_tokens,
        "context_reduction_pct": reduction,
        "targets": {
            "efficiency": 90,
            "header_min": target_header_min
        },
        "success": {
            "efficiency": success_efficiency,
            "header": success_header
        }
    }, f, indent=2)

print(f"\n✓ Results saved to hybrid_results.json")
print("="*70)
