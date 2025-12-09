"""
P0: Large-scale RQ1 validation experiment
Compares Baseline vs HyCoRAG on 100 samples with proper metrics.
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
from hycorag.evaluation.metrics import calculate_em, calculate_f1
import time
import json

print("="*70)
print("P0: RQ1 Complete Validation (100 Samples)")
print("="*70)

# Configuration
MAX_SAMPLES = 100
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

# Run Baseline
print("\n" + "="*70)
print("Running Baseline Pipeline...")
print("="*70)

baseline_results = {
    "answers": [],
    "em_scores": [],
    "f1_scores": [],
    "context_lengths": [],
    "latencies": []
}

for i, sample in enumerate(dataset):
    print(f"[{i+1}/{len(dataset)}] {sample.question[:60]}...")
    
    start_time = time.time()
    result = baseline_pipeline.run(sample.question, top_k=3)
    latency = time.time() - start_time
    
    # Calculate metrics
    best_em = 0
    best_f1 = 0
    for gold_ans in sample.answers:
        em = calculate_em(result.answer, gold_ans)
        f1 = calculate_f1(result.answer, gold_ans)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    
    baseline_results["answers"].append(result.answer)
    baseline_results["em_scores"].append(best_em)
    baseline_results["f1_scores"].append(best_f1)
    baseline_results["context_lengths"].append(result.metadata.get("context_length", 0))
    baseline_results["latencies"].append(latency)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: EM={sum(baseline_results['em_scores'][-10:])/10:.3f}, "
              f"F1={sum(baseline_results['f1_scores'][-10:])/10:.3f}")

# Run HyCoRAG
print("\n" + "="*70)
print("Running HyCoRAG Pipeline...")
print("="*70)

hycorag_results = {
    "answers": [],
    "em_scores": [],
    "f1_scores": [],
    "context_lengths": [],
    "concept_counts": [],
    "latencies": []
}

for i, sample in enumerate(dataset):
    print(f"[{i+1}/{len(dataset)}] {sample.question[:60]}...")
    
    start_time = time.time()
    result = hycorag_pipeline.run(sample.question, top_k=3)
    latency = time.time() - start_time
    
    # Calculate metrics
    best_em = 0
    best_f1 = 0
    for gold_ans in sample.answers:
        em = calculate_em(result.answer, gold_ans)
        f1 = calculate_f1(result.answer, gold_ans)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    
    hycorag_results["answers"].append(result.answer)
    hycorag_results["em_scores"].append(best_em)
    hycorag_results["f1_scores"].append(best_f1)
    hycorag_results["context_lengths"].append(result.metadata.get("context_length", 0))
    hycorag_results["concept_counts"].append(result.metadata.get("total_concepts", 0))
    hycorag_results["latencies"].append(latency)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: EM={sum(hycorag_results['em_scores'][-10:])/10:.3f}, "
              f"F1={sum(hycorag_results['f1_scores'][-10:])/10:.3f}")

# Aggregate results
print("\n" + "="*70)
print("P0 RESULTS: RQ1 Complete Validation")
print("="*70)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

baseline_summary = {
    "avg_em": avg(baseline_results["em_scores"]),
    "avg_f1": avg(baseline_results["f1_scores"]),
    "avg_context_length": avg(baseline_results["context_lengths"]),
    "avg_latency": avg(baseline_results["latencies"]),
    "total_samples": len(baseline_results["em_scores"])
}

hycorag_summary = {
    "avg_em": avg(hycorag_results["em_scores"]),
    "avg_f1": avg(hycorag_results["f1_scores"]),
    "avg_context_length": avg(hycorag_results["context_lengths"]),
    "avg_concept_count": avg(hycorag_results["concept_counts"]),
    "avg_latency": avg(hycorag_results["latencies"]),
    "total_samples": len(hycorag_results["em_scores"])
}

print("\n### Baseline ###")
for k, v in baseline_summary.items():
    print(f"  {k}: {v:.3f}")

print("\n### HyCoRAG ###")
for k, v in hycorag_summary.items():
    print(f"  {k}: {v:.3f}")

print("\n### Comparison ###")
context_reduction = (1 - hycorag_summary["avg_context_length"] / baseline_summary["avg_context_length"]) * 100
em_change = hycorag_summary["avg_em"] - baseline_summary["avg_em"]
f1_change = hycorag_summary["avg_f1"] - baseline_summary["avg_f1"]
latency_change = (hycorag_summary["avg_latency"] / baseline_summary["avg_latency"] - 1) * 100

print(f"  Context Reduction: {context_reduction:.1f}%")
print(f"  EM Change: {em_change:+.3f} ({em_change/max(baseline_summary['avg_em'], 0.001)*100:+.1f}%)")
print(f"  F1 Change: {f1_change:+.3f} ({f1_change/max(baseline_summary['avg_f1'], 0.001)*100:+.1f}%)")
print(f"  Latency Change: {latency_change:+.1f}%")

# Save results
if SAVE_RESULTS:
    results = {
        "baseline": baseline_summary,
        "hycorag": hycorag_summary,
        "comparison": {
            "context_reduction_pct": context_reduction,
            "em_change": em_change,
            "f1_change": f1_change,
            "latency_change_pct": latency_change
        },
        "config": {
            "max_samples": MAX_SAMPLES,
            "dataset": "RealHiTBench",
            "llm": "Qwen2.5-3B-Instruct"
        }
    }
    
    with open("p0_rq1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to p0_rq1_results.json")

print("\n" + "="*70)
print("P0 Validation Complete!")
print("="*70)
