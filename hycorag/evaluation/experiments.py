import time
import numpy as np
from typing import Dict, Any, Literal
from tqdm import tqdm

from ..data.datasets import HotpotQADataset, NQOpenDataset, RAGIGBenchDataset, MMTabDataset, RealHiTBenchDataset, ComTQADataset
from ..rag.pipeline import BaselineRAGPipeline, HyCoRAGPipeline, LLMClient
from ..rag.retriever import BaseRetriever
from ..models.concept_distill import HybridConceptDistiller
from ..models.concept_router import ConceptRouter
from .metrics import calculate_em, calculate_f1, calculate_hit_at_k

def run_pipeline_on_dataset(pipeline, dataset, top_k=3) -> Dict[str, Any]:
    """Helper to run a pipeline on a dataset and collect metrics."""
    ems = []
    f1s = []
    context_lengths = []
    concept_counts = []
    
    print(f"Running evaluation on {len(dataset)} samples...")
    
    # Pre-indexing: Collect all docs from the dataset samples to build a mini-corpus
    # This ensures "closed-world" retrieval for testing purposes
    # CRITICAL: We need to preserve structure metadata for HyCoRAG
    corpus = {}
    corpus_metadata = {}  # NEW: Store metadata separately
    
    for i, item in enumerate(dataset):
        for j, doc in enumerate(item.documents):
            # Simple ID generation. In real large-scale, use real doc IDs.
            doc_id = f"doc_{item.qid}_{j}"
            corpus[doc_id] = doc
            # Preserve structure metadata if available
            if item.metadata and 'structure' in item.metadata:
                corpus_metadata[doc_id] = item.metadata
            
    if hasattr(pipeline.retriever, 'index'):
        print(f"Indexing {len(corpus)} documents from dataset...")
        # Pass metadata to retriever
        if hasattr(pipeline.retriever, 'index_with_metadata'):
            pipeline.retriever.index_with_metadata(corpus, corpus_metadata)
        else:
            pipeline.retriever.index(corpus)
            # Store metadata separately for retrieval
            if corpus_metadata:
                pipeline.retriever._corpus_metadata = corpus_metadata
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        # Run Pipeline
        try:
            result = pipeline.run(item.question, top_k=top_k)
            
            # Metrics
            # Handle multiple valid answers if present
            best_em = 0
            best_f1 = 0
            for gold_ans in item.answers:
                em = calculate_em(result.answer, gold_ans)
                f1 = calculate_f1(result.answer, gold_ans)
                if em > best_em: best_em = em
                if f1 > best_f1: best_f1 = f1
            
            ems.append(best_em)
            f1s.append(best_f1)
            
            # Metadata Stats
            if result.metadata:
                context_lengths.append(result.metadata.get("context_length", 0))
                concept_counts.append(result.metadata.get("total_concepts", 0))
                
        except Exception as e:
            print(f"Error processing sample {item.qid}: {e}")
            continue

    return {
        "avg_em": float(np.mean(ems)) if ems else 0.0,
        "avg_f1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_context_length": float(np.mean(context_lengths)) if context_lengths else 0.0,
        "avg_concept_count": float(np.mean(concept_counts)) if concept_counts else 0.0,
        "total_samples": len(ems)
    }

def run_text_qa_experiment(
    dataset_name: Literal["hotpotqa", "nq_open"], 
    mode: Literal["baseline", "distill_only", "full"],
    retriever: BaseRetriever,
    llm: LLMClient,
    distiller: HybridConceptDistiller,
    router: ConceptRouter,
    max_samples: int = 10
) -> Dict[str, Any]:
    
    print(f"\n[Experiment] Text QA | Dataset: {dataset_name} | Mode: {mode}")
    
    # 1. Load Data
    if dataset_name == "hotpotqa":
        dataset = HotpotQADataset.from_hf(split="validation", max_samples=max_samples) # Use val for exp
    elif dataset_name == "nq_open":
        dataset = NQOpenDataset.from_hf(split="validation", max_samples=max_samples)
    else:
        raise ValueError(f"Unknown text dataset: {dataset_name}")
        
    # 2. Setup Pipeline
    if mode == "baseline":
        pipeline = BaselineRAGPipeline(retriever, llm)
    else:
        pipeline = HyCoRAGPipeline(retriever, llm, distiller, router, mode=mode)
        
    # 3. Run
    metrics = run_pipeline_on_dataset(pipeline, dataset)
    return metrics

def run_table_mm_experiment(
    dataset_name: Literal["mmtab", "rag_igbench", "comtqa"],
    mode: Literal["baseline", "distill_only", "full"],
    retriever: BaseRetriever,
    llm: LLMClient,
    distiller: HybridConceptDistiller,
    router: ConceptRouter,
    max_samples: int = 10
) -> Dict[str, Any]:

    print(f"\n[Experiment] Table/MM QA | Dataset: {dataset_name} | Mode: {mode}")
    
    # 1. Load Data
    if dataset_name == "mmtab":
        dataset = MMTabDataset.from_hf(split="test", max_samples=max_samples)
    elif dataset_name == "rag_igbench":
        dataset = RAGIGBenchDataset.from_hf(split="test", max_samples=max_samples)
    elif dataset_name == "comtqa":
        dataset = ComTQADataset.from_hf(split="train", max_samples=max_samples)
    else:
        raise ValueError(f"Unknown table/mm dataset: {dataset_name}")

    # 2. Setup Pipeline (Same as text for now, but HyCoRAG should handle Table inputs)
    if mode == "baseline":
        pipeline = BaselineRAGPipeline(retriever, llm)
    else:
        pipeline = HyCoRAGPipeline(retriever, llm, distiller, router, mode=mode)
        
    # 3. Run
    metrics = run_pipeline_on_dataset(pipeline, dataset)
    return metrics

def run_hierarchical_experiment(
    dataset_name: Literal["realhitbench"],
    mode: Literal["baseline", "distill_only", "full"],
    retriever: BaseRetriever,
    llm: LLMClient,
    distiller: HybridConceptDistiller,
    router: ConceptRouter,
    max_samples: int = 10
) -> Dict[str, Any]:
    
    print(f"\n[Experiment] Hierarchical Table QA | Dataset: {dataset_name} | Mode: {mode}")
    
    if dataset_name == "realhitbench":
        dataset = RealHiTBenchDataset.from_local(data_path="RealHiTBench", max_samples=max_samples)
    else:
        raise ValueError(f"Unknown hier dataset: {dataset_name}")
        
    if len(dataset) == 0:
        print("Dataset empty (stub). Skipping.")
        return {}

    if mode == "baseline":
        pipeline = BaselineRAGPipeline(retriever, llm)
    else:
        pipeline = HyCoRAGPipeline(retriever, llm, distiller, router, mode=mode)
        
    return run_pipeline_on_dataset(pipeline, dataset)
